"""Select five classifiers with training-only CV, then export verified artifacts.

Run `python train_models.py --download` for initial acquisition, then run without
the flag to reproduce the tuned experiment offline. Use --baseline to explicitly
reproduce the original fixed configurations instead of running parameter search.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import (
    CLASS_NAMES, DATASET_URL, FEATURE_NAMES, LABEL_MAPPING, POSITIVE_LABEL,
    TARGET_NAME, Dataset, download_dataset, file_sha256, load_dataset,
)
from evaluation import METRIC_NAMES, evaluate_classifier
from paths import (
    COMPARISON_FILE, CV_RESULTS_FILE, EVALUATION_REPORTS_FILE, EXPERIMENT_DETAILS_FILE,
    INSPECTION_FILE, METADATA_FILE, MODEL_DIR, PROJECT_ROOT, PROVENANCE_FILE, RAW_DATA_FILE,
    RAW_DESCRIPTION_FILE, TEST_DATA_FILE, TEST_PREDICTIONS_FILE,
    TRAIN_COMPARISON_FILE, ensure_project_directories,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

PARAMETER_GRIDS = {
    "logistic_regression": {"classifier__C": [0.1, 1.0, 10.0]},
    "decision_tree": {"classifier__max_depth": [3, 5, None], "classifier__min_samples_leaf": [1, 3]},
    "knn": {"classifier__n_neighbors": [3, 5, 9], "classifier__weights": ["uniform", "distance"]},
    "gaussian_nb": {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]},
    "svm": {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale", 0.01]},
}


@dataclass
class ModelSpec:
    name: str
    filename: str
    pipeline: Pipeline


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata: dict


def build_models() -> dict[str, ModelSpec]:
    """Create unfitted baselines with explicit, documented configurations."""
    return {
        "logistic_regression": ModelSpec(
            "Logistic Regression", "logistic_regression.joblib",
            Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
            ))]),
        ),
        "decision_tree": ModelSpec(
            "Decision Tree", "decision_tree.joblib",
            Pipeline([("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE, max_depth=None, min_samples_leaf=1,
            ))]),
        ),
        "knn": ModelSpec(
            "KNN", "knn.joblib",
            Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(
                n_neighbors=5, weights="uniform",
            ))]),
        ),
        "gaussian_nb": ModelSpec(
            "Naive Bayes", "gaussian_nb.joblib",
            Pipeline([("classifier", GaussianNB(var_smoothing=1e-9))]),
        ),
        "svm": ModelSpec(
            "Support Vector Machine", "svm.joblib",
            Pipeline([("scaler", StandardScaler()), ("classifier", SVC(
                C=1.0, kernel="rbf", gamma="scale", probability=True,
                random_state=RANDOM_STATE,
            ))]),
        ),
    }


def create_split(dataset: Dataset) -> DataSplit:
    """Split once; retain source IDs and verify observation separation."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features, dataset.target, test_size=TEST_SIZE,
        stratify=dataset.target, random_state=RANDOM_STATE,
    )
    train_ids = dataset.sample_ids.loc[X_train.index].tolist()
    test_ids = dataset.sample_ids.loc[X_test.index].tolist()
    if set(train_ids) & set(test_ids):
        raise ValueError("Source IDs overlap between training and test partitions.")
    train_observations = set(map(tuple, X_train.to_numpy()))
    test_observations = set(map(tuple, X_test.to_numpy()))
    if train_observations & test_observations:
        raise ValueError("Duplicate observations cross the train/test boundary.")
    metadata = {
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "stratified": True,
        "train_size": len(X_train), "test_rows": len(X_test),
        "train_class_counts": {str(c): int((y_train == c).sum()) for c in CLASS_NAMES},
        "test_class_counts": {str(c): int((y_test == c).sum()) for c in CLASS_NAMES},
        "train_sample_ids": train_ids, "test_sample_ids": test_ids,
        "train_row_indices": X_train.index.tolist(),
        "test_row_indices": X_test.index.tolist(),
        "source_id_overlap": 0, "duplicate_feature_overlap": 0,
    }
    return DataSplit(X_train, X_test, y_train, y_test, metadata)


def write_json(path: Path, payload: dict) -> None:
    """Write portable JSON without NaN or implicit numeric rounding."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def publish_file(source: Path, target: Path) -> None:
    """Replace a file using a sibling that inherits the output directory ACL.

    Moving a file directly out of a private Windows temporary directory would
    preserve its private ACL. Copying to a sibling first keeps exports readable
    under the same permissions as other project outputs.
    """
    sibling = target.with_name(f".{target.name}.{uuid4().hex}.publishing")
    try:
        with source.open("rb") as reader, sibling.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        sibling.replace(target)
    finally:
        sibling.unlink(missing_ok=True)


def _train_into(destination: Path, download: bool, tune: bool) -> pd.DataFrame:
    """Build a complete verified experiment in a staging directory."""
    ensure_project_directories()
    def output(path: Path) -> Path:
        staged = destination / path.relative_to(PROJECT_ROOT)
        staged.parent.mkdir(parents=True, exist_ok=True)
        return staged
    if download:
        download_dataset()
    if not RAW_DESCRIPTION_FILE.exists() or not PROVENANCE_FILE.exists():
        raise FileNotFoundError("Raw description or provenance missing; run with --download.")
    source = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    for source_file in (RAW_DATA_FILE, RAW_DESCRIPTION_FILE):
        expected = source["files"][source_file.name]["sha256"]
        if file_sha256(source_file) != expected:
            raise ValueError(f"Source file differs from its acquisition record: {source_file.name}")
    dataset = load_dataset()
    split = create_split(dataset)
    write_json(output(INSPECTION_FILE), dataset.inspection)
    test_frame = split.X_test.assign(**{TARGET_NAME: split.y_test})
    test_frame.to_csv(output(TEST_DATA_FILE), index=False)
    exported_test = pd.read_csv(output(TEST_DATA_FILE))
    exported_X = exported_test.loc[:, list(FEATURE_NAMES)]
    exported_y = exported_test[TARGET_NAME]
    np.testing.assert_array_equal(exported_X.to_numpy(), split.X_test.to_numpy())
    np.testing.assert_array_equal(exported_y.to_numpy(), split.y_test.to_numpy())

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(split.X_train, split.y_train)) if tune else []
    selection = {
        "mode": "tuned" if tune else "baseline", "scoring": "f1",
        "positive_label": 1, "cv_folds": 3 if tune else 0,
        "cv_shuffle": True if tune else False, "cv_random_state": RANDOM_STATE,
        "training_rows": len(split.X_train), "test_data_used_for_selection": False,
    }
    cv_fold_ids = [{
        "fold": number,
        "train_sample_ids": dataset.sample_ids.loc[split.X_train.iloc[fit].index].tolist(),
        "validation_sample_ids": dataset.sample_ids.loc[split.X_train.iloc[validation].index].tolist(),
    } for number, (fit, validation) in enumerate(folds, start=1)]
    test_rows, train_rows, cv_rows = [], [], []
    model_details, model_manifest, reports = {}, {}, {}
    predictions = pd.DataFrame({
        "sample_id": split.metadata["test_sample_ids"],
        TARGET_NAME: split.y_test.to_numpy(),
    })
    for key, spec in build_models().items():
        print(f"{'Selecting' if tune else 'Training'} {spec.name} on {len(split.X_train)} training rows...", flush=True)
        search_details = None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("error", ConvergenceWarning)
            if tune:
                search = GridSearchCV(
                    spec.pipeline, PARAMETER_GRIDS[key], scoring="f1", cv=folds,
                    refit=True, n_jobs=1, return_train_score=True, error_score="raise",
                )
                search.fit(split.X_train, split.y_train)
                spec.pipeline = search.best_estimator_
                search_details = {
                    "parameter_grid": PARAMETER_GRIDS[key],
                    "best_parameters": search.best_params_,
                    "best_cv_f1": float(search.best_score_),
                    "candidate_count": len(search.cv_results_["params"]),
                    "best_index": int(search.best_index_),
                }
                for index, parameters in enumerate(search.cv_results_["params"]):
                    cv_rows.append({
                        "Model": spec.name, "candidate": index,
                        "parameters": json.dumps(parameters, sort_keys=True),
                        "mean_validation_F1": float(search.cv_results_["mean_test_score"][index]),
                        "std_validation_F1": float(search.cv_results_["std_test_score"][index]),
                        "mean_training_F1": float(search.cv_results_["mean_train_score"][index]),
                        "rank": int(search.cv_results_["rank_test_score"][index]),
                        **{f"fold_{fold + 1}_validation_F1": float(search.cv_results_[f"split{fold}_test_score"][index]) for fold in range(3)},
                        "selected": index == search.best_index_,
                    })
            else:
                spec.pipeline.fit(split.X_train, split.y_train)
        warning_messages = list(dict.fromkeys(str(item.message) for item in captured))
        test_result = evaluate_classifier(spec.pipeline, split.X_test, split.y_test)
        train_result = evaluate_classifier(spec.pipeline, split.X_train, split.y_train)
        for metric, value in test_result.metrics.items():
            lower = -1 if metric == "MCC" else 0
            if value is None or not np.isfinite(value) or not lower <= value <= 1:
                raise ValueError(f"Invalid held-out metric for {spec.name}: {metric}={value}")
        artifact = output(MODEL_DIR / spec.filename)
        joblib.dump(spec.pipeline, artifact, compress=3)
        reloaded = joblib.load(artifact)
        disk_result = evaluate_classifier(reloaded, exported_X, exported_y)
        np.testing.assert_array_equal(test_result.predictions, disk_result.predictions)
        np.testing.assert_allclose(
            test_result.positive_scores, disk_result.positive_scores, rtol=1e-12, atol=1e-14,
        )
        for metric in METRIC_NAMES:
            np.testing.assert_allclose(test_result.metrics[metric], disk_result.metrics[metric])
        if "scaler" in reloaded.named_steps:
            scaler = reloaded.named_steps["scaler"]
            if int(scaler.n_samples_seen_) != len(split.X_train):
                raise ValueError("Scaler sample count does not match the training partition.")
            np.testing.assert_allclose(scaler.mean_, split.X_train.mean().to_numpy())
        test_rows.append({"Model": spec.name, **test_result.metrics})
        train_rows.append({"Model": spec.name, **train_result.metrics})
        model_manifest[key] = {
            "name": spec.name, "filename": spec.filename,
            "classes": reloaded.classes_.tolist(), "sha256": file_sha256(artifact),
        }
        model_details[key] = {
            "name": spec.name,
            "preprocessing": "StandardScaler" if "scaler" in spec.pipeline.named_steps else "none",
            "classifier": type(spec.pipeline.named_steps["classifier"]).__name__,
            "classifier_parameters": spec.pipeline.named_steps["classifier"].get_params(),
            "train_metrics": train_result.metrics, "test_metrics": test_result.metrics,
            "logistic_iterations": (
                spec.pipeline.named_steps["classifier"].n_iter_.tolist()
                if key == "logistic_regression" else None
            ),
            "artifact_reload_verified": True,
            "selection": search_details, "training_warnings": warning_messages,
        }
        reports[key] = test_result.to_dict()
        predictions[f"{key}_prediction"] = disk_result.predictions
        predictions[f"{key}_malignant_probability"] = disk_result.positive_scores

    comparison = pd.DataFrame(test_rows, columns=["Model", *METRIC_NAMES])
    comparison.to_csv(output(COMPARISON_FILE), index=False)
    pd.DataFrame(train_rows, columns=["Model", *METRIC_NAMES]).to_csv(output(TRAIN_COMPARISON_FILE), index=False)
    predictions.to_csv(output(TEST_PREDICTIONS_FILE), index=False)
    write_json(output(EVALUATION_REPORTS_FILE), reports)
    pd.DataFrame(cv_rows, columns=[
        "Model", "candidate", "parameters", "mean_validation_F1", "std_validation_F1",
        "mean_training_F1", "rank", "fold_1_validation_F1", "fold_2_validation_F1",
        "fold_3_validation_F1", "selected",
    ]).to_csv(output(CV_RESULTS_FILE), index=False)
    runtime = {"python": platform.python_version(), "packages": {
        package: version(package) for package in
        ("streamlit", "scikit-learn", "pandas", "numpy", "joblib", "matplotlib", "scipy")
    }}
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1, "created_at_utc": timestamp,
        "dataset": "Breast Cancer Wisconsin (Diagnostic)", "source_url": DATASET_URL,
        "feature_names": list(FEATURE_NAMES), "feature_count": len(FEATURE_NAMES),
        "feature_type": "finite numeric", "target_name": TARGET_NAME,
        "label_mapping": LABEL_MAPPING, "class_names": CLASS_NAMES,
        "positive_label": POSITIVE_LABEL, "split": split.metadata,
        "runtime": runtime, "models": model_manifest,
        "selection": selection,
        "raw_data_sha256": file_sha256(RAW_DATA_FILE),
        "test_csv_sha256": file_sha256(output(TEST_DATA_FILE)),
        "metric_conventions": {
            "auc": "ROC AUC using probability of malignant (1)",
            "precision_recall_f1": "binary; positive label 1; zero_division=0",
            "mcc": "0 for a constant actual or predicted marginal, with a notice",
            "confusion_matrix_order": [0, 1], "one_class_auc": None,
        },
    }
    write_json(output(METADATA_FILE), metadata)
    write_json(output(EXPERIMENT_DETAILS_FILE), {
        "created_at_utc": timestamp, "runtime": runtime,
        "source": source, "inspection": dataset.inspection, "split": split.metadata,
        "experiment": "Training-only stratified 3-fold CV; fixed grid; positive-class F1 selection; refit on 455 training rows." if tune else "Five fixed baseline configurations; no hyperparameter search or cross-validation.",
        "selection": selection, "cv_fold_sample_ids": cv_fold_ids,
        "ranking_criterion": "Positive-class F1; descriptive held-out comparison only.",
        "models": model_details,
        "verification": {
            "exported_test_matches_held_out_data": True,
            "all_saved_models_reloaded_and_matched": True,
            "scalers_fitted_on_training_only": True,
            "full_dataset_refit_performed": False,
        },
    })
    print("\nHeld-out test comparison (display rounded; CSV retains full precision):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSaved all five pipelines; exported-CSV predictions and probabilities verified.")
    return comparison


def train_and_export(download: bool = False, tune: bool = True) -> pd.DataFrame:
    """Stage and verify all outputs before publishing; commit metadata last.

    During publication the inference loader rejects mismatched hashes instead of
    serving a mixed bundle. On a publication error, restore replaced files.
    """
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="experiment-", dir=temporary_root) as temporary:
        staging = Path(temporary) / "output"
        comparison = _train_into(staging, download, tune)
        if tune and METADATA_FILE.exists() and COMPARISON_FILE.exists():
            previous = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            if previous.get("selection", {}).get("mode", "baseline") == "baseline":
                shutil.copy2(COMPARISON_FILE, staging / "results" / "baseline_model_comparison.csv")
        outputs = sorted(path for path in staging.rglob("*") if path.is_file())
        outputs.sort(key=lambda path: path.relative_to(staging) == METADATA_FILE.relative_to(PROJECT_ROOT))
        backups, published = {}, []
        try:
            for index, staged in enumerate(outputs):
                target = PROJECT_ROOT / staged.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    backup = Path(temporary) / f"backup-{index}"
                    shutil.copy2(target, backup)
                    backups[target] = backup
                publish_file(staged, target)
                published.append(target)
        except OSError:
            for target in reversed(published):
                if target in backups:
                    publish_file(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing source files from UCI.")
    parser.add_argument("--baseline", action="store_true", help="Explicitly export original baseline settings instead of CV-selected models.")
    args = parser.parse_args()
    train_and_export(download=args.download, tune=not args.baseline)


if __name__ == "__main__":
    main()
