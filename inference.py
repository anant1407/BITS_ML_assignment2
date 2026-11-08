"""Load repository artifacts and evaluate uploads without any training code."""

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from data_utils import FEATURE_NAMES, TARGET_NAME, TestData, file_sha256
from evaluation import EvaluationResult, METRIC_NAMES, evaluate_classifier
from paths import METADATA_FILE, MODEL_DIR

MODEL_KEYS = ("logistic_regression", "decision_tree", "knn", "gaussian_nb", "svm")


class ArtifactError(RuntimeError):
    """The repository's saved model bundle cannot be loaded consistently."""


def read_metadata(path: Path = METADATA_FILE) -> dict:
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or metadata["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Unexpected saved feature schema.")
        if metadata["target_name"] != TARGET_NAME or metadata["positive_label"] != 1:
            raise ValueError("Unexpected target encoding.")
        if metadata["label_mapping"] != {"B": 0, "M": 1}:
            raise ValueError("Unexpected class mapping.")
        if set(metadata["models"]) != set(MODEL_KEYS):
            raise ValueError("All five classifiers must be present.")
        for key, entry in metadata["models"].items():
            if entry["filename"] != f"{key}.joblib" or entry["classes"] != [0, 1]:
                raise ValueError("Unexpected saved model filename or classes.")
        for package in ("scikit-learn", "numpy", "scipy", "joblib"):
            expected = metadata["runtime"]["packages"][package]
            if version(package) != expected:
                raise ValueError(f"{package} must match the saved version ({expected}). Install requirements.txt.")
        return metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"Saved model metadata is missing or incompatible. {exc}") from exc


def artifact_signature(metadata_path: Path = METADATA_FILE, model_dir: Path = MODEL_DIR) -> str:
    """Validate bundle hashes; the signature changes whenever artifacts change."""
    metadata = read_metadata(metadata_path)
    try:
        hashes = [file_sha256(Path(metadata_path))]
        for key in MODEL_KEYS:
            entry = metadata["models"][key]
            digest = file_sha256(Path(model_dir) / entry["filename"])
            if digest != entry["sha256"]:
                raise ArtifactError("Saved models and metadata differ. Finish exporting the model bundle and reload.")
            hashes.append(digest)
        return ":".join(hashes)
    except OSError as exc:
        raise ArtifactError("A saved model is missing or unreadable. Include all five .joblib files in model/.") from exc


@dataclass
class ModelBundle:
    metadata: dict
    models: dict


def load_model_bundle(signature: str, metadata_path: Path = METADATA_FILE,
                      model_dir: Path = MODEL_DIR) -> ModelBundle:
    """Load only the known repository models, checking for mixed exports."""
    if artifact_signature(metadata_path, model_dir) != signature:
        raise ArtifactError("Model files changed during loading. Reload the page.")
    metadata = read_metadata(metadata_path)
    models = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            for key in MODEL_KEYS:
                pipeline = joblib.load(Path(model_dir) / metadata["models"][key]["filename"])
                if list(pipeline.feature_names_in_) != list(FEATURE_NAMES):
                    raise ValueError("A pipeline has a different feature schema.")
                np.testing.assert_array_equal(pipeline.classes_, [0, 1])
                models[key] = pipeline
    except Exception as exc:
        raise ArtifactError("A saved classifier could not be loaded. Restore the matching artifacts and pinned dependencies.") from exc
    if artifact_signature(metadata_path, model_dir) != signature:
        raise ArtifactError("Model files changed during loading. Reload the page.")
    return ModelBundle(metadata, models)


@dataclass
class UploadEvaluation:
    results: dict[str, EvaluationResult]
    comparison: pd.DataFrame
    predictions: pd.DataFrame


def evaluate_upload(bundle: ModelBundle, data: TestData) -> UploadEvaluation:
    """Evaluate the current upload afresh; no session result cache or fitting."""
    results, rows = {}, []
    predictions = pd.DataFrame({"row": range(1, len(data.target) + 1), TARGET_NAME: data.target.to_numpy()})
    for key in MODEL_KEYS:
        result = evaluate_classifier(bundle.models[key], data.features, data.target)
        results[key] = result
        rows.append({"Model": bundle.metadata["models"][key]["name"], **result.metrics})
        predictions[f"{key}_prediction"] = result.predictions
        predictions[f"{key}_malignant_probability"] = result.positive_scores
    return UploadEvaluation(results, pd.DataFrame(rows, columns=["Model", *METRIC_NAMES]), predictions)
