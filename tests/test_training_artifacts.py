"""Integration checks for data provenance, leakage, and saved-model parity.

Run after `python train_models.py --download`. Only local repository artifacts
are loaded. These checks never fit a model or access the network.
"""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from data_utils import FEATURE_NAMES, TARGET_NAME, file_sha256, load_dataset
from evaluation import METRIC_NAMES, evaluate_classifier
from paths import (
    COMPARISON_FILE, EVALUATION_REPORTS_FILE, METADATA_FILE, MODEL_DIR,
    PROVENANCE_FILE, RAW_DATA_FILE, RAW_DESCRIPTION_FILE, TEST_DATA_FILE,
    CV_RESULTS_FILE, EXPERIMENT_DETAILS_FILE,
)
from train_models import create_split


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.split = create_split(cls.dataset)
        cls.metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    def test_cv_folds_use_only_training_samples(self):
        details = json.loads(EXPERIMENT_DETAILS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(details["selection"]["mode"], "tuned")
        self.assertEqual(details["selection"]["scoring"], "f1")
        self.assertFalse(details["selection"]["test_data_used_for_selection"])
        folds = details["cv_fold_sample_ids"]
        self.assertEqual(len(folds), 3)
        train_ids = set(self.metadata["split"]["train_sample_ids"])
        test_ids = set(self.metadata["split"]["test_sample_ids"])
        all_validation = []
        for fold in folds:
            fitting, validation = set(fold["train_sample_ids"]), set(fold["validation_sample_ids"])
            self.assertFalse(fitting & validation)
            self.assertEqual(fitting | validation, train_ids)
            self.assertFalse((fitting | validation) & test_ids)
            all_validation.extend(fold["validation_sample_ids"])
        self.assertEqual(len(all_validation), 455)
        self.assertEqual(set(all_validation), train_ids)

    def test_cv_selected_parameters_maximize_recorded_validation_f1(self):
        details = json.loads(EXPERIMENT_DETAILS_FILE.read_text(encoding="utf-8"))
        candidates = pd.read_csv(CV_RESULTS_FILE)
        self.assertEqual(len(candidates), 24)
        np.testing.assert_allclose(
            candidates["mean_validation_F1"],
            candidates[[f"fold_{fold}_validation_F1" for fold in (1, 2, 3)]].mean(axis=1),
        )
        for key, model in details["models"].items():
            with self.subTest(model=key):
                rows = candidates[candidates.Model == model["name"]]
                chosen = rows[rows.selected]
                self.assertEqual(len(chosen), 1)
                self.assertAlmostEqual(float(chosen.iloc[0].mean_validation_F1), float(rows.mean_validation_F1.max()))
                parameters = json.loads(chosen.iloc[0].parameters)
                self.assertEqual(parameters, model["selection"]["best_parameters"])
                pipeline = joblib.load(MODEL_DIR / self.metadata["models"][key]["filename"])
                for name, value in parameters.items():
                    self.assertEqual(pipeline.get_params()[name], value)

    def test_raw_schema_and_encoding_match_independent_reference(self):
        reference = load_breast_cancer()
        np.testing.assert_array_equal(self.dataset.features.to_numpy(), reference.data)
        np.testing.assert_array_equal(self.dataset.target.to_numpy(), 1 - reference.target)
        self.assertEqual(self.dataset.inspection["class_counts"], {"0": 357, "1": 212})
        self.assertEqual(list(self.dataset.features.columns), list(FEATURE_NAMES))
        self.assertNotIn(TARGET_NAME, self.dataset.features)
        provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
        for path in (RAW_DATA_FILE, RAW_DESCRIPTION_FILE):
            self.assertEqual(file_sha256(path), provenance["files"][path.name]["sha256"])

    def test_duplicates_and_conflicting_labels_require_explicit_decision(self):
        raw = pd.read_csv(RAW_DATA_FILE, header=None)
        for conflicting in (False, True):
            with self.subTest(conflicting=conflicting), tempfile.TemporaryDirectory() as temp:
                modified = raw.copy()
                modified.iloc[1, 2:] = modified.iloc[0, 2:].to_numpy()
                modified.iloc[1, 1] = ("B" if raw.iloc[0, 1] == "M" else "M") if conflicting else raw.iloc[0, 1]
                path = Path(temp) / "duplicate.data"
                modified.to_csv(path, header=False, index=False)
                with self.assertRaisesRegex(ValueError, "documented cleaning decision"):
                    load_dataset(path)

    def test_fixed_split_has_no_overlap_and_csv_matches_held_out_values(self):
        self.assertEqual(self.split.metadata, self.metadata["split"])
        self.assertEqual((len(self.split.X_train), len(self.split.X_test)), (455, 114))
        self.assertFalse(set(self.split.metadata["train_sample_ids"]) & set(self.split.metadata["test_sample_ids"]))
        self.assertEqual(self.split.metadata["test_class_counts"], {"0": 72, "1": 42})
        frame = pd.read_csv(TEST_DATA_FILE)
        self.assertEqual(list(frame.columns), [*FEATURE_NAMES, TARGET_NAME])
        np.testing.assert_array_equal(frame[list(FEATURE_NAMES)], self.split.X_test)
        np.testing.assert_array_equal(frame[TARGET_NAME], self.split.y_test)
        self.assertEqual(file_sha256(TEST_DATA_FILE), self.metadata["test_csv_sha256"])

    def test_all_saved_models_match_metrics_reports_and_training_partition(self):
        comparison = pd.read_csv(COMPARISON_FILE).set_index("Model")
        reports = json.loads(EVALUATION_REPORTS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(comparison), 5)
        self.assertEqual(len(self.metadata["models"]), 5)
        for key, entry in self.metadata["models"].items():
            with self.subTest(model=key):
                path = MODEL_DIR / entry["filename"]
                self.assertEqual(file_sha256(path), entry["sha256"])
                pipeline = joblib.load(path)
                self.assertEqual(list(pipeline.feature_names_in_), list(FEATURE_NAMES))
                result = evaluate_classifier(pipeline, self.split.X_test, self.split.y_test)
                np.testing.assert_allclose(
                    [result.metrics[m] for m in METRIC_NAMES],
                    comparison.loc[entry["name"], list(METRIC_NAMES)].to_numpy(dtype=float),
                    rtol=1e-12, atol=1e-14,
                )
                np.testing.assert_array_equal(result.predictions, reports[key]["predictions"])
                np.testing.assert_allclose(result.positive_scores, reports[key]["positive_scores"], rtol=1e-12)
                self.assertEqual(int(result.confusion_matrix.sum()), 114)
                classifier = pipeline.named_steps["classifier"]
                if key == "decision_tree":
                    fitted_rows = classifier.tree_.n_node_samples[0]
                elif key == "gaussian_nb":
                    fitted_rows = classifier.class_count_.sum()
                elif key == "knn":
                    fitted_rows = classifier.n_samples_fit_
                elif key == "svm":
                    fitted_rows = classifier.shape_fit_[0]
                else:
                    fitted_rows = pipeline.named_steps["scaler"].n_samples_seen_
                self.assertEqual(int(fitted_rows), 455)
                if "scaler" in pipeline.named_steps:
                    scaler = pipeline.named_steps["scaler"]
                    self.assertEqual(int(scaler.n_samples_seen_), 455)
                    np.testing.assert_allclose(scaler.mean_, self.split.X_train.mean().to_numpy())


if __name__ == "__main__":
    unittest.main()
