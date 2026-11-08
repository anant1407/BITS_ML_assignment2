"""CSV contract and interactive inference checks against repository artifacts."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from data_utils import FEATURE_NAMES, parse_test_csv
from inference import ArtifactError, artifact_signature, evaluate_upload, load_model_bundle
from paths import COMPARISON_FILE, METADATA_FILE, MODEL_DIR, PROJECT_ROOT, TEST_DATA_FILE


class UploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = TEST_DATA_FILE.read_bytes()
        cls.frame = pd.read_csv(TEST_DATA_FILE)

    def test_reordered_columns_and_utf8_bom(self):
        reordered = self.frame.loc[:, list(reversed(self.frame.columns))]
        data = parse_test_csv(reordered.to_csv(index=False).encode("utf-8-sig"))
        self.assertEqual(list(data.features), list(FEATURE_NAMES))
        np.testing.assert_array_equal(data.features, self.frame[list(FEATURE_NAMES)])
        self.assertEqual(len(data.target), 114)

    def test_schema_errors(self):
        header = self.payload.decode().splitlines()[0]
        examples = [
            (b"", "empty"), (b"   \n", "empty"), (b"\xff", "UTF-8"),
            ((header + "\n").encode(), "no data rows"),
            (self.frame.drop(columns="diagnosis").to_csv(index=False).encode(), "Missing target"),
            (self.frame.drop(columns="radius_mean").to_csv(index=False).encode(), "radius_mean"),
            (self.frame.assign(sample_id=1).to_csv(index=False).encode(), "Unexpected columns"),
            ((header + ",radius_mean\n").encode(), "Duplicate column names"),
            ((header + "\n1,2\n").encode(), "fields; expected"),
            ((header + '\n"unclosed').encode(), "malformed"),
        ]
        for payload, message in examples:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_test_csv(payload)

    def test_invalid_numeric_and_target_values(self):
        for value in ("not numeric", "", "NaN", "inf", "-inf"):
            changed = self.frame.astype({"radius_mean": "object"})
            changed.loc[0, "radius_mean"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "radius_mean"):
                parse_test_csv(changed.to_csv(index=False).encode())
        for value in ("M", "", "NaN", "inf", "2", "0.5"):
            changed = self.frame.astype({"diagnosis": "object"})
            changed.loc[0, "diagnosis"] = value
            with self.subTest(label=value), self.assertRaisesRegex(ValueError, "diagnosis must contain"):
                parse_test_csv(changed.to_csv(index=False).encode())

    def test_inference_matches_results_without_fitting_or_network(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting")), patch(
            "data_utils.urlopen", side_effect=AssertionError("No downloads")
        ):
            bundle = load_model_bundle(artifact_signature())
            actual = evaluate_upload(bundle, parse_test_csv(self.payload))
            expected = pd.read_csv(COMPARISON_FILE)
            self.assertEqual(actual.comparison["Model"].tolist(), expected["Model"].tolist())
            np.testing.assert_allclose(actual.comparison.iloc[:, 1:], expected.iloc[:, 1:], rtol=1e-12)
            for result in actual.results.values():
                self.assertEqual(result.confusion_matrix.sum(), 114)
            one_class = self.frame[self.frame.diagnosis == 0].head(5)
            revised = evaluate_upload(bundle, parse_test_csv(one_class.to_csv(index=False).encode()))
            self.assertEqual(len(revised.predictions), 5)
            self.assertTrue(all(result.metrics["AUC"] is None for result in revised.results.values()))

    def test_missing_mismatched_and_incompatible_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = json.loads(METADATA_FILE.read_text())
            path = directory / "metadata.json"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ArtifactError, "missing or unreadable"):
                artifact_signature(path, directory)
            metadata["models"]["knn"]["sha256"] = "changed"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ArtifactError, "models and metadata differ"):
                artifact_signature(path, MODEL_DIR)
            metadata["runtime"]["packages"]["scikit-learn"] = "0.0.0"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ArtifactError, "Install requirements.txt"):
                artifact_signature(path, MODEL_DIR)

    def test_artifact_signature_changes_for_new_metadata_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            metadata = json.loads(METADATA_FILE.read_text())
            path.write_text(json.dumps(metadata))
            old_signature = artifact_signature(path, MODEL_DIR)
            metadata["created_at_utc"] = "new model bundle revision"
            path.write_text(json.dumps(metadata))
            self.assertNotEqual(old_signature, artifact_signature(path, MODEL_DIR))


class AppTests(unittest.TestCase):
    def test_sample_all_models_and_visible_comparison(self):
        page = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertFalse(page.exception)
        self.assertEqual(len(page.metric), 0)
        page.checkbox(key="use_sample").check().run(timeout=30)
        self.assertFalse(page.exception)
        expected = pd.read_csv(COMPARISON_FILE).set_index("Model")
        for key in page.selectbox(key="classifier").options:
            # Selectbox options are display labels; select_index exercises every model.
            index = list(page.selectbox(key="classifier").options).index(key)
            page.selectbox(key="classifier").select_index(index).run(timeout=30)
            self.assertFalse(page.exception)
            self.assertEqual(len(page.metric), 6)
            for metric in page.metric:
                self.assertEqual(metric.value, f"{expected.loc[key, metric.label]:.4f}")
            self.assertEqual(len(page.dataframe[-1].value), 5)
            self.assertEqual(page.dataframe[-1].value["Model"].tolist(), expected.index.tolist())

    def test_upload_changes_clear_stale_results_and_explain_one_class(self):
        page = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        with patch("streamlit.file_uploader", return_value=io.BytesIO(TEST_DATA_FILE.read_bytes())):
            page.run(timeout=30)
        self.assertFalse(page.exception)
        self.assertEqual(len(page.metric), 6)
        sample = pd.read_csv(TEST_DATA_FILE)
        changed = sample[sample.diagnosis == 0].head(3).to_csv(index=False).encode()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(changed)):
            page.run(timeout=30)
        self.assertFalse(page.exception)
        self.assertEqual(next(m.value for m in page.metric if m.label == "AUC"), "N/A")
        self.assertTrue(any("AUC is unavailable" in n.value for n in page.info))
        self.assertIn("3 valid samples", page.success[0].value)
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"invalid")):
            page.run(timeout=30)
        self.assertFalse(page.exception)
        self.assertTrue(page.error)
        self.assertEqual(len(page.metric), 0)

    def test_artifact_error_is_displayed(self):
        with patch("inference.artifact_signature", side_effect=ArtifactError("Saved model is missing.")):
            page = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertFalse(page.exception)
        self.assertIn("Saved model is missing", page.error[0].value)


if __name__ == "__main__":
    unittest.main()
