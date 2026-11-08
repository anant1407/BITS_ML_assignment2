"""Known-answer checks for evaluation semantics, independent of model quality."""

import json
import unittest

import numpy as np
import pandas as pd

from evaluation import evaluate_classifier


class FixedClassifier:
    """Supply known labels/scores, including nonstandard probability order."""

    def __init__(self, predictions, scores, classes=(0, 1)):
        self.classes_ = np.array(classes)
        self.predictions = np.array(predictions)
        self.scores = np.array(scores)

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        return np.column_stack([
            self.scores if label == 1 else 1 - self.scores for label in self.classes_
        ])

    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation must never fit a model.")


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        model = FixedClassifier([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
        result = evaluate_classifier(model, np.zeros((4, 2)), [0, 0, 1, 1])
        expected = {"Accuracy": 0.5, "AUC": 0.75, "Precision": 0.5,
                    "Recall": 0.5, "F1": 0.5, "MCC": 0.0}
        self.assertEqual(result.metrics, expected)
        np.testing.assert_array_equal(result.confusion_matrix, [[1, 1], [1, 1]])
        self.assertEqual(result.classification_report["Malignant"]["support"], 2)
        self.assertEqual(result.notices, [])

    def test_auc_uses_positive_class_not_column_position(self):
        model = FixedClassifier([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9], classes=(1, 0))
        result = evaluate_classifier(model, np.zeros((4, 2)), [0, 0, 1, 1])
        self.assertEqual(result.metrics["AUC"], 0.75)
        np.testing.assert_array_equal(result.positive_scores, model.scores)

    def test_one_class_data_and_strict_json(self):
        for label in (0, 1):
            with self.subTest(label=label):
                model = FixedClassifier([label, label], [float(label)] * 2)
                result = evaluate_classifier(model, np.zeros((2, 2)), [label, label])
                self.assertIsNone(result.metrics["AUC"])
                self.assertEqual(result.metrics["Accuracy"], 1)
                self.assertEqual(result.metrics["MCC"], 0)
                self.assertTrue(any("AUC is unavailable" in n for n in result.notices))
                self.assertTrue(any("MCC" in n for n in result.notices))
                self.assertEqual(result.confusion_matrix.shape, (2, 2))
                self.assertIsNone(json.loads(json.dumps(result.to_dict(), allow_nan=False))["metrics"]["AUC"])

    def test_no_positive_predictions(self):
        result = evaluate_classifier(
            FixedClassifier([0, 0], [0.1, 0.2]), np.zeros((2, 2)), [0, 1]
        )
        self.assertEqual(result.metrics["Precision"], 0)
        self.assertEqual(result.metrics["Recall"], 0)
        self.assertEqual(result.metrics["F1"], 0)
        self.assertEqual(result.metrics["AUC"], 1)
        self.assertTrue(any("Precision is undefined" in n for n in result.notices))

    def test_rejects_empty_invalid_or_misaligned_labels(self):
        model = FixedClassifier([0, 1], [0.1, 0.9])
        for X, y in [(np.zeros((0, 2)), []), (np.zeros((2, 2)), [0, 2]),
                     (np.zeros((2, 2)), [0, np.nan]), (np.zeros((2, 2)), [1])]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                evaluate_classifier(model, X, y)
        with self.assertRaisesRegex(ValueError, "indexes must align"):
            evaluate_classifier(model, pd.DataFrame({"x": [1, 2]}),
                                pd.Series([0, 1], index=[1, 0]))

    def test_rejects_invalid_probabilities(self):
        for scores in ([np.nan, 0.7], [0.1, 1.2]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                evaluate_classifier(FixedClassifier([0, 1], scores), np.zeros((2, 2)), [0, 1])


if __name__ == "__main__":
    unittest.main()
