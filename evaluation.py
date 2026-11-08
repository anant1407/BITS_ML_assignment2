"""Evaluate fitted binary classifiers consistently in training and Streamlit."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


@dataclass
class EvaluationResult:
    metrics: dict[str, float | None]
    predictions: np.ndarray
    positive_scores: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: dict
    notices: list[str]

    def to_dict(self) -> dict:
        """Convert arrays for strict JSON export, preserving metric precision."""
        return {
            "metrics": self.metrics,
            "predictions": self.predictions.tolist(),
            "positive_scores": self.positive_scores.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "notices": self.notices,
        }


def evaluate_classifier(fitted_pipeline, X, y, positive_label: int = 1) -> EvaluationResult:
    """Evaluate an already fitted model; malignant (1) is the positive class.

    Precision/recall/F1 use zero_division=0. AUC is None for a one-class input.
    MCC is 0 when either marginal is constant, matching sklearn's convention;
    a notice explains that degeneracy. No fitting occurs here.
    """
    if positive_label != 1:
        raise ValueError("This project's schema defines malignant (1) as positive.")
    actual = np.asarray(y)
    if actual.ndim != 1 or not len(actual) or len(actual) != len(X):
        raise ValueError("Expected nonempty, equally sized features and one-dimensional labels.")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must be encoded as 0 or 1 without missing values.")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("Feature and target row indexes must align.")
    classes = np.asarray(fitted_pipeline.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("The saved classifier must have been trained on classes 0 and 1.")
    predicted = np.asarray(fitted_pipeline.predict(X))
    probabilities = np.asarray(fitted_pipeline.predict_proba(X), dtype=float)
    if predicted.shape != actual.shape or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Classifier returned invalid predicted labels.")
    if probabilities.shape != (len(actual), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Classifier returned invalid probability dimensions or values.")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("Classifier probabilities must be in [0, 1] and sum to one.")
    scores = probabilities[:, int(np.flatnonzero(classes == positive_label)[0])]
    notices = []
    auc = float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None
    if auc is None:
        notices.append("AUC is unavailable because the test labels contain only one class.")
    if not (predicted == 1).any():
        notices.append("Precision is undefined without positive predictions; reported as 0.")
    if not (actual == 1).any():
        notices.append("Recall is undefined without actual positive samples; reported as 0.")
    if not (actual == 1).any() and not (predicted == 1).any():
        notices.append("F1 is undefined without actual or predicted positives; reported as 0.")
    constant_marginal = np.unique(actual).size < 2 or np.unique(predicted).size < 2
    if constant_marginal:
        notices.append("MCC has a constant actual or predicted class; reported as 0 by convention.")
    metrics = {
        "Accuracy": float(accuracy_score(actual, predicted)), "AUC": auc,
        "Precision": float(precision_score(actual, predicted, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "F1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "MCC": 0.0 if constant_marginal else float(matthews_corrcoef(actual, predicted)),
    }
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    report = classification_report(
        actual, predicted, labels=[0, 1], target_names=["Benign", "Malignant"],
        output_dict=True, zero_division=0,
    )
    if (np.bincount(actual.astype(int), minlength=2) == 0).any() or (
        np.bincount(predicted.astype(int), minlength=2) == 0
    ).any():
        notices.append("Undefined per-class report metrics use zero_division=0.")
    return EvaluationResult(metrics, predicted, scores, matrix, report, notices)
