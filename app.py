"""Interactive evaluation of saved classifiers; this app never trains models."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st

from data_utils import DATASET_URL, FEATURE_NAMES, parse_test_csv
from evaluation import METRIC_NAMES
from inference import ArtifactError, artifact_signature, evaluate_upload, load_model_bundle
from paths import TEST_DATA_FILE


@st.cache_resource(show_spinner="Loading saved classifiers...")
def load_resources(signature: str):
    """Cache only models; artifact content hashes invalidate this resource."""
    return load_model_bundle(signature)


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
    st.caption("MACHINE LEARNING / FIVE-MODEL COMPARISON")
    st.title("Breast Cancer Classifier Comparison")
    st.write("Explore how five classifiers distinguish benign and malignant samples from 30 cell-nucleus measurements. This is an educational evaluation using the UCI Wisconsin Diagnostic dataset.")
    st.caption("Positive class: malignant (1). Benign: 0. Upload labeled, unseen test samples; saved models evaluate them without retraining.")
    try:
        bundle = load_resources(artifact_signature())
    except ArtifactError as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        st.header("Experiment")
        split = bundle.metadata["split"]
        st.write(f"Training samples: **{split['train_size']}**")
        st.write(f"Held-out samples: **{split['test_rows']}**")
        st.write(f"Stratified split · seed {split['random_state']}")
        selection = bundle.metadata.get("selection", {})
        st.write("Selection: training-only 3-fold CV, positive-class F1" if selection.get("mode") == "tuned" else "Fixed baseline configurations")
        st.markdown(f"[Dataset source on UCI]({DATASET_URL})")
        st.caption("Results describe the supplied samples. The app cannot verify whether arbitrary uploads were previously used for training.")
        with st.expander("Required CSV columns"):
            st.code(", ".join([*FEATURE_NAMES, "diagnosis"]), language=None)

    st.subheader("1. Choose your test data")
    st.write("Use a UTF-8 CSV with the 30 numeric features and a diagnosis column containing 0 or 1. Omit IDs and index columns. Feature columns may be reordered.")
    sample = None
    try:
        sample = TEST_DATA_FILE.read_bytes()
        st.download_button("Download sample test CSV", sample, "test_data.csv", "text/csv", key="sample_download")
    except OSError:
        st.warning("The sample test CSV is unavailable. You can still upload your own labeled test data.")
    uploaded = st.file_uploader("Upload labeled test CSV", type=["csv"], key="test_csv")
    use_sample = st.checkbox("Use the bundled test data", key="use_sample", disabled=sample is None)
    if uploaded is not None:
        payload = uploaded.getvalue()
        st.caption("Evaluating the uploaded CSV." if not use_sample else "Your uploaded CSV takes precedence over the bundled sample.")
    elif use_sample and sample is not None:
        payload = sample
        st.caption("Evaluating the repository's held-out test partition.")
    else:
        st.info("Upload a labeled test CSV or select the bundled test data to begin.")
        return
    try:
        data = parse_test_csv(payload)
    except ValueError as exc:
        st.error(str(exc))
        return
    counts = data.target.value_counts()
    st.success(f"{len(data.target)} valid samples · {int(counts.get(0, 0))} benign · {int(counts.get(1, 0))} malignant")
    with st.expander("Preview test data", expanded=False):
        st.dataframe(data.frame.head(10), hide_index=True, width="stretch")
    try:
        evaluation = evaluate_upload(bundle, data)
    except Exception:
        st.error("The saved classifiers could not evaluate this file. Check the CSV schema and use the matching saved model bundle.")
        return

    st.subheader("2. Inspect a classifier")
    selected = st.selectbox(
        "Classifier", list(bundle.models),
        format_func=lambda key: bundle.metadata["models"][key]["name"], key="classifier",
    )
    result = evaluation.results[selected]
    for column, metric in zip(st.columns(6), METRIC_NAMES):
        value = result.metrics[metric]
        column.metric(metric, "N/A" if value is None else f"{value:.4f}")
    for notice in result.notices:
        st.info(notice)
    matrix_column, report_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        figure, axis = plt.subplots(figsize=(4.8, 3.8))
        ConfusionMatrixDisplay(result.confusion_matrix, display_labels=["Benign", "Malignant"]).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d",
        )
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("Actual diagnosis")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with report_column:
        st.markdown("**Classification report**")
        report = pd.DataFrame({
            name: result.classification_report[name]
            for name in ("Benign", "Malignant", "macro avg", "weighted avg")
        }).T.rename(columns={"f1-score": "F1"})
        st.dataframe(report.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "F1": "{:.4f}", "support": "{:.0f}"}), width="stretch")
        st.caption("Precision, recall, and F1 above refer to malignant cases. The report also shows each class; undefined values use zero with an explanation.")

    st.subheader("3. Compare all five models")
    st.caption("Every row evaluates the same current test data. AUC uses malignant-class probabilities; the other metrics use predicted labels.")
    st.dataframe(evaluation.comparison.style.format({name: "{:.4f}" for name in METRIC_NAMES}, na_rep="N/A"), hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.download_button("Download comparison CSV", evaluation.comparison.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv", key="metrics_download")
    right.download_button("Download predictions CSV", evaluation.predictions.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv", key="predictions_download")


if __name__ == "__main__":
    main()
