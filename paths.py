"""Project paths anchored to this file, independent of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "wdbc.data"
RAW_DESCRIPTION_FILE = RAW_DATA_DIR / "wdbc.names"
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"
METADATA_FILE = MODEL_DIR / "metadata.json"
COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
EXPERIMENT_DETAILS_FILE = RESULTS_DIR / "experiment_details.json"
PROVENANCE_FILE = RAW_DATA_DIR / "provenance.json"
INSPECTION_FILE = RESULTS_DIR / "dataset_inspection.json"
EVALUATION_REPORTS_FILE = RESULTS_DIR / "evaluation_reports.json"
TRAIN_COMPARISON_FILE = RESULTS_DIR / "training_comparison.csv"
TEST_PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
CV_RESULTS_FILE = RESULTS_DIR / "cv_results.csv"


def ensure_project_directories() -> None:
    """Create output and raw-data directories when explicitly requested."""
    for directory in (RAW_DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
