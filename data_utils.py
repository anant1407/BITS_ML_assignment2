"""Explicit UCI acquisition, dataset inspection, and canonical training schema.

Importing this module never downloads data or fits preprocessing.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from paths import PROVENANCE_FILE, RAW_DATA_FILE, RAW_DESCRIPTION_FILE

DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
CHARACTERISTICS = (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
)
FEATURE_NAMES = tuple(
    f"{name}_{statistic}"
    for statistic in ("mean", "se", "worst")
    for name in CHARACTERISTICS
)
TARGET_NAME = "diagnosis"
LABEL_MAPPING = {"B": 0, "M": 1}
CLASS_NAMES = {0: "Benign", 1: "Malignant"}
POSITIVE_LABEL = 1


@dataclass
class TestData:
    """Validated upload in canonical order; labels never enter predictors."""

    features: pd.DataFrame
    target: pd.Series

    @property
    def frame(self) -> pd.DataFrame:
        return self.features.assign(**{TARGET_NAME: self.target})


def parse_test_csv(payload: bytes) -> TestData:
    """Parse a UTF-8 labeled test CSV and report actionable schema errors."""
    if not payload:
        raise ValueError("The CSV is empty. Upload a file with a header and labeled test rows.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must be UTF-8 text. Export it as CSV UTF-8 and try again.") from exc
    if not text.strip() or "\x00" in text:
        raise ValueError("The CSV is empty or unreadable. Upload a UTF-8 CSV file.")
    try:
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows, None)
        if not header:
            raise ValueError("The CSV is empty or has no header.")
        duplicate_names = [name for name, count in Counter(header).items() if count > 1]
        if duplicate_names:
            raise ValueError("Duplicate column names: " + ", ".join(duplicate_names))
        if TARGET_NAME not in header:
            raise ValueError("Missing target column 'diagnosis'. True labels (0=benign, 1=malignant) are required for evaluation.")
        missing = [name for name in FEATURE_NAMES if name not in header]
        extra = [name for name in header if name not in {*FEATURE_NAMES, TARGET_NAME}]
        problems = []
        if missing:
            problems.append("Missing required features: " + ", ".join(missing))
        if extra:
            problems.append("Unexpected columns: " + ", ".join(extra))
        if problems:
            raise ValueError(". ".join(problems) + ". Use the 30 sample features plus diagnosis; omit IDs and index columns.")
        row_count = 0
        for line_number, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {line_number} has {len(row)} fields; expected {len(header)}.")
            row_count += 1
        if not row_count:
            raise ValueError("The CSV has no data rows. Include at least one labeled test sample.")
        frame = pd.read_csv(io.StringIO(text))
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("The CSV is malformed. Check quoting, commas, and the header.") from exc
    X = frame.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    invalid = [name for name in FEATURE_NAMES if pd.api.types.is_bool_dtype(frame[name])
               or not np.isfinite(X[name].to_numpy(dtype=float)).all()]
    if invalid:
        raise ValueError("Non-numeric, missing, or infinite predictor values in: " + ", ".join(invalid))
    y = pd.to_numeric(frame[TARGET_NAME], errors="coerce")
    if pd.api.types.is_bool_dtype(frame[TARGET_NAME]) or not y.isin([0, 1]).all():
        raise ValueError("diagnosis must contain only 0 (benign) or 1 (malignant), with no missing labels.")
    y = y.astype("int64")
    y.name = TARGET_NAME
    return TestData(X, y)


def file_sha256(path: Path) -> str:
    """Hash source files and artifacts for traceability."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset() -> dict:
    """Fetch missing UCI files; retain existing files and acquisition metadata.

    A previously recorded hash mismatch is an error, not a silent replacement.
    """
    previous = (
        json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
        if PROVENANCE_FILE.exists() else {"files": {}}
    )
    files = {}
    downloads = {}
    for path in (RAW_DATA_FILE, RAW_DESCRIPTION_FILE):
        url = DOWNLOAD_ROOT + path.name
        if path.exists():
            digest = file_sha256(path)
            old_record = previous.get("files", {}).get(path.name, {})
            if old_record.get("sha256", digest) != digest:
                raise ValueError(f"Source hash changed: {path.name}. Investigate before training.")
            files[path.name] = {
                "url": url,
                "retrieved_at_utc": old_record.get("retrieved_at_utc"),
                "sha256": digest,
                "acquisition": old_record.get("acquisition", "existing local file"),
            }
        else:
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            if not payload or payload.lstrip().lower().startswith(b"<!doctype html"):
                raise ValueError(f"UCI returned an unexpected response for {path.name}.")
            downloads[path] = payload
            files[path.name] = {
                "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "acquisition": "downloaded from UCI",
            }
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in downloads.items():
        path.write_bytes(payload)
    provenance = {"dataset": "Breast Cancer Wisconsin (Diagnostic)",
                  "source_url": DATASET_URL, "files": files}
    PROVENANCE_FILE.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return provenance


@dataclass
class Dataset:
    """Predictors and encoded labels, with source IDs retained outside X."""

    features: pd.DataFrame
    target: pd.Series
    sample_ids: pd.Series
    inspection: dict


def load_dataset(path: Path = RAW_DATA_FILE) -> Dataset:
    """Load and inspect local raw data without fitting or implicit downloads."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {path}. Run train_models.py --download first."
        )
    frame = pd.read_csv(path, header=None, dtype={0: "string", 1: "string"})
    if frame.shape[1] != len(FEATURE_NAMES) + 2:
        raise ValueError("Expected ID, diagnosis, and exactly 30 input features.")
    frame.columns = ["sample_id", TARGET_NAME, *FEATURE_NAMES]
    if len(frame) < 500 or len(FEATURE_NAMES) < 12:
        raise ValueError("Dataset fails the minimum 500-instance/12-feature requirement.")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Source sample IDs must be present and unique.")
    if not frame[TARGET_NAME].isin(LABEL_MAPPING).all():
        raise ValueError("Raw diagnosis values must be B or M with no missing labels.")
    X = frame.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Raw predictors contain missing or non-finite values.")
    y = frame[TARGET_NAME].map(LABEL_MAPPING).astype("int64")
    y.name = TARGET_NAME
    observations = X.assign(**{TARGET_NAME: y})
    duplicate_observations = int(observations.duplicated().sum())
    conflicting_groups = int(
        (observations.groupby(list(FEATURE_NAMES))[TARGET_NAME].nunique() > 1).sum()
    )
    if duplicate_observations or conflicting_groups:
        raise ValueError(
            "Repeated observations or conflicting labels require a documented "
            "cleaning decision before splitting. No rows were silently removed."
        )
    counts = {str(label): int((y == label).sum()) for label in CLASS_NAMES}
    if not all(counts.values()):
        raise ValueError("The training dataset must contain both diagnosis classes.")
    inspection = {
        "instances": len(X), "predictor_count": X.shape[1],
        "feature_names": list(FEATURE_NAMES),
        "feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
        "missing_values": int(frame.isna().sum().sum()),
        "non_finite_predictor_values": int((~np.isfinite(X.to_numpy())).sum()),
        "exact_duplicate_records": int(frame.duplicated().sum()),
        "duplicate_observations_excluding_id": duplicate_observations,
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "conflicting_label_groups": conflicting_groups,
        "class_counts": counts,
        "class_proportions": {label: count / len(y) for label, count in counts.items()},
        "label_mapping": LABEL_MAPPING, "positive_label": POSITIVE_LABEL,
        "feature_ranges": {
            name: {"min": float(X[name].min()), "max": float(X[name].max()),
                   "mean": float(X[name].mean()), "std": float(X[name].std())}
            for name in FEATURE_NAMES
        },
        "cleaning": "No rows removed; ID excluded; B/M encoded as 0/1.",
        "raw_data_sha256": file_sha256(path),
    }
    return Dataset(X, y, frame["sample_id"].copy(), inspection)
