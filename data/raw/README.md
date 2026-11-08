# Original UCI data

Source: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

- `wdbc.data`: Original records, preserved without modification.
- `wdbc.names`: Original UCI variable descriptions.
- `provenance.json`: Source URLs, actual acquisition timestamps in UTC, and SHA-256 file hashes.

Acquire missing files explicitly with `python train_models.py --download`.
Subsequent runs without `--download` use the local files and check their recorded
hashes. Importing application or data modules performs no network requests.

The raw column order is ID, B/M diagnosis, then three blocks (mean, standard
error, worst) of ten measurements. `data_utils.py` assigns canonical names,
excludes ID from predictors, and maps B to 0 and M to 1. Original IDs are retained
separately for verifying the split. No source records are silently removed.
