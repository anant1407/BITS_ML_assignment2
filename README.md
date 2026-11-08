# Machine Learning Assignment 2

## a. Problem statement

Predict whether a breast-mass sample is benign or malignant from 30 numeric
measurements of cell nuclei. Implement Logistic Regression, Decision Tree, KNN,
Gaussian Naive Bayes, and SVM on one common dataset; compare all six required
metrics on the same held-out samples. Provide a Streamlit interface that evaluates
labeled test CSVs using saved pipelines without training on uploaded data.

This repository contains the training workflow, test data, saved models, and
interactive app. Local planning files are excluded from the submission repository.
The project is published on GitHub and deployed on Streamlit Community Cloud.
Final PDF preparation and portal submission remain separate tasks.

## b. Dataset description

Source: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Original files were retrieved on 8 November 2026. Exact acquisition timestamps,
URLs, and SHA-256 hashes are recorded in `data/raw/provenance.json`.

| Property | Verified value |
|---|---|
| Instances | 569 |
| Numeric input features | 30 |
| Classes | Benign: 357; malignant: 212 |
| Target | `diagnosis`, benign `B -> 0`, malignant `M -> 1` |
| Positive class | Malignant (`1`) |
| Missing/non-finite values | 0 |
| Exact duplicate records | 0 |
| Duplicate observations excluding ID | 0 |
| Duplicate feature rows / conflicting-label groups | 0 / 0 |
| Cleaning | No rows removed; ID excluded from predictors; target explicitly encoded |

The 30 features are three blocks (mean, standard error, worst) of ten measurements:
radius, texture, perimeter, area, smoothness, compactness, concavity, concave points,
symmetry, and fractal dimension. Canonical names include `radius_mean`,
`radius_se`, and `radius_worst`. Per-column data types and min/max/mean/standard
deviation are generated locally in `results/dataset_inspection.json` during
training; this optional summary is excluded from Git. The same inspection is
retained inside `results/experiment_details.json`.

One stratified 80/20 split uses `random_state=42` for every classifier:

| Partition | Rows | Benign | Malignant |
|---|---:|---:|---:|
| Training | 455 | 285 | 170 |
| Test | 114 | 72 | 42 |

Original sample IDs and row indexes are retained in experiment metadata, with no
ID or duplicate-observation overlap between partitions. The exported
`test_data.csv` contains the exact unscaled held-out features and target, without
ID or an extra index column. Its 31 columns are the canonical 30 features followed
by `diagnosis`. Do not fit models on this CSV.

## c. GitHub repository link

[BITS_ML_assignment2 on GitHub](https://github.com/anant1407/BITS_ML_assignment2)

Clone URL: `https://github.com/anant1407/BITS_ML_assignment2.git`.

**Live Streamlit application:** [Open the classifier comparison app](https://bitsmlassignment2git-kinwytsby5hbpxjdukvmtt.streamlit.app/).

## d. Models used and evaluation

| Classifier | Preparation and selected configuration |
|---|---|
| Logistic Regression | StandardScaler; `C=1`, `solver=lbfgs`, `max_iter=2000`, seed 42 |
| Decision Tree | No scaling; unrestricted depth, `min_samples_leaf=1`, seed 42 |
| KNN | StandardScaler; 3 neighbors, uniform weights |
| Gaussian Naive Bayes | No scaling; `var_smoothing=1e-9` |
| SVM | StandardScaler; RBF kernel, `C=10`, `gamma=0.01`, probabilities enabled, seed 42 |

Preprocessing is fitted only on the training partition and saved inside each
pipeline. After completing and verifying the baseline app, a fixed 24-candidate
search used stratified three-fold cross-validation on the 455 training rows only.
The scoring criterion was declared as positive-class F1 before searching.
Every candidate includes its preprocessing pipeline, so scaling is fitted within
each fold. The folds are shuffled with seed 42 and shared across models.

| Classifier | Search grid | Selected mean validation F1 |
|---|---|---:|
| Logistic Regression | `C`: 0.1, 1, 10 | 0.964599 |
| Decision Tree | depth: 3, 5, unrestricted; minimum leaf: 1, 3 | 0.893478 |
| KNN | neighbors: 3, 5, 9; weights: uniform, distance | 0.954341 |
| Gaussian Naive Bayes | smoothing: 1e-9, 1e-8, 1e-7 | 0.921915 |
| SVM | `C`: 0.1, 1, 10; `gamma`: scale, 0.01 | 0.957849 |

Each selected configuration was refitted on all 455 training rows before final
test evaluation. The original 114-row test set remained unchanged and was never
used for parameter selection. Logistic Regression completed without a convergence
warning. Classifier parameters, actual iteration counts, warnings, CV fold sample
IDs, and selection details are recorded in `results/experiment_details.json`.
All candidate scores, including individual fold scores, are in `results/cv_results.csv`.
No pipeline was refitted on the full 569-row dataset.

Held-out results below are rounded to six decimals. The CSV retains full precision.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree | 0.929825 | 0.924603 | 0.904762 | 0.904762 | 0.904762 | 0.849206 |
| KNN | 0.938596 | 0.982474 | 0.972973 | 0.857143 | 0.911392 | 0.868766 |
| Naive Bayes | 0.938596 | 0.993386 | 1.000000 | 0.833333 | 0.909091 | 0.871489 |
| Support Vector Machine | 0.982456 | 0.996032 | 1.000000 | 0.952381 | 0.975610 | 0.962622 |

`evaluation.py` uses malignant-class probabilities for ROC AUC and predicted labels
for the other five metrics. It locates probabilities using the fitted `classes_`
order. Precision, recall, and F1 use `zero_division=0`; notices explain undefined
cases. A one-class test input returns AUC as `None` (JSON `null`) and a notice.
MCC uses 0 with an explanatory notice when an actual or predicted class marginal
is constant. Confusion matrices always use `[Benign, Malignant]` for both axes.

The chosen `SVC(probability=True)` API is supported by the pinned scikit-learn
1.9.1, but that version emits a deprecation warning recommending an explicit
calibrator for a future release. Keep the tested dependency pins when loading
these artifacts; a future API migration requires retraining and validation.

## e. Observations and overall winner

The following observations use the exported test metrics and confusion matrices,
plus `results/training_comparison.csv` for training/test comparisons. A false
positive is a benign sample classified as malignant; a false negative is a
malignant sample classified as benign.

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 68 | 4 | 4 | 38 |
| KNN | 71 | 1 | 6 | 36 |
| Naive Bayes | 72 | 0 | 7 | 35 |
| Support Vector Machine | 72 | 0 | 2 | 40 |

Every row totals the same 114 samples: 72 benign and 42 malignant.

| ML model | Observation about measured performance |
|---|---|
| Logistic Regression | This linear baseline is competitive: accuracy 0.964912, F1 0.951220, and MCC 0.924518. It detects 39/42 malignant cases, missing 3, and produces 1 false positive. Its AUC of 0.996032 ties SVM, although its predicted labels produce two more errors. Training F1 is 0.982036 versus test F1 0.951220. High ranking performance does not imply identical errors at each classifier's decision boundary. |
| Decision Tree | Training accuracy and F1 are both 1.0, while test accuracy is 0.929825 and F1 is 0.904762. This is the largest training/test gap among these models and is consistent with overfitting. The CV search nevertheless selected unrestricted depth with a minimum leaf size of 1. It makes 4 false positives and 4 false negatives and has the lowest test AUC (0.924603) and MCC (0.849206). |
| KNN | The selected 3-neighbor model achieves precision 0.972973 but recall 0.857143: it makes 1 false positive and misses 6 malignant cases. Its test F1 is 0.911392 versus training F1 0.978979. The baseline 5-neighbor model had test F1 0.938272. The 3-neighbor configuration remains selected because it had the best training-CV mean F1 (0.954341); the held-out score was not used to reverse that decision. |
| Gaussian Naive Bayes | It has precision 1.0 with no false positives, but misses 7/42 malignant cases, giving the lowest recall (0.833333). Its F1 is 0.909091 and MCC is 0.871489. Its high AUC (0.993386) describes ranking across thresholds; the actual predicted labels still miss more positive cases. Conditional feature independence is a possible limitation of this model, but this experiment does not establish that assumption as the cause of its errors. |
| Support Vector Machine | The selected RBF SVM detects 40/42 malignant cases, with 2 false negatives and no false positives. It has the highest test accuracy (0.982456), recall (0.952381), F1 (0.975610), and MCC (0.962622), while tying Logistic Regression on AUC (0.996032). Training F1 is 0.982036. Its selected C=10 and gamma=0.01 improve test F1 relative to the baseline SVM's 0.962963. |
| Overall winner for this experiment | **SVM**, using the predeclared positive-class F1 criterion for the descriptive test comparison. Its highest F1 is supported by the highest recall and MCC, with no false positives in this particular test set. Logistic Regression remains a competitive alternative with the same AUC. No inference-latency or probability-calibration comparison was performed. |

The original baseline test scores are preserved in
`results/baseline_model_comparison.csv`. Hyperparameters were selected separately
for each algorithm through training-only CV. Logistic Regression has the highest
selected mean CV F1 (0.964599), whereas SVM has the highest final test F1. These
are different evaluations; the descriptive test winner does not replace the
CV-based parameter-selection procedure.

### Limits of the comparison

The test set contains only 114 samples, including 42 malignant cases. One changed
prediction changes accuracy by about 0.88 percentage points; one additional
missed malignant case changes recall by about 2.38 percentage points. SVM's
accuracy advantage over Logistic Regression is two correctly classified samples.
Small differences on this single split do not establish statistical superiority.

These results concern one historical public dataset and one stratified split.
No external-population evaluation, nested CV, threshold optimization, calibration
study, or clinical validation was performed. Statements about why a model behaves
a certain way are interpretations unless directly supported by the recorded
experiment. All numerical observations above come from the saved outputs; no
scores, commits, or repository links have been invented.

## Python environment

The local project uses **CPython 3.14.3** in `venv/`, with system-site packages
disabled. The local `.python-version` file is excluded from Git; the required
interpreter version is documented here. For deployment,
select **Python 3.14** in Streamlit Community Cloud's Advanced settings; the cloud
manages patch releases. `.python-version` does not configure Community Cloud.

Streamlit documents support for released Python versions still receiving security
updates: [deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

From the repository root in PowerShell, create the environment on a fresh checkout:

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The environment already exists in this workspace. Use its interpreter directly;
activation is optional, so no PowerShell execution-policy changes are needed.

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe train_models.py --download
.\venv\Scripts\python.exe train_models.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m streamlit run app.py
```

Use `--download` for initial acquisition of missing UCI files. Existing source files
and their acquisition timestamps are preserved; hash mismatches fail explicitly.
After acquisition, training without that flag is entirely offline. Running the
training command regenerates the test CSV, pipelines, and results from the same
fixed split. The app evaluates uploaded data using the exported models.

The default training command runs the bounded CV search and exports the selected
models. Use `train_models.py --baseline` only to explicitly replace them with the
original fixed configurations. All outputs are staged and verified before
publication, with metadata published last and rollback on publication errors.
Output files inherit workspace directory permissions on Windows. During an
export the app rejects a mismatched model/metadata bundle instead of serving
mixed model versions.

`requirements.txt` pins the six direct dependencies to locally verified versions.
Use these versions for training and loading saved models in subsequent phases.

Setup verification completed on 8 November 2026: all six direct dependencies
imported at their pinned versions, `pip check` found no broken requirements,
project paths and the training scaffold worked from another working directory,
and Streamlit's `AppTest` rendered the initial page without exceptions.

Parts 2-6 verification: **10 tests passed**. These include hand-calculated metric
values, reversed probability-column order, one-class inputs, malformed labels and
probabilities, source-schema comparison against scikit-learn's independent local
copy, duplicate/conflicting-label rejection, split and CSV equality, artifact
hashes, saved model/results agreement, and training-only fitted sample counts and
scaler statistics. The training command also checks reloaded predictions and
probabilities against the original evaluation using the exported CSV.

Parts 7-8 verification: **21 tests passed** in the complete suite after tuning.
New checks cover invalid CSVs, column reordering, missing/incompatible artifacts,
all model dropdown choices, uploaded-file changes, one-class notices, app/result
agreement, metadata cache invalidation, and training-only CV selection. App
inference was exercised with fitting and network downloads explicitly blocked.

### Reproducibility verification

Run the committed test suite using the `unittest` command above. It checks upload
validation, all five classifiers, metric parity, saved artifacts, data separation,
and CV selection. Reproduce training first when changing the dataset or models.

Additional local QA verified dependency consistency, training from another working
directory, inference-path inspection, app rendering, and a temporary headless
Streamlit server. The local QA helper (`tools/`) and its generated report
(`results/verification_report.json`) are excluded from Git and are not needed
to run the submitted app, training workflow, or committed tests.

Verified on 8 November 2026 in a newly installed Python 3.14.3 environment:
**all 21 tests passed**, dependencies had no conflicts, retraining reproduced the
metrics and test CSV, all five artifacts loaded, the app rendered successfully,
and the local Streamlit health endpoint returned HTTP 200. The verifier also
confirmed the required README section order and agreement of all five metric and
confusion-matrix rows with the saved results. Cloud deployment is not part of
this local verification run.

## Using the Streamlit app

Open the live application linked above, or run the local app:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

1. Download the sample test CSV and upload it, or select **Use the bundled test data**.
2. For your own test data, supply a UTF-8 CSV with the 30 feature columns and
   `diagnosis` encoded as 0 or 1. Do not include an ID or exported index column.
   Feature order may vary; the app restores the canonical order.
3. Review the sample/class counts and optional data preview.
4. Choose a classifier to view all six metrics, its confusion matrix, and the
   per-class report. A one-class upload shows AUC as unavailable with an explanation.
5. Compare all five classifiers on the same upload and download comparison scores
   or row-level predictions, including malignant-class probabilities.

An uploaded file takes precedence over the bundled-data checkbox. Invalid inputs
show actionable errors and clear prior results. The app rejects malformed CSVs,
duplicate/missing/extra columns, nonnumeric/missing/infinite predictors, and
missing or unknown target labels. Uploads should contain unseen test samples;
the app cannot independently verify their provenance.

Only repository-supplied model artifacts are loaded. Model loading is resource
cached using metadata and artifact content hashes; results are recomputed for
the current input on reruns. `app.py` and `inference.py` never import the training
entrypoint or run fitting, parameter search, or dataset downloads.

### Hosted deployment validation

Validated on 8 November 2026 against the public Streamlit Community Cloud app.
The runtime reports repository `anant1407/bits_ml_assignment2`, branch `main`,
entrypoint `app.py`, Streamlit 1.67.0, and Python 3.14.10 on Linux. The cloud
Python 3.14 minor version matches local development; the cloud manages its patch
release, which differs from the local Python 3.14.3 installation.

All five classifiers' six metric cards and comparison-table values matched
`results/model_comparison.csv`. The live sample CSV download matched local test
data, and an actual multipart upload reproduced the same results. Classification
reports and confusion-matrix images were emitted for every model; the downloaded
SVM matrix showed 72 true negatives, 0 false positives, 2 false negatives, and
40 true positives. Comparison and prediction downloads also matched local results.
Reordered CSV columns preserved results, one-class data displayed AUC as N/A,
and an invalid upload showed a validation error with prior results cleared.

These checks used public HTTP and Streamlit's normal runtime messages because
browser access was unavailable. The downloaded SVM chart was visually inspected;
a full browser layout inspection was not performed. The local validation evidence
is under the ignored `tmp/deployment-validation/` directory.

## Files and paths

| Path | Purpose |
|---|---|
| `paths.py` | Shared paths anchored to the source directory |
| `data_utils.py` | UCI acquisition, local loading/inspection, canonical schema, and uploaded CSV validation |
| `evaluation.py` | Shared metrics, predictions, scores, notices, and reports |
| `train_models.py` | Common split, training-only grid search, staged export, and reload verification |
| `inference.py` | Consistent repository artifact loading and upload evaluation |
| `app.py` | Interactive upload, model selection, metrics, reports, and CSV downloads |
| `data/raw/` | Original dataset and variable descriptions |
| `model/` | Saved fitted pipelines and schema metadata |
| `results/` | Comparison results and experiment details |
| `results/figures/` | Exported figures |
| `tests/` | Focused metric and saved-artifact correctness checks |

### Generated artifacts

- `test_data.csv`: Exact held-out features and encoded labels.
- `model/*.joblib`: Five complete fitted pipelines, including preprocessing where needed.
- `model/metadata.json`: Feature schema, class mapping, split IDs, runtime/package
  versions, model filenames/hashes, test CSV hash, and metric conventions.
- `results/dataset_inspection.json` (local only, generated during training): Dataset integrity checks and feature summaries.
- `results/model_comparison.csv`: Five-model held-out comparison at full precision.
- `results/training_comparison.csv`: Training scores for gap analysis.
- `results/baseline_model_comparison.csv`: Original fixed-configuration test scores.
- `results/cv_results.csv`: All 24 parameter candidates, fold/mean/std validation
  F1, mean training F1, ranks, and selected-candidate indicators.
- `results/evaluation_reports.json`: Test predictions, positive scores, confusion
  matrices, classification reports, and notices for every model.
- `results/test_predictions.csv`: Test sample IDs, true labels, and each model's
  predicted label and malignant-class probability.
- `results/experiment_details.json`: Acquisition metadata, inspection, exact split,
  classifier parameters, training/test scores, runtime versions, and verification.
- `results/verification_report.json` (local only, generated by the verifier): Actual fresh-environment verification outcome,
  package versions, training reproducibility, tests, model loading, and server health.

All data and artifact paths are derived from `Path(__file__).resolve().parent`,
so they do not depend on the process working directory. Importing path constants
does not create files; `ensure_project_directories()` creates directories only
when called explicitly.

Local environments, caches, credentials, temporary files, planning checklists,
interpreter-selection files, the assignment handout, empty-directory placeholders,
local verification helpers, and machine-local verification reports are excluded by `.gitignore`. Application
and training source, dependencies, README, test data, saved model/preprocessing
artifacts, and supporting experiment results remain eligible for version control.
Raw training inputs and tests are retained so documented
reproduction commands work from a checkout.

## References and implementation notes

- [UCI dataset and original variable descriptions](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- [scikit-learn preprocessing and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

The project uses the libraries' classifier implementations, an explicit local
training/evaluation workflow, and observations based on its actual saved results.
Before submission, the author should understand and be able to explain the split,
preprocessing, CV selection, positive-class metrics, model persistence, and the
inference-only app, consistent with the assignment's learning-support policy.
