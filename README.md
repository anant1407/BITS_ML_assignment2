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

BITS_ML_assignment2 on GitHub: `https://github.com/anant1407/BITS_ML_assignment2.git`.

**Live Streamlit application:** `https://bitsmlassignment2git-kinwytsby5hbpxjdukvmtt.streamlit.app/` .

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