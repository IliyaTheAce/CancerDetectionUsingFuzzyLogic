# Cancer Detection Research Project

## 1. Project Goal

We are preparing a research paper for the **2026 International Congress on Cancer Prevention at University of Zanjan**.

The instructor's main requirement is:

> Apply **Fuzzy Logic** in an **Artificial Intelligence / Machine Learning context** for **cancer diagnosis/detection**.

The cancer type is **prostate cancer**.

The project is an **experimental paper**, not a review.

Maximum paper length: **10 pages**.

The paper format should be based on papers/templates from previous years of the same conference.

---

## 2. Dataset

**Prostate Cancer Data - Mendeley Data**

URL: https://data.mendeley.com/datasets/6db5t5b52x/1

DOI: 10.17632/6db5t5b52x.1

Verified from the downloaded files, not from the web page alone:

| File | Rows in the downloaded file |
| --- | ---: |
| `Data/data-1.csv` | 478 |
| `Data/data-2.csv` | 120 |

The released files originally had 298 and 300 patients. Patients were moved from the external file into the training file. Their original `No.` values are kept, including repeats. The two files together still contain **598 patients**.

The code does **not** keep the file boundary as the train/test split. Both files are stacked into one pool. The pool is then split once with a stratified 80/20 split (`seed=42`) into:

| Split | Patients | positive | negative |
| --- | ---: | ---: | ---: |
| Training set | 478 | 206 | 272 |
| Test set | 120 | 52 | 68 |

`source_file` records which downloaded file a row came from. It is not a cohort. The test set is not the published 300-patient external cohort, and a paper has to say that.

The target column is `Prostate Cancer` (`positive` / `negative`). This is binary classification of the recorded label. It is not a claim of clinical diagnosis, stage, or Gleason grade. Those fields are not in the data.

The test set is used only for the final comparison. It does not set medians, the scaler, membership knots, feature decisions, or classifier parameters.

Column meanings are in `Docs/DatasetColumns.md`.

---

## 3. Research Direction

The model is a **fuzzy inference system** on three clinical features. There is **no neural learning** and no ANFIS step. Fuzzy parameters are not tuned by a network.

```text
Clinical dataset (598 patients)
      ↓
Cleaning (row-level checks on the pool)
      ↓
Stratified 80/20 split
      ↓
Training medians, feature selection, membership knots
      ↓
27-rule Sugeno system
      ↓
Comparison on the test set
      against logistic regression, SVM, a decision tree, and XGBoost
      ↓
100 repeated splits for stability
```

The four other algorithms are baselines. They are not part of the proposed method. They answer whether the fuzzy system is competitive with ordinary classifiers on the same inputs.

---

## 4. What the fuzzy system is

Selected features, chosen only on the training set:

| Feature | Meaning | Direction in cancer |
| --- | --- | --- |
| `PSAD` | PSA density, `tPSA / PV` | higher |
| `f_t_psa` | free-to-total PSA | lower |
| `PV` | prostate volume | lower |

Each feature has three linguistic terms (`low`, `medium`, `high`) from the training 25th, 50th, and 75th percentiles. The full rule grid is 3 × 3 × 3 = 27 rules. Each consequent is a ridge-regression weight fit to the training labels. The penalty is fixed at 1. This is not a neural network.

A patient is called positive when the score passes the cutoff that maximized accuracy on the training out-of-fold predictions. On the current run that cutoff is 0.456.

Why this shape:

* Three inputs keep the rule base small enough for 478 training patients.
* Quartile knots do not use the cancer label.
* Shoulder memberships stay fully high above the 75th percentile, which matters because `PSAD` is heavily skewed.
* The rules can be read as sentences. That is the reason to use fuzzy logic here.

Details: `Docs/FeatureSelection.md`, `Docs/Fuzzification.md`, and `Docs/Code.md`.

---

## 5. Final comparison

Same three features for every model. Fit on the training set only. Report the test set as the result that counts.

| Model | Role | Setting, fixed before the test set |
| --- | --- | --- |
| Fuzzy 27-rule system | Proposed method | Quartiles from the training set. Consequents from ridge regression on the training labels, penalty 1. Cutoff from training cross-validation accuracy. |
| Logistic regression | Linear baseline | `C=1`, features standardized on the training set. Cutoff from training cross-validation accuracy. |
| SVM | Nonlinear baseline | RBF kernel, `C=1`, `gamma="scale"`, same standardization. Cutoff from training cross-validation accuracy. |
| Decision tree | Rule-like baseline | Depth 3, at least 15 patients per leaf. Cutoff from training cross-validation accuracy. |
| XGBoost | Boosted-tree baseline | 100 trees, depth 2, learning rate 0.1, minimum child weight 5. Cutoff from training cross-validation accuracy. |

Metrics, same definition for every row:

* Accuracy
* Sensitivity (recall of `positive`)
* Specificity
* Precision
* F1-score
* ROC-AUC

Also reported, and not the headline result:

* 5-fold stratified cross-validation on the training set, for every model
* Training-set fit, so memorization is visible
* 100 seeded stratified 80/20 repeats for stability (`src/repeated.py`)

The test set is not used to pick a feature, a knot, a consequent, or a hyperparameter. The measured table is in `Docs/Code.md`.

---

## 6. Why fuzzy logic is in the paper

`PSAD`, free-to-total PSA, and prostate volume are graded, not yes/no, measurements. The system states rules of the form:

```text
IF PSAD is high
AND f/t PSA is low
AND prostate volume is low
THEN cancer risk is high
```

The comparison then shows how that readable score stands next to a linear model, a kernel model, a single tree, and a boosted tree.

---

## 7. Language and libraries

Python is the implementation language.

```text
Python
├── pandas
├── numpy
├── scipy
├── scikit-learn
├── matplotlib
└── xgboost
```

Run the project virtualenv:

```text
.\Scripts\python.exe main.py
```

---

## 8. Paper outline

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Related Work
6. Materials and Methods
7. Fuzzy Model
8. Experimental Results (test-set comparison and repeated splits)
9. Discussion
10. Conclusion
11. References

Keep the paper within **10 pages**.

Possible title:

> A Fuzzy Inference System for Prostate Cancer Classification on Clinical Data, Compared with Standard Classifiers

Persian direction:

> سیستم استنتاج فازی برای طبقه‌بندی سرطان پروستات روی داده‌های بالینی و مقایسه آن با طبقه‌بندهای استاندارد

---

## 9. Constraints

* This is a research comparison, not a diagnostic device.
* Do not claim a definitive clinical diagnosis.
* Do not invent dataset fields, units, or performance numbers.
* Do not choose a feature because it sounds medically familiar. Age was measured and then rejected because its separation was weak.
* Do not tune anything on the test set.
* Do not add a neural network to make the method sound more advanced. Fuzzy inference is the method.

---

## 10. Status

Done:

* Dataset inspected and documented
* Cleaning, with a row-level log
* Pooled stratified 80/20 split (`seed=42`)
* Feature selection: `PSAD`, `f_t_psa`, `PV`
* Fuzzification and the 27-rule system
* Test-set comparison against logistic regression, SVM, a decision tree, and XGBoost
* 100 repeated splits and paper charts under `charts/`

Not in this project:

* Neural learning
* ANFIS
* A second neural classifier

Code for every step is explained in `Docs/Code.md`.

The earlier neuro-fuzzy roadmap is kept only as a backup in `Docs/backup/2026-10-05/`.
