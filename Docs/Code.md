# What the code does, and why

This file walks through every step of the program: what that step is, why it is there, and what it writes. The numbers in the last section come from running `main.py` on the downloaded files. They are not filled in by hand.

The proposed method is a fuzzy inference system, plus fuzzy boost: the same shallow XGBoost, given the 27 rule firing strengths and age. Logistic regression, an RBF support vector machine, a decision tree, and a plain XGBoost are the comparison. There is no neural network and no ANFIS step. The earlier roadmap that included neural learning is only in `Docs/backup/2026-10-05/`.

## What the program is

The program classifies the recorded prostate-cancer label in a clinical table. It does not stage a tumor, read a biopsy, or produce a diagnosis a clinic should act on.

Both downloaded files are stacked into one pool of 598 patients, then split:

| Split | Role | Patients |
| --- | --- | ---: |
| Training set | Every choice is fit here | 478 (206 positive, 272 negative) |
| Test set | Scored once, at the end | 120 (52 positive, 68 negative) |

The split is stratified by the label, 80/20, with `random_state=42`. The file a row came from is kept only as `source_file`. Rows had already been moved between `data-1.csv` and `data-2.csv` by hand, so the file boundary is no longer a cohort. The test set is not the published 300-patient external cohort, and a paper has to say that.

The test set does not set a median, a scale, a membership knot, a selected feature, a fuzzy consequent, or a classifier parameter.

Run it from the project folder with the project virtualenv:

```text
.\Scripts\python.exe main.py
```

`main.py` calls the steps in order: clean and split, select features, fuzzify, apply the rules, compare, repeat 100 splits, then write charts under `charts/`.

## Step 1 — `src/preprocess.py`

### What it is

This step loads both files into one pool, removes values that cannot be the measurement the column claims to be, rebuilds the ratio columns from their own formulas, splits the pool, fills holes with the training median, and writes a scaled copy for later models that need one.

The target `Prostate Cancer` is copied to `prostate_cancer` as 0 for `negative` and 1 for `positive`. `No.` is copied unchanged, including repeated values. It is never a feature. A separate `row_id` is added only so two rows with the same `No.` can be joined. No repeated id is deleted.

### Why the invalid cells are blanked

A neutrophil count larger than the total white-cell count is not an absolute count. A platelet value of 1.75 is not a credible platelet count in these units (the next lowest platelet value in the files is 71). Those cells are set to missing and then filled. Leaving them in would treat a data-entry error as a biological signal.

The very large PSA values are kept. A PSA in the hundreds or thousands can be a real measurement, and both of those rows are labeled positive. Deleting them because they are extreme would throw away the signal.

### Why ratios are recomputed

`PSAD` is PSA density, `tPSA / PV`. `AST/ALT` is `AST / ALT`. The stored numbers are often just rounded. A few are not: one row stores `AST/ALT = 1.50` where `10 / 15` is about `0.67`. The code replaces the stored ratio with the formula so later rules are not using a broken ratio. Gaps larger than `0.005` are written to the cleaning log; smaller gaps are rounding and are not listed one by one.

`f_t_psa` (`fPSA / tPSA`) is created here. The raw file has no free-to-total column. It is a derived feature, not a new lab test.

### Why the median comes from the training set only

Each missing cell is filled with the median of that column on the 478 training patients, including when the hole is in the test set. If the test median were used, the final test set would have helped prepare the data.

Feature columns are cast to float before filling. Training medians can be half-integers (for example `66.5`), and an integer column cannot hold that value.

### Why a scaled copy exists

`RobustScaler` is fit on the training set and applied to both. It is saved in `preprocess.joblib` together with the medians. The fuzzy system does not use this scaled table. Membership functions are defined on the original units (a PSA density, a ratio, a volume), because those units are what a rule sentence refers to. The comparison step scales again, and only for logistic regression and the SVM, inside those models.

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/train_clean.csv` | 478 cleaned training rows, original units |
| `Data/processed/test_clean.csv` | 120 cleaned test rows, original units |
| `Data/processed/train_scaled.csv` | Same rows after the robust scaler |
| `Data/processed/test_scaled.csv` | Same, with the training scaler |
| `Data/processed/cleaning_log.csv` | Every blanked, recomputed, or imputed cell |
| `Data/processed/preprocess.joblib` | Medians, scaler, column list, and split settings |

The log from the current files: one impossible monocyte and two impossible neutrophils in `data-1`; one impossible platelet and one impossible neutrophil in `data-2`; two stored `PSAD` values and 67 stored `AST/ALT` values recomputed in `data-1`; 17 stored `AST/ALT` values recomputed in `data-2`; then training medians fill the remaining holes (four cells in train, one in test).

## Step 2 — `src/features.py`

### What it is

This step chooses the inputs of the fuzzy system. It also builds three blood-count ratios and tests them. All scores use the training set only. The test file is only cut down to the columns that were accepted.

### Why so few inputs

Each selected feature gets three linguistic terms. The rule base is every combination of those terms. Two features make 9 rules, three make 27, four make 81. Twenty-seven rules is a complete grid that can still be read, and it is a reasonable size next to 478 training patients. A fourth weak feature would multiply the rules without adding a clear separation. That is why the cutoff is strict.

### Why these extra ratios were tried

`nlr` (`PMN / LYMPH`), `mlr` (`MONOC / LYMPH`), and `plr` (`PLT / LYMPH`) are common inflammation ratios, and the columns to compute them are already in the file. They were tested rather than assumed. On this training set their AUCs are about 0.51 to 0.52, so they were rejected. `f_t_psa` was kept.

### Why a feature is accepted or rejected

Features are considered from the strongest univariate separation to the weakest. The AUC is oriented so that 0.5 means no separation: if the raw AUC is below 0.5, the reported AUC is `1 - raw AUC` and the direction is recorded as lower in cancer.

Three rules, in order:

1. Reject a feature whose Spearman correlation with an already accepted feature is at least 0.80. Otherwise the same fact enters the rules twice.
2. Reject a feature that is that highly correlated with a feature already rejected by rule 1. This drops the raw copy of a marker that was itself a copy. `fPSA` is rejected here because it tracks `tPSA`, and `tPSA` was already rejected because it tracks `PSAD`.
3. Reject a feature whose oriented AUC is below 0.60. A Mann-Whitney p-value is stored so the table shows the strength of the shift, but it is not the acceptance rule. Age, red-cell count, and hemoglobin do differ between classes, and they are still rejected: the shift is too small for a fuzzy input. Median age is 73 versus 71 years.

### Why these three survived

| Feature | Oriented AUC | Direction in cancer | Why it stays |
| --- | ---: | --- | --- |
| `PSAD` | 0.789 | higher | Strongest single column. It is total PSA divided by prostate volume, so it already carries both. |
| `f_t_psa` | 0.665 | lower | The free-to-total ratio. Raw free PSA was rejected as a copy of total PSA. This ratio is the part that is not a copy, and it is lower in cancer, which matches the usual clinical reading of that ratio. |
| `PV` | 0.619 | lower | Prostate volume. Not redundant with `PSAD` at the 0.80 line (Spearman about −0.36). |

They stay because each one separates the classes and they are not duplicates. The claim is not that they improve a linear model. The fuzzy system can use a direction that a linear coefficient barely uses.

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/feature_scores.csv` | Every candidate, with AUC, direction, p-value, and the accept/reject reason |
| `Data/processed/train_model.csv` | Id, cohort, the three features, and the label |
| `Data/processed/test_model.csv` | The same columns for the test set |

The longer account of the ranking is `Docs/FeatureSelection.md`.

## Step 3 — `src/fuzzify.py`

### What it is

This step turns each of the three numbers into three degrees: `low`, `medium`, and `high`. A degree is between 0 and 1. For one feature, the three degrees sum to 1 at every patient. That is a Ruspini partition, and the rule step relies on it.

The knots are the training 25th percentile, the median, and the 75th percentile. The same three numbers are applied to the test set. The cancer label is not used to place a knot.

| Feature | 25th, fully low at or below | Median, medium peaks here | 75th, fully high at or above |
| --- | ---: | ---: | ---: |
| `PSAD` | 0.134 | 0.262 | 0.608 |
| `f_t_psa` | 0.101 | 0.135 | 0.207 |
| `PV` | 36.46 | 53.52 | 74.74 |

### Why quartiles, and why not a Gaussian

Quartiles are a property of the training measurements. They do not ask which knot would classify cancer best, so the membership shapes are not secretly trained on the label.

A Gaussian centered on those same quartiles was tried as an idea and rejected. `PSAD` is heavily skewed. At the 90th percentile, a Gaussian fit to these quartiles gives a membership near 0 for low, medium, and high. The largest density values, which are the strongest `PSAD` signal, would have dropped out of the system. The piecewise linear shoulders do the opposite: everything at or above the 75th percentile is fully `high`, including the extreme densities, and everything at or below the 25th percentile is fully `low`.

Between the 25th percentile and the median, `low` falls in a straight line and `medium` rises. Between the median and the 75th percentile, `medium` falls and `high` rises. The names `low` / `medium` / `high` describe the size of the number. They do not yet mean cancer risk. A high `PSAD` points toward cancer; a high `f_t_psa` points away from it. That direction is applied in the next step, not in the name of the term.

The figure is `charts/membership_functions.png`. The horizontal `high` line continues to the right of the 75th percentile. The axis stops a little past that shoulder so the middle triangle is visible.

### Why the class averages are printed

After fuzzification, the training positive patients have higher average membership in `PSAD_high`, `f_t_psa_low`, and `PV_low` than the negative patients. That check confirms the directions already measured in step 2. The knots were not moved to make this table look better.

| Class | `PSAD` low | medium | high | `f_t_psa` low | medium | high | `PV` low | medium | high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| negative | 0.563 | 0.261 | 0.176 | 0.275 | 0.239 | 0.486 | 0.310 | 0.239 | 0.451 |
| positive | 0.182 | 0.226 | 0.591 | 0.459 | 0.330 | 0.210 | 0.465 | 0.264 | 0.271 |

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/membership_params.csv` | The three knots per feature |
| `Data/processed/train_memberships.csv` | Clean columns plus nine degree columns |
| `Data/processed/test_memberships.csv` | The same, using the training knots |

The shape formulas are in `Docs/Fuzzification.md`. The membership figure is written later by `src/charts.py`.

## Step 4 — `src/fuzzy_rules.py`

### What it is

This step is the classifier. It still builds all 27 combinations of the linguistic terms. The consequent of each rule is now a weight from a ridge regression fit on the training labels. The 27 firing strengths are the inputs. There is no extra intercept, and there is no neural network. The penalty is fixed at 1 so a rule that almost never fires cannot take an extreme weight.

The old direction points stay in the table as `direction_consequent`. They are a reference, scored at 0.5, and they are not the model being compared.

A patient's firing strength for a rule is the product of the three memberships. Those 27 strengths sum to 1, and the code checks that. The score is the weighted sum of the learned consequents. A few learned weights fall slightly outside 0–1. That is what an unbounded ridge fit does. They are not clipped after seeing the test set.

The decision cutoff is the one that maximizes accuracy on the training out-of-fold scores. Ties go to the cutoff closest to 0.5. On this run that cutoff is 0.456. It is chosen before the test set is scored, then applied once.

Knots and consequents are re-fit inside each cross-validation fold. The test set uses the knots and consequents fit on all 478 training patients.

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/fuzzy_rules.csv` | All 27 rules, the direction reference, and the learned consequent |
| `Data/processed/train_fuzzy_scores.csv` | Training score and 0/1 prediction |
| `Data/processed/test_fuzzy_scores.csv` | Test score and 0/1 prediction |
| `Data/processed/fuzzy_metrics.csv` | Learned system: cross-validation, training fit, and holdout |
| `Data/processed/fuzzy_ablation.csv` | The old direction-point rules at cutoff 0.5 |

On the 120-patient test set the learned system scores 0.758 accuracy, 0.654 sensitivity, 0.838 specificity, 0.701 F1, and 0.793 ROC-AUC (34 true positives, 57 true negatives, 11 false positives, 18 false negatives). The old direction rules at 0.5 score 0.717 accuracy, F1 0.679, and ROC-AUC 0.736. Learning the consequents improved ranking and accuracy on this split.

## Step 5 — `src/compare.py`

### What it is

This step fits four classifiers on the same three columns, and fuzzy boost, and puts them next to the fuzzy scores.

Fuzzy boost uses the same depth-2 XGBoost as the plain booster (100 rounds, learning rate 0.1, minimum child weight 5). It also receives the 27 rule firing strengths, with quartile knots refit inside each training fold, and age. Age failed the univariate screen (oriented AUC about 0.55), so it is not a rule input. It stayed because the booster's training out-of-fold residuals still tracked age, and adding it raised accuracy, F1, and ROC-AUC on the training folds of other splits. Red-cell count had a residual correlation of the same size and did not raise those scores, so it stays out.

| Model | What it is | Why this setting |
| --- | --- | --- |
| Logistic regression | A straight weighted sum of the three features, then a logistic curve. | The linear baseline. `C=1`. The three columns are standardized with a `StandardScaler` fit on the training set, because a volume near 50 and a ratio near 0.14 are not comparable units. |
| SVM | An RBF kernel support vector machine. | A nonlinear baseline that does not use rules. `C=1`, `gamma="scale"`, with the same standardization. |
| Decision tree | One tree of if-then splits. | The closest ordinary relative of a rule system. Depth is 3, so a path can use each feature. A leaf must hold at least 15 training patients, so the tree cannot isolate one person. |
| XGBoost | One hundred shallow trees added together. | The boosted-tree baseline. Each tree has depth 2, the learning rate is 0.1, and a leaf must carry child weight at least 5. |
| Fuzzy boost | That same booster, on the three features, age, and the 27 rule firing strengths. | Age is fixed from the primary training residuals. Knots are refit inside each fold. The cutoff is still the training accuracy cutoff. |

Nothing in this list was chosen by looking at the test set.

Every model, including the fuzzy system, uses one cutoff: the value that maximizes accuracy on its own training out-of-fold scores. Ties go to the cutoff closest to that model's natural center, 0.5 for probabilities and 0 for the SVM decision function. The same frozen cutoff is then applied to the test set. ROC-AUC does not use the cutoff.

### Why three views are saved

| View | Who | What it answers |
| --- | --- | --- |
| `cross_validation` | Every model, training set | Stratified 5-fold out-of-fold scores (`shuffle`, `random_state=42`). For the fuzzy system, knots and consequents are re-fit inside each fold. |
| `training_fit` | Every model, training set | The same 478 rows the model was fit on. Useful only to see memorization. XGBoost's training ROC-AUC is 0.881 and its test ROC-AUC is 0.780. The training number is not a result. |
| `held_out` | Every model, test set | Fit on all 478 training patients, scored once on the remaining 120. |

ROC-AUC is the comparison that does not depend on the cutoff. Accuracy, sensitivity, specificity, precision, and F1 describe the one operating point above.

### Why the importances are saved

`comparison_importance.csv` records how each fitted model uses the three columns.

| Model | `PSAD` | `f_t_psa` | `PV` | How to read the number |
| --- | ---: | ---: | ---: | --- |
| Logistic regression | +3.452 | −0.541 | −0.194 | Coefficient after standardization. Positive means a higher value raises cancer log-odds. Almost all of the linear signal is `PSAD`. |
| Decision tree | 0.879 | 0.027 | 0.094 | Normalized impurity reduction. The tree mostly splits on `PSAD`. |
| XGBoost | 0.660 | 0.186 | 0.154 | Normalized importance. Still led by `PSAD`. |

The RBF SVM has no single coefficient per feature, so it is not in that table.

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/comparison_metrics.csv` | Every model, cohort, and view |
| `Data/processed/comparison_scores.csv` | The continuous score of each model for each patient |
| `Data/processed/comparison_importance.csv` | Coefficients and split importances |

## Step 6 — `src/repeated.py`

### What it is

This step repeats the whole comparison on 100 seeded stratified 80/20 splits of the same cleaned pool. Each repeat redoes everything that is fit to data on that repeat's training set: medians, knots, consequents, the four classifiers, and every cutoff. The model inputs stay `PSAD`, `f_t_psa`, and `PV`. Feature selection is still rerun and recorded, so the report shows how often that choice would have come out the same (56 of 100 repeats on the current run).

The spread across repeats is a stability range. The test sets overlap between repeats, so it is not a confidence interval and not a significance test.

### What it writes

| File | What it is |
| --- | --- |
| `Data/processed/repeated_metrics.csv` | One row per seed × model |
| `Data/processed/repeated_summary.csv` | Mean, sd, median, and 2.5/97.5 percentiles |
| `Data/processed/repeated_paired.csv` | Fuzzy minus each baseline, per metric |
| `Data/processed/repeated_feature_selection.csv` | Accepted features on each seed |

## Step 7 — `src/charts.py`

This step only reads the CSVs above and writes figures under `charts/`:

| Figure | Content |
| --- | --- |
| `charts/membership_functions.png` | The three Ruspini partitions |
| `charts/metrics_comparison.png` | Holdout metrics by model |
| `charts/accuracy_f1_auc.png` | Accuracy, F1, and ROC-AUC bars |
| `charts/roc_curves.png` | Holdout ROC curves |
| `charts/confusion_matrix_fuzzy.png` | Fuzzy holdout confusion matrix |
| `charts/repeated_metrics.png` | Metric spread across 100 splits |
| `charts/repeated_mean_sd.png` | Mean ± sd across repeats |

## Holdout result

These rows are the result on the 120-patient test set from the seed-42 split. Each model was fit on the 478 training patients only. The cutoff for each model was frozen from that model's training cross-validation.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fuzzy system | 0.758 | 0.654 | 0.838 | 0.756 | 0.701 | 0.793 | 34 | 57 | 11 | 18 |
| Fuzzy boost | 0.817 | 0.673 | 0.926 | 0.875 | 0.761 | 0.834 | 35 | 63 | 5 | 17 |
| Logistic regression | 0.692 | 0.615 | 0.750 | 0.653 | 0.634 | 0.731 | 32 | 51 | 17 | 20 |
| SVM (RBF) | 0.683 | 0.577 | 0.765 | 0.652 | 0.612 | 0.698 | 30 | 52 | 16 | 22 |
| Decision tree | 0.700 | 0.365 | 0.956 | 0.864 | 0.514 | 0.715 | 19 | 65 | 3 | 33 |
| XGBoost | 0.742 | 0.577 | 0.868 | 0.769 | 0.659 | 0.780 | 30 | 59 | 9 | 22 |

How to read that table:

* **This is not proof that the fuzzy system is better.** The holdout has 120 patients. The split is random, not an independent published external cohort. A paper can report this table. It cannot call it the dataset's independent external validation.
* **Accuracy, F1, and ROC-AUC.** At the frozen accuracy cutoff, fuzzy boost is the highest of the six (0.817 accuracy, 0.761 F1, 0.834 ROC-AUC). The fuzzy system is next on accuracy and F1 (0.758 and 0.701). Plain XGBoost is next on ROC-AUC (0.780).
* **The old rules.** Direction-point consequents at 0.5, on this same holdout, reach accuracy 0.717 and F1 0.679, with ROC-AUC 0.736. The learned consequents are the better ranking model on this split.
* **Memorization check.** XGBoost's training ROC-AUC is 0.881 and its holdout ROC-AUC is 0.780. Quote the holdout row.

Training cross-validation, for context only:

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fuzzy system | 0.743 | 0.655 | 0.809 | 0.687 | 0.780 |
| Fuzzy boost | 0.757 | 0.641 | 0.846 | 0.695 | 0.796 |
| Logistic regression | 0.711 | 0.689 | 0.728 | 0.673 | 0.740 |
| SVM (RBF) | 0.688 | 0.607 | 0.750 | 0.627 | 0.723 |
| Decision tree | 0.720 | 0.393 | 0.967 | 0.547 | 0.763 |
| XGBoost | 0.738 | 0.587 | 0.853 | 0.659 | 0.784 |

Across 100 repeated splits, mean test accuracy is about 0.757 for fuzzy boost, 0.733 for the fuzzy system, and 0.732 for XGBoost. Mean ROC-AUC is about 0.813 for fuzzy boost, 0.785 for fuzzy, and 0.787 for XGBoost. Fuzzy boost is ahead of the fuzzy system on accuracy in 71 of 100 splits and on ROC-AUC in 87. The single seed-42 holdout is one draw from that spread, and it sits at the high end of fuzzy boost's accuracy range.

## What this code deliberately does not do

* It does not train a neural network, and it does not move membership knots by gradient descent.
* It does not search hyperparameters on the test set.
* It does not delete a row because its `No.` is repeated.
* It does not add age, blood counts, or the inflammation ratios after they failed the selection rules.
* It does not claim that the accuracy cutoff is a clinical decision threshold.
* It does not treat the file boundary between `data-1.csv` and `data-2.csv` as the train/test split.
