# Cancer Detection Research Project

## 1. Project Goal

We are preparing a research paper for the **2026 International Congress on Cancer Prevention at University of Zanjan**.

The instructor's main requirement is:

> Apply **Fuzzy Logic** in an **Artificial Intelligence / Machine Learning context** for **cancer diagnosis/detection**.

The preferred cancer type is currently **prostate cancer**, with **breast cancer** as a possible alternative.

The project should ideally be a **research/experimental paper**, but a review paper is acceptable if implementation or data access becomes a problem.

Maximum paper length: **10 pages**.

The paper format should be based on papers/templates from previous years of the same conference.

---

## 2. Current Dataset Choice

The currently preferred dataset is:

**Prostate Cancer Data - Mendeley Data**

URL:

https://data.mendeley.com/datasets/6db5t5b52x/1

This dataset was selected because it appears suitable for binary prostate cancer classification and, importantly, contains an **independent validation cohort**, which can make the experimental design stronger than a simple random train/test split.

Current understanding:

* Training/internal cohort: approximately 298 patients
* Independent validation cohort: approximately 300 patients
* Binary cancer/non-cancer outcome
* Dataset is clinical/tabular rather than image-based

**Important:** Before implementation, inspect the actual files and verify:

* exact feature names
* target column
* missing values
* categorical/numerical variables
* class distribution
* train/validation structure
* whether the independent cohort is genuinely external and suitable for final validation

Do **not** invent dataset properties. Base all implementation decisions on the actual dataset.

---

## 3. Main Research Direction

The current preferred approach is:

> **Neuro-Fuzzy / Fuzzy Logic-based Machine Learning for Prostate Cancer Detection**

The main idea is to keep **Fuzzy Logic as the central component**, while using a neural network to learn/optimize fuzzy-system parameters.

Preferred model family:

### ANFIS

**Adaptive Neuro-Fuzzy Inference System**

Conceptually:

```text
Clinical Dataset
      ↓
Data Preprocessing
      ↓
Feature Selection / Engineering
      ↓
Fuzzification
      ↓
Fuzzy Rules
      ↕
Neural Learning / Parameter Optimization
      ↓
Cancer Classification
      ↓
Evaluation
```

The project should not simply add a generic neural network alongside fuzzy logic. The neural component should have a meaningful role in learning or optimizing the fuzzy system.

---

## 4. Experimental Design

We should compare multiple approaches so the paper demonstrates whether the proposed neuro-fuzzy model actually provides value.

Recommended baselines:

1. Logistic Regression
2. Traditional Fuzzy Inference System
3. Neuro-Fuzzy / ANFIS (**proposed model**)

Potential additional ML baselines if time permits:

* Random Forest
* SVM
* A small conventional Neural Network

Evaluation metrics:

* Accuracy
* Sensitivity / Recall
* Specificity
* Precision
* F1-score
* ROC-AUC

The independent validation cohort should be used for **external/final validation** rather than relying only on a random train/test split.

---

## 5. Why Fuzzy Logic Matters

The paper should explain why fuzzy logic is relevant instead of treating it as decoration.

Clinical variables often have gradual rather than binary meanings.

For example:

```text
Age
→ young / middle / old

PSA
→ low / medium / high

Risk-related clinical features
→ low / medium / high
```

A fuzzy system can represent rules such as:

```text
IF PSA is high
AND Age is high
AND other risk factors are high
THEN Cancer Risk is high
```

The exact variables and rules must be determined **after inspecting the real dataset**.

---

## 6. Proposed Paper Contribution

The paper should aim to present:

> A fuzzy-logic-centered AI model, enhanced through neural learning, for prostate cancer detection, with comparison against conventional classification approaches and evaluation on an independent validation cohort.

Possible title direction:

**English:**

> Design and Evaluation of a Neuro-Fuzzy Model for Prostate Cancer Detection Using Clinical Data

Possible Persian equivalent:

> طراحی و ارزیابی مدل عصبی-فازی برای تشخیص سرطان پروستات با استفاده از داده‌های بالینی

The final title can be adjusted after the exact methodology is finalized.

---

## 7. Python vs Rust Decision

We discussed implementing the project in Rust.

### Decision:

**Python is preferred for the research implementation.**

Reason:

The difficult part is not implementing fuzzy mathematics itself. The main issue is the ecosystem:

* Data preprocessing
* ML libraries
* Fuzzy logic libraries
* Visualization
* Experimentation
* Metrics
* Statistical analysis
* Reproducing research workflows

Python has a much stronger ecosystem for this work.

Rust is technically possible, but it would add unnecessary implementation effort, especially if ANFIS or a similar neuro-fuzzy method must be implemented manually.

Rust can be considered later for deployment/API work, but it is **not the preferred language for the research prototype**.

Likely Python stack:

```text
Python
├── pandas
├── numpy
├── scikit-learn
├── matplotlib
├── seaborn (optional)
└── fuzzy / ANFIS implementation
```

Do not commit to a specific ANFIS library until its maturity and API are checked.

---

## 8. Expected Research Workflow

### Phase 1 - Dataset inspection

* Download the Mendeley dataset
* Inspect all files
* Determine target and features
* Check missing values
* Check class balance
* Understand the independent validation cohort

### Phase 2 - Literature review

Search recent papers on:

* Fuzzy Logic + cancer diagnosis
* Neuro-Fuzzy + prostate cancer
* ANFIS + prostate cancer
* Explainable AI + cancer diagnosis
* Fuzzy Machine Learning for medical diagnosis

The literature review should help define:

* feature selection strategy
* fuzzy membership functions
* model architecture
* evaluation methodology
* novelty/positioning

### Phase 3 - Baselines

Implement conventional models first:

```text
Logistic Regression
Random Forest / SVM
(optional Neural Network)
```

### Phase 4 - Fuzzy model

Build a traditional fuzzy inference system:

```text
Input features
→ membership functions
→ fuzzy rules
→ inference
→ defuzzification/classification
```

### Phase 5 - Neuro-Fuzzy model

Implement ANFIS or an equivalent neuro-fuzzy architecture.

The neural component should learn/optimize fuzzy parameters rather than merely being an unrelated second classifier.

### Phase 6 - Evaluation

Evaluate on:

* validation data
* independent/external cohort

Generate:

* confusion matrices
* ROC curves
* metric tables
* comparison tables

### Phase 7 - Paper

Suggested structure:

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Related Work
6. Materials and Methods
7. Proposed Fuzzy/Neuro-Fuzzy Model
8. Experimental Results
9. Discussion
10. Conclusion
11. References

Keep the final paper within **10 pages**.

---

## 9. Important Constraints

* This is a **research paper**, not a production medical diagnostic system.
* Do not claim that the model provides definitive clinical diagnosis.
* Prefer wording such as **classification**, **detection**, or **risk prediction** depending on the actual target variable.
* Do not fabricate missing clinical information.
* Do not choose features simply because they sound medically relevant. Use only available dataset features.
* Do not report fabricated performance values. All metrics must come from actual experiments.
* Avoid unnecessary model complexity just to make the paper sound more advanced.
* The core methodological contribution must remain **Fuzzy Logic / Neuro-Fuzzy AI**.

---

## 10. Current Status

### Decided

* Cancer type: **Prostate cancer preferred**
* Dataset: **Mendeley Prostate Cancer Data**
* Core methodology: **Fuzzy Logic + Neural Learning**
* Preferred model: **ANFIS / Neuro-Fuzzy**
* Language: **Python**
* Research style: **Experimental research paper**
* Maximum length: **10 pages**

### Not yet decided

* Exact feature subset
* Exact preprocessing pipeline
* Exact membership functions
* Fuzzy rule-generation method
* Exact ANFIS implementation
* Baseline model set
* Feature selection method
* Hyperparameter optimization
* Final paper title
* Exact conference formatting based on previous papers

### Immediate next step

**Inspect the actual Mendeley dataset before designing the model.**

Do not start writing the final methodology until the dataset structure has been verified.
