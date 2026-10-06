"""Compare the fuzzy system with four classifiers on the same three features.

Every model is fit on the training set only. The test set is scored once,
after fitting, and is not used to choose a setting.

Logistic regression and the RBF SVM are scaled with a StandardScaler fit on
the training set, because PSAD, the free-to-total ratio, and prostate
volume are not in the same units. The tree and XGBoost split on thresholds,
so they keep the original units.

Settings are fixed in advance:

- logistic regression and the SVM use the library defaults C=1
- the tree is depth 3 with at least 15 patients in a leaf, so a path can
  use each selected feature without memorizing single patients
- XGBoost uses depth-2 trees, 100 rounds, and a minimum child weight of 5,
  which is the same idea: a boosted tree with capacity close to this sample

Every model, the fuzzy system included, gets its cutoff the same way: the
value that maximizes accuracy on its own stratified 5-fold out-of-fold
scores on the training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.fuzzify import SELECTED_FEATURES
from src.fuzzy_rules import classification_metrics, threshold_for_accuracy
from src.preprocess import OUTPUT_DIR, ROW_ID_COLUMN, TARGET_CODE_COLUMN

RANDOM_STATE = 42
CV_SPLITS = 5

# Higher score means higher cancer risk. The SVM score is centered on 0.
PROBABILITY_CENTER = 0.5
SVM_CENTER = 0.0

MODEL_LABELS = {
    "fuzzy": "Fuzzy system",
    "logistic_regression": "Logistic regression",
    "svm": "SVM (RBF)",
    "decision_tree": "Decision tree",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "fuzzy": "#54A24B",
    "logistic_regression": "#4C78A8",
    "svm": "#F58518",
    "decision_tree": "#E45756",
    "xgboost": "#72B7B2",
}


@dataclass
class FittedClassifier:
    estimator: object
    kind: str
    threshold: float
    out_of_fold: np.ndarray

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return _positive_score(self.estimator, frame, self.kind)


def _estimators() -> dict[str, tuple[object, float, str]]:
    """Return name -> (estimator, cutoff tie-break center, score kind)."""
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
            ),
        ]
    )
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )
    boosting = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    return {
        "logistic_regression": (logistic, PROBABILITY_CENTER, "proba"),
        "svm": (svm, SVM_CENTER, "decision"),
        "decision_tree": (tree, PROBABILITY_CENTER, "proba"),
        "xgboost": (boosting, PROBABILITY_CENTER, "proba"),
    }


CLASSIFIER_NAMES = tuple(_estimators())


def _positive_score(estimator: object, frame: pd.DataFrame, kind: str) -> np.ndarray:
    values = frame[SELECTED_FEATURES]
    if kind == "proba":
        return estimator.predict_proba(values)[:, 1]
    return estimator.decision_function(values)


def _cv_score(
    estimator: object,
    features: pd.DataFrame,
    target: np.ndarray,
    kind: str,
) -> np.ndarray:
    folds = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    method = "predict_proba" if kind == "proba" else "decision_function"
    predicted = cross_val_predict(
        estimator,
        features,
        target,
        cv=folds,
        method=method,
        n_jobs=1,
    )
    if kind == "proba":
        return predicted[:, 1]
    return predicted


def fit_classifiers(train: pd.DataFrame, target: np.ndarray) -> dict[str, FittedClassifier]:
    fitted = {}
    for name, (estimator, center, kind) in _estimators().items():
        out_of_fold = _cv_score(estimator, train[SELECTED_FEATURES], target, kind)
        threshold = threshold_for_accuracy(target, out_of_fold, center=center)
        estimator.fit(train[SELECTED_FEATURES], target)
        fitted[name] = FittedClassifier(estimator, kind, threshold, out_of_fold)
    return fitted


def _importance_rows(fitted: dict[str, FittedClassifier]) -> pd.DataFrame:
    rows = []
    logistic = fitted["logistic_regression"].estimator.named_steps["model"]
    for feature, weight in zip(SELECTED_FEATURES, logistic.coef_[0]):
        rows.append(
            {
                "model": "logistic_regression",
                "feature": feature,
                "value": float(weight),
                "meaning": "coefficient on the training-standardized feature",
            }
        )
    for model_name in ("decision_tree", "xgboost"):
        importances = fitted[model_name].estimator.feature_importances_
        for feature, weight in zip(SELECTED_FEATURES, importances):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "value": float(weight),
                    "meaning": "normalized importance of splits on this feature",
                }
            )
    return pd.DataFrame(rows)


def _fuzzy_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(OUTPUT_DIR / "fuzzy_metrics.csv")
    rows = []
    for record in metrics.to_dict(orient="records"):
        cohort = record.pop("cohort")
        view = record.pop("view")
        rows.append({"model": "fuzzy", "cohort": cohort, "view": view, **record})
    return rows


def _fuzzy_scores(frame: pd.DataFrame, file_name: str) -> np.ndarray:
    scores = pd.read_csv(OUTPUT_DIR / file_name)
    merged = frame.merge(
        scores.loc[:, [ROW_ID_COLUMN, "fuzzy_score"]],
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if merged["fuzzy_score"].isna().any():
        raise AssertionError(f"{file_name} does not match the model rows")
    return merged["fuzzy_score"].to_numpy()


def run() -> None:
    train = pd.read_csv(OUTPUT_DIR / "train_model.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_model.csv")
    y_train = train[TARGET_CODE_COLUMN].to_numpy()
    y_test = test[TARGET_CODE_COLUMN].to_numpy()

    id_columns = [ROW_ID_COLUMN, "No.", "cohort", TARGET_CODE_COLUMN]
    score_frames = {
        "train": train.loc[:, id_columns].copy(),
        "test": test.loc[:, id_columns].copy(),
    }
    score_frames["train"]["fuzzy"] = _fuzzy_scores(train, "train_fuzzy_scores.csv")
    score_frames["test"]["fuzzy"] = _fuzzy_scores(test, "test_fuzzy_scores.csv")

    rows = _fuzzy_rows()
    fitted = fit_classifiers(train, y_train)
    for name, model in fitted.items():
        train_score = model.score(train)
        test_score = model.score(test)
        test_metrics = classification_metrics(y_test, test_score, model.threshold)
        rows.extend(
            [
                {
                    "model": name,
                    "cohort": "train",
                    "view": "cross_validation",
                    **classification_metrics(y_train, model.out_of_fold, model.threshold),
                },
                {
                    "model": name,
                    "cohort": "train",
                    "view": "training_fit",
                    **classification_metrics(y_train, train_score, model.threshold),
                },
                {"model": name, "cohort": "test", "view": "held_out", **test_metrics},
            ]
        )
        score_frames["train"][name] = train_score
        score_frames["test"][name] = test_score

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
    _importance_rows(fitted).to_csv(OUTPUT_DIR / "comparison_importance.csv", index=False)
    pd.concat(score_frames.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "comparison_scores.csv",
        index=False,
    )


if __name__ == "__main__":
    run()
