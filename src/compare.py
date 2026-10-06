"""Compare the fuzzy system with standard classifiers and with fuzzy boost.

Every model is fit on the training set only. The test set is scored once,
after fitting, and is not used to choose a setting.

The four standard classifiers use the three clinical features only, with
settings fixed in advance:

- logistic regression and the RBF SVM use C=1, on features standardized
  from the training set
- the tree is depth 3 with at least 15 patients in a leaf
- XGBoost uses depth-2 trees, 100 rounds, learning rate 0.1, and a minimum
  child weight of 5

Fuzzy boost uses that same tree capacity. Its inputs add two things the
other models do not take:

- the 27 rule firing strengths. Knots come from ACTIVE_MEMBERSHIP in
  src/fuzzify.py. Quartile knots are fit on the training rows only, and
  again inside each cross-validation fold. Literature knots stay fixed.
- age. Age failed the univariate screen, so it is not a fuzzy input. It
  stayed in the booster because the training out-of-fold residuals still
  tracked age, and adding it raised accuracy, F1, and ROC-AUC on the
  training folds of the other splits. Red-cell count had a residual
  correlation of the same size and did not raise those scores, so it stays
  out.

Every model gets its cutoff the same way: the value that maximizes accuracy
on its own stratified 5-fold out-of-fold scores on the training set.
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

from src import fuzzify
from src.fuzzify import SELECTED_FEATURES, _apply, _fit_all
from src.fuzzy_rules import RULES, classification_metrics, firing_matrix, threshold_for_accuracy
from src.preprocess import OUTPUT_DIR, ROW_ID_COLUMN, TARGET_CODE_COLUMN

RANDOM_STATE = 42
CV_SPLITS = 5

# Higher score means higher cancer risk. The SVM score is centered on 0.
PROBABILITY_CENTER = 0.5
SVM_CENTER = 0.0
# Not a fuzzy input. See the module note on why the booster still uses it.
BOOST_EXTRA_FEATURES = ("Age",)

MODEL_ORDER = ("fuzzy", "fuzzy_boost", "logistic_regression", "svm", "decision_tree", "xgboost")
MODEL_LABELS = {
    "fuzzy": "Fuzzy system",
    "fuzzy_boost": "Fuzzy boost",
    "logistic_regression": "Logistic regression",
    "svm": "SVM (RBF)",
    "decision_tree": "Decision tree",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "fuzzy": "#54A24B",
    "fuzzy_boost": "#B279A2",
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
    knots: dict[str, tuple[float, float, float]] | None = None

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return _positive_score(self.estimator, _model_matrix(frame, self.knots), self.kind)


def _xgboost() -> XGBClassifier:
    return XGBClassifier(
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
    return {
        "logistic_regression": (logistic, PROBABILITY_CENTER, "proba"),
        "svm": (svm, SVM_CENTER, "decision"),
        "decision_tree": (tree, PROBABILITY_CENTER, "proba"),
        "xgboost": (_xgboost(), PROBABILITY_CENTER, "proba"),
        "fuzzy_boost": (_xgboost(), PROBABILITY_CENTER, "proba"),
    }


CLASSIFIER_NAMES = tuple(_estimators())


def _rule_names() -> list[str]:
    return [
        f"rule_{int(rule.rule_id):02d}_PSAD_{rule.PSAD}_ft_{rule.f_t_psa}_PV_{rule.PV}"
        for rule in RULES.itertuples(index=False)
    ]


def _model_matrix(
    frame: pd.DataFrame,
    knots: dict[str, tuple[float, float, float]] | None,
) -> pd.DataFrame:
    """The three clinical features, or those plus age and the 27 rule firing strengths."""
    raw = frame.loc[:, list(SELECTED_FEATURES)]
    if knots is None:
        return raw
    missing = [column for column in BOOST_EXTRA_FEATURES if column not in frame.columns]
    if missing:
        raise AssertionError(f"fuzzy boost is missing columns: {missing}")
    extra = frame.loc[:, list(BOOST_EXTRA_FEATURES)]
    weights = firing_matrix(_apply(frame, knots))
    rules = pd.DataFrame(weights, index=frame.index, columns=_rule_names())
    return pd.concat([raw, extra, rules], axis=1)


def _positive_score(estimator: object, values: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "proba":
        return estimator.predict_proba(values)[:, 1]
    return estimator.decision_function(values)


def _cv_score(
    estimator: object,
    features: pd.DataFrame,
    target: np.ndarray,
    kind: str,
) -> np.ndarray:
    folds = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    method = "predict_proba" if kind == "proba" else "decision_function"
    predicted = cross_val_predict(estimator, features, target, cv=folds, method=method, n_jobs=1)
    if kind == "proba":
        return predicted[:, 1]
    return predicted


def _cv_fuzzy_boost(train: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Out-of-fold booster scores, with knots refit on each training fold."""
    scores = np.zeros(len(train), dtype=float)
    folds = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fit_index, held_index in folds.split(train, target):
        knots = _fit_all(train.iloc[fit_index])
        model = _xgboost()
        model.fit(_model_matrix(train.iloc[fit_index], knots), target[fit_index])
        held = _model_matrix(train.iloc[held_index], knots)
        scores[held_index] = model.predict_proba(held)[:, 1]
    return scores


def _fit_fuzzy_boost(
    train: pd.DataFrame,
    target: np.ndarray,
    estimator: XGBClassifier,
) -> FittedClassifier:
    out_of_fold = _cv_fuzzy_boost(train, target)
    knots = _fit_all(train)
    estimator.fit(_model_matrix(train, knots), target)
    threshold = threshold_for_accuracy(target, out_of_fold, center=PROBABILITY_CENTER)
    return FittedClassifier(estimator, "proba", threshold, out_of_fold, knots)


def fit_classifiers(train: pd.DataFrame, target: np.ndarray) -> dict[str, FittedClassifier]:
    fitted = {}
    for name, (estimator, center, kind) in _estimators().items():
        if name == "fuzzy_boost":
            fitted[name] = _fit_fuzzy_boost(train, target, estimator)
            continue
        out_of_fold = _cv_score(estimator, train.loc[:, list(SELECTED_FEATURES)], target, kind)
        threshold = threshold_for_accuracy(target, out_of_fold, center=center)
        estimator.fit(train.loc[:, list(SELECTED_FEATURES)], target)
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
    for model_name in ("decision_tree", "xgboost", "fuzzy_boost"):
        importances = fitted[model_name].estimator.feature_importances_
        if model_name == "fuzzy_boost":
            feature_names = list(SELECTED_FEATURES) + list(BOOST_EXTRA_FEATURES) + _rule_names()
            meaning = "normalized importance of splits on this feature"
        else:
            feature_names = list(SELECTED_FEATURES)
            meaning = "normalized importance of splits on this feature"
        if len(feature_names) != len(importances):
            raise AssertionError(f"{model_name} importance length does not match its features")
        for feature, weight in zip(feature_names, importances):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "value": float(weight),
                    "meaning": meaning,
                }
            )
    return pd.DataFrame(rows)


def _fuzzy_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(OUTPUT_DIR / "fuzzy_metrics.csv")
    rows = []
    for record in metrics.to_dict(orient="records"):
        record.pop("membership", None)
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


def _attach_extra(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add booster-only columns that the three-feature model table does not carry."""
    source = pd.read_csv(OUTPUT_DIR / source_name)
    extra = source.loc[:, [ROW_ID_COLUMN, *BOOST_EXTRA_FEATURES]]
    attached = frame.merge(extra, on=ROW_ID_COLUMN, how="left", validate="one_to_one")
    if attached[list(BOOST_EXTRA_FEATURES)].isna().any().any():
        raise AssertionError(f"{source_name} does not cover every model row")
    return attached


def run() -> None:
    train = _attach_extra(pd.read_csv(OUTPUT_DIR / "train_model.csv"), "train_clean.csv")
    test = _attach_extra(pd.read_csv(OUTPUT_DIR / "test_model.csv"), "test_clean.csv")
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
        for cohort, view, metrics in (
            ("train", "cross_validation", classification_metrics(y_train, model.out_of_fold, model.threshold)),
            ("train", "training_fit", classification_metrics(y_train, train_score, model.threshold)),
            ("test", "held_out", test_metrics),
        ):
            rows.append({"model": name, "cohort": cohort, "view": view, **metrics})
        score_frames["train"][name] = train_score
        score_frames["test"][name] = test_score

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "membership", fuzzify.ACTIVE_MEMBERSHIP)
    metrics.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
    _importance_rows(fitted).to_csv(OUTPUT_DIR / "comparison_importance.csv", index=False)
    pd.concat(score_frames.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "comparison_scores.csv",
        index=False,
    )


if __name__ == "__main__":
    run()
