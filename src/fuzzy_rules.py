"""Score cancer risk with a 27-rule Sugeno system.

The rule grid is the full 3 x 3 x 3 product of the linguistic terms.
Each consequent is a ridge-regression weight fit to the training labels,
with the 27 firing strengths as the inputs and no extra intercept.
Firing strengths already sum to 1, so the score is a weighted vote of
those learned risks. This is least squares on fixed memberships, not a
neural network, and the knots are not moved.

The ridge penalty is fixed at 1. It keeps a rule that almost never fires
from taking an extreme weight.

The decision threshold is the cutoff that maximizes accuracy on the
training out-of-fold scores. Ties go to the cutoff closest to 0.5.
Knots and consequents are re-estimated inside each fold, so those
out-of-fold scores never see their own fold. The test set is scored once
with knots and consequents fit on the full training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.fuzzify import SELECTED_FEATURES, TERMS, _apply, _fit_all
from src.preprocess import OUTPUT_DIR, ROW_ID_COLUMN, TARGET_CODE_COLUMN, TARGET_COLUMN

# Direction points remain in the rule table as a reference, not as the score.
CANCER_POINTS = {
    "PSAD": {"low": 0, "medium": 1, "high": 2},
    "f_t_psa": {"low": 2, "medium": 1, "high": 0},
    "PV": {"low": 2, "medium": 1, "high": 0},
}
MAX_POINTS = 3 * 2
DIRECTION_THRESHOLD = 0.5
RIDGE_ALPHA = 1.0
CV_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class FuzzySystem:
    knots: dict[str, tuple[float, float, float]]
    consequents: np.ndarray
    threshold: float
    out_of_fold: np.ndarray


def build_rules() -> pd.DataFrame:
    rows = []
    rule_id = 1
    for psad_term in TERMS:
        for ft_term in TERMS:
            for pv_term in TERMS:
                points = (
                    CANCER_POINTS["PSAD"][psad_term]
                    + CANCER_POINTS["f_t_psa"][ft_term]
                    + CANCER_POINTS["PV"][pv_term]
                )
                rows.append(
                    {
                        "rule_id": rule_id,
                        "PSAD": psad_term,
                        "f_t_psa": ft_term,
                        "PV": pv_term,
                        "points": points,
                        "direction_consequent": points / MAX_POINTS,
                    }
                )
                rule_id += 1
    rules = pd.DataFrame(rows)
    if len(rules) != len(TERMS) ** len(SELECTED_FEATURES):
        raise AssertionError("the rule base is not the full 27-rule grid")
    return rules


RULES = build_rules()


def firing_matrix(frame: pd.DataFrame, rules: pd.DataFrame = RULES) -> np.ndarray:
    columns = []
    for rule in rules.itertuples(index=False):
        weight = np.ones(len(frame), dtype=float)
        for feature in SELECTED_FEATURES:
            term = getattr(rule, feature)
            weight = weight * frame[f"{feature}_{term}"].to_numpy()
        columns.append(weight)
    weights = np.column_stack(columns)
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise AssertionError("rule firing strengths do not sum to 1")
    return weights


def fit_consequents(weights: np.ndarray, target: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    model.fit(weights, target)
    return model.coef_.astype(float)


def threshold_for_accuracy(
    y_true: np.ndarray,
    score: np.ndarray,
    center: float = 0.5,
) -> float:
    """Cutoff with the highest accuracy. Ties resolve toward `center`."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    cuts = np.unique(score)
    cuts = np.concatenate([cuts, [float(cuts.max()) + 1.0]])
    predicted = score[None, :] >= cuts[:, None]
    accuracy = (predicted == y_true[None, :].astype(bool)).mean(axis=1)
    best = accuracy.max()
    candidates = cuts[np.isclose(accuracy, best, rtol=0.0, atol=1e-12)]
    return float(candidates[np.argmin(np.abs(candidates - center))])


def classification_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity_denominator = tn + fp
    if specificity_denominator == 0:
        raise AssertionError("specificity is undefined because there is no negative case")
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, pos_label=1)),
        "specificity": float(tn / specificity_denominator),
        "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _fit_once(train: pd.DataFrame, target: np.ndarray) -> tuple[dict, np.ndarray]:
    knots = _fit_all(train)
    consequents = fit_consequents(firing_matrix(_apply(train, knots)), target)
    return knots, consequents


def score_fuzzy(
    frame: pd.DataFrame,
    knots: dict[str, tuple[float, float, float]],
    consequents: np.ndarray,
) -> np.ndarray:
    return firing_matrix(_apply(frame, knots)) @ consequents


def fit_fuzzy(train: pd.DataFrame, target: np.ndarray) -> FuzzySystem:
    out_of_fold = np.zeros(len(train), dtype=float)
    folds = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fit_index, held_index in folds.split(train, target):
        knots, consequents = _fit_once(train.iloc[fit_index], target[fit_index])
        out_of_fold[held_index] = score_fuzzy(train.iloc[held_index], knots, consequents)
    threshold = threshold_for_accuracy(target, out_of_fold, center=0.5)
    knots, consequents = _fit_once(train, target)
    return FuzzySystem(knots, consequents, threshold, out_of_fold)


def _risk_word(value: float) -> str:
    if value >= 2 / 3:
        return "high"
    if value <= 1 / 3:
        return "low"
    return "medium"


def _describe(consequents: np.ndarray) -> pd.DataFrame:
    described = RULES.copy()
    described["consequent"] = consequents
    described["risk"] = [_risk_word(float(value)) for value in consequents]
    described["rule"] = [
        (
            f"IF PSAD is {rule.PSAD} AND f_t_psa is {rule.f_t_psa} "
            f"AND PV is {rule.PV} THEN risk is {rule.consequent:.3f}"
        )
        for rule in described.itertuples(index=False)
    ]
    return described


def _score_frame(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> pd.DataFrame:
    scored = frame.loc[:, [ROW_ID_COLUMN, "No.", "cohort", TARGET_COLUMN, TARGET_CODE_COLUMN]].copy()
    scored["fuzzy_score"] = score
    scored["fuzzy_prediction"] = (score >= threshold).astype(int)
    return scored


def run() -> None:
    train = pd.read_csv(OUTPUT_DIR / "train_model.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_model.csv")
    y_train = train[TARGET_CODE_COLUMN].to_numpy()
    y_test = test[TARGET_CODE_COLUMN].to_numpy()

    system = fit_fuzzy(train, y_train)
    train_score = score_fuzzy(train, system.knots, system.consequents)
    test_score = score_fuzzy(test, system.knots, system.consequents)

    # The earlier hand-set consequents, on the same knots, at their own cutoff.
    # This is a reference, not a second chance to pick a winner.
    direction = RULES["direction_consequent"].to_numpy()
    ablation = pd.DataFrame(
        [
            {
                "variant": "direction_points",
                "cohort": cohort,
                **classification_metrics(
                    frame[TARGET_CODE_COLUMN].to_numpy(),
                    score_fuzzy(frame, system.knots, direction),
                    DIRECTION_THRESHOLD,
                ),
            }
            for cohort, frame in (("train", train), ("test", test))
        ]
    )

    metrics = pd.DataFrame(
        [
            {
                "cohort": "train",
                "view": "cross_validation",
                **classification_metrics(y_train, system.out_of_fold, system.threshold),
            },
            {
                "cohort": "train",
                "view": "training_fit",
                **classification_metrics(y_train, train_score, system.threshold),
            },
            {
                "cohort": "test",
                "view": "held_out",
                **classification_metrics(y_test, test_score, system.threshold),
            },
        ]
    )

    rules = _describe(system.consequents)
    rules.to_csv(OUTPUT_DIR / "fuzzy_rules.csv", index=False)
    _score_frame(train, train_score, system.threshold).to_csv(
        OUTPUT_DIR / "train_fuzzy_scores.csv",
        index=False,
    )
    _score_frame(test, test_score, system.threshold).to_csv(
        OUTPUT_DIR / "test_fuzzy_scores.csv",
        index=False,
    )
    metrics.to_csv(OUTPUT_DIR / "fuzzy_metrics.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "fuzzy_ablation.csv", index=False)


if __name__ == "__main__":
    run()
