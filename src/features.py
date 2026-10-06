"""Choose a small feature set for the fuzzy model.

All scores are computed on the training set only. The test set is only
subsetted with the frozen feature list.

Rules, applied in AUC order:
1. Reject a feature when its Spearman correlation with an already accepted
   feature is at least 0.80.
2. Reject a feature when it is that highly correlated with a feature
   rejected by rule 1. This removes raw copies of a redundant marker.
3. Reject a feature when its univariate ROC AUC is below 0.60.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.preprocess import (
    COHORT_COLUMN,
    ID_COLUMN,
    OUTPUT_DIR,
    ROW_ID_COLUMN,
    TARGET_CODE_COLUMN,
    TARGET_COLUMN,
)

AUC_MINIMUM = 0.60
CORRELATION_LIMIT = 0.80

BASE_FEATURES = [
    "Age",
    "BMI",
    "RBC",
    "HGB",
    "WBC",
    "PLT",
    "PMN",
    "LYMPH",
    "MONOC",
    "tPSA",
    "fPSA",
    "PV",
    "PSAD",
    "GLU",
    "CRE",
    "ALT",
    "AST",
    "AST/ALT",
    "f_t_psa",
]
ENGINEERED_RATIOS = {
    "nlr": ("PMN", "LYMPH"),
    "mlr": ("MONOC", "LYMPH"),
    "plr": ("PLT", "LYMPH"),
}


def add_engineered_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    for name, (numerator, denominator) in ENGINEERED_RATIOS.items():
        engineered[name] = engineered[numerator] / engineered[denominator]
    return engineered


def candidate_columns() -> list[str]:
    return [*BASE_FEATURES, *ENGINEERED_RATIOS]


def score_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = candidate_columns()
    target = frame[TARGET_CODE_COLUMN].to_numpy()
    rows = []
    for column in features:
        values = frame[column].to_numpy()
        raw_auc = float(roc_auc_score(target, values))
        if raw_auc >= 0.5:
            auc = raw_auc
            direction = "higher_in_cancer"
        else:
            auc = 1.0 - raw_auc
            direction = "lower_in_cancer"
        positive = values[target == 1]
        negative = values[target == 0]
        _statistic, p_value = stats.mannwhitneyu(positive, negative, alternative="two-sided")
        rows.append(
            {
                "feature": column,
                "auc": auc,
                "direction": direction,
                "p_value": float(p_value),
                "median_positive": float(np.median(positive)),
                "median_negative": float(np.median(negative)),
            }
        )
    scores = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    correlations = frame[features].corr(method="spearman")
    return scores, correlations


def select_features(scores: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    accepted: list[str] = []
    redundant: set[str] = set()
    decisions = []

    for record in scores.to_dict(orient="records"):
        feature = record["feature"]
        accepted_match = _correlated_with(feature, accepted, correlations)
        redundant_match = _correlated_with(feature, redundant, correlations)

        if accepted_match is not None:
            coefficient = float(correlations.loc[feature, accepted_match])
            decision = "rejected"
            reason = (
                f"Spearman {coefficient:.3f} with accepted feature {accepted_match}"
            )
            redundant.add(feature)
        elif redundant_match is not None:
            coefficient = float(correlations.loc[feature, redundant_match])
            decision = "rejected"
            reason = (
                f"Spearman {coefficient:.3f} with {redundant_match}, "
                "which is already redundant"
            )
        elif record["auc"] < AUC_MINIMUM:
            decision = "rejected"
            reason = f"univariate AUC {record['auc']:.3f} is below {AUC_MINIMUM:.2f}"
        else:
            decision = "accepted"
            reason = "AUC at least 0.60 and not redundant"
            accepted.append(feature)

        decisions.append({**record, "decision": decision, "reason": reason})

    return pd.DataFrame(decisions)


def _correlated_with(
    feature: str,
    others: list[str] | set[str],
    correlations: pd.DataFrame,
) -> str | None:
    for other in others:
        if abs(float(correlations.loc[feature, other])) >= CORRELATION_LIMIT:
            return other
    return None


def _model_frame(frame: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    columns = [
        ROW_ID_COLUMN,
        ID_COLUMN,
        COHORT_COLUMN,
        *selected,
        TARGET_COLUMN,
        TARGET_CODE_COLUMN,
    ]
    return frame.loc[:, columns].copy()


def selected_features(train: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    scores, correlations = score_features(add_engineered_ratios(train))
    report = select_features(scores, correlations)
    selected = report.loc[report["decision"] == "accepted", "feature"].tolist()
    return selected, report


def run() -> None:
    train = add_engineered_ratios(pd.read_csv(OUTPUT_DIR / "train_clean.csv"))
    test = add_engineered_ratios(pd.read_csv(OUTPUT_DIR / "test_clean.csv"))

    selected, report = selected_features(train)
    if not selected:
        raise RuntimeError("feature selection accepted no features")

    report.to_csv(OUTPUT_DIR / "feature_scores.csv", index=False)
    _model_frame(train, selected).to_csv(OUTPUT_DIR / "train_model.csv", index=False)
    _model_frame(test, selected).to_csv(OUTPUT_DIR / "test_model.csv", index=False)


if __name__ == "__main__":
    run()
