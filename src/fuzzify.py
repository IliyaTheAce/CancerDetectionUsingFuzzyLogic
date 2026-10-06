"""Fuzzify the three selected features with a quantile partition.

Knots are the 25th, 50th, and 75th percentiles of the training set.
The same knots are applied to the test set.

Outer terms saturate: values below the 25th percentile are fully low,
and values above the 75th percentile are fully high. A Gaussian centered
on these quartiles does not do that. On PSAD, the 90th percentile already
falls so far past the upper Gaussian that every membership is about 0,
including the patients with the strongest density signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import OUTPUT_DIR

SELECTED_FEATURES = ["PSAD", "f_t_psa", "PV"]
TERMS = ("low", "medium", "high")
QUANTILES = (0.25, 0.50, 0.75)


def fit_knots(values: pd.Series) -> tuple[float, float, float]:
    low, medium, high = (float(values.quantile(q)) for q in QUANTILES)
    if not low < medium < high:
        raise ValueError(f"{values.name} quartiles are not strictly increasing")
    return low, medium, high


def membership_degrees(
    values: np.ndarray,
    low_knot: float,
    medium_knot: float,
    high_knot: float,
) -> dict[str, np.ndarray]:
    """Ruspini partition: low and high are shoulders, medium is a triangle."""
    x = np.asarray(values, dtype=float)
    low = np.zeros_like(x)
    medium = np.zeros_like(x)
    high = np.zeros_like(x)

    low = np.where(x <= low_knot, 1.0, low)
    rising = (x > low_knot) & (x < medium_knot)
    low = np.where(rising, (medium_knot - x) / (medium_knot - low_knot), low)
    medium = np.where(rising, (x - low_knot) / (medium_knot - low_knot), medium)

    medium = np.where(x == medium_knot, 1.0, medium)

    falling = (x > medium_knot) & (x < high_knot)
    medium = np.where(falling, (high_knot - x) / (high_knot - medium_knot), medium)
    high = np.where(falling, (x - medium_knot) / (high_knot - medium_knot), high)
    high = np.where(x >= high_knot, 1.0, high)
    return {"low": low, "medium": medium, "high": high}


def _fit_all(internal: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    return {feature: fit_knots(internal[feature]) for feature in SELECTED_FEATURES}


def _apply(frame: pd.DataFrame, knots: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    fuzzified = frame.copy()
    for feature, (low_knot, medium_knot, high_knot) in knots.items():
        degrees = membership_degrees(frame[feature].to_numpy(), low_knot, medium_knot, high_knot)
        for term in TERMS:
            fuzzified[f"{feature}_{term}"] = degrees[term]
    return fuzzified


def _assert_partition(frame: pd.DataFrame, knots: dict[str, tuple[float, float, float]]) -> None:
    for feature, (low_knot, medium_knot, high_knot) in knots.items():
        columns = [f"{feature}_{term}" for term in TERMS]
        total = frame[columns].sum(axis=1)
        if not np.allclose(total, 1.0):
            raise AssertionError(f"{feature} memberships do not sum to 1")
        if not np.allclose(frame.loc[frame[feature] <= low_knot, f"{feature}_low"], 1.0):
            raise AssertionError(f"{feature} values at or below q25 are not fully low")
        if not np.allclose(frame.loc[frame[feature] >= high_knot, f"{feature}_high"], 1.0):
            raise AssertionError(f"{feature} values at or above q75 are not fully high")
        del medium_knot


def _parameter_table(knots: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for feature, (low_knot, medium_knot, high_knot) in knots.items():
        rows.append(
            {
                "feature": feature,
                "q25_low_shoulder": low_knot,
                "q50_medium_peak": medium_knot,
                "q75_high_shoulder": high_knot,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    train = pd.read_csv(OUTPUT_DIR / "train_model.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_model.csv")
    knots = _fit_all(train)

    train_fuzzy = _apply(train, knots)
    test_fuzzy = _apply(test, knots)
    _assert_partition(train_fuzzy, knots)
    _assert_partition(test_fuzzy, knots)

    _parameter_table(knots).to_csv(OUTPUT_DIR / "membership_params.csv", index=False)
    train_fuzzy.to_csv(OUTPUT_DIR / "train_memberships.csv", index=False)
    test_fuzzy.to_csv(OUTPUT_DIR / "test_memberships.csv", index=False)


if __name__ == "__main__":
    run()
