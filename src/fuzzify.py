"""Fuzzify PSAD, f/t PSA, and prostate volume.

The shape is always the same Ruspini partition: low and high are shoulders,
medium is a triangle, and the three degrees sum to 1. What changes is where
the three knots sit.

Switch the knots with ACTIVE_MEMBERSHIP, or from the command line:

    python main.py --membership mri

The fuzzy system and fuzzy boost both call _fit_all, so that one name is
the membership they use. Logistic regression, the SVM, the tree, and plain
XGBoost keep the raw numbers.

quartile
    Training 25th percentile, median, and 75th percentile. These are refit
    on each training fold and on each repeated split. The label is not used.

eau
    EAU 2024 and the Schoots risk table: PSA density 0.10 / 0.15 / 0.20.
    Catalona free-to-total steps 0.10 / 0.15 / 0.25. ERSPC volume classes
    30 / 40 / 50 mL.

mri
    EAU MRI pathway plus the 2025 biopsy thresholds: PSA density 0.10 / 0.12
    / 0.20 (PI-RADS 3 line at 0.12, negative-MRI line at 0.20). Free-to-total
    intervals 0.10 / 0.16 / 0.25. Volume 35 / 44 / 60 mL.

classic
    Pre-MRI cutoffs. PSA density below 0.09 is the EAU "unlikely" line, 0.15
    is the classic biopsy line, and 0.20 is the top of the Schoots intermediate
    band. Free-to-total 0.10 / 0.15 / 0.25. Volume 35 / 50 / 65 mL.

gray_zone
    Published gray-zone cuts. Free-to-total studies cluster from 0.14 to 0.25,
    with 0.16 as an interval edge inside that range. Volume ROC cuts are
    43.5 mL at PSA 2.5–10 and 61.5 mL at PSA 10–30, with 35 mL as the
    small-gland line. PSA density stays on the 0.10 / 0.15 / 0.20 bands.

Outer terms saturate: at or below the low knot is fully low, and at or above
the high knot is fully high. A Gaussian centered on the quartiles does not
do that. On PSAD, the 90th percentile already falls so far past the upper
Gaussian that every membership is about 0, including the patients with the
strongest density signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import OUTPUT_DIR

SELECTED_FEATURES = ["PSAD", "f_t_psa", "PV"]
TERMS = ("low", "medium", "high")
QUANTILES = (0.25, 0.50, 0.75)

# Default knots for the fuzzy system and fuzzy boost.
# Override with: python main.py --membership mri
# One of: quartile, eau, mri, classic, gray_zone
ACTIVE_MEMBERSHIP = "quartile"

# low shoulder, medium peak, high shoulder. Units: PSAD ng/mL/cm3, f/t ratio, PV mL.
LITERATURE: dict[str, dict[str, tuple[float, float, float]]] = {
    "eau": {
        "PSAD": (0.10, 0.15, 0.20),
        "f_t_psa": (0.10, 0.15, 0.25),
        "PV": (30.0, 40.0, 50.0),
    },
    "mri": {
        "PSAD": (0.10, 0.12, 0.20),
        "f_t_psa": (0.10, 0.16, 0.25),
        "PV": (35.0, 44.0, 60.0),
    },
    "classic": {
        "PSAD": (0.09, 0.15, 0.20),
        "f_t_psa": (0.10, 0.15, 0.25),
        "PV": (35.0, 50.0, 65.0),
    },
    "gray_zone": {
        "PSAD": (0.10, 0.15, 0.20),
        "f_t_psa": (0.14, 0.16, 0.25),
        "PV": (35.0, 43.5, 61.5),
    },
}

MEMBERSHIP_SOURCES = {
    "quartile": "training 25th percentile, median, and 75th percentile",
    "eau": "EAU 2024 and Schoots PSA-density bands; Catalona free-to-total steps; ERSPC volume classes",
    "mri": "EAU MRI pathway and 2025 biopsy thresholds; free-to-total intervals 0.10/0.16/0.25; volume 35/44/60 mL",
    "classic": "PSA density 0.09/0.15/0.20; Catalona and 0.15 free-to-total cutoff; volume 35/50/65 mL",
    "gray_zone": "gray-zone free-to-total range 0.14-0.25; volume ROC 43.5 mL and 61.5 mL",
}

MEMBERSHIP_NAMES = ("quartile", *LITERATURE)


def _check_fixed_knots() -> None:
    for name, table in LITERATURE.items():
        missing = [feature for feature in SELECTED_FEATURES if feature not in table]
        if missing:
            raise ValueError(f"{name} is missing knots for {missing}")
        for feature, (low_knot, medium_knot, high_knot) in table.items():
            if not low_knot < medium_knot < high_knot:
                raise ValueError(f"{name} {feature} knots are not strictly increasing")


_check_fixed_knots()


def set_membership(name: str) -> str:
    """Select the knots used by the fuzzy system and fuzzy boost."""
    global ACTIVE_MEMBERSHIP
    if name not in MEMBERSHIP_NAMES:
        known = ", ".join(MEMBERSHIP_NAMES)
        raise ValueError(f"unknown membership {name!r}; choose one of: {known}")
    ACTIVE_MEMBERSHIP = name
    return ACTIVE_MEMBERSHIP


def fit_knots(values: pd.Series) -> tuple[float, float, float]:
    low, medium, high = (float(values.quantile(q)) for q in QUANTILES)
    if not low < medium < high:
        raise ValueError(f"{values.name} quartiles are not strictly increasing")
    return low, medium, high


def knots_for(
    frame: pd.DataFrame,
    name: str | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Knots for one named membership. Quartiles are taken from `frame`."""
    if name is None:
        name = ACTIVE_MEMBERSHIP
    if name == "quartile":
        return {feature: fit_knots(frame[feature]) for feature in SELECTED_FEATURES}
    if name not in LITERATURE:
        known = ", ".join(MEMBERSHIP_NAMES)
        raise ValueError(f"unknown membership {name!r}; choose one of: {known}")
    return {feature: LITERATURE[name][feature] for feature in SELECTED_FEATURES}


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
    return knots_for(internal, ACTIVE_MEMBERSHIP)


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
            raise AssertionError(f"{feature} values at or below the low knot are not fully low")
        if not np.allclose(frame.loc[frame[feature] >= high_knot, f"{feature}_high"], 1.0):
            raise AssertionError(f"{feature} values at or above the high knot are not fully high")
        del medium_knot


def _parameter_table(name: str, knots: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for feature, (low_knot, medium_knot, high_knot) in knots.items():
        rows.append(
            {
                "membership": name,
                "feature": feature,
                "low_shoulder": low_knot,
                "medium_peak": medium_knot,
                "high_shoulder": high_knot,
                "source": MEMBERSHIP_SOURCES[name],
            }
        )
    return pd.DataFrame(rows)


def _catalog(train: pd.DataFrame) -> pd.DataFrame:
    tables = [_parameter_table(name, knots_for(train, name)) for name in MEMBERSHIP_NAMES]
    return pd.concat(tables, ignore_index=True)


def run() -> None:
    train = pd.read_csv(OUTPUT_DIR / "train_model.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_model.csv")
    knots = _fit_all(train)

    train_fuzzy = _apply(train, knots)
    test_fuzzy = _apply(test, knots)
    _assert_partition(train_fuzzy, knots)
    _assert_partition(test_fuzzy, knots)

    _catalog(train).to_csv(OUTPUT_DIR / "membership_catalog.csv", index=False)
    _parameter_table(ACTIVE_MEMBERSHIP, knots).to_csv(OUTPUT_DIR / "membership_params.csv", index=False)
    train_fuzzy.to_csv(OUTPUT_DIR / "train_memberships.csv", index=False)
    test_fuzzy.to_csv(OUTPUT_DIR / "test_memberships.csv", index=False)


if __name__ == "__main__":
    run()
