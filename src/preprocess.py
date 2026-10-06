"""Load, clean, and split the prostate cancer patients.

data-1.csv and data-2.csv are read as one pool of 598 patients. Rows had
already been moved between the two files by hand, so the file a row sits
in is no longer a cohort. It is kept only as `source_file`.

The pool is split once, stratified by the label, into a training set and
a test set. The same split function is reused with other seeds by
`src/repeated.py`.

`No.` is kept exactly as stored, including repeated values. `row_id` is a
unique number given after the two files are stacked. It is only used to
join tables. No row is dropped because its `No.` repeats.

Invalid cells are values that cannot be a real absolute count:
a white-cell subtype larger than the total white-cell count, or a
platelet count below 20 x 10^9/L. Those cells are set to missing. This
step and the ratio recomputation look at one row at a time, so they run
on the whole pool before the split. Missing cells are then filled with
the training-set median only. Extreme PSA values are kept, because a very
high PSA is a real clinical signal and both such rows are labeled positive.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = DATA_DIR / "processed"

SOURCE_FILES = ("data-1.csv", "data-2.csv")
EXPECTED_PATIENTS = 598

ID_COLUMN = "No."
ROW_ID_COLUMN = "row_id"
SOURCE_COLUMN = "source_file"
TARGET_COLUMN = "Prostate Cancer"
TARGET_CODE_COLUMN = "prostate_cancer"
COHORT_COLUMN = "cohort"

TEST_FRACTION = 0.20
SPLIT_SEED = 42

CLINICAL_FEATURES = [
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
]
ENGINEERED_FEATURES = ["f_t_psa"]
FEATURE_COLUMNS = CLINICAL_FEATURES + ENGINEERED_FEATURES

WHITE_CELL_PARTS = ["PMN", "LYMPH", "MONOC"]
# Below this level a platelet count is not a credible routine value.
# The only such cell in these files is 1.75; the next lowest is 71.
MIN_PLAUSIBLE_PLATELET = 20.0
# Stored PSAD and AST/ALT are rounded. Only larger gaps are logged.
RATIO_LOG_TOLERANCE = 0.005

TARGET_TO_CODE = {"negative": 0, "positive": 1}


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = [ID_COLUMN, *CLINICAL_FEATURES, TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    frame = frame.loc[:, expected].copy()
    frame[CLINICAL_FEATURES] = frame[CLINICAL_FEATURES].astype(float)
    frame[SOURCE_COLUMN] = path.stem

    unknown_labels = set(frame[TARGET_COLUMN].unique()) - set(TARGET_TO_CODE)
    if unknown_labels:
        raise ValueError(f"{path.name} has unexpected labels: {sorted(unknown_labels)}")

    frame[TARGET_CODE_COLUMN] = frame[TARGET_COLUMN].map(TARGET_TO_CODE).astype(int)
    return frame


def load_pool() -> pd.DataFrame:
    pool = pd.concat(
        [load_source(DATA_DIR / name) for name in SOURCE_FILES],
        ignore_index=True,
    )
    if len(pool) != EXPECTED_PATIENTS:
        raise AssertionError(f"expected {EXPECTED_PATIENTS} patients, found {len(pool)}")
    pool[ROW_ID_COLUMN] = np.arange(1, len(pool) + 1)
    return pool


def _log(records: list[dict], frame: pd.DataFrame, idx: int, where: str, **fields: object) -> None:
    records.append(
        {
            "row_id": int(frame.at[idx, ROW_ID_COLUMN]),
            "patient_no": int(frame.at[idx, ID_COLUMN]),
            "where": where,
            **fields,
        }
    )


def invalidate_impossible_counts(frame: pd.DataFrame, records: list[dict]) -> None:
    """Blank counts that contradict the definition of the column."""
    for column in WHITE_CELL_PARTS:
        invalid = frame[column] > frame["WBC"]
        for idx in frame.index[invalid]:
            _log(
                records,
                frame,
                idx,
                frame.at[idx, SOURCE_COLUMN],
                column=column,
                original_value=float(frame.at[idx, column]),
                new_value=np.nan,
                action="set_missing",
                reason="subtype count is larger than total WBC, so it is not an absolute count",
            )
        frame.loc[invalid, column] = np.nan

    invalid_platelets = frame["PLT"] < MIN_PLAUSIBLE_PLATELET
    for idx in frame.index[invalid_platelets]:
        _log(
            records,
            frame,
            idx,
            frame.at[idx, SOURCE_COLUMN],
            column="PLT",
            original_value=float(frame.at[idx, "PLT"]),
            new_value=np.nan,
            action="set_missing",
            reason=f"platelet count is below {MIN_PLAUSIBLE_PLATELET:g} x 10^9/L",
        )
    frame.loc[invalid_platelets, "PLT"] = np.nan


def recompute_derived_ratios(frame: pd.DataFrame, records: list[dict]) -> None:
    """Replace stored ratios with the value of their own formula."""
    replacements = {
        "PSAD": frame["tPSA"] / frame["PV"],
        "AST/ALT": frame["AST"] / frame["ALT"],
    }
    for column, recomputed in replacements.items():
        gap = (frame[column] - recomputed).abs()
        material = gap > RATIO_LOG_TOLERANCE
        for idx in frame.index[material]:
            _log(
                records,
                frame,
                idx,
                frame.at[idx, SOURCE_COLUMN],
                column=column,
                original_value=float(frame.at[idx, column]),
                new_value=float(recomputed.at[idx]),
                action="recompute",
                reason="stored ratio does not match its formula beyond rounding",
            )
        frame[column] = recomputed

    frame["f_t_psa"] = frame["fPSA"] / frame["tPSA"]


def clean_pool(frame: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    cleaned = frame.copy()
    invalidate_impossible_counts(cleaned, records)
    recompute_derived_ratios(cleaned, records)
    return cleaned


def split_pool(pool: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, test_index = train_test_split(
        pool.index,
        test_size=TEST_FRACTION,
        stratify=pool[TARGET_CODE_COLUMN],
        random_state=seed,
    )
    train = pool.loc[np.sort(train_index)].copy()
    test = pool.loc[np.sort(test_index)].copy()
    train[COHORT_COLUMN] = "train"
    test[COHORT_COLUMN] = "test"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_medians(train: pd.DataFrame) -> pd.Series:
    return train[FEATURE_COLUMNS].median()


def impute_medians(
    frame: pd.DataFrame,
    medians: pd.Series,
    records: list[dict],
) -> pd.DataFrame:
    imputed = frame.copy()
    # Medians can be half-integers (e.g. 66.5). Columns that still look
    # like integers cannot hold those fills, so cast before assigning.
    imputed[FEATURE_COLUMNS] = imputed[FEATURE_COLUMNS].astype(float)
    for column in FEATURE_COLUMNS:
        missing = imputed[column].isna()
        fill_value = float(medians[column])
        for idx in imputed.index[missing]:
            _log(
                records,
                imputed,
                idx,
                imputed.at[idx, COHORT_COLUMN],
                column=column,
                original_value=np.nan,
                new_value=fill_value,
                action="median_impute",
                reason="median fitted on the training set only",
            )
        imputed.loc[missing, column] = fill_value
    return imputed


def prepare_split(
    pool: pd.DataFrame,
    seed: int,
    records: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split a cleaned pool and impute both parts with training medians."""
    train, test = split_pool(pool, seed)
    medians = fit_medians(train)
    train = impute_medians(train, medians, records)
    test = impute_medians(test, medians, records)
    _assert_ready(train)
    _assert_ready(test)
    return train, test, medians


def scale_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    scaler = RobustScaler()
    scaler.fit(train[FEATURE_COLUMNS])

    def apply_scaler(frame: pd.DataFrame) -> pd.DataFrame:
        scaled_values = scaler.transform(frame[FEATURE_COLUMNS])
        scaled = frame.copy()
        scaled[FEATURE_COLUMNS] = scaled_values
        return scaled

    return apply_scaler(train), apply_scaler(test), scaler


def _column_order() -> list[str]:
    return [
        ROW_ID_COLUMN,
        ID_COLUMN,
        SOURCE_COLUMN,
        COHORT_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        TARGET_CODE_COLUMN,
    ]


def save_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    records: list[dict],
    medians: pd.Series,
    scaler: RobustScaler,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    order = _column_order()

    train.loc[:, order].to_csv(OUTPUT_DIR / "train_clean.csv", index=False)
    test.loc[:, order].to_csv(OUTPUT_DIR / "test_clean.csv", index=False)
    train_scaled.loc[:, order].to_csv(OUTPUT_DIR / "train_scaled.csv", index=False)
    test_scaled.loc[:, order].to_csv(OUTPUT_DIR / "test_scaled.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)

    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "medians": medians,
            "scaler": scaler,
            "target_mapping": TARGET_TO_CODE,
            "split_seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION,
            "fitted_on": "training set of the seeded stratified split",
        },
        OUTPUT_DIR / "preprocess.joblib",
    )


def _assert_ready(frame: pd.DataFrame) -> None:
    if len(frame) == 0:
        raise AssertionError("split has no rows")
    if frame[ROW_ID_COLUMN].duplicated().any():
        raise AssertionError("row_id values are not unique")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise AssertionError("cleaned features still contain missing values")
    if frame["fPSA"].gt(frame["tPSA"]).any():
        raise AssertionError("free PSA exceeds total PSA")
    for column in WHITE_CELL_PARTS:
        if frame[column].gt(frame["WBC"]).any():
            raise AssertionError(f"{column} still exceeds WBC")
    if not np.allclose(frame["PSAD"], frame["tPSA"] / frame["PV"]):
        raise AssertionError("PSAD does not match tPSA / PV")
    if not np.allclose(frame["AST/ALT"], frame["AST"] / frame["ALT"]):
        raise AssertionError("AST/ALT does not match AST / ALT")
    if not np.allclose(frame["f_t_psa"], frame["fPSA"] / frame["tPSA"]):
        raise AssertionError("f_t_psa does not match fPSA / tPSA")


def run() -> None:
    records: list[dict] = []
    pool = clean_pool(load_pool(), records)
    train, test, medians = prepare_split(pool, SPLIT_SEED, records)

    if set(train[ROW_ID_COLUMN]) & set(test[ROW_ID_COLUMN]):
        raise AssertionError("a patient is in both the training and the test set")
    if len(train) + len(test) != EXPECTED_PATIENTS:
        raise AssertionError("the split lost or added patients")

    train_scaled, test_scaled, scaler = scale_features(train, test)
    save_outputs(train, test, train_scaled, test_scaled, records, medians, scaler)


if __name__ == "__main__":
    run()
