"""Repeat the whole comparison on 100 seeded stratified 80/20 splits.

Each repeat starts from the same cleaned pool of 598 patients and redoes
everything that is fit to data, using that repeat's training set only:
the imputation medians, the membership knots, the rule consequents, the
four classifiers, and every model's accuracy cutoff. The test set of that
repeat is scored once.

The model inputs stay PSAD, f_t_psa, and PV in every repeat. The feature
selection rules are still rerun on each training set, and the accepted
list is recorded, so the report shows how often that choice would have
come out the same.

The spread across repeats shows how much a result moves with the split.
The test sets overlap between repeats, so a percentile band of these
numbers is a stability range. It is not a confidence interval and not a
significance test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.compare import CLASSIFIER_NAMES, fit_classifiers
from src.features import selected_features
from src.fuzzify import SELECTED_FEATURES
from src.fuzzy_rules import classification_metrics, fit_fuzzy, score_fuzzy
from src.preprocess import OUTPUT_DIR, TARGET_CODE_COLUMN, clean_pool, load_pool, prepare_split

N_REPEATS = 100
SEEDS = tuple(range(N_REPEATS))
MODELS = ("fuzzy", *CLASSIFIER_NAMES)
SUMMARY_METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")
PAIRED_METRICS = ("accuracy", "f1", "roc_auc")


def _one_repeat(pool: pd.DataFrame, seed: int) -> tuple[list[dict], dict]:
    train, test, _medians = prepare_split(pool, seed, records=[])
    y_train = train[TARGET_CODE_COLUMN].to_numpy()
    y_test = test[TARGET_CODE_COLUMN].to_numpy()

    accepted, _report = selected_features(train)
    selection = {
        "seed": seed,
        "accepted": ", ".join(accepted),
        "same_as_model_inputs": set(accepted) == set(SELECTED_FEATURES),
    }

    rows = []
    system = fit_fuzzy(train, y_train)
    fuzzy_test = score_fuzzy(test, system.knots, system.consequents)
    rows.append({"seed": seed, "model": "fuzzy", **classification_metrics(y_test, fuzzy_test, system.threshold)})

    for name, model in fit_classifiers(train, y_train).items():
        rows.append(
            {
                "seed": seed,
                "model": name,
                **classification_metrics(y_test, model.score(test), model.threshold),
            }
        )
    return rows, selection


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        part = results.loc[results["model"] == model]
        for metric in SUMMARY_METRICS:
            values = part[metric].to_numpy()
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "p2_5": float(np.percentile(values, 2.5)),
                    "median": float(np.median(values)),
                    "p97_5": float(np.percentile(values, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def _paired(results: pd.DataFrame) -> pd.DataFrame:
    wide = {metric: results.pivot(index="seed", columns="model", values=metric) for metric in PAIRED_METRICS}
    rows = []
    for baseline in CLASSIFIER_NAMES:
        for metric in PAIRED_METRICS:
            difference = (wide[metric]["fuzzy"] - wide[metric][baseline]).to_numpy()
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "mean_fuzzy_minus_baseline": float(difference.mean()),
                    "p2_5": float(np.percentile(difference, 2.5)),
                    "p97_5": float(np.percentile(difference, 97.5)),
                    "fuzzy_higher": int((difference > 1e-12).sum()),
                    "tied": int((np.abs(difference) <= 1e-12).sum()),
                    "fuzzy_lower": int((difference < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    pool = clean_pool(load_pool(), records=[])
    rows: list[dict] = []
    selections: list[dict] = []
    for seed in SEEDS:
        repeat_rows, selection = _one_repeat(pool, seed)
        rows.extend(repeat_rows)
        selections.append(selection)

    results = pd.DataFrame(rows)
    summary = _summary(results)
    paired = _paired(results)
    selection_frame = pd.DataFrame(selections)

    results.to_csv(OUTPUT_DIR / "repeated_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "repeated_summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "repeated_paired.csv", index=False)
    selection_frame.to_csv(OUTPUT_DIR / "repeated_feature_selection.csv", index=False)


if __name__ == "__main__":
    run()
