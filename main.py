"""Run the full research pipeline and write the paper charts."""

import argparse

import src.fuzzify as memberships
from src.charts import print_chart_data, run as build_charts
from src.compare import run as compare
from src.features import run as select_features
from src.fuzzify import MEMBERSHIP_NAMES, run as fuzzify, set_membership
from src.fuzzy_rules import run as apply_rules
from src.preprocess import run as preprocess
from src.repeated import run as repeat_splits


def main(membership: str | None = None) -> None:
    if membership is not None:
        set_membership(membership)
    print(f"Membership: {memberships.ACTIVE_MEMBERSHIP}")

    preprocess()
    select_features()
    fuzzify()
    apply_rules()
    compare()
    repeat_splits()

    print_chart_data()
    print("Charts:")
    for path in build_charts():
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prostate-cancer fuzzy pipeline.")
    parser.add_argument(
        "--membership",
        choices=MEMBERSHIP_NAMES,
        default=memberships.ACTIVE_MEMBERSHIP,
        help="Knot set for the fuzzy system and fuzzy boost (default: %(default)s).",
    )
    args = parser.parse_args()
    main(args.membership)
