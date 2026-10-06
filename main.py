"""Run the full research pipeline and write the paper charts."""

from src.charts import print_chart_data, run as build_charts
from src.compare import run as compare
from src.features import run as select_features
from src.fuzzify import run as fuzzify
from src.fuzzy_rules import run as apply_rules
from src.preprocess import run as preprocess
from src.repeated import run as repeat_splits


def main() -> None:
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
    main()
