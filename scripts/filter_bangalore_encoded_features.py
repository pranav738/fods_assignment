from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ORIGINAL_NUMERIC_FEATURES = [
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Roadwork and Construction Activity",
]

DATE_DERIVED_FEATURES = [
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

CATEGORICAL_PREFIXES = (
    "Weather Conditions_",
    "Area Name_",
    "Road/Intersection Name_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the encoded dataset to retain only features corresponding "
            "to the original Bangalore traffic data."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("datasets/processed/enriched_sample_encoded.csv"),
        help="Path to the encoded dataset to filter.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/processed/bangalore_traffic_encoded_filtered.csv"),
        help="Where to save the filtered dataset.",
    )
    return parser.parse_args()


def determine_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for feature in ORIGINAL_NUMERIC_FEATURES:
        if feature in columns:
            selected.append(feature)
    for feature in DATE_DERIVED_FEATURES:
        if feature in columns:
            selected.append(feature)
    for prefix in CATEGORICAL_PREFIXES:
        selected.extend([column for column in columns if column.startswith(prefix)])
    return selected


def filter_dataset(input_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path)
    columns_to_keep = determine_columns(list(dataframe.columns))
    if not columns_to_keep:
        raise ValueError("No matching columns were found in the encoded dataset.")
    return dataframe.loc[:, columns_to_keep]


def main() -> None:
    args = parse_args()
    filtered = filter_dataset(args.input_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output_path, index=False)
    print(
        "Filtered dataset written to",
        args.output_path,
        "with",
        len(filtered.columns),
        "columns and",
        len(filtered.index),
        "rows.",
    )


if __name__ == "__main__":
    main()
