from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ORIGINAL_DIRECT_COLUMNS = {
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
}

DERIVED_FROM_DATE = {
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_holiday",
}

CATEGORICAL_PREFIXES = (
    "Weather Conditions_",
    "Area Name_",
    "Road/Intersection Name_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter encoded dataset to columns corresponding to original Bangalore traffic features."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("datasets/processed/enriched_sample_encoded.csv"),
        help="Path to the encoded dataset with numerical features.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/processed/traffic_features_encoded_filtered.csv"),
        help="Where to write the filtered encoded dataset.",
    )
    return parser.parse_args()


def select_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    for column in columns:
        if column in ORIGINAL_DIRECT_COLUMNS or column in DERIVED_FROM_DATE:
            selected.append(column)
            continue
        if column.startswith(CATEGORICAL_PREFIXES):
            selected.append(column)
    missing_direct = sorted(ORIGINAL_DIRECT_COLUMNS - set(columns))
    return selected, missing_direct


def main() -> None:
    args = parse_args()
    encoded_df = pd.read_csv(args.input_path)
    selected_columns, missing_direct = select_columns(list(encoded_df.columns))
    filtered_df = encoded_df[selected_columns]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output_path, index=False)
    print(
        "Saved filtered dataset with",
        len(filtered_df.index),
        "rows and",
        len(filtered_df.columns),
        "columns to",
        args.output_path,
    )
    if missing_direct:
        print("Missing direct columns not present in encoded dataset:", missing_direct)


if __name__ == "__main__":
    main()
