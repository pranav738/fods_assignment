from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAY_NAME_TO_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

DAYS_IN_WEEK = 7.0
DAYS_IN_YEAR = 365.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode enriched Bangalore traffic dataset with numeric features only."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("datasets/raw/enriched_sample.csv"),
        help="Path to the enriched CSV to encode.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/processed/enriched_sample_encoded.csv"),
        help="Where to write the numeric encoded dataset.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def boolean_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("int8")
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    numeric = pd.to_numeric(series, errors="coerce")
    combined = mapped.fillna(numeric)
    if combined.isna().any():
        problematic = series[combined.isna()].unique()
        raise ValueError(f"Unexpected boolean values found: {problematic}")
    return combined.astype("int8")


def encode_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    day_numbers = df["day_of_week"].map(DAY_NAME_TO_INDEX)
    if day_numbers.isna().any():
        missing = df.loc[day_numbers.isna(), "day_of_week"].unique()
        raise ValueError(f"Unrecognised day names found: {missing}")
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_numbers / DAYS_IN_WEEK)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_numbers / DAYS_IN_WEEK)
    return df.drop(columns=["day_of_week"])


def encode_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Date column contains invalid entries that cannot be parsed.")
    df = encode_day_of_week(df)
    day_of_year = df["Date"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    return df.drop(columns=["Date"])


def encode_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = boolean_to_int(df[column])
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, dtype="uint8", drop_first=False)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric = df.select_dtypes(exclude=["number"])
    if len(non_numeric.columns) > 0:
        raise ValueError(
            f"Non-numeric columns remain after encoding: {list(non_numeric.columns)}"
        )
    return df


def encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_temporal_columns(df)
    categorical_columns = [
        "Weather Conditions",
        "Area Name",
        "Road/Intersection Name",
    ]
    df = encode_categoricals(df, categorical_columns)
    boolean_columns = ["is_holiday", "Roadwork and Construction Activity"]
    df = encode_boolean_columns(df, boolean_columns)
    df = ensure_numeric(df)
    return df


def main() -> None:
    args = parse_args()
    input_df = load_dataset(args.input_path)
    encoded_df = encode_dataset(input_df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(args.output_path, index=False)
    print(
        "Encoded dataset written to",
        args.output_path,
        "with",
        len(encoded_df.columns),
        "columns and",
        len(encoded_df.index),
        "rows.",
    )


if __name__ == "__main__":
    main()
