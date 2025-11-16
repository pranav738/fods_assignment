"""Utility script to inspect weather-feature correlations.

Run with:

    uv run python test_coll.py

This mirrors the exploratory steps used earlier so you can tweak or extend
them while experimenting with feature engineering ideas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Resolution helper: always anchor on the project root (the script lives there) so we
# can locate the dataset reliably even when executing from another working directory.
DATA_FILE = Path(__file__).resolve().parent / "bangalore 2022-01-01 to 2024-08-09 merged.csv"


# Columns that we want to analyse. These are drawn directly from the weather feed and
# line up with the features you were considering adding to the traffic dataset.
NUMERIC_COLUMNS = [
    "tempmax",
    "tempmin",
    "temp",
    "feelslikemax",
    "feelslikemin",
    "dew",
    "humidity",
    "precip",
    "precipcover",
    "windgust",
    "windspeed",
    "winddir",
    "sealevelpressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
    "severerisk",
]


def load_data() -> pd.DataFrame:
    """Read the CSV and keep only the numeric columns we care about."""

    if not DATA_FILE.exists():  # Guardrail so failures are explicit.
        msg = f"Expected weather file missing: {DATA_FILE}"
        raise FileNotFoundError(msg)

    # Pandas handles type inference, and selecting the columns up front keeps the
    # DataFrame footprint compact and focused on correlation-ready numeric data.
    data = pd.read_csv(DATA_FILE, usecols=NUMERIC_COLUMNS)
    return data


def compute_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for the provided numeric data."""

    # Pearson correlation assumes numeric columns. The formula, for any two columns x
    # and y, is cov(x, y) divided by (std(x) * std(y)). Pandas implements this formula
    # in DataFrame.corr(method="pearson"). Rows with missing values are dropped
    # pairwise automatically.
    corr_matrix = data.corr(method="pearson")
    return corr_matrix


def main() -> None:
    """Load the data, compute correlations, and print the results."""

    data = load_data()

    print("Loaded weather rows:", len(data))
    print("\nPreview of numeric columns (first 5 rows):")
    print(data.head())

    corr_matrix = compute_correlations(data)

    print("\nCorrelation matrix (rounded to three decimals):")
    print(corr_matrix.round(3))


if __name__ == "__main__":
    main()
