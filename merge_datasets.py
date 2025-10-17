from pathlib import Path

import pandas as pd


CUTOFF_DATE = pd.Timestamp("2023-07-01")
BASE_DIR = Path(__file__).resolve().parent
SOURCE_BEFORE = BASE_DIR / "bangalore 2022-01-01 to 2024-06-30.csv"
SOURCE_AFTER = BASE_DIR / "bangalore 2023-07-01 to 2024-08-09.csv"
OUTPUT_PATH = BASE_DIR / "bangalore 2022-01-01 to 2024-08-09 merged.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_before_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    mask = pd.to_datetime(df["datetime"]) < cutoff
    return df.loc[mask].copy()


def main() -> None:
    before_df = filter_before_cutoff(load_dataset(SOURCE_BEFORE), CUTOFF_DATE)
    after_df = load_dataset(SOURCE_AFTER)
    merged_df = pd.concat([before_df, after_df], ignore_index=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
