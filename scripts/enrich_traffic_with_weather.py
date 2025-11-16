from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


DATE_COLUMN_TRAFFIC = "Date"
DATE_COLUMN_WEATHER = "datetime"


WEATHER_FEATURES = [
    "tempmax",
    "tempmin",
    "feelslikemax",
    "feelslikemin",
    "dew",
    "humidity",
    "precip",
    "precipcover",
    "windgust",
    "windspeed",
    "winddir",
    "cloudcover",
    "visibility",
    "solarradiation",
    "uvindex",
]


def time_to_minutes(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, errors="coerce")
    return (
        times.dt.hour * 60
        + times.dt.minute
        + times.dt.second.div(60)
        + times.dt.microsecond.div(60_000_000)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich Bangalore traffic dataset with weather features."
    )
    parser.add_argument(
        "--traffic-path",
        type=Path,
        default=Path("Banglore_traffic_Dataset.csv"),
        help="Path to the traffic dataset CSV.",
    )
    parser.add_argument(
        "--weather-path",
        type=Path,
        default=Path("bangalore 2022-01-01 to 2024-08-09 merged.csv"),
        help="Path to the weather dataset CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("Banglore_traffic_with_weather.csv"),
        help="Where to write the enriched dataset.",
    )
    parser.add_argument(
        "--holidays-path",
        type=Path,
        default=Path("holiidays.md"),
        help="Path to the markdown file listing holidays.",
    )
    return parser.parse_args()


def load_traffic(path: Path) -> pd.DataFrame:
    traffic = pd.read_csv(path)
    traffic[DATE_COLUMN_TRAFFIC] = pd.to_datetime(traffic[DATE_COLUMN_TRAFFIC]).dt.date
    return traffic


def load_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path)
    weather[DATE_COLUMN_WEATHER] = pd.to_datetime(weather[DATE_COLUMN_WEATHER]).dt.date
    required_columns = [DATE_COLUMN_WEATHER, *WEATHER_FEATURES, "sunrise", "sunset"]
    missing = [column for column in required_columns if column not in weather.columns]
    if missing:
        raise ValueError(f"Weather dataset missing required columns: {missing}")
    weather = weather[required_columns].drop_duplicates(subset=DATE_COLUMN_WEATHER)
    return weather


def load_holiday_dates(path: Path) -> set[date]:
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    holidays: set[date] = set()
    current_year: int | None = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("#").strip()
                if header.isdigit():
                    current_year = int(header)
                continue
            if "-" not in line or current_year is None:
                continue
            month_part, days_part = line.split("-", 1)
            month_key = month_part.strip().lower()
            month_abbr = month_key[:3]
            month_number = month_map.get(month_abbr)
            if month_number is None:
                continue
            days_clean = days_part.replace(" ", "")
            for token in days_clean.split(","):
                if not token:
                    continue
                try:
                    day_value = int(token)
                except ValueError:
                    continue
                holidays.add(date(current_year, month_number, day_value))
    return holidays


def is_holiday(day: date, specific_dates: set[date]) -> bool:
    if day in specific_dates:
        return True
    if day.weekday() == 6:
        return True
    if day.weekday() == 5:
        occurrence = (day.day - 1) // 7 + 1
        if occurrence in {2, 4}:
            return True
    return False


def enrich_datasets(
    traffic: pd.DataFrame, weather: pd.DataFrame, holiday_dates: set[date]
) -> pd.DataFrame:
    weather_aligned = weather.rename(columns={DATE_COLUMN_WEATHER: DATE_COLUMN_TRAFFIC})
    merged = traffic.merge(weather_aligned, on=DATE_COLUMN_TRAFFIC, how="left")
    merged["day_of_week"] = pd.to_datetime(merged[DATE_COLUMN_TRAFFIC]).dt.day_name()
    # Add holiday feature based on holiidays.md
    merged["is_holiday"] = merged[DATE_COLUMN_TRAFFIC].apply(
        lambda value: is_holiday(value, holiday_dates)
    )
    # Convert roadwork column to boolean flag
    roadwork_col = "Roadwork and Construction Activity"
    merged[roadwork_col] = (
        merged[roadwork_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": True, "no": False})
    )
    sunrise_minutes = time_to_minutes(merged["sunrise"])
    sunset_minutes = time_to_minutes(merged["sunset"])
    merged["minutes_of_daylight"] = sunset_minutes - sunrise_minutes
    return merged


def main() -> None:
    args = parse_args()
    traffic = load_traffic(args.traffic_path)
    weather = load_weather(args.weather_path)
    holiday_dates = load_holiday_dates(args.holidays_path)
    enriched = enrich_datasets(traffic, weather, holiday_dates)
    enriched.to_csv(args.output_path, index=False)
    print(
        "Enriched dataset written to",
        args.output_path,
        "with",
        len(enriched.index),
        "rows.",
    )


if __name__ == "__main__":
    main()
