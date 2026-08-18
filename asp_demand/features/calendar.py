"""Japan calendar: build/generate the CSV and join it onto aggregated buckets."""

from __future__ import annotations

from pathlib import Path

import jpholiday
import pandas as pd

from asp_demand.config import CALENDAR_CSV

CALENDAR_FLAGS = [
    "is_weekend",
    "is_holiday",
    "is_business_day",
    "day_before_holiday",
    "day_after_holiday",
    "in_long_weekend",
    "is_nenmatsu",
]


def build_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    """Build a per-date Japan calendar frame for ``[start_year, end_year]`` inclusive."""
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    holiday_names = [jpholiday.is_holiday_name(d.date()) or "" for d in dates]

    df = pd.DataFrame({"date": dates})
    df["holiday_name"] = holiday_names
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5
    df["is_holiday"] = df["holiday_name"] != ""
    df["is_business_day"] = ~(df["is_weekend"] | df["is_holiday"])
    df["day_before_holiday"] = df["is_holiday"].shift(-1, fill_value=False)
    df["day_after_holiday"] = df["is_holiday"].shift(1, fill_value=False)

    non_business = ~df["is_business_day"]
    runs = (non_business != non_business.shift()).cumsum()
    run_sizes = non_business.groupby(runs).transform("sum")
    df["in_long_weekend"] = non_business & (run_sizes >= 3)

    # Year-end / New-Year period (年末年始): Dec 29–Jan 3, plus Dec 28 as the lead-in
    # day-before-holiday. A big de-facto-holiday demand anomaly that jpholiday misses
    # (it only marks Jan 1), so an explicit flag helps the model.
    month, day = df["date"].dt.month, df["date"].dt.day
    df["is_nenmatsu"] = ((month == 12) & (day >= 28)) | ((month == 1) & (day <= 3))

    df[CALENDAR_FLAGS] = df[CALENDAR_FLAGS].astype(int)
    return df


def generate_calendar(
    start_year: int, end_year: int, path: Path = CALENDAR_CSV
) -> int:
    """Build the calendar and write it to ``path``; return the number of rows."""
    df = build_calendar(start_year, end_year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def load_calendar(path: Path = CALENDAR_CSV) -> pd.DataFrame:
    """Load the Japan calendar CSV (date + holiday/weekday flags)."""
    cal = pd.read_csv(path, parse_dates=["date"])
    missing = [flag for flag in CALENDAR_FLAGS if flag not in cal.columns]
    if missing:
        raise ValueError(
            f"calendar at {path} is missing {missing}; regenerate it "
            "(POST /calendar or `python scripts/gen_japan_calendar.py`)"
        )
    for flag in CALENDAR_FLAGS:
        cal[flag] = cal[flag].astype(int)
    return cal


def join_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Left-join calendar flags onto a frame with a ``bucket_start`` column (by date)."""
    out = df.copy()
    out["date"] = out["bucket_start"].dt.normalize()
    cols = ["date", *CALENDAR_FLAGS]
    return out.merge(calendar[cols], on="date", how="left")
