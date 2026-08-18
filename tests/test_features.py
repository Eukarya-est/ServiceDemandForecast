"""Unit tests for feature engineering (calendar join + lag shapes)."""

from __future__ import annotations

import pandas as pd

from asp_demand.features.build import build_features
from asp_demand.features.calendar import build_calendar


def _calendar() -> pd.DataFrame:
    dates = pd.date_range("2025-05-01", "2025-05-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "is_weekend": (dates.weekday >= 5).astype(int),
            "is_holiday": 0,
            "is_business_day": (dates.weekday < 5).astype(int),
            "day_before_holiday": 0,
            "day_after_holiday": 0,
            "in_long_weekend": 0,
            "is_nenmatsu": 0,
        }
    )


def test_build_calendar_flags_year_end_period() -> None:
    cal = build_calendar(2024, 2025).set_index("date")["is_nenmatsu"]
    # Dec 28 (lead-in) .. Jan 3 are flagged; the 27th and 4th are not.
    assert cal["2024-12-27"] == 0
    assert cal.loc["2024-12-28":"2025-01-03"].eq(1).all()
    assert cal["2025-01-04"] == 0
    assert int(cal.sum()) == 7 * 2  # 7 nenmatsu days per year, 2 years


def test_hourly_features_have_lags_and_calendar() -> None:
    idx = pd.date_range("2025-05-06", periods=200, freq="h")
    hourly = pd.DataFrame({"bucket_start": idx, "elb": "e", "request_count": range(200)})
    df, features = build_features("hourly", hourly, _calendar())

    assert {"lag_1", "lag_24", "lag_168", "rollmean_24", "tod", "is_weekend", "is_nenmatsu"} <= set(
        features
    )
    # lag_1 at row i equals request_count at row i-1
    assert df["lag_1"].iloc[5] == df["request_count"].iloc[4]
    assert pd.isna(df["lag_1"].iloc[0])


def test_daily_features_have_expected_columns() -> None:
    idx = pd.date_range("2025-05-01", periods=20, freq="D")
    daily = pd.DataFrame({"bucket_start": idx, "elb": "e", "request_count": range(20)})
    df, features = build_features("daily", daily, _calendar())

    assert {"lag_1", "lag_7", "lag_14", "rollmean_7", "dow_sin"} <= set(features)
    assert "tod" not in features  # no intra-day cycle for daily
    assert df["lag_7"].iloc[7] == df["request_count"].iloc[0]


def test_12h_features_use_two_periods_per_day() -> None:
    idx = pd.date_range("2025-05-01", periods=40, freq="12h")
    frame = pd.DataFrame({"bucket_start": idx, "elb": "e", "request_count": range(40)})
    df, features = build_features("12h", frame, _calendar())

    # periods_per_day=2 -> lags [1, 2, 14], intra-day position 0/1
    assert {"lag_1", "lag_2", "lag_14", "rollmean_2", "tod"} <= set(features)
    assert set(df["tod"].unique()) <= {0, 1}
