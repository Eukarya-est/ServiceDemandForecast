"""Build LightGBM feature matrices for any bucketing granularity.

``build_features`` takes a granularity name, a raw aggregated frame (``bucket_start``,
``elb``, ``request_count``) and the calendar, and returns ``(engineered_df, feature_names)``.
Time, lag and rolling features are driven by the granularity's :class:`GranularitySpec`
(``periods_per_day`` for the intra-day cycle; ``lags``/``rolling`` in buckets). Lag/rolling
are computed per ELB; warm-up NaNs are left in place — LightGBM handles missing values, and
the recursive forecaster fills them as it walks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asp_demand.config import granularity_spec
from asp_demand.features.calendar import CALENDAR_FLAGS, join_calendar

TARGET = "request_count"


def _add_lag_features(
    df: pd.DataFrame, lags: tuple[int, ...], rolling: tuple[int, ...]
) -> list[str]:
    names: list[str] = []
    grouped = df.groupby("elb")[TARGET]
    for lag in lags:
        col = f"lag_{lag}"
        df[col] = grouped.shift(lag)
        names.append(col)
    for window in rolling:
        col = f"rollmean_{window}"
        df[col] = df.groupby("elb")[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w).mean()
        )
        names.append(col)
    return names


def build_features(
    granularity: str, frame: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features for ``granularity`` using its :class:`GranularitySpec`."""
    spec = granularity_spec(granularity)
    df = join_calendar(frame, calendar).sort_values("bucket_start").reset_index(drop=True)
    ts = df["bucket_start"]

    df["dayofweek"] = ts.dt.dayofweek
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    time_features = ["dayofweek", "day", "month", "dow_sin", "dow_cos"]

    periods = spec.periods_per_day
    if periods > 1:
        # position within the day, 0..periods-1 (e.g. hour for hourly, 0/1 for 12h)
        minutes_per_bucket = 1440 // periods
        tod = (ts.dt.hour * 60 + ts.dt.minute) // minutes_per_bucket
        df["tod"] = tod.astype(int)
        df["tod_sin"] = np.sin(2 * np.pi * df["tod"] / periods)
        df["tod_cos"] = np.cos(2 * np.pi * df["tod"] / periods)
        time_features += ["tod", "tod_sin", "tod_cos"]

    lag_features = _add_lag_features(df, spec.lags, spec.rolling)
    return df, [*time_features, *CALENDAR_FLAGS, *lag_features]
