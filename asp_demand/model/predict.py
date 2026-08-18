"""Quantile forecasting (P50/P90/P95) and backtesting against actuals.

``forecast`` does recursive multi-step prediction into the future (recursing on the P50
median for lags). ``backtest`` does one-step-ahead prediction over a *historical* window
using the real lag values, so predictions can be compared to actuals. Both emit the
configured ``QUANTILES`` as columns ``p50``/``p90``/``p95`` (sorted to avoid crossing).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from asp_demand.config import (
    QUANTILES,
    backtest_path,
    features_path,
    forecast_path,
    granularity_spec,
    model_path,
    parquet_path,
)
from asp_demand.features.build import TARGET, build_features
from asp_demand.features.calendar import load_calendar

QUANTILE_COLS = list(QUANTILES)  # ["p50", "p90", "p95"]


def _step(granularity: str) -> pd.Timedelta:
    """The forecast step (bucket width) for a granularity."""
    return pd.Timedelta(granularity_spec(granularity).freq)


def _load_models(run_dir: Path, granularity: str) -> dict[str, lgb.Booster]:
    return {
        label: lgb.Booster(model_file=str(model_path(run_dir, granularity, label)))
        for label in QUANTILES
    }


def _history(run_dir: Path, granularity: str, elb: str | None) -> tuple[pd.DataFrame, str]:
    history = pd.read_parquet(parquet_path(run_dir, granularity))
    if history.empty:
        raise ValueError(f"no {granularity} data in {run_dir}; run preprocessing first")
    chosen = str(history.groupby("elb").size().idxmax()) if elb is None else elb
    series = (
        history[history["elb"] == chosen]
        .sort_values("bucket_start")
        .reset_index(drop=True)[["bucket_start", "elb", TARGET]]
        .copy()
    )
    return series, chosen


def _non_crossing(quantile_preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Enforce p50 <= p90 <= p95 row-wise (independent quantile models can cross)."""
    out: dict[str, np.ndarray] = {}
    running: np.ndarray | None = None
    for label in QUANTILES:  # ordered low -> high
        values = np.maximum(quantile_preds[label], 0.0)
        running = values if running is None else np.maximum(values, running)
        out[label] = running
    return out


def forecast(
    granularity: str, run_dir: str | Path, horizon: int, elb: str | None = None
) -> pd.DataFrame:
    """Forecast ``horizon`` future buckets; recurse on P50 to feed lags.

    Returns a frame with ``bucket_start`` and ``p50``/``p90``/``p95`` for one ELB.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    step = _step(granularity)
    rd = Path(run_dir)

    series, elb = _history(rd, granularity, elb)
    boosters = _load_models(rd, granularity)
    features = json.loads(features_path(rd, granularity).read_text())
    calendar = load_calendar()

    predictions: list[dict[str, Any]] = []
    for _ in range(horizon):
        next_start = series["bucket_start"].iloc[-1] + step
        placeholder = pd.DataFrame({"bucket_start": [next_start], "elb": [elb], TARGET: [np.nan]})
        series = pd.concat([series, placeholder], ignore_index=True)

        engineered, _ = build_features(granularity, series, calendar)
        x = engineered[features].iloc[[-1]]
        row = _non_crossing({label: np.asarray(b.predict(x)) for label, b in boosters.items()})
        point = {label: float(row[label][0]) for label in QUANTILES}

        series.loc[series.index[-1], TARGET] = point["p50"]  # recurse on the median
        predictions.append({"bucket_start": next_start, **point})

    return pd.DataFrame(predictions)


def backtest(
    granularity: str, run_dir: str | Path, start: date, end: date, elb: str | None = None
) -> pd.DataFrame:
    """One-step-ahead prediction over the historical window ``[start, end]`` vs actuals.

    Uses the *real* lag values (not recursively fed predictions), so each bucket's
    prediction is an honest one-step-ahead estimate. Returns ``bucket_start``, ``actual``
    and ``p50``/``p90``/``p95``. Raises if the window has no actuals in this run.
    """
    if end < start:
        raise ValueError(f"--to {end} is before --from {start}")
    step = _step(granularity)
    rd = Path(run_dir)

    series, _ = _history(rd, granularity, elb)
    boosters = _load_models(rd, granularity)
    features = json.loads(features_path(rd, granularity).read_text())
    calendar = load_calendar()

    engineered, _ = build_features(granularity, series, calendar)
    win_start = pd.Timestamp(start)
    win_end = pd.Timestamp(end) + (pd.Timedelta(days=1) - step)
    window = engineered[
        (engineered["bucket_start"] >= win_start) & (engineered["bucket_start"] <= win_end)
    ]
    if window.empty:
        raise ValueError(f"no {granularity} actuals in [{start}, {end}] for this run")

    x = window[features]
    preds = _non_crossing({label: np.asarray(b.predict(x)) for label, b in boosters.items()})
    out = pd.DataFrame({
        "bucket_start": window["bucket_start"].to_numpy(),
        "actual": window[TARGET].to_numpy(dtype=float),
    })
    for label in QUANTILES:
        out[label] = preds[label]
    return out


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE / MAPE of ``predicted`` vs ``actual`` (MAPE skips zero actuals)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    mask = actual != 0
    mape = (
        float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    return {"mae": mae, "mape": mape}


def plan_window(
    granularity: str, last: pd.Timestamp, start: date, end: date
) -> tuple[int, int, pd.Timestamp, pd.Timestamp]:
    """Translate a date window into recursive-forecast steps (pure; no I/O or model).

    ``last`` is the final bucket present in history. Returns
    ``(horizon, gap_steps, win_start, win_end)`` where ``horizon`` covers from the first
    forecast bucket through ``win_end`` and ``gap_steps`` is the recursive lead-in before
    the window begins. Raises if the window is malformed or overlaps existing history.
    """
    if end < start:
        raise ValueError(f"--to {end} is before --from {start}")
    step = _step(granularity)
    # last bucket of a day is one step before the next midnight (== end for daily)
    win_start = pd.Timestamp(start)
    win_end = pd.Timestamp(end) + (pd.Timedelta(days=1) - step)
    last = pd.Timestamp(last)
    if win_start <= last:
        raise ValueError(
            f"window starts {win_start.date()} but the run's history already runs to {last}; "
            f"pick a window after the data (or use backtest to compare against actuals)"
        )
    first = last + step
    horizon = round((win_end - first) / step) + 1
    gap_steps = round((win_start - first) / step)
    return horizon, gap_steps, win_start, win_end


def forecast_window(
    granularity: str, run_dir: str | Path, start: date, end: date, elb: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Forecast the inclusive date window ``[start, end]`` (dates in the run's tz).

    Returns ``(windowed_df, gap_steps)``; ``gap_steps`` > 0 means the window is further
    than one step past the data, so it is reached through extra recursive lead-in.
    """
    rd = Path(run_dir)
    history = pd.read_parquet(parquet_path(rd, granularity))
    if history.empty:
        raise ValueError(f"no {granularity} data in {rd}; run preprocessing first")
    last = pd.Timestamp(history["bucket_start"].max())

    horizon, gap_steps, win_start, win_end = plan_window(granularity, last, start, end)
    df = forecast(granularity, rd, horizon, elb=elb)
    window = df[(df["bucket_start"] >= win_start) & (df["bucket_start"] <= win_end)]
    return window.reset_index(drop=True), gap_steps


def write_forecast(df: pd.DataFrame, run_dir: str | Path, granularity: str) -> Path:
    """Write a forecast frame to ``run_dir/forecast_<granularity>.csv``; return the path."""
    path = forecast_path(Path(run_dir), granularity)
    df.to_csv(path, index=False)
    return path


def write_backtest(df: pd.DataFrame, run_dir: str | Path, granularity: str) -> Path:
    """Write a backtest frame to ``run_dir/backtest_<granularity>.csv``; return the path."""
    path = backtest_path(Path(run_dir), granularity)
    df.to_csv(path, index=False)
    return path
