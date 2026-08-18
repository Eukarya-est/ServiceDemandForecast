"""Integration tests for quantile training, forecasting, and backtesting."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from asp_demand.config import QUANTILES, model_path, parquet_path
from asp_demand.model.predict import backtest, forecast, score
from asp_demand.model.train import train_model


def _make_run(tmp_path: Path) -> Path:
    """A run dir with ~10 days of synthetic hourly data (diurnal + weekly signal)."""
    idx = pd.date_range("2025-05-01", periods=24 * 10, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    counts = 1000 + 400 * np.sin(2 * np.pi * hour / 24) + 100 * (dow < 5)
    rd = tmp_path / "run"
    rd.mkdir()
    pd.DataFrame({"bucket_start": idx, "elb": "e", "request_count": counts.astype(int)}).to_parquet(
        parquet_path(rd, "hourly"), index=False
    )
    return rd


def test_train_writes_a_model_per_quantile(tmp_path: Path) -> None:
    rd = _make_run(tmp_path)
    metrics = train_model("hourly", rd)
    assert set(metrics) == {"mae", "rmse", "mape"}
    for label in QUANTILES:
        assert model_path(rd, "hourly", label).exists()


def test_train_reports_per_quantile_progress(tmp_path: Path) -> None:
    rd = _make_run(tmp_path)
    seen: list[tuple[int, int]] = []
    train_model("hourly", rd, progress_cb=lambda done, total: seen.append((done, total)))
    n = len(QUANTILES)
    assert seen[0] == (0, n)  # reported up front
    assert seen[-1] == (n, n)  # finished all quantiles
    assert all(total == n for _, total in seen)


def test_forecast_emits_non_crossing_quantiles(tmp_path: Path) -> None:
    rd = _make_run(tmp_path)
    train_model("hourly", rd)
    df = forecast("hourly", rd, horizon=12)

    assert list(df.columns) == ["bucket_start", "p50", "p90", "p95"]
    assert len(df) == 12
    # quantiles must not cross and stay non-negative
    assert (df["p50"] <= df["p90"]).all()
    assert (df["p90"] <= df["p95"]).all()
    assert (df["p50"] >= 0).all()


def test_backtest_returns_actual_and_quantiles(tmp_path: Path) -> None:
    rd = _make_run(tmp_path)
    train_model("hourly", rd)
    # a day that exists in history (1–10 May)
    df = backtest("hourly", rd, date(2025, 5, 9), date(2025, 5, 9))

    assert list(df.columns) == ["bucket_start", "actual", "p50", "p90", "p95"]
    assert len(df) == 24  # a full hourly day
    assert (df["p90"] <= df["p95"]).all()
    s = score(df["actual"].to_numpy(), df["p50"].to_numpy())
    assert s["mae"] >= 0
