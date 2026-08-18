"""Unit tests for forecast plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asp_demand.config import backtest_path, forecast_path, parquet_path
from asp_demand.viz import plot_forecast


def test_plot_forecast_writes_html(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "bucket_start": pd.date_range("2025-01-01", periods=3, freq="h"),
            "elb": "e",
            "request_count": [1, 2, 3],
        }
    ).to_parquet(parquet_path(tmp_path, "hourly"), index=False)
    pd.DataFrame(
        {
            "bucket_start": pd.date_range("2025-01-01 03:00", periods=3, freq="h"),
            "p50": [4.0, 5.0, 6.0], "p90": [5.0, 6.0, 7.0], "p95": [6.0, 7.0, 8.0],
        }
    ).to_csv(forecast_path(tmp_path, "hourly"), index=False)

    out = plot_forecast(tmp_path, "hourly", height=1000)
    assert out == tmp_path / "forecast_hourly.html"
    assert out.stat().st_size > 0
    html = out.read_text(errors="ignore")
    assert "</html>" in html.lower()
    # tall canvas, quick-range buttons (under legend), pan+wheel-zoom, all 3 quantiles
    compact = html.replace(" ", "")
    assert '"height":1000' in compact
    assert '"label":"7d"' in compact  # quick-range buttons present
    assert '"dragmode":"pan"' in compact  # drag pans through dates
    assert '"scrollZoom":true' in compact  # mouse-wheel zoom
    assert "Plotly.relayout" in html and "jump through time" in html  # jump bar injected
    for q in ("P50", "P90", "P95"):
        assert q in html


def test_plot_backtest_overlays_actual(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "bucket_start": pd.date_range("2025-01-01", periods=3, freq="h"),
            "actual": [10.0, 11.0, 12.0],
            "p50": [9.0, 10.0, 11.0], "p90": [12.0, 13.0, 14.0], "p95": [13.0, 14.0, 15.0],
        }
    ).to_csv(backtest_path(tmp_path, "hourly"), index=False)

    out = plot_forecast(tmp_path, "hourly", kind="backtest")
    assert out == tmp_path / "backtest_hourly.html"
    html = out.read_text(errors="ignore")
    assert "actual" in html and "P95" in html


def test_lttb_decimation_caps_points_and_keeps_endpoints() -> None:
    from asp_demand.viz import _lttb_indices

    x = pd.Series(pd.date_range("2025-01-01", periods=10_000, freq="h"))
    y = pd.Series(np.sin(np.arange(10_000) / 50.0))
    idx = _lttb_indices(x, y, max_points=500)

    assert len(idx) == 500
    assert idx[0] == 0 and idx[-1] == 9999  # endpoints preserved
    assert np.all(np.diff(idx) > 0)  # strictly increasing (no dupes/reorder)
    # below the cap -> no decimation
    assert len(_lttb_indices(x.head(100), y.head(100), 500)) == 100


def test_forecast_chart_uses_webgl_for_actual(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "bucket_start": pd.date_range("2025-01-01", periods=5000, freq="h"),
            "elb": "e",
            "request_count": np.arange(5000),
        }
    ).to_parquet(parquet_path(tmp_path, "hourly"), index=False)
    pd.DataFrame(
        {
            "bucket_start": pd.date_range("2025-07-28", periods=3, freq="h"),
            "p50": [1.0, 2.0, 3.0], "p90": [2.0, 3.0, 4.0], "p95": [3.0, 4.0, 5.0],
        }
    ).to_csv(forecast_path(tmp_path, "hourly"), index=False)

    out = plot_forecast(tmp_path, "hourly", max_points=800)
    html = out.read_text(errors="ignore")
    assert "scattergl" in html.lower()  # actual trace rendered via WebGL


def test_plot_forecast_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predict"):
        plot_forecast(tmp_path, "hourly")
