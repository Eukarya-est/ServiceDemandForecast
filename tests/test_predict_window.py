"""Unit tests for date-window -> horizon planning (pure, no model/IO)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from asp_demand.model.predict import plan_window


def test_hourly_window_aligned_no_gap() -> None:
    # history ends 2024-12-31 23:00 -> forecast 2025-01-01..01-03 = 72 hourly buckets
    horizon, gap, win_start, win_end = plan_window(
        "hourly", pd.Timestamp("2024-12-31 23:00"), date(2025, 1, 1), date(2025, 1, 3)
    )
    assert horizon == 72
    assert gap == 0
    assert win_start == pd.Timestamp("2025-01-01 00:00")
    assert win_end == pd.Timestamp("2025-01-03 23:00")


def test_hourly_window_with_gap() -> None:
    # history ends a day early -> 24-step lead-in before the requested day
    horizon, gap, _, _ = plan_window(
        "hourly", pd.Timestamp("2024-12-30 23:00"), date(2025, 1, 1), date(2025, 1, 1)
    )
    assert gap == 24
    assert horizon == 48  # 24 lead-in + 24 in-window


def test_daily_window() -> None:
    horizon, gap, win_start, win_end = plan_window(
        "daily", pd.Timestamp("2024-12-31"), date(2025, 1, 1), date(2025, 1, 3)
    )
    assert (horizon, gap) == (3, 0)
    assert win_start == pd.Timestamp("2025-01-01")
    assert win_end == pd.Timestamp("2025-01-03")


def test_overlap_with_history_raises() -> None:
    with pytest.raises(ValueError, match="already runs to"):
        plan_window("hourly", pd.Timestamp("2025-01-02 05:00"), date(2025, 1, 1), date(2025, 1, 3))


def test_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="before"):
        plan_window("daily", pd.Timestamp("2024-12-31"), date(2025, 1, 5), date(2025, 1, 1))
