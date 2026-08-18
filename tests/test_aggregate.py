"""Unit tests for log aggregation (UTC->JST bucketing, gap filling, daily rollup)."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import pandas as pd

from asp_demand.config import forecast_path, model_path, parquet_path
from asp_demand.preprocessing.aggregate import aggregate, clean_cache, downsample, run

ELB = "app/example-alb/abc"


def _write_log(root: Path, day: date, lines: list[str]) -> None:
    day_dir = root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(day_dir / "test.log.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _line(ts_utc: str) -> str:
    return f"https {ts_utc} {ELB} 1.2.3.4:5 6.7.8.9:0 0 0 0 200 200 1 1 rest"


def test_utc_bucketing_and_complete_day(tmp_path: Path) -> None:
    _write_log(
        tmp_path,
        date(2025, 5, 6),
        [
            _line("2025-05-06T00:10:00.000000Z"),
            _line("2025-05-06T00:40:00.000000Z"),
            _line("2025-05-06T05:30:00.000000Z"),
        ],
    )
    hourly = aggregate(
        date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, tz="UTC", cache_dir=tmp_path / "cache"
    )

    assert len(hourly) == 24  # complete day, zero-filled
    by_hour = dict(zip(hourly["bucket_start"].astype(str), hourly["request_count"], strict=True))
    assert by_hour["2025-05-06 00:00:00"] == 2
    assert by_hour["2025-05-06 05:00:00"] == 1
    assert by_hour["2025-05-06 12:00:00"] == 0
    assert int(hourly["request_count"].sum()) == 3


def test_downsample_to_coarser_granularities() -> None:
    idx = pd.date_range("2025-05-06", periods=24, freq="h")
    hourly = pd.DataFrame({"bucket_start": idx, "elb": "e", "request_count": [1] * 24})

    daily = downsample(hourly, "1D")
    assert len(daily) == 1 and int(daily["request_count"].iloc[0]) == 24

    six = downsample(hourly, "6h")
    assert len(six) == 4  # 24h / 6h
    assert list(six["request_count"]) == [6, 6, 6, 6]
    assert list(six["bucket_start"].dt.hour) == [0, 6, 12, 18]


def test_tz_shifts_day(tmp_path: Path) -> None:
    # UTC 16:30 on May 6 -> JST 01:30 on May 7.
    _write_log(tmp_path, date(2025, 5, 6), [_line("2025-05-06T16:30:00.000000Z")])

    # Requesting the JST date the event lands on includes it...
    jst7 = aggregate(
        date(2025, 5, 7), date(2025, 5, 7), root=tmp_path, tz="Asia/Tokyo",
        cache_dir=tmp_path / "cache",
    )
    by = dict(zip(jst7["bucket_start"].astype(str), jst7["request_count"], strict=True))
    assert by["2025-05-07 01:00:00"] == 1
    assert sum(by.values()) == 1
    # ...and requesting the prior JST date excludes it (boundary filtering).
    jst6 = aggregate(
        date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, tz="Asia/Tokyo",
        cache_dir=tmp_path / "cache2",
    )
    assert int(jst6["request_count"].sum()) == 0


def test_progress_flag_does_not_change_results(tmp_path: Path) -> None:
    lines = [_line("2025-05-06T00:10:00.000000Z"), _line("2025-05-06T00:20:00.000000Z")]
    _write_log(tmp_path, date(2025, 5, 6), lines)

    cache = tmp_path / "cache"
    plain = aggregate(date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, cache_dir=cache)
    # refresh=True re-reads through the parallel path; result must match.
    with_bar = aggregate(
        date(2025, 5, 6), date(2025, 5, 6), root=tmp_path,
        cache_dir=cache, progress=True, refresh=True,
    )
    assert list(plain["request_count"]) == list(with_bar["request_count"])


def test_complete_day_zero_fills_gaps(tmp_path: Path) -> None:
    # Two events 2 hours apart -> the whole day is emitted, middle hour zero-filled.
    _write_log(
        tmp_path,
        date(2025, 5, 6),
        [_line("2025-05-06T00:10:00.000000Z"), _line("2025-05-06T02:10:00.000000Z")],
    )
    hourly = aggregate(
        date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, tz="UTC", cache_dir=tmp_path / "cache"
    )
    assert len(hourly) == 24
    by = dict(zip(hourly["bucket_start"].astype(str), hourly["request_count"], strict=True))
    assert (by["2025-05-06 00:00:00"], by["2025-05-06 01:00:00"], by["2025-05-06 02:00:00"]) == (
        1, 0, 1,
    )


def test_run_writes_timestamped_run_dir(tmp_path: Path) -> None:
    _write_log(tmp_path, date(2025, 5, 6), [_line("2025-05-06T00:10:00.000000Z")])
    run_dir = tmp_path / "runs" / "250506000000"

    returned, frames = run(
        date(2025, 5, 6), date(2025, 5, 6),
        root=tmp_path, run_dir=run_dir, cache_dir=tmp_path / "cache",
    )
    assert returned == run_dir
    # every granularity is emitted
    for name in ("hourly", "3h", "6h", "12h", "daily"):
        assert parquet_path(run_dir, name).exists()
        assert int(frames[name]["request_count"].sum()) == 1


def test_clean_cache_scopes(tmp_path: Path) -> None:
    cdir = tmp_path / "cache"
    (cdir / "UTC").mkdir(parents=True)
    (cdir / "UTC" / "2025-05-06.parquet").write_bytes(b"x" * 10)
    (cdir / "2025-01-01.parquet").write_bytes(b"y" * 20)  # legacy flat / orphaned

    # dry-run reports but removes nothing
    count, freed = clean_cache(cache_dir=cdir, dry_run=True)
    assert (count, freed) == (2, 30)
    assert (cdir / "UTC" / "2025-05-06.parquet").exists()

    # orphaned removes only the flat root file, keeps the per-tz cache
    count, _ = clean_cache(orphaned=True, cache_dir=cdir)
    assert count == 1
    assert not (cdir / "2025-01-01.parquet").exists()
    assert (cdir / "UTC" / "2025-05-06.parquet").exists()

    # tz-scoped removes only that timezone (and prunes the empty dir)
    count, _ = clean_cache(tz="UTC", cache_dir=cdir)
    assert count == 1
    assert not (cdir / "UTC").exists()


def test_path_helpers_layout() -> None:
    run_dir = Path("/data/runs/250619120000")
    assert parquet_path(run_dir, "hourly").name == "hourly.parquet"
    assert model_path(run_dir, "daily").name == "daily_p50.txt"  # default quantile
    assert model_path(run_dir, "daily", "p90").name == "daily_p90.txt"
    assert forecast_path(run_dir, "hourly").name == "forecast_hourly.csv"


def test_resolve_run_dir_handles_bare_name_and_path() -> None:
    from asp_demand.config import RUNS_DIR, resolve_run_dir

    # a bare run name (what the API/UI sends) resolves under data/runs/
    assert resolve_run_dir("260619140655") == RUNS_DIR / "260619140655"
    # an existing path (what the CLI passes) is used as-is
    assert resolve_run_dir(str(RUNS_DIR)) == RUNS_DIR


def test_per_day_cache_avoids_reread(tmp_path: Path) -> None:
    import shutil

    cache = tmp_path / "cache"
    _write_log(tmp_path, date(2025, 5, 6), [_line("2025-05-06T00:10:00.000000Z")])

    first = aggregate(date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, cache_dir=cache)
    assert int(first["request_count"].sum()) == 1

    # Remove the source: a cached day must still return its counts...
    shutil.rmtree(tmp_path / "2025")
    cached = aggregate(date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, cache_dir=cache)
    assert int(cached["request_count"].sum()) == 1

    # ...while refresh=True re-reads the (now empty) source.
    refreshed = aggregate(
        date(2025, 5, 6), date(2025, 5, 6), root=tmp_path, cache_dir=cache, refresh=True
    )
    assert refreshed.empty
