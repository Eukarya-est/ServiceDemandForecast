"""Aggregate raw ALB logs into hourly and daily request counts.

The raw logs are partitioned by **UTC** date in ``YYYY/MM/DD`` folders, but ``start`` /
``end`` are interpreted as dates in the chosen bucketing **timezone** (``tz``, default
JST; set to ``UTC`` to match the AWS Athena aggregates). Because a target-tz day spans
two UTC folders, we scan the requested folders **padded by one day on each side**, bucket
events in ``tz``, keep only buckets whose tz-date is in ``[start, end]``, and zero-fill to
complete days — so daily totals are whole days.

Reading is the bottleneck. It is both CPU-bound (gzip + Python line parsing, which the
GIL serialises) and I/O-bound (objects fetched over the network from S3). A **process**
pool wins in both cases — each worker bypasses the GIL for true parallel parsing and runs
its own blocking reads. Each UTC folder day's counts are cached (per timezone) to
``data/cache/<tz>/<date>.parquet`` so a day is never re-aggregated; later runs only read
new/uncached days.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from asp_demand.config import (
    CACHE_DIR,
    DEFAULT_TZ,
    GRANULARITIES,
    RAW_LOG_URI,
    granularity_spec,
    make_run_dir,
    parquet_path,
    update_latest,
)
from asp_demand.preprocessing.log_parser import parse_file
from asp_demand.storage import get_fs

DayCounter = Counter[tuple[datetime, str]]

# Avoid fork() in a possibly multi-threaded parent (s3fs starts an event-loop
# thread); forkserver gives clean, deadlock-free workers on Linux.
_START_METHOD = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"


def _default_workers() -> int:
    """Default worker count (capped; raise via --workers for I/O-bound S3 reads)."""
    return min(32, max(4, os.cpu_count() or 4))


def _date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)
    return days


def _glob_day(fs: Any, base: str, day: date) -> list[str]:
    """List ``*.log.gz`` paths for a single UTC partition day."""
    pattern = f"{base.rstrip('/')}/{day.year:04d}/{day.month:02d}/{day.day:02d}/*.log.gz"
    return sorted(fs.glob(pattern))


def iter_log_files(fs: Any, base: str, start: date, end: date) -> Iterator[str]:
    """Yield all ``*.log.gz`` paths in the UTC-partitioned tree for ``[start, end]``."""
    for day in _date_range(start, end):
        yield from _glob_day(fs, base, day)


def _count_file(fs: Any, path: str, tz: str) -> DayCounter:
    """Stream one ``.gz`` object and return tz-hourly ``(bucket, elb) -> count``."""
    zone = ZoneInfo(tz)
    counts: DayCounter = Counter()
    for ts_utc, elb in parse_file(fs, path):
        ts_local = ts_utc.astimezone(zone)
        hour_bucket = ts_local.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        counts[(hour_bucket, elb)] += 1
    return counts


# --- process-pool workers (build the filesystem lazily, once per worker) -----
@cache
def _cached_fs(uri: str) -> tuple[Any, str]:
    return get_fs(uri)


def _glob_day_task(uri: str, day: date) -> list[str]:
    fs, base = _cached_fs(uri)
    return _glob_day(fs, base, day)


def _count_task(uri: str, path: str, tz: str) -> DayCounter:
    fs, _ = _cached_fs(uri)
    return _count_file(fs, path, tz)


def _counter_to_frame(counts: DayCounter) -> pd.DataFrame:
    rows = [
        {"bucket_start": bucket, "elb": elb, "request_count": count}
        for (bucket, elb), count in counts.items()
    ]
    frame = pd.DataFrame(rows, columns=["bucket_start", "elb", "request_count"])
    if frame.empty:
        return frame.astype(
            {"bucket_start": "datetime64[ns]", "elb": "object", "request_count": "int64"}
        )
    return frame.sort_values("bucket_start").reset_index(drop=True)


def _reindex_full_days(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Zero-fill each ELB's series to every hour of the complete days ``[start, end]``."""
    if df.empty:
        return df
    index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    frames: list[pd.DataFrame] = []
    for elb, group in df.groupby("elb"):
        filled = (
            group.set_index("bucket_start")["request_count"]
            .reindex(index, fill_value=0)
            .rename_axis("bucket_start")
            .reset_index(name="request_count")
        )
        filled["elb"] = elb
        frames.append(filled[["bucket_start", "elb", "request_count"]])
    return pd.concat(frames, ignore_index=True)


def downsample(hourly: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate the hourly base series to a coarser ``freq`` (e.g. 3h, 6h, 12h, 1D)."""
    if hourly.empty:
        return hourly.copy()
    coarser = hourly.copy()
    coarser["bucket_start"] = coarser["bucket_start"].dt.floor(freq)
    rolled = coarser.groupby(["bucket_start", "elb"])["request_count"].sum().reset_index()
    return rolled.sort_values("bucket_start").reset_index(drop=True)


def _cache_path(cache_dir: Path, day: date) -> Path:
    return cache_dir / f"{day.isoformat()}.parquet"


def _prune_empty_dirs(root: Path) -> None:
    """Remove now-empty subdirectories under ``root`` (deepest first)."""
    subdirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    )
    for d in subdirs:
        with contextlib.suppress(OSError):  # OSError -> directory not empty
            d.rmdir()


def clean_cache(
    tz: str | None = None,
    *,
    orphaned: bool = False,
    cache_dir: Path = CACHE_DIR,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Remove cached aggregation parquet; return ``(files_removed, bytes_freed)``.

    ``orphaned`` -> only legacy flat files directly under ``cache_dir`` (keeps the
    per-tz subdirs). ``tz`` -> only that timezone's cache. Neither -> everything.
    ``dry_run`` reports without deleting.
    """
    if not cache_dir.exists():
        return (0, 0)
    if orphaned:
        targets = list(cache_dir.glob("*.parquet"))  # flat root only
    elif tz is not None:
        targets = list((cache_dir / tz.replace("/", "-")).rglob("*.parquet"))
    else:
        targets = list(cache_dir.rglob("*.parquet"))

    total = sum(p.stat().st_size for p in targets)
    if not dry_run:
        for path in targets:
            path.unlink(missing_ok=True)
        _prune_empty_dirs(cache_dir)
    return (len(targets), total)


def _process_days(
    uri: str,
    days: list[date],
    cache_dir: Path,
    tz: str,
    workers: int,
    progress: bool,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Read the given UTC folder days across a process pool; cache tz-hourly counts."""
    # Drop any stale cache for days we are about to reprocess (matters on refresh).
    for day in days:
        _cache_path(cache_dir, day).unlink(missing_ok=True)

    # Glob each day concurrently (S3 listing is itself a round-trip).
    day_files: dict[date, list[str]] = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context(_START_METHOD)
    ) as pool:
        glob_futures = {pool.submit(_glob_day_task, uri, day): day for day in days}
        for gfut in as_completed(glob_futures):
            day_files[glob_futures[gfut]] = gfut.result()

    tasks = [(day, path) for day, paths in day_files.items() for path in paths]
    per_day: dict[date, DayCounter] = defaultdict(Counter)
    if tasks:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context(_START_METHOD)
        ) as pool:
            count_futures = {pool.submit(_count_task, uri, path, tz): day for day, path in tasks}
            completed = as_completed(count_futures)
            if progress:
                completed = tqdm(completed, total=len(tasks), desc="preprocess", unit="file")
            if progress_cb:
                progress_cb(0, len(tasks))
            for done, cfut in enumerate(completed, start=1):
                per_day[count_futures[cfut]] += cfut.result()
                if progress_cb:
                    progress_cb(done, len(tasks))
                if cancel_check and cancel_check():
                    for pending in count_futures:
                        pending.cancel()  # drop not-yet-started files
                    raise InterruptedError("preprocess cancelled")

    # Cache every day that actually had files (even if it parsed to zero rows),
    # so it is treated as done and skipped next time.
    for day in days:
        if day_files.get(day):
            _counter_to_frame(per_day.get(day, Counter())).to_parquet(
                _cache_path(cache_dir, day), index=False
            )


def aggregate(
    start: date,
    end: date,
    root: str | Path | None = None,
    *,
    tz: str | None = None,
    workers: int | None = None,
    progress: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Aggregate logs for the tz-dates ``[start, end]`` into the hourly base frame.

    ``start`` / ``end`` are dates in ``tz`` (default ``ASP_TZ`` -> JST; use ``UTC`` to
    match Athena). UTC folders are scanned padded by ±1 day so target-tz days are whole.
    Coarser granularities are derived from this base via :func:`downsample`.
    ``root`` is a local path or ``s3://`` URI. Caches live under
    ``cache_dir/<tz>/`` (default ``data/cache``); ``refresh=True`` ignores them.
    """
    tz_name = tz or DEFAULT_TZ
    uri = RAW_LOG_URI if root is None else str(root)
    base = CACHE_DIR if cache_dir is None else Path(cache_dir)
    cdir = base / tz_name.replace("/", "-")
    cdir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or _default_workers()

    # Pad by one UTC day each side: a target-tz day pulls from adjacent UTC folders.
    folder_days = _date_range(start - timedelta(days=1), end + timedelta(days=1))
    todo = [day for day in folder_days if refresh or not _cache_path(cdir, day).exists()]
    if progress and len(todo) < len(folder_days):
        print(f"cache: {len(folder_days) - len(todo)}/{len(folder_days)} day(s) reused", flush=True)

    if todo:
        _process_days(uri, todo, cdir, tz_name, n_workers, progress, progress_cb, cancel_check)

    # Sum across folder caches (a tz hour can span two UTC folders), then keep only
    # buckets whose tz-date is in the requested range and zero-fill to complete days.
    frames = [
        df
        for day in folder_days
        if _cache_path(cdir, day).exists()
        and not (df := pd.read_parquet(_cache_path(cdir, day))).empty
    ]
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        raw["bucket_start"] = pd.to_datetime(raw["bucket_start"])
        merged = raw.groupby(["bucket_start", "elb"])["request_count"].sum().reset_index()
        bucket_date = merged["bucket_start"].dt.date
        merged = merged[(bucket_date >= start) & (bucket_date <= end)]
    else:
        merged = _counter_to_frame(Counter())

    return _reindex_full_days(merged, start, end)


def run(
    start: date,
    end: date,
    root: str | Path | None = None,
    *,
    tz: str | None = None,
    run_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    granularities: tuple[str, ...] | None = None,
    workers: int | None = None,
    progress: bool = False,
    refresh: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Aggregate into a timestamped run dir; return ``(run_dir, {granularity: frame})``.

    The hourly base is read once and downsampled to every requested granularity
    (default: all). A fresh ``data/runs/<yymmddhhmmss>/`` is created (and pointed to by
    ``latest``) unless ``run_dir`` is given. When ``progress`` is true, the absolute
    output location is printed once and a per-file progress bar is shown.
    """
    grans = GRANULARITIES if granularities is None else tuple(granularities)
    rd = make_run_dir() if run_dir is None else Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"Preprocessed output -> {rd.resolve()} ({', '.join(grans)})", flush=True)

    base = aggregate(
        start,
        end,
        root=root,
        tz=tz,
        workers=workers,
        progress=progress,
        refresh=refresh,
        cache_dir=cache_dir,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in grans:
        frame = downsample(base, granularity_spec(name).freq)
        frame.to_parquet(parquet_path(rd, name), index=False)
        frames[name] = frame
    update_latest(rd)
    return rd, frames
