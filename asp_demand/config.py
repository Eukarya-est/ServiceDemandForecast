"""Central paths and constants for the ASP demand pipeline.

Outputs are grouped per **run** in a timestamped directory
``data/runs/<yymmddhhmmss>/`` holding ``hourly.parquet`` / ``daily.parquet`` (from
preprocess), the trained models, and ``forecast_{hourly,daily}.csv`` (from predict).
``preprocess`` creates a fresh run dir and points ``data/runs/latest`` at it; ``train``
and ``predict`` operate on a given run dir (default: the latest).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env (if present) without overriding real environment variables — so an
# EC2 instance role / shell exports always win over a committed-by-mistake file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
RAW_LOG_DIR = DATA_DIR / "aplb-access-log"
CALENDAR_CSV = DATA_DIR / "calendar" / "japan_calendar.csv"
CACHE_DIR = DATA_DIR / "cache"  # shared per-day aggregation cache (reused across runs)
RUNS_DIR = DATA_DIR / "runs"  # timestamped run outputs
LATEST_LINK = RUNS_DIR / "latest"

# Raw ALB logs may live locally or in S3 (read as-is, no copy). A local default
# keeps the bundled sample working out of the box; set ASP_RAW_LOG_URI to e.g.
# s3://my-bucket/aplb-access-log to stream from S3.
RAW_LOG_URI = os.getenv("ASP_RAW_LOG_URI", str(RAW_LOG_DIR))

# Optional custom S3 endpoint (on-prem S3-compatible store or VPC endpoint).
# Credentials/region come from the standard AWS_* env vars via boto3/s3fs.
S3_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL") or None

UTC = ZoneInfo("UTC")
JST = ZoneInfo("Asia/Tokyo")

# Timezone used to bucket demand. JST suits Japan demand modelling (holiday/weekday
# features); set ASP_TZ=UTC to match the AWS Athena aggregates exactly.
DEFAULT_TZ = os.getenv("ASP_TZ", "Asia/Tokyo")

# Only one ELB exists in the sample data; used as a default for serving.
DEFAULT_ELB = "app/example-alb/0123456789abcdef"

# Forecast quantiles (label -> alpha), ordered low->high. P50 is the point forecast.
QUANTILES: dict[str, float] = {"p50": 0.5, "p90": 0.9, "p95": 0.95}


@dataclass(frozen=True)
class GranularitySpec:
    """A demand bucketing resolution and its forecasting feature scheme."""

    name: str
    freq: str  # pandas offset alias for flooring/downsampling and the forecast step
    periods_per_day: int  # buckets per day (drives intra-day + lag features)
    lags: tuple[int, ...]  # autoregressive lags (in buckets)
    rolling: tuple[int, ...]  # rolling-mean windows (in buckets)


# All granularities derive from the hourly base by downsampling (so the raw-log read
# and cache are shared). Sub-daily lags follow [1, 1-day, 1-week]; daily keeps its
# established [1d, 1w, 2w] scheme.
_GRANULARITIES = {
    spec.name: spec
    for spec in (
        GranularitySpec("hourly", "1h", 24, (1, 24, 168), (24, 168)),
        GranularitySpec("3h", "3h", 8, (1, 8, 56), (8, 56)),
        GranularitySpec("6h", "6h", 4, (1, 4, 28), (4, 28)),
        GranularitySpec("12h", "12h", 2, (1, 2, 14), (2, 14)),
        GranularitySpec("daily", "1D", 1, (1, 7, 14), (7,)),
    )
}
GRANULARITIES = tuple(_GRANULARITIES)  # names, for CLI choices / iteration


def granularity_spec(name: str) -> GranularitySpec:
    """Look up a :class:`GranularitySpec` by name (raises on unknown)."""
    try:
        return _GRANULARITIES[name]
    except KeyError:
        raise ValueError(f"unknown granularity {name!r}; choose from {GRANULARITIES}") from None


RUN_TIMESTAMP_FMT = "%y%m%d%H%M%S"


def make_run_dir(base: Path = RUNS_DIR) -> Path:
    """Create and return a fresh timestamped run directory (``yymmddhhmmss``)."""
    run_dir = base / datetime.now().strftime(RUN_TIMESTAMP_FMT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest(run_dir: Path, link: Path | None = None) -> None:
    """Point a ``latest`` symlink at ``run_dir`` using a relative target.

    The link defaults to ``<run_dir>/../latest`` (a sibling of the run), so it stays
    next to its runs and never clobbers the global link from an out-of-tree run dir.
    Relative target so it still resolves if the data dir is mounted at a different
    absolute path (e.g. host vs container).
    """
    if link is None:
        link = run_dir.parent / "latest"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    target = os.path.relpath(run_dir.resolve(), link.parent.resolve())
    link.symlink_to(target, target_is_directory=True)


def resolve_run_dir(run_dir: str | Path | None) -> Path:
    """Resolve a run reference to a directory.

    ``None`` -> the ``latest`` run. A path that exists (e.g. ``data/runs/<ts>`` from the
    CLI) is used as-is. A bare run name (e.g. ``<ts>`` from the API/UI dropdown) is looked
    up under ``RUNS_DIR``. Raises only when ``None`` and there is no ``latest``.
    """
    if run_dir is None:
        if LATEST_LINK.exists():
            return LATEST_LINK.resolve()
        raise ValueError("no run dir given and no latest run found; run preprocess first")
    given = Path(run_dir)
    if given.exists():
        return given
    return RUNS_DIR / run_dir  # treat as a run name under data/runs/


def parquet_path(run_dir: Path, granularity: str) -> Path:
    """Aggregated counts parquet for a granularity within a run."""
    return run_dir / f"{granularity}.parquet"


def model_path(run_dir: Path, granularity: str, quantile: str = "p50") -> Path:
    """Saved LightGBM booster for a granularity + quantile within a run."""
    return run_dir / f"{granularity}_{quantile}.txt"


def features_path(run_dir: Path, granularity: str) -> Path:
    """Saved feature-name list for a granularity within a run."""
    return run_dir / f"{granularity}_features.json"


def metrics_path(run_dir: Path, granularity: str) -> Path:
    """Saved training metrics for a granularity within a run."""
    return run_dir / f"metrics_{granularity}.json"


def forecast_path(run_dir: Path, granularity: str) -> Path:
    """Forecast CSV for a granularity within a run."""
    return run_dir / f"forecast_{granularity}.csv"


def backtest_path(run_dir: Path, granularity: str) -> Path:
    """Backtest CSV (predicted vs actual over history) for a granularity within a run."""
    return run_dir / f"backtest_{granularity}.csv"
