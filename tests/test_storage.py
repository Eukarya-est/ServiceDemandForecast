"""Unit tests for the local/S3 storage abstraction."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

from asp_demand.preprocessing.aggregate import iter_log_files
from asp_demand.storage import build_storage_options, get_fs


def test_storage_options_only_for_s3_with_endpoint() -> None:
    endpoint = "https://minio.local:9000"
    assert build_storage_options("s3://b/p", endpoint) == {
        "client_kwargs": {"endpoint_url": endpoint}
    }
    # No endpoint -> no options (standard AWS S3).
    assert build_storage_options("s3://b/p", None) == {}
    # Local paths never get S3 options even if an endpoint is set.
    assert build_storage_options("data/aplb-access-log", endpoint) == {}


def test_iter_log_files_globs_local_partitions(tmp_path: Path) -> None:
    day_dir = tmp_path / "2025" / "05" / "06"
    day_dir.mkdir(parents=True)
    for name in ("a.log.gz", "b.log.gz"):
        with gzip.open(day_dir / name, "wt", encoding="utf-8") as fh:
            fh.write("x\n")
    (day_dir / "ignore.txt").write_text("nope")

    fs, base = get_fs(str(tmp_path))
    found = list(iter_log_files(fs, base, date(2025, 5, 6), date(2025, 5, 6)))
    assert len(found) == 2
    assert all(p.endswith(".log.gz") for p in found)
