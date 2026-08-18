"""Stream-parse AWS ALB access logs.

ALB log lines are space-delimited. For demand-by-count we only need:
  field 1 (index 1): ``time``  -> ISO-8601 UTC timestamp
  field 2 (index 2): ``elb``   -> load balancer resource id
Both are unquoted, so a simple split on the first few spaces is enough and we
avoid parsing the heavy quoted fields that follow.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from asp_demand.config import UTC


def parse_timestamp(raw: str) -> datetime:
    """Parse an ALB ISO-8601 UTC timestamp (e.g. ``2025-05-06T22:45:00.991274Z``)."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def parse_line(line: str) -> tuple[datetime, str] | None:
    """Return ``(utc_timestamp, elb)`` for a log line, or ``None`` if unparseable."""
    parts = line.split(" ", 3)
    if len(parts) < 3:
        return None
    try:
        ts = parse_timestamp(parts[1])
    except ValueError:
        return None
    return ts, parts[2]


def parse_file(fs: Any, path: str) -> Iterator[tuple[datetime, str]]:
    """Yield ``(utc_timestamp, elb)`` for each parseable line in a ``.gz`` log object.

    ``fs`` is an fsspec filesystem (local or S3) and ``path`` a path it understands;
    the gzip stream is decompressed on the fly so nothing is written to disk.
    """
    with fs.open(path, "rb") as raw, gzip.open(
        raw, "rt", encoding="utf-8", errors="replace"
    ) as fh:
        for line in fh:
            record = parse_line(line)
            if record is not None:
                yield record
