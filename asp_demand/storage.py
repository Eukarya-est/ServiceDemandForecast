"""Filesystem abstraction over local paths and S3, via fsspec/s3fs.

Lets the pipeline read raw ALB logs straight from ``s3://bucket/prefix`` on-premise
(no local copy) using the same code path as a local directory. Credentials and region
are picked up from the standard ``AWS_*`` environment variables by boto3/s3fs; an
optional custom endpoint (on-prem S3-compatible store / VPC endpoint) is injected here.
"""

from __future__ import annotations

from typing import Any

import fsspec

from asp_demand import config


def build_storage_options(uri: str, endpoint_url: str | None) -> dict[str, Any]:
    """Return fsspec storage options for ``uri`` (pure; easy to unit-test)."""
    if uri.startswith("s3://") and endpoint_url:
        return {"client_kwargs": {"endpoint_url": endpoint_url}}
    return {}


def storage_options(uri: str) -> dict[str, Any]:
    """Storage options for ``uri`` using the configured S3 endpoint, if any."""
    return build_storage_options(uri, config.S3_ENDPOINT_URL)


def get_fs(uri: str) -> tuple[Any, str]:
    """Resolve ``uri`` to an ``(fsspec_filesystem, base_path)`` pair.

    ``base_path`` is the scheme-stripped path (e.g. ``bucket/prefix`` for S3),
    which the returned filesystem accepts directly in ``glob``/``open``.
    """
    fs, base_path = fsspec.core.url_to_fs(uri, **storage_options(uri))
    return fs, base_path
