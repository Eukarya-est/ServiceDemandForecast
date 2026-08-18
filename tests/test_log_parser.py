"""Unit tests for ALB log line parsing."""

from __future__ import annotations

from datetime import timedelta

from asp_demand.preprocessing.log_parser import parse_line

SAMPLE = (
    'https 2025-05-06T22:45:00.991274Z app/example-alb/0123456789abcdef '
    '203.0.113.10:3927 192.0.2.20:1080 0.001 0.261 0.000 200 200 694 505323 '
    '"POST https://example.com:443/x HTTP/1.1" "-" rest of line ignored'
)


def test_parse_line_extracts_timestamp_and_elb() -> None:
    result = parse_line(SAMPLE)
    assert result is not None
    ts, elb = result
    assert elb == "app/example-alb/0123456789abcdef"
    assert ts.utcoffset() == timedelta(0)
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2025, 5, 6, 22, 45)


def test_parse_line_rejects_short_lines() -> None:
    assert parse_line("garbage") is None
    assert parse_line("") is None


def test_parse_line_rejects_bad_timestamp() -> None:
    assert parse_line("https not-a-timestamp app/example-alb/abc rest") is None
