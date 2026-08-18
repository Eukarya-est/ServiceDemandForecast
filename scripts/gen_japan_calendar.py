"""Generate the committed Japan calendar CSV from the ``jpholiday`` package.

Thin wrapper around :func:`asp_demand.features.calendar.generate_calendar`. The Hydra
pipeline calls the same function, so this script and the pipeline stay in sync.

Usage:
    python scripts/gen_japan_calendar.py [start_year] [end_year]
"""

from __future__ import annotations

import sys

from asp_demand.config import CALENDAR_CSV
from asp_demand.features.calendar import generate_calendar


def main() -> None:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    rows = generate_calendar(start_year, end_year)
    print(f"Wrote {rows} rows to {CALENDAR_CSV}")


if __name__ == "__main__":
    main()
