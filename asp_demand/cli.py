"""Command-line interface: preprocess, train, predict."""

from __future__ import annotations

import argparse
from datetime import date, datetime

from asp_demand.config import GRANULARITIES, resolve_run_dir


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _cmd_preprocess(args: argparse.Namespace) -> None:
    from asp_demand.preprocessing.aggregate import run

    run_dir, frames = run(
        args.start,
        args.end,
        root=args.raw_log_uri,
        tz=args.tz,
        run_dir=args.run_dir,
        workers=args.workers,
        progress=True,
        refresh=args.refresh,
    )
    for name, frame in frames.items():
        total = int(frame["request_count"].sum())
        print(f"{name:>6}: {len(frame):>5} rows -> request_count total {total}")
    print(f"run dir -> {run_dir.resolve()}")


def _cmd_plot(args: argparse.Namespace) -> None:
    from asp_demand.viz import plot_forecast

    run_dir = resolve_run_dir(args.run_dir)
    path = plot_forecast(
        run_dir, args.granularity, out=args.out, height=args.height,
        kind=args.kind, max_points=args.max_points,
    )
    print(f"plot saved -> {path.resolve()}")


def _cmd_cache_clean(args: argparse.Namespace) -> None:
    from asp_demand.config import CACHE_DIR
    from asp_demand.preprocessing.aggregate import clean_cache

    count, nbytes = clean_cache(tz=args.tz, orphaned=args.orphaned, dry_run=args.dry_run)
    scope = "orphaned" if args.orphaned else (f"tz={args.tz}" if args.tz else "all")
    verb = "would remove" if args.dry_run else "removed"
    print(f"cache ({scope}) at {CACHE_DIR}: {verb} {count} file(s), {nbytes / 1024 / 1024:.1f} MiB")


def _cmd_train(args: argparse.Namespace) -> None:
    from asp_demand.model.train import train_model

    run_dir = resolve_run_dir(args.run_dir)
    metrics = train_model(args.granularity, run_dir)
    print(f"{args.granularity} model trained in {run_dir}. validation metrics: {metrics}")


def _cmd_predict(args: argparse.Namespace) -> None:
    from asp_demand.model.predict import forecast, forecast_window, write_forecast

    run_dir = resolve_run_dir(args.run_dir)
    if args.from_date is not None:
        if args.horizon is not None:
            raise ValueError("use either --horizon or --from/--to, not both")
        end = args.to_date or args.from_date
        df, gap = forecast_window(args.granularity, run_dir, args.from_date, end)
        if gap > 0:
            print(
                f"note: {gap} step(s) of recursive lead-in before the window "
                "(history ends earlier than the window; accuracy degrades with distance)"
            )
    else:
        df = forecast(args.granularity, run_dir, args.horizon or 24)

    print(df.assign(bucket_start=df["bucket_start"].astype(str)).to_string(index=False))
    if not args.no_save:
        path = write_forecast(df, run_dir, args.granularity)
        print(f"forecast saved -> {path.resolve()}")


def _cmd_backtest(args: argparse.Namespace) -> None:
    from asp_demand.model.predict import backtest, score, write_backtest

    run_dir = resolve_run_dir(args.run_dir)
    end = args.to_date or args.from_date
    df = backtest(args.granularity, run_dir, args.from_date, end)
    print(df.assign(bucket_start=df["bucket_start"].astype(str)).to_string(index=False))
    print(f"P50 vs actual: {score(df['actual'].to_numpy(), df['p50'].to_numpy())}")
    if not args.no_save:
        path = write_backtest(df, run_dir, args.granularity)
        print(f"backtest saved -> {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asp-demand", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preprocess", help="aggregate raw ALB logs into hourly/daily counts")
    pre.add_argument(
        "--start", type=_parse_date, required=True, help="UTC folder start date YYYY-MM-DD"
    )
    pre.add_argument(
        "--end", type=_parse_date, required=True, help="UTC folder end date YYYY-MM-DD"
    )
    pre.add_argument(
        "--raw-log-uri",
        default=None,
        help="local path or s3://bucket/prefix (default: $ASP_RAW_LOG_URI)",
    )
    pre.add_argument(
        "--tz",
        default=None,
        help="bucketing timezone, e.g. Asia/Tokyo or UTC (default: $ASP_TZ -> JST)",
    )
    pre.add_argument(
        "--workers", type=int, default=None, help="parallel file readers (default: auto)"
    )
    pre.add_argument(
        "--refresh", action="store_true", help="ignore the per-day cache and re-read"
    )
    pre.add_argument(
        "--run-dir", default=None, help="output run dir (default: new data/runs/<yymmddhhmmss>)"
    )
    pre.set_defaults(func=_cmd_preprocess)

    train = sub.add_parser("train", help="train a LightGBM model")
    train.add_argument("--granularity", choices=GRANULARITIES, required=True)
    train.add_argument(
        "--run-dir", default=None, help="run dir with the parquet (default: latest)"
    )
    train.set_defaults(func=_cmd_train)

    pred = sub.add_parser("predict", help="forecast future demand")
    pred.add_argument("--granularity", choices=GRANULARITIES, required=True)
    pred.add_argument(
        "--horizon", type=int, default=None, help="number of future buckets (default: 24)"
    )
    pred.add_argument(
        "--from", dest="from_date", type=_parse_date, default=None,
        help="forecast a date window instead of a horizon: window start YYYY-MM-DD",
    )
    pred.add_argument(
        "--to", dest="to_date", type=_parse_date, default=None,
        help="window end YYYY-MM-DD (default: same as --from)",
    )
    pred.add_argument(
        "--run-dir", default=None, help="run dir with the model (default: latest)"
    )
    pred.add_argument(
        "--no-save", action="store_true", help="print only; do not write the forecast CSV"
    )
    pred.set_defaults(func=_cmd_predict)

    bt = sub.add_parser("backtest", help="predict over a historical window and compare to actuals")
    bt.add_argument("--granularity", choices=GRANULARITIES, required=True)
    bt.add_argument("--from", dest="from_date", type=_parse_date, required=True, help="YYYY-MM-DD")
    bt.add_argument(
        "--to", dest="to_date", type=_parse_date, default=None, help="YYYY-MM-DD (default: --from)"
    )
    bt.add_argument("--run-dir", default=None, help="run dir with the model (default: latest)")
    bt.add_argument("--no-save", action="store_true", help="print only; do not write the CSV")
    bt.set_defaults(func=_cmd_backtest)

    pl = sub.add_parser("plot", help="render an interactive forecast/backtest chart (Plotly HTML)")
    pl.add_argument("--granularity", choices=GRANULARITIES, required=True)
    pl.add_argument(
        "--kind", choices=("forecast", "backtest"), default="forecast", help="which CSV to chart"
    )
    pl.add_argument(
        "--run-dir", default=None, help="run dir with the forecast (default: latest)"
    )
    pl.add_argument(
        "--out", default=None, help="output HTML path (default: <run>/<kind>_<g>.html)"
    )
    pl.add_argument(
        "--height", type=int, default=900, help="chart height in px (taller = page scrolls)"
    )
    pl.add_argument(
        "--max-points", type=int, default=4000,
        help="decimate the actual trace to this many points (rendering speed)",
    )
    pl.set_defaults(func=_cmd_plot)

    cc = sub.add_parser("cache-clean", help="remove cached aggregation parquet")
    cc.add_argument("--tz", default=None, help="only this timezone's cache (default: all)")
    cc.add_argument(
        "--orphaned",
        action="store_true",
        help="only legacy flat files in the cache root (keeps per-tz caches)",
    )
    cc.add_argument("--dry-run", action="store_true", help="list what would be removed")
    cc.set_defaults(func=_cmd_cache_clean)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()
