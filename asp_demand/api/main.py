"""FastAPI control plane: one endpoint per pipeline task + a static vanilla-JS UI.

Each endpoint is a thin wrapper over the same functions the CLI uses. Fast tasks
(calendar, predict, visualize, runs) are synchronous; long tasks (preprocess, train)
return a ``job_id`` and run on a background thread — poll ``/jobs/{id}``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from asp_demand.api.jobs import JobContext, JobManager
from asp_demand.config import (
    CALENDAR_CSV,
    GRANULARITIES,
    RUNS_DIR,
    forecast_path,
    granularity_spec,
    model_path,
    parquet_path,
    resolve_run_dir,
)

app = FastAPI(title="ASP Demand Prediction API", version="0.2.0")
jobs = JobManager()


# --------------------------------------------------------------------------- models
class CalendarRequest(BaseModel):
    start_year: int = 2023
    end_year: int = 2026


class PreprocessRequest(BaseModel):
    start: date
    end: date
    tz: str | None = None
    raw_log_uri: str | None = None
    workers: int | None = None
    refresh: bool = False


class TrainRequest(BaseModel):
    granularity: str
    run_dir: str | None = None


class PredictRequest(BaseModel):
    granularity: str
    run_dir: str | None = None
    horizon: int | None = None
    from_date: date | None = None
    to_date: date | None = None


class BacktestRequest(BaseModel):
    granularity: str
    run_dir: str | None = None
    from_date: date
    to_date: date | None = None


class JobRef(BaseModel):
    job_id: str


def _rows(df: Any) -> list[dict[str, Any]]:
    """Serialize a forecast/backtest frame to JSON-able rows (str time, float values)."""
    out: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row = {k: (str(v) if k == "bucket_start" else float(v)) for k, v in record.items()}
        out.append(row)
    return out


def _check_granularity(name: str) -> None:
    if name not in GRANULARITIES:
        raise HTTPException(422, f"unknown granularity {name!r}; choose from {list(GRANULARITIES)}")


def _resolve(run_dir: str | None) -> Path:
    try:
        return resolve_run_dir(run_dir)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ----------------------------------------------------------------------- task endpoints
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/calendar")
def calendar(req: CalendarRequest) -> dict[str, Any]:
    """1. GENERATE-DATE-INFO — build the Japan calendar CSV (jpholiday)."""
    from asp_demand.features.calendar import generate_calendar

    rows = generate_calendar(req.start_year, req.end_year)
    return {"rows": rows, "path": str(CALENDAR_CSV)}


@app.post("/preprocess", response_model=JobRef)
def preprocess(req: PreprocessRequest) -> JobRef:
    """2. PREPROCESS — aggregate logs into a new run dir (async job)."""
    from asp_demand.preprocessing.aggregate import run as preprocess_run

    def task(ctx: JobContext) -> dict[str, Any]:
        run_dir, frames = preprocess_run(
            req.start, req.end, root=req.raw_log_uri, tz=req.tz,
            workers=req.workers, refresh=req.refresh,
            progress_cb=ctx.report, cancel_check=ctx.is_cancelled,
        )
        return {"run_dir": run_dir.name, "rows": {g: len(f) for g, f in frames.items()}}

    return JobRef(job_id=jobs.submit("preprocess", task).id)


@app.post("/train", response_model=JobRef)
def train(req: TrainRequest) -> JobRef:
    """3. TRAIN — fit a LightGBM model in a run dir (async job)."""
    _check_granularity(req.granularity)
    from asp_demand.model.train import train_model

    run_dir = _resolve(req.run_dir)

    def task(ctx: JobContext) -> dict[str, Any]:
        metrics = train_model(req.granularity, run_dir, progress_cb=ctx.report)
        return {"run_dir": run_dir.name, "granularity": req.granularity, "metrics": metrics}

    return JobRef(job_id=jobs.submit(f"train:{req.granularity}", task).id)


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """4. PREDICT — forecast a horizon or a date window (sync)."""
    _check_granularity(req.granularity)
    from asp_demand.model.predict import forecast, forecast_window, write_forecast

    run_dir = _resolve(req.run_dir)
    if not model_path(run_dir, req.granularity).exists():
        raise HTTPException(
            409, f"{req.granularity} model not trained in {run_dir.name}; train first"
        )
    try:
        if req.from_date is not None:
            df, gap = forecast_window(
                req.granularity, run_dir, req.from_date, req.to_date or req.from_date
            )
        else:
            df, gap = forecast(req.granularity, run_dir, req.horizon or 24), 0
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc

    path = write_forecast(df, run_dir, req.granularity)
    return {
        "granularity": req.granularity,
        "run_dir": run_dir.name,
        "step": granularity_spec(req.granularity).freq,
        "gap_steps": gap,
        "saved": str(path),
        "forecast": _rows(df),
    }


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    """Predict over a historical window and compare to actuals (sync)."""
    _check_granularity(req.granularity)
    from asp_demand.model.predict import backtest as run_backtest
    from asp_demand.model.predict import score, write_backtest

    run_dir = _resolve(req.run_dir)
    if not model_path(run_dir, req.granularity).exists():
        raise HTTPException(
            409, f"{req.granularity} model not trained in {run_dir.name}; train first"
        )
    try:
        df = run_backtest(req.granularity, run_dir, req.from_date, req.to_date or req.from_date)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc

    path = write_backtest(df, run_dir, req.granularity)
    return {
        "granularity": req.granularity,
        "run_dir": run_dir.name,
        "metrics": score(df["actual"].to_numpy(), df["p50"].to_numpy()),
        "saved": str(path),
        "backtest": _rows(df),
    }


@app.get("/visualize")
def visualize(
    granularity: str, run_dir: str | None = None, height: int = 900,
    kind: str = "forecast", max_points: int = 4000,
) -> FileResponse:
    """5. DATA-VISUALIZATION — render the interactive Plotly chart and return its HTML."""
    _check_granularity(granularity)
    from asp_demand.viz import plot_forecast

    rd = _resolve(run_dir)
    try:
        path = plot_forecast(rd, granularity, height=height, kind=kind, max_points=max_points)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:  # unknown kind
        raise HTTPException(422, str(exc)) from exc
    return FileResponse(path, media_type="text/html")


# ------------------------------------------------------------------------ state endpoints
@app.get("/runs")
def list_runs() -> dict[str, Any]:
    """List run dirs and which artifacts each granularity has (drives the UI)."""
    runs: list[dict[str, Any]] = []
    if RUNS_DIR.exists():
        dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.is_symlink()]
        for d in sorted(dirs, key=lambda p: p.name, reverse=True):
            runs.append(
                {
                    "run": d.name,
                    "granularities": {
                        g: {
                            "parquet": parquet_path(d, g).exists(),
                            "model": model_path(d, g).exists(),
                            "forecast": forecast_path(d, g).exists(),
                        }
                        for g in GRANULARITIES
                    },
                }
            )
    latest = resolve_run_dir(None).name if (RUNS_DIR / "latest").exists() else None
    return {"latest": latest, "granularities": list(GRANULARITIES), "runs": runs}


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.as_dict() for j in jobs.list()]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job id")
    return job.as_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    if jobs.get(job_id) is None:
        raise HTTPException(404, "unknown job id")
    return {"job_id": job_id, "cancelled": jobs.cancel(job_id)}


# ------------------------------------------------------------------------------- static UI
@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
