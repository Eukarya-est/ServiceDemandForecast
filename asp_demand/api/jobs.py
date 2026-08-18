"""Tiny in-process job manager for long-running API tasks (preprocess, train).

Jobs run on a background thread pool; the HTTP handler returns a ``job_id`` immediately
and the client polls ``/jobs/{id}``. State lives in memory (single uvicorn process) —
fine for an internal control panel; swap for a real queue if you scale out.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from threading import Event, Lock
from typing import Any

# A task reports coarse progress as (done, total) item counts.
Reporter = Callable[[int, int], None]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class JobContext:
    """Handed to a task: report progress and observe cancellation cooperatively."""

    report: Reporter
    is_cancelled: Callable[[], bool]

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise InterruptedError("job cancelled")


@dataclass
class Job:
    """A unit of background work and its lifecycle state."""

    id: str
    kind: str
    status: str = "pending"  # pending | running | succeeded | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, int] | None = None  # {"done": n, "total": m}
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    """Submit callables that return a JSON-able dict; track status/result/error."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._events: dict[str, Event] = {}
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")

    def submit(self, kind: str, fn: Callable[[JobContext], dict[str, Any]]) -> Job:
        """Run ``fn(ctx)`` in the background. ``ctx.report(done, total)`` updates progress;
        ``ctx.is_cancelled()`` / ``ctx.raise_if_cancelled()`` let cooperative tasks stop."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, created_at=_now(), updated_at=_now())
        with self._lock:
            self._jobs[job.id] = job
            self._events[job.id] = Event()
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[JobContext], dict[str, Any]]) -> None:
        self._update(job, status="running")
        event = self._events[job.id]
        ctx = JobContext(
            report=lambda done, total: self._update(job, progress={"done": done, "total": total}),
            is_cancelled=event.is_set,
        )
        try:
            result = fn(ctx)
        except Exception as exc:
            # A failure after cancel was requested is reported as cancelled, not failed.
            if event.is_set():
                self._update(job, status="cancelled")
            else:
                self._update(job, status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            self._update(job, status="succeeded", result=result)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation; returns False if the job is unknown or already finished."""
        with self._lock:
            job = self._jobs.get(job_id)
            event = self._events.get(job_id)
            if job is None or event is None or job.status in ("succeeded", "failed", "cancelled"):
                return False
        event.set()
        return True

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
