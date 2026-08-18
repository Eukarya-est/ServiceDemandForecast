"""Unit tests for the background job manager."""

from __future__ import annotations

import time
from typing import Any

from asp_demand.api.jobs import JobContext, JobManager


def _wait(manager: JobManager, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.status in ("succeeded", "failed", "cancelled"):
            return job.as_dict()
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_job_succeeds_and_stores_result() -> None:
    manager = JobManager()
    job = manager.submit("ok", lambda _ctx: {"value": 42})
    done = _wait(manager, job.id)
    assert done["status"] == "succeeded"
    assert done["result"] == {"value": 42}
    assert done["error"] is None


def test_job_failure_is_captured() -> None:
    manager = JobManager()

    def boom(_ctx: JobContext) -> dict[str, Any]:
        raise ValueError("nope")

    job = manager.submit("bad", boom)
    done = _wait(manager, job.id)
    assert done["status"] == "failed"
    assert "nope" in done["error"]


def test_progress_is_reported() -> None:
    manager = JobManager()

    def work(ctx: JobContext) -> dict[str, Any]:
        ctx.report(1, 3)
        ctx.report(3, 3)
        return {"ok": True}

    job = manager.submit("prog", work)
    done = _wait(manager, job.id)
    assert done["status"] == "succeeded"
    assert done["progress"] == {"done": 3, "total": 3}


def test_cancel_marks_job_cancelled() -> None:
    manager = JobManager()

    def work(ctx: JobContext) -> dict[str, Any]:
        # runs until cancelled (so the test can only pass via cancellation)
        while True:
            ctx.raise_if_cancelled()
            time.sleep(0.01)

    job = manager.submit("cancelme", work)
    time.sleep(0.05)  # let it start looping
    assert manager.cancel(job.id) is True
    done = _wait(manager, job.id, timeout=10.0)
    assert done["status"] == "cancelled"
    assert manager.cancel(job.id) is False  # already finished
    assert manager.cancel("nope") is False  # unknown


def test_list_and_unknown_lookup() -> None:
    manager = JobManager()
    job = manager.submit("ok", lambda _ctx: {"x": 1})
    _wait(manager, job.id)
    assert any(j.id == job.id for j in manager.list())
    assert manager.get("does-not-exist") is None
