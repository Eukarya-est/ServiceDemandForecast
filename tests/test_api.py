"""Smoke tests for the FastAPI control plane (no heavy jobs triggered)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from asp_demand.api.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_runs_shape() -> None:
    body = client.get("/runs").json()
    assert "runs" in body and isinstance(body["runs"], list)
    assert "hourly" in body["granularities"] and "6h" in body["granularities"]


def test_unknown_granularity_is_422() -> None:
    res = client.post("/predict", json={"granularity": "bogus"})
    assert res.status_code == 422


def test_unknown_job_is_404() -> None:
    assert client.get("/jobs/deadbeef").status_code == 404


def test_root_redirects_to_ui() -> None:
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (307, 308)
    assert res.headers["location"] == "/ui/"
