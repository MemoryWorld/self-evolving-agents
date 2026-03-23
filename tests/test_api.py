"""API tests."""

import time

from fastapi.testclient import TestClient

from self_evolving.core.agent import BaseAgent
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.service.api import create_app


def _wait_for_job_completion(client: TestClient, job_id: str, timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not complete in time")


def test_run_qa_and_fetch_persisted_data(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: Paris")

    app = create_app(str(tmp_path / "sea.db"))
    client = TestClient(app)

    response = client.post(
        "/runs/qa",
        json={
            "goal": "What is the capital of France?",
            "reference_answer": "Paris",
            "agent_id": "api-agent",
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    job = response.json()
    assert job["kind"] == "qa_run"
    assert job["job_id"]

    completed_job = _wait_for_job_completion(client, job["job_id"])
    payload = completed_job["result"]
    assert payload["task_id"]
    assert payload["agent_id"] == "api-agent"
    assert payload["run_id"]

    runs = client.get("/runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 1

    detail = client.get(f"/runs/{payload['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["run_id"] == payload["run_id"]


def test_run_benchmark_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SEA_BENCHMARK_DIR", str(tmp_path / "benchmarks"))

    def fake_call(self, messages):
        text = messages[-1]["content"].lower()
        if "capital of france" in text:
            return "ANSWER: Paris"
        if "12 * 7" in text:
            return "ANSWER: 84"
        return "ANSWER: unknown"

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call)
    monkeypatch.setattr(
        OPROOptimizer,
        "optimize",
        lambda self, initial_prompt, eval_fn, task_description="": initial_prompt + " optimized",
    )

    app = create_app(str(tmp_path / "sea.db"))
    client = TestClient(app)

    response = client.post(
        "/benchmarks/qa",
        json={
            "tasks": [
                {"goal": "What is the capital of France?", "reference_answer": "Paris"},
                {"goal": "What is 12 * 7?", "reference_answer": "84"},
            ],
            "variants": ["baseline", "memory"],
        },
    )

    assert response.status_code == 200
    job = response.json()
    assert job["kind"] == "qa_benchmark"

    completed_job = _wait_for_job_completion(client, job["job_id"])
    payload = completed_job["result"]
    assert payload["task_count"] == 2
    assert "baseline" in payload["variants"]
    assert "memory" in payload["variants"]

    jobs = client.get("/jobs")
    assert jobs.status_code == 200
    assert len(jobs.json()) >= 1
