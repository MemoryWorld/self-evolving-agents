"""API tests."""

import time

from fastapi.testclient import TestClient

from self_evolving.core.agent import BaseAgent
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.evolution.memory.episodic import EpisodicMemory
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
    monkeypatch.setattr(EpisodicMemory, "_distil", lambda self, trajectory: [trajectory.goal])
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
    assert payload["evaluation_protocol"]["prompt_optimization"] == "resubstitution"
    assert payload["data_source"] == "model_execution"

    jobs = client.get("/jobs")
    assert jobs.status_code == 200
    assert len(jobs.json()) >= 1


def test_api_rebuilds_agent_with_prior_memory_and_persisted_accesses(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: Paris")
    monkeypatch.setattr(EpisodicMemory, "_distil", lambda self, trajectory: ["France capital Paris"])
    path = str(tmp_path / "sea.db")
    # Recreate the app as well as the agent, keeping only the SQLite path and agent ID.
    for task_id in ["first", "second"]:
        with TestClient(create_app(path)) as client:
            response = client.post("/runs/qa", json={"goal": "France capital?",
                                   "reference_answer": "Paris", "agent_id": "stable",
                                   "task_id": task_id, "use_memory": True})
            job = _wait_for_job_completion(client, response.json()["job_id"])
            assert job["status"] == "completed", job.get("error")
            detail = client.get(f"/runs/{job['result']['run_id']}").json()
            if task_id == "second":
                assert "[Past experience 1]: France capital Paris" in detail["steps"][0]["observation"]
                entries = client.get("/agents/stable/memory").json()
                assert len(entries) == 2
                assert sum(item["access_count"] for item in entries) == 1


def test_api_same_agent_concurrent_run_reports_conflict(tmp_path, monkeypatch):
    from threading import Barrier
    barrier = Barrier(2)

    def answer(self, messages):
        barrier.wait(timeout=5)
        return "ANSWER: Paris"

    monkeypatch.setattr(BaseAgent, "_call_llm", answer)
    monkeypatch.setattr(EpisodicMemory, "_distil", lambda self, trajectory: [trajectory.task_id])
    with TestClient(create_app(str(tmp_path / "sea.db"))) as client:
        jobs = []
        for task_id in ["one", "two"]:
            response = client.post("/runs/qa", json={"goal": "France capital?",
                                   "reference_answer": "Paris", "agent_id": "shared",
                                   "task_id": task_id})
            jobs.append(response.json()["job_id"])
        results = [_wait_for_job_completion(client, job_id, timeout=8) for job_id in jobs]
        assert sorted(job["status"] for job in results) == ["completed", "failed"]
        failed = next(job for job in results if job["status"] == "failed")
        assert "reload and rerun" in failed["error"]
        entries = client.get("/agents/shared/memory").json()
        winner = next(job for job in results if job["status"] == "completed")
        assert [entry["content"] for entry in entries] == [winner["result"]["task_id"]]


def test_api_accepts_disjoint_tuning_tasks_and_keeps_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("SEA_BENCHMARK_DIR", str(tmp_path / "benchmarks"))
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: correct")
    monkeypatch.setattr(OPROOptimizer, "_propose", lambda self, description: "candidate")
    with TestClient(create_app(str(tmp_path / "sea.db"))) as client:
        response = client.post("/benchmarks/qa", json={
            "tasks": [{"goal": "evaluation question", "reference_answer": "correct"}],
            "tuning_tasks": [{"goal": "tuning question", "reference_answer": "correct"}],
            "variants": ["prompt_optimization"],
        })
        job = _wait_for_job_completion(client, response.json()["job_id"])
        assert job["status"] == "completed", job.get("error")
        assert job["result"]["evaluation_protocol"]["prompt_optimization"] == "heldout"
        assert job["result"]["variants"]["prompt_optimization"]["metadata"]["evaluation_protocol"]["tuning_task_count"] == 1
