"""Tests for dashboard data loaders."""

import json

import httpx

from self_evolving.dashboard.data import (
    build_benchmark_comparison,
    load_agent_memory,
    load_benchmark_sessions,
    load_job,
    load_jobs,
    load_benchmark_variant,
    load_recent_runs,
    summarize_memory,
    trigger_benchmark,
    trigger_run,
)
from self_evolving.persistence.sqlite_store import SQLiteStore


def test_dashboard_data_loaders_for_sqlite(tmp_path):
    db_path = str(tmp_path / "sea.db")
    store = SQLiteStore(db_path)

    class Entry:
        source_task = "task-1"
        content = "Capital of France is Paris."
        success = True
        importance = 1.5
        access_count = 0
        embedding = [1.0, 0.0]

    store.save_memory_entries("agent-1", [Entry()])

    recent_runs = load_recent_runs(db_path)
    assert recent_runs == []

    memory = load_agent_memory(db_path, "agent-1")
    assert len(memory) == 1
    assert memory[0]["content"] == "Capital of France is Paris."

    summary = summarize_memory(memory)
    assert summary["total_entries"] == 1
    assert summary["avg_importance"] == 1.5
    assert summary["total_accesses"] == 0


def test_dashboard_data_loaders_for_benchmarks(tmp_path):
    session_dir = tmp_path / "benchmarks" / "20260101-000000"
    session_dir.mkdir(parents=True)

    summary = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "task_count": 2,
        "variants": {"baseline": {"success_rate": 1.0}},
    }
    variant = {"name": "baseline", "success_rate": 1.0}

    (session_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (session_dir / "baseline.json").write_text(json.dumps(variant), encoding="utf-8")

    sessions = load_benchmark_sessions(str(tmp_path / "benchmarks"))
    assert len(sessions) == 1
    assert sessions[0]["task_count"] == 2

    loaded_variant = load_benchmark_variant(str(session_dir), "baseline")
    assert loaded_variant["name"] == "baseline"

    comparison = build_benchmark_comparison(sessions[0])
    assert len(comparison) == 1
    assert comparison[0]["variant"] == "baseline"
    assert comparison[0]["success_rate"] == 1.0


def test_dashboard_trigger_helpers(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = []
    gets = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return DummyResponse({"ok": True, "url": url})

    def fake_get(url, params=None, timeout=0):
        gets.append((url, params, timeout))
        return DummyResponse({"ok": True, "url": url})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    run_payload = trigger_run("http://127.0.0.1:8000", {"goal": "x"})
    benchmark_payload = trigger_benchmark("http://127.0.0.1:8000", {"tasks": []})
    jobs_payload = load_jobs("http://127.0.0.1:8000")
    job_payload = load_job("http://127.0.0.1:8000", "job-1")

    assert run_payload["ok"] is True
    assert benchmark_payload["ok"] is True
    assert jobs_payload["ok"] is True
    assert job_payload["ok"] is True
    assert calls[0][0].endswith("/runs/qa")
    assert calls[1][0].endswith("/benchmarks/qa")
    assert gets[0][0].endswith("/jobs")
    assert gets[1][0].endswith("/jobs/job-1")
