"""Tests for dashboard data loaders."""

import json

from self_evolving.dashboard.data import (
    load_agent_memory,
    load_benchmark_sessions,
    load_benchmark_variant,
    load_recent_runs,
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
