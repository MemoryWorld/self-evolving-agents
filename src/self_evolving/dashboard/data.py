"""Data loaders for dashboard views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from self_evolving.persistence.sqlite_store import SQLiteStore


def load_recent_runs(db_path: str, limit: int = 50) -> list[dict[str, Any]]:
    store = SQLiteStore(db_path)
    return store.list_runs(limit=limit)


def load_run_detail(db_path: str, run_id: str) -> dict[str, Any] | None:
    store = SQLiteStore(db_path)
    return store.get_run(run_id)


def load_agent_memory(db_path: str, agent_id: str, limit: int = 100) -> list[dict[str, Any]]:
    store = SQLiteStore(db_path)
    return store.list_memory(agent_id, limit=limit)


def load_benchmark_sessions(root_dir: str) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        return []

    sessions = []
    for session_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        summary_path = session_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary["session_dir"] = str(session_dir)
        sessions.append(summary)
    return sessions


def load_benchmark_variant(session_dir: str, variant: str) -> dict[str, Any] | None:
    path = Path(session_dir) / f"{variant}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_benchmark_comparison(session: dict[str, Any]) -> list[dict[str, Any]]:
    variants = session.get("variants", {})
    rows = []
    for name, payload in variants.items():
        rows.append(
            {
                "variant": name,
                "success_rate": payload.get("success_rate", 0.0),
                "mean_reward": payload.get("mean_reward", 0.0),
                "mean_steps": payload.get("mean_steps", 0.0),
                "evolution_gain": payload.get("evolution_gain"),
                "stability": payload.get("stability", 0.0),
            }
        )
    return rows


def summarize_memory(memories: list[dict[str, Any]]) -> dict[str, Any]:
    if not memories:
        return {
            "total_entries": 0,
            "avg_importance": 0.0,
            "total_accesses": 0,
        }

    total_entries = len(memories)
    avg_importance = sum(memory["importance"] for memory in memories) / total_entries
    total_accesses = sum(memory["access_count"] for memory in memories)
    return {
        "total_entries": total_entries,
        "avg_importance": avg_importance,
        "total_accesses": total_accesses,
    }
