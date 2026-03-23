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
