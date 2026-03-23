"""SQLite persistence for runs, steps, and episodic memory."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class SQLiteStore:
    """Lightweight local persistence backend."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("SEA_DB_PATH", ".data/sea.db")
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    env_name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    num_steps INTEGER NOT NULL,
                    final_feedback_type TEXT,
                    final_feedback_value TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    observation TEXT NOT NULL,
                    action TEXT NOT NULL,
                    feedback_type TEXT,
                    feedback_value TEXT,
                    tool_calls_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    source_task TEXT NOT NULL,
                    content TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at
                ON runs (created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_memories_agent_id
                ON memories (agent_id, created_at DESC);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "embedding_json" not in columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN embedding_json TEXT NOT NULL DEFAULT '[]'"
                )

    def save_trajectory(self, *, agent: Any, env_name: str, trajectory: Any) -> str:
        run_id = str(uuid.uuid4())
        final_feedback_type = None
        final_feedback_value = None
        if trajectory.final_feedback is not None:
            final_feedback_type = trajectory.final_feedback.type.value
            final_feedback_value = self._json_dump(trajectory.final_feedback.value)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, agent_id, env_name, goal, model, system_prompt,
                    success, total_reward, num_steps, final_feedback_type,
                    final_feedback_value, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    trajectory.task_id,
                    agent.agent_id,
                    env_name,
                    trajectory.goal,
                    agent.model,
                    agent.state.system_prompt,
                    int(trajectory.success),
                    float(trajectory.total_reward),
                    len(trajectory.steps),
                    final_feedback_type,
                    final_feedback_value,
                    self._json_dump(trajectory.metadata),
                    time.time(),
                ),
            )

            for step in trajectory.steps:
                feedback_type = None
                feedback_value = None
                if step.feedback is not None:
                    feedback_type = step.feedback.type.value
                    feedback_value = self._json_dump(step.feedback.value)

                conn.execute(
                    """
                    INSERT INTO steps (
                        run_id, step_index, observation, action,
                        feedback_type, feedback_value, tool_calls_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        step.step_index,
                        step.observation,
                        step.action,
                        feedback_type,
                        feedback_value,
                        self._json_dump(step.tool_calls),
                    ),
                )

        return run_id

    def save_memory_entries(self, agent_id: str, entries: list[Any]) -> None:
        if not entries:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memories (
                    agent_id, source_task, content, success,
                    importance, access_count, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        agent_id,
                        entry.source_task,
                        entry.content,
                        int(entry.success),
                        float(entry.importance),
                        int(entry.access_count),
                        self._json_dump(getattr(entry, "embedding", [])),
                        time.time(),
                    )
                    for entry in entries
                ],
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, task_id, agent_id, env_name, goal, model, success,
                       total_reward, num_steps, created_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                return None

            step_rows = conn.execute(
                """
                SELECT step_index, observation, action, feedback_type,
                       feedback_value, tool_calls_json
                FROM steps
                WHERE run_id = ?
                ORDER BY step_index ASC
                """,
                (run_id,),
            ).fetchall()

        run = dict(run_row)
        run["success"] = bool(run["success"])
        run["metadata"] = self._json_load(run.pop("metadata_json"))
        run["final_feedback_value"] = self._json_load(run["final_feedback_value"])
        run["steps"] = []
        for row in step_rows:
            item = dict(row)
            item["feedback_value"] = self._json_load(item["feedback_value"])
            item["tool_calls"] = self._json_load(item.pop("tool_calls_json"))
            run["steps"].append(item)
        return run

    def list_memory(self, agent_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_task, content, success, importance, access_count, embedding_json
                FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (agent_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["success"] = bool(item["success"])
            item["embedding"] = self._json_load(item.pop("embedding_json"))
            result.append(item)
        return result

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value)

    @staticmethod
    def _json_load(value: str | None) -> Any:
        if value is None:
            return None
        return json.loads(value)
