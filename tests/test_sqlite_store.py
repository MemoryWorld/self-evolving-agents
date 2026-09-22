"""Tests for SQLite persistence."""

from self_evolving.core.types import Feedback, FeedbackType, Step, Trajectory
from self_evolving.persistence.sqlite_store import SQLiteStore


class DummyState:
    system_prompt = "test system prompt"


class DummyAgent:
    agent_id = "agent-test"
    model = "dummy-model"
    state = DummyState()


def test_save_and_get_run(tmp_path):
    store = SQLiteStore(str(tmp_path / "sea.db"))
    trajectory = Trajectory(task_id="task-1", goal="What is 2+2?")
    trajectory.steps.append(
        Step(
            observation="Question: What is 2+2?",
            action="ANSWER: 4",
            feedback=Feedback(type=FeedbackType.BINARY, value=True),
            step_index=0,
        )
    )
    trajectory.final_feedback = Feedback(type=FeedbackType.BINARY, value=True)

    run_id = store.save_trajectory(agent=DummyAgent(), env_name="simple_qa", trajectory=trajectory)
    data = store.get_run(run_id)

    assert data is not None
    assert data["run_id"] == run_id
    assert data["task_id"] == "task-1"
    assert data["success"] is True
    assert len(data["steps"]) == 1
    assert data["steps"][0]["action"] == "ANSWER: 4"


def test_save_and_list_memory(tmp_path):
    store = SQLiteStore(str(tmp_path / "sea.db"))

    class Entry:
        source_task = "task-1"
        content = "Always answer with the known fact."
        success = True
        importance = 1.5
        access_count = 0
        embedding = [0.1, 0.2, 0.3]

    store.save_memory_entries("agent-test", [Entry()])
    memories = store.list_memory("agent-test")

    assert len(memories) == 1
    assert memories[0]["content"] == "Always answer with the known fact."
    assert memories[0]["success"] is True
    assert memories[0]["embedding"] == [0.1, 0.2, 0.3]


def test_legacy_memory_schema_migrates_without_losing_entries(tmp_path):
    import sqlite3
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "agent_id TEXT, source_task TEXT, content TEXT, success INTEGER, "
                 "importance REAL, access_count INTEGER, created_at REAL)")
    conn.execute("INSERT INTO memories (agent_id, source_task, content, success, importance, "
                 "access_count, created_at) VALUES ('a', 'old', 'kept', 1, 1.5, 2, 1)")
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    snapshot = store.load_memory_snapshot("a")
    assert snapshot["entries"][0]["content"] == "kept"
    assert snapshot["entries"][0]["embedding"] == []
    assert snapshot["stored_count"] == 0  # old schema did not record this counter
    store.save_memory_snapshot("a", snapshot["entries"], stored_count=1, expected_revision=0)
    assert store.list_memory("a")[0]["access_count"] == 2


def test_connections_close_after_success_and_error(tmp_path):
    import sqlite3
    import pytest
    store = SQLiteStore(str(tmp_path / "connections.db"))
    with store._connect() as connection:
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with store._connect() as failed_connection:
            raise RuntimeError("simulate failure")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        failed_connection.execute("SELECT 1")
