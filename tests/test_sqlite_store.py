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
