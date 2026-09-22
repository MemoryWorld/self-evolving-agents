"""Cold start, checkpoint/reload, compaction and stale-writer regressions."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.core.types import AgentState, Feedback, FeedbackType, Trajectory
from self_evolving.evolution.memory.episodic import EpisodicMemory, MemoryEntry
from self_evolving.persistence.sqlite_store import SQLiteStore, MemoryConflictError


class ConstantEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


def make_agent(path, monkeypatch, *, agent_id="same-agent", max_entries=100, summarize_after=10):
    agent = BaseAgent(model="test", agent_id=agent_id)
    agent.memory = EpisodicMemory(embedder=ConstantEmbedder(), max_entries=max_entries,
                                  summarize_after=summarize_after)
    agent.store = SQLiteStore(str(path))
    monkeypatch.setattr(agent.memory, "_distil", lambda trajectory: [f"lesson {trajectory.task_id}"])
    monkeypatch.setattr(agent.memory, "_summarize", lambda combined: f"compressed {combined}")
    monkeypatch.setattr(agent, "_call_llm", lambda messages: "ANSWER: Paris")
    return agent


def run_task(agent, task_id):
    return agent.run(SimpleQAEnvironment([("France capital?", "Paris")]),
                     "France capital?", task_id=task_id)


def test_empty_memory_writes_then_new_agent_loads_and_injects(tmp_path, monkeypatch):
    path = tmp_path / "memory.db"
    first = make_agent(path, monkeypatch)
    assert not first.memory
    run_task(first, "first")
    assert len(first.memory) == 1
    second = make_agent(path, monkeypatch)
    trajectory = run_task(second, "second")
    assert "[Past experience 1]: lesson first" in trajectory.steps[0].observation
    snapshot = SQLiteStore(str(path)).load_memory_snapshot("same-agent")
    assert [entry["content"] for entry in snapshot["entries"]] == ["lesson first", "lesson second"]
    assert snapshot["entries"][0]["access_count"] == 1
    assert snapshot["stored_count"] == 2
    assert snapshot["revision"] == 2
    isolated = make_agent(path, monkeypatch, agent_id="different-agent")
    assert "Past experience" not in run_task(isolated, "third").steps[0].observation


def test_summary_cadence_and_contents_survive_new_agent_each_run(tmp_path, monkeypatch):
    path = tmp_path / "memory.db"
    # Eight entries ensure summarization replaces two entries, not just renames one.
    for index in range(8):
        agent = make_agent(path, monkeypatch, summarize_after=8)
        run_task(agent, str(index))
    snapshot = SQLiteStore(str(path)).load_memory_snapshot("same-agent")
    assert snapshot["stored_count"] == 8
    assert len(snapshot["entries"]) == 7
    assert snapshot["entries"][0]["content"] == "[Summary] compressed lesson 0 | lesson 1"
    assert all(item["content"] not in {"lesson 0", "lesson 1"} for item in snapshot["entries"])
    restored = make_agent(path, monkeypatch, summarize_after=8)
    restored.load_memory()
    assert restored.memory.dump() == snapshot["entries"]
    assert restored.memory.stored_count == 8


def test_trim_preserves_order_and_does_not_resurrect_evicted_rows(tmp_path, monkeypatch):
    path = tmp_path / "memory.db"
    agent = make_agent(path, monkeypatch, max_entries=2)
    agent.memory.load([
        MemoryEntry("low", "old", False, importance=0.1).__dict__,
        MemoryEntry("high", "middle", True, importance=2).__dict__,
    ])
    run_task(agent, "new")
    restored = make_agent(path, monkeypatch, max_entries=2)
    restored.load_memory()
    assert [item["content"] for item in restored.memory.dump()] == ["high", "lesson new"]
    restored.memory.load([], stored_count=3)
    restored.save_memory()
    assert restored.store.load_memory_snapshot("same-agent")["entries"] == []


def test_snapshot_rolls_back_failed_replacement(tmp_path):
    store = SQLiteStore(str(tmp_path / "memory.db"))
    valid = MemoryEntry("kept", "t1", True).__dict__
    store.save_memory_snapshot("a", [valid], stored_count=1, expected_revision=0)
    broken = {**valid, "content": None}
    with pytest.raises(Exception):
        store.save_memory_snapshot("a", [broken], stored_count=2, expected_revision=1)
    snapshot = store.load_memory_snapshot("a")
    assert snapshot["revision"] == 1
    assert snapshot["entries"][0]["content"] == "kept"


def test_concurrent_snapshots_reject_stale_writer_without_losing_winner(tmp_path):
    path = str(tmp_path / "memory.db")
    stores = [SQLiteStore(path), SQLiteStore(path)]
    barrier = Barrier(2)

    def write(index):
        revision = stores[index].load_memory_snapshot("a")["revision"]
        barrier.wait(timeout=5)
        try:
            stores[index].save_memory_snapshot("a", [MemoryEntry(str(index), "task", True).__dict__],
                                               stored_count=1, expected_revision=revision)
            return "saved", str(index)
        except MemoryConflictError:
            return "conflict", str(index)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, [0, 1]))
    assert sorted(result[0] for result in results) == ["conflict", "saved"]
    winner = next(value for status, value in results if status == "saved")
    snapshot = stores[0].load_memory_snapshot("a")
    assert snapshot["revision"] == 1
    assert snapshot["entries"][0]["content"] == winner


def test_legacy_append_invalidates_existing_snapshot_and_remains_readable(tmp_path):
    store = SQLiteStore(str(tmp_path / "memory.db"))
    store.save_memory_entries("a", [MemoryEntry("first", "task", True)])
    snapshot = store.load_memory_snapshot("a")
    assert snapshot["entries"][0]["content"] == "first"
    store.save_memory_entries("a", [MemoryEntry("second", "task", True)])
    with pytest.raises(MemoryConflictError):
        store.save_memory_snapshot("a", [], stored_count=0, expected_revision=snapshot["revision"])
    assert [item["content"] for item in store.list_memory("a")] == ["second", "first"]


def test_state_loads_into_empty_memory_and_dump_does_not_alias():
    agent = BaseAgent(model="test")
    agent.memory = EpisodicMemory(embedder=ConstantEmbedder())
    state = AgentState(agent_id="restored", system_prompt="prompt",
                       memory_entries=[MemoryEntry("kept", "task", True, embedding=[1, 0]).__dict__],
                       metadata={"memory_stored_count": 6})
    agent.load_state(state)
    assert agent.agent_id == "restored"
    assert agent.memory.stored_count == 6
    exported = agent.get_state()
    exported.memory_entries[0]["embedding"][0] = 999
    assert agent.memory.dump()[0]["embedding"] == [1, 0]
    agent.load_state(AgentState(agent_id="restored", system_prompt="empty"))
    assert len(agent.memory) == 0


def test_distillation_receives_failure_feedback_and_reflection(monkeypatch):
    from types import SimpleNamespace
    import litellm
    prompts = []

    def completion(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="LESSON: check units"))])

    monkeypatch.setattr(litellm, "completion", completion)
    memory = EpisodicMemory(embedder=ConstantEmbedder())
    trajectory = Trajectory(task_id="failed", goal="convert units",
                            final_feedback=Feedback(FeedbackType.BINARY, False),
                            metadata={"reflection": "Confused meters with centimeters"})
    memory.store(trajectory)
    assert "Confused meters with centimeters" in prompts[0]
    assert memory.dump()[0]["success"] is False
