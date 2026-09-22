"""Actual failure/retry behavior and cleanup, without model requests."""
import pytest

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.mechanisms.reflection.reflexion import ReflexionAgent, ReflexionReflector


class FixedReflector(ReflexionReflector):
    def reflect(self, trajectory):
        if not trajectory.success:
            trajectory.metadata["reflection"] = "Check the capital, not the country."
        return trajectory


def test_failed_attempt_retries_with_reflection_and_resets_for_next_task(monkeypatch):
    agent = BaseAgent(model="test", system_prompt="original")
    original_reflector = object()
    agent.reflector = original_reflector
    wrapper = ReflexionAgent(agent, FixedReflector(max_rounds=3))
    seen = []

    def answer(messages):
        seen.append(messages)
        return "ANSWER: Paris" if "Reflection from attempt" in messages[0]["content"] else "ANSWER: wrong"

    monkeypatch.setattr(agent, "_call_llm", answer)
    progress = []
    trajectory = wrapper.run(SimpleQAEnvironment([("capital?", "Paris")]), "capital?",
                             progress_callback=lambda value, stage, detail: progress.append(value))
    assert trajectory.success
    assert trajectory.metadata["attempt_count"] == 2
    assert trajectory.metadata["total_attempt_steps"] == 2
    assert "Check the capital" in seen[1][0]["content"]
    assert len(seen[1]) == 2  # new episode has no old conversation messages
    assert agent.state.system_prompt == "original"
    assert agent.reflector is original_reflector
    assert progress[-1] == 100
    assert progress == sorted(progress)
    wrapper.run(SimpleQAEnvironment([("capital?", "Paris")]), "capital?")
    assert seen[2][0]["content"] == "original"


def test_exhaustion_is_bounded_and_restores_prompt(monkeypatch):
    agent = BaseAgent(model="test", system_prompt="original")
    monkeypatch.setattr(agent, "_call_llm", lambda messages: "wrong")
    trajectory = ReflexionAgent(agent, FixedReflector(max_rounds=2)).run(
        SimpleQAEnvironment([("capital?", "Paris")]), "capital?")
    assert not trajectory.success
    assert trajectory.metadata["attempt_count"] == 2
    assert agent.state.system_prompt == "original"
    assert agent.reflector is None


def test_exception_during_retry_restores_prompt_and_reflector(monkeypatch):
    agent = BaseAgent(model="test", system_prompt="original")
    calls = 0

    def answer(messages):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated execution failure")
        return "wrong"

    monkeypatch.setattr(agent, "_call_llm", answer)
    with pytest.raises(RuntimeError, match="simulated"):
        ReflexionAgent(agent, FixedReflector(max_rounds=3)).run(
            SimpleQAEnvironment([("capital?", "Paris")]), "capital?")
    assert calls == 2
    assert agent.state.system_prompt == "original"
    assert agent.reflector is None


def test_zero_rounds_rejected():
    with pytest.raises(ValueError, match="max_rounds"):
        ReflexionReflector(max_rounds=0)


def test_attempt_aggregate_is_saved_in_final_run_metadata(tmp_path, monkeypatch):
    from self_evolving.persistence.sqlite_store import SQLiteStore
    agent = BaseAgent(model="test")
    agent.store = SQLiteStore(str(tmp_path / "runs.db"))
    monkeypatch.setattr(agent, "_call_llm", lambda messages: "wrong")
    trajectory = ReflexionAgent(agent, FixedReflector(max_rounds=2)).run(
        SimpleQAEnvironment([("capital?", "Paris")]), "capital?")
    persisted = agent.store.get_run(trajectory.metadata["run_id"])
    assert persisted["metadata"]["attempt_count"] == 2
    assert persisted["metadata"]["total_attempt_steps"] == 2
    assert len({attempt["run_id"] for attempt in persisted["metadata"]["reflexion_attempts"]}) == 2
