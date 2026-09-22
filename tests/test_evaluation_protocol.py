"""Tuning/evaluation separation, artifacts and score interpretation."""
import json
from datetime import datetime as RealDatetime
from pathlib import Path

import pytest

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.benchmark import BenchmarkRunner
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.persistence.sqlite_store import SQLiteStore


def test_optimizer_only_scores_tuning_tasks_before_heldout_evaluation(tmp_path, monkeypatch):
    calls = []
    proposals = []

    def answer(self, messages):
        calls.append((messages[0]["content"], messages[-1]["content"]))
        return "ANSWER: correct" if messages[0]["content"] == "candidate" else "wrong"

    def propose(self, task_description):
        proposals.append(list(calls))
        return "candidate"

    monkeypatch.setattr(BaseAgent, "_call_llm", answer)
    monkeypatch.setattr(OPROOptimizer, "_propose", propose)
    runner = BenchmarkRunner(tasks=[("heldout question", "correct")],
                             tuning_tasks=[("tuning question", "correct")],
                             output_dir=str(tmp_path))
    summary = runner.run(["prompt_optimization"])
    # The baseline evaluates heldout first, but those scores are never supplied to OPRO.
    assert calls[0][1] == "Question: heldout question"
    assert all(question == "Question: tuning question" for _, question in calls[1:-1])
    assert calls[-1] == ("candidate", "Question: heldout question")
    result = summary["variants"]["prompt_optimization"]
    assert result["metadata"]["history"][0]["score"] == 0
    assert result["metadata"]["best_prompt"] == "candidate"
    assert summary["evaluation_protocol"]["prompt_optimization"] == "heldout"
    assert len(proposals) == 3


def test_default_protocol_never_claims_heldout(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: correct")
    monkeypatch.setattr(OPROOptimizer, "_propose", lambda self, description: "candidate")
    summary = BenchmarkRunner([("question", "correct")], output_dir=str(tmp_path)).run(
        ["prompt_optimization"])
    protocol = summary["evaluation_protocol"]
    assert protocol["prompt_optimization"] == "resubstitution"
    assert "substring_smoke_test" in protocol["scoring"]
    artifact = json.loads((Path(summary["session_dir"]) / "summary.json").read_text())
    assert artifact["evaluation_protocol"] == protocol


@pytest.mark.parametrize("tuning", [[(" QUESTION ", "different reference")], []])
def test_overlap_or_empty_tuning_rejected(tmp_path, tuning):
    with pytest.raises(ValueError):
        BenchmarkRunner([("question", "answer")], tuning_tasks=tuning, output_dir=str(tmp_path))


def test_new_optimizer_run_does_not_use_prior_task_history(monkeypatch):
    optimizer = OPROOptimizer(max_iterations=1)
    histories = []

    def propose(description):
        histories.append(optimizer.history)
        return "candidate"

    monkeypatch.setattr(optimizer, "_propose", propose)
    optimizer.optimize("first", lambda prompt: 0.25)
    optimizer.optimize("second", lambda prompt: 0.5)
    assert histories[1] == [("second", 0.5)]
    assert all(prompt != "first" for prompt, _ in optimizer.history)


def test_same_second_sessions_have_unique_artifacts_and_agents(tmp_path, monkeypatch):
    from self_evolving.evaluation import benchmark

    class FrozenDatetime:
        @staticmethod
        def now(tz):
            return RealDatetime(2026, 9, 22, tzinfo=tz)

    monkeypatch.setattr(benchmark, "datetime", FrozenDatetime)
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: correct")
    store = SQLiteStore(str(tmp_path / "runs.db"))
    runner = BenchmarkRunner([("question", "correct")], output_dir=str(tmp_path / "runs"), store=store)
    first = runner.run(["baseline"])
    second = runner.run(["baseline"])
    assert first["session_dir"] != second["session_dir"]
    assert len({run["agent_id"] for run in store.list_runs()}) == 2
    assert len(list((tmp_path / "runs").glob("*/summary.json"))) == 2


@pytest.mark.parametrize("pairs", [[], [("q", " ")], [("", "a")], [("Q", "a"), (" q ", "b")]])
def test_qa_rejects_invalid_reference_data(pairs):
    with pytest.raises(ValueError):
        SimpleQAEnvironment(pairs)


def test_qa_missing_goal_and_step_before_reset_cannot_pass():
    env = SimpleQAEnvironment([("q", "a")])
    with pytest.raises(RuntimeError):
        env.step("anything")
    with pytest.raises(ValueError):
        env.reset("missing")


def test_qa_substring_scoring_is_explicitly_only_a_smoke_test():
    env = SimpleQAEnvironment([("two plus two", "4")])
    env.reset("two plus two")
    # Documents a known limitation; this must not be called exact-match accuracy.
    assert env.step("ANSWER: 42")[1].value is True
