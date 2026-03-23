"""Tests for benchmark runner artifacts and summary structure."""

import json

from self_evolving.core.agent import BaseAgent
from self_evolving.evaluation.benchmark import BenchmarkRunner, BenchmarkTask
from self_evolving.evolution.prompt.opro import OPROOptimizer


def test_benchmark_runner_writes_artifacts(tmp_path, monkeypatch):
    def fake_call_llm(self, messages):
        prompt = messages[0]["content"]
        question = messages[-1]["content"]

        if "capital of france" in question.lower():
            return "ANSWER: Paris"
        if "12 * 7" in question:
            return "ANSWER: 84"
        if "hamlet" in question.lower():
            return "ANSWER: Shakespeare"
        if "red planet" in question.lower():
            return "ANSWER: Mars"
        return "ANSWER: unknown"

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(OPROOptimizer, "optimize", lambda self, initial_prompt, eval_fn, task_description="": initial_prompt + " optimized")

    runner = BenchmarkRunner(
        tasks=[
            BenchmarkTask("What is the capital of France?", "Paris"),
            BenchmarkTask("What is 12 * 7?", "84"),
        ],
        output_dir=str(tmp_path / "benchmarks"),
    )

    summary = runner.run()

    assert summary["task_count"] == 2
    assert "baseline" in summary["variants"]
    assert "memory" in summary["variants"]
    assert "reflexion" in summary["variants"]
    assert "prompt_optimization" in summary["variants"]

    session_dirs = list((tmp_path / "benchmarks").iterdir())
    assert len(session_dirs) == 1

    session_dir = session_dirs[0]
    assert (session_dir / "summary.json").exists()
    assert (session_dir / "baseline.json").exists()
    assert (session_dir / "memory.json").exists()
    assert (session_dir / "reflexion.json").exists()
    assert (session_dir / "prompt_optimization.json").exists()

    stored_summary = json.loads((session_dir / "summary.json").read_text())
    assert stored_summary["task_count"] == 2
