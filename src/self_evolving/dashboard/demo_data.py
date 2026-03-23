"""Offline demo data generation for the dashboard."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.benchmark import BenchmarkRunner, BenchmarkTask
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.persistence.sqlite_store import SQLiteStore


def _demo_answer(question: str) -> str:
    q = question.lower()
    if "capital of france" in q:
        return "ANSWER: Paris"
    if "12 * 7" in question:
        return "ANSWER: 84"
    if "hamlet" in q:
        return "ANSWER: Shakespeare"
    if "red planet" in q:
        return "ANSWER: Mars"
    if "atomic number 79" in q:
        return "ANSWER: Gold"
    return "ANSWER: unknown"


@contextmanager
def patched_demo_llm() -> Iterator[None]:
    original_call = BaseAgent._call_llm
    original_optimize = OPROOptimizer.optimize

    def fake_call(self, messages):
        return _demo_answer(messages[-1]["content"])

    def fake_optimize(self, initial_prompt, eval_fn, task_description="general agent task"):
        self._history = [
            (initial_prompt, eval_fn(initial_prompt)),
            (initial_prompt + " [optimized]", eval_fn(initial_prompt)),
        ]
        return initial_prompt + " [optimized]"

    BaseAgent._call_llm = fake_call
    OPROOptimizer.optimize = fake_optimize
    try:
        yield
    finally:
        BaseAgent._call_llm = original_call
        OPROOptimizer.optimize = original_optimize


def generate_demo_data(
    db_path: str = ".data/sea.db",
    benchmark_dir: str = "runs/benchmarks",
) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(benchmark_dir).mkdir(parents=True, exist_ok=True)

    tasks = [
        BenchmarkTask("What is the capital of France?", "Paris"),
        BenchmarkTask("What is 12 * 7?", "84"),
        BenchmarkTask("Who wrote Hamlet?", "Shakespeare"),
        BenchmarkTask("What planet is known as the Red Planet?", "Mars"),
    ]

    with patched_demo_llm():
        store = SQLiteStore(db_path)
        agent = BaseAgent(agent_id="demo-agent")
        agent.store = store
        agent.memory = EpisodicMemory()
        env = SimpleQAEnvironment([(task.goal, task.reference_answer) for task in tasks])

        generated_runs = []
        for index, task in enumerate(tasks):
            trajectory = agent.run(env, goal=task.goal, task_id=f"demo_task_{index}")
            generated_runs.append(trajectory.metadata.get("run_id"))

        runner = BenchmarkRunner(tasks, output_dir=benchmark_dir)
        benchmark_summary = runner.run()

    return {
        "db_path": db_path,
        "benchmark_dir": benchmark_dir,
        "generated_runs": generated_runs,
        "benchmark_summary": benchmark_summary,
    }
