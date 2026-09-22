"""Deterministic synthetic dashboard data; no providers or global patches."""

from __future__ import annotations

from pathlib import Path

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.benchmark import BenchmarkRunner, BenchmarkTask
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.evolution.memory.embedders import HashingEmbedder
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.mechanisms.reflection.reflexion import ReflexionReflector
from self_evolving.persistence.sqlite_store import SQLiteStore


def _demo_answer(question: str) -> str:
    question = question.rsplit("[Current observation]:", 1)[-1]
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


class DemoAgent(BaseAgent):
    def __init__(self, **kwargs):
        kwargs["model"] = "offline/deterministic-fixture"
        super().__init__(**kwargs)

    def _call_llm(self, messages):
        return _demo_answer(messages[-1]["content"])


class DemoMemory(EpisodicMemory):
    def __init__(self, **kwargs):
        kwargs["embedder"] = HashingEmbedder()
        super().__init__(**kwargs)

    def _distil(self, trajectory):
        return [f"Synthetic lesson for {trajectory.goal}: inspect the current question."]

    def _summarize(self, combined):
        return f"Synthetic summary: {combined[:120]}"


class DemoReflector(ReflexionReflector):
    def reflect(self, trajectory):
        if not trajectory.success:
            trajectory.metadata["reflection"] = "Synthetic reflection: inspect the question."
        return trajectory


class DemoOptimizer(OPROOptimizer):
    def _propose(self, task_description):
        return BaseAgent.DEFAULT_SYSTEM + " Check the current question carefully."


def generate_demo_data(
    db_path: str = ".data/sea.db",
    benchmark_dir: str = "runs/benchmarks",
) -> dict:
    """Run actual control flow with canned local answers, not model-quality scores."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(benchmark_dir).mkdir(parents=True, exist_ok=True)

    tasks = [
        BenchmarkTask("What is the capital of France?", "Paris"),
        BenchmarkTask("What is 12 * 7?", "84"),
        BenchmarkTask("Who wrote Hamlet?", "Shakespeare"),
        BenchmarkTask("What planet is known as the Red Planet?", "Mars"),
    ]

    store = SQLiteStore(db_path)
    agent = DemoAgent(agent_id="demo-agent")
    agent.store = store
    agent.memory = DemoMemory()
    env = SimpleQAEnvironment([(task.goal, task.reference_answer) for task in tasks])

    generated_runs = []
    for index, task in enumerate(tasks):
        trajectory = agent.run(env, goal=task.goal, task_id=f"demo_task_{index}")
        generated_runs.append(trajectory.metadata.get("run_id"))

    runner = BenchmarkRunner(
        tasks, output_dir=benchmark_dir,
        tuning_tasks=[BenchmarkTask("Which element has atomic number 79?", "Gold")],
        agent_factory=DemoAgent, memory_factory=DemoMemory,
        reflector_factory=DemoReflector, optimizer_factory=DemoOptimizer,
        data_source="synthetic_deterministic_fixture",
    )
    benchmark_summary = runner.run()

    return {
        "data_source": "synthetic_deterministic_fixture",
        "db_path": db_path,
        "benchmark_dir": benchmark_dir,
        "generated_runs": generated_runs,
        "benchmark_summary": benchmark_summary,
    }
