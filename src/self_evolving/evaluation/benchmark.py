"""Benchmark runner for comparing self-evolving agent variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.metrics import EvolutionMetrics
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.mechanisms.reflection.reflexion import ReflexionAgent, ReflexionReflector
from self_evolving.persistence.sqlite_store import SQLiteStore


@dataclass
class BenchmarkTask:
    goal: str
    reference_answer: str


@dataclass
class VariantResult:
    name: str
    success_rate: float
    mean_reward: float
    mean_steps: float
    evolution_gain: Optional[float]
    adaptation_speed: Optional[int]
    stability: float
    metadata: dict = field(default_factory=dict)
    episodes: list[dict] = field(default_factory=list)


class BenchmarkRunner:
    """Run and persist benchmark comparisons across agent variants."""

    DEFAULT_VARIANTS = ("baseline", "memory", "reflexion", "prompt_optimization")

    def __init__(
        self,
        tasks: Iterable[BenchmarkTask | tuple[str, str]],
        output_dir: str = "runs/benchmarks",
        model: Optional[str] = None,
        max_steps: int = 20,
        store: Optional[SQLiteStore] = None,
        agent_id_prefix: str = "benchmark",
    ):
        self.tasks = [
            task if isinstance(task, BenchmarkTask) else BenchmarkTask(*task)
            for task in tasks
        ]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_steps = max_steps
        self.store = store
        self.agent_id_prefix = agent_id_prefix

    def run(
        self,
        variants: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> dict:
        variants = variants or list(self.DEFAULT_VARIANTS)
        ordered_variants = ["baseline"] + [variant for variant in variants if variant != "baseline"]
        session_dir = self._make_session_dir()
        if progress_callback:
            progress_callback(0.0, "starting", {"variant": "baseline"})

        def make_variant_progress(variant_index: int, variant_name: str):
            def callback(progress: float, stage: str, detail: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                overall = ((variant_index + (progress / 100.0)) / len(ordered_variants)) * 100.0
                progress_callback(
                    overall,
                    stage,
                    {
                        "variant": variant_name,
                        "variant_index": variant_index,
                        "variant_count": len(ordered_variants),
                        **detail,
                    },
                )
            return callback

        baseline_result = self._run_baseline(
            progress_callback=make_variant_progress(0, "baseline")
        )
        results: dict[str, VariantResult] = {"baseline": baseline_result}
        self._write_variant_artifact(session_dir, baseline_result)

        for variant_index, variant in enumerate(ordered_variants[1:], start=1):
            if variant == "baseline":
                continue
            result = self._run_variant(
                variant,
                baseline_result.success_rate,
                progress_callback=make_variant_progress(variant_index, variant),
            )
            results[variant] = result
            self._write_variant_artifact(session_dir, result)

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "session_dir": str(session_dir),
            "task_count": len(self.tasks),
            "variants": {name: asdict(result) for name, result in results.items()},
        }
        self._write_summary_artifact(session_dir, summary)
        if progress_callback:
            progress_callback(100.0, "completed", {"session_dir": str(session_dir)})
        return summary

    def _run_variant(
        self,
        variant: str,
        baseline_success_rate: float,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> VariantResult:
        if variant == "memory":
            return self._run_memory(baseline_success_rate, progress_callback=progress_callback)
        if variant == "reflexion":
            return self._run_reflexion(baseline_success_rate, progress_callback=progress_callback)
        if variant == "prompt_optimization":
            return self._run_prompt_optimization(
                baseline_success_rate,
                progress_callback=progress_callback,
            )
        raise ValueError(f"Unsupported benchmark variant: {variant}")

    def _run_baseline(
        self,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> VariantResult:
        agent = self._make_agent("baseline")
        metrics = EvolutionMetrics()
        episodes = self._run_tasks_with_agent(
            agent,
            metrics,
            variant_name="baseline",
            progress_callback=progress_callback,
        )
        report = metrics.report()
        return self._build_result("baseline", report, episodes)

    def _run_memory(
        self,
        baseline_success_rate: float,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> VariantResult:
        agent = self._make_agent("memory")
        agent.memory = EpisodicMemory()
        metrics = EvolutionMetrics(baseline_success_rate=baseline_success_rate)
        episodes = self._run_tasks_with_agent(
            agent,
            metrics,
            variant_name="memory",
            progress_callback=progress_callback,
        )
        report = metrics.report()
        return self._build_result("memory", report, episodes)

    def _run_reflexion(
        self,
        baseline_success_rate: float,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> VariantResult:
        agent = self._make_agent("reflexion")
        wrapped = ReflexionAgent(agent, ReflexionReflector(model=agent.model, max_rounds=2))
        metrics = EvolutionMetrics(baseline_success_rate=baseline_success_rate)
        episodes = self._run_tasks_with_runner(
            wrapped,
            metrics,
            variant_name="reflexion",
            progress_callback=progress_callback,
        )
        report = metrics.report()
        return self._build_result("reflexion", report, episodes)

    def _run_prompt_optimization(
        self,
        baseline_success_rate: float,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> VariantResult:
        initial_prompt = BaseAgent.DEFAULT_SYSTEM
        if progress_callback:
            progress_callback(5.0, "optimizing_prompt", {"variant": "prompt_optimization"})
        optimizer = OPROOptimizer(model=self.model, max_iterations=3, batch_size=min(4, len(self.tasks)))
        eval_fn = self._make_eval_fn()
        best_prompt = optimizer.optimize(
            initial_prompt=initial_prompt,
            eval_fn=eval_fn,
            task_description="benchmark question answering",
        )
        if progress_callback:
            progress_callback(30.0, "optimized_prompt", {"variant": "prompt_optimization"})

        agent = self._make_agent("prompt_optimization", system_prompt=best_prompt)
        metrics = EvolutionMetrics(baseline_success_rate=baseline_success_rate)
        def on_task_progress(progress: float, stage: str, detail: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            scaled = 30.0 + (progress * 0.7)
            progress_callback(scaled, stage, detail)

        episodes = self._run_tasks_with_agent(
            agent,
            metrics,
            variant_name="prompt_optimization",
            progress_callback=on_task_progress,
        )
        report = metrics.report()
        return self._build_result(
            "prompt_optimization",
            report,
            episodes,
            metadata={
                "best_prompt": best_prompt,
                "history": [
                    {"prompt": prompt, "score": score}
                    for prompt, score in optimizer.history
                ],
            },
        )

    def _make_eval_fn(self):
        tasks = list(self.tasks)

        def eval_fn(prompt: str) -> float:
            agent = BaseAgent(model=self.model, max_steps=self.max_steps, system_prompt=prompt)
            env = SimpleQAEnvironment([(task.goal, task.reference_answer) for task in tasks])
            successes = 0
            for index, task in enumerate(tasks):
                trajectory = agent.run(env, goal=task.goal, task_id=f"opro_eval_{index}")
                if trajectory.success:
                    successes += 1
            return successes / len(tasks) if tasks else 0.0

        return eval_fn

    def _make_agent(self, variant: str, system_prompt: Optional[str] = None) -> BaseAgent:
        agent = BaseAgent(
            model=self.model,
            max_steps=self.max_steps,
            system_prompt=system_prompt,
            agent_id=f"{self.agent_id_prefix}-{variant}",
        )
        agent.store = self.store
        return agent

    def _run_tasks_with_agent(
        self,
        agent: BaseAgent,
        metrics: EvolutionMetrics,
        *,
        variant_name: str,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> list[dict]:
        env = SimpleQAEnvironment([(task.goal, task.reference_answer) for task in self.tasks])
        episodes = []
        for index, task in enumerate(self.tasks):
            def on_progress(progress: float, stage: str, detail: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                overall = ((index + (progress / 100.0)) / len(self.tasks)) * 100.0
                progress_callback(
                    overall,
                    stage,
                    {
                        **detail,
                        "variant": variant_name,
                        "task_index": index,
                        "task_count": len(self.tasks),
                    },
                )

            trajectory = agent.run(
                env,
                goal=task.goal,
                task_id=f"task_{index}",
                progress_callback=on_progress,
            )
            metrics.record(trajectory, evolution_round=index)
            episodes.append(self._trajectory_to_dict(trajectory))
        return episodes

    def _run_tasks_with_runner(
        self,
        runner,
        metrics: EvolutionMetrics,
        *,
        variant_name: str,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ) -> list[dict]:
        env = SimpleQAEnvironment([(task.goal, task.reference_answer) for task in self.tasks])
        episodes = []
        for index, task in enumerate(self.tasks):
            def on_progress(progress: float, stage: str, detail: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                overall = ((index + (progress / 100.0)) / len(self.tasks)) * 100.0
                progress_callback(
                    overall,
                    stage,
                    {
                        **detail,
                        "variant": variant_name,
                        "task_index": index,
                        "task_count": len(self.tasks),
                    },
                )

            trajectory = runner.run(
                env,
                goal=task.goal,
                task_id=f"task_{index}",
                progress_callback=on_progress,
            )
            metrics.record(trajectory, evolution_round=index)
            episodes.append(self._trajectory_to_dict(trajectory))
        return episodes

    @staticmethod
    def _trajectory_to_dict(trajectory) -> dict:
        return {
            "task_id": trajectory.task_id,
            "goal": trajectory.goal,
            "success": trajectory.success,
            "num_steps": len(trajectory.steps),
            "total_reward": trajectory.total_reward,
            "metadata": dict(trajectory.metadata),
        }

    @staticmethod
    def _build_result(name: str, report, episodes: list[dict], metadata: Optional[dict] = None) -> VariantResult:
        return VariantResult(
            name=name,
            success_rate=report.success_rate,
            mean_reward=report.mean_reward,
            mean_steps=report.mean_steps,
            evolution_gain=report.evolution_gain,
            adaptation_speed=report.adaptation_speed,
            stability=report.stability,
            metadata=metadata or {},
            episodes=episodes,
        )

    def _make_session_dir(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        session_dir = self.output_dir / stamp
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def _write_variant_artifact(session_dir: Path, result: VariantResult) -> None:
        path = session_dir / f"{result.name}.json"
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    @staticmethod
    def _write_summary_artifact(session_dir: Path, summary: dict) -> None:
        path = session_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    tasks = [
        BenchmarkTask("What is the capital of France?", "Paris"),
        BenchmarkTask("What is 12 * 7?", "84"),
        BenchmarkTask("Who wrote Hamlet?", "Shakespeare"),
        BenchmarkTask("What planet is known as the Red Planet?", "Mars"),
    ]
    runner = BenchmarkRunner(tasks)
    summary = runner.run()
    print(json.dumps(summary, indent=2))
