"""
Evaluation metrics for self-evolving agents.

Based on §7 of 2507.21046 — evaluation goals:
  - Task success rate
  - Evolution gain (improvement over baseline)
  - Stability (variance across runs)
  - Adaptation speed (steps to reach target performance)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import statistics

from self_evolving.core.types import Trajectory


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    total_reward: float
    num_steps: int
    had_reflection: bool = False
    evolution_round: int = 0


@dataclass
class EvaluationReport:
    success_rate: float
    mean_reward: float
    reward_std: float
    mean_steps: float
    evolution_gain: Optional[float]     # improvement vs. baseline
    adaptation_speed: Optional[int]     # episodes to reach 80% success
    stability: float                    # 1 - reward_std (higher = more stable)
    raw_results: list[EpisodeResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=== Evaluation Report ===",
            f"  Success rate:     {self.success_rate:.1%}",
            f"  Mean reward:      {self.mean_reward:.3f} ± {self.reward_std:.3f}",
            f"  Mean steps:       {self.mean_steps:.1f}",
            f"  Stability:        {self.stability:.3f}",
        ]
        if self.evolution_gain is not None:
            lines.append(f"  Evolution gain:   {self.evolution_gain:+.1%}")
        if self.adaptation_speed is not None:
            lines.append(f"  Adaptation speed: {self.adaptation_speed} episodes")
        return "\n".join(lines)


class EvolutionMetrics:
    """
    Accumulates episode results and computes evaluation reports.

    Usage:
        metrics = EvolutionMetrics()
        for trajectory in trajectories:
            metrics.record(trajectory)
        report = metrics.report()
        print(report)
    """

    def __init__(self, baseline_success_rate: Optional[float] = None):
        self._results: list[EpisodeResult] = []
        self._baseline = baseline_success_rate

    def record(self, trajectory: Trajectory, evolution_round: int = 0) -> EpisodeResult:
        result = EpisodeResult(
            task_id=trajectory.task_id,
            success=trajectory.success,
            total_reward=trajectory.total_reward,
            num_steps=len(trajectory.steps),
            had_reflection="reflection" in trajectory.metadata,
            evolution_round=evolution_round,
        )
        self._results.append(result)
        return result

    def report(self) -> EvaluationReport:
        if not self._results:
            return EvaluationReport(
                success_rate=0.0, mean_reward=0.0, reward_std=0.0,
                mean_steps=0.0, evolution_gain=None,
                adaptation_speed=None, stability=1.0,
            )

        successes = [r.success for r in self._results]
        rewards = [r.total_reward for r in self._results]
        steps = [r.num_steps for r in self._results]

        success_rate = sum(successes) / len(successes)
        mean_reward = statistics.mean(rewards)
        reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        mean_steps = statistics.mean(steps)
        stability = max(0.0, 1.0 - reward_std)

        evolution_gain = None
        if self._baseline is not None:
            evolution_gain = success_rate - self._baseline

        adaptation_speed = self._compute_adaptation_speed(successes, target=0.8)

        return EvaluationReport(
            success_rate=success_rate,
            mean_reward=mean_reward,
            reward_std=reward_std,
            mean_steps=mean_steps,
            evolution_gain=evolution_gain,
            adaptation_speed=adaptation_speed,
            stability=stability,
            raw_results=list(self._results),
        )

    def _compute_adaptation_speed(
        self, successes: list[bool], target: float = 0.8, window: int = 5
    ) -> Optional[int]:
        """Number of episodes until rolling success rate reaches target."""
        for i in range(window, len(successes) + 1):
            window_rate = sum(successes[i - window: i]) / window
            if window_rate >= target:
                return i
        return None

    def reset(self) -> None:
        self._results = []

    @property
    def n_episodes(self) -> int:
        return len(self._results)
