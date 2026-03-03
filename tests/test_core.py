"""Unit tests for core types and environment — no LLM calls needed."""
import pytest
from self_evolving.core.types import (
    Trajectory, Step, Feedback, FeedbackType, EvolutionTarget, EvolutionStage,
)
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.metrics import EvolutionMetrics


def make_trajectory(success: bool, n_steps: int = 2) -> Trajectory:
    traj = Trajectory(task_id="test", goal="test goal")
    for i in range(n_steps):
        traj.steps.append(Step(
            observation=f"obs_{i}",
            action=f"action_{i}",
            step_index=i,
        ))
    traj.final_feedback = Feedback(
        type=FeedbackType.BINARY,
        value=success,
        source="environment",
    )
    return traj


class TestFeedback:
    def test_binary_scalar(self):
        fb = Feedback(type=FeedbackType.BINARY, value=True, source="env")
        assert fb.scalar == 1.0

    def test_scalar_clamped(self):
        fb = Feedback(type=FeedbackType.SCALAR, value=0.75, source="env")
        assert fb.scalar == 0.75

    def test_textual_raises(self):
        fb = Feedback(type=FeedbackType.TEXTUAL, value="good job", source="env")
        with pytest.raises(ValueError):
            _ = fb.scalar


class TestTrajectory:
    def test_success_flag(self):
        assert make_trajectory(success=True).success is True
        assert make_trajectory(success=False).success is False

    def test_step_count(self):
        t = make_trajectory(success=True, n_steps=5)
        assert len(t.steps) == 5


class TestSimpleQAEnvironment:
    def test_correct_answer(self):
        env = SimpleQAEnvironment([("What is 2+2?", "4")])
        env.reset("What is 2+2?")
        obs, fb, done = env.step("The answer is 4.")
        assert fb.scalar == 1.0
        assert done is True

    def test_wrong_answer(self):
        env = SimpleQAEnvironment([("What is 2+2?", "4")])
        env.reset("What is 2+2?")
        _, fb, done = env.step("The answer is 5.")
        assert fb.scalar == 0.0


class TestEvolutionMetrics:
    def test_success_rate(self):
        metrics = EvolutionMetrics()
        metrics.record(make_trajectory(success=True))
        metrics.record(make_trajectory(success=False))
        metrics.record(make_trajectory(success=True))
        report = metrics.report()
        assert report.success_rate == pytest.approx(2 / 3)

    def test_evolution_gain(self):
        metrics = EvolutionMetrics(baseline_success_rate=0.5)
        for _ in range(4):
            metrics.record(make_trajectory(success=True))
        report = metrics.report()
        assert report.evolution_gain == pytest.approx(0.5)

    def test_empty_report(self):
        report = EvolutionMetrics().report()
        assert report.success_rate == 0.0
