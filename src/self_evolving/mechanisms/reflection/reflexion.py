"""
Reflexion — verbal reinforcement learning via self-reflection.

Reference: Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
           Covered in §5.1 / §5.2 of 2507.21046.

Core idea:
  After a failed episode, ask the agent to produce a verbal reflection
  (what went wrong, what to do differently).
  Prepend this reflection to the context in the *next* episode.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Callable, Optional

import litellm

from self_evolving.mechanisms.reflection.base import BaseReflector
from self_evolving.core.types import Trajectory

logger = logging.getLogger(__name__)


REFLEXION_PROMPT = """You are an AI agent reflecting on a recent task attempt.

Goal: {goal}
Outcome: {outcome}
Trajectory summary:
{trajectory_summary}

Write a concise reflection (2-4 sentences) that:
1. Identifies the key mistake or gap.
2. Proposes a concrete strategy for the next attempt.

Reflection:"""


class ReflexionReflector(BaseReflector):
    """
    Generates verbal reflections for failed trajectories.
    Reflections are stored in trajectory.metadata["reflection"]
    and can be prepended to the agent's system prompt on retry.

    Usage:
        reflector = ReflexionReflector()
        agent.reflector = reflector

        trajectory = agent.run(env, goal)
        if not trajectory.success:
            reflection = trajectory.metadata.get("reflection", "")
            agent.state.system_prompt += f"\n\nPast reflection: {reflection}"
    """

    def __init__(self, model: Optional[str] = None, max_rounds: int = 3):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.model = model or os.getenv("SEA_WEAK_MODEL", "deepseek/deepseek-chat")
        self.max_rounds = max_rounds

    def reflect(self, trajectory: Trajectory) -> Trajectory:
        if trajectory.success:
            return trajectory

        summary = self._summarise(trajectory)
        outcome = "FAILURE"
        prompt = REFLEXION_PROMPT.format(
            goal=trajectory.goal,
            outcome=outcome,
            trajectory_summary=summary,
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=256,
            )
            reflection = (resp.choices[0].message.content or "").strip()
            trajectory.metadata["reflection"] = reflection
            logger.info(f"Reflexion: {reflection[:100]}")
        except Exception as e:
            logger.warning(f"Reflexion generation failed: {e}")
        return trajectory

    def _summarise(self, trajectory: Trajectory) -> str:
        lines = []
        for s in trajectory.steps[:8]:
            lines.append(f"  obs={s.observation[:80]!r}  action={s.action[:80]!r}")
        return "\n".join(lines) or "(no steps recorded)"


class ReflexionAgent:
    """
    Wraps BaseAgent with Reflexion retry loop.

    On failure, appends the verbal reflection to the system prompt and retries.
    Stops after max_rounds or on first success.
    """

    def __init__(self, agent, reflector: Optional[ReflexionReflector] = None):
        from self_evolving.core.agent import BaseAgent
        self.agent: BaseAgent = agent
        self.reflector = reflector or ReflexionReflector(model=agent.model)

    def run(
        self,
        env,
        goal: str,
        task_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str, dict[str, Any]], None]] = None,
    ):
        original_prompt = self.agent.state.system_prompt
        original_reflector = self.agent.reflector
        self.agent.reflector = self.reflector
        attempts = []
        try:
            for attempt in range(self.reflector.max_rounds):
                def on_progress(progress: float, stage: str, detail: dict[str, Any]) -> None:
                    if progress_callback is None:
                        return
                    scaled = ((attempt + (progress / 100.0)) / self.reflector.max_rounds) * 100.0
                    progress_callback(scaled, "attempt_completed" if stage == "completed" else stage,
                                      {**detail, "attempt": attempt + 1,
                                       "max_attempts": self.reflector.max_rounds})

                trajectory = self.agent.run(env, goal, task_id, progress_callback=on_progress)
                attempts.append({"attempt": attempt + 1, "success": trajectory.success,
                                 "num_steps": len(trajectory.steps),
                                 "total_reward": trajectory.total_reward,
                                 "run_id": trajectory.metadata.get("run_id"),
                                 "reflection": trajectory.metadata.get("reflection")})
                if trajectory.success:
                    break
                reflection = trajectory.metadata.get("reflection", "")
                if reflection:
                    self.agent.state.system_prompt = (
                        original_prompt + f"\n\n[Reflection from attempt {attempt+1}]: {reflection}"
                    )
                if attempt + 1 < self.reflector.max_rounds:
                    logger.info(f"Reflexion retry {attempt+2}/{self.reflector.max_rounds}")
            trajectory.metadata["reflexion_attempts"] = attempts
            trajectory.metadata["attempt_count"] = len(attempts)
            trajectory.metadata["total_attempt_steps"] = sum(a["num_steps"] for a in attempts)
            if self.agent.store is not None and trajectory.metadata.get("run_id"):
                self.agent.store.update_run_metadata(trajectory.metadata["run_id"], trajectory.metadata)
            if progress_callback:
                progress_callback(100.0, "completed", {"attempt_count": len(attempts)})
            return trajectory
        finally:
            # Also restore state if execution, persistence, or callbacks fail.
            self.agent.state.system_prompt = original_prompt
            self.agent.reflector = original_reflector
