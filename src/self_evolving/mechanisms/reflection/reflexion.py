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
from typing import Optional

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
            reflection = resp.choices[0].message.content.strip()
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
        self.agent.reflector = self.reflector

    def run(self, env, goal: str, task_id: Optional[str] = None):
        from self_evolving.core.environment import Environment
        original_prompt = self.agent.state.system_prompt

        for attempt in range(self.reflector.max_rounds):
            trajectory = self.agent.run(env, goal, task_id)
            if trajectory.success:
                self.agent.state.system_prompt = original_prompt
                return trajectory

            reflection = trajectory.metadata.get("reflection", "")
            if reflection:
                self.agent.state.system_prompt = (
                    original_prompt + f"\n\n[Reflection from attempt {attempt+1}]: {reflection}"
                )
            logger.info(f"Reflexion retry {attempt+2}/{self.reflector.max_rounds}")

        self.agent.state.system_prompt = original_prompt
        return trajectory
