"""
Self-Refine — iterative output refinement within a single episode.

Reference: Madaan et al. (2023) "Self-Refine: Iterative Refinement with Self-Feedback"
           Covered in §5.2.1 of 2507.21046.

Core idea (intra-test-time):
  generate → critique → refine → critique → refine → ... → stop
"""
from __future__ import annotations
import logging
import os
from typing import Optional

import litellm

from self_evolving.mechanisms.reflection.base import BaseReflector
from self_evolving.core.types import Trajectory, Step, Feedback, FeedbackType

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are a critic evaluating an AI agent's answer.

Task: {goal}
Current answer: {answer}

Identify specific flaws and suggest concrete improvements.
If the answer is already correct and complete, output exactly: "STOP"
Otherwise output: "CRITIQUE: <your critique>"
"""

REFINE_PROMPT = """You are an AI agent refining your answer based on feedback.

Task: {goal}
Previous answer: {answer}
Critique: {critique}

Write an improved answer:"""


class SelfRefineReflector(BaseReflector):
    """
    Intra-test-time iterative refinement.

    Attach to an agent; after the initial answer is generated,
    this reflector runs critique→refine cycles before finalising.

    Usage:
        agent.reflector = SelfRefineReflector(max_rounds=2)
        trajectory = agent.run(env, goal)
        # trajectory.steps contains refinement steps
    """

    def __init__(self, model: Optional[str] = None, max_rounds: int = 2):
        self.model = model or os.getenv("SEA_MODEL", "deepseek/deepseek-chat")
        self.max_rounds = max_rounds

    def reflect(self, trajectory: Trajectory) -> Trajectory:
        if not trajectory.steps:
            return trajectory

        # Take the last action as the initial answer
        last_step = trajectory.steps[-1]
        answer = last_step.action

        for round_idx in range(self.max_rounds):
            critique = self._critique(trajectory.goal, answer)
            if critique == "STOP":
                logger.info(f"SelfRefine stopped at round {round_idx+1} (STOP signal)")
                break

            refined = self._refine(trajectory.goal, answer, critique)
            if not refined:
                break

            # Append refinement as additional steps in the trajectory
            trajectory.steps.append(
                Step(
                    observation=f"[Self-Critique]: {critique}",
                    action=refined,
                    feedback=Feedback(
                        type=FeedbackType.TEXTUAL,
                        value=critique,
                        source="self",
                    ),
                    step_index=len(trajectory.steps),
                )
            )
            answer = refined
            logger.info(f"SelfRefine round {round_idx+1} complete")

        trajectory.metadata["self_refine_rounds"] = round_idx + 1
        return trajectory

    def _critique(self, goal: str, answer: str) -> str:
        prompt = CRITIQUE_PROMPT.format(goal=goal, answer=answer[:800])
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("STOP"):
                return "STOP"
            return text.replace("CRITIQUE:", "").strip()
        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            return "STOP"

    def _refine(self, goal: str, answer: str, critique: str) -> Optional[str]:
        prompt = REFINE_PROMPT.format(goal=goal, answer=answer[:800], critique=critique[:400])
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return None
