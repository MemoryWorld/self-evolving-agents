"""
Reward Scorer — LLM-as-judge scalar reward generation.

Covers both:
  - Scalar rewards (§5.1 of 2507.21046): numeric feedback r ∈ [0, 1]
  - Textual feedback (§5.1): critique paired with a score

Used by OPRO, population-based evolution, and the evaluation module.
"""
from __future__ import annotations
import logging
import os
import re
from typing import Optional

import litellm

from self_evolving.core.types import Trajectory

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an impartial judge evaluating an AI agent's performance.

Task goal: {goal}
Agent's final answer: {answer}
Reference (if available): {reference}

Score the answer on a scale from 0.0 to 1.0:
  1.0 = perfectly correct and complete
  0.5 = partially correct or incomplete
  0.0 = incorrect or no useful answer

Output format (strictly follow):
SCORE: <number>
REASON: <one sentence>
"""


class RewardScorer:
    """
    Generates scalar rewards for agent trajectories using an LLM judge.

    Usage:
        scorer = RewardScorer()
        score = scorer.score(trajectory, reference="Paris")
        # score ∈ [0.0, 1.0]
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("SEA_WEAK_MODEL", "deepseek/deepseek-chat")

    def score(self, trajectory: Trajectory, reference: str = "") -> float:
        """Return a scalar reward in [0, 1] for the trajectory."""
        if not trajectory.steps:
            return 0.0

        final_answer = trajectory.steps[-1].action
        prompt = JUDGE_PROMPT.format(
            goal=trajectory.goal,
            answer=final_answer[:800],
            reference=reference or "(none)",
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=128,
            )
            text = resp.choices[0].message.content or ""
            match = re.search(r"SCORE:\s*([\d.]+)", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Scoring failed: {e}")

        # Fallback: use binary success signal
        return 1.0 if trajectory.success else 0.0

    def score_batch(
        self, trajectories: list[Trajectory], references: Optional[list[str]] = None
    ) -> list[float]:
        refs = references or [""] * len(trajectories)
        return [self.score(t, r) for t, r in zip(trajectories, refs)]
