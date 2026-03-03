"""
OPRO-style Prompt Optimizer.

Reference: "Large Language Models as Optimizers" (Yang et al., 2024)
           Covered in both surveys under prompt evolution / §3.2.2.

Algorithm:
  1. Maintain a scored history of (prompt, score) pairs.
  2. Ask the meta-LLM to propose a better prompt given the history.
  3. Evaluate the new prompt on a mini validation set.
  4. Keep if improved; discard otherwise.
  5. Repeat for max_iterations rounds.
"""
from __future__ import annotations
import logging
import os
from typing import Callable, Optional

import litellm

logger = logging.getLogger(__name__)


class OPROOptimizer:
    """
    Optimises a system prompt by treating the LLM as the optimiser.

    Usage:
        optimizer = OPROOptimizer(model="deepseek/deepseek-chat")

        def eval_fn(prompt: str) -> float:
            # run agent with prompt on a small eval set, return 0-1 score
            ...

        best_prompt = optimizer.optimize(initial_prompt, eval_fn)
        agent.state.system_prompt = best_prompt
    """

    META_PROMPT = """You are a prompt engineer optimising a system prompt for an AI agent.

Below are previous prompts and their performance scores (higher = better).
Analyse the pattern and generate ONE improved prompt.

History (oldest → newest):
{history}

Task description: {task_description}

Rules:
- Output ONLY the new prompt text, no commentary.
- The new prompt should be more specific, clearer, and likely to score higher.
- Keep it under 300 words.

New improved prompt:"""

    def __init__(
        self,
        model: Optional[str] = None,
        max_iterations: int = 5,
        batch_size: int = 4,
    ):
        self.model = model or os.getenv("SEA_MODEL", "deepseek/deepseek-chat")
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self._history: list[tuple[str, float]] = []   # (prompt, score)

    def optimize(
        self,
        initial_prompt: str,
        eval_fn: Callable[[str], float],
        task_description: str = "general agent task",
    ) -> str:
        """
        Run OPRO optimisation loop.
        Returns the best prompt found.
        """
        current_prompt = initial_prompt
        current_score = eval_fn(current_prompt)
        self._history.append((current_prompt, current_score))
        logger.info(f"OPRO start  score={current_score:.3f}")

        best_prompt, best_score = current_prompt, current_score

        for iteration in range(self.max_iterations):
            candidate = self._propose(task_description)
            if not candidate:
                continue

            score = eval_fn(candidate)
            self._history.append((candidate, score))
            logger.info(f"OPRO iter={iteration+1}  score={score:.3f}  best={best_score:.3f}")

            if score > best_score:
                best_score = score
                best_prompt = candidate

        logger.info(f"OPRO done   best_score={best_score:.3f}")
        return best_prompt

    def _propose(self, task_description: str) -> Optional[str]:
        history_str = "\n".join(
            f"Score {score:.3f}: {prompt[:200]}"
            for prompt, score in self._history[-self.batch_size:]
        )
        meta_prompt = self.META_PROMPT.format(
            history=history_str,
            task_description=task_description,
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.9,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OPRO proposal failed: {e}")
            return None

    @property
    def history(self) -> list[tuple[str, float]]:
        return list(self._history)
