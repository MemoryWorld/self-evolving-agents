"""
Episodic Memory — inter-test-time memory evolution.

Inspired by:
  - Expel (Zhao et al., 2024): extract reusable insights from trajectories
  - MemGPT / Mem0: hierarchical working + long-term memory
  - Agent Workflow Memory (AWM): store successful sub-routines

Design:
  store(trajectory)  → distil key lessons into memory entries
  retrieve(query)    → fuzzy keyword search over stored entries
  summarize()        → compress oldest entries to save context window
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import litellm

from self_evolving.core.types import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    source_task: str
    success: bool
    importance: float = 1.0     # higher = retrieved first
    access_count: int = 0


class EpisodicMemory:
    """
    A growing pool of distilled experiences.

    Usage:
        memory = EpisodicMemory(model="deepseek/deepseek-chat")
        agent.memory = memory

        # After each task:
        memory.store(trajectory)

        # During a new task (via agent._augment_with_memory):
        hits = memory.retrieve("how to sort a list", top_k=3)
    """

    DISTIL_PROMPT = """You are a memory distiller for an AI agent.
Given a task trajectory, extract 1-3 concise, reusable lessons.
Format: one lesson per line, starting with "LESSON:".
Focus on: what worked, what failed, key strategies to remember.

Task goal: {goal}
Outcome: {outcome}
Key steps:
{steps}
"""

    def __init__(
        self,
        model: Optional[str] = None,
        max_entries: int = 100,
        summarize_after: int = 10,
    ):
        self.model = model or os.getenv("SEA_WEAK_MODEL", "deepseek/deepseek-chat")
        self.max_entries = max_entries
        self.summarize_after = summarize_after
        self._entries: list[MemoryEntry] = []
        self._stored_count: int = 0

    # ------------------------------------------------------------------

    def store(self, trajectory: Trajectory) -> list[MemoryEntry]:
        """Distil a trajectory into memory entries."""
        lessons = self._distil(trajectory)
        new_entries = []
        for lesson in lessons:
            entry = MemoryEntry(
                content=lesson,
                source_task=trajectory.task_id,
                success=trajectory.success,
                importance=1.5 if trajectory.success else 0.8,
            )
            self._entries.append(entry)
            new_entries.append(entry)

        self._stored_count += 1
        self._maybe_summarize()
        self._trim()
        return new_entries

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k relevant memory entries for a query (keyword match)."""
        if not self._entries:
            return []

        query_words = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words), 1)
            score = overlap * entry.importance
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, entry in scored[:top_k]:
            if score > 0:
                entry.access_count += 1
                results.append(entry.content)
        return results

    def load(self, raw_entries: list[dict]) -> None:
        self._entries = [MemoryEntry(**e) for e in raw_entries]

    def dump(self) -> list[dict]:
        return [e.__dict__ for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------

    def _distil(self, trajectory: Trajectory) -> list[str]:
        steps_summary = "\n".join(
            f"  Step {s.step_index}: action={s.action[:120]!r}"
            for s in trajectory.steps[:6]
        )
        outcome = "SUCCESS" if trajectory.success else "FAILURE"
        prompt = self.DISTIL_PROMPT.format(
            goal=trajectory.goal,
            outcome=outcome,
            steps=steps_summary,
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            return [
                line.replace("LESSON:", "").strip()
                for line in raw.splitlines()
                if line.strip().startswith("LESSON:")
            ]
        except Exception as e:
            logger.warning(f"Memory distillation failed: {e}")
            outcome_str = "succeeded" if trajectory.success else "failed"
            return [f"Task '{trajectory.goal}' {outcome_str}."]

    def _maybe_summarize(self) -> None:
        if self._stored_count % self.summarize_after != 0:
            return
        if len(self._entries) < self.summarize_after:
            return
        # Compress the oldest quarter of entries into a single summary entry
        n = max(1, len(self._entries) // 4)
        old = self._entries[:n]
        self._entries = self._entries[n:]
        combined = " | ".join(e.content for e in old)
        summary_prompt = (
            f"Compress these agent lessons into 2 concise sentences:\n{combined}"
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            summary = resp.choices[0].message.content or combined[:200]
        except Exception:
            summary = combined[:200]
        self._entries.insert(
            0,
            MemoryEntry(
                content=f"[Summary] {summary}",
                source_task="summarized",
                success=True,
                importance=1.2,
            ),
        )

    def _trim(self) -> None:
        if len(self._entries) > self.max_entries:
            # Drop lowest-importance entries
            self._entries.sort(key=lambda e: e.importance, reverse=True)
            self._entries = self._entries[: self.max_entries]
