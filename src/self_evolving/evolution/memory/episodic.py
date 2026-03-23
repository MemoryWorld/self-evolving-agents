"""
Episodic Memory — inter-test-time memory evolution.

Now supports vector retrieval with a pluggable embedder backend.
Default behavior uses a lightweight local hashing embedder so the project
works without extra model dependencies.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import litellm

from self_evolving.core.types import Trajectory
from self_evolving.evolution.memory.embedders import (
    BaseEmbedder,
    build_embedder,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    source_task: str
    success: bool
    importance: float = 1.0
    access_count: int = 0
    embedding: list[float] = field(default_factory=list)


class EpisodicMemory:
    """
    A growing pool of distilled experiences.

    Retrieval uses:
    - vector similarity as the primary score
    - lexical overlap as a lightweight tiebreak / fallback
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
        embedder: Optional[BaseEmbedder] = None,
        lexical_weight: float = 0.15,
    ):
        self.model = model or os.getenv("SEA_WEAK_MODEL", "deepseek/deepseek-chat")
        self.max_entries = max_entries
        self.summarize_after = summarize_after
        self.embedder = embedder or build_embedder(os.getenv("SEA_MEMORY_EMBEDDER", "hashing"))
        self.lexical_weight = lexical_weight
        self._entries: list[MemoryEntry] = []
        self._stored_count: int = 0

    def store(self, trajectory: Trajectory) -> list[MemoryEntry]:
        """Distil a trajectory into memory entries."""
        lessons = self._distil(trajectory)
        new_entries: list[MemoryEntry] = []
        for lesson in lessons:
            entry = MemoryEntry(
                content=lesson,
                source_task=trajectory.task_id,
                success=trajectory.success,
                importance=1.5 if trajectory.success else 0.8,
                embedding=self.embedder.embed(lesson),
            )
            self._entries.append(entry)
            new_entries.append(entry)

        self._stored_count += 1
        self._maybe_summarize()
        self._trim()
        return new_entries

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k relevant memory entries for a query."""
        if not self._entries:
            return []

        query_embedding = self.embedder.embed(query)
        query_words = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries:
            vector_score = cosine_similarity(query_embedding, entry.embedding)
            lexical_score = self._lexical_overlap(query_words, entry.content)
            score = vector_score * entry.importance + lexical_score * self.lexical_weight
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, entry in scored[:top_k]:
            if score > 0:
                entry.access_count += 1
                results.append(entry.content)
        return results

    def load(self, raw_entries: list[dict]) -> None:
        self._entries = [MemoryEntry(**entry) for entry in raw_entries]

    def dump(self) -> list[dict]:
        return [entry.__dict__ for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def _distil(self, trajectory: Trajectory) -> list[str]:
        steps_summary = "\n".join(
            f"  Step {step.step_index}: action={step.action[:120]!r}"
            for step in trajectory.steps[:6]
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
        except Exception as exc:
            logger.warning(f"Memory distillation failed: {exc}")
            outcome_str = "succeeded" if trajectory.success else "failed"
            return [f"Task '{trajectory.goal}' {outcome_str}."]

    def _maybe_summarize(self) -> None:
        if self._stored_count % self.summarize_after != 0:
            return
        if len(self._entries) < self.summarize_after:
            return

        n_old = max(1, len(self._entries) // 4)
        old_entries = self._entries[:n_old]
        self._entries = self._entries[n_old:]
        combined = " | ".join(entry.content for entry in old_entries)
        summary_prompt = f"Compress these agent lessons into 2 concise sentences:\n{combined}"

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
                embedding=self.embedder.embed(summary),
            ),
        )

    def _trim(self) -> None:
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda entry: entry.importance, reverse=True)
            self._entries = self._entries[: self.max_entries]

    @staticmethod
    def _lexical_overlap(query_words: set[str], content: str) -> float:
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & content_words) / max(len(query_words), 1)
