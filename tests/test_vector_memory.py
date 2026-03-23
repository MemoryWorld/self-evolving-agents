"""Tests for vector-based episodic memory."""

from self_evolving.evolution.memory.episodic import EpisodicMemory


class DummyEmbedder:
    def embed(self, text: str) -> list[float]:
        text = text.lower()
        if "france" in text or "paris" in text:
            return [1.0, 0.0, 0.0]
        if "hamlet" in text or "shakespeare" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_vector_retrieval_prefers_semantic_match():
    memory = EpisodicMemory(embedder=DummyEmbedder())
    memory.load(
        [
            {
                "content": "Capital of France is Paris.",
                "source_task": "task-france",
                "success": True,
                "importance": 1.5,
                "access_count": 0,
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "content": "Hamlet was written by Shakespeare.",
                "source_task": "task-hamlet",
                "success": True,
                "importance": 1.5,
                "access_count": 0,
                "embedding": [0.0, 1.0, 0.0],
            },
        ]
    )

    hits = memory.retrieve("What is the capital city of France?", top_k=1)
    assert hits == ["Capital of France is Paris."]


def test_store_adds_embedding():
    class Trajectory:
        task_id = "task-1"
        goal = "What is the capital of France?"
        success = True
        steps = []

    memory = EpisodicMemory(embedder=DummyEmbedder())
    entries = memory.store(Trajectory())

    assert len(entries) == 1
    assert entries[0].embedding == [1.0, 0.0, 0.0]
