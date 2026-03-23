"""Embedding backends for episodic memory retrieval."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEmbedder(ABC):
    """Simple embedding interface."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense vector for the input text."""


class HashingEmbedder(BaseEmbedder):
    """
    Lightweight local embedder based on feature hashing.

    This is not as semantically strong as a transformer encoder, but it gives
    us a deterministic vector space without extra runtime dependencies.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = [token.lower() for token in text.split() if token.strip()]
        if not tokens:
            return vec.tolist()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Optional local semantic embedder if sentence-transformers is installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it or use HashingEmbedder."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(va, vb) / denom)


def build_embedder(name: Optional[str] = None) -> BaseEmbedder:
    name = (name or "hashing").lower()
    if name == "sentence_transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()
