"""Base class for all reflection mechanisms."""
from __future__ import annotations
from abc import ABC, abstractmethod
from self_evolving.core.types import Trajectory


class BaseReflector(ABC):
    """
    Reflectors operate on a completed Trajectory and return an (optionally)
    improved version or attach reflection notes to it.
    """

    @abstractmethod
    def reflect(self, trajectory: Trajectory) -> Trajectory:
        """Process a trajectory; return updated trajectory."""
