"""
Core type definitions.

Formalises the POMDP agent system from 2508.07407 §2:
  Agent system Π = (Γ, {ψi}, {Ci}, {Wi})
  Environment  E = (G, S, A, T, R, Ω, O, γ)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class FeedbackType(str, Enum):
    SCALAR = "scalar"       # numeric reward  r ∈ ℝ
    TEXTUAL = "textual"     # natural-language critique
    BINARY = "binary"       # success / failure


class EvolutionStage(str, Enum):
    """When to evolve — §4 of 2507.21046."""
    INTRA_TEST = "intra_test"   # within a single task episode
    INTER_TEST = "inter_test"   # across task episodes


class EvolutionTarget(str, Enum):
    """What to evolve — §3 of 2507.21046."""
    MODEL = "model"
    MEMORY = "memory"
    PROMPT = "prompt"
    TOOLS = "tools"
    ARCHITECTURE = "architecture"


@dataclass
class Message:
    role: str           # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Feedback:
    """Unified feedback signal — covers scalar rewards + textual critiques."""
    type: FeedbackType
    value: float | str          # scalar or text
    source: str = "environment" # "environment" | "self" | "critic"
    timestamp: float = field(default_factory=time.time)

    @property
    def scalar(self) -> float:
        if self.type == FeedbackType.SCALAR:
            return float(self.value)
        if self.type == FeedbackType.BINARY:
            return 1.0 if self.value else 0.0
        # textual → caller must interpret
        raise ValueError("Cannot convert textual feedback to scalar directly.")


@dataclass
class Step:
    observation: str
    action: str
    feedback: Optional[Feedback] = None
    tool_calls: list[dict] = field(default_factory=list)
    step_index: int = 0


@dataclass
class Trajectory:
    """A full task episode — sequence of (obs, action, feedback) triples."""
    task_id: str
    goal: str
    steps: list[Step] = field(default_factory=list)
    final_feedback: Optional[Feedback] = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        if self.final_feedback is None:
            return False
        if self.final_feedback.type == FeedbackType.BINARY:
            return bool(self.final_feedback.value)
        return self.final_feedback.scalar >= 0.5

    @property
    def total_reward(self) -> float:
        rewards = []
        for step in self.steps:
            if step.feedback is None:
                continue
            try:
                rewards.append(step.feedback.scalar)
            except ValueError:
                continue
        return sum(rewards)


@dataclass
class AgentState:
    """Mutable snapshot of an agent at a given point in time."""
    agent_id: str
    system_prompt: str
    memory_entries: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    evolution_history: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvolutionRecord:
    """Tracks one evolution event for auditability."""
    target: EvolutionTarget
    stage: EvolutionStage
    before: Any
    after: Any
    trigger: str            # what caused this evolution
    gain: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
