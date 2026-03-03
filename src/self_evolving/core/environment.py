"""
Abstract Environment interface — POMDP E = (G, S, A, T, R, Ω, O, γ).
Concrete environments (QA, coding, tool-use, etc.) subclass this.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from self_evolving.core.types import Feedback, FeedbackType


class Environment(ABC):
    """
    Minimal POMDP environment contract.

    Subclass and implement:
      - reset(goal) → initial observation
      - step(action) → (observation, feedback, done)
      - is_done()    → bool
    """

    def __init__(self, name: str = "env"):
        self.name = name
        self._goal: str = ""
        self._step_count: int = 0

    @abstractmethod
    def reset(self, goal: str) -> str:
        """Reset env for a new task goal; return the first observation."""
        self._goal = goal
        self._step_count = 0
        return goal

    @abstractmethod
    def step(self, action: str) -> tuple[str, Feedback, bool]:
        """
        Execute action.
        Returns (next_observation, feedback, done).
        """

    @abstractmethod
    def is_done(self) -> bool:
        """Return True when the episode is over."""

    def render(self) -> str:
        return f"[{self.name}] step={self._step_count} goal={self._goal!r}"


class SimpleQAEnvironment(Environment):
    """
    Minimal question-answering environment for smoke-testing.
    Gold answer must appear (case-insensitive) in the agent's response.
    """

    def __init__(self, qa_pairs: list[tuple[str, str]]):
        super().__init__("simple_qa")
        self._qa_pairs = qa_pairs
        self._current_answer: str = ""
        self._done: bool = False

    def reset(self, goal: str) -> str:
        super().reset(goal)
        self._current_answer = ""
        self._done = False
        for question, answer in self._qa_pairs:
            if question.strip().lower() == goal.strip().lower():
                self._current_answer = answer
                break
        return f"Question: {goal}"

    def step(self, action: str) -> tuple[str, Feedback, bool]:
        self._step_count += 1
        correct = self._current_answer.lower() in action.lower()
        self._done = True
        feedback = Feedback(
            type=FeedbackType.BINARY,
            value=correct,
            source="environment",
        )
        obs = "Correct!" if correct else f"Wrong. Expected: {self._current_answer}"
        return obs, feedback, self._done

    def is_done(self) -> bool:
        return self._done


class ToolUseEnvironment(Environment):
    """
    Environment where the agent can call registered Python callables as tools.
    Tool results are returned as observations.
    """

    def __init__(self, tools: dict[str, callable], max_steps: int = 20):
        super().__init__("tool_use")
        self._tools = tools
        self._max_steps = max_steps
        self._done: bool = False

    def reset(self, goal: str) -> str:
        super().reset(goal)
        self._done = False
        tool_list = ", ".join(self._tools.keys())
        return f"Goal: {goal}\nAvailable tools: {tool_list}"

    def step(self, action: str) -> tuple[str, Feedback, bool]:
        self._step_count += 1

        # Parse simple TOOL_NAME(args) format
        result = self._execute_action(action)
        done = self._step_count >= self._max_steps or "DONE:" in action
        self._done = done

        feedback = Feedback(
            type=FeedbackType.TEXTUAL,
            value=result,
            source="environment",
        )
        return result, feedback, done

    def _execute_action(self, action: str) -> str:
        for tool_name, fn in self._tools.items():
            if action.strip().startswith(tool_name):
                try:
                    # Naive arg extraction — production code should use JSON
                    args_str = action[len(tool_name):].strip().strip("()")
                    result = fn(args_str) if args_str else fn()
                    return str(result)
                except Exception as e:
                    return f"Tool error: {e}"
        return f"Unknown action: {action}"

    def is_done(self) -> bool:
        return self._done
