"""
BaseAgent — the core agent abstraction.

Implements the agent system Π = (Γ, {ψi}, {Ci}, {Wi}) from 2508.07407 §2.
Self-evolution hooks are injected via EvolutionMixin.
"""
from __future__ import annotations
import os
import uuid
import logging
from typing import Optional, TYPE_CHECKING

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from self_evolving.core.types import (
    AgentState, Trajectory, Step, Feedback, FeedbackType, Message,
    EvolutionRecord, EvolutionTarget, EvolutionStage,
)
from self_evolving.core.environment import Environment

if TYPE_CHECKING:
    from self_evolving.evolution.memory.episodic import EpisodicMemory
    from self_evolving.mechanisms.reflection.base import BaseReflector
    from self_evolving.persistence.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    A self-evolving LLM agent.

    Core loop (per task):
        obs = env.reset(goal)
        while not done:
            action = agent.act(obs)
            obs, feedback, done = env.step(action)
            agent.observe(feedback)
        agent.evolve(trajectory)   ← inter-test-time evolution hook
    """

    DEFAULT_SYSTEM = (
        "You are a capable AI agent. Think step by step. "
        "When you have a final answer, prefix it with 'ANSWER:'."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 20,
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.model = model or os.getenv("SEA_MODEL", "deepseek/deepseek-chat")
        self.state = AgentState(
            agent_id=self.agent_id,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM,
        )
        self.max_steps = max_steps

        # Optional pluggable modules (set after construction)
        self.memory: Optional[EpisodicMemory] = None
        self.reflector: Optional[BaseReflector] = None
        self.store: Optional[SQLiteStore] = None

        self._conversation: list[Message] = []
        self._current_trajectory: Optional[Trajectory] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, env: Environment, goal: str, task_id: Optional[str] = None) -> Trajectory:
        """Execute a full task episode and return the trajectory."""
        task_id = task_id or str(uuid.uuid4())[:8]
        obs = env.reset(goal)
        trajectory = Trajectory(task_id=task_id, goal=goal)
        self._current_trajectory = trajectory
        self._reset_conversation()

        for step_idx in range(self.max_steps):
            # Augment observation with relevant memories
            augmented_obs = self._augment_with_memory(obs)
            action = self.act(augmented_obs)

            obs, feedback, done = env.step(action)
            step = Step(
                observation=augmented_obs,
                action=action,
                feedback=feedback,
                step_index=step_idx,
            )
            trajectory.steps.append(step)
            self.observe(feedback)

            if done:
                trajectory.final_feedback = feedback
                break

        # Intra-test-time reflection (within episode)
        if self.reflector:
            trajectory = self.reflector.reflect(trajectory)

        # Store episode in memory
        if self.memory:
            memory_entries = self.memory.store(trajectory)
            if self.store:
                self.store.save_memory_entries(self.agent_id, memory_entries)

        if self.store:
            trajectory.metadata["run_id"] = self.store.save_trajectory(
                agent=self,
                env_name=env.name,
                trajectory=trajectory,
            )

        return trajectory

    def act(self, observation: str) -> str:
        """Generate the next action given the current observation."""
        self._conversation.append(Message(role="user", content=observation))
        response = self._call_llm(self._build_messages())
        self._conversation.append(Message(role="assistant", content=response))
        return response

    def observe(self, feedback: Feedback) -> None:
        """Process feedback from the environment (intra-test hook)."""
        if feedback.type == FeedbackType.TEXTUAL:
            # Append environment feedback to conversation context
            self._conversation.append(
                Message(role="user", content=f"[Environment feedback]: {feedback.value}")
            )

    def evolve(self, trajectories: list[Trajectory]) -> list[EvolutionRecord]:
        """
        Inter-test-time evolution hook.
        Subclasses or plugged-in evolution modules override this.
        Returns a list of EvolutionRecord describing what changed.
        """
        records: list[EvolutionRecord] = []
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_conversation(self) -> None:
        self._conversation = []

    def _build_messages(self) -> list[dict]:
        system = self.state.system_prompt
        msgs = [{"role": "system", "content": system}]
        for m in self._conversation:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def _augment_with_memory(self, observation: str) -> str:
        if self.memory is None:
            return observation
        relevant = self.memory.retrieve(observation, top_k=3)
        if not relevant:
            return observation
        memory_block = "\n".join(
            f"[Past experience {i+1}]: {e}" for i, e in enumerate(relevant)
        )
        return f"{memory_block}\n\n[Current observation]: {observation}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_llm(self, messages: list[dict]) -> str:
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self) -> AgentState:
        return self.state

    def load_state(self, state: AgentState) -> None:
        self.state = state
        if self.memory and state.memory_entries:
            self.memory.load(state.memory_entries)

    def __repr__(self) -> str:
        return f"BaseAgent(id={self.agent_id}, model={self.model})"
