"""
Example 2 — Episodic memory evolution (inter-test-time).
The agent improves across tasks by learning from past experiences.
Run: python examples/02_memory_evolution.py
"""
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.evaluation.metrics import EvolutionMetrics

TASKS = [
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What is the molecular weight of water?", "18"),
]

def main():
    env = SimpleQAEnvironment(TASKS)
    memory = EpisodicMemory(summarize_after=3)
    agent = BaseAgent()
    agent.memory = memory
    metrics = EvolutionMetrics()

    print("Running tasks with episodic memory...\n")
    for i, (question, _) in enumerate(TASKS):
        trajectory = agent.run(env, goal=question, task_id=f"task_{i}")
        metrics.record(trajectory, evolution_round=i)
        status = "✓" if trajectory.success else "✗"
        print(f"{status} [{i+1}/{len(TASKS)}] {question!r}")

    print(f"\nMemory entries stored: {len(memory)}")
    print()
    print(metrics.report())

if __name__ == "__main__":
    main()
