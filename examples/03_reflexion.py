"""
Example 3 — Reflexion: verbal reinforcement with retry.
Demonstrates intra-test-time self-reflection on failure.
Run: python examples/03_reflexion.py
"""
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.mechanisms.reflection.reflexion import ReflexionAgent, ReflexionReflector

TASKS = [
    ("Name the three primary colors in light (RGB).", "red"),
    ("What element has atomic number 79?", "gold"),
    ("What is the powerhouse of the cell?", "mitochondria"),
]

def main():
    env = SimpleQAEnvironment(TASKS)
    base_agent = BaseAgent()
    reflexion_agent = ReflexionAgent(base_agent, ReflexionReflector(max_rounds=2))

    for question, _ in TASKS:
        trajectory = reflexion_agent.run(env, goal=question)
        status = "✓" if trajectory.success else "✗"
        reflection = trajectory.metadata.get("reflection", "")
        print(f"{status} {question!r}")
        if reflection:
            print(f"   Reflection: {reflection[:120]}")
    print()

if __name__ == "__main__":
    main()
