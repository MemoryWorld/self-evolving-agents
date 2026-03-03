"""
Example 1 — Basic agent on a QA task.
Run: python examples/01_basic_agent.py
"""
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.metrics import EvolutionMetrics

QA_PAIRS = [
    ("What is the capital of France?", "Paris"),
    ("What is 12 * 7?", "84"),
    ("Who wrote Hamlet?", "Shakespeare"),
]

def main():
    env = SimpleQAEnvironment(QA_PAIRS)
    agent = BaseAgent()
    metrics = EvolutionMetrics()

    for question, _ in QA_PAIRS:
        trajectory = agent.run(env, goal=question)
        metrics.record(trajectory)
        status = "✓" if trajectory.success else "✗"
        print(f"{status} {question!r}")

    print()
    print(metrics.report())

if __name__ == "__main__":
    main()
