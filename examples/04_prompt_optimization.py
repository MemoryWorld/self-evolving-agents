"""
Example 4 — OPRO prompt optimisation (inter-test-time).
Shows how the agent's system prompt evolves to improve task performance.
Run: python examples/04_prompt_optimization.py
"""
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evolution.prompt.opro import OPROOptimizer
from self_evolving.evaluation.metrics import EvolutionMetrics

EVAL_TASKS = [
    ("What is the speed of light in km/s?", "300000"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the largest ocean on Earth?", "Pacific"),
]

def make_eval_fn(tasks):
    """Build an evaluation function for OPRO."""
    def eval_fn(prompt: str) -> float:
        env = SimpleQAEnvironment(tasks)
        agent = BaseAgent(system_prompt=prompt)
        successes = 0
        for question, _ in tasks:
            t = agent.run(env, goal=question)
            if t.success:
                successes += 1
        return successes / len(tasks)
    return eval_fn

def main():
    initial_prompt = "You are a helpful AI. Answer questions concisely."

    optimizer = OPROOptimizer(max_iterations=3)
    eval_fn = make_eval_fn(EVAL_TASKS)

    print("Running OPRO prompt optimisation...\n")
    best_prompt = optimizer.optimize(
        initial_prompt=initial_prompt,
        eval_fn=eval_fn,
        task_description="factual question answering",
    )

    print("Optimisation history:")
    for i, (prompt, score) in enumerate(optimizer.history):
        print(f"  [{i}] score={score:.2f}  prompt={prompt[:80]!r}")

    print(f"\nBest prompt: {best_prompt!r}")

    # Final evaluation
    metrics = EvolutionMetrics(baseline_success_rate=optimizer.history[0][1])
    env = SimpleQAEnvironment(EVAL_TASKS)
    agent = BaseAgent(system_prompt=best_prompt)
    for question, _ in EVAL_TASKS:
        t = agent.run(env, goal=question)
        metrics.record(t)

    print()
    print(metrics.report())

if __name__ == "__main__":
    main()
