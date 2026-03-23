"""
Example 6 — Benchmark runner.
Compares baseline, memory, reflexion, and prompt optimization variants
and writes JSON artifacts under runs/benchmarks/.

Run:
    python examples/06_benchmark_runner.py
"""

from dotenv import load_dotenv

load_dotenv()

from self_evolving.evaluation.benchmark import BenchmarkRunner, BenchmarkTask


def main():
    tasks = [
        BenchmarkTask("What is the capital of France?", "Paris"),
        BenchmarkTask("What is 12 * 7?", "84"),
        BenchmarkTask("Who wrote Hamlet?", "Shakespeare"),
        BenchmarkTask("What planet is known as the Red Planet?", "Mars"),
    ]

    runner = BenchmarkRunner(tasks)
    summary = runner.run()
    print(summary)


if __name__ == "__main__":
    main()
