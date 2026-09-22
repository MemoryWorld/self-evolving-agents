"""
Example 7 — Generate offline demo data for the dashboard.

This script populates:
- SQLite runs and memory entries
- benchmark JSON artifacts

Run:
    python examples/07_generate_demo_data.py
"""

import argparse
import os

# An offline CLI must also avoid import-time provider metadata refreshes.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from self_evolving.dashboard.demo_data import generate_demo_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic offline data, not model scores")
    parser.add_argument("--db-path", default=".data/sea.db")
    parser.add_argument("--benchmark-dir", default="runs/benchmarks")
    args = parser.parse_args()
    result = generate_demo_data(args.db_path, args.benchmark_dir)
    print(result)


if __name__ == "__main__":
    main()
