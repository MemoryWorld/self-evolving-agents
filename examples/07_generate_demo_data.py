"""
Example 7 — Generate offline demo data for the dashboard.

This script populates:
- SQLite runs and memory entries
- benchmark JSON artifacts

Run:
    python examples/07_generate_demo_data.py
"""

from dotenv import load_dotenv

load_dotenv()

from self_evolving.dashboard.demo_data import generate_demo_data


def main():
    result = generate_demo_data()
    print(result)


if __name__ == "__main__":
    main()
