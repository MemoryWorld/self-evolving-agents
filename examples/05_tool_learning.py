"""
Example 5 — Tool learning (LATM / CREATOR style).
Agent creates and reuses custom Python tools at runtime.
Run: python examples/05_tool_learning.py
"""
from dotenv import load_dotenv
load_dotenv()

from self_evolving.evolution.tools.learner import ToolLearner

def main():
    learner = ToolLearner()

    print("=== Tool Learning Demo ===\n")

    # Task 1: create a string reversal tool
    fn = learner.create_tool(
        description="reverse a string",
        example_input="hello",
        expected_output="olleh",
        tool_name="reverse_string",
    )
    if fn:
        print(f"reverse_string('world') = {fn('world')}")
    else:
        print("reverse_string: creation failed")

    # Task 2: create a word counter
    fn2 = learner.create_tool(
        description="count the number of words in a sentence",
        example_input="the quick brown fox",
        expected_output="4",
        tool_name="word_count",
    )
    if fn2:
        print(f"word_count('hello world foo') = {fn2('hello world foo')}")
    else:
        print("word_count: creation failed")

    print(f"\nRegistered tools: {learner.list_tools()}")

if __name__ == "__main__":
    main()
