"""
Tool Learner — agents learn to create and reuse tools.

Reference: LATM (Cai et al., 2023), CREATOR (Qian et al., 2023)
           Covered in §3.3 of 2507.21046.

Algorithm:
  1. When a task requires a capability not in the tool registry, ask the LLM
     to write a Python function for it.
  2. Sandbox-execute the generated code on test cases.
  3. If tests pass, register the function as a new tool.
  4. Reuse the tool in future tasks.
"""
from __future__ import annotations
import logging
import os
import textwrap
from typing import Optional, Callable

import litellm

logger = logging.getLogger(__name__)


TOOL_GEN_PROMPT = """You are a tool-creation assistant for an AI agent.
The agent needs a new Python function to accomplish a sub-task.

Sub-task description: {description}
Example input: {example_input}
Expected output: {expected_output}

Write a single Python function called `tool_fn` that solves this sub-task.
Output ONLY the function code — no imports outside the standard library, no main block.

```python
def tool_fn(...):
    ...
```
"""


class ToolLearner:
    """
    Dynamically generates and registers Python tool functions.

    Usage:
        learner = ToolLearner()
        fn = learner.create_tool(
            description="reverse a string",
            example_input="hello",
            expected_output="olleh",
        )
        # fn is now a callable and also stored in learner.registry
        result = fn("world")   # → "dlrow"
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("SEA_MODEL", "deepseek/deepseek-chat")
        self.registry: dict[str, Callable] = {}

    def create_tool(
        self,
        description: str,
        example_input: str = "",
        expected_output: str = "",
        tool_name: Optional[str] = None,
    ) -> Optional[Callable]:
        """
        Ask the LLM to generate a tool function, validate it, register it.
        Returns the callable or None if generation/validation fails.
        """
        code = self._generate_code(description, example_input, expected_output)
        if not code:
            return None

        fn = self._safe_exec(code)
        if fn is None:
            return None

        # Smoke-test against the provided example
        if example_input and expected_output:
            try:
                result = str(fn(example_input))
                if expected_output not in result and result not in expected_output:
                    logger.warning(
                        f"Tool smoke-test failed: got {result!r}, expected {expected_output!r}"
                    )
                    return None
            except Exception as e:
                logger.warning(f"Tool smoke-test exception: {e}")
                return None

        name = tool_name or description[:40].replace(" ", "_").lower()
        self.registry[name] = fn
        logger.info(f"New tool registered: {name!r}")
        return fn

    def get_tool(self, name: str) -> Optional[Callable]:
        return self.registry.get(name)

    def list_tools(self) -> list[str]:
        return list(self.registry.keys())

    # ------------------------------------------------------------------

    def _generate_code(
        self, description: str, example_input: str, expected_output: str
    ) -> Optional[str]:
        prompt = TOOL_GEN_PROMPT.format(
            description=description,
            example_input=example_input,
            expected_output=expected_output,
        )
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            # Extract code block
            if "```python" in raw:
                raw = raw.split("```python")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return textwrap.dedent(raw).strip()
        except Exception as e:
            logger.warning(f"Tool generation failed: {e}")
            return None

    def _safe_exec(self, code: str) -> Optional[Callable]:
        """Execute generated code in an isolated namespace."""
        namespace: dict = {}
        try:
            exec(code, namespace)  # noqa: S102
            fn = namespace.get("tool_fn")
            if callable(fn):
                return fn
            logger.warning("Generated code does not define `tool_fn`.")
            return None
        except Exception as e:
            logger.warning(f"Code execution error: {e}")
            return None
