"""
Self-Evolving Agents Framework
Based on: "A Survey of Self-Evolving Agents" (arXiv:2507.21046)
      and "A Comprehensive Survey of Self-Evolving AI Agents" (arXiv:2508.07407)
"""
import os

# Importing the framework should not fetch provider metadata. Real completions
# still require an explicit call; users can opt into a refreshed cost map.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import Environment
from self_evolving.core.types import AgentState, Trajectory, Feedback

__version__ = "0.1.0"
__all__ = ["BaseAgent", "Environment", "AgentState", "Trajectory", "Feedback"]
