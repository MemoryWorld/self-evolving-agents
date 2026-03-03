# Self-Evolving Agents Framework

A modular Python framework for building and experimenting with **self-evolving LLM agents** — agents that continuously improve through memory, reflection, prompt optimisation, and tool learning.

> Built on top of two comprehensive surveys:
> - [A Survey of Self-Evolving Agents: On Path to ASI](https://arxiv.org/abs/2507.21046) (arXiv:2507.21046)
> - [A Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407) (arXiv:2508.07407)

---

## Framework Overview

```
What to evolve  ×  When to evolve  ×  How to evolve
     (§3)               (§4)               (§5)
```

| Module | Covers | Methods |
|--------|--------|---------|
| `evolution/memory` | Memory evolution | Expel, MemGPT-style episodic memory |
| `evolution/prompt` | Prompt optimisation | OPRO |
| `evolution/tools`  | Tool learning | LATM, CREATOR |
| `mechanisms/reflection` | Intra/inter-test reflection | Reflexion, Self-Refine |
| `mechanisms/reward` | Scalar reward generation | LLM-as-judge |
| `evaluation` | Evolution metrics | Success rate, evolution gain, stability |

---

## Benchmark Results

All examples verified with **DeepSeek-V3** (`deepseek-chat`) and **Qwen2.5-32B** (local, RTX 5090).

| Example | Method | Success Rate | Notes |
|---------|--------|-------------|-------|
| 01 Basic Agent | Baseline | **100%** | 3/3 QA tasks |
| 02 Memory Evolution | EpisodicMemory | **75%** | 3/4 tasks; memory distillation active |
| 03 Reflexion | ReflexionAgent | **67–100%** | Verbal reflection on failure |
| 04 Prompt Optimization | OPRO | **75%** | 3 iterations; prompt evolved |
| 05 Tool Learning | ToolLearner | **100%** | `reverse_string`, `word_count` generated at runtime |

---

## Installation

```bash
git clone https://github.com/MemoryWorld/self-evolving-agents.git
cd self-evolving-agents
pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set your API key (DeepSeek, OpenAI, etc.)
```

---

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evolution.memory.episodic import EpisodicMemory

# Build agent with episodic memory
agent = BaseAgent()
agent.memory = EpisodicMemory()

env = SimpleQAEnvironment([("What is the capital of France?", "Paris")])
trajectory = agent.run(env, goal="What is the capital of France?")
print("Success:", trajectory.success)
```

---

## Examples

| File | Demonstrates |
|------|-------------|
| `examples/01_basic_agent.py` | Plain agent on QA tasks |
| `examples/02_memory_evolution.py` | Episodic memory across tasks |
| `examples/03_reflexion.py` | Verbal reflection on failure + retry |
| `examples/04_prompt_optimization.py` | OPRO prompt evolution |
| `examples/05_tool_learning.py` | Runtime tool creation (LATM/CREATOR) |

Run any example:
```bash
python examples/01_basic_agent.py
```

---

## Architecture

```
src/self_evolving/
├── core/
│   ├── agent.py          # BaseAgent — Π = (Γ, {ψi}, {Ci}, {Wi})
│   ├── environment.py    # POMDP environment interface + QA / tool-use envs
│   └── types.py          # Trajectory, Feedback, EvolutionRecord, ...
├── evolution/
│   ├── memory/
│   │   └── episodic.py   # EpisodicMemory — distil & retrieve past experiences
│   ├── prompt/
│   │   └── opro.py       # OPROOptimizer — LLM-as-optimiser for system prompts
│   └── tools/
│       └── learner.py    # ToolLearner — generate & register Python tools
├── mechanisms/
│   ├── reflection/
│   │   ├── reflexion.py  # ReflexionReflector + ReflexionAgent
│   │   └── self_refine.py# SelfRefineReflector — intra-episode critique+refine
│   └── reward/
│       └── scorer.py     # RewardScorer — LLM-as-judge scalar rewards
└── evaluation/
    └── metrics.py        # EvolutionMetrics — success rate, gain, stability
```

---

## Evolution Taxonomy

Following the surveys' three-dimensional taxonomy:

### What to evolve (§3)
- **Memory** — episodic storage of distilled lessons (`EpisodicMemory`)
- **Prompt** — automatic system prompt improvement (`OPROOptimizer`)
- **Tools** — runtime generation of reusable Python callables (`ToolLearner`)

### When to evolve (§4)
- **Intra-test-time** — within a single episode (`SelfRefineReflector`)
- **Inter-test-time** — across episodes (`EpisodicMemory`, `OPROOptimizer`, `ReflexionAgent`)

### How to evolve (§5)
- **Reward-based** — scalar reward signals (`RewardScorer`)
- **Imitation / Demonstration** — learn from past trajectories (`EpisodicMemory`)
- **Textual feedback** — verbal critiques (`ReflexionReflector`, `SelfRefineReflector`)

---

## Configuration

Edit `configs/default.yaml` or set environment variables in `.env`:

```yaml
model:
  primary: "deepseek/deepseek-chat"

evolution:
  when: "inter_test"
  what: ["memory", "prompt"]

reflection:
  strategy: "reflexion"
  max_reflection_rounds: 3
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [ ] Multi-agent evolution (population-based, co-evolution)
- [ ] TextGrad-style gradient-based prompt optimisation
- [ ] Benchmark harnesses (HotpotQA, ALFWorld, WebArena)
- [ ] Vector-store backed memory (FAISS / LanceDB)
- [ ] Fine-tuning loop integration (LoRA adapter evolution)

---

## Citation

```bibtex
@article{gao2025survey,
  title={A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence},
  author={Gao, Huan-ang and others},
  journal={arXiv:2507.21046},
  year={2025}
}

@article{fang2025comprehensive,
  title={A Comprehensive Survey of Self-Evolving AI Agents},
  author={Fang, Jinyuan and others},
  journal={arXiv:2508.07407},
  year={2025}
}
```

---

## License

MIT
