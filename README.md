# Self-Evolving Agents Framework

A modular Python framework for building and experimenting with self-evolving LLM agents.

This project started as a research-oriented implementation of ideas from self-evolving agent surveys. It has now been reframed as an engineering-focused agent platform roadmap: a system that can run agents, log trajectories, evolve prompts and memory, learn tools, and grow toward a serviceable benchmarkable platform.

## What It Does Today

The current codebase already supports:

- a reusable `BaseAgent` execution loop
- pluggable environments
- episodic memory with distilled lessons
- vector-based episodic memory retrieval with a local embedder fallback
- SQLite persistence for runs, steps, and memory
- a FastAPI service layer for executing and inspecting runs
- a benchmark runner that compares agent variants and writes JSON artifacts
- prompt optimization with an OPRO-style loop
- verbal reflection with retry
- runtime tool generation and registration
- LLM-as-judge reward scoring
- evaluation metrics and basic unit tests

In practical terms, the repo is already a working prototype for:

- running small agent tasks
- recording trajectories
- trying different evolution mechanisms
- comparing simple behaviors across tasks

It is not yet a production system, training framework, or distributed runtime.

## Current Architecture

```text
src/self_evolving/
├── core/
│   ├── agent.py
│   ├── environment.py
│   └── types.py
├── evolution/
│   ├── memory/episodic.py
│   ├── prompt/opro.py
│   └── tools/learner.py
├── mechanisms/
│   ├── reflection/
│   │   ├── base.py
│   │   ├── reflexion.py
│   │   └── self_refine.py
│   └── reward/scorer.py
└── evaluation/metrics.py
```

## Module Summary

### Core agent
- `src/self_evolving/core/agent.py`
- Handles the main loop:
  - reset environment
  - generate action
  - receive feedback
  - record trajectory
  - optionally use memory and reflection

### Environments
- `src/self_evolving/core/environment.py`
- Includes:
  - `SimpleQAEnvironment`
  - `ToolUseEnvironment`

### Episodic memory
- `src/self_evolving/evolution/memory/episodic.py`
- Stores lessons distilled from past trajectories.
- Retrieval now uses vector similarity as the primary score.
- Default setup uses a lightweight local hashing embedder so the project works without extra model downloads.
- The embedder backend is pluggable, so this can later be swapped to `sentence-transformers` or an external embedding model.

### Embedders
- `src/self_evolving/evolution/memory/embedders.py`
- Includes:
  - `HashingEmbedder` for zero-extra-dependency local vector retrieval
  - `SentenceTransformerEmbedder` as an optional stronger local backend

### Persistence
- `src/self_evolving/persistence/sqlite_store.py`
- Persists:
  - runs
  - steps
  - episodic memory entries

### API service
- `src/self_evolving/service/api.py`
- Exposes endpoints for:
  - health
  - run QA task
  - list runs
  - inspect run detail
  - inspect agent memory

### Prompt evolution
- `src/self_evolving/evolution/prompt/opro.py`
- Maintains prompt history and uses a meta-LLM to propose better prompts.

### Tool learning
- `src/self_evolving/evolution/tools/learner.py`
- Generates Python tool functions with the LLM, validates them, and registers them for reuse.

### Reflection
- `src/self_evolving/mechanisms/reflection/reflexion.py`
- Adds post-failure reflection and retry behavior.

### Reward scoring
- `src/self_evolving/mechanisms/reward/scorer.py`
- Uses an LLM judge to turn outcomes into scalar rewards.

### Evaluation
- `src/self_evolving/evaluation/metrics.py`
- Tracks:
  - success rate
  - evolution gain
  - stability
  - adaptation speed

### Benchmark runner
- `src/self_evolving/evaluation/benchmark.py`
- Compares:
  - baseline
  - memory
  - reflexion
  - prompt optimization
- Writes JSON artifacts for each variant plus a session summary.

## What This Repo Is Right Now

This repository is best described as:

"A working single-process LLM agent experimentation framework with memory, prompt evolution, reflection, tool learning, and evaluation."

That is strong enough for:

- research prototyping
- portfolio demonstration
- algorithmic experimentation
- framework design discussions

That is not yet enough for:

- production serving
- API deployment
- persistent run management
- serious benchmarking
- secure tool execution
- distributed execution

## Major Gaps

### Engineering gaps
- no dashboard
- no benchmark automation
- no container delivery files
- no CI workflow

### Systems / platform gaps
- no safe tool sandbox
- no multi-agent orchestration runtime
- no experiment artifact store
- no observability / tracing layer

### Intentionally out of scope for now
- multi-GPU execution
- distributed runtime
- LoRA / finetuning loop integration

Those can be added later, but the current focus is to make the repo excellent within a single-machine engineering scope first.

## Priority Roadmap

The immediate upgrade path is:

1. add stronger tests and config loading
2. add persistence for runs, memory, prompts, and tools
3. upgrade memory from keyword retrieval to vector retrieval
4. add a benchmark runner
5. harden the FastAPI service and add async job execution
6. add a safer tool sandbox
7. add a lightweight dashboard
8. add Docker and CI

The detailed execution breakdown is in:

- [TASKS.md](./TASKS.md)

## Installation

```bash
git clone https://github.com/MemoryWorld/self-evolving-agents.git
cd self-evolving-agents
pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set your API key
```

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment

agent = BaseAgent()
env = SimpleQAEnvironment([("What is the capital of France?", "Paris")])

trajectory = agent.run(env, goal="What is the capital of France?")
print("Success:", trajectory.success)
```

## API Quick Start

Run the API:

```bash
uvicorn self_evolving.service.api:create_app --factory --reload
```

Create a persisted QA run:

```bash
curl -X POST http://127.0.0.1:8000/runs/qa \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "What is the capital of France?",
    "reference_answer": "Paris",
    "agent_id": "demo-agent",
    "use_memory": true
  }'
```

Inspect stored runs:

```bash
curl http://127.0.0.1:8000/runs
```

Inspect stored memory for one agent:

```bash
curl http://127.0.0.1:8000/agents/demo-agent/memory
```

## Examples

### 1. Basic agent
```bash
python examples/01_basic_agent.py
```

### 2. Memory evolution
```bash
python examples/02_memory_evolution.py
```

### 3. Reflexion retry
```bash
python examples/03_reflexion.py
```

### 4. Prompt optimization
```bash
python examples/04_prompt_optimization.py
```

### 5. Tool learning
```bash
python examples/05_tool_learning.py
```

### 6. Benchmark runner
```bash
python examples/06_benchmark_runner.py
```

## Tests

```bash
pytest tests/ -v
```

## Project Direction

The target outcome for this repo is no longer just:

"Implement the survey ideas."

The target outcome is:

"Turn self-evolving agents into a portfolio-grade LLM systems project with persistent experiments, semantic memory, benchmark automation, API serving, safer tool execution, and engineering-grade delivery."

## License

MIT
