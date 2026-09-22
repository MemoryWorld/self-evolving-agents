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
- a FastAPI service layer for executing runs, benchmarks, and inspection queries
- a benchmark runner that compares agent variants and writes JSON artifacts
- a Streamlit dashboard with a lightweight control plane for runs, memory, and benchmark artifacts
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
- Retrieval is `cosine × importance + 0.15 × lexical overlap`, with a linear scan of the retained entries. Access count is tracked, not used as a ranking factor.
- Empty memory now accumulates its first lesson. Completed episodes checkpoint the **full** memory: summaries, evictions, access counts and the stored-episode count survive reconstruction with the same database and `agent_id`.
- Retrieval outside `agent.run` requires an explicit `agent.save_memory()` to checkpoint its updated access counts. Failed episodes that raise before checkpoint do not save these changes.

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
- Memory checkpoints replace one agent's snapshot atomically with an optimistic revision check. Concurrent stale writers fail with `MemoryConflictError` instead of overwriting a newer snapshot; API jobs expose this as `failed` with an error message.
- `save_memory_entries` and `list_memory` retain their append/query interfaces. Internal snapshot loads preserve oldest-first order; legacy entries migrate automatically, but their unknown historical store count starts at zero.
- For read-only QA, reload with `agent.load_memory()` and rerun after a conflict. For tools with external side effects, reconcile those effects before retrying. Memory and trajectory writes are separate transactions, not an exactly-once workflow.

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
- Retries use a fresh conversation and the previous attempt's reflection. The original prompt and reflector are restored on success, exhaustion, or exceptions.
- Each attempt is persisted separately; the final run also records attempt IDs/counts and total attempt steps. Reflection text is included as unverified context when distilling a lesson, not treated as verified knowledge or a model-weight update.

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
- Explicit `tuning_tasks` are used only to score OPRO prompt candidates; the selected prompt is then evaluated on `tasks`. Exact normalized goal overlap is rejected. Without `tuning_tasks`, the protocol is labeled `resubstitution`, never held-out evaluation.
- Sessions have unique directories and agent IDs, so concurrent sessions do not overwrite artifacts or inherit a previous benchmark's memories. Summaries include the task manifest, protocol and data source.
- The QA scorer is a case-insensitive reference substring smoke test, not exact-match accuracy. Memory uses sequential online adaptation; Reflexion allows up to two attempts. Final-attempt mean steps and total retry steps are both identified, but reflection/OPRO token cost and latency are **not** measured. These variants do not represent equal-cost quality comparisons.

### Dashboard
- `app.py`
- `src/self_evolving/dashboard/data.py`
- Shows:
  - recent runs
  - run detail
  - agent memory
  - benchmark sessions and per-variant artifacts

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
- no container delivery files
- CI runs the offline regression suite and synthetic demo on Windows and Linux
- background jobs exist, but job state is in memory and is lost on API restart

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
6. add a lightweight dashboard and control plane
7. add a safer tool sandbox
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

For a reproducible offline acceptance run, no API keys or `.env` file are needed:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python examples/07_generate_demo_data.py --db-path .data/offline-demo.db --benchmark-dir .data/offline-benchmarks
```

The tests block provider calls and socket connections (except Windows' internal asyncio socket-pair setup). The demo uses explicit deterministic agent/memory/reflection/optimizer implementations, never global patches; its scores are **synthetic fixtures**, not real model results. Package imports use LiteLLM's bundled cost map by default to avoid a metadata download.

For an actual model-backed prompt experiment, provide independent tuning and evaluation tasks:

```python
from self_evolving.evaluation.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    tuning_tasks=[("What is 3 * 7?", "21"), ("Who wrote Hamlet?", "Shakespeare")],
    tasks=[("What is 8 * 9?", "72"), ("What is the capital of France?", "Paris")],
    model="your/configured-model",
)
summary = runner.run(["baseline", "prompt_optimization"])
print(summary["evaluation_protocol"])
```

These tiny examples demonstrate split wiring only. Disjoint question strings do not establish semantic independence; review task families and answer leakage before reporting a real benchmark result. Model-backed examples below require your own provider configuration and may incur API charges.

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

Create a QA run job:

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

The API now returns a job record immediately. Poll the job until it reaches `completed`:

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

Create a benchmark job:

```bash
curl -X POST http://127.0.0.1:8000/benchmarks/qa \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {"goal": "What is the capital of France?", "reference_answer": "Paris"},
      {"goal": "What is 12 * 7?", "reference_answer": "84"}
    ],
    "tuning_tasks": [
      {"goal": "What is 3 * 7?", "reference_answer": "21"}
    ],
    "variants": ["baseline", "memory", "reflexion", "prompt_optimization"]
  }'
```

Inspect recent jobs:

Completed benchmark jobs return `evaluation_protocol` and `data_source` alongside their variant results; the dashboard's session summary displays the same fields. If `tuning_tasks` is omitted, OPRO output is explicitly marked `resubstitution`.

```bash
curl http://127.0.0.1:8000/jobs
```

Inspect persisted runs:

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

### 7. Dashboard
```bash
streamlit run app.py
```

### 8. Generate offline demo data
```bash
python examples/07_generate_demo_data.py
```

This populates:
- SQLite runs
- agent memory entries
- benchmark JSON artifacts

Use it when you want the dashboard to have data immediately without calling a real external model.

The generated runs use model label `offline/deterministic-fixture` and benchmark JSON uses `data_source=synthetic_deterministic_fixture`. No learned-performance claim can be inferred from those values.

## Next Step

The next practical step after the current background-job control plane is:

- persist job state and execution logs beyond the API process

That would let the system:
- survive API restarts without losing in-flight job history
- retain richer execution traces for debugging and demos
- support retries, cancellation, and queued scheduling more cleanly
- prepare the project for container deployment and CI smoke tests

In other words, the next upgrade is turning the current local control plane into a more durable single-node agent platform.

## Tests

```bash
pytest tests/ -v
```

Regression coverage includes empty-memory cold start; app/agent reconstruction and memory injection; summary/eviction/counter restoration; snapshot rollback and stale-writer conflicts; actual Reflexion failure/retry/cleanup and persisted attempt evidence; OPRO tuning/evaluation separation; unique same-second artifacts; and the fully offline demo.

Remaining limits: the default hashing vectors are not a transformer semantic model; embedding model/version migration is not automatic. Memory lacks trusted-source filtering, semantic duplicate detection and tenant authorization. Tool learning is an experimental Python execution path, not a safe sandbox. Background job state remains process-local. No real-model improvement, training result, throughput or token-cost benchmark has been measured by the offline suite.

## Project Direction

The target outcome for this repo is no longer just:

"Implement the survey ideas."

The target outcome is:

"Turn self-evolving agents into a portfolio-grade LLM systems project with persistent experiments, semantic memory, benchmark automation, API serving, safer tool execution, and engineering-grade delivery."

## License

MIT
