# Self-Evolving Agents Execution Tasks

This file turns the current prototype into an engineering-focused roadmap that can be executed step by step.

## Goal

Upgrade the project from a research prototype into a portfolio-ready LLM systems project with:

- service layer
- API layer
- benchmark harness
- persistence
- vector memory
- safer tool sandbox
- dashboard / observability
- Docker delivery
- CI workflow

Multi-GPU and distributed execution are intentionally out of scope for now.

## Phase 1: Stabilize The Core

### 1. Add a proper config loader
- Create a typed config object for model, memory, benchmark, service, and storage settings.
- Load config from YAML + env vars.
- Remove hard-coded defaults scattered across modules.
- Exit criteria:
  - one config entrypoint
  - examples can run with config only

### 2. Add structured logging
- Replace ad-hoc logs with consistent structured logs.
- Include task id, agent id, module, latency, and success fields.
- Exit criteria:
  - every run emits machine-readable logs

### 3. Expand tests
- Add tests for:
  - memory store / retrieve
  - prompt optimizer history behavior
  - tool learner validation failures
  - reward scorer parsing fallback
  - reflexion retry flow
- Exit criteria:
  - core modules have unit coverage beyond smoke tests

## Phase 2: Persistence And Better Memory

### 4. Add run persistence
- Persist trajectories, reflections, prompt history, and tool registry.
- Start simple with SQLite.
- Define tables or JSON schema for:
  - runs
  - steps
  - memories
  - prompts
  - tools
- Exit criteria:
  - runs survive process restart
  - historical runs can be queried

### 5. Replace keyword memory with vector memory
- Add embedding-based retrieval.
- Recommended first version:
  - sentence-transformers embeddings
  - FAISS index or SQLite + local vector store
- Keep keyword retrieval as fallback.
- Exit criteria:
  - memory retrieval uses semantic similarity
  - benchmark comparison exists: keyword vs vector

### 6. Add memory management policies
- Add recency + importance + similarity scoring.
- Add deduplication and summarization policy.
- Exit criteria:
  - memory quality does not degrade sharply after many episodes

## Phase 3: Benchmark Harness

### 7. Build a reusable benchmark runner
- Add a benchmark CLI that runs:
  - baseline
  - memory-enabled
  - reflexion-enabled
  - prompt-optimized
  - tool-learning
- Save metrics and artifacts per run.
- Exit criteria:
  - one command produces comparable experiment results

### 8. Add benchmark datasets / tasks
- Keep current QA tasks.
- Add more realistic tasks:
  - multi-step QA
  - tool-use tasks
  - simple coding tasks
  - retrieval-augmented tasks
- Exit criteria:
  - project has a small but non-trivial benchmark suite

### 9. Add evaluation artifacts
- Save:
  - success rate
  - reward distribution
  - adaptation speed
  - prompt evolution history
  - memory hit statistics
  - tool reuse counts
- Exit criteria:
  - results can be plotted and compared over time

## Phase 4: Service Layer And API

### 10. Add FastAPI service
- Add endpoints for:
  - create run
  - execute task
  - list runs
  - inspect trajectory
  - inspect memory
  - inspect prompts
  - inspect tools
- Exit criteria:
  - project can run as an API service

### 11. Add a job-oriented execution model
- Separate request handling from run execution.
- Support async task submission and status polling.
- Exit criteria:
  - long-running evaluations do not block the API thread

### 12. Add API docs and examples
- Provide curl examples and response schema examples.
- Exit criteria:
  - another person can use the API without reading source first

## Phase 5: Safer Tool Sandbox

### 13. Replace raw exec with sandboxed tool execution
- Minimum acceptable version:
  - subprocess-based isolated execution
  - restricted builtins
  - timeouts
  - output capture
- Better version:
  - dedicated worker process or container sandbox
- Exit criteria:
  - generated tools no longer run in the main interpreter directly

### 14. Add tool test cases
- Let the model generate code.
- Validate it against multiple examples, not just one.
- Exit criteria:
  - tool registration quality is measurably better

## Phase 6: Dashboard And Observability

### 15. Add a lightweight dashboard
- First version can be Streamlit.
- Show:
  - runs
  - trajectories
  - reflections
  - prompt versions
  - memory entries
  - benchmark charts
- Exit criteria:
  - project has a visual inspection layer

### 16. Add metrics and timing
- Track:
  - step latency
  - run latency
  - LLM call count
  - token usage if available
  - memory retrieval hit rate
- Exit criteria:
  - dashboard shows operational metrics, not just final scores

## Phase 7: Delivery

### 17. Add Docker support
- Add Dockerfile.
- Add docker-compose for API + storage.
- Exit criteria:
  - project runs in one reproducible container workflow

### 18. Add CI workflow
- Add GitHub Actions for:
  - lint
  - tests
  - packaging check
- Exit criteria:
  - pull requests are automatically validated

### 19. Add project documentation for engineering use
- Add:
  - architecture diagram
  - module responsibilities
  - benchmark workflow
  - service deployment workflow
- Exit criteria:
  - repo is understandable as an engineering project, not just a paper implementation

## Suggested Build Order

1. config loader
2. tests
3. SQLite persistence
4. vector memory
5. benchmark runner
6. FastAPI service
7. safer tool sandbox
8. dashboard
9. Docker
10. CI

## Portfolio Outcome Target

When these phases are done, this repo should be describable as:

"A modular LLM agent experimentation platform with persistent runs, semantic episodic memory, prompt evolution, tool learning, benchmark automation, API serving, and visual observability."
