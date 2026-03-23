"""FastAPI service for running and inspecting self-evolving agents."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evaluation.benchmark import BenchmarkRunner, BenchmarkTask
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.persistence.sqlite_store import SQLiteStore
from self_evolving.service.jobs import JobManager


class QARunRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_steps: int = Field(default=20, ge=1, le=100)
    use_memory: bool = True


class RunResponse(BaseModel):
    run_id: Optional[str]
    task_id: str
    agent_id: str
    goal: str
    success: bool
    total_reward: float
    num_steps: int
    reflection: Optional[str] = None


class BenchmarkTaskRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)


class BenchmarkRequest(BaseModel):
    tasks: list[BenchmarkTaskRequest] = Field(..., min_length=1)
    variants: list[str] = Field(default_factory=lambda: list(BenchmarkRunner.DEFAULT_VARIANTS))
    model: Optional[str] = None
    max_steps: int = Field(default=20, ge=1, le=100)


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    progress: float
    stage: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _build_agent(request: QARunRequest, store: SQLiteStore) -> BaseAgent:
    agent = BaseAgent(
        model=request.model,
        system_prompt=request.system_prompt,
        max_steps=request.max_steps,
        agent_id=request.agent_id,
    )
    agent.store = store

    if request.use_memory:
        memory = EpisodicMemory()
        memory.load(store.list_memory(agent.agent_id))
        agent.memory = memory

    return agent


def create_app(db_path: str | None = None) -> FastAPI:
    db_path = db_path or os.getenv("SEA_DB_PATH", ".data/sea.db")
    benchmark_dir = os.getenv("SEA_BENCHMARK_DIR", "runs/benchmarks")
    store = SQLiteStore(db_path)
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        app.state.jobs = jobs
        yield
        jobs.shutdown()

    app = FastAPI(
        title="Self-Evolving Agents API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.jobs = jobs

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs/qa", response_model=JobResponse)
    async def run_qa_task(request: QARunRequest) -> JobResponse:
        store: SQLiteStore = app.state.store
        jobs: JobManager = app.state.jobs

        def job_fn(progress_callback):
            env = SimpleQAEnvironment([(request.goal, request.reference_answer)])
            agent = _build_agent(request, store)
            trajectory = agent.run(
                env,
                goal=request.goal,
                task_id=request.task_id,
                progress_callback=progress_callback,
            )
            return RunResponse(
                run_id=trajectory.metadata.get("run_id"),
                task_id=trajectory.task_id,
                agent_id=agent.agent_id,
                goal=trajectory.goal,
                success=trajectory.success,
                total_reward=trajectory.total_reward,
                num_steps=len(trajectory.steps),
                reflection=trajectory.metadata.get("reflection"),
            ).model_dump()

        job = jobs.submit(
            kind="qa_run",
            metadata={
                "goal": request.goal,
                "agent_id": request.agent_id or "auto",
                "task_id": request.task_id,
            },
            fn=job_fn,
        )
        return JobResponse(**job)

    @app.get("/runs")
    async def list_runs(limit: int = 20) -> list[dict]:
        store: SQLiteStore = app.state.store
        return store.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        store: SQLiteStore = app.state.store
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/agents/{agent_id}/memory")
    async def get_agent_memory(agent_id: str, limit: int = 50) -> list[dict]:
        store: SQLiteStore = app.state.store
        return store.list_memory(agent_id, limit=limit)

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_jobs(limit: int = 20) -> list[JobResponse]:
        jobs: JobManager = app.state.jobs
        return [JobResponse(**job) for job in jobs.list_jobs(limit=limit)]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        jobs: JobManager = app.state.jobs
        try:
            return JobResponse(**jobs.get_job(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.post("/benchmarks/qa", response_model=JobResponse)
    async def run_qa_benchmark(request: BenchmarkRequest) -> JobResponse:
        jobs: JobManager = app.state.jobs

        def job_fn(progress_callback):
            tasks = [
                BenchmarkTask(task.goal, task.reference_answer)
                for task in request.tasks
            ]
            runner = BenchmarkRunner(
                tasks=tasks,
                output_dir=benchmark_dir,
                model=request.model,
                max_steps=request.max_steps,
                store=app.state.store,
            )
            summary = runner.run(
                variants=request.variants,
                progress_callback=progress_callback,
            )
            return {
                "session_dir": summary["session_dir"],
                "task_count": summary["task_count"],
                "variants": summary["variants"],
            }

        job = jobs.submit(
            kind="qa_benchmark",
            metadata={
                "task_count": len(request.tasks),
                "variants": request.variants,
            },
            fn=job_fn,
        )
        return JobResponse(**job)

    return app
