"""FastAPI service for running and inspecting self-evolving agents."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from self_evolving.core.agent import BaseAgent
from self_evolving.core.environment import SimpleQAEnvironment
from self_evolving.evolution.memory.episodic import EpisodicMemory
from self_evolving.persistence.sqlite_store import SQLiteStore


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
    store = SQLiteStore(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        yield

    app = FastAPI(
        title="Self-Evolving Agents API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store = store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs/qa", response_model=RunResponse)
    async def run_qa_task(request: QARunRequest) -> RunResponse:
        store: SQLiteStore = app.state.store
        env = SimpleQAEnvironment([(request.goal, request.reference_answer)])
        agent = _build_agent(request, store)
        trajectory = agent.run(env, goal=request.goal, task_id=request.task_id)

        return RunResponse(
            run_id=trajectory.metadata.get("run_id"),
            task_id=trajectory.task_id,
            agent_id=agent.agent_id,
            goal=trajectory.goal,
            success=trajectory.success,
            total_reward=trajectory.total_reward,
            num_steps=len(trajectory.steps),
            reflection=trajectory.metadata.get("reflection"),
        )

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

    return app
