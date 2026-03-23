"""In-process background job management for API-triggered work."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


JobStatus = str


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: JobStatus
    progress: float
    stage: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Simple thread-pool-backed job manager for local development."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sea-job")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        *,
        kind: str,
        metadata: dict[str, Any],
        fn: Callable[[Callable[[float, str, dict[str, Any] | None], None]], dict[str, Any]],
    ) -> dict[str, Any]:
        now = time.time()
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            progress=0.0,
            stage="queued",
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run_job, job_id, fn)
        return self.get_job(job_id)

    def _run_job(
        self,
        job_id: str,
        fn: Callable[[Callable[[float, str, dict[str, Any] | None], None]], dict[str, Any]],
    ) -> None:
        self.update(job_id, status="running", progress=1.0, stage="starting")
        try:
            result = fn(lambda progress, stage, detail=None: self.update(job_id, progress=progress, stage=stage, detail=detail))
            self.update(job_id, status="completed", progress=100.0, stage="completed", result=result)
        except Exception as exc:
            self.update(job_id, status="failed", stage="failed", error=str(exc))

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        stage: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = max(0.0, min(100.0, progress))
            if stage is not None:
                record.stage = stage
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            if detail:
                record.metadata = {**record.metadata, **detail}
            record.updated_at = time.time()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return asdict(record)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )[:limit]
            return [asdict(item) for item in items]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
