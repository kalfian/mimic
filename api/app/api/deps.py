"""Request-scoped access to the process services created in ``app.main``'s lifespan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.core.errors import ErrorCode, PipelineError
from app.core.jobstore import JobRecord, JobStore
from app.core.runner import JobRunner
from app.core.storage import JobStorage, is_valid_job_id


@dataclass(frozen=True, slots=True)
class Services:
    settings: Settings
    storage: JobStorage
    store: JobStore
    runner: JobRunner


def get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:  # only possible if the app was used without its lifespan
        raise PipelineError(ErrorCode.INTERNAL_ERROR, "Server is not initialized.")
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_job(job_id: str, services: ServicesDep) -> JobRecord:
    """Path param ``job_id`` -> record. Malformed and unknown ids are both ``not_found``."""
    if not is_valid_job_id(job_id):
        raise PipelineError(ErrorCode.NOT_FOUND)
    record = services.store.get(job_id)
    if record is None:
        raise PipelineError(ErrorCode.NOT_FOUND)
    return record


JobDep = Annotated[JobRecord, Depends(get_job)]
