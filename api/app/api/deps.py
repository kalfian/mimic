"""Request-scoped access to the process services created in ``app.main``'s lifespan.

``get_job`` (``JobDep``) is the single choke point for every job-by-id route, so it is also where
job visibility is enforced (PLAN-auth U3, A10): it requires a full session (``CurrentUserDep``:
401 / 403 ``password_change_required``) and answers **404** ``not_found`` for a job the user may
not see, exactly like an unknown id, so existence never leaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.auth.deps import CurrentUserDep
from app.auth.policy import can_access_job
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


def get_job(job_id: str, services: ServicesDep, user: CurrentUserDep) -> JobRecord:
    """Path param ``job_id`` -> record the current user may see.

    Malformed ids, unknown ids and jobs of other users (or legacy ownerless jobs, for non-admins)
    are all ``not_found``. Authentication runs first (dependencies resolve before this body), so
    an anonymous caller gets 401 even for a malformed id.
    """
    if not is_valid_job_id(job_id):
        raise PipelineError(ErrorCode.NOT_FOUND)
    record = services.store.get(job_id)
    if record is None or not can_access_job(user, record.owner_id):
        raise PipelineError(ErrorCode.NOT_FOUND)
    return record


JobDep = Annotated[JobRecord, Depends(get_job)]
