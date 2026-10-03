"""Job routes (PLAN §5): upload, list, status, result, preview video, keyframes, delete.

Authorization (PLAN-auth §3.3, §4, §14): every route needs a full session (401 / 403
``password_change_required``). A new job is owned by its uploader. Job-by-id routes go through
``JobDep`` (``app.api.deps.get_job``), which answers 404 ``not_found`` for a job the caller may not
see (users: their own jobs only; admins: every job incl. legacy ownerless ones).

Upload validation runs synchronously before the 202 (PLAN §5): form fields, extension,
size (``Content-Length`` pre-check, then a hard limit while copying), container sniff, ffprobe
(video stream, duration bounds) and a first-frame decode. On any failure the job directory is
removed and no job row is created.

The multipart body is parsed by hand (``request.form``) instead of FastAPI ``File``/``Form``
parameters so that (1) the size pre-check runs *before* the body is read and (2) every
validation failure uses the contract error body instead of FastAPI's ``{"detail": ...}``.
"""

from __future__ import annotations

import logging
import unicodedata
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import JobDep, ServicesDep
from app.api.schemas import (
    ALLOWED_EXTENSIONS,
    KEYFRAME_NAME_RE,
    ErrorBody,
    ErrorDetail,
    InterpretRequest,
    JobCreated,
    JobList,
    JobOptions,
    JobOwner,
    JobSource,
    JobStatus,
    PixelRatioOption,
    ResultEnvelope,
)
from app.auth.deps import CurrentUserDep
from app.auth.models import UserRecord
from app.auth.policy import is_valid_user_id
from app.core.errors import RERUN_UNAVAILABLE_MESSAGE, ErrorCode, PipelineError
from app.core.jobstore import LIST_LIMIT_MAX, InvalidCursor, JobRecord
from app.core.stages import JobState, stage_label
from app.core.storage import MEASUREMENT_FILE
from app.pipeline.probe import validate_upload

log = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

#: Allowance for multipart framing + small fields on top of the file-size limit when
#: pre-checking ``Content-Length``. The exact limit is enforced on the file bytes while copying.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_FILENAME_CHARS = 200
#: Default page size of ``GET /api/jobs``.
LIST_LIMIT_DEFAULT = 50
#: ``GET /api/jobs?owner=me``: the caller's own jobs.
OWNER_ME = "me"

_PIXEL_RATIO_OPTIONS: frozenset[str] = frozenset({"auto", "1", "2", "3"})
_TRUE = frozenset({"true", "1", "on", "yes"})
_FALSE = frozenset({"false", "0", "off", "no"})


def _error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {c: {"model": ErrorBody} for c in codes}


_UPLOAD_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "pixel_ratio": {
                            "type": "string",
                            "enum": ["auto", "1", "2", "3"],
                            "default": "auto",
                        },
                        "use_interpreter": {
                            "type": "string",
                            "enum": ["true", "false"],
                            "default": "false",
                        },
                    },
                }
            }
        },
    }
}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def to_job_status(rec: JobRecord) -> JobStatus:
    error = (
        ErrorDetail(code=rec.error_code, message=rec.error_message or "")
        if rec.error_code is not None
        else None
    )
    # ``owner`` is null for legacy jobs (no owner_id). A dangling owner_id (no users row; only
    # possible with old code or manual edits) also maps to null: there is no username to show.
    owner = (
        JobOwner(id=rec.owner_id, username=rec.owner_username)
        if rec.owner_id is not None and rec.owner_username is not None
        else None
    )
    return JobStatus(
        id=rec.id,
        status=rec.status,
        stage=rec.stage,
        stage_label=stage_label(rec.stage),
        progress=rec.progress,
        error=error,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        source=rec.source,
        options=rec.options,
        owner=owner,
    )


def _invalid(message: str) -> PipelineError:
    return PipelineError(ErrorCode.INVALID_REQUEST, message)


def _form_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid(f"Field {field!r} must be a text value.")
    return value.strip()


def parse_pixel_ratio(value: Any) -> PixelRatioOption:
    text = _form_text(value, "pixel_ratio")
    if not text:
        return "auto"
    text = text.lower()
    if text not in _PIXEL_RATIO_OPTIONS:
        raise _invalid("pixel_ratio must be one of auto, 1, 2, 3.")
    return text  # type: ignore[return-value]


def parse_use_interpreter(value: Any) -> bool:
    """Opt-in (PLAN P9): absent/empty -> False."""
    text = _form_text(value, "use_interpreter")
    if not text:
        return False
    text = text.lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise _invalid("use_interpreter must be true or false.")


def clean_filename(raw: str | None) -> str:
    """Display-only original name: basename, no control chars, bounded length."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    if len(name) > MAX_FILENAME_CHARS:
        stem, dot, ext = name.rpartition(".")
        name = (
            (stem[: MAX_FILENAME_CHARS - len(ext) - 1] + dot + ext)
            if dot
            else name[:MAX_FILENAME_CHARS]
        )
    return name or "upload"


def resolve_owner_filter(user: UserRecord, owner: str | None) -> str | None:
    """``owner`` query value -> ``list_jobs(owner_id=...)``.

    Users always get their own jobs (``owner`` may be omitted, ``me`` or their own id; another
    id is 403 ``forbidden``). Admins: omitted = every job incl. legacy ones, ``me`` = their own,
    a user id = that user's jobs (an unknown id simply yields an empty list). A value that is
    neither ``me`` nor a user id is 422 ``invalid_request``.
    """
    if owner is not None:
        owner = owner.strip()
        if owner == OWNER_ME:
            return user.id
        if not is_valid_user_id(owner):
            raise _invalid("owner must be 'me' or a user id.")
        if owner != user.id and not user.is_admin:
            raise PipelineError(ErrorCode.FORBIDDEN)
        return owner
    return None if user.is_admin else user.id


def _precheck_content_length(request: Request, max_bytes: int) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        return  # chunked: the copy limit still applies
    try:
        length = int(raw)
    except ValueError as exc:
        raise _invalid("Invalid Content-Length header.") from exc
    if length > max_bytes + MULTIPART_OVERHEAD_BYTES:
        raise PipelineError(
            ErrorCode.FILE_TOO_LARGE,
            f"The file is larger than {max_bytes // (1024 * 1024)} MB. Trim it or export it at "
            "a lower resolution.",
        )


# --------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------


@router.post(
    "/api/jobs",
    status_code=202,
    response_model=JobCreated,
    responses=_error_responses(401, 403, 413, 415, 422),
    openapi_extra=_UPLOAD_OPENAPI,
)
async def create_job(request: Request, services: ServicesDep, user: CurrentUserDep) -> JobCreated:
    """Upload a recording. The job is owned by the caller (PLAN-auth U3)."""
    settings, storage, store = services.settings, services.storage, services.store
    _precheck_content_length(request, settings.max_upload_bytes)

    try:
        form = await request.form(max_files=1, max_fields=8, max_part_size=64 * 1024)
    except StarletteHTTPException as exc:  # malformed multipart -> Starlette raises 400
        raise _invalid(f"Malformed multipart body: {exc.detail}") from exc

    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise _invalid("Missing file field 'file'.")
        options = JobOptions(
            pixel_ratio=parse_pixel_ratio(form.get("pixel_ratio")),
            use_interpreter=parse_use_interpreter(form.get("use_interpreter")),
        )
        filename = clean_filename(upload.filename)
        ext = filename.rpartition(".")[2].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT)

        job_id = uuid.uuid4().hex
        try:
            path = await run_in_threadpool(
                storage.save_upload, job_id, upload.file, ext, max_bytes=settings.max_upload_bytes
            )
            fp = await run_in_threadpool(validate_upload, path, settings)
            assert fp.duration_s is not None
            source = JobSource(
                filename=filename,
                width=fp.width,
                height=fp.height,
                fps=round(fp.fps_nominal, 3),
                duration_s=round(fp.duration_s, 3),
            )
            rec = store.create(
                job_id,
                original_filename=filename,
                ext=ext,
                options=options,
                source=source,
                owner_id=user.id,
            )
        except BaseException:
            storage.delete_job(job_id)
            raise
    finally:
        await form.close()

    services.runner.submit(job_id)
    log.info("job %s: queued for user %s (%s, %s)", job_id, user.id, ext, options.model_dump_json())
    return JobCreated(id=job_id, created_at=rec.created_at)


@router.get("/api/jobs", response_model=JobList, responses=_error_responses(401, 403, 422))
def list_jobs(
    user: CurrentUserDep,
    services: ServicesDep,
    owner: Annotated[
        str | None,
        Query(description="'me' or a user id. Ids other than your own are admin-only."),
    ] = None,
    status: Annotated[JobState | None, Query(description="Only jobs in this state.")] = None,
    limit: Annotated[int, Query(ge=1, le=LIST_LIMIT_MAX)] = LIST_LIMIT_DEFAULT,
    cursor: Annotated[
        str | None, Query(max_length=512, description="next_cursor of the previous page.")
    ] = None,
) -> JobList:
    """Jobs visible to the caller, newest first, keyset-paginated (PLAN-auth §3.3).

    Users see only their own jobs. Admins see every job (legacy ownerless ones included) unless
    ``owner`` narrows it down.
    """
    owner_id = resolve_owner_filter(user, owner)
    try:
        page = services.store.list_jobs(
            owner_id=owner_id, status=status, limit=limit, cursor=cursor or None
        )
    except InvalidCursor as exc:
        raise _invalid("cursor is not valid; start again without it.") from exc
    except ValueError as exc:  # limit out of range (also validated by Query above)
        raise _invalid(str(exc)) from exc
    return JobList(items=[to_job_status(r) for r in page.items], next_cursor=page.next_cursor)


@router.get(
    "/api/jobs/{job_id}", response_model=JobStatus, responses=_error_responses(401, 403, 404)
)
def get_job_status(job: JobDep) -> JobStatus:
    return to_job_status(job)


@router.get(
    "/api/jobs/{job_id}/result",
    response_model=ResultEnvelope,
    responses=_error_responses(401, 403, 404, 409),
)
def get_result(job: JobDep, services: ServicesDep) -> Response:
    if job.status is not JobState.SUCCEEDED:
        message = (
            "The job failed; there is no result."
            if job.status is JobState.FAILED
            else "The result is not ready yet."
        )
        raise PipelineError(ErrorCode.NOT_READY, message)
    try:
        body = services.storage.artifact_path(job.id, "result.json").read_bytes()
    except FileNotFoundError as exc:
        log.error("job %s: succeeded but result.json is missing", job.id)
        raise PipelineError(ErrorCode.INTERNAL_ERROR, "The result file is missing.") from exc
    # Served verbatim: it was validated as a ResultEnvelope before it was written.
    return Response(content=body, media_type="application/json")


@router.post(
    "/api/jobs/{job_id}/interpret",
    status_code=202,
    response_model=JobStatus,
    responses=_error_responses(401, 403, 404, 409, 422),
)
def rerun_interpretation(job: JobDep, body: InterpretRequest, services: ServicesDep) -> JobStatus:
    """Re-run AI labeling (Layer B) + assemble + render on the stored measurement.

    Only for ``succeeded`` jobs; the video is not decoded again. The job goes back through
    ``queued`` → ``interpreting`` → ``generating`` → ``done``; poll ``GET /api/jobs/{id}`` as after
    an upload. ``use_interpreter: true`` is the explicit opt-in that sends the keyframes to the
    configured interpreter (PLAN P9). Allowed on the caller's own jobs (admins: any job).
    """
    if job.status in (JobState.QUEUED, JobState.PROCESSING):
        raise PipelineError(ErrorCode.ALREADY_RUNNING)
    if job.status is JobState.FAILED:
        raise PipelineError(ErrorCode.NOT_READY, "The job failed; there is no result to re-label.")
    storage = services.storage
    if not storage.exists(job.id, MEASUREMENT_FILE) or not storage.exists(job.id, "result.json"):
        raise PipelineError(ErrorCode.NOT_READY, RERUN_UNAVAILABLE_MESSAGE)
    options = job.options.model_copy(update={"use_interpreter": body.use_interpreter})
    rec = services.runner.submit_reinterpret(job.id, options)
    log.info("job %s: interpretation re-run queued (use_interpreter=%s)", job.id,
             body.use_interpreter)  # fmt: skip
    return to_job_status(rec)


@router.get(
    "/api/jobs/{job_id}/video",
    response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {}}}, **_error_responses(401, 403, 404)},
)
def get_video(job: JobDep, services: ServicesDep) -> FileResponse:
    """H.264 preview. Starlette's ``FileResponse`` answers ``Range`` requests with 206."""
    path = services.storage.artifact_path(job.id, "preview.mp4")
    if not path.is_file():
        raise PipelineError(ErrorCode.NOT_FOUND, "The preview video is not available.")
    return FileResponse(path, media_type="video/mp4")


@router.get(
    "/api/jobs/{job_id}/keyframes/{name}",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}, **_error_responses(401, 403, 404)},
)
def get_keyframe(job: JobDep, name: str, services: ServicesDep) -> FileResponse:
    if KEYFRAME_NAME_RE.fullmatch(name) is None:
        raise PipelineError(ErrorCode.NOT_FOUND, "Unknown keyframe.")
    path = services.storage.artifact_path(job.id, f"keyframes/{name}")
    if not path.is_file():
        raise PipelineError(ErrorCode.NOT_FOUND, "Keyframe not found.")
    return FileResponse(path, media_type="image/png")


@router.delete(
    "/api/jobs/{job_id}",
    status_code=204,
    response_class=Response,
    responses=_error_responses(401, 403, 404),
)
def delete_job(job: JobDep, user: CurrentUserDep, services: ServicesDep) -> Response:
    """Delete a job, its row and its files (PLAN-auth §14). Owner or admin; anyone else gets 404.

    Works in any state. A queued or running job (incl. an interpretation re-run) is removed
    right away; the runner's deleted-job guard (A14) skips its status writes and removes the
    directory again if the pipeline re-created it.
    """
    services.store.delete(job.id)  # False = deleted concurrently: same outcome, still 204
    try:
        services.storage.delete_job(job.id)
    except Exception as exc:  # best effort: the row is gone, the runner guard retries running jobs
        log.warning("job %s: deleting its files failed (%s)", job.id, type(exc).__name__)
    log.info("user %s deleted job %s", user.id, job.id)
    return Response(status_code=204)
