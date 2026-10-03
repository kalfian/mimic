"""HTTP API models (PLAN §5). FROZEN CONTRACT after Phase 0.

Mirrored in ``web/lib/types.ts`` via ``make contract``. All URLs in responses are API-relative
(``/api/jobs/<id>/...``); the client prefixes ``NEXT_PUBLIC_API_BASE_URL``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.auth.policy import USERNAME_PATTERN, normalize_username
from app.core.errors import ErrorCode
from app.core.stages import JobState, Stage
from app.generate import Outputs
from app.models.ir import InterpreterProvider, JobId, MotionSpec

#: Job ids are ``uuid4().hex``; validate on every route.
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
#: Upload extensions accepted (container is also sniffed).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"mp4", "mov", "webm", "m4v"})
#: Servable keyframe file names (whitelist for ``GET /api/jobs/{id}/keyframes/{name}``).
KEYFRAME_NAME_RE = re.compile(
    r"^(state_a|state_b|mid_25|mid_50|mid_75|annotated_a|annotated_b|el_e[1-9][0-9]*)\.png$"
)

PixelRatioOption = Literal["auto", "1", "2", "3"]
KeyframeKind = Literal[
    "state_a", "state_b", "mid_25", "mid_50", "mid_75", "annotated_a", "annotated_b", "element"
]
#: Account role (PLAN-auth U1). Same values as ``app.auth.models.Role``.
UserRole = Literal["admin", "user"]


class ApiModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        serialize_by_alias=True,
        json_schema_serialization_defaults_required=True,
    )


# ---- errors ---------------------------------------------------------------------------------


class ErrorDetail(ApiModel):
    code: ErrorCode
    message: str


class ErrorBody(ApiModel):
    """Every non-2xx JSON response: ``{"error": {"code", "message"}}``."""

    error: ErrorDetail


# ---- health ---------------------------------------------------------------------------------


class HealthInterpreter(ApiModel):
    mode: InterpreterProvider
    available: bool = Field(
        description=(
            "claude_cli: shutil.which(MIMIC_CLAUDE_BIN) is not None. openai_compat: "
            "MIMIC_LLM_BASE_URL + MIMIC_LLM_API_KEY + MIMIC_LLM_MODEL are configured (no network "
            "call). none: false."
        )
    )
    model: str | None = None


class HealthLimits(ApiModel):
    """Upload limits the server enforces (from settings), so the client can pre-check them."""

    max_upload_mb: int
    max_duration_s: float
    min_duration_s: float
    duration_tolerance_s: float = Field(description="Extra seconds tolerated over the maximum.")


class Health(ApiModel):
    """``GET /api/health``. Never calls ``claude`` or the LLM gateway.

    ``interpreter`` and ``limits`` are null for anonymous callers and for sessions with a
    pending forced password change (PLAN-auth A12); ``status``/``version``/``ffmpeg`` are public.
    """

    status: Literal["ok"] = "ok"
    version: str
    ffmpeg: bool
    interpreter: HealthInterpreter | None = None
    limits: HealthLimits | None = None


# ---- interpreter check ----------------------------------------------------------------------

StructuredOutputMode = Literal["json_schema", "json_object", "prompt_only"]
InterpreterCheckErrorCode = Literal[
    "not_configured",
    "unauthorized",
    "unreachable",
    "timeout",
    "model_not_found",
    "no_image_support",
    "bad_response",
]


class InterpreterCheckError(ApiModel):
    code: InterpreterCheckErrorCode
    message: str = Field(description="Human-readable; never contains credentials.")


class InterpreterCheck(ApiModel):
    """``POST /api/interpreter/check``: actively test the configured interpreter.

    ``ok: false`` is a normal 200 answer (with ``error``). Probe fields are null when that probe
    did not run (e.g. ``supports_images`` for ``claude_cli`` without a deep check).
    """

    mode: InterpreterProvider
    ok: bool
    latency_ms: int | None = Field(default=None, ge=0)
    model: str | None = None
    model_listed: bool | None = Field(
        default=None, description="openai_compat: the model id appears in GET /models."
    )
    supports_images: bool | None = None
    structured_output: StructuredOutputMode | None = Field(
        default=None, description="Best structured-output mode that worked."
    )
    error: InterpreterCheckError | None = None
    checked_at: AwareDatetime


# ---- jobs -----------------------------------------------------------------------------------


class JobOptions(ApiModel):
    """Upload form options echoed back. ``use_interpreter`` is opt-in (default off, PLAN P9)."""

    pixel_ratio: PixelRatioOption = "auto"
    use_interpreter: bool = False


class JobSource(ApiModel):
    """Probe summary; null until probing finished."""

    filename: str
    width: int
    height: int
    fps: float
    duration_s: float


class InterpretRequest(ApiModel):
    """``POST /api/jobs/{id}/interpret`` body. ``use_interpreter`` is required (no default): the
    request itself is the user's explicit choice to send keyframes to the AI interpreter
    (``true``) or to go back to heuristic labels (``false``)."""

    use_interpreter: bool


class JobCreated(ApiModel):
    """``POST /api/jobs`` -> 202."""

    id: JobId
    status: Literal["queued"] = "queued"
    created_at: AwareDatetime


class JobOwner(ApiModel):
    """The account that uploaded a job."""

    id: str
    username: str


class JobStatus(ApiModel):
    """``GET /api/jobs/{id}``. Poll until ``status`` is terminal.

    ``error`` is normally null unless ``status`` is ``failed``. Exception: a ``succeeded`` job
    whose interpretation re-run failed keeps its previous result and reports that failure here.
    ``owner`` is null only for legacy jobs uploaded before accounts existed (admin-only).
    """

    id: JobId
    status: JobState
    stage: Stage
    stage_label: str
    progress: float = Field(ge=0.0, le=1.0)
    error: ErrorDetail | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    source: JobSource | None = None
    options: JobOptions
    owner: JobOwner | None = None


class JobList(ApiModel):
    """``GET /api/jobs``: newest first (``created_at`` desc, ``id`` desc), keyset-paginated.

    Pass ``next_cursor`` back as ``cursor`` for the next page; null on the last page.
    """

    items: list[JobStatus]
    next_cursor: str | None = None


# ---- accounts (PLAN-auth §3.1) --------------------------------------------------------------


class Me(ApiModel):
    """``GET /api/auth/me``, ``POST /api/auth/login``, ``POST /api/auth/password``."""

    id: str
    username: str
    role: UserRole
    must_change_password: bool
    created_at: AwareDatetime


class AuthStatus(ApiModel):
    """``GET /api/auth/status`` (public). ``setup_required``: no admin exists yet."""

    setup_required: bool


class LoginRequest(ApiModel):
    """``POST /api/auth/login`` body. The username is stripped + lowercased server-side."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)

    @field_validator("username", mode="after")
    @classmethod
    def _normalize_username(cls, v: str) -> str:
        return normalize_username(v)


class ChangePasswordRequest(ApiModel):
    """``POST /api/auth/password`` body. The policy is checked in code (``weak_password``)."""

    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


class AdminUser(ApiModel):
    """One account as seen by an admin."""

    id: str
    username: str
    role: UserRole
    is_active: bool
    must_change_password: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime
    last_login_at: AwareDatetime | None = None
    job_count: int = Field(ge=0)


class AdminUserList(ApiModel):
    """``GET /api/admin/users``: every account, sorted by username (no pagination)."""

    items: list[AdminUser]


class CreateUserRequest(ApiModel):
    """``POST /api/admin/users`` body. The username is stripped + lowercased, then validated."""

    username: str = Field(pattern=USERNAME_PATTERN)
    role: UserRole = "user"

    @field_validator("username", mode="before")
    @classmethod
    def _normalize_username(cls, v: object) -> object:
        return normalize_username(v) if isinstance(v, str) else v


class UpdateUserRequest(ApiModel):
    """``PATCH /api/admin/users/{id}`` body. At least one field must be set (non-null)."""

    role: UserRole | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> UpdateUserRequest:
        if self.role is None and self.is_active is None:
            raise ValueError("set at least one of role, is_active")
        return self


class TemporaryPassword(ApiModel):
    """``POST /api/admin/users`` (201) and ``.../reset-password`` (200).

    ``temporary_password`` is shown once; the account must change it at next sign-in.
    """

    user: AdminUser
    temporary_password: str


# ---- result ---------------------------------------------------------------------------------


class KeyframeArtifact(ApiModel):
    """One PNG. ``t_ms`` is null for composite images (element crops)."""

    name: str = Field(pattern=KEYFRAME_NAME_RE.pattern)
    kind: KeyframeKind
    t_ms: int | None = None
    element_id: str | None = Field(default=None, description="Set for kind=element.")
    url: str


class Artifacts(ApiModel):
    video_url: str | None = Field(description="H.264 preview; null if the transcode failed.")
    keyframes: list[KeyframeArtifact] = Field(default_factory=list)


class ResultEnvelope(ApiModel):
    """``GET /api/jobs/{id}/result`` and ``data/jobs/<id>/result.json``."""

    job_id: JobId
    spec: MotionSpec
    outputs: Outputs
    artifacts: Artifacts

    @model_validator(mode="after")
    def _same_job(self) -> ResultEnvelope:
        if self.spec.job_id != self.job_id:
            raise ValueError("spec.job_id must equal job_id")
        return self


#: Models exported to ``docs/contract/motion-spec.schema.json`` and ``web/lib/types.ts``
#: (serialization mode: every field required, nullable where it applies).
CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    ResultEnvelope,
    JobCreated,
    JobStatus,
    JobList,
    InterpretRequest,
    Health,
    InterpreterCheck,
    ErrorBody,
    Me,
    AuthStatus,
    AdminUser,
    AdminUserList,
    TemporaryPassword,
)

#: Request bodies exported in *validation* mode, so fields with a server default are optional
#: in TypeScript (e.g. ``CreateUserRequest.role``, ``UpdateUserRequest.*``).
CONTRACT_REQUEST_MODELS: tuple[type[BaseModel], ...] = (
    LoginRequest,
    ChangePasswordRequest,
    CreateUserRequest,
    UpdateUserRequest,
)
