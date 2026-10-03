"""HTTP API models (PLAN §5). FROZEN CONTRACT after Phase 0.

Mirrored in ``web/lib/types.ts`` via ``make contract``. All URLs in responses are API-relative
(``/api/jobs/<id>/...``); the client prefixes ``NEXT_PUBLIC_API_BASE_URL``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

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
    """``GET /api/health``. Never calls ``claude`` or the LLM gateway."""

    status: Literal["ok"] = "ok"
    version: str
    ffmpeg: bool
    interpreter: HealthInterpreter
    limits: HealthLimits


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


class JobStatus(ApiModel):
    """``GET /api/jobs/{id}``. Poll until ``status`` is terminal.

    ``error`` is normally null unless ``status`` is ``failed``. Exception: a ``succeeded`` job
    whose interpretation re-run failed keeps its previous result and reports that failure here.
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


#: Models exported to ``docs/contract/motion-spec.schema.json`` and ``web/lib/types.ts``.
CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    ResultEnvelope,
    JobCreated,
    JobStatus,
    InterpretRequest,
    Health,
    InterpreterCheck,
    ErrorBody,
)
