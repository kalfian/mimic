"""Error codes and the single exception type used by validation and the pipeline (PLAN §4.4, §5).

HTTP-time codes (upload validation, routing) are returned with ``http_status`` and the body
``{"error": {"code", "message"}}``. Pipeline-time codes surface through ``JobStatus.error``;
their ``http_status`` is only used if one is ever raised inside a request.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Closed set of error codes shared with the frontend (``web/lib/types.ts``)."""

    # upload validation (sync, before 202)
    UNSUPPORTED_FORMAT = "unsupported_format"
    FILE_TOO_LARGE = "file_too_large"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    DECODE_FAILED = "decode_failed"
    INVALID_REQUEST = "invalid_request"
    # pipeline (async, via job status)
    NO_MOTION_DETECTED = "no_motion_detected"
    UNSUPPORTED_MOTION = "unsupported_motion"
    NO_STABLE_STATE = "no_stable_state"
    INTERNAL_ERROR = "internal_error"
    INTERRUPTED = "interrupted"
    # routing
    NOT_FOUND = "not_found"
    NOT_READY = "not_ready"
    ALREADY_RUNNING = "already_running"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.TOO_LONG: 422,
    ErrorCode.TOO_SHORT: 422,
    ErrorCode.DECODE_FAILED: 422,
    ErrorCode.INVALID_REQUEST: 422,
    ErrorCode.NO_MOTION_DETECTED: 422,
    ErrorCode.UNSUPPORTED_MOTION: 422,
    ErrorCode.NO_STABLE_STATE: 422,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.INTERRUPTED: 500,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.NOT_READY: 409,
    ErrorCode.ALREADY_RUNNING: 409,
}

#: Default user-facing message per code. Callers may pass a more specific message.
#: Tone: say what happened, then what to do (PRD §27 recording guidelines). One or two short
#: sentences, no internals (ffmpeg stderr, paths). ``web/lib/format.ts`` ``describeError`` adds
#: the title and guidance bullets; keep both in step.
DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_FORMAT: "Unsupported file type. Upload an MP4, MOV, M4V or WebM video.",
    ErrorCode.FILE_TOO_LARGE: (
        "The file is larger than the upload limit. Trim it or export it at a lower resolution."
    ),
    ErrorCode.TOO_LONG: "The recording is too long. Trim it to just the interaction.",
    ErrorCode.TOO_SHORT: (
        "The recording is too short. Wait about 1 second before interacting and keep recording "
        "until the animation has finished."
    ),
    ErrorCode.DECODE_FAILED: (
        "The video could not be decoded. Re-export it as an H.264 MP4 and upload it again."
    ),
    ErrorCode.INVALID_REQUEST: "The request is invalid.",
    ErrorCode.NO_MOTION_DETECTED: (
        "No UI motion was found in the recording. Make sure the interaction and its animation "
        "happen on screen while recording."
    ),
    ErrorCode.UNSUPPORTED_MOTION: (
        "The whole page moves (scrolling or a page transition), which is not supported yet. "
        "Record a single component interaction without scrolling."
    ),
    ErrorCode.NO_STABLE_STATE: (
        "The UI is already moving when the recording starts. Start recording, wait about 1 "
        "second without touching anything, then interact."
    ),
    ErrorCode.INTERNAL_ERROR: (
        "Something went wrong while analyzing the recording. Try again; if it keeps failing, "
        "try another recording."
    ),
    ErrorCode.INTERRUPTED: (
        "Processing stopped because the server restarted. Upload the video again."
    ),
    ErrorCode.NOT_FOUND: "This job does not exist. It may have been deleted.",
    ErrorCode.NOT_READY: "The result is not ready yet.",
    ErrorCode.ALREADY_RUNNING: "This job is still being processed. Wait until it finishes.",
}


#: ``not_ready`` message when a result cannot be re-labeled (``measurement.json`` missing/old).
RERUN_UNAVAILABLE_MESSAGE = (
    "This result cannot be re-labeled because its stored measurements are missing or from an "
    "older version. Upload the video again."
)


class PipelineError(Exception):
    """Expected, user-explainable failure. Anything else is reported as ``internal_error``."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message or DEFAULT_MESSAGES[self.code]
        self.http_status = http_status or ERROR_HTTP_STATUS[self.code]
        super().__init__(f"{self.code}: {self.message}")

    def to_body(self) -> dict[str, dict[str, str]]:
        """Serialized error body ``{"error": {"code", "message"}}`` (PLAN §5)."""
        return {"error": {"code": self.code.value, "message": self.message}}
