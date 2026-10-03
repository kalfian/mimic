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
    # auth / accounts (PLAN-auth §3.2)
    UNAUTHENTICATED = "unauthenticated"
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_DISABLED = "account_disabled"
    FORBIDDEN = "forbidden"
    PASSWORD_CHANGE_REQUIRED = "password_change_required"
    ORIGIN_NOT_ALLOWED = "origin_not_allowed"
    TOO_MANY_ATTEMPTS = "too_many_attempts"
    SETUP_REQUIRED = "setup_required"
    LAST_ADMIN = "last_admin"
    SELF_ACTION_FORBIDDEN = "self_action_forbidden"
    USERNAME_TAKEN = "username_taken"
    WEAK_PASSWORD = "weak_password"
    CURRENT_PASSWORD_INCORRECT = "current_password_incorrect"


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
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.INVALID_CREDENTIALS: 401,
    ErrorCode.ACCOUNT_DISABLED: 403,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.PASSWORD_CHANGE_REQUIRED: 403,
    ErrorCode.ORIGIN_NOT_ALLOWED: 403,
    ErrorCode.TOO_MANY_ATTEMPTS: 429,
    ErrorCode.SETUP_REQUIRED: 409,
    ErrorCode.LAST_ADMIN: 409,
    ErrorCode.SELF_ACTION_FORBIDDEN: 409,
    ErrorCode.USERNAME_TAKEN: 409,
    ErrorCode.WEAK_PASSWORD: 422,
    # 422 on purpose: a 401 would read as "session expired" to the client.
    ErrorCode.CURRENT_PASSWORD_INCORRECT: 422,
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
    ErrorCode.UNAUTHENTICATED: ("You are not signed in, or your session expired. Sign in again."),
    ErrorCode.INVALID_CREDENTIALS: "Wrong username or password.",
    ErrorCode.ACCOUNT_DISABLED: "This account is disabled. Ask an admin to enable it.",
    ErrorCode.FORBIDDEN: "You don't have permission to do this.",
    ErrorCode.PASSWORD_CHANGE_REQUIRED: "Set a new password before continuing.",
    ErrorCode.ORIGIN_NOT_ALLOWED: (
        "This request came from a page that is not allowed to use this API."
    ),
    ErrorCode.TOO_MANY_ATTEMPTS: "Too many failed attempts. Wait a few minutes and try again.",
    ErrorCode.SETUP_REQUIRED: (
        "No admin account exists yet. On the API host, run `make create-admin`."
    ),
    ErrorCode.LAST_ADMIN: ("This is the last active admin. Make another user an admin first."),
    ErrorCode.SELF_ACTION_FORBIDDEN: "You can't do this to your own account.",
    ErrorCode.USERNAME_TAKEN: "That username is already taken.",
    ErrorCode.WEAK_PASSWORD: (
        "Passwords need at least 12 characters and must not match the username."
    ),
    ErrorCode.CURRENT_PASSWORD_INCORRECT: "The current password is wrong.",
}


#: ``not_ready`` message when a result cannot be re-labeled (``measurement.json`` missing/old).
RERUN_UNAVAILABLE_MESSAGE = (
    "This result cannot be re-labeled because its stored measurements are missing or from an "
    "older version. Upload the video again."
)


class PipelineError(Exception):
    """Expected, user-explainable failure. Anything else is reported as ``internal_error``.

    ``headers`` are extra response headers for HTTP-time errors (e.g. ``Retry-After`` on
    ``too_many_attempts``); the app's ``PipelineError`` handler copies them onto the response.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str | None = None,
        http_status: int | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message or DEFAULT_MESSAGES[self.code]
        self.http_status = http_status or ERROR_HTTP_STATUS[self.code]
        self.headers: dict[str, str] = dict(headers or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_body(self) -> dict[str, dict[str, str]]:
        """Serialized error body ``{"error": {"code", "message"}}`` (PLAN §5)."""
        return {"error": {"code": self.code.value, "message": self.message}}
