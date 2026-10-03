/**
 * Single error type for everything the data layer can fail with.
 *
 * - HTTP errors carry the backend's `{"error": {"code", "message"}}` body (PLAN §5).
 * - Job failures (`JobStatus.error`, pipeline-time codes) have `origin: "job"` and no HTTP status.
 * - Transport problems (offline, CORS, aborted, unparsable body) use client-only codes.
 *
 * Kept in its own module so `api.ts` and `mock/mockApi.ts` can both use it without a cycle.
 */

import type { ErrorCode, ErrorDetail } from "./types";

/** Codes that only the client produces (never sent by the backend). */
export type ClientErrorCode = "network_error" | "aborted" | "invalid_response";

export type ApiErrorCode = ErrorCode | ClientErrorCode;

/** Where the error came from. */
export type ApiErrorOrigin = "http" | "job" | "network" | "client";

// A Record keyed by ErrorCode, so adding a backend code without listing it here fails typecheck.
const BACKEND_CODE_SET: Record<ErrorCode, true> = {
  unsupported_format: true,
  file_too_large: true,
  too_long: true,
  too_short: true,
  decode_failed: true,
  invalid_request: true,
  no_motion_detected: true,
  unsupported_motion: true,
  no_stable_state: true,
  internal_error: true,
  interrupted: true,
  not_found: true,
  not_ready: true,
  already_running: true,
};
const BACKEND_CODES: ReadonlySet<string> = new Set(Object.keys(BACKEND_CODE_SET));

/** True if `code` is one of the backend's closed set of error codes. */
export function isBackendErrorCode(code: unknown): code is ErrorCode {
  return typeof code === "string" && BACKEND_CODES.has(code);
}

/** Fallback code when a response has no parsable error body. */
function codeForStatus(status: number): ApiErrorCode {
  switch (status) {
    case 404:
      return "not_found";
    case 409:
      return "not_ready";
    case 413:
      return "file_too_large";
    case 415:
      return "unsupported_format";
    case 422:
      return "invalid_request";
    default:
      return "internal_error";
  }
}

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  /** HTTP status, or null for job failures and transport errors. */
  readonly status: number | null;
  readonly origin: ApiErrorOrigin;

  constructor(code: ApiErrorCode, message: string, options: { status?: number | null; origin?: ApiErrorOrigin } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = options.status ?? null;
    this.origin = options.origin ?? (options.status != null ? "http" : "client");
  }

  /**
   * Worth retrying automatically (polling backoff): transport errors and 5xx / 408 / 429.
   * 4xx validation errors and job failures are final.
   */
  get isTransient(): boolean {
    if (this.origin === "network") return true;
    if (this.origin !== "http" || this.status == null) return false;
    return this.status >= 500 || this.status === 408 || this.status === 429;
  }

  /** Build from an HTTP status + (maybe) parsed JSON body. Never throws. */
  static fromResponse(status: number, body: unknown): ApiError {
    const detail = (body as { error?: Partial<ErrorDetail> } | null | undefined)?.error;
    const code = isBackendErrorCode(detail?.code) ? detail.code : codeForStatus(status);
    const message =
      typeof detail?.message === "string" && detail.message.length > 0 ? detail.message : `Request failed (HTTP ${status}).`;
    return new ApiError(code, message, { status, origin: "http" });
  }

  /** Build from a failed job's `JobStatus.error`. */
  static fromJob(detail: ErrorDetail | null): ApiError {
    if (!detail) return new ApiError("internal_error", "The job failed without an error description.", { origin: "job" });
    return new ApiError(detail.code, detail.message, { origin: "job" });
  }

  static network(message = "Could not reach the analysis server."): ApiError {
    return new ApiError("network_error", message, { origin: "network" });
  }

  static aborted(): ApiError {
    return new ApiError("aborted", "The request was cancelled.", { origin: "client" });
  }

  static invalidResponse(message = "The server returned an unexpected response."): ApiError {
    return new ApiError("invalid_response", message, { origin: "client" });
  }

  /** Normalize anything thrown (fetch TypeError, AbortError, ApiError) into an ApiError. */
  static from(err: unknown): ApiError {
    if (err instanceof ApiError) return err;
    if (err instanceof DOMException && err.name === "AbortError") return ApiError.aborted();
    if (err instanceof Error && err.name === "AbortError") return ApiError.aborted();
    if (err instanceof TypeError) return ApiError.network();
    return new ApiError("internal_error", err instanceof Error ? err.message : String(err), { origin: "client" });
  }
}
