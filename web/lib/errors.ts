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

/**
 * Codes that only the client produces (never sent by the backend).
 *
 * - `cookie_rejected`: login succeeded but the very next `GET /api/auth/me` was 401, i.e. the
 *   browser did not keep or send the session cookie. Almost always a hostname mismatch
 *   (web on `127.0.0.1`, API on `localhost` or the other way round). See `session.ts`.
 */
export type ClientErrorCode = "network_error" | "aborted" | "invalid_response" | "cookie_rejected";

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
  unauthenticated: true,
  invalid_credentials: true,
  account_disabled: true,
  forbidden: true,
  password_change_required: true,
  origin_not_allowed: true,
  too_many_attempts: true,
  setup_required: true,
  last_admin: true,
  self_action_forbidden: true,
  username_taken: true,
  weak_password: true,
  current_password_incorrect: true,
};
const BACKEND_CODES: ReadonlySet<string> = new Set(Object.keys(BACKEND_CODE_SET));

/** True if `code` is one of the backend's closed set of error codes. */
export function isBackendErrorCode(code: unknown): code is ErrorCode {
  return typeof code === "string" && BACKEND_CODES.has(code);
}

/** Fallback code when a response has no parsable error body. */
function codeForStatus(status: number): ApiErrorCode {
  switch (status) {
    case 401:
      return "unauthenticated";
    case 403:
      return "forbidden";
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
    case 429:
      return "too_many_attempts";
    default:
      return "internal_error";
  }
}

/**
 * Parse a `Retry-After` header: delay-seconds or an HTTP-date. Returns whole seconds (>= 0), or
 * null if absent/unparsable. `now` is injectable for tests.
 */
export function parseRetryAfter(value: string | null | undefined, now: number = Date.now()): number | null {
  if (value == null) return null;
  const v = value.trim();
  if (v === "") return null;
  if (/^\d+$/.test(v)) return Number(v);
  const at = Date.parse(v);
  if (Number.isNaN(at)) return null;
  return Math.max(0, Math.ceil((at - now) / 1000));
}

export interface ApiErrorOptions {
  status?: number | null;
  origin?: ApiErrorOrigin;
  /** Seconds from `Retry-After` (`too_many_attempts`). */
  retryAfterS?: number | null;
  /** The backend's own `error.message`, when there was one. */
  serverMessage?: string | null;
}

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  /** HTTP status, or null for job failures and transport errors. */
  readonly status: number | null;
  readonly origin: ApiErrorOrigin;
  /**
   * Seconds to wait before retrying, from the `Retry-After` response header (set on 429
   * `too_many_attempts`). Null when the header is absent or the browser can't read it (the API
   * must list it in `Access-Control-Expose-Headers` for cross-origin reads).
   */
  readonly retryAfterS: number | null;
  /**
   * The message the backend sent (`{"error": {"message"}}` or `JobStatus.error.message`), or null
   * if it sent none and `message` is a generic fallback. Server messages are specific (e.g. the
   * exact `weak_password` rule that failed), so prefer them in the UI (`describeApiError`).
   */
  readonly serverMessage: string | null;

  constructor(code: ApiErrorCode, message: string, options: ApiErrorOptions = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = options.status ?? null;
    this.origin = options.origin ?? (options.status != null ? "http" : "client");
    this.retryAfterS = options.retryAfterS ?? null;
    this.serverMessage = options.serverMessage ?? null;
  }

  /**
   * Worth retrying automatically (polling backoff): transport errors and 5xx / 408.
   * 4xx errors (incl. every auth/admin code and 429 `too_many_attempts`, which must not be
   * hammered) and job failures are final.
   */
  get isTransient(): boolean {
    if (this.origin === "network") return true;
    if (this.origin !== "http" || this.status == null) return false;
    return this.status >= 500 || this.status === 408;
  }

  /**
   * Build from an HTTP status + (maybe) parsed JSON body. Never throws.
   * `retryAfter` is the raw `Retry-After` header value, if any.
   */
  static fromResponse(status: number, body: unknown, options: { retryAfter?: string | null } = {}): ApiError {
    const detail = (body as { error?: Partial<ErrorDetail> } | null | undefined)?.error;
    const code = isBackendErrorCode(detail?.code) ? detail.code : codeForStatus(status);
    const serverMessage = typeof detail?.message === "string" && detail.message.length > 0 ? detail.message : null;
    return new ApiError(code, serverMessage ?? `Request failed (HTTP ${status}).`, {
      status,
      origin: "http",
      retryAfterS: parseRetryAfter(options.retryAfter),
      serverMessage,
    });
  }

  /** Build from a failed job's `JobStatus.error`. */
  static fromJob(detail: ErrorDetail | null): ApiError {
    if (!detail) return new ApiError("internal_error", "The job failed without an error description.", { origin: "job" });
    return new ApiError(detail.code, detail.message, { origin: "job", serverMessage: detail.message });
  }

  /**
   * Login succeeded but the session cookie did not come back (`GET /api/auth/me` → 401 right
   * after). `pageHost` / `apiHost` make the message concrete when known. The message states only
   * what happened; the advice ("use the same hostname") is the `cookie_rejected` guidance in
   * `format.ts`, so `describeApiError` never shows it twice.
   */
  static cookieRejected(hosts: { pageHost?: string | null; apiHost?: string | null } = {}): ApiError {
    const { pageHost, apiHost } = hosts;
    const where =
      pageHost && apiHost && pageHost !== apiHost
        ? ` This page is on "${pageHost}" but the API is on "${apiHost}".`
        : "";
    return new ApiError(
      "cookie_rejected",
      `You signed in, but the browser did not keep the session cookie.${where}`,
      { origin: "client" },
    );
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
