/**
 * Typed client for the Mimic FastAPI backend (PLAN §5, P7; PLAN-auth §8.2).
 *
 * The browser calls the API directly at `NEXT_PUBLIC_API_BASE_URL` (no Next rewrites), so uploads
 * keep XHR progress and avoid proxy body limits. With `NEXT_PUBLIC_API_MOCK=1` every function is
 * routed to `mock/` instead (fake lifecycle, fake accounts, the contract fixture), see `lib/README.md`.
 *
 * Auth: the API keeps the session in an HttpOnly cookie (`mimic_session`) that JS never sees, so
 * every request (fetch and the XHR upload) is sent with credentials. Nothing auth-related is
 * stored in localStorage/sessionStorage. A 401 `unauthenticated` or 403 `password_change_required`
 * from any call is broadcast through `onAuthEvent` (the session store in `session.ts` listens).
 *
 * Every function rejects with `ApiError` (never a raw TypeError/Response).
 */

import { ApiError } from "./errors";
import { normalizeResult } from "./spec";
import type { MockScenario } from "./mock/mockApi";
import type {
  AdminUser,
  AdminUserList,
  AuthStatus,
  ChangePasswordRequest,
  CreateUserRequest,
  Health,
  InterpreterCheck,
  InterpreterCheckError,
  InterpreterProvider,
  InterpretRequest,
  JobCreated,
  JobList,
  JobState,
  JobStatus,
  LoginRequest,
  Me,
  PixelRatioOption,
  ResultEnvelope,
  TemporaryPassword,
  UpdateUserRequest,
  UserRole,
} from "./types";

export { ApiError } from "./errors";
export type { ApiErrorCode, ApiErrorOrigin, ClientErrorCode } from "./errors";

/* ---------- configuration (NEXT_PUBLIC_* are inlined at build time; keep the direct references) ---------- */

/** Backend origin without trailing slash. Default matches `make dev-api`. */
export const API_BASE_URL: string = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

/** `NEXT_PUBLIC_API_MOCK=1` (or `true`) routes the whole client to the in-browser mock. */
export const IS_MOCK_API: boolean = /^(1|true)$/i.test(process.env.NEXT_PUBLIC_API_MOCK ?? "");

// Loaded lazily so the mock + fixture only end up in the bundle when mock mode is on.
const loadMock = () => import("./mock/mockApi");
const loadMockAuth = () => import("./mock/mockAuth");

/* ---------- auth events ---------- */

/**
 * - `unauthenticated`: some request got 401 `unauthenticated` (no session, expired, revoked by an
 *   admin, user disabled/deleted). The session store switches to `anonymous`.
 * - `password_change_required`: some request got 403 `password_change_required`. The session
 *   store re-reads `/api/auth/me` so routing sends the user to `/account/password`.
 */
export type AuthEvent = "unauthenticated" | "password_change_required";

const authListeners = new Set<(event: AuthEvent, error: ApiError) => void>();

/** Subscribe to auth events from every API call. Returns the unsubscribe function. */
export function onAuthEvent(listener: (event: AuthEvent, error: ApiError) => void): () => void {
  authListeners.add(listener);
  return () => {
    authListeners.delete(listener);
  };
}

/** Broadcast an auth event (called by the client itself; exported for tests). */
export function notifyAuthEvent(event: AuthEvent, error: ApiError): void {
  for (const listener of [...authListeners]) {
    try {
      listener(event, error);
    } catch {
      // A broken listener must not turn into a rejected API call.
    }
  }
}

function reportAuthError(err: ApiError): void {
  if (err.code === "unauthenticated") notifyAuthEvent("unauthenticated", err);
  else if (err.code === "password_change_required") notifyAuthEvent("password_change_required", err);
}

/**
 * Run the mock or the real implementation, normalize any rejection to `ApiError` and report auth
 * failures. Every exported endpoint goes through here, so fetch, XHR and mock behave the same.
 */
async function call<T>(mock: () => Promise<T>, real: () => Promise<T>): Promise<T> {
  try {
    return await (IS_MOCK_API ? mock() : real());
  } catch (err) {
    const e = ApiError.from(err);
    reportAuthError(e);
    throw e;
  }
}

/** Hostname of `API_BASE_URL` (null if it can't be parsed). */
export function apiHostname(): string | null {
  try {
    return new URL(API_BASE_URL).hostname;
  } catch {
    return null;
  }
}

/**
 * The session cookie is host-only and set by the API host, so the page and the API must use the
 * same hostname (ports may differ). Returns both hostnames when they differ (the login page can
 * warn before the user even tries), or null when they match, in mock mode, or outside a browser.
 */
export function hostnameMismatch(pageHostname: string | null = currentPageHostname()): { pageHost: string; apiHost: string } | null {
  if (IS_MOCK_API) return null;
  const apiHost = apiHostname();
  if (!pageHostname || !apiHost) return null;
  return pageHostname.toLowerCase() === apiHost.toLowerCase() ? null : { pageHost: pageHostname, apiHost };
}

function currentPageHostname(): string | null {
  return typeof window !== "undefined" && window.location ? window.location.hostname : null;
}

/* ---------- upload options ---------- */

/** Form options for `POST /api/jobs`. */
export interface UploadOptions {
  /** Display scale of the recording; `auto` lets the backend guess (PLAN P4). */
  pixelRatio: PixelRatioOption;
  /** Send keyframes to Claude for labeling. Opt-in, default OFF (PLAN P9, org privacy policy). */
  useInterpreter: boolean;
}

export const DEFAULT_UPLOAD_OPTIONS: Readonly<UploadOptions> = Object.freeze({
  pixelRatio: "auto",
  useInterpreter: false,
});

export interface UploadProgress {
  /** Bytes sent so far. */
  loaded: number;
  /** Total bytes, null if the browser can't tell. */
  total: number | null;
  /** 0..1, null if `total` is unknown. */
  fraction: number | null;
}

export interface UploadRequestOptions {
  signal?: AbortSignal;
  /**
   * Mock mode only: force a scenario. If omitted the mock reads `?fail=<code>` / `?scenario=<name>`
   * from the current page URL. Ignored against the real API.
   */
  mockScenario?: MockScenario;
}

/* ---------- helpers ---------- */

function jobPath(id: string): string {
  return `/api/jobs/${encodeURIComponent(id)}`;
}

function userPath(id: string): string {
  return `/api/admin/users/${encodeURIComponent(id)}`;
}

/**
 * Resolve an API-relative artifact URL (`/api/jobs/<id>/video`, keyframe URLs) to something the
 * browser can load. Absolute, `data:` and `blob:` URLs pass through unchanged.
 *
 * `<video>`/`<img>` without a `crossorigin` attribute send the session cookie on their own
 * (same-site, SameSite=Lax), so these URLs work for signed-in users. Don't add `crossorigin`.
 */
export function assetUrl(path: string): string;
export function assetUrl(path: string | null | undefined): string | null;
export function assetUrl(path: string | null | undefined): string | null {
  if (path == null || path === "") return null;
  if (/^(https?:|data:|blob:)/i.test(path)) return path;
  if (IS_MOCK_API) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function readJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (text === "") return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

type HttpMethod = "GET" | "POST" | "PATCH" | "DELETE";

interface RequestOptions {
  method?: HttpMethod;
  /** JSON-encoded when present. */
  json?: unknown;
  signal?: AbortSignal;
  /** Response body check; omit for 204 endpoints (the body is ignored). */
  validate?: (body: unknown) => boolean;
}

/**
 * `fetch` with the session cookie (`credentials: "include"`), JSON in/out, no caching.
 * Resolves `undefined` for 204 (and for any 2xx when no `validate` is given).
 */
async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", json, signal, validate } = options;
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: json === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      body: json === undefined ? undefined : JSON.stringify(json),
      credentials: "include",
      cache: "no-store",
      signal,
    });
  } catch (err) {
    throw ApiError.from(err);
  }
  if (res.ok && (res.status === 204 || !validate)) {
    // Drain the body so the connection can be reused; its content doesn't matter.
    try {
      await res.text();
    } catch {
      // ignore
    }
    return undefined as T;
  }
  let body: unknown;
  try {
    body = await readJson(res);
  } catch (err) {
    throw ApiError.from(err);
  }
  if (!res.ok) throw ApiError.fromResponse(res.status, body, { retryAfter: res.headers.get("Retry-After") });
  if (validate && !validate(body)) throw ApiError.invalidResponse();
  return body as T;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
// `interpreter` / `limits` are null for anonymous callers (PLAN-auth A12).
const looksLikeHealth = (b: unknown) =>
  isObject(b) && b.status === "ok" && (b.interpreter == null || isObject(b.interpreter)) && (b.limits == null || isObject(b.limits));
const looksLikeJobStatus = (b: unknown) => isObject(b) && typeof b.id === "string" && typeof b.status === "string";
const looksLikeJobCreated = looksLikeJobStatus;
const looksLikeResult = (b: unknown) => isObject(b) && isObject(b.spec) && isObject(b.outputs) && isObject(b.artifacts);
const looksLikeInterpreterCheck = (b: unknown) => isObject(b) && typeof b.ok === "boolean" && typeof b.mode === "string";
const looksLikeAuthStatus = (b: unknown) => isObject(b) && typeof b.setup_required === "boolean";
const looksLikeMe = (b: unknown) =>
  isObject(b) && typeof b.id === "string" && typeof b.username === "string" && typeof b.role === "string" && typeof b.must_change_password === "boolean";
const looksLikeJobList = (b: unknown) =>
  isObject(b) && Array.isArray(b.items) && b.items.every(looksLikeJobStatus) && (b.next_cursor === null || typeof b.next_cursor === "string");
const looksLikeAdminUser = (b: unknown) =>
  isObject(b) && typeof b.id === "string" && typeof b.username === "string" && typeof b.is_active === "boolean" && typeof b.job_count === "number";
const looksLikeAdminUserList = (b: unknown) => isObject(b) && Array.isArray(b.items) && b.items.every(looksLikeAdminUser);
const looksLikeTemporaryPassword = (b: unknown) => isObject(b) && looksLikeAdminUser(b.user) && typeof b.temporary_password === "string";

/* ---------- interpreter connection check ---------- */
// The response type is generated from the backend (`InterpreterCheck` in ./types). The aliases
// below keep the names the UI imports from `@/lib/api`.

export type { InterpreterCheck };

/** Layer B backend configured server-side (env only — no credentials ever reach the browser). */
export type InterpreterMode = InterpreterProvider;

/** How the configured model returns structured output, as detected by the check. */
export type StructuredOutputMode = NonNullable<InterpreterCheck["structured_output"]>;

export type InterpreterCheckErrorCode = InterpreterCheckError["code"];

/* ---------- health + interpreter ---------- */

/**
 * `GET /api/health` (public). `interpreter` and `limits` are null for anonymous callers and while
 * a forced password change is pending; a full session gets everything.
 */
export async function getHealth(signal?: AbortSignal): Promise<Health> {
  return call(
    async () => (await loadMock()).mockGetHealth(signal),
    () => request<Health>("/api/health", { signal, validate: looksLikeHealth }),
  );
}

/**
 * `POST /api/interpreter/check` — **admin only** (403 `forbidden` for users). Actively tests the
 * server-configured interpreter (the "Test connection" button). A failed check resolves with
 * `ok: false` + `error`; only transport/HTTP problems reject. Can take several seconds.
 */
export async function checkInterpreter(signal?: AbortSignal): Promise<InterpreterCheck> {
  return call(
    async () => (await loadMock()).mockCheckInterpreter(signal),
    () => request<InterpreterCheck>("/api/interpreter/check", { method: "POST", signal, validate: looksLikeInterpreterCheck }),
  );
}

/* ---------- auth ---------- */

/** `GET /api/auth/status` (public) — `setup_required: true` until `make create-admin` ran. */
export async function getAuthStatus(signal?: AbortSignal): Promise<AuthStatus> {
  return call(
    async () => (await loadMockAuth()).mockGetAuthStatus(signal),
    () => request<AuthStatus>("/api/auth/status", { signal, validate: looksLikeAuthStatus }),
  );
}

/**
 * `POST /api/auth/login` — sets the session cookie and resolves the account. Prefer `login()` from
 * `session.ts`, which also verifies the cookie stuck (`cookie_rejected`) and updates the store.
 *
 * Rejects with `invalid_credentials` (401), `account_disabled` (403, only after a correct
 * password), `setup_required` (409), `too_many_attempts` (429, `retryAfterS`), `invalid_request`
 * (422, empty fields), `origin_not_allowed` (403).
 */
export async function login(username: string, password: string, signal?: AbortSignal): Promise<Me> {
  const body: LoginRequest = { username, password };
  return call(
    async () => (await loadMockAuth()).mockLogin(username, password, signal),
    () => request<Me>("/api/auth/login", { method: "POST", json: body, signal, validate: looksLikeMe }),
  );
}

/** `POST /api/auth/logout` → 204. Idempotent (also 204 without a session). Clears the cookie. */
export async function logout(signal?: AbortSignal): Promise<void> {
  return call(
    async () => (await loadMockAuth()).mockLogout(signal),
    () => request<void>("/api/auth/logout", { method: "POST", signal }),
  );
}

/**
 * `GET /api/auth/me` — the signed-in account (also while a forced password change is pending,
 * see `must_change_password`). 401 `unauthenticated` without a valid session.
 */
export async function getMe(signal?: AbortSignal): Promise<Me> {
  return call(
    async () => (await loadMockAuth()).mockGetMe(signal),
    () => request<Me>("/api/auth/me", { signal, validate: looksLikeMe }),
  );
}

/**
 * `POST /api/auth/password` — change the own password (also the forced change). Resolves the
 * account with `must_change_password: false`; the server revokes every session of the account and
 * sets a **new** session cookie for this browser.
 *
 * Rejects with `current_password_incorrect` (422), `weak_password` (422, `message` names the rule
 * that failed), `too_many_attempts` (429), `unauthenticated` (401).
 */
export async function changePassword(currentPassword: string, newPassword: string, signal?: AbortSignal): Promise<Me> {
  const body: ChangePasswordRequest = { current_password: currentPassword, new_password: newPassword };
  return call(
    async () => (await loadMockAuth()).mockChangePassword(currentPassword, newPassword, signal),
    () => request<Me>("/api/auth/password", { method: "POST", json: body, signal, validate: looksLikeMe }),
  );
}

/* ---------- admin: users ---------- */

/** `GET /api/admin/users` — every account, sorted by username. Admin only (403 `forbidden`). */
export async function listUsers(signal?: AbortSignal): Promise<AdminUserList> {
  return call(
    async () => (await loadMockAuth()).mockListUsers(signal),
    () => request<AdminUserList>("/api/admin/users", { signal, validate: looksLikeAdminUserList }),
  );
}

export interface CreateUserInput {
  /** Stripped + lowercased by the server, then checked against `USERNAME_PATTERN`. */
  username: string;
  role?: UserRole;
}

/**
 * `POST /api/admin/users` → 201 with a server-generated **temporary password, shown once**. The
 * account must change it at first sign-in. Rejects with `username_taken` (409), `invalid_request`
 * (422, bad username).
 */
export async function createUser(input: CreateUserInput, signal?: AbortSignal): Promise<TemporaryPassword> {
  const body: CreateUserRequest = { username: input.username, role: input.role ?? "user" };
  return call(
    async () => (await loadMockAuth()).mockCreateUser(body, signal),
    () => request<TemporaryPassword>("/api/admin/users", { method: "POST", json: body, signal, validate: looksLikeTemporaryPassword }),
  );
}

export interface UpdateUserInput {
  role?: UserRole;
  isActive?: boolean;
}

/**
 * `PATCH /api/admin/users/{id}` — change role and/or enable/disable (at least one field). A role
 * change or a disable revokes all of the target's sessions (yours too, if you change your own
 * role). Rejects with `last_admin` (409), `self_action_forbidden` (409, disabling yourself),
 * `not_found` (404).
 */
export async function updateUser(id: string, input: UpdateUserInput, signal?: AbortSignal): Promise<AdminUser> {
  const body: UpdateUserRequest = {};
  if (input.role !== undefined) body.role = input.role;
  if (input.isActive !== undefined) body.is_active = input.isActive;
  return call(
    async () => (await loadMockAuth()).mockUpdateUser(id, body, signal),
    () => request<AdminUser>(userPath(id), { method: "PATCH", json: body, signal, validate: looksLikeAdminUser }),
  );
}

/**
 * `POST /api/admin/users/{id}/reset-password` — new server-generated temporary password (shown
 * once), forces a change at next sign-in and revokes the target's sessions. Rejects with
 * `self_action_forbidden` (409, use `changePassword` for yourself), `not_found` (404).
 */
export async function resetUserPassword(id: string, signal?: AbortSignal): Promise<TemporaryPassword> {
  return call(
    async () => (await loadMockAuth()).mockResetUserPassword(id, signal),
    () => request<TemporaryPassword>(`${userPath(id)}/reset-password`, { method: "POST", signal, validate: looksLikeTemporaryPassword }),
  );
}

/**
 * `DELETE /api/admin/users/{id}` → 204. Deletes the account, its sessions and **all its jobs and
 * files**. Rejects with `last_admin` (409), `self_action_forbidden` (409), `not_found` (404).
 */
export async function deleteUser(id: string, signal?: AbortSignal): Promise<void> {
  return call(
    async () => (await loadMockAuth()).mockDeleteUser(id, signal),
    () => request<void>(userPath(id), { method: "DELETE", signal }),
  );
}

/* ---------- jobs ---------- */

export interface ListJobsOptions {
  /**
   * `"me"` = own jobs. A user id = that user's jobs (admins only; 403 `forbidden` otherwise).
   * Omitted: users get their own jobs, admins get every job (incl. legacy `owner: null` ones).
   */
  owner?: "me" | (string & {});
  status?: JobState;
  /** 1..200, default 50 (server). */
  limit?: number;
  /** `next_cursor` of the previous page. */
  cursor?: string | null;
  signal?: AbortSignal;
}

/** `GET /api/jobs` — newest first, keyset-paginated (`next_cursor` null on the last page). */
export async function listJobs(options: ListJobsOptions = {}): Promise<JobList> {
  const { owner, status, limit, cursor, signal } = options;
  return call(
    async () => (await loadMock()).mockListJobs({ owner, status, limit, cursor }, signal),
    () => {
      const params = new URLSearchParams();
      if (owner) params.set("owner", owner);
      if (status) params.set("status", status);
      if (limit != null) params.set("limit", String(limit));
      if (cursor) params.set("cursor", cursor);
      const qs = params.toString();
      return request<JobList>(`/api/jobs${qs ? `?${qs}` : ""}`, { signal, validate: looksLikeJobList });
    },
  );
}

/** `GET /api/jobs/{id}` — poll until `status` is `succeeded` or `failed` (prefer `useJobPolling`). */
export async function getJob(id: string, signal?: AbortSignal): Promise<JobStatus> {
  return call(
    async () => (await loadMock()).mockGetJob(id, signal),
    () => request<JobStatus>(jobPath(id), { signal, validate: looksLikeJobStatus }),
  );
}

/** `GET /api/jobs/{id}/result` — rejects with code `not_ready` (409) until the job succeeded. */
export async function getResult(id: string, signal?: AbortSignal): Promise<ResultEnvelope> {
  // stored schema 0.1 results come back verbatim without `spec.mode`: fill it in ("transition")
  const env = await call(
    async () => (await loadMock()).mockGetResult(id, signal),
    () => request<ResultEnvelope>(`${jobPath(id)}/result`, { signal, validate: looksLikeResult }),
  );
  return normalizeResult(env);
}

/**
 * `DELETE /api/jobs/{id}` → 204. Owner or admin; anyone else gets 404 `not_found` (same rule as
 * reads). Works on queued/running jobs too (the server stops writing to it and removes its files).
 * Afterwards every read of the job is 404.
 */
export async function deleteJob(jobId: string, signal?: AbortSignal): Promise<void> {
  return call(
    async () => (await loadMock()).mockDeleteJob(jobId, signal),
    () => request<void>(jobPath(jobId), { method: "DELETE", signal }),
  );
}

export interface RerunInterpretationOptions {
  /**
   * `true` (default) = send this job's keyframes to the server-configured AI interpreter — the
   * explicit opt-in (PLAN P9), so only call it from a user action. `false` = back to heuristic labels.
   */
  useInterpreter?: boolean;
  signal?: AbortSignal;
}

/**
 * `POST /api/jobs/{id}/interpret` — re-run AI labeling (+ IR assembly + the four outputs) on a
 * **succeeded** job. Measurements are reused; only labels, roles, hierarchy, interaction type and
 * the generated text can change. Resolves with the job's `JobStatus` (`queued`); the job then goes
 * `interpreting` → `generating` → `succeeded`, so restart polling (`useJobPolling(...).restart()`)
 * **after** this resolves.
 *
 * Rejects with `already_running` (409) while the job is queued/processing, `not_ready` (409) for a
 * failed job or a result made by an older version (message says "Upload the video again"),
 * `not_found` (404). If the re-run itself fails, the job returns to `succeeded` with the previous
 * result and `JobStatus.error` set (see lib/README.md "Re-run interpretation").
 */
export async function rerunInterpretation(jobId: string, options: RerunInterpretationOptions = {}): Promise<JobStatus> {
  const useInterpreter = options.useInterpreter ?? true;
  const body: InterpretRequest = { use_interpreter: useInterpreter };
  return call(
    async () => (await loadMock()).mockRerunInterpretation(jobId, useInterpreter, options.signal),
    () => request<JobStatus>(`${jobPath(jobId)}/interpret`, { method: "POST", json: body, signal: options.signal, validate: looksLikeJobStatus }),
  );
}

/**
 * `POST /api/jobs` (multipart) via XHR so upload progress is observable. Sent with the session
 * cookie (`withCredentials`); the job is owned by the signed-in user.
 *
 * Resolves with `JobCreated` on 202. Validation failures reject with the backend code
 * (`unsupported_format`, `file_too_large`, `too_long`, `too_short`, `decode_failed`, `invalid_request`).
 */
export async function uploadVideo(
  file: File,
  options: Partial<UploadOptions> = {},
  onProgress?: (progress: UploadProgress) => void,
  request: UploadRequestOptions = {},
): Promise<JobCreated> {
  const opts: UploadOptions = { ...DEFAULT_UPLOAD_OPTIONS, ...options };
  return call(
    async () => (await loadMock()).mockUploadVideo(file, opts, onProgress, request),
    () => xhrUpload(file, opts, onProgress, request.signal),
  );
}

function xhrUpload(
  file: File,
  opts: UploadOptions,
  onProgress: ((progress: UploadProgress) => void) | undefined,
  signal: AbortSignal | undefined,
): Promise<JobCreated> {
  return new Promise<JobCreated>((resolve, reject) => {
    if (signal?.aborted) {
      reject(ApiError.aborted());
      return;
    }

    const form = new FormData();
    form.append("file", file, file.name);
    form.append("pixel_ratio", opts.pixelRatio);
    form.append("use_interpreter", opts.useInterpreter ? "true" : "false");

    const xhr = new XMLHttpRequest();
    const onAbortSignal = () => xhr.abort();
    const cleanup = () => signal?.removeEventListener("abort", onAbortSignal);

    xhr.open("POST", `${API_BASE_URL}/api/jobs`);
    // Send the session cookie (cross-origin localhost:3000 → localhost:8000 is same-site).
    xhr.withCredentials = true;
    xhr.setRequestHeader("Accept", "application/json");
    xhr.responseType = "text";

    xhr.upload.onprogress = (e) => {
      const total = e.lengthComputable ? e.total : null;
      onProgress?.({ loaded: e.loaded, total, fraction: total ? Math.min(1, e.loaded / total) : null });
    };
    xhr.onload = () => {
      cleanup();
      let body: unknown;
      try {
        body = xhr.responseText ? (JSON.parse(xhr.responseText) as unknown) : null;
      } catch {
        body = undefined;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (looksLikeJobCreated(body)) resolve(body as JobCreated);
        else reject(ApiError.invalidResponse());
        return;
      }
      reject(ApiError.fromResponse(xhr.status, body, { retryAfter: xhr.getResponseHeader("Retry-After") }));
    };
    xhr.onerror = () => {
      cleanup();
      reject(ApiError.network());
    };
    xhr.ontimeout = () => {
      cleanup();
      reject(ApiError.network("The upload timed out."));
    };
    xhr.onabort = () => {
      cleanup();
      reject(ApiError.aborted());
    };

    signal?.addEventListener("abort", onAbortSignal, { once: true });
    xhr.send(form);
  });
}

/** The subset used by the job poller; injectable for tests. */
export interface JobClient {
  getJob(id: string, signal?: AbortSignal): Promise<JobStatus>;
  getResult(id: string, signal?: AbortSignal): Promise<ResultEnvelope>;
}

export const defaultJobClient: JobClient = { getJob, getResult };
