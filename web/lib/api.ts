/**
 * Typed client for the Mimic FastAPI backend (PLAN §5, P7).
 *
 * The browser calls the API directly at `NEXT_PUBLIC_API_BASE_URL` (no Next rewrites), so uploads
 * keep XHR progress and avoid proxy body limits. With `NEXT_PUBLIC_API_MOCK=1` every function is
 * routed to `mock/mockApi.ts` instead (fake lifecycle + the contract fixture), see `lib/README.md`.
 *
 * Every function rejects with `ApiError` (never a raw TypeError/Response).
 */

import { ApiError } from "./errors";
import type { MockScenario } from "./mock/mockApi";
import type {
  Health,
  InterpreterCheck,
  InterpreterCheckError,
  InterpreterProvider,
  InterpretRequest,
  JobCreated,
  JobStatus,
  PixelRatioOption,
  ResultEnvelope,
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

/**
 * Resolve an API-relative artifact URL (`/api/jobs/<id>/video`, keyframe URLs) to something the
 * browser can load. Absolute, `data:` and `blob:` URLs pass through unchanged.
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

async function requestJson<T>(
  path: string,
  signal: AbortSignal | undefined,
  isValid: (body: unknown) => boolean,
  method: "GET" | "POST" = "GET",
  jsonBody?: unknown,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers:
        jsonBody === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      body: jsonBody === undefined ? undefined : JSON.stringify(jsonBody),
      cache: "no-store",
      signal,
    });
  } catch (err) {
    throw ApiError.from(err);
  }
  let body: unknown;
  try {
    body = await readJson(res);
  } catch (err) {
    throw ApiError.from(err);
  }
  if (!res.ok) throw ApiError.fromResponse(res.status, body);
  if (!isValid(body)) throw ApiError.invalidResponse();
  return body as T;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const looksLikeHealth = (b: unknown) => isObject(b) && b.status === "ok" && isObject(b.interpreter);
const looksLikeJobStatus = (b: unknown) => isObject(b) && typeof b.id === "string" && typeof b.status === "string";
const looksLikeJobCreated = looksLikeJobStatus;
const looksLikeResult = (b: unknown) => isObject(b) && isObject(b.spec) && isObject(b.outputs) && isObject(b.artifacts);
const looksLikeInterpreterCheck = (b: unknown) => isObject(b) && typeof b.ok === "boolean" && typeof b.mode === "string";

/* ---------- interpreter connection check ---------- */
// The response type is generated from the backend (`InterpreterCheck` in ./types). The aliases
// below keep the names the UI imports from `@/lib/api`.

export type { InterpreterCheck };

/** Layer B backend configured server-side (env only — no credentials ever reach the browser). */
export type InterpreterMode = InterpreterProvider;

/** How the configured model returns structured output, as detected by the check. */
export type StructuredOutputMode = NonNullable<InterpreterCheck["structured_output"]>;

export type InterpreterCheckErrorCode = InterpreterCheckError["code"];

/* ---------- endpoints ---------- */

/** `GET /api/health` — interpreter availability drives the "Use AI labeling" toggle. */
export async function getHealth(signal?: AbortSignal): Promise<Health> {
  if (IS_MOCK_API) return (await loadMock()).mockGetHealth(signal);
  return requestJson<Health>("/api/health", signal, looksLikeHealth);
}

/**
 * `POST /api/interpreter/check` — actively test the server-configured interpreter (the "Test
 * connection" button). A failed check resolves with `ok: false` + `error`; only transport/HTTP
 * problems reject. Can take several seconds (the backend calls the model).
 */
export async function checkInterpreter(signal?: AbortSignal): Promise<InterpreterCheck> {
  if (IS_MOCK_API) return (await loadMock()).mockCheckInterpreter(signal);
  return requestJson<InterpreterCheck>("/api/interpreter/check", signal, looksLikeInterpreterCheck, "POST");
}

/** `GET /api/jobs/{id}` — poll until `status` is `succeeded` or `failed` (prefer `useJobPolling`). */
export async function getJob(id: string, signal?: AbortSignal): Promise<JobStatus> {
  if (IS_MOCK_API) return (await loadMock()).mockGetJob(id, signal);
  return requestJson<JobStatus>(jobPath(id), signal, looksLikeJobStatus);
}

/** `GET /api/jobs/{id}/result` — rejects with code `not_ready` (409) until the job succeeded. */
export async function getResult(id: string, signal?: AbortSignal): Promise<ResultEnvelope> {
  if (IS_MOCK_API) return (await loadMock()).mockGetResult(id, signal);
  return requestJson<ResultEnvelope>(`${jobPath(id)}/result`, signal, looksLikeResult);
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
  if (IS_MOCK_API) return (await loadMock()).mockRerunInterpretation(jobId, useInterpreter, options.signal);
  const body: InterpretRequest = { use_interpreter: useInterpreter };
  return requestJson<JobStatus>(`${jobPath(jobId)}/interpret`, options.signal, looksLikeJobStatus, "POST", body);
}

/**
 * `POST /api/jobs` (multipart) via XHR so upload progress is observable.
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
  if (IS_MOCK_API) return (await loadMock()).mockUploadVideo(file, opts, onProgress, request);

  return new Promise<JobCreated>((resolve, reject) => {
    const { signal } = request;
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
      reject(ApiError.fromResponse(xhr.status, body));
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
