/**
 * Framework-free job poller: `GET /api/jobs/{id}` until terminal, then `GET /result` once.
 * `useJobPolling` is a thin React wrapper; scripts/tests can call `pollJob` directly.
 *
 * Cadence (PLAN §10.2): every 800 ms, 1.5 s once the job has been watched for 30 s.
 * Transient failures (network, 5xx) back off exponentially (1 s, 2 s, 4 s … ≤ 10 s) and give up
 * after 5 in a row. 4xx (e.g. 404 not_found) and job failures are final immediately.
 */

import { defaultJobClient, type JobClient } from "./api";
import { ApiError } from "./errors";
import type { JobStatus, ResultEnvelope } from "./types";

export type PollPhase =
  /** Before the first status response. */
  | "loading"
  /** Job queued/processing; polling. */
  | "polling"
  /** Job succeeded; fetching the result envelope. */
  | "fetching_result"
  /** Result available. Terminal. */
  | "succeeded"
  /** Job failed or a request failed permanently; see `error`. Terminal. */
  | "failed";

export interface JobPollState {
  phase: PollPhase;
  job: JobStatus | null;
  result: ResultEnvelope | null;
  /** Set only when `phase === "failed"`. `error.origin === "job"` → pipeline failure. */
  error: ApiError | null;
  /** Consecutive transient failures being retried right now (0 when healthy) → "reconnecting…". */
  retrying: number;
  /** Last transient error while `retrying > 0`. */
  lastTransientError: ApiError | null;
}

export interface PollOptions {
  client?: JobClient;
  intervalMs?: number;
  slowIntervalMs?: number;
  /** Switch to `slowIntervalMs` after watching this long. */
  slowAfterMs?: number;
  maxConsecutiveErrors?: number;
  backoffBaseMs?: number;
  backoffMaxMs?: number;
  signal?: AbortSignal;
  /** Called after every state change (never synchronously inside `pollJob`'s call). */
  onUpdate?: (state: JobPollState) => void;
}

export const POLL_DEFAULTS = {
  intervalMs: 800,
  slowIntervalMs: 1500,
  slowAfterMs: 30_000,
  maxConsecutiveErrors: 5,
  backoffBaseMs: 1000,
  backoffMaxMs: 10_000,
} as const;

export const INITIAL_POLL_STATE: JobPollState = Object.freeze({
  phase: "loading",
  job: null,
  result: null,
  error: null,
  retrying: 0,
  lastTransientError: null,
}) as JobPollState;

export function isTerminalPhase(phase: PollPhase): boolean {
  return phase === "succeeded" || phase === "failed";
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const timer = setTimeout(done, ms);
    function done() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", done);
      resolve();
    }
    signal?.addEventListener("abort", done, { once: true });
  });
}

/**
 * Poll a job to a terminal state. Resolves with the final state (also when aborted: the last
 * state seen). Never rejects.
 */
export async function pollJob(id: string, options: PollOptions = {}): Promise<JobPollState> {
  const o = { ...POLL_DEFAULTS, ...options };
  const client = options.client ?? defaultJobClient;
  const { signal, onUpdate } = options;
  const startedAt = Date.now();

  let state: JobPollState = INITIAL_POLL_STATE;
  const emit = (patch: Partial<JobPollState>) => {
    state = { ...state, ...patch };
    if (!signal?.aborted) onUpdate?.(state);
  };
  const backoff = (n: number) => Math.min(o.backoffMaxMs, o.backoffBaseMs * 2 ** (n - 1));

  /** Shared transient-error handling. Returns true if the caller should retry. */
  const handleError = async (err: unknown, failures: number): Promise<boolean> => {
    const e = ApiError.from(err);
    if (signal?.aborted || e.code === "aborted") return false;
    if (!e.isTransient || failures >= o.maxConsecutiveErrors) {
      emit({ phase: "failed", error: e, retrying: 0, lastTransientError: null });
      return false;
    }
    emit({ retrying: failures, lastTransientError: e });
    await sleep(backoff(failures), signal);
    return !signal?.aborted;
  };

  // 1. Poll status.
  let failures = 0;
  for (;;) {
    if (signal?.aborted) return state;
    let job: JobStatus;
    try {
      job = await client.getJob(id, signal);
    } catch (err) {
      failures += 1;
      if (await handleError(err, failures)) continue;
      return state;
    }
    if (signal?.aborted) return state;
    failures = 0;

    if (job.status === "failed") {
      emit({ phase: "failed", job, error: ApiError.fromJob(job.error), retrying: 0, lastTransientError: null });
      return state;
    }
    if (job.status === "succeeded") {
      emit({ phase: "fetching_result", job, retrying: 0, lastTransientError: null });
      break;
    }
    emit({ phase: "polling", job, retrying: 0, lastTransientError: null });
    const elapsed = Date.now() - startedAt;
    await sleep(elapsed >= o.slowAfterMs ? o.slowIntervalMs : o.intervalMs, signal);
  }

  // 2. Fetch the result once (retry on transient errors and on a not_ready race).
  failures = 0;
  for (;;) {
    if (signal?.aborted) return state;
    try {
      const result = await client.getResult(id, signal);
      if (signal?.aborted) return state;
      emit({ phase: "succeeded", result, retrying: 0, lastTransientError: null });
      return state;
    } catch (err) {
      failures += 1;
      const e = ApiError.from(err);
      if (e.code === "not_ready" && failures < o.maxConsecutiveErrors) {
        await sleep(o.intervalMs, signal);
        continue;
      }
      if (await handleError(e, failures)) continue;
      return state;
    }
  }
}
