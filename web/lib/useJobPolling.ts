/**
 * React hook: poll a job until it is terminal, then fetch its result once (PLAN §10.2).
 * Client components only (`"use client"` in the component that calls it).
 *
 *   const { phase, job, stage, progress, result, error, isLoading, retry, restart } = useJobPolling(id);
 *
 * - `phase`: loading → polling → fetching_result → succeeded | failed
 * - `error` is set only when failed; `error.origin === "job"` = pipeline failure (code from
 *   JobStatus.error), otherwise a request failure (e.g. `not_found`, `network_error`).
 * - `retrying > 0` while transient errors are being retried with backoff ("Reconnecting…").
 * - `restart()` polls the same id again from scratch, e.g. after `rerunInterpretation(id)` resolved
 *   (the job is queued again). State resets to `loading` until the first status arrives.
 * - `retry()` is the same as `restart()` (kept for the error state's "Try again").
 * - Changing `id` restarts; unmount aborts in-flight requests.
 */

import { useCallback, useEffect, useState } from "react";

import type { ApiError } from "./errors";
import { INITIAL_POLL_STATE, isTerminalPhase, pollJob, type JobPollState, type PollPhase } from "./jobPoller";
import type { JobState, JobStatus, ResultEnvelope, Stage } from "./types";
import { STAGE_LABELS } from "./types";

export interface UseJobPollingOptions {
  intervalMs?: number;
  slowIntervalMs?: number;
  slowAfterMs?: number;
}

export interface UseJobPollingResult {
  phase: PollPhase;
  /** Latest `GET /api/jobs/{id}` response (null before the first one). */
  job: JobStatus | null;
  /** `job.status` shortcut. */
  status: JobState | null;
  stage: Stage | null;
  /** Backend label for `stage` (use `stageLabel()` from format.ts for the AI-off variant). */
  stageLabel: string | null;
  /** 0..1 overall progress. */
  progress: number;
  result: ResultEnvelope | null;
  error: ApiError | null;
  /** True until a terminal phase (succeeded with result, or failed). */
  isLoading: boolean;
  isTerminal: boolean;
  retrying: number;
  lastTransientError: ApiError | null;
  /** Poll the same id again from scratch (after `rerunInterpretation`). Stable identity. */
  restart: () => void;
  /** Alias of `restart` (error state "Try again"). */
  retry: () => void;
}

interface Snapshot {
  key: string;
  state: JobPollState;
}

export function useJobPolling(id: string | null | undefined, options: UseJobPollingOptions = {}): UseJobPollingResult {
  const { intervalMs, slowIntervalMs, slowAfterMs } = options;
  const [attempt, setAttempt] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const key = `${id ?? ""}#${attempt}`;

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    void pollJob(id, {
      intervalMs,
      slowIntervalMs,
      slowAfterMs,
      signal: controller.signal,
      onUpdate: (state) => setSnapshot({ key, state }),
    });
    return () => controller.abort();
  }, [id, key, intervalMs, slowIntervalMs, slowAfterMs]);

  const restart = useCallback(() => setAttempt((n) => n + 1), []);

  // A snapshot from a previous id/attempt is stale: show the initial state instead.
  const state = id && snapshot?.key === key ? snapshot.state : INITIAL_POLL_STATE;
  const job = state.job;
  const terminal = isTerminalPhase(state.phase);

  return {
    phase: state.phase,
    job,
    status: job?.status ?? null,
    stage: job?.stage ?? null,
    stageLabel: job ? (job.stage_label || STAGE_LABELS[job.stage]) : null,
    progress: state.phase === "succeeded" ? 1 : (job?.progress ?? 0),
    result: state.result,
    error: state.error,
    isLoading: Boolean(id) && !terminal,
    isTerminal: terminal,
    retrying: state.retrying,
    lastTransientError: state.lastTransientError,
    restart,
    retry: restart,
  };
}
