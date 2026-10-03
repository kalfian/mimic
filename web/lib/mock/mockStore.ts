/**
 * Persistence for mock mode (NEXT_PUBLIC_API_MOCK=1): fake accounts, the fake "current session"
 * and which mock job belongs to whom, so a page reload keeps you signed in and keeps the job list.
 *
 * Browser: `localStorage` (keys below). Node tests / SSR: an in-memory map.
 *
 * MOCK ONLY. Nothing secret is stored: mock accounts have no passwords at all (any password except
 * `wrong` signs in) and temporary passwords are never persisted. The real client keeps the session
 * in an HttpOnly cookie and never touches Web Storage.
 */

import { ApiError } from "../errors";

/** localStorage key of the mock accounts + current mock session (PLAN-auth §8.3). */
export const MOCK_AUTH_STORAGE_KEY = "mimic.mock.auth";
/** localStorage key of the mock job registry (owners, filenames, deleted ids). */
export const MOCK_JOBS_STORAGE_KEY = "mimic.mock.jobs";

const memory = new Map<string, string>();

function browserStorage(): Storage | null {
  // Check `window` first: touching `globalThis.localStorage` in Node prints a warning.
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage ?? null;
  } catch {
    return null; // disabled storage (privacy mode, sandboxed iframe)
  }
}

export function readMockState<T>(key: string, isValid: (value: unknown) => value is T): T | null {
  let raw: string | null = null;
  const storage = browserStorage();
  try {
    raw = storage ? storage.getItem(key) : (memory.get(key) ?? null);
  } catch {
    raw = null;
  }
  if (raw == null) return null;
  try {
    const value = JSON.parse(raw) as unknown;
    return isValid(value) ? value : null;
  } catch {
    return null;
  }
}

export function writeMockState(key: string, value: unknown): void {
  const raw = JSON.stringify(value);
  const storage = browserStorage();
  try {
    if (storage) storage.setItem(key, raw);
    else memory.set(key, raw);
  } catch {
    memory.set(key, raw); // quota / disabled storage: keep it for this page at least
  }
}

export function clearMockState(key: string): void {
  memory.delete(key);
  try {
    browserStorage()?.removeItem(key);
  } catch {
    // ignore
  }
}

/** `?<name>=` from the current page URL ("" outside a browser). */
export function currentSearch(): string {
  return typeof window !== "undefined" && window.location ? window.location.search : "";
}

/** Resolves after `ms`; rejects with `ApiError` code `aborted` when `signal` fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(ApiError.aborted());
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(ApiError.aborted());
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Shared simulated latency for every mock endpoint (see `configureMock` in mockApi.ts). */
export const mockTiming = { timeScale: 1, latencyMs: 120 };

export const mockLatency = (signal?: AbortSignal) => sleep(mockTiming.latencyMs * mockTiming.timeScale, signal);
