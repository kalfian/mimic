/**
 * Session store (PLAN-auth §8.2): who is signed in, shared by every component through
 * `useSession()`. A module-level store read with `useSyncExternalStore`, so no React provider is
 * needed. No JSX. The session itself is an HttpOnly cookie the browser manages; this store only
 * holds the result of `GET /api/auth/me` in memory (never in Web Storage).
 *
 *   const session = useSession();          // also starts loading the session on first use
 *   session.status: "loading" | "anonymous" | "authenticated" | "error"
 *
 * State changes:
 * - `loadSession()` (first `useSession()`): `/me` → authenticated | 401 → anonymous | other → error.
 * - `login()`: `POST /login`, then `/me` to prove the cookie stuck (401 there → `cookie_rejected`).
 * - `logout()`, `changePassword()`: as named.
 * - Any API call answering 401 `unauthenticated` → anonymous. If the user *was* signed in, `error`
 *   is that `unauthenticated` error, so the login page can say "your session expired".
 * - Any API call answering 403 `password_change_required` → `me.must_change_password = true`
 *   (+ a `/me` refresh), so `routeDecision` sends the user to `/account/password`.
 */

import { useEffect, useSyncExternalStore } from "react";

import * as api from "./api";
import { ApiError } from "./errors";
import type { Me } from "./types";

export type SessionStatus = "loading" | "anonymous" | "authenticated" | "error";

export interface SessionState {
  status: SessionStatus;
  /** Set only when `authenticated`. */
  me: Me | null;
  /**
   * - `error`: why `/me` failed (network, 5xx). Offer "Try again" → `refreshSession()`.
   * - `anonymous`: `unauthenticated` when a signed-in session ended (expired, revoked, disabled),
   *   null after a normal sign-out or when the user was never signed in.
   */
  error: ApiError | null;
}

const LOADING: SessionState = Object.freeze({ status: "loading", me: null, error: null }) as SessionState;

let state: SessionState = LOADING;
let loadPromise: Promise<SessionState> | null = null;
/** Bumped by every operation that owns the outcome; stale async results are dropped. */
let generation = 0;
const listeners = new Set<() => void>();

function setState(next: SessionState): void {
  state = Object.freeze(next) as SessionState;
  for (const l of [...listeners]) l();
}

/** Current snapshot (stable identity until the next change). */
export function getSession(): SessionState {
  return state;
}

export function subscribeSession(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function authenticated(me: Me): SessionState {
  return { status: "authenticated", me, error: null };
}

/** After `/me` said 401: anonymous, keeping a "session expired" error set by the auth event. */
function anonymousAfter401(): SessionState {
  return { status: "anonymous", me: null, error: state.status === "anonymous" ? state.error : null };
}

async function fetchMe(): Promise<SessionState> {
  const gen = ++generation;
  try {
    const me = await api.getMe();
    if (gen === generation) setState(authenticated(me));
  } catch (err) {
    const e = ApiError.from(err);
    if (gen === generation) setState(e.code === "unauthenticated" ? anonymousAfter401() : { status: "error", me: null, error: e });
  }
  return state;
}

/**
 * Resolve the session once (idempotent: later calls return the same promise / current state).
 * `useSession()` calls it, so components rarely need to.
 */
export function loadSession(): Promise<SessionState> {
  if (state.status !== "loading") return Promise.resolve(state);
  if (!loadPromise) {
    loadPromise = fetchMe().finally(() => {
      loadPromise = null;
    });
  }
  return loadPromise;
}

/**
 * Re-read `/api/auth/me` (e.g. "Try again" in the error state, or after an admin action that may
 * have affected the own account). Keeps showing the current state while it runs, except from
 * `error`, which goes back to `loading`.
 */
export function refreshSession(): Promise<SessionState> {
  if (state.status === "error") setState(LOADING);
  const p = fetchMe();
  if (state.status === "loading") {
    loadPromise = p.finally(() => {
      loadPromise = null;
    });
  }
  return p;
}

/**
 * Sign in. Resolves the account (check `me.must_change_password`; `routeDecision` redirects to
 * `/account/password` by itself). Rejects with `ApiError`: `invalid_credentials`,
 * `account_disabled`, `setup_required`, `too_many_attempts` (`retryAfterS`), `invalid_request`,
 * `origin_not_allowed`, `network_error`, or the client code **`cookie_rejected`** when the API
 * accepted the password but the browser didn't send the cookie back (hostname mismatch).
 * On failure the store stays `anonymous`; show the error in the form.
 */
export async function login(username: string, password: string): Promise<Me> {
  const gen = ++generation;
  await api.login(username, password);
  let me: Me;
  try {
    me = await api.getMe();
  } catch (err) {
    const e = ApiError.from(err);
    if (gen === generation) setState({ status: "anonymous", me: null, error: null });
    if (e.code === "unauthenticated") {
      const hosts = api.hostnameMismatch();
      throw ApiError.cookieRejected(hosts ?? { apiHost: api.apiHostname() });
    }
    throw e;
  }
  if (gen === generation) setState(authenticated(me));
  return me;
}

/**
 * Sign out (`POST /api/auth/logout`, then anonymous). If the request fails (API unreachable) it
 * rejects and the state is unchanged: the server session would still be valid.
 */
export async function logout(): Promise<void> {
  const gen = ++generation;
  await api.logout();
  if (gen === generation) setState({ status: "anonymous", me: null, error: null });
}

/**
 * Change the own password (voluntary or forced). Resolves the updated account
 * (`must_change_password: false`) and updates the store. Rejects with `current_password_incorrect`,
 * `weak_password` (server message names the rule), `too_many_attempts`, `unauthenticated`.
 */
export async function changePassword(currentPassword: string, newPassword: string): Promise<Me> {
  const gen = ++generation;
  const me = await api.changePassword(currentPassword, newPassword);
  if (gen === generation) setState(authenticated(me));
  return me;
}

function handleAuthEvent(event: api.AuthEvent, error: ApiError): void {
  if (event === "unauthenticated") {
    if (state.status === "authenticated") setState({ status: "anonymous", me: null, error });
    else if (state.status === "error") setState({ status: "anonymous", me: null, error: null });
    return;
  }
  // password_change_required: flip the flag now (routing reacts at once), then confirm via /me.
  if (state.status === "authenticated" && state.me && !state.me.must_change_password) {
    setState(authenticated({ ...state.me, must_change_password: true }));
    void fetchMe();
  }
}

api.onAuthEvent(handleAuthEvent);

/** Back to the initial `loading` state (tests). */
export function resetSessionForTests(): void {
  generation++;
  loadPromise = null;
  setState(LOADING);
}

/* ---------- React ---------- */

const getServerSnapshot = () => LOADING;

/**
 * The session, re-rendering on every change. Starts `loadSession()` on first use. On the server
 * (prerender) it is always `loading`, so protected content is never rendered before the client
 * knows who is signed in.
 */
export function useSession(): SessionState {
  const snapshot = useSyncExternalStore(subscribeSession, getSession, getServerSnapshot);
  useEffect(() => {
    void loadSession();
  }, []);
  return snapshot;
}

/** True for an authenticated admin with a full session (no pending forced change). */
export function isAdminSession(session: SessionState): boolean {
  return session.status === "authenticated" && session.me?.role === "admin" && !session.me.must_change_password;
}
