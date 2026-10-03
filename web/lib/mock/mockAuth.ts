/**
 * In-browser fake of the auth + admin API (PLAN-auth §8.3), used when `NEXT_PUBLIC_API_MOCK=1`.
 *
 * ⚠ MOCK ONLY — NOT SECURITY. Any non-empty password signs in, except the literal `wrong`
 * (→ `invalid_credentials`). Accounts and the "current session" live in localStorage
 * (`mimic.mock.auth`) so a reload keeps you signed in; no password is ever stored. The real API
 * keeps sessions server-side behind an HttpOnly cookie.
 *
 * Seeded accounts (any password works):
 *   admin    admin                       alice  user (owns 2 seeded jobs)
 *   user     user (owns the fixture job) newbie user, must change password at sign-in
 *   disabled user, inactive (→ account_disabled)
 *
 * The guards mirror the backend: `last_admin`, `self_action_forbidden`, `username_taken`, the
 * password policy (`weak_password`), `current_password_incorrect` (current password `wrong`), and
 * the login throttle (5 failures per username in 15 min → `too_many_attempts` + Retry-After).
 *
 * Page-URL scenarios: `?fail=<code>`, see `MOCK_AUTH_FAILURES` / lib/README.md "Auth".
 */

import { passwordPolicyViolation } from "../authPolicy";
import { ApiError } from "../errors";
import type { AdminUser, AdminUserList, AuthStatus, CreateUserRequest, ErrorCode, Me, TemporaryPassword, UpdateUserRequest, UserRole } from "../types";
import { PASSWORD_MIN_LENGTH, USERNAME_PATTERN } from "../types";
import { deleteJobsOf, jobCountOf, MOCK_USER_IDS } from "./mockJobsRegistry";
import { clearMockState, currentSearch, MOCK_AUTH_STORAGE_KEY, mockLatency, readMockState, writeMockState } from "./mockStore";

/* ---------- scenarios ---------- */

/**
 * `?fail=<code>` values per operation. A code only affects the operations it is listed for, so
 * `?fail=last_admin` on /admin/users makes role changes / disable / delete fail while the list
 * still loads.
 *
 * - `network_error` and `unauthenticated` on list calls fail **once per page load**, so the error
 *   state's "Try again" (or signing in again) then works.
 * - `cookie_rejected`: login "succeeds" but no session is kept, so the follow-up `/me` is 401.
 */
export const MOCK_AUTH_FAILURES = {
  authStatus: ["setup_required"],
  login: ["setup_required", "invalid_credentials", "account_disabled", "too_many_attempts", "cookie_rejected", "origin_not_allowed", "network_error"],
  changePassword: ["current_password_incorrect", "weak_password", "too_many_attempts"],
  listUsers: ["network_error", "unauthenticated", "forbidden"],
  createUser: ["username_taken"],
  updateUser: ["last_admin", "self_action_forbidden", "not_found"],
  resetUserPassword: ["self_action_forbidden", "not_found"],
  deleteUser: ["last_admin", "self_action_forbidden", "not_found"],
  listJobs: ["network_error", "unauthenticated"],
  deleteJob: ["not_found"],
} as const;

export type MockAuthOperation = keyof typeof MOCK_AUTH_FAILURES;
export type MockAuthFailure = (typeof MOCK_AUTH_FAILURES)[MockAuthOperation][number];

/** Seconds reported in `Retry-After` for `?fail=too_many_attempts`. */
export const MOCK_RETRY_AFTER_S = 90;

const ALL_FAILURES: ReadonlySet<string> = new Set(Object.values(MOCK_AUTH_FAILURES).flat());
const failedOnce = new Set<string>();

/** `?fail=<code>` if it is an auth/admin scenario (default: the current page URL). */
export function readMockAuthFailure(search: string = currentSearch()): MockAuthFailure | null {
  const fail = new URLSearchParams(search).get("fail");
  return fail && ALL_FAILURES.has(fail) ? (fail as MockAuthFailure) : null;
}

/** The scenario for `op`, or null if `?fail=` is absent or doesn't apply to `op`. */
export function mockFailureFor(op: MockAuthOperation, search: string = currentSearch()): MockAuthFailure | null {
  const fail = readMockAuthFailure(search);
  return fail && (MOCK_AUTH_FAILURES[op] as readonly string[]).includes(fail) ? fail : null;
}

/** Like `mockFailureFor`, but only the first time per page load (list calls). */
export function mockFailOnce(op: MockAuthOperation): MockAuthFailure | null {
  const fail = mockFailureFor(op);
  if (!fail || failedOnce.has(`${op}:${fail}`)) return null;
  failedOnce.add(`${op}:${fail}`);
  return fail;
}

/* ---------- state ---------- */

interface MockUser {
  id: string;
  username: string;
  role: UserRole;
  is_active: boolean;
  must_change_password: boolean;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
}

interface AuthState {
  v: 1;
  users: MockUser[];
  /** The signed-in mock user, or null. Stands in for the HttpOnly session cookie. */
  sessionUserId: string | null;
}

function seedUsers(): MockUser[] {
  const at = "2026-09-01T08:00:00.000000Z";
  const u = (username: keyof typeof MOCK_USER_IDS, role: UserRole, extra: Partial<MockUser> = {}): MockUser => ({
    id: MOCK_USER_IDS[username],
    username,
    role,
    is_active: true,
    must_change_password: false,
    created_at: at,
    updated_at: at,
    last_login_at: null,
    ...extra,
  });
  return [
    u("admin", "admin", { last_login_at: "2026-10-02T17:45:00.000000Z" }),
    u("alice", "user", { last_login_at: "2026-09-30T08:10:00.000000Z" }),
    u("disabled", "user", { is_active: false }),
    u("newbie", "user", { must_change_password: true }),
    u("user", "user", { last_login_at: "2026-10-03T07:55:00.000000Z" }),
  ];
}

function isAuthState(value: unknown): value is AuthState {
  const v = value as AuthState | null;
  return typeof v === "object" && v !== null && v.v === 1 && Array.isArray(v.users) && (v.sessionUserId === null || typeof v.sessionUserId === "string");
}

let state: AuthState | null = null;
/** Login / password-change failures per key, ms timestamps (in memory, like the backend). */
const attempts = new Map<string, number[]>();

function load(): AuthState {
  if (!state) state = readMockState(MOCK_AUTH_STORAGE_KEY, isAuthState) ?? { v: 1, users: seedUsers(), sessionUserId: null };
  return state;
}

function save(): void {
  if (state) writeMockState(MOCK_AUTH_STORAGE_KEY, state);
}

/** Back to the seeded accounts, signed out (tests, "reset mock data"). */
export function resetMockAuth(): void {
  clearMockState(MOCK_AUTH_STORAGE_KEY);
  state = { v: 1, users: seedUsers(), sessionUserId: null };
  attempts.clear();
  failedOnce.clear();
  save();
}

/**
 * Sign in as a seeded/created mock user without a password (tests, dev shortcuts). Pass null to
 * sign out. Throws for an unknown username.
 */
export function mockSignInAs(username: string | null): void {
  const s = load();
  if (username === null) {
    s.sessionUserId = null;
  } else {
    const user = s.users.find((u) => u.username === username);
    if (!user) throw new Error(`No mock user "${username}".`);
    s.sessionUserId = user.id;
  }
  save();
}

/* ---------- helpers ---------- */

const MINUTE = 60_000;
const ATTEMPT_WINDOW_MS = 15 * MINUTE;
const MAX_FAILURES = 5;
const USERNAME_RE = new RegExp(USERNAME_PATTERN);
const USER_ID_RE = /^[0-9a-f]{32}$/;

function httpError(code: ErrorCode, status: number, message: string, retryAfterS?: number): ApiError {
  return ApiError.fromResponse(status, { error: { code, message } }, { retryAfter: retryAfterS == null ? null : String(retryAfterS) });
}

const MESSAGES = {
  unauthenticated: "You are not signed in, or your session expired. Sign in again.",
  invalid_credentials: "Wrong username or password.",
  account_disabled: "This account is disabled. Ask an admin to enable it.",
  forbidden: "You don't have permission to do this.",
  password_change_required: "Set a new password before continuing.",
  origin_not_allowed: "This request came from a page that is not allowed to use this API.",
  too_many_attempts: "Too many failed attempts. Wait a few minutes and try again.",
  setup_required: "No admin account exists yet. On the API host, run `make create-admin`.",
  last_admin: "This is the last active admin. Make another user an admin first.",
  self_action_forbidden: "You can't do this to your own account.",
  username_taken: "That username is already taken.",
  current_password_incorrect: "The current password is wrong.",
  not_found: "User not found.",
} as const;

const STATUS: Record<keyof typeof MESSAGES, number> = {
  unauthenticated: 401,
  invalid_credentials: 401,
  account_disabled: 403,
  forbidden: 403,
  password_change_required: 403,
  origin_not_allowed: 403,
  too_many_attempts: 429,
  setup_required: 409,
  last_admin: 409,
  self_action_forbidden: 409,
  username_taken: 409,
  current_password_incorrect: 422,
  not_found: 404,
};

function fail(code: keyof typeof MESSAGES, retryAfterS?: number): ApiError {
  return httpError(code, STATUS[code], MESSAGES[code], retryAfterS);
}

/** Turn a `?fail=` scenario into the error the real API would return. */
function scenarioError(code: MockAuthFailure): ApiError {
  switch (code) {
    case "network_error":
      return ApiError.network();
    case "too_many_attempts":
      return fail("too_many_attempts", MOCK_RETRY_AFTER_S);
    case "weak_password":
      return httpError("weak_password", 422, `Use at least ${PASSWORD_MIN_LENGTH} characters.`);
    case "cookie_rejected":
      throw new Error("cookie_rejected is handled by mockLogin");
    default:
      return fail(code);
  }
}

function nowIso(): string {
  return new Date().toISOString();
}

function toMe(u: MockUser): Me {
  return { id: u.id, username: u.username, role: u.role, must_change_password: u.must_change_password, created_at: u.created_at };
}

function toAdminUser(u: MockUser): AdminUser {
  return { ...u, job_count: jobCountOf(u.id) };
}

function temporaryPassword(): string {
  // Same shape as the backend's secrets.token_urlsafe(12): 16 URL-safe characters.
  const bytes = new Uint8Array(12);
  globalThis.crypto.getRandomValues(bytes);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function recentFailures(key: string, now: number): number[] {
  const list = (attempts.get(key) ?? []).filter((at) => now - at < ATTEMPT_WINDOW_MS);
  attempts.set(key, list);
  return list;
}

function throttle(key: string): void {
  const now = Date.now();
  const list = recentFailures(key, now);
  if (list.length >= MAX_FAILURES) {
    throw fail("too_many_attempts", Math.max(1, Math.ceil((list[0] + ATTEMPT_WINDOW_MS - now) / 1000)));
  }
}

function recordFailure(key: string): void {
  recentFailures(key, Date.now()).push(Date.now());
}

/** Password policy, mirrors api/app/auth/policy.py `check_password_policy` (shared with the forms). */
export const mockPasswordPolicyViolation = passwordPolicyViolation;

/* ---------- session guards (used by mockApi.ts for the job routes) ---------- */

/** The signed-in mock user (forced password change allowed), or null. */
function sessionUser(): MockUser | null {
  const s = load();
  const user = s.users.find((u) => u.id === s.sessionUserId) ?? null;
  if (!user || !user.is_active) {
    if (s.sessionUserId !== null) {
      s.sessionUserId = null; // stale/revoked session
      save();
    }
    return null;
  }
  return user;
}

/** Current account for a route that allows the forced-change state (`/me`, password). 401 otherwise. */
export function requireMockSession(): Me {
  const user = sessionUser();
  if (!user) throw fail("unauthenticated");
  return toMe(user);
}

/** Current account with a full session: 401 anonymous, 403 `password_change_required`. */
export function requireMockUser(): Me {
  const me = requireMockSession();
  if (me.must_change_password) throw fail("password_change_required");
  return me;
}

/** Full session + admin role: additionally 403 `forbidden` for users. */
export function requireMockAdmin(): Me {
  const me = requireMockUser();
  if (me.role !== "admin") throw fail("forbidden");
  return me;
}

/** Full session or null, never throws (health). */
export function optionalMockUser(): Me | null {
  const user = sessionUser();
  return user && !user.must_change_password ? toMe(user) : null;
}

/** `{id, username}` of a mock account, or null if it no longer exists. */
export function mockUserRef(id: string | null): { id: string; username: string } | null {
  if (id === null) return null;
  const user = load().users.find((u) => u.id === id);
  return user ? { id: user.id, username: user.username } : null;
}

/** Simulate a revoked/expired session (used by `?fail=unauthenticated`). */
export function dropMockSession(): ApiError {
  const s = load();
  s.sessionUserId = null;
  save();
  return fail("unauthenticated");
}

/* ---------- auth endpoints (same signatures as lib/api.ts) ---------- */

export async function mockGetAuthStatus(signal?: AbortSignal): Promise<AuthStatus> {
  await mockLatency(signal);
  const hasAdmin = load().users.some((u) => u.role === "admin" && u.is_active);
  return { setup_required: mockFailureFor("authStatus") === "setup_required" || !hasAdmin };
}

export async function mockLogin(username: string, password: string, signal?: AbortSignal): Promise<Me> {
  await mockLatency(signal);
  const scenario = mockFailureFor("login");
  const s = load();
  if (scenario === "setup_required" || !s.users.some((u) => u.role === "admin" && u.is_active)) throw fail("setup_required");
  if (scenario === "origin_not_allowed" || scenario === "network_error" || scenario === "too_many_attempts") throw scenarioError(scenario);

  const name = username.trim().toLowerCase();
  if (name === "" || password === "") throw httpError("invalid_request", 422, "Enter a username and a password.");
  throttle(`login:${name}`);

  const user = s.users.find((u) => u.username === name);
  if (scenario === "invalid_credentials" || !user || password === "wrong") {
    recordFailure(`login:${name}`);
    throw fail("invalid_credentials");
  }
  if (scenario === "account_disabled" || !user.is_active) throw fail("account_disabled");

  attempts.delete(`login:${name}`);
  user.last_login_at = nowIso();
  // cookie_rejected: the API answered 200, but "the browser dropped the cookie" (no session kept).
  s.sessionUserId = scenario === "cookie_rejected" ? null : user.id;
  save();
  return toMe(user);
}

export async function mockLogout(signal?: AbortSignal): Promise<void> {
  await mockLatency(signal);
  const s = load();
  s.sessionUserId = null;
  save();
}

export async function mockGetMe(signal?: AbortSignal): Promise<Me> {
  await mockLatency(signal);
  return requireMockSession();
}

export async function mockChangePassword(currentPassword: string, newPassword: string, signal?: AbortSignal): Promise<Me> {
  await mockLatency(signal);
  const me = requireMockSession();
  const scenario = mockFailureFor("changePassword");
  if (scenario) throw scenarioError(scenario);
  if (currentPassword === "" || newPassword === "") throw httpError("invalid_request", 422, "The request is invalid.");

  const key = `password:${me.id}`;
  throttle(key);
  if (currentPassword === "wrong") {
    recordFailure(key);
    throw fail("current_password_incorrect");
  }
  const violation = mockPasswordPolicyViolation(newPassword, me.username, currentPassword);
  if (violation) throw httpError("weak_password", 422, violation);

  attempts.delete(key);
  const s = load();
  const user = s.users.find((u) => u.id === me.id)!;
  user.must_change_password = false;
  user.updated_at = nowIso();
  // The real API revokes every session and sets a new cookie; in the mock this browser stays signed in.
  s.sessionUserId = user.id;
  save();
  return toMe(user);
}

/* ---------- admin endpoints ---------- */

function findTarget(id: string): MockUser {
  const user = USER_ID_RE.test(id) ? load().users.find((u) => u.id === id) : undefined;
  if (!user) throw fail("not_found");
  return user;
}

function activeAdminsExcept(id: string): number {
  return load().users.filter((u) => u.role === "admin" && u.is_active && u.id !== id).length;
}

/** Role change / disable / reset revoke the target's sessions; in the mock only ours exists. */
function revokeSessionsOf(id: string): void {
  const s = load();
  if (s.sessionUserId === id) s.sessionUserId = null;
}

export async function mockListUsers(signal?: AbortSignal): Promise<AdminUserList> {
  await mockLatency(signal);
  const scenario = mockFailOnce("listUsers");
  if (scenario === "unauthenticated") throw dropMockSession();
  requireMockAdmin();
  if (scenario) throw scenarioError(scenario);
  const items = [...load().users].sort((a, b) => a.username.localeCompare(b.username)).map(toAdminUser);
  return { items };
}

export async function mockCreateUser(body: CreateUserRequest, signal?: AbortSignal): Promise<TemporaryPassword> {
  await mockLatency(signal);
  requireMockAdmin();
  if (mockFailureFor("createUser") === "username_taken") throw fail("username_taken");
  const username = body.username.trim().toLowerCase();
  if (!USERNAME_RE.test(username)) {
    throw httpError(
      "invalid_request",
      422,
      "Usernames are 3–32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter or digit.",
    );
  }
  const s = load();
  if (s.users.some((u) => u.username === username)) throw fail("username_taken");
  const at = nowIso();
  const user: MockUser = {
    id: randomHexId(),
    username,
    role: body.role ?? "user",
    is_active: true,
    must_change_password: true,
    created_at: at,
    updated_at: at,
    last_login_at: null,
  };
  s.users.push(user);
  save();
  return { user: toAdminUser(user), temporary_password: temporaryPassword() };
}

export async function mockUpdateUser(id: string, body: UpdateUserRequest, signal?: AbortSignal): Promise<AdminUser> {
  await mockLatency(signal);
  const me = requireMockAdmin();
  const scenario = mockFailureFor("updateUser");
  if (scenario) throw scenarioError(scenario);
  const target = findTarget(id);
  const role = body.role ?? null;
  const isActive = body.is_active ?? null;
  if (role === null && isActive === null) throw httpError("invalid_request", 422, "Set role or is_active.");
  if (isActive === false && target.id === me.id) throw fail("self_action_forbidden");

  const demotes = role === "user" && target.role === "admin";
  const disables = isActive === false && target.is_active;
  if ((demotes || disables) && target.role === "admin" && target.is_active && activeAdminsExcept(target.id) === 0) throw fail("last_admin");

  const roleChanged = role !== null && role !== target.role;
  const activeChanged = isActive !== null && isActive !== target.is_active;
  if (roleChanged) target.role = role;
  if (activeChanged) target.is_active = isActive;
  if (roleChanged || activeChanged) target.updated_at = nowIso();
  if (roleChanged || disables) revokeSessionsOf(target.id);
  save();
  return toAdminUser(target);
}

export async function mockResetUserPassword(id: string, signal?: AbortSignal): Promise<TemporaryPassword> {
  await mockLatency(signal);
  const me = requireMockAdmin();
  const scenario = mockFailureFor("resetUserPassword");
  if (scenario) throw scenarioError(scenario);
  const target = findTarget(id);
  if (target.id === me.id) throw fail("self_action_forbidden");
  target.must_change_password = true;
  target.updated_at = nowIso();
  revokeSessionsOf(target.id);
  save();
  return { user: toAdminUser(target), temporary_password: temporaryPassword() };
}

export async function mockDeleteUser(id: string, signal?: AbortSignal): Promise<void> {
  await mockLatency(signal);
  const me = requireMockAdmin();
  const scenario = mockFailureFor("deleteUser");
  if (scenario) throw scenarioError(scenario);
  const target = findTarget(id);
  if (target.id === me.id) throw fail("self_action_forbidden");
  if (target.role === "admin" && target.is_active && activeAdminsExcept(target.id) === 0) throw fail("last_admin");
  const s = load();
  deleteJobsOf(target.id);
  s.users = s.users.filter((u) => u.id !== target.id);
  revokeSessionsOf(target.id);
  save();
}

function randomHexId(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}
