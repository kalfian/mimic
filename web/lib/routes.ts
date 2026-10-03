/**
 * Client-side route policy (PLAN-auth §8.1, A9). Pure, no React, no Next imports.
 *
 * The API enforces every permission; this only decides what the browser shows, so protected pages
 * are never rendered for the wrong session. `AuthGate` calls:
 *
 *   const d = routeDecision(pathname, session, { search: window.location.search, next });
 *   loading   → neutral placeholder (never the page)
 *   error     → "can't reach the server" + Try again (`refreshSession()`)
 *   redirect  → router.replace(d.to)
 *   forbidden → 403 state (signed in, not allowed: users on /admin/*)
 *   render    → children
 *
 * | Path                 | Access                         | Anonymous        | Forced change       |
 * |----------------------|--------------------------------|------------------|---------------------|
 * | /login               | public, anonymous only         | render           | → /account/password |
 * | /account/password    | any session (forced allowed)   | → /login         | render              |
 * | /admin, /admin/*     | admin (users: forbidden)       | → /login         | → /account/password |
 * | everything else      | full session                   | → /login         | → /account/password |
 *
 * An authenticated user on /login goes to `next` (validated by `safeNext`) or `/`.
 */

import type { SessionState } from "./session";

export const LOGIN_PATH = "/login";
export const CHANGE_PASSWORD_PATH = "/account/password";
export const JOBS_PATH = "/jobs";
export const ADMIN_USERS_PATH = "/admin/users";

export type RouteAccess = "public" | "session" | "admin" | "user";

export type RouteDecision =
  | { kind: "loading" }
  | { kind: "render" }
  | { kind: "forbidden" }
  | { kind: "error"; error: NonNullable<SessionState["error"]> }
  | { kind: "redirect"; to: string };

/** `/jobs/` → `/jobs`; empty → `/`. */
export function normalizePathname(pathname: string): string {
  if (!pathname) return "/";
  const p = pathname.length > 1 ? pathname.replace(/\/+$/, "") : pathname;
  return p === "" ? "/" : p;
}

export function routeAccess(pathname: string): RouteAccess {
  const p = normalizePathname(pathname);
  if (p === LOGIN_PATH) return "public";
  if (p === CHANGE_PASSWORD_PATH) return "session";
  if (p === "/admin" || p.startsWith("/admin/")) return "admin";
  return "user";
}

const BASE = "http://mimic.invalid";
// Control characters: the URL parser silently drops tab/CR/LF, which can turn "/\t/x" into "//x".
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

/**
 * A redirect target from untrusted input (`?next=`): only same-app paths. Rejects absolute and
 * protocol-relative URLs (`https://…`, `//evil`), backslashes (`/\evil`), control characters,
 * encoded variants of those (`/%2F%2Fevil`, `/%5Cevil`) and the login page itself (redirect loop).
 * Returns `fallback` (default `/`) when rejected. Accepts the raw `searchParams` value (string or array).
 */
export function safeNext(raw: string | string[] | null | undefined, fallback: string = "/"): string {
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (typeof value !== "string" || value === "" || value.length > 2048) return fallback;
  if (!value.startsWith("/") || value.startsWith("//") || value.includes("\\") || CONTROL_CHARS.test(value)) return fallback;
  let decoded: string;
  try {
    decoded = decodeURIComponent(value);
  } catch {
    return fallback; // malformed %-escape
  }
  if (decoded.startsWith("//") || decoded.includes("\\") || CONTROL_CHARS.test(decoded)) return fallback;
  let url: URL;
  try {
    url = new URL(value, BASE);
  } catch {
    return fallback;
  }
  if (url.origin !== BASE) return fallback;
  if (normalizePathname(url.pathname) === LOGIN_PATH) return fallback;
  return `${url.pathname}${url.search}${url.hash}`;
}

function withNext(path: string, next: string | null | undefined): string {
  const target = next ? safeNext(next) : "/";
  return target === "/" ? path : `${path}?next=${encodeURIComponent(target)}`;
}

/** `/login`, or `/login?next=<path>` (omitted for `/`). */
export function loginPath(next?: string | null): string {
  return withNext(LOGIN_PATH, next);
}

/** `/account/password`, or with `?next=<path>`. */
export function changePasswordPath(next?: string | null): string {
  return withNext(CHANGE_PASSWORD_PATH, next);
}

export interface RouteDecisionOptions {
  /** Current query string (`?a=b` or ""), kept in the `next` of a login redirect. */
  search?: string;
  /** `?next=` of the current URL (the /login and /account/password pages). Validated here. */
  next?: string | string[] | null;
}

export function routeDecision(pathname: string, session: SessionState, options: RouteDecisionOptions = {}): RouteDecision {
  const path = normalizePathname(pathname);
  const access = routeAccess(path);
  const here = `${path}${options.search && options.search !== "?" ? options.search : ""}`;

  if (session.status === "loading") return { kind: "loading" };

  if (access === "public") {
    if (session.status !== "authenticated" || !session.me) return { kind: "render" };
    const next = safeNext(options.next);
    if (session.me.must_change_password) return { kind: "redirect", to: changePasswordPath(next) };
    return { kind: "redirect", to: next };
  }

  if (session.status === "error") {
    return session.error ? { kind: "error", error: session.error } : { kind: "loading" };
  }
  if (session.status === "anonymous" || !session.me) return { kind: "redirect", to: loginPath(here) };

  if (access === "session") return { kind: "render" };
  if (session.me.must_change_password) return { kind: "redirect", to: changePasswordPath(here) };
  if (access === "admin" && session.me.role !== "admin") return { kind: "forbidden" };
  return { kind: "render" };
}

/**
 * Where to go after a successful sign-in or password change: the forced change first, then
 * `next` (validated) or `/`.
 */
export function afterAuthPath(me: { must_change_password: boolean }, next?: string | string[] | null): string {
  const target = safeNext(next);
  return me.must_change_password ? changePasswordPath(target) : target;
}
