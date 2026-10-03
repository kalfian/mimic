// Route policy (PLAN-auth §8.1) + safeNext open-redirect cases.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiError } from "./errors";
import { afterAuthPath, changePasswordPath, loginPath, routeAccess, routeDecision, safeNext } from "./routes";
import type { SessionState } from "./session";
import type { Me } from "./types";

const me = (role: Me["role"], must_change_password = false): Me => ({
  id: "a0000000000000000000000000000009",
  username: role === "admin" ? "root" : "ann",
  role,
  must_change_password,
  created_at: "2026-10-01T00:00:00Z",
});
const loading: SessionState = { status: "loading", me: null, error: null };
const anonymous: SessionState = { status: "anonymous", me: null, error: null };
const failed: SessionState = { status: "error", me: null, error: ApiError.network() };
const user: SessionState = { status: "authenticated", me: me("user"), error: null };
const admin: SessionState = { status: "authenticated", me: me("admin"), error: null };
const forcedUser: SessionState = { status: "authenticated", me: me("user", true), error: null };
const forcedAdmin: SessionState = { status: "authenticated", me: me("admin", true), error: null };

describe("routeAccess", () => {
  it("classifies paths", () => {
    assert.equal(routeAccess("/login"), "public");
    assert.equal(routeAccess("/login/"), "public");
    assert.equal(routeAccess("/account/password"), "session");
    assert.equal(routeAccess("/admin"), "admin");
    assert.equal(routeAccess("/admin/users"), "admin");
    assert.equal(routeAccess("/administrator"), "user"); // prefix must be a whole segment
    for (const p of ["/", "/jobs", "/jobs/abc", "/whatever"]) assert.equal(routeAccess(p), "user");
  });
});

describe("routeDecision", () => {
  it("loading session → loading on every route (no protected flash)", () => {
    for (const p of ["/", "/login", "/account/password", "/jobs", "/admin/users"]) assert.deepEqual(routeDecision(p, loading), { kind: "loading" });
  });

  it("/login", () => {
    assert.deepEqual(routeDecision("/login", anonymous), { kind: "render" });
    assert.deepEqual(routeDecision("/login", failed), { kind: "render" }); // the form shows its own errors
    assert.deepEqual(routeDecision("/login", user), { kind: "redirect", to: "/" });
    assert.deepEqual(routeDecision("/login", user, { next: "/jobs?status=failed" }), { kind: "redirect", to: "/jobs?status=failed" });
    assert.deepEqual(routeDecision("/login", user, { next: "//evil.com" }), { kind: "redirect", to: "/" });
    assert.deepEqual(routeDecision("/login", forcedUser, { next: "/jobs" }), { kind: "redirect", to: "/account/password?next=%2Fjobs" });
    assert.deepEqual(routeDecision("/login", forcedUser), { kind: "redirect", to: "/account/password" });
  });

  it("/account/password allows the forced-change state", () => {
    assert.deepEqual(routeDecision("/account/password", anonymous), { kind: "redirect", to: "/login?next=%2Faccount%2Fpassword" });
    assert.deepEqual(routeDecision("/account/password", forcedUser), { kind: "render" });
    assert.deepEqual(routeDecision("/account/password", forcedAdmin), { kind: "render" });
    assert.deepEqual(routeDecision("/account/password", user), { kind: "render" });
    assert.equal(routeDecision("/account/password", failed).kind, "error");
  });

  it("full-session pages: /, /jobs, /jobs/[id]", () => {
    for (const p of ["/", "/jobs", "/jobs/3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e"]) {
      assert.deepEqual(routeDecision(p, user), { kind: "render" });
      assert.deepEqual(routeDecision(p, admin), { kind: "render" });
      assert.equal(routeDecision(p, failed).kind, "error");
    }
    assert.deepEqual(routeDecision("/", anonymous), { kind: "redirect", to: "/login" });
    assert.deepEqual(routeDecision("/jobs", anonymous, { search: "?owner=me" }), { kind: "redirect", to: "/login?next=%2Fjobs%3Fowner%3Dme" });
    assert.deepEqual(routeDecision("/jobs/x", forcedUser), { kind: "redirect", to: "/account/password?next=%2Fjobs%2Fx" });
    assert.deepEqual(routeDecision("/", forcedAdmin), { kind: "redirect", to: "/account/password" });
  });

  it("/admin/* is admin-only: users get forbidden (not a redirect)", () => {
    assert.deepEqual(routeDecision("/admin/users", admin), { kind: "render" });
    assert.deepEqual(routeDecision("/admin/users", user), { kind: "forbidden" });
    assert.deepEqual(routeDecision("/admin/users/", user), { kind: "forbidden" });
    assert.deepEqual(routeDecision("/admin/users", anonymous), { kind: "redirect", to: "/login?next=%2Fadmin%2Fusers" });
    assert.deepEqual(routeDecision("/admin/users", forcedAdmin), { kind: "redirect", to: "/account/password?next=%2Fadmin%2Fusers" });
  });

  it("error state carries the error", () => {
    const d = routeDecision("/jobs", failed);
    assert.equal(d.kind, "error");
    if (d.kind === "error") assert.equal(d.error.code, "network_error");
  });
});

describe("safeNext", () => {
  it("accepts same-app paths", () => {
    assert.equal(safeNext("/jobs"), "/jobs");
    assert.equal(safeNext("/jobs/abc?x=1#t"), "/jobs/abc?x=1#t");
    assert.equal(safeNext(["/admin/users", "/other"]), "/admin/users");
    assert.equal(safeNext("/a/../jobs"), "/jobs"); // normalized by URL
  });

  it("rejects everything else", () => {
    const bad = [
      null,
      undefined,
      "",
      "jobs",
      "//evil.com",
      "///evil.com",
      "https://evil.com",
      "http:/evil.com",
      "javascript:alert(1)",
      "/\\evil.com",
      "\\\\evil.com",
      "/%2F%2Fevil.com",
      "/%2fevil.com",
      "/%5Cevil.com",
      "%2F%2Fevil.com",
      "/\t/evil.com",
      "/\n/evil.com",
      "/%E0%A4%A", // malformed escape
      "/login",
      "/login?next=/jobs",
      `/${"a".repeat(3000)}`,
    ];
    for (const raw of bad) assert.equal(safeNext(raw), "/", `safeNext(${JSON.stringify(raw)})`);
    assert.equal(safeNext("//evil.com", "/jobs"), "/jobs");
  });

  it("login/password paths and afterAuthPath", () => {
    assert.equal(loginPath(), "/login");
    assert.equal(loginPath("/"), "/login");
    assert.equal(loginPath("//evil"), "/login");
    assert.equal(loginPath("/jobs"), "/login?next=%2Fjobs");
    assert.equal(changePasswordPath("/jobs"), "/account/password?next=%2Fjobs");
    assert.equal(afterAuthPath({ must_change_password: false }, "/jobs"), "/jobs");
    assert.equal(afterAuthPath({ must_change_password: true }, "/jobs"), "/account/password?next=%2Fjobs");
    assert.equal(afterAuthPath({ must_change_password: false }, "https://evil.com"), "/");
  });
});
