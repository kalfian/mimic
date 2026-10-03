// Session store against the mock API (NEXT_PUBLIC_API_MOCK=1). node --test runs each file in its
// own process, so setting the env before the dynamic imports is safe.
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

process.env.NEXT_PUBLIC_API_MOCK = "1";

const api = await import("./api");
const session = await import("./session");
const { configureMock, resetMockData } = await import("./mock/mockApi");
const { mockSignInAs } = await import("./mock/mockAuth");
const { routeDecision } = await import("./routes");

const isCode = (code: string) => (e: unknown) => e instanceof api.ApiError && e.code === code;

beforeEach(() => {
  resetMockData();
  configureMock({ timeScale: 0, latencyMs: 0 });
  session.resetSessionForTests();
});
afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("session store", () => {
  it("starts loading, resolves anonymous without a session, notifies subscribers", async () => {
    assert.equal(session.getSession().status, "loading");
    const seen: string[] = [];
    const unsubscribe = session.subscribeSession(() => seen.push(session.getSession().status));
    const [a, b] = await Promise.all([session.loadSession(), session.loadSession()]); // idempotent
    assert.equal(a, b);
    assert.deepEqual(session.getSession(), { status: "anonymous", me: null, error: null });
    assert.deepEqual(seen, ["anonymous"]);
    unsubscribe();
  });

  it("restores an existing session (reload)", async () => {
    mockSignInAs("user");
    const s = await session.loadSession();
    assert.equal(s.status, "authenticated");
    assert.equal(s.me?.username, "user");
  });

  it("login → authenticated; logout → anonymous", async () => {
    await session.loadSession();
    const me = await session.login("Admin", "any password");
    assert.equal(me.role, "admin");
    assert.equal(session.getSession().status, "authenticated");
    assert.ok(session.isAdminSession(session.getSession()));
    await session.logout();
    assert.deepEqual(session.getSession(), { status: "anonymous", me: null, error: null });
    await assert.rejects(api.getMe(), isCode("unauthenticated"));
  });

  it("`wrong` → invalid_credentials, store stays anonymous", async () => {
    await session.loadSession();
    await assert.rejects(session.login("user", "wrong"), isCode("invalid_credentials"));
    assert.equal(session.getSession().status, "anonymous");
  });

  it("cookie_rejected: login accepted but /me is 401", async () => {
    (globalThis as { window?: unknown }).window = { location: { search: "?fail=cookie_rejected", hostname: "localhost" } };
    await session.loadSession();
    await assert.rejects(session.login("user", "x"), (e: unknown) => {
      assert.ok(e instanceof api.ApiError);
      assert.equal(e.code, "cookie_rejected");
      assert.equal(e.origin, "client");
      // What happened; the "same hostname" advice is the format.ts guidance (shown once).
      assert.match(e.message, /did not keep the session cookie/);
      return true;
    });
    assert.equal(session.getSession().status, "anonymous");
  });

  it("an unauthenticated response anywhere → anonymous with a 'session expired' error", async () => {
    mockSignInAs("user");
    await session.loadSession();
    mockSignInAs(null); // the server revoked the session (disabled, expired …)
    await assert.rejects(api.listJobs(), isCode("unauthenticated"));
    const s = session.getSession();
    assert.equal(s.status, "anonymous");
    assert.equal(s.error?.code, "unauthenticated");
    assert.deepEqual(routeDecision("/jobs", s), { kind: "redirect", to: "/login?next=%2Fjobs" });
    // a later /me 401 keeps the "expired" reason
    await session.refreshSession();
    assert.equal(session.getSession().error?.code, "unauthenticated");
  });

  it("forced password change: login → routing to /account/password → changePassword → full session", async () => {
    await session.loadSession();
    const me = await session.login("newbie", "temporary");
    assert.equal(me.must_change_password, true);
    assert.deepEqual(routeDecision("/", session.getSession()), { kind: "redirect", to: "/account/password" });
    await assert.rejects(session.changePassword("temporary", "short"), isCode("weak_password"));
    const after = await session.changePassword("temporary", "a much longer passphrase");
    assert.equal(after.must_change_password, false);
    assert.equal(session.getSession().me?.must_change_password, false);
    assert.deepEqual(routeDecision("/", session.getSession()), { kind: "render" });
  });

  it("password_change_required from any call flips must_change_password", async () => {
    mockSignInAs("user");
    await session.loadSession();
    api.notifyAuthEvent("password_change_required", new api.ApiError("password_change_required", "x", { status: 403 }));
    assert.equal(session.getSession().me?.must_change_password, true);
    // then /me (still false in the mock) confirms the real state
    await new Promise((r) => setTimeout(r, 5));
    assert.equal(session.getSession().me?.must_change_password, false);
  });
});
