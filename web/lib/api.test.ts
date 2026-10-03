// The real (non-mock) client against a stubbed fetch / XMLHttpRequest: credentials on every
// request, methods, 204, Retry-After, auth events, cookie_rejected, session error state.
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

delete process.env.NEXT_PUBLIC_API_MOCK;
delete process.env.NEXT_PUBLIC_API_BASE_URL;

const api = await import("./api");
const session = await import("./session");

interface Call {
  url: string;
  init: RequestInit;
}
let calls: Call[] = [];
let responder: (url: string, init: RequestInit) => Response | Promise<Response> = () => json(200, {});
const realFetch = globalThis.fetch;

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
}
const err = (code: string, message = "m") => ({ error: { code, message } });
const ME = { id: "a".repeat(32), username: "ann", role: "user", must_change_password: false, created_at: "2026-10-01T00:00:00Z" };
const ADMIN_USER = {
  id: "b".repeat(32),
  username: "bob",
  role: "user",
  is_active: true,
  must_change_password: false,
  created_at: "x",
  updated_at: "x",
  last_login_at: null,
  job_count: 0,
};

beforeEach(() => {
  calls = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    calls.push({ url, init });
    return responder(url, init);
  }) as typeof fetch;
  session.resetSessionForTests();
});
afterEach(() => {
  globalThis.fetch = realFetch;
  delete (globalThis as { window?: unknown }).window;
  delete (globalThis as { XMLHttpRequest?: unknown }).XMLHttpRequest;
});

describe("requests", () => {
  it("every fetch sends credentials, no-store, and the right method", async () => {
    responder = (url) => {
      if (url.endsWith("/api/auth/me")) return json(200, ME);
      if (url.endsWith("/api/auth/status")) return json(200, { setup_required: false });
      if (url.includes("/api/admin/users/") && url.endsWith("reset-password")) return json(200, { user: ADMIN_USER, temporary_password: "t".repeat(16) });
      if (url.endsWith("/api/admin/users")) return json(201, { user: ADMIN_USER, temporary_password: "t".repeat(16) });
      if (url.includes("/api/admin/users/")) return json(200, ADMIN_USER);
      if (url.includes("/api/jobs?") || url.endsWith("/api/jobs")) return json(200, { items: [], next_cursor: null });
      return json(204, null);
    };
    assert.equal((await api.getMe()).username, "ann");
    assert.deepEqual(await api.getAuthStatus(), { setup_required: false });
    assert.equal(await api.logout(), undefined);
    assert.equal(await api.deleteJob("j1"), undefined);
    assert.equal(await api.deleteUser("b".repeat(32)), undefined);
    assert.equal((await api.createUser({ username: "bob" })).user.username, "bob");
    assert.equal((await api.updateUser("b".repeat(32), { isActive: false })).username, "bob");
    assert.equal((await api.resetUserPassword("b".repeat(32))).temporary_password.length, 16);
    await api.listJobs({ owner: "me", status: "processing", limit: 20, cursor: "abc" });
    await api.listJobs();

    for (const c of calls) {
      assert.equal(c.init.credentials, "include", c.url);
      assert.equal(c.init.cache, "no-store", c.url);
    }
    const sig = calls.map((c) => `${c.init.method} ${c.url.replace("http://localhost:8000", "")}`);
    assert.deepEqual(sig, [
      "GET /api/auth/me",
      "GET /api/auth/status",
      "POST /api/auth/logout",
      "DELETE /api/jobs/j1",
      `DELETE /api/admin/users/${"b".repeat(32)}`,
      "POST /api/admin/users",
      `PATCH /api/admin/users/${"b".repeat(32)}`,
      `POST /api/admin/users/${"b".repeat(32)}/reset-password`,
      "GET /api/jobs?owner=me&status=processing&limit=20&cursor=abc",
      "GET /api/jobs",
    ]);
    assert.deepEqual(JSON.parse(String(calls[5].init.body)), { username: "bob", role: "user" });
    assert.deepEqual(JSON.parse(String(calls[6].init.body)), { is_active: false }); // only the given fields
  });

  it("login / changePassword send the contract bodies", async () => {
    responder = () => json(200, ME);
    await api.login("ann", "pw");
    await api.changePassword("old", "new password!!");
    assert.deepEqual(JSON.parse(String(calls[0].init.body)), { username: "ann", password: "pw" });
    assert.deepEqual(JSON.parse(String(calls[1].init.body)), { current_password: "old", new_password: "new password!!" });
  });

  it("health with null interpreter/limits (anonymous) is valid", async () => {
    responder = () => json(200, { status: "ok", version: "1", ffmpeg: true, interpreter: null, limits: null });
    const h = await api.getHealth();
    assert.equal(h.interpreter, null);
  });

  it("429 carries retryAfterS and is not transient", async () => {
    responder = () => json(429, err("too_many_attempts", "Too many."), { "Retry-After": "120" });
    await assert.rejects(api.login("ann", "x"), (e: unknown) => {
      assert.ok(e instanceof api.ApiError);
      assert.equal(e.code, "too_many_attempts");
      assert.equal(e.retryAfterS, 120);
      assert.equal(e.serverMessage, "Too many.");
      assert.equal(e.isTransient, false);
      return true;
    });
  });

  it("malformed bodies → invalid_response", async () => {
    responder = () => json(200, { nope: true });
    await assert.rejects(api.getMe(), (e: unknown) => e instanceof api.ApiError && e.code === "invalid_response");
    await assert.rejects(api.listUsers(), (e: unknown) => e instanceof api.ApiError && e.code === "invalid_response");
  });
});

describe("auth events", () => {
  it("401 unauthenticated and 403 password_change_required are broadcast; other errors are not", async () => {
    const events: string[] = [];
    const off = api.onAuthEvent((e) => events.push(e));
    responder = () => json(401, err("unauthenticated"));
    await assert.rejects(api.listJobs());
    responder = () => json(403, err("password_change_required"));
    await assert.rejects(api.getJob("x"));
    responder = () => json(401, err("invalid_credentials"));
    await assert.rejects(api.login("a", "b"));
    responder = () => json(403, err("forbidden"));
    await assert.rejects(api.checkInterpreter());
    responder = () => json(401, null); // no body → mapped from the status
    await assert.rejects(api.getMe());
    off();
    responder = () => json(401, err("unauthenticated"));
    await assert.rejects(api.getMe());
    assert.deepEqual(events, ["unauthenticated", "password_change_required", "unauthenticated"]);
  });

  it("XHR upload: withCredentials + auth event on 401", async () => {
    const events: string[] = [];
    const off = api.onAuthEvent((e) => events.push(e));
    const instances: FakeXhr[] = [];
    class FakeXhr {
      withCredentials = false;
      status = 0;
      responseText = "";
      responseType = "";
      upload: { onprogress: ((e: unknown) => void) | null } = { onprogress: null };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      ontimeout: (() => void) | null = null;
      onabort: (() => void) | null = null;
      method = "";
      url = "";
      constructor() {
        instances.push(this);
      }
      open(method: string, url: string) {
        this.method = method;
        this.url = url;
      }
      setRequestHeader() {}
      getResponseHeader() {
        return null;
      }
      abort() {}
      send() {
        this.status = 401;
        this.responseText = JSON.stringify(err("unauthenticated"));
        queueMicrotask(() => this.onload?.());
      }
    }
    (globalThis as { XMLHttpRequest?: unknown }).XMLHttpRequest = FakeXhr;
    const file = new File([new Uint8Array(8)], "a.mp4", { type: "video/mp4" });
    await assert.rejects(api.uploadVideo(file), (e: unknown) => e instanceof api.ApiError && e.code === "unauthenticated" && e.status === 401);
    off();
    assert.equal(instances.length, 1);
    assert.equal(instances[0].withCredentials, true);
    assert.equal(instances[0].method, "POST");
    assert.equal(instances[0].url, "http://localhost:8000/api/jobs");
    assert.deepEqual(events, ["unauthenticated"]);
  });
});

describe("hostnames + session against the real client", () => {
  it("hostnameMismatch", () => {
    assert.equal(api.apiHostname(), "localhost");
    assert.equal(api.hostnameMismatch("localhost"), null);
    assert.deepEqual(api.hostnameMismatch("127.0.0.1"), { pageHost: "127.0.0.1", apiHost: "localhost" });
    assert.equal(api.hostnameMismatch(null), null);
  });

  it("login OK but /me 401 → cookie_rejected naming both hostnames", async () => {
    (globalThis as { window?: unknown }).window = { location: { search: "", hostname: "127.0.0.1" } };
    responder = (url) => (url.endsWith("/api/auth/login") ? json(200, ME) : json(401, err("unauthenticated")));
    await assert.rejects(session.login("ann", "pw"), (e: unknown) => {
      assert.ok(e instanceof api.ApiError);
      assert.equal(e.code, "cookie_rejected");
      assert.match(e.message, /"127\.0\.0\.1".*"localhost"/);
      return true;
    });
    assert.equal(session.getSession().status, "anonymous");
  });

  it("API unreachable → error state; refreshSession recovers", async () => {
    responder = () => {
      throw new TypeError("Failed to fetch");
    };
    await session.loadSession();
    assert.equal(session.getSession().status, "error");
    assert.equal(session.getSession().error?.code, "network_error");
    responder = () => json(200, ME);
    await session.refreshSession();
    assert.equal(session.getSession().status, "authenticated");
  });

  it("logout failure keeps the session (server session would still be valid)", async () => {
    responder = () => json(200, ME);
    await session.loadSession();
    responder = () => {
      throw new TypeError("Failed to fetch");
    };
    await assert.rejects(session.logout(), (e: unknown) => e instanceof api.ApiError && e.code === "network_error");
    assert.equal(session.getSession().status, "authenticated");
  });
});
