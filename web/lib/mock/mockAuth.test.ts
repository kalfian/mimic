// The mock accounts mirror the backend guards (PLAN-auth §3.3, §4, §8.3).
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { ApiError } from "../errors";
import { configureMock, mockGetJob, mockListJobs, resetMockData } from "./mockApi";
import {
  mockChangePassword,
  mockCreateUser,
  mockDeleteUser,
  mockFailureFor,
  mockGetAuthStatus,
  mockGetMe,
  mockListUsers,
  mockLogin,
  mockLogout,
  mockResetUserPassword,
  mockSignInAs,
  mockUpdateUser,
  readMockAuthFailure,
  MOCK_RETRY_AFTER_S,
} from "./mockAuth";
import { MOCK_USER_IDS } from "./mockJobsRegistry";

const rejectsWith = (p: Promise<unknown>, code: string, status?: number) =>
  assert.rejects(p, (e: unknown) => e instanceof ApiError && e.code === code && (status === undefined || e.status === status));

/** Pretend to be on a page with this query string (mock scenarios read `window.location.search`). */
function onPage(search: string) {
  (globalThis as { window?: unknown }).window = { location: { search, hostname: "localhost" } };
}

beforeEach(() => {
  resetMockData();
  configureMock({ timeScale: 0, latencyMs: 0 });
});
afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("mock login", () => {
  it("any password except `wrong`; username is normalized; session kept", async () => {
    await rejectsWith(mockGetMe(), "unauthenticated", 401);
    const me = await mockLogin("  Admin ", "anything at all");
    assert.equal(me.username, "admin");
    assert.equal(me.role, "admin");
    assert.equal((await mockGetMe()).id, MOCK_USER_IDS.admin);
    await mockLogout();
    await rejectsWith(mockGetMe(), "unauthenticated");
    await mockLogout(); // idempotent
  });

  it("wrong password / unknown user → invalid_credentials; disabled → account_disabled; empty → invalid_request", async () => {
    await rejectsWith(mockLogin("user", "wrong"), "invalid_credentials", 401);
    await rejectsWith(mockLogin("nobody", "x"), "invalid_credentials", 401);
    await rejectsWith(mockLogin("disabled", "wrong"), "invalid_credentials"); // wrong password first
    await rejectsWith(mockLogin("disabled", "right"), "account_disabled", 403);
    await rejectsWith(mockLogin("user", ""), "invalid_request", 422);
  });

  it("5 failures → too_many_attempts with Retry-After; success resets", async () => {
    for (let i = 0; i < 5; i++) await rejectsWith(mockLogin("alice", "wrong"), "invalid_credentials");
    await assert.rejects(mockLogin("alice", "right"), (e: unknown) => {
      assert.ok(e instanceof ApiError);
      assert.equal(e.code, "too_many_attempts");
      assert.equal(e.status, 429);
      assert.ok(e.retryAfterS != null && e.retryAfterS > 800 && e.retryAfterS <= 900, String(e.retryAfterS));
      assert.equal(e.isTransient, false);
      return true;
    });
    // other usernames are not affected
    await mockLogin("user", "right");
  });

  it("newbie must change password: job routes 403 until changed", async () => {
    const me = await mockLogin("newbie", "temp");
    assert.equal(me.must_change_password, true);
    await rejectsWith(mockListJobs(), "password_change_required", 403);
    await rejectsWith(mockChangePassword("wrong", "a long enough passphrase"), "current_password_incorrect", 422);
    await rejectsWith(mockChangePassword("temp", "short"), "weak_password", 422);
    await assert.rejects(mockChangePassword("same passphrase!", "same passphrase!"), (e: unknown) => {
      return e instanceof ApiError && e.code === "weak_password" && /different/.test(e.message);
    });
    const after = await mockChangePassword("temp", "a long enough passphrase");
    assert.equal(after.must_change_password, false);
    assert.deepEqual((await mockListJobs()).items, []);
  });

  it("weak_password: password equal to the username (case-insensitive)", async () => {
    mockSignInAs("admin");
    const { user } = await mockCreateUser({ username: "longusername1", role: "user" });
    mockSignInAs(user.username);
    await rejectsWith(mockChangePassword("x", "LongUsername1"), "weak_password");
  });
});

describe("mock admin", () => {
  it("non-admins and forced-change admins are rejected", async () => {
    await rejectsWith(mockListUsers(), "unauthenticated", 401);
    mockSignInAs("user");
    await rejectsWith(mockListUsers(), "forbidden", 403);
    mockSignInAs("admin");
    const { user } = await mockCreateUser({ username: "boss", role: "admin" });
    assert.equal(user.must_change_password, true);
    mockSignInAs("boss");
    await rejectsWith(mockListUsers(), "password_change_required", 403);
  });

  it("list is sorted with job counts", async () => {
    mockSignInAs("admin");
    const { items } = await mockListUsers();
    assert.deepEqual(
      items.map((u) => u.username),
      ["admin", "alice", "disabled", "newbie", "user"],
    );
    assert.equal(items.find((u) => u.username === "alice")?.job_count, 2);
    assert.equal(items.find((u) => u.username === "user")?.job_count, 2);
    assert.equal(items.find((u) => u.username === "newbie")?.last_login_at, null);
  });

  it("create: normalized, validated, unique, temporary password shown once", async () => {
    mockSignInAs("admin");
    const created = await mockCreateUser({ username: " Bob.Smith ", role: "user" });
    assert.equal(created.user.username, "bob.smith");
    assert.equal(created.user.must_change_password, true);
    assert.match(created.temporary_password, /^[A-Za-z0-9_-]{16}$/);
    await rejectsWith(mockCreateUser({ username: "bob.smith" }), "username_taken", 409);
    await rejectsWith(mockCreateUser({ username: "ab" }), "invalid_request", 422);
    await rejectsWith(mockCreateUser({ username: "-bad" }), "invalid_request", 422);
    const again = await mockResetUserPassword(created.user.id);
    assert.notEqual(again.temporary_password, created.temporary_password);
  });

  it("last active admin can't be demoted, disabled or deleted", async () => {
    mockSignInAs("admin");
    const id = MOCK_USER_IDS.admin;
    await rejectsWith(mockUpdateUser(id, { role: "user" }), "last_admin", 409);
    await rejectsWith(mockUpdateUser(id, { is_active: false }), "self_action_forbidden", 409);
    // with a second admin, demoting yourself works and signs you out (sessions revoked)
    await mockUpdateUser(MOCK_USER_IDS.alice, { role: "admin" });
    const demoted = await mockUpdateUser(id, { role: "user" });
    assert.equal(demoted.role, "user");
    await rejectsWith(mockGetMe(), "unauthenticated");
    // alice is now the last active admin
    mockSignInAs("alice");
    await rejectsWith(mockDeleteUser(MOCK_USER_IDS.alice), "self_action_forbidden");
    await mockUpdateUser(MOCK_USER_IDS.user, { role: "admin" });
    await mockUpdateUser(MOCK_USER_IDS.user, { is_active: false }); // disabled admin doesn't count
    mockSignInAs("alice");
    const list = await mockListUsers();
    assert.equal(list.items.filter((u) => u.role === "admin" && u.is_active).length, 1);
  });

  it("self guards; not_found; disable revokes; enable; no-op", async () => {
    mockSignInAs("admin");
    await rejectsWith(mockResetUserPassword(MOCK_USER_IDS.admin), "self_action_forbidden", 409);
    await rejectsWith(mockDeleteUser(MOCK_USER_IDS.admin), "self_action_forbidden", 409);
    await rejectsWith(mockUpdateUser("f".repeat(32), { role: "user" }), "not_found", 404);
    await rejectsWith(mockUpdateUser("not-an-id", { role: "user" }), "not_found", 404);
    await rejectsWith(mockUpdateUser(MOCK_USER_IDS.user, {}), "invalid_request", 422);
    const disabled = await mockUpdateUser(MOCK_USER_IDS.user, { is_active: false });
    assert.equal(disabled.is_active, false);
    await rejectsWith(mockLogin("user", "x"), "account_disabled");
    mockSignInAs("admin");
    assert.equal((await mockUpdateUser(MOCK_USER_IDS.user, { is_active: true })).is_active, true);
    assert.equal((await mockUpdateUser(MOCK_USER_IDS.user, { role: "user" })).role, "user");
  });

  it("delete removes the account and its jobs", async () => {
    mockSignInAs("admin");
    const { items } = await mockListJobs({ owner: MOCK_USER_IDS.alice });
    assert.equal(items.length, 2);
    await mockDeleteUser(MOCK_USER_IDS.alice);
    assert.equal((await mockListJobs({ owner: MOCK_USER_IDS.alice })).items.length, 0);
    await rejectsWith(mockGetJob(items[0].id), "not_found");
    await rejectsWith(mockLogin("alice", "x"), "invalid_credentials");
    mockSignInAs("admin");
    assert.ok(!(await mockListUsers()).items.some((u) => u.username === "alice"));
  });
});

describe("?fail= scenarios", () => {
  it("reads only auth/admin codes and applies them per operation", () => {
    assert.equal(readMockAuthFailure("?fail=last_admin"), "last_admin");
    assert.equal(readMockAuthFailure("?fail=too_long"), null); // upload scenario, not ours
    assert.equal(mockFailureFor("updateUser", "?fail=last_admin"), "last_admin");
    assert.equal(mockFailureFor("createUser", "?fail=last_admin"), null);
    assert.equal(mockFailureFor("login", "?fail=cookie_rejected"), "cookie_rejected");
  });

  it("login scenarios", async () => {
    onPage("?fail=setup_required");
    assert.deepEqual(await mockGetAuthStatus(), { setup_required: true });
    await rejectsWith(mockLogin("admin", "x"), "setup_required", 409);

    onPage("?fail=too_many_attempts");
    await assert.rejects(mockLogin("admin", "x"), (e: unknown) => e instanceof ApiError && e.code === "too_many_attempts" && e.retryAfterS === MOCK_RETRY_AFTER_S);

    for (const [code, status] of [
      ["invalid_credentials", 401],
      ["account_disabled", 403],
      ["origin_not_allowed", 403],
    ] as const) {
      onPage(`?fail=${code}`);
      await rejectsWith(mockLogin("admin", "x"), code, status);
    }
    onPage("?fail=network_error");
    await rejectsWith(mockLogin("admin", "x"), "network_error");

    onPage("?fail=cookie_rejected");
    const me = await mockLogin("admin", "x"); // the API said yes …
    assert.equal(me.username, "admin");
    await rejectsWith(mockGetMe(), "unauthenticated"); // … but no session stuck
  });

  it("admin + password scenarios", async () => {
    mockSignInAs("admin");
    onPage("?fail=username_taken");
    await rejectsWith(mockCreateUser({ username: "fresh" }), "username_taken");
    onPage("?fail=last_admin");
    await rejectsWith(mockUpdateUser(MOCK_USER_IDS.user, { is_active: false }), "last_admin");
    await rejectsWith(mockDeleteUser(MOCK_USER_IDS.user), "last_admin");
    onPage("?fail=self_action_forbidden");
    await rejectsWith(mockResetUserPassword(MOCK_USER_IDS.user), "self_action_forbidden");
    onPage("?fail=not_found");
    await rejectsWith(mockUpdateUser(MOCK_USER_IDS.user, { role: "admin" }), "not_found");
    onPage("?fail=weak_password");
    await rejectsWith(mockChangePassword("a", "b"), "weak_password");
    onPage("?fail=current_password_incorrect");
    await rejectsWith(mockChangePassword("a", "b"), "current_password_incorrect");
  });

  it("list failures happen once per page load (Try again works)", async () => {
    mockSignInAs("admin");
    onPage("?fail=network_error");
    await rejectsWith(mockListUsers(), "network_error");
    assert.ok((await mockListUsers()).items.length > 0);
    await rejectsWith(mockListJobs(), "network_error");
    assert.ok((await mockListJobs()).items.length > 0);

    onPage("?fail=unauthenticated");
    await rejectsWith(mockListJobs(), "unauthenticated", 401);
    await rejectsWith(mockGetMe(), "unauthenticated"); // the mock session was dropped too
  });
});
