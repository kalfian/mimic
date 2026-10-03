// Admin user management: pure row guards + the controller against the mock API.
import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

process.env.NEXT_PUBLIC_API_MOCK = "1";

const { activeAdminCount, AdminUsersController, deleteConfirmationMatches, userActionGuards } = await import("./adminUsers");
const { configureMock, resetMockData } = await import("./mock/mockApi");
const { mockSignInAs } = await import("./mock/mockAuth");
const { MOCK_USER_IDS } = await import("./mock/mockJobsRegistry");
type AdminUser = import("./types").AdminUser;

const u = (id: string, role: AdminUser["role"], is_active = true): AdminUser => ({
  id,
  username: id,
  role,
  is_active,
  must_change_password: false,
  created_at: "",
  updated_at: "",
  last_login_at: null,
  job_count: 0,
});

describe("userActionGuards", () => {
  it("own row: no disable / reset / delete; role change signs you out", () => {
    const users = [u("me", "admin"), u("other", "admin")];
    const g = userActionGuards(users[0], users, "me");
    assert.equal(g.isSelf, true);
    assert.deepEqual(g.can, { promote: false, demote: true, disable: false, enable: false, resetPassword: false, delete: false });
    assert.equal(g.roleChangeSignsOut, true);
    assert.ok(g.reasons.disable && g.reasons.resetPassword && g.reasons.delete);
  });

  it("last active admin: no demote / disable / delete, with a reason", () => {
    const users = [u("me", "admin"), u("solo", "admin"), u("off", "admin", false)];
    // `me` is not active-admin here only for the sake of the test: make solo the only active one
    users[0] = u("me", "user");
    const g = userActionGuards(users[1], users, "me");
    assert.equal(g.isLastActiveAdmin, true);
    assert.equal(g.can.demote, false);
    assert.equal(g.can.disable, false);
    assert.equal(g.can.delete, false);
    assert.equal(g.can.resetPassword, true);
    assert.match(g.reasons.demote ?? "", /last active admin/);
    assert.equal(activeAdminCount(users), 1);
  });

  it("regular user row", () => {
    const users = [u("me", "admin"), u("ann", "user"), u("dis", "user", false)];
    assert.deepEqual(userActionGuards(users[1], users, "me").can, {
      promote: true,
      demote: false,
      disable: true,
      enable: false,
      resetPassword: true,
      delete: true,
    });
    assert.equal(userActionGuards(users[2], users, "me").can.enable, true);
    assert.equal(userActionGuards(users[2], users, "me").can.disable, false);
  });

  it("delete confirmation", () => {
    assert.equal(deleteConfirmationMatches(" Alice ", "alice"), true);
    assert.equal(deleteConfirmationMatches("alic", "alice"), false);
  });
});

describe("AdminUsersController (mock API)", () => {
  beforeEach(() => {
    resetMockData();
    configureMock({ timeScale: 0, latencyMs: 0 });
    mockSignInAs("admin");
  });

  it("loads, creates (temporary password), updates, resets, deletes", async () => {
    const c = new AdminUsersController({ currentUserId: () => MOCK_USER_IDS.admin });
    c.start();
    await c.refresh();
    assert.equal(c.getSnapshot().isLoading, false);
    assert.equal(c.getSnapshot().users.length, 5);

    const created = await c.createUser({ username: "carol" });
    assert.ok(created.ok);
    if (created.ok) assert.equal(created.data.temporary_password.length, 16);
    assert.deepEqual(
      c.getSnapshot().users.map((x) => x.username),
      ["admin", "alice", "carol", "disabled", "newbie", "user"],
    );
    const dup = await c.createUser({ username: "carol" });
    assert.equal(dup.ok, false);
    if (!dup.ok) assert.equal(dup.error.code, "username_taken");

    const disabled = await c.updateUser(MOCK_USER_IDS.user, { isActive: false });
    assert.ok(disabled.ok && disabled.data.is_active === false);
    assert.equal(c.getSnapshot().users.find((x) => x.id === MOCK_USER_IDS.user)?.is_active, false);

    const self = await c.updateUser(MOCK_USER_IDS.admin, { isActive: false });
    assert.ok(!self.ok && self.error.code === "self_action_forbidden");
    const last = await c.updateUser(MOCK_USER_IDS.admin, { role: "user" });
    assert.ok(!last.ok && last.error.code === "last_admin");

    const reset = await c.resetPassword(MOCK_USER_IDS.alice);
    assert.ok(reset.ok && reset.data.user.must_change_password);

    const del = await c.deleteUser(MOCK_USER_IDS.alice);
    assert.ok(del.ok);
    assert.ok(!c.getSnapshot().users.some((x) => x.id === MOCK_USER_IDS.alice));
    c.stop();
  });

  it("changing the own role reports the revoked session", async () => {
    let revoked = 0;
    const c = new AdminUsersController({ currentUserId: () => MOCK_USER_IDS.admin, onOwnSessionRevoked: () => revoked++ });
    await c.refresh();
    assert.ok((await c.updateUser(MOCK_USER_IDS.alice, { role: "admin" })).ok);
    assert.equal(revoked, 0);
    assert.ok((await c.updateUser(MOCK_USER_IDS.admin, { role: "user" })).ok);
    assert.equal(revoked, 1);
  });

  it("non-admin: forbidden error state", async () => {
    mockSignInAs("user");
    const c = new AdminUsersController({ currentUserId: () => MOCK_USER_IDS.user });
    await c.refresh();
    assert.equal(c.getSnapshot().error?.code, "forbidden");
    assert.equal(c.getSnapshot().isLoading, false);
  });
});
