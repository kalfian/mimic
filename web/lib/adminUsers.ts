/**
 * Framework-free user management for `/admin/users` (PLAN-auth §8.1): the account list, the
 * mutations, and the pure guards that decide which row actions are offered. `useAdminUsers` is
 * the React wrapper.
 *
 * Mutations never throw: they resolve `{ ok: true, data }` or `{ ok: false, error }` so dialogs
 * can show `last_admin`, `self_action_forbidden`, `username_taken` … inline. The guards mirror the
 * server's, but the server stays authoritative (another admin may change things meanwhile), so
 * always handle the 409s too. After a 404/409 the list refreshes itself.
 */

import {
  createUser,
  deleteUser,
  listUsers,
  resetUserPassword,
  updateUser,
  type CreateUserInput,
  type UpdateUserInput,
} from "./api";
import { ApiError } from "./errors";
import { getSession, refreshSession } from "./session";
import type { AdminUser, AdminUserList, TemporaryPassword } from "./types";

export type MutationResult<T> = { ok: true; data: T } | { ok: false; error: ApiError };

export interface AdminUsersClient {
  listUsers(signal?: AbortSignal): Promise<AdminUserList>;
  createUser(input: CreateUserInput): Promise<TemporaryPassword>;
  updateUser(id: string, input: UpdateUserInput): Promise<AdminUser>;
  resetUserPassword(id: string): Promise<TemporaryPassword>;
  deleteUser(id: string): Promise<void>;
}

export interface AdminUsersOptions {
  client?: AdminUsersClient;
  /** Id of the signed-in admin (default: the session store). */
  currentUserId?: () => string | null;
  /** Called after changing the own role (the server revoked our sessions). Default `refreshSession`. */
  onOwnSessionRevoked?: () => void;
}

export interface AdminUsersSnapshot {
  /** Sorted by username. */
  users: AdminUser[];
  /** First load in progress. */
  isLoading: boolean;
  isRefreshing: boolean;
  /** Last load failure (`forbidden` for non-admins, `network_error` …); users are kept. */
  error: ApiError | null;
}

const defaultClient: AdminUsersClient = { listUsers, createUser, updateUser, resetUserPassword, deleteUser };

/* ---------- pure guards ---------- */

export function activeAdminCount(users: readonly AdminUser[]): number {
  return users.filter((u) => u.role === "admin" && u.is_active).length;
}

export type GuardedAction = "promote" | "demote" | "disable" | "enable" | "resetPassword" | "delete";

export interface UserActionGuards {
  isSelf: boolean;
  /** The only active admin: can't be demoted, disabled or deleted. */
  isLastActiveAdmin: boolean;
  can: Record<GuardedAction, boolean>;
  /** Why an action that applies to this row is unavailable (tooltip / helper text). */
  reasons: Partial<Record<GuardedAction, string>>;
  /** Changing the own role signs you out (sessions revoked): confirm first. */
  roleChangeSignsOut: boolean;
}

const LAST_ADMIN_REASON = "This is the last active admin. Make another user an admin first.";

/** Which row actions to offer for `user`, as seen by the admin `meId`. Mirrors the API guards. */
export function userActionGuards(user: AdminUser, users: readonly AdminUser[], meId: string | null): UserActionGuards {
  const isSelf = user.id === meId;
  const isLastActiveAdmin = user.role === "admin" && user.is_active && activeAdminCount(users.filter((u) => u.id !== user.id)) === 0;
  const reasons: Partial<Record<GuardedAction, string>> = {};
  const can: Record<GuardedAction, boolean> = {
    promote: user.role === "user",
    demote: user.role === "admin" && !isLastActiveAdmin,
    disable: user.is_active && !isSelf && !isLastActiveAdmin,
    enable: !user.is_active,
    resetPassword: !isSelf,
    delete: !isSelf && !isLastActiveAdmin,
  };
  if (user.role === "admin" && isLastActiveAdmin) reasons.demote = LAST_ADMIN_REASON;
  if (user.is_active) {
    if (isSelf) reasons.disable = "You can't disable your own account.";
    else if (isLastActiveAdmin) reasons.disable = LAST_ADMIN_REASON;
  }
  if (isSelf) {
    reasons.resetPassword = "Change your own password from the account menu.";
    reasons.delete = "You can't delete your own account.";
  } else if (isLastActiveAdmin) {
    reasons.delete = LAST_ADMIN_REASON;
  }
  return { isSelf, isLastActiveAdmin, can, reasons, roleChangeSignsOut: isSelf };
}

/** The delete dialog requires typing the username (case/whitespace-insensitive). */
export function deleteConfirmationMatches(typed: string, username: string): boolean {
  return typed.trim().toLowerCase() === username;
}

function sortUsers(users: AdminUser[]): AdminUser[] {
  return [...users].sort((a, b) => (a.username < b.username ? -1 : a.username > b.username ? 1 : 0));
}

/* ---------- controller ---------- */

const INITIAL: AdminUsersSnapshot = Object.freeze({ users: [], isLoading: true, isRefreshing: false, error: null }) as AdminUsersSnapshot;

export class AdminUsersController {
  private snapshot: AdminUsersSnapshot = INITIAL;
  private readonly listeners = new Set<() => void>();
  private readonly client: AdminUsersClient;
  private readonly currentUserId: () => string | null;
  private readonly onOwnSessionRevoked: () => void;
  private loadController: AbortController | null = null;
  private running = false;

  constructor(options: AdminUsersOptions = {}) {
    this.client = options.client ?? defaultClient;
    this.currentUserId = options.currentUserId ?? (() => getSession().me?.id ?? null);
    this.onOwnSessionRevoked = options.onOwnSessionRevoked ?? (() => void refreshSession());
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): AdminUsersSnapshot => this.snapshot;

  start = (): void => {
    if (this.running) return;
    this.running = true;
    void this.refresh();
  };

  stop = (): void => {
    this.running = false;
    this.loadController?.abort();
    this.loadController = null;
    if (this.snapshot.isRefreshing) this.set({ isRefreshing: false });
  };

  /** Reload the list. Never rejects. */
  refresh = async (): Promise<void> => {
    this.loadController?.abort();
    const controller = new AbortController();
    this.loadController = controller;
    if (!this.snapshot.isLoading) this.set({ isRefreshing: true });
    try {
      const list = await this.client.listUsers(controller.signal);
      if (controller.signal.aborted) return;
      this.set({ users: sortUsers(list.items), isLoading: false, isRefreshing: false, error: null });
    } catch (err) {
      if (controller.signal.aborted) return;
      const e = ApiError.from(err);
      if (e.code === "aborted") return;
      this.set({ isLoading: false, isRefreshing: false, error: e });
    } finally {
      if (this.loadController === controller) this.loadController = null;
    }
  };

  /** Create an account. `data.temporary_password` is shown once (never store it). */
  createUser = (input: CreateUserInput): Promise<MutationResult<TemporaryPassword>> =>
    this.mutate(
      () => this.client.createUser(input),
      (created) => this.upsert(created.user),
    );

  /** Change role and/or enable/disable. Changing the own role signs you out. */
  updateUser = (id: string, input: UpdateUserInput): Promise<MutationResult<AdminUser>> => {
    const before = this.snapshot.users.find((u) => u.id === id);
    return this.mutate(
      () => this.client.updateUser(id, input),
      (user) => {
        this.upsert(user);
        if (id === this.currentUserId() && before && before.role !== user.role) this.onOwnSessionRevoked();
      },
    );
  };

  /** New temporary password (shown once); the account must change it at next sign-in. */
  resetPassword = (id: string): Promise<MutationResult<TemporaryPassword>> =>
    this.mutate(
      () => this.client.resetUserPassword(id),
      (reset) => this.upsert(reset.user),
    );

  /** Delete the account and all its jobs. */
  deleteUser = (id: string): Promise<MutationResult<void>> =>
    this.mutate(
      () => this.client.deleteUser(id),
      () => this.set({ users: this.snapshot.users.filter((u) => u.id !== id) }),
    );

  private async mutate<T>(run: () => Promise<T>, apply: (data: T) => void): Promise<MutationResult<T>> {
    try {
      const data = await run();
      apply(data);
      return { ok: true, data };
    } catch (err) {
      const error = ApiError.from(err);
      // The list is stale (user deleted, another admin changed roles): reload it in the background.
      if (error.status === 404 || error.status === 409) void this.refresh();
      return { ok: false, error };
    }
  }

  private upsert(user: AdminUser): void {
    this.set({ users: sortUsers([...this.snapshot.users.filter((u) => u.id !== user.id), user]) });
  }

  private set(patch: Partial<AdminUsersSnapshot>): void {
    this.snapshot = Object.freeze({ ...this.snapshot, ...patch }) as AdminUsersSnapshot;
    for (const l of [...this.listeners]) l();
  }
}
