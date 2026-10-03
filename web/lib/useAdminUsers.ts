/**
 * React hook for `/admin/users` (client components only), see `adminUsers.ts`.
 *
 *   const { users, isLoading, error, refresh, createUser, updateUser, resetPassword, deleteUser } = useAdminUsers();
 *   const r = await createUser({ username, role });
 *   if (r.ok) reveal(r.data.temporary_password); else showError(r.error);   // username_taken …
 *
 * Row actions: `userActionGuards(user, users, me.id)` (re-exported here).
 */

import { useEffect, useMemo, useSyncExternalStore } from "react";

import { AdminUsersController, type AdminUsersSnapshot } from "./adminUsers";

export { activeAdminCount, deleteConfirmationMatches, userActionGuards } from "./adminUsers";
export type { GuardedAction, MutationResult, UserActionGuards } from "./adminUsers";

export interface UseAdminUsersResult extends AdminUsersSnapshot {
  refresh: AdminUsersController["refresh"];
  createUser: AdminUsersController["createUser"];
  updateUser: AdminUsersController["updateUser"];
  resetPassword: AdminUsersController["resetPassword"];
  deleteUser: AdminUsersController["deleteUser"];
}

export function useAdminUsers(): UseAdminUsersResult {
  const controller = useMemo(() => new AdminUsersController(), []);

  useEffect(() => {
    controller.start();
    return controller.stop;
  }, [controller]);

  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
  return {
    ...snapshot,
    refresh: controller.refresh,
    createUser: controller.createUser,
    updateUser: controller.updateUser,
    resetPassword: controller.resetPassword,
    deleteUser: controller.deleteUser,
  };
}
