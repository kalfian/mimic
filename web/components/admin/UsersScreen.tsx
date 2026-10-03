"use client";

import { useRef, useState, type RefObject } from "react";

import { CreateUserDialog } from "@/components/admin/CreateUserDialog";
import { TemporaryPasswordReveal } from "@/components/admin/TemporaryPasswordReveal";
import { UserRows } from "@/components/admin/UserRows";
import { Forbidden } from "@/components/auth/Forbidden";
import { IconLock, IconPlus, IconRefresh, IconTrash } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { INPUT_CLASS } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import type { ApiError } from "@/lib/errors";
import { useSession } from "@/lib/session";
import type { AdminUser, UserRole } from "@/lib/types";
import { deleteConfirmationMatches, useAdminUsers } from "@/lib/useAdminUsers";

export type UserAction = "promote" | "demote" | "disable" | "enable" | "reset" | "delete";

type Pending =
  | { kind: "create" }
  | { kind: "reveal"; user: AdminUser; password: string; reason: "created" | "reset" }
  | { kind: "confirm"; action: Exclude<UserAction, "enable">; user: AdminUser };

/**
 * `/admin/users` (admins; `AuthGate` shows the 403 state to users). Row actions are offered by
 * `userActionGuards()` (own row / last active admin are disabled with a reason), and the server's
 * 409s are still shown in the dialog that caused them.
 */
export function UsersScreen() {
  const session = useSession();
  const me = session.me;
  const admin = useAdminUsers();
  const [pending, setPending] = useState<Pending | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [typed, setTyped] = useState("");
  const [result, setResult] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const createRef = useRef<HTMLButtonElement>(null);
  /** The control that started the current flow; focus goes back there when it ends. */
  const returnRef = useRef<HTMLElement | null>(null);

  if (!me) return null;

  if (admin.error?.code === "forbidden") return <Forbidden me={me} />;

  const close = () => {
    if (busy) return;
    setPending(null);
    setError(null);
    setTyped("");
  };

  const start = (next: Pending) => {
    const el = document.activeElement;
    returnRef.current = el instanceof HTMLElement && el !== document.body ? el : headingRef.current;
    setError(null);
    setTyped("");
    setPending(next);
  };

  const onRowAction = async (action: UserAction, user: AdminUser) => {
    setResult(null);
    if (action !== "enable") {
      start({ kind: "confirm", action, user });
      return;
    }
    const r = await admin.updateUser(user.id, { isActive: true });
    setResult(
      r.ok
        ? { tone: "ok", text: `${user.username} is enabled and can sign in again.` }
        : { tone: "danger", text: `Couldn’t enable ${user.username}: ${errorCopy(r.error).message}` },
    );
  };

  const create = async (input: { username: string; role: UserRole }): Promise<ApiError | null> => {
    const r = await admin.createUser(input);
    if (!r.ok) return r.error;
    setPending({ kind: "reveal", user: r.data.user, password: r.data.temporary_password, reason: "created" });
    return null;
  };

  const confirm = async () => {
    if (pending?.kind !== "confirm") return;
    const { action, user } = pending;
    setBusy(true);
    setError(null);
    let ok = false;
    let message = "";
    if (action === "reset") {
      const r = await admin.resetPassword(user.id);
      setBusy(false);
      if (!r.ok) {
        setError(r.error);
        return;
      }
      setPending({ kind: "reveal", user: r.data.user, password: r.data.temporary_password, reason: "reset" });
      return;
    }
    if (action === "delete") {
      const r = await admin.deleteUser(user.id);
      ok = r.ok;
      if (!r.ok) setError(r.error);
      else {
        message = `Deleted ${user.username}${user.job_count ? ` and ${jobCount(user.job_count)}` : ""}.`;
        returnRef.current = headingRef.current; // the row is gone
      }
    } else {
      const input = action === "disable" ? { isActive: false } : { role: (action === "promote" ? "admin" : "user") as UserRole };
      const r = await admin.updateUser(user.id, input);
      ok = r.ok;
      if (!r.ok) setError(r.error);
      else
        message =
          action === "disable"
            ? `${user.username} is disabled and was signed out.`
            : action === "promote"
              ? `${user.username} is now an admin.`
              : `${user.username} is now a user.`;
    }
    setBusy(false);
    if (ok) {
      setPending(null);
      setTyped("");
      setResult({ tone: "ok", text: message });
    }
  };

  const users = admin.users;
  const admins = users.filter((u) => u.role === "admin" && u.is_active).length;
  const disabled = users.filter((u) => !u.is_active).length;

  return (
    <div className="mx-auto max-w-[1240px] space-y-6 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0 space-y-1.5">
          <p className="caption">Admin</p>
          <h1 ref={headingRef} tabIndex={-1} className="text-xl font-semibold tracking-tight text-ink outline-none">
            Users
          </h1>
          <p className="text-sm text-ink-3">
            {admin.isLoading ? (
              "Loading accounts…"
            ) : (
              <>
                <span className="nums">{users.length}</span> {users.length === 1 ? "account" : "accounts"} · <span className="nums">{admins}</span>{" "}
                active {admins === 1 ? "admin" : "admins"}
                {disabled ? (
                  <>
                    {" "}
                    · <span className="nums">{disabled}</span> disabled
                  </>
                ) : null}
                {admin.isRefreshing ? " · updating…" : ""}
              </>
            )}
          </p>
        </div>
        <Button
          ref={createRef}
          variant="primary"
          size="sm"
          icon={<IconPlus />}
          onClick={() => start({ kind: "create" })}
          disabled={admin.isLoading && !admin.error}
        >
          Create user
        </Button>
      </header>

      {result ? (
        <Notice tone={result.tone} live={result.tone === "ok" ? "polite" : "assertive"} onDismiss={() => setResult(null)}>
          {result.text}
        </Notice>
      ) : null}

      {admin.isLoading ? (
        <div aria-busy="true" className="space-y-px overflow-hidden rounded-md border border-line">
          <p role="status" className="sr-only">
            Loading accounts
          </p>
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="hatch h-12" />
          ))}
        </div>
      ) : admin.error && users.length === 0 ? (
        <Notice
          tone="danger"
          live="assertive"
          title={errorCopy(admin.error).title}
          guidance={errorCopy(admin.error).guidance}
          actions={
            <Button size="sm" icon={<IconRefresh />} loading={admin.isRefreshing} onClick={() => void admin.refresh()}>
              Try again
            </Button>
          }
        >
          {errorCopy(admin.error).message}
        </Notice>
      ) : (
        <UserRows users={users} meId={me.id} onAction={(a, u) => void onRowAction(a, u)} />
      )}

      <CreateUserDialog open={pending?.kind === "create"} onCancel={close} onCreate={create} returnFocusRef={createRef} />

      <TemporaryPasswordReveal
        open={pending?.kind === "reveal"}
        user={pending?.kind === "reveal" ? pending.user : null}
        password={pending?.kind === "reveal" ? pending.password : ""}
        reason={pending?.kind === "reveal" ? pending.reason : "created"}
        onDone={close}
        returnFocusRef={returnRef}
      />

      {pending?.kind === "confirm" ? (
        <ConfirmFor
          action={pending.action}
          user={pending.user}
          self={pending.user.id === me.id}
          busy={busy}
          error={error}
          typed={typed}
          onType={setTyped}
          onCancel={close}
          onConfirm={() => void confirm()}
          returnFocusRef={returnRef}
        />
      ) : null}
    </div>
  );
}

function jobCount(n: number): string {
  return `${n} ${n === 1 ? "job" : "jobs"}`;
}

function ConfirmFor({
  action,
  user,
  self,
  busy,
  error,
  typed,
  onType,
  onCancel,
  onConfirm,
  returnFocusRef,
}: {
  action: Exclude<UserAction, "enable">;
  user: AdminUser;
  self: boolean;
  busy: boolean;
  error: ApiError | null;
  typed: string;
  onType: (v: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
  returnFocusRef: RefObject<HTMLElement | null>;
}) {
  const name = <span className="font-mono">{user.username}</span>;
  const common = { open: true, busy, error, onCancel, onConfirm, returnFocusRef };

  switch (action) {
    case "promote":
      return (
        <ConfirmDialog
          {...common}
          caption="Change role"
          title={<>Make {name} an admin?</>}
          description={
            <>
              Admins see every job and can create, reset, disable and delete accounts. {name} is signed out and gets the new role at the next sign-in.
            </>
          }
          confirmLabel="Make admin"
          busyLabel="Saving…"
        />
      );
    case "demote":
      return self ? (
        <ConfirmDialog
          {...common}
          caption="Change role"
          title="Remove your own admin role?"
          description={
            <>
              <strong className="font-medium text-ink">You will be signed out right away.</strong> When you sign in again you are a regular user: you
              see only your own jobs and can’t manage accounts. Only another admin can undo this.
            </>
          }
          confirmLabel="Remove and sign out"
          busyLabel="Saving…"
          confirmVariant="danger"
        />
      ) : (
        <ConfirmDialog
          {...common}
          caption="Change role"
          title={<>Make {name} a regular user?</>}
          description={
            <>
              {name} keeps their {jobCount(user.job_count)} but from now on sees only their own. They are signed out and sign in again.
            </>
          }
          confirmLabel="Make user"
          busyLabel="Saving…"
        />
      );
    case "disable":
      return (
        <ConfirmDialog
          {...common}
          caption="Disable account"
          title={<>Disable {name}?</>}
          description={
            <>
              {name} is signed out everywhere and can’t sign in until an admin enables the account again. Their {jobCount(user.job_count)} are kept.
            </>
          }
          confirmLabel="Disable account"
          busyLabel="Disabling…"
          confirmVariant="danger"
        />
      );
    case "reset":
      return (
        <ConfirmDialog
          {...common}
          caption="Reset password"
          title={<>Reset the password of {name}?</>}
          description={
            <>
              {name} is signed out everywhere. Mimic generates a temporary password, shown to you once, and {name} must replace it at the next
              sign-in.
            </>
          }
          confirmLabel="Reset password"
          busyLabel="Resetting…"
          confirmIcon={<IconLock />}
        />
      );
    case "delete": {
      const matches = deleteConfirmationMatches(typed, user.username);
      return (
        <ConfirmDialog
          {...common}
          caption="Delete account"
          title={<>Delete {name}?</>}
          description={
            user.job_count > 0 ? (
              <>
                This deletes the account <strong className="font-medium text-ink">and its {jobCount(user.job_count)}</strong>: recordings, results and
                keyframes. It can’t be undone.
              </>
            ) : (
              <>This deletes the account. It has no jobs. It can’t be undone.</>
            )
          }
          confirmLabel={user.job_count > 0 ? `Delete account and ${jobCount(user.job_count)}` : "Delete account"}
          busyLabel="Deleting…"
          confirmVariant="danger"
          confirmIcon={<IconTrash />}
          confirmDisabled={!matches}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (matches && !busy) onConfirm();
            }}
            className="space-y-1.5"
          >
            <label htmlFor="delete-confirm" className="block text-sm text-ink-2">
              Type <span className="font-mono font-medium text-ink select-all">{user.username}</span> to confirm
            </label>
            <input
              id="delete-confirm"
              value={typed}
              onChange={(e) => onType(e.target.value)}
              autoComplete="off"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              readOnly={busy}
              className={`${INPUT_CLASS} font-mono text-sm`}
            />
          </form>
        </ConfirmDialog>
      );
    }
  }
}
