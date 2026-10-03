"use client";

import { IconLock, IconMore, IconSignOut, IconTrash, IconUser } from "@/components/icons";
import { useNow } from "@/components/job/useNow";
import type { UserAction } from "@/components/admin/UsersScreen";
import { Menu, type MenuEntry } from "@/components/ui/Menu";
import { formatLastLogin, USER_ROLE_LABELS } from "@/lib/format";
import type { AdminUser } from "@/lib/types";
import { userActionGuards, type UserActionGuards } from "@/lib/useAdminUsers";

const DATE = new Intl.DateTimeFormat("en", { day: "numeric", month: "short", year: "numeric" });
const DATE_TIME = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" });

function rowMenu(user: AdminUser, g: UserActionGuards, onAction: (a: UserAction, u: AdminUser) => void): MenuEntry[] {
  const items: MenuEntry[] = [];
  if (user.role === "user") {
    items.push({ key: "promote", label: "Make admin", icon: <IconUser />, onSelect: () => onAction("promote", user), disabled: !g.can.promote });
  } else {
    items.push({
      key: "demote",
      label: g.isSelf ? "Remove my admin role…" : "Make regular user",
      icon: <IconUser />,
      onSelect: () => onAction("demote", user),
      disabled: !g.can.demote,
      reason: g.reasons.demote,
    });
  }
  if (user.is_active) {
    items.push({
      key: "disable",
      label: "Disable account",
      icon: <IconSignOut />,
      onSelect: () => onAction("disable", user),
      disabled: !g.can.disable,
      reason: g.reasons.disable,
    });
  } else {
    items.push({ key: "enable", label: "Enable account", icon: <IconSignOut />, onSelect: () => onAction("enable", user), disabled: !g.can.enable });
  }
  items.push({
    key: "reset",
    label: "Reset password",
    icon: <IconLock />,
    onSelect: () => onAction("reset", user),
    disabled: !g.can.resetPassword,
    reason: g.reasons.resetPassword,
  });
  items.push({ key: "sep", separator: true });
  items.push({
    key: "delete",
    label: "Delete account",
    icon: <IconTrash />,
    tone: "danger",
    onSelect: () => onAction("delete", user),
    disabled: !g.can.delete,
    reason: g.reasons.delete,
  });
  return items;
}

function RoleBadge({ user, guards }: { user: AdminUser; guards: UserActionGuards }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      {user.role === "admin" ? (
        <span className="rounded-sm bg-ink px-1 font-mono text-2xs text-on-ink">admin</span>
      ) : (
        <span className="text-sm text-ink-2">{USER_ROLE_LABELS.user}</span>
      )}
      {guards.isLastActiveAdmin ? (
        <span className="text-2xs text-ink-3" title="Can't be demoted, disabled or deleted until another admin exists">
          last admin
        </span>
      ) : null}
    </span>
  );
}

function AccountState({ user }: { user: AdminUser }) {
  return (
    <span className="flex flex-col gap-1">
      <span className="flex items-center gap-2 text-sm">
        <span aria-hidden="true" className={`size-1.5 shrink-0 rounded-full ${user.is_active ? "bg-ok" : "bg-line-strong"}`} />
        <span className={user.is_active ? "text-ink" : "text-ink-3"}>{user.is_active ? "Active" : "Disabled"}</span>
      </span>
      {user.must_change_password ? (
        <span className="w-fit rounded-sm border border-warn/40 bg-warn-soft px-1 font-mono text-2xs text-warn">must change password</span>
      ) : null}
    </span>
  );
}

function Username({ user, isSelf }: { user: AdminUser; isSelf: boolean }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span className={`font-mono text-sm break-all ${user.is_active ? "text-ink" : "text-ink-3"}`}>{user.username}</span>
      {isSelf ? <span className="rounded-sm border border-line px-1 text-2xs text-ink-3">you</span> : null}
    </span>
  );
}

function LastLogin({ iso, now }: { iso: string | null; now: number | null }) {
  if (iso === null) return <span className="text-ink-3">Never</span>;
  return (
    <time dateTime={iso} title={DATE_TIME.format(Date.parse(iso))}>
      {now === null ? "…" : formatLastLogin(iso, now)}
    </time>
  );
}

function RowMenu({ user, guards, onAction }: { user: AdminUser; guards: UserActionGuards; onAction: (a: UserAction, u: AdminUser) => void }) {
  return (
    <Menu
      label={`Actions for ${user.username}`}
      trigger={<IconMore />}
      triggerClassName="focus-ring inline-flex size-8 items-center justify-center rounded-sm text-ink-3 transition-colors duration-150 hover:bg-surface-2 hover:text-ink aria-expanded:bg-surface-2 aria-expanded:text-ink"
      items={rowMenu(user, guards, onAction)}
      minWidth="15rem"
    />
  );
}

const TH = "caption px-3 py-2 text-left font-normal";

/** Accounts as a table (md+) and as cards (narrow screens). */
export function UserRows({ users, meId, onAction }: { users: AdminUser[]; meId: string; onAction: (a: UserAction, u: AdminUser) => void }) {
  const now = useNow(30_000);
  const rows = users.map((user) => ({ user, guards: userActionGuards(user, users, meId) }));

  return (
    <>
      <div className="hidden overflow-x-auto rounded-md border border-line bg-surface md:block">
        <table className="w-full min-w-[52rem] border-collapse text-sm">
          <caption className="sr-only">Accounts, sorted by username</caption>
          <thead className="border-b border-line">
            <tr>
              <th scope="col" className={`${TH} pl-4`}>
                Username
              </th>
              <th scope="col" className={`${TH} w-32`}>
                Role
              </th>
              <th scope="col" className={`${TH} w-48`}>
                Status
              </th>
              <th scope="col" className={`${TH} w-16 text-right`}>
                Jobs
              </th>
              <th scope="col" className={`${TH} w-32`}>
                Last sign-in
              </th>
              <th scope="col" className={`${TH} w-32`}>
                Created
              </th>
              <th scope="col" className={`${TH} w-14 pr-4`}>
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ user, guards }) => (
              <tr key={user.id} className="border-b border-line align-top transition-colors duration-150 last:border-b-0 hover:bg-surface-2/50">
                <th scope="row" className="py-3 pr-3 pl-4 text-left font-normal">
                  <Username user={user} isSelf={guards.isSelf} />
                </th>
                <td className="px-3 py-3">
                  <RoleBadge user={user} guards={guards} />
                </td>
                <td className="px-3 py-3">
                  <AccountState user={user} />
                </td>
                <td className="nums px-3 py-3 text-right text-xs text-ink-2">{user.job_count}</td>
                <td className="px-3 py-3 text-ink-2">
                  <LastLogin iso={user.last_login_at} now={now} />
                </td>
                <td className="px-3 py-3 text-ink-2">
                  <time dateTime={user.created_at}>{DATE.format(Date.parse(user.created_at))}</time>
                </td>
                <td className="py-2 pr-4 pl-3 text-right">
                  <RowMenu user={user} guards={guards} onAction={onAction} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ul className="divide-y divide-line rounded-md border border-line bg-surface md:hidden" aria-label="Accounts, sorted by username">
        {rows.map(({ user, guards }) => (
          <li key={user.id} className="flex items-start gap-3 py-3 pr-2 pl-4">
            <div className="min-w-0 flex-1 space-y-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <Username user={user} isSelf={guards.isSelf} />
                <RoleBadge user={user} guards={guards} />
              </div>
              <AccountState user={user} />
              <p className="flex flex-wrap gap-x-2 text-xs text-ink-3">
                <span>
                  <span className="nums">{user.job_count}</span> {user.job_count === 1 ? "job" : "jobs"}
                </span>
                <span aria-hidden="true">·</span>
                <span>
                  Last sign-in <LastLogin iso={user.last_login_at} now={now} />
                </span>
              </p>
            </div>
            <RowMenu user={user} guards={guards} onAction={onAction} />
          </li>
        ))}
      </ul>
    </>
  );
}
