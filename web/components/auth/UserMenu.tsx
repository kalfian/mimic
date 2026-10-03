"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { IconChevronDown, IconLock, IconRefresh, IconSignOut } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { Menu, type MenuEntry } from "@/components/ui/Menu";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import { ApiError } from "@/lib/errors";
import { USER_ROLE_LABELS } from "@/lib/format";
import { CHANGE_PASSWORD_PATH, changePasswordPath } from "@/lib/routes";
import { logout } from "@/lib/session";
import type { Me } from "@/lib/types";

/**
 * Account menu: who is signed in (username + role), Change password, Sign out. A failed sign-out
 * (API unreachable: the server session would still be valid) is reported under the header with a
 * retry instead of pretending it worked.
 */
export function UserMenu({ me, forced, pathname }: { me: Me; forced: boolean; pathname: string }) {
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const signOut = async () => {
    setSigningOut(true);
    setError(null);
    try {
      await logout(); // the store goes anonymous; AuthGate sends the page to /login
    } catch (err) {
      setError(ApiError.from(err));
    } finally {
      setSigningOut(false);
    }
  };

  const items: MenuEntry[] = [];
  if (!forced && pathname !== CHANGE_PASSWORD_PATH) {
    items.push({
      key: "password",
      label: "Change password",
      icon: <IconLock />,
      onSelect: () => router.push(changePasswordPath(pathname)),
    });
    items.push({ key: "sep", separator: true });
  }
  items.push({ key: "signout", label: signingOut ? "Signing out…" : "Sign out", icon: <IconSignOut />, onSelect: () => void signOut() });

  const isAdmin = me.role === "admin";

  return (
    <>
      <Menu
        label={`Account: ${me.username}${isAdmin ? ", admin" : ""}`}
        triggerClassName="focus-ring flex h-8 max-w-[11rem] items-center gap-1.5 rounded-md border border-line px-2 text-sm text-ink transition-colors duration-150 hover:border-line-strong hover:bg-surface-2 aria-expanded:bg-surface-2"
        trigger={
          <>
            <span className="truncate font-mono text-xs">{me.username}</span>
            {isAdmin ? <span className="rounded-sm bg-ink px-1 font-mono text-2xs text-on-ink">admin</span> : null}
            <IconChevronDown size={14} className="shrink-0 text-ink-3" />
          </>
        }
        header={
          <div className="space-y-0.5">
            <p className="caption">Signed in as</p>
            <p className="font-mono text-sm break-all text-ink">{me.username}</p>
            <p className="text-xs text-ink-3">
              {USER_ROLE_LABELS[me.role]}
              {isAdmin ? " · can manage users and see every job" : " · sees own jobs only"}
            </p>
          </div>
        }
        items={items}
      />
      {error ? (
        <div className="fixed inset-x-0 top-14 z-30 px-4">
          <Notice
            tone="danger"
            live="assertive"
            title="Couldn’t sign out"
            className="mx-auto max-w-md bg-danger-soft"
            onDismiss={() => setError(null)}
            actions={
              <Button size="sm" icon={<IconRefresh />} loading={signingOut} onClick={() => void signOut()}>
                Try again
              </Button>
            }
          >
            {errorCopy(error).message} You are still signed in on the server.
          </Notice>
        </div>
      ) : null}
    </>
  );
}
