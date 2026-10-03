"use client";

import { useRef, type RefObject } from "react";

import { IconAlert } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { CopyButton } from "@/components/ui/CopyButton";
import { Dialog } from "@/components/ui/Dialog";
import { USER_ROLE_LABELS } from "@/lib/format";
import type { AdminUser } from "@/lib/types";

/**
 * Shows a server-generated temporary password ONCE (after create or reset). The password lives
 * only in the parent's state while this dialog is open; closing it drops it for good.
 */
export function TemporaryPasswordReveal({
  open,
  user,
  password,
  reason,
  onDone,
  returnFocusRef,
}: {
  open: boolean;
  user: AdminUser | null;
  password: string;
  reason: "created" | "reset";
  onDone: () => void;
  returnFocusRef?: RefObject<HTMLElement | null>;
}) {
  const fieldRef = useRef<HTMLInputElement>(null);
  if (!user) return null;
  return (
    <Dialog
      open={open}
      onClose={onDone}
      caption={reason === "created" ? "Account created" : "Password reset"}
      title={
        <>
          Temporary password for <span className="font-mono">{user.username}</span>
        </>
      }
      description={
        <>
          Give it to <span className="font-mono text-ink">{user.username}</span> over a private channel. They sign in with it once and must choose
          their own password right away{reason === "reset" ? "; their other sessions were signed out" : ""}.
        </>
      }
      initialFocusRef={fieldRef}
      returnFocusRef={returnFocusRef}
      width="md"
      footer={
        <Button variant="primary" onClick={onDone}>
          Done
        </Button>
      }
    >
      <div className="space-y-2">
        <label htmlFor="temp-password" className="block text-sm font-medium text-ink">
          Temporary password
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={fieldRef}
            id="temp-password"
            readOnly
            value={password}
            onFocus={(e) => e.currentTarget.select()}
            spellCheck={false}
            autoComplete="off"
            aria-describedby="temp-password-once"
            className="focus-ring h-10 min-w-0 flex-1 basis-full rounded-sm sm:basis-0 border border-line-strong bg-surface-2 px-3 font-mono text-base tracking-wide text-ink"
          />
          <CopyButton text={password} label="Copy" size="md" />
        </div>
      </div>

      <p id="temp-password-once" className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn-soft px-3 py-2.5 text-sm text-ink">
        <IconAlert className="mt-0.5 shrink-0 text-warn" />
        <span>
          <span className="font-medium">This password won’t be shown again.</span> Mimic doesn’t store it. If it gets lost, reset the password again.
        </span>
      </p>

      <dl className="grid grid-cols-[6rem_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
        <dt className="text-ink-3">Username</dt>
        <dd className="font-mono text-ink">{user.username}</dd>
        <dt className="text-ink-3">Role</dt>
        <dd className="text-ink">{USER_ROLE_LABELS[user.role]}</dd>
        <dt className="text-ink-3">Sign-in</dt>
        <dd className="text-ink-2">Must change password</dd>
      </dl>
    </Dialog>
  );
}
