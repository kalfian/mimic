"use client";

import { useState, type FormEvent, type RefObject } from "react";

import { IconPlus } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { TextField } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import { normalizeUsername, USERNAME_HINT, usernameProblem } from "@/lib/authPolicy";
import type { ApiError } from "@/lib/errors";
import { USER_ROLE_LABELS } from "@/lib/format";
import type { UserRole } from "@/lib/types";

const ROLE_HELP: Record<UserRole, string> = {
  user: "Analyzes recordings and sees only their own jobs.",
  admin: "Also sees every job and manages accounts.",
};

/**
 * Username + role. Submitting resolves to the temporary-password reveal in the parent. Errors:
 * client username check (same rule as the server), `username_taken` / `invalid_request` on the
 * field, anything else inline.
 */
export function CreateUserDialog({
  open,
  onCancel,
  onCreate,
  returnFocusRef,
}: {
  open: boolean;
  onCancel: () => void;
  /** Resolves null on success (the parent switches to the reveal) or the error to show. */
  onCreate: (input: { username: string; role: UserRole }) => Promise<ApiError | null>;
  returnFocusRef?: RefObject<HTMLElement | null>;
}) {
  if (!open) return null;
  return <CreateUserForm onCancel={onCancel} onCreate={onCreate} returnFocusRef={returnFocusRef} />;
}

function CreateUserForm({
  onCancel,
  onCreate,
  returnFocusRef,
}: {
  onCancel: () => void;
  onCreate: (input: { username: string; role: UserRole }) => Promise<ApiError | null>;
  returnFocusRef?: RefObject<HTMLElement | null>;
}) {
  const [username, setUsername] = useState("");
  const [role, setRole] = useState<UserRole>("user");
  const [busy, setBusy] = useState(false);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const normalized = normalizeUsername(username);
  const showNormalized = username !== "" && normalized !== username && normalized !== "";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    const problem = usernameProblem(username);
    setFieldError(problem);
    setError(null);
    if (problem) {
      document.getElementById("new-username")?.focus();
      return;
    }
    setBusy(true);
    const failure = await onCreate({ username: normalized, role });
    setBusy(false);
    if (!failure) return;
    if (failure.code === "username_taken" || failure.code === "invalid_request") {
      setFieldError(errorCopy(failure).message);
      document.getElementById("new-username")?.focus();
    } else {
      setError(failure);
    }
  };

  return (
    <Dialog
      open
      onClose={onCancel}
      caption="New account"
      title="Create user"
      description="Mimic generates a temporary password for the account. The person signs in with it once and then chooses their own."
      busy={busy}
      returnFocusRef={returnFocusRef}
      footer={
        <>
          <Button onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button type="submit" form="create-user-form" variant="primary" icon={<IconPlus />} loading={busy}>
            {busy ? "Creating…" : "Create user"}
          </Button>
        </>
      }
    >
      <form id="create-user-form" onSubmit={submit} noValidate className="space-y-5">
        <TextField
          id="new-username"
          label="Username"
          autoComplete="off"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            if (fieldError) setFieldError(null);
          }}
          readOnly={busy}
          error={fieldError}
          hint={
            showNormalized ? (
              <>
                Saved as <span className="font-mono text-ink-2">{normalized}</span>. {USERNAME_HINT}
              </>
            ) : (
              USERNAME_HINT
            )
          }
          className="font-mono text-sm"
        />

        <fieldset className="space-y-2" disabled={busy}>
          <legend className="mb-1.5 text-sm font-medium text-ink">Role</legend>
          {(["user", "admin"] as const).map((r) => (
            <label
              key={r}
              className="flex cursor-pointer items-start gap-3 rounded-md border border-line px-3 py-2.5 transition-colors duration-150 hover:border-line-strong has-[input:checked]:border-ink has-[input:focus-visible]:outline-2 has-[input:focus-visible]:outline-offset-2 has-[input:focus-visible]:outline-[var(--focus)]"
            >
              <input
                type="radio"
                name="role"
                value={r}
                checked={role === r}
                onChange={() => setRole(r)}
                className="mt-1 size-3.5 shrink-0 accent-[var(--ink)]"
              />
              <span className="min-w-0">
                <span className="block text-sm font-medium text-ink">{USER_ROLE_LABELS[r]}</span>
                <span className="block text-xs text-ink-3">{ROLE_HELP[r]}</span>
              </span>
            </label>
          ))}
        </fieldset>

        {error ? (
          <Notice tone="danger" live="assertive" title={errorCopy(error).title} guidance={errorCopy(error).guidance}>
            {errorCopy(error).message}
          </Notice>
        ) : null}
      </form>
    </Dialog>
  );
}
