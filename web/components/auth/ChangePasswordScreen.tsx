"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState, type FormEvent } from "react";

import { useCountdown } from "@/components/auth/useCountdown";
import { IconArrowLeft, IconCheck } from "@/components/icons";
import { Button, buttonClass } from "@/components/ui/Button";
import { PasswordField } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import { newPasswordProblems, passwordLength, PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH } from "@/lib/authPolicy";
import { ApiError } from "@/lib/errors";
import { formatElapsed, formatRetryAfter } from "@/lib/format";
import { afterAuthPath } from "@/lib/routes";
import { changePassword, logout, useSession } from "@/lib/session";
import type { Me } from "@/lib/types";

type Problems = { current?: string; newPassword?: string; confirmPassword?: string };

/**
 * `/account/password`. Forced variant (temporary password after create/reset: explains why, the
 * only way out is sign out) and voluntary variant (from the account menu). Client checks mirror the
 * server policy for instant feedback; the server's answer is still shown when it disagrees.
 */
export function ChangePasswordScreen({ next }: { next: string }) {
  const session = useSession();
  const me = session.me;
  // Keep the variant the page opened with; `me` flips to must_change_password:false on success.
  const [forced] = useState(() => me?.must_change_password ?? false);
  if (!me) return null; // AuthGate never renders this route without a session.
  return <ChangePasswordForm me={me} forced={forced} next={next} />;
}

function ChangePasswordForm({ me, forced, next }: { me: Me; forced: boolean; next: string }) {
  const router = useRouter();
  const [current, setCurrent] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [problems, setProblems] = useState<Problems>({});
  const [confirmTouched, setConfirmTouched] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [failure, setFailure] = useState<{ error: ApiError; until: number | null } | null>(null);
  const [done, setDone] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const currentRef = useRef<HTMLInputElement>(null);
  const newRef = useRef<HTMLInputElement>(null);
  const confirmRef = useRef<HTMLInputElement>(null);

  const countdown = useCountdown(failure?.until ?? null);
  const throttled = failure?.error.code === "too_many_attempts";
  const remaining = throttled && failure?.until != null ? (countdown ?? failure.error.retryAfterS ?? 0) : null;
  const locked = remaining !== null && remaining > 0;

  const clearProblem = (key: keyof Problems) => {
    if (problems[key]) setProblems((p) => ({ ...p, [key]: undefined }));
  };

  const length = passwordLength(newPassword);
  const rules = [
    {
      key: "length",
      label: `${PASSWORD_MIN_LENGTH}–${PASSWORD_MAX_LENGTH} characters`,
      met: length >= PASSWORD_MIN_LENGTH && length <= PASSWORD_MAX_LENGTH,
    },
    {
      key: "username",
      label: "Not your username",
      met: newPassword !== "" && newPassword.normalize("NFKC").toLowerCase() !== me.username.toLowerCase(),
    },
    {
      key: "current",
      label: forced ? "Not the temporary password" : "Not your current password",
      met: newPassword !== "" && newPassword.normalize("NFKC") !== current.normalize("NFKC"),
    },
  ];
  const confirmMismatch = confirmTouched && confirm !== "" && confirm !== newPassword ? "The passwords don't match." : null;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting || locked) return;
    const found: Problems = { ...newPasswordProblems({ newPassword, confirmPassword: confirm, username: me.username, currentPassword: current }) };
    if (!current) found.current = forced ? "Enter the temporary password." : "Enter your current password.";
    if (!newPassword) found.newPassword = "Choose a new password.";
    setProblems(found);
    setConfirmTouched(true);
    if (found.current || found.newPassword || found.confirmPassword) {
      (found.current ? currentRef : found.newPassword ? newRef : confirmRef).current?.focus();
      return;
    }
    setSubmitting(true);
    setFailure(null);
    try {
      const updated = await changePassword(current, newPassword);
      if (forced) {
        router.replace(afterAuthPath(updated, next));
        return;
      }
      setDone(true);
      setCurrent("");
      setNewPassword("");
      setConfirm("");
    } catch (err) {
      const error = ApiError.from(err);
      if (error.code === "current_password_incorrect") {
        setProblems({ current: errorCopy(error).message });
        currentRef.current?.focus();
      } else if (error.code === "weak_password") {
        setProblems({ newPassword: errorCopy(error).message });
        newRef.current?.focus();
      } else if (error.code !== "unauthenticated") {
        // unauthenticated: the session store goes anonymous and AuthGate redirects to /login.
        setFailure({ error, until: error.code === "too_many_attempts" && error.retryAfterS != null ? Date.now() + error.retryAfterS * 1000 : null });
      }
    } finally {
      setSubmitting(false);
    }
  };

  const signOut = async () => {
    setSigningOut(true);
    try {
      await logout();
    } catch (err) {
      setFailure({ error: ApiError.from(err), until: null });
    } finally {
      setSigningOut(false);
    }
  };

  return (
    <div className="mx-auto grid max-w-[1240px] gap-10 px-4 py-10 sm:px-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:gap-20 lg:px-8 lg:py-16">
      <section aria-labelledby="password-title" className="min-w-0">
        {!forced ? (
          <Link href={next} className="focus-ring -ml-1 mb-4 inline-flex items-center gap-1.5 rounded-sm px-1 text-sm text-ink-3 hover:text-ink">
            <IconArrowLeft />
            Back
          </Link>
        ) : null}
        <header className="mb-8 space-y-2">
          <p className={`caption ${forced ? "text-warn" : ""}`}>{forced ? "Action required" : "Account"}</p>
          <h1 id="password-title" className="text-xl font-semibold tracking-tight text-ink">
            {forced ? "Choose your own password" : "Change password"}
          </h1>
          <p className="text-md text-ink-2">
            {forced ? (
              <>
                You signed in as <span className="font-mono text-ink">{me.username}</span> with a temporary password from an admin. Set your own to
                continue.
              </>
            ) : (
              <>
                For <span className="font-mono text-ink">{me.username}</span>. Other browsers and devices are signed out afterwards.
              </>
            )}
          </p>
        </header>

        {done ? (
          <div className="space-y-5">
            <Notice tone="ok" live="polite" title="Password changed">
              Other browsers and devices were signed out. This one stays signed in.
            </Notice>
            <Link href={afterAuthPath(me, next)} className={buttonClass("primary")}>
              Continue
            </Link>
          </div>
        ) : (
          <form onSubmit={onSubmit} noValidate className="space-y-5">
            <PasswordField
              ref={currentRef}
              id="pw-current"
              label={forced ? "Temporary password" : "Current password"}
              name="current-password"
              autoComplete="current-password"
              value={current}
              onChange={(e) => {
                setCurrent(e.target.value);
                clearProblem("current");
              }}
              error={problems.current}
              hint={forced ? "The one the admin gave you." : undefined}
              readOnly={submitting}
            />
            <PasswordField
              ref={newRef}
              id="pw-new"
              label="New password"
              name="new-password"
              autoComplete="new-password"
              value={newPassword}
              onChange={(e) => {
                setNewPassword(e.target.value);
                clearProblem("newPassword");
              }}
              error={problems.newPassword}
              aside={
                <span className={`nums ${length > PASSWORD_MAX_LENGTH ? "text-danger" : ""}`} aria-hidden="true">
                  {length} / {PASSWORD_MIN_LENGTH}+
                </span>
              }
              hint={<RuleList rules={rules} />}
              readOnly={submitting}
            />
            <PasswordField
              ref={confirmRef}
              id="pw-confirm"
              label="Confirm new password"
              name="confirm-password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => {
                setConfirm(e.target.value);
                clearProblem("confirmPassword");
              }}
              onBlur={() => setConfirmTouched(true)}
              error={problems.confirmPassword ?? confirmMismatch}
              readOnly={submitting}
            />

            {failure ? <SubmitFailure error={failure.error} remaining={remaining} /> : null}

            <div className="flex flex-wrap items-center gap-3 border-t border-line pt-6">
              <Button type="submit" variant="primary" loading={submitting} disabled={locked}>
                {submitting ? "Saving…" : forced ? "Set password and continue" : "Change password"}
              </Button>
              {forced ? (
                <Button variant="ghost" loading={signingOut} onClick={() => void signOut()}>
                  Sign out instead
                </Button>
              ) : (
                <Link href={next} className={buttonClass("ghost")}>
                  Cancel
                </Link>
              )}
            </div>
          </form>
        )}
      </section>

      <aside aria-labelledby="pw-about" className="min-w-0 lg:border-l lg:border-line lg:pl-10">
        <h2 id="pw-about" className="caption mb-3">
          Password rules
        </h2>
        <ul className="max-w-xl space-y-2 text-sm text-ink-2">
          <li>Length matters more than symbols: a few unrelated words make a good password.</li>
          <li>
            <span className="nums">{PASSWORD_MIN_LENGTH}</span> to <span className="nums">{PASSWORD_MAX_LENGTH}</span> characters. No rules about
            digits or symbols.
          </li>
          <li>Changing it signs out every other browser and device using this account.</li>
          {forced ? <li>Until you set it, Mimic only lets you change the password or sign out.</li> : null}
        </ul>
      </aside>
    </div>
  );
}

function RuleList({ rules }: { rules: { key: string; label: string; met: boolean }[] }) {
  return (
    <span className="mt-1 flex flex-col gap-0.5">
      {rules.map((r) => (
        <span key={r.key} className={`inline-flex items-center gap-1 ${r.met ? "text-ok" : "text-ink-3"}`}>
          {r.met ? (
            <IconCheck size={12} />
          ) : (
            <span aria-hidden="true" className="inline-flex size-3 items-center justify-center">
              <span className="size-1 rounded-full bg-current" />
            </span>
          )}
          {r.label}
          <span className="sr-only">{r.met ? " (met)" : " (not met yet)"}</span>
        </span>
      ))}
    </span>
  );
}

function SubmitFailure({ error, remaining }: { error: ApiError; remaining: number | null }) {
  if (error.code === "too_many_attempts") {
    const waiting = remaining !== null && remaining > 0;
    return (
      <Notice tone="warn" live="assertive" title={waiting ? "Too many attempts" : "You can try again"}>
        {remaining === null ? (
          "Wait a few minutes, then try again."
        ) : waiting ? (
          <>
            Password changes are paused for this account.{" "}
            <span aria-hidden="true">
              Try again in <span className="nums text-ink">{formatElapsed(remaining * 1000)}</span>.
            </span>
            <span className="sr-only">{formatRetryAfter(error.retryAfterS ?? remaining)}.</span>
          </>
        ) : (
          "The pause is over."
        )}
      </Notice>
    );
  }
  const d = errorCopy(error);
  return (
    <Notice tone="danger" live="assertive" title={d.title} guidance={d.guidance}>
      {d.message}
    </Notice>
  );
}
