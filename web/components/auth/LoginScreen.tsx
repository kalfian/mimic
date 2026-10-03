"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { SetupRequired } from "@/components/auth/SetupRequired";
import { useCountdown } from "@/components/auth/useCountdown";
import { IconRefresh } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { PasswordField, TextField } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import { getAuthStatus, hostnameMismatch, IS_MOCK_API } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import { describeError, formatElapsed, formatRetryAfter } from "@/lib/format";
import { login, useSession } from "@/lib/session";

// Dev helper, loaded only in mock mode (keeps the mock out of a real build's login bundle).
const MockLoginNote = dynamic(() => import("@/components/auth/MockLoginNote"), { ssr: false });

type Status = { kind: "checking" } | { kind: "ready" } | { kind: "setup" } | { kind: "error"; error: ApiError };

/**
 * `/login`. Resolves `GET /api/auth/status` first: setup required → instructions instead of a
 * form. After a successful sign-in the session store becomes authenticated and `AuthGate`
 * redirects (to `next`, or to the forced password change), so this screen never navigates itself.
 */
export function LoginScreen({ next }: { next: string }) {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  const [attempt, setAttempt] = useState(0);
  const [checkingAgain, setCheckingAgain] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getAuthStatus(controller.signal).then(
      (s) => {
        if (controller.signal.aborted) return;
        setStatus(s.setup_required ? { kind: "setup" } : { kind: "ready" });
        setCheckingAgain(false);
      },
      (err: unknown) => {
        if (controller.signal.aborted) return;
        const e = ApiError.from(err);
        if (e.code === "aborted") return;
        setStatus({ kind: "error", error: e });
        setCheckingAgain(false);
      },
    );
    return () => controller.abort();
  }, [attempt]);

  const recheck = () => {
    setCheckingAgain(true);
    setAttempt((n) => n + 1);
  };

  return (
    <div className="mx-auto grid max-w-[1240px] gap-10 px-4 py-10 sm:px-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:gap-20 lg:px-8 lg:py-16">
      <div className="min-w-0">
        {status.kind === "checking" ? (
          <CheckingServer />
        ) : status.kind === "setup" ? (
          <SetupRequired onCheckAgain={recheck} checking={checkingAgain} />
        ) : status.kind === "error" ? (
          <StatusError error={status.error} onRetry={recheck} retrying={checkingAgain} />
        ) : (
          <LoginForm next={next} onSetupRequired={() => setStatus({ kind: "setup" })} />
        )}
      </div>
      <AccountsAside />
    </div>
  );
}

function Heading({ caption, title, children }: { caption: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 space-y-2">
      <p className="caption">{caption}</p>
      <h1 id="auth-title" className="text-xl font-semibold tracking-tight text-ink">
        {title}
      </h1>
      {children ? <p className="text-md text-ink-2">{children}</p> : null}
    </header>
  );
}

function CheckingServer() {
  return (
    <div aria-busy="true">
      <Heading caption="Sign in" title="Sign in to Mimic" />
      <p role="status" className="reveal-late text-sm text-ink-3">
        Contacting the API server…
      </p>
    </div>
  );
}

function StatusError({ error, onRetry, retrying }: { error: ApiError; onRetry: () => void; retrying: boolean }) {
  // On 127.0.0.1 vs localhost the API refuses the page's origin (CORS), which looks like
  // "server unreachable"; name the real cause.
  const d = errorCopy(error, { hostnameMismatch: hostnameMismatch() });
  return (
    <section aria-labelledby="auth-title">
      <Heading caption="Sign in" title="Sign in to Mimic" />
      <Notice
        tone="danger"
        live="assertive"
        title={d.title}
        guidance={d.guidance}
        actions={
          <Button size="sm" icon={<IconRefresh />} loading={retrying} onClick={onRetry}>
            Try again
          </Button>
        }
      >
        {d.message}
      </Notice>
    </section>
  );
}

type FormError = { error: ApiError; until: number | null } | null;

function LoginForm({ next, onSetupRequired }: { next: string; onSetupRequired: () => void }) {
  const session = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [failure, setFailure] = useState<FormError>(null);
  const [fieldErrors, setFieldErrors] = useState<{ username?: string; password?: string }>({});
  const passwordRef = useRef<HTMLInputElement>(null);
  const usernameRef = useRef<HTMLInputElement>(null);
  const [mismatch] = useState(() => hostnameMismatch());

  const countdown = useCountdown(failure?.until ?? null);
  const throttled = failure?.error.code === "too_many_attempts";
  const remaining = throttled && failure?.until != null ? (countdown ?? failure.error.retryAfterS ?? 0) : null;
  const locked = remaining !== null && remaining > 0;

  // "Your session ended" only when a signed-in session was cut off (not after a normal sign-out).
  const sessionEnded = session.status === "anonymous" && session.error?.code === "unauthenticated" && failure === null;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting || locked) return;
    const errs: { username?: string; password?: string } = {};
    if (!username.trim()) errs.username = "Enter your username.";
    if (!password) errs.password = "Enter your password.";
    setFieldErrors(errs);
    if (errs.username || errs.password) {
      setFailure(null);
      (errs.username ? usernameRef : passwordRef).current?.focus();
      return;
    }

    setSubmitting(true);
    setFailure(null);
    try {
      await login(username, password);
      // AuthGate takes over: redirect to `next` or to the forced password change.
    } catch (err) {
      const error = ApiError.from(err);
      if (error.code === "setup_required") {
        onSetupRequired();
        return;
      }
      setFailure({ error, until: error.code === "too_many_attempts" && error.retryAfterS != null ? Date.now() + error.retryAfterS * 1000 : null });
      if (error.code === "invalid_credentials") {
        setPassword("");
        passwordRef.current?.focus();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const fill = (name: string) => {
    setUsername(name);
    setPassword("mock-password");
    setFieldErrors({});
    setFailure(null);
  };

  return (
    <section aria-labelledby="auth-title">
      <Heading caption="Sign in" title="Sign in to Mimic">
        Use the account an admin created for you.
      </Heading>

      <form onSubmit={onSubmit} noValidate className="space-y-5" aria-describedby={next !== "/" ? "login-next" : undefined}>
        {sessionEnded ? (
          <Notice tone="info" title="Your session ended">
            You were signed out (the session expired, or an admin changed your account). Sign in again to continue.
          </Notice>
        ) : null}
        {mismatch ? <HostnameWarning pageHost={mismatch.pageHost} apiHost={mismatch.apiHost} /> : null}

        <TextField
          ref={usernameRef}
          id="login-username"
          label="Username"
          name="username"
          autoComplete="username"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          error={fieldErrors.username}
          readOnly={submitting}
          className="font-mono text-sm"
        />
        <PasswordField
          ref={passwordRef}
          id="login-password"
          label="Password"
          name="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={fieldErrors.password}
          readOnly={submitting}
        />

        {failure ? <LoginFailure error={failure.error} remaining={remaining} /> : null}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-1">
          <Button type="submit" variant="primary" loading={submitting} disabled={locked} className="min-w-[7.5rem]">
            {submitting ? "Signing in…" : "Sign in"}
          </Button>
          {next !== "/" ? (
            <p id="login-next" className="min-w-0 truncate text-xs text-ink-3">
              Then back to <span className="font-mono text-ink-2">{next}</span>
            </p>
          ) : null}
        </div>
      </form>

      {IS_MOCK_API ? (
        <div className="mt-10">
          <MockLoginNote onPick={fill} />
        </div>
      ) : null}
    </section>
  );
}

function LoginFailure({ error, remaining }: { error: ApiError; remaining: number | null }) {
  const d = errorCopy(error);
  if (error.code === "too_many_attempts") {
    const waiting = remaining !== null && remaining > 0;
    return (
      <Notice tone="warn" live="assertive" title={waiting ? "Too many failed attempts" : "You can try again"}>
        {remaining === null ? (
          "Sign-in is paused for this username. Wait a few minutes, then try again."
        ) : waiting ? (
          <>
            Sign-in is paused for this username. {/* The ticking digits are hidden from screen readers; the wait is announced once. */}
            <span aria-hidden="true">
              Try again in <span className="nums text-ink">{formatElapsed(remaining * 1000)}</span>.
            </span>
            <span className="sr-only">{formatRetryAfter(error.retryAfterS ?? remaining)}.</span>
          </>
        ) : (
          "The pause is over. Check the password before you sign in again."
        )}
      </Notice>
    );
  }
  if (error.code === "cookie_rejected") {
    // The generic copy + guidance, and the two hostnames (mono) when they are known to differ.
    const base = describeError("cookie_rejected");
    const hosts = hostnameMismatch();
    return (
      <Notice tone="danger" live="assertive" title={base.title} guidance={base.guidance}>
        {base.message}
        {hosts ? (
          <>
            {" "}
            This page is on <span className="font-mono text-ink">{hosts.pageHost}</span>, the API on{" "}
            <span className="font-mono text-ink">{hosts.apiHost}</span>.
          </>
        ) : null}
      </Notice>
    );
  }
  if (error.code === "invalid_credentials") {
    return (
      <Notice tone="danger" live="assertive" title={d.message}>
        Check both and try again. After 5 failed attempts, sign-in pauses for a few minutes.
      </Notice>
    );
  }
  return (
    <Notice tone="danger" live="assertive" title={d.title} guidance={d.guidance}>
      {d.message}
    </Notice>
  );
}

function HostnameWarning({ pageHost, apiHost }: { pageHost: string; apiHost: string }) {
  return (
    <Notice tone="warn" title="Hostnames don’t match">
      This page is on <span className="font-mono text-ink">{pageHost}</span> but the API is on <span className="font-mono text-ink">{apiHost}</span>.
      The browser won’t send the session cookie across them, so sign-in won’t stick. Open the app on{" "}
      <span className="font-mono text-ink">{apiHost}</span>, or point NEXT_PUBLIC_API_BASE_URL at{" "}
      <span className="font-mono text-ink">{pageHost}</span>.
    </Notice>
  );
}

const ACCOUNT_FACTS: [string, ReactNode][] = [
  ["No sign-up", "An admin creates every account and hands you a temporary password. You choose your own at first sign-in."],
  ["Forgot it?", "Ask an admin to reset your password. You’ll get a new temporary one."],
  ["Your jobs", "You see only the recordings you uploaded. Admins can see every job."],
  [
    "Sessions",
    <>
      You stay signed in for up to <span className="nums">7</span> days, or until <span className="nums">12</span> h without activity.
    </>,
  ],
];

function AccountsAside() {
  return (
    <aside aria-labelledby="accounts-title" className="min-w-0 lg:border-l lg:border-line lg:pl-10">
      <h2 id="accounts-title" className="caption mb-3">
        About accounts
      </h2>
      <dl className="grid max-w-xl gap-x-6 gap-y-3 text-sm sm:grid-cols-[8rem_minmax(0,1fr)]">
        {ACCOUNT_FACTS.map(([term, text]) => (
          <div key={term} className="space-y-0.5 sm:contents">
            <dt className="text-ink-3">{term}</dt>
            <dd className="text-ink-2">{text}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}
