"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { IconRefresh } from "@/components/icons";
import { Forbidden } from "@/components/auth/Forbidden";
import { Button } from "@/components/ui/Button";
import { errorCopy } from "@/components/ui/errorCopy";
import { hostnameMismatch } from "@/lib/api";
import type { ApiError } from "@/lib/errors";
import { routeDecision } from "@/lib/routes";
import { refreshSession, useSession } from "@/lib/session";

/**
 * Client route protection (PLAN-auth A9). Until the session is known — and while a redirect is
 * pending — it renders a neutral placeholder, never the page, so protected content can't flash.
 * The placeholder is also what the server prerenders (the session store is `loading` there).
 *
 * The decision itself comes from `routeDecision()` (lib/routes.ts). The query string is read from
 * `window.location` only when redirecting (it only affects the redirect target), so this needs no
 * `useSearchParams` / Suspense boundary.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const session = useSession();
  const decision = routeDecision(pathname, session);
  const redirectTo = decision.kind === "redirect" ? decision.to : null;

  useEffect(() => {
    if (redirectTo === null) return;
    const search = window.location.search;
    const next = new URLSearchParams(search).get("next");
    const d = routeDecision(pathname, session, { search, next });
    if (d.kind === "redirect") router.replace(d.to);
  }, [redirectTo, pathname, session, router]);

  switch (decision.kind) {
    case "render":
      return <>{children}</>;
    case "forbidden":
      return <Forbidden me={session.me} />;
    case "error":
      return <SessionError error={decision.error} />;
    case "redirect":
      return <Placeholder text="Redirecting…" />;
    case "loading":
      return <Placeholder text="Checking your session…" />;
  }
}

function Placeholder({ text }: { text: string }) {
  return (
    <div className="mx-auto max-w-[1240px] px-4 py-10 sm:px-6 lg:px-8" aria-busy="true">
      <p role="status" className="reveal-late caption">
        {text}
      </p>
    </div>
  );
}

function SessionError({ error }: { error: ApiError }) {
  // 127.0.0.1 vs localhost: the API refuses the page's origin, which looks like "unreachable".
  const d = errorCopy(error, { hostnameMismatch: hostnameMismatch() });
  return (
    <div className="mx-auto max-w-[1240px] px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
      <section aria-labelledby="session-error-title" className="max-w-xl space-y-4">
        <p className="caption text-danger">Session unknown</p>
        <h1 id="session-error-title" className="text-xl font-semibold tracking-tight text-ink">
          Can’t check who is signed in
        </h1>
        <p className="text-md text-ink-2">
          {d.title}. {d.message}
        </p>
        {d.guidance.length > 0 ? (
          <ul className="space-y-1 text-sm text-ink-2">
            {d.guidance.map((g) => (
              <li key={g} className="flex gap-2">
                <span aria-hidden="true" className="text-ink-3">
                  –
                </span>
                {g}
              </li>
            ))}
          </ul>
        ) : null}
        <p className="font-mono text-2xs text-ink-3">
          {error.code}
          {error.status ? ` · HTTP ${error.status}` : ""}
        </p>
        <Button variant="primary" icon={<IconRefresh />} onClick={() => void refreshSession()}>
          Try again
        </Button>
      </section>
    </div>
  );
}
