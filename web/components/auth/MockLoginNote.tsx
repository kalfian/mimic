"use client";

import { resetMockData } from "@/lib/mock/mockApi";
import { MOCK_AUTH_FAILURES } from "@/lib/mock/mockAuth";

/**
 * Dev-only helper on /login (mock mode): the seeded accounts as one-click fills, and links that
 * reload the page with a login `?fail=` scenario (lib/README.md "Mock accounts").
 */

const ACCOUNTS: { username: string; note: string }[] = [
  { username: "admin", note: "admin, the only one" },
  { username: "user", note: "owns the sample result" },
  { username: "alice", note: "2 jobs, one failed" },
  { username: "newbie", note: "must change password" },
  { username: "disabled", note: "account disabled" },
];

const chip =
  "focus-ring inline-block rounded-sm border border-line px-1.5 py-0.5 text-ink-2 hover:border-ink-3 hover:text-ink aria-[current=true]:border-ink aria-[current=true]:bg-ink aria-[current=true]:text-on-ink";

export default function MockLoginNote({ onPick }: { onPick: (username: string) => void }) {
  const fail = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("fail");
  const scenarios = [...new Set([...MOCK_AUTH_FAILURES.authStatus, ...MOCK_AUTH_FAILURES.login])];

  return (
    <details className="rounded-md border border-dashed border-line-strong font-mono text-2xs" open>
      <summary className="focus-ring cursor-pointer rounded-md px-3 py-2 text-ink-2 select-none">
        Mock API · not security: any password signs in except <span className="text-ink">wrong</span>
      </summary>
      <div className="space-y-3 border-t border-line px-3 py-3">
        <div>
          <p className="mb-1 text-ink-3">Seeded accounts (fills the form)</p>
          <ul className="grid gap-1">
            {ACCOUNTS.map((a) => (
              <li key={a.username}>
                <button type="button" className={`${chip} w-full text-left`} onClick={() => onPick(a.username)}>
                  <span className="text-ink">{a.username}</span> <span className="text-ink-3">· {a.note}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="mb-1 text-ink-3">Force a state (?fail=, reloads)</p>
          <ul className="flex flex-wrap gap-1">
            <li>
              <a href="/login" aria-current={fail === null ? "true" : undefined} className={chip}>
                none
              </a>
            </li>
            {scenarios.map((s) => (
              <li key={s}>
                <a href={`/login?fail=${s}`} aria-current={fail === s ? "true" : undefined} className={chip}>
                  {s}
                </a>
              </li>
            ))}
          </ul>
          <p className="mt-1.5 text-ink-3">Or: password wrong 5× for one username → real throttle with a countdown.</p>
        </div>
        {/* A plain link: the handler wipes the mock first, then the browser reloads /login. */}
        <a href="/login" className={chip} onClick={() => resetMockData()}>
          Reset mock accounts + jobs
        </a>
      </div>
    </details>
  );
}
