"use client";

import { IconRefresh, IconTerminal } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { CopyButton } from "@/components/ui/CopyButton";
import { SETUP_COMMAND } from "@/lib/format";

const STEPS = [
  "On the machine that runs the API, open a terminal in the Mimic repository.",
  "Run the command below and choose the admin’s username and password when it asks.",
  "Come back here, check again, and sign in with that account.",
];

/** No admin exists yet (409 `setup_required` / `GET /api/auth/status`): instructions, no form. */
export function SetupRequired({ onCheckAgain, checking }: { onCheckAgain: () => void; checking: boolean }) {
  return (
    <section aria-labelledby="auth-title">
      <header className="mb-8 space-y-2">
        <p className="caption text-warn">Setup required</p>
        <h1 id="auth-title" className="text-xl font-semibold tracking-tight text-ink">
          Create the first admin
        </h1>
        <p className="text-md text-ink-2">
          No admin account exists yet, so nobody can sign in. There are no default credentials: the first admin is created on the server.
        </p>
      </header>

      <ol className="space-y-3">
        {STEPS.map((step, i) => (
          <li key={step} className="flex gap-3 text-sm text-ink-2">
            <span className="nums w-5 shrink-0 pt-px text-2xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
            <span className="min-w-0">{step}</span>
          </li>
        ))}
      </ol>

      <div className="mt-5 flex flex-wrap items-center gap-3 rounded-md border border-line bg-surface py-2 pr-2 pl-3">
        <IconTerminal className="shrink-0 text-ink-3" />
        <code className="min-w-0 flex-1 font-mono text-sm break-all text-ink">
          <span className="text-ink-3 select-none" aria-hidden="true">
            ${" "}
          </span>
          {SETUP_COMMAND}
        </code>
        <CopyButton text={SETUP_COMMAND} label="Copy command" />
      </div>
      <p className="mt-2 text-xs text-ink-3">Recordings analyzed before accounts existed are assigned to this first admin.</p>

      <div className="mt-8 flex flex-wrap items-center gap-3 border-t border-line pt-6">
        <Button variant="primary" icon={<IconRefresh />} loading={checking} onClick={onCheckAgain}>
          {checking ? "Checking…" : "Check again"}
        </Button>
        <p role="status" className="text-sm text-ink-3">
          {checking ? "Asking the API whether an admin exists…" : ""}
        </p>
      </div>
    </section>
  );
}
