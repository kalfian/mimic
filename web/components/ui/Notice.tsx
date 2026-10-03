import type { ReactNode } from "react";

import { IconAlert, IconCheck, IconCross, IconInfo } from "@/components/icons";

export type NoticeTone = "danger" | "warn" | "info" | "ok";

const TONE: Record<NoticeTone, { box: string; icon: string; title: string }> = {
  danger: { box: "border-danger/40 bg-danger-soft", icon: "text-danger", title: "text-danger" },
  warn: { box: "border-warn/40 bg-warn-soft", icon: "text-warn", title: "text-ink" },
  info: { box: "border-line bg-surface", icon: "text-ink-3", title: "text-ink" },
  ok: { box: "border-ok/40 bg-ok-soft", icon: "text-ok", title: "text-ok" },
};

const ICON: Record<NoticeTone, typeof IconAlert> = { danger: IconCross, warn: IconAlert, info: IconInfo, ok: IconCheck };

/**
 * Inline message box (same frame as the interpreter check results): tone icon + optional title +
 * body + optional numbered guidance + actions. `live` picks the announcement politeness:
 * "assertive" → role=alert (errors raised by the user's own action), "polite" → role=status.
 */
export function Notice({
  tone = "info",
  title,
  children,
  guidance,
  actions,
  live = "off",
  className = "",
  id,
  onDismiss,
}: {
  tone?: NoticeTone;
  title?: ReactNode;
  children?: ReactNode;
  guidance?: readonly string[];
  actions?: ReactNode;
  live?: "assertive" | "polite" | "off";
  className?: string;
  id?: string;
  /** Adds a close button (results the user may want to keep reading, never errors that block). */
  onDismiss?: () => void;
}) {
  const t = TONE[tone];
  const Icon = ICON[tone];
  const role = live === "assertive" ? "alert" : live === "polite" ? "status" : undefined;
  return (
    <div id={id} role={role} className={`flex items-start gap-2.5 rounded-md border px-3 py-2.5 text-sm ${t.box} ${className}`}>
      <Icon className={`mt-0.5 shrink-0 ${t.icon}`} />
      <div className="min-w-0 flex-1 space-y-1">
        {title ? <p className={`font-medium ${t.title}`}>{title}</p> : null}
        {children ? <div className="text-ink-2">{children}</div> : null}
        {guidance && guidance.length > 0 ? (
          <ul className="space-y-1 pt-0.5">
            {guidance.map((g) => (
              <li key={g} className="flex gap-2 text-ink-2">
                <span aria-hidden="true" className="text-ink-3">
                  –
                </span>
                <span className="min-w-0">{g}</span>
              </li>
            ))}
          </ul>
        ) : null}
        {actions ? <div className="flex flex-wrap items-center gap-2 pt-1.5">{actions}</div> : null}
      </div>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="focus-ring -my-1 -mr-1.5 inline-flex size-7 shrink-0 items-center justify-center rounded-sm text-ink-3 transition-colors duration-150 hover:bg-surface-2 hover:text-ink"
        >
          <IconCross size={14} />
        </button>
      ) : null}
    </div>
  );
}
