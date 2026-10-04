import type { ReactNode } from "react";

import { IconAlert, IconInfo } from "@/components/icons";
import { Panel } from "@/components/ui/Panel";
import { INTERPRETATION_STATUS_LABELS, WARNING_LABELS } from "@/lib/format";
import { continuousOf } from "@/lib/spec";
import type { MotionSpec, WarningSeverity } from "@/lib/types";

interface Item {
  key: string;
  severity: WarningSeverity;
  title: string;
  message: string;
  /** Items about element labels can host the "Label with AI" action. */
  aboutLabels?: boolean;
}

/** "frames_subsampled" → "Frames subsampled". Used for codes this build has no copy for yet. */
function humanizeCode(code: string): string {
  const words = code.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "Note";
}

/** Anything other than an explicit "warn" is shown as info, so a new severity never breaks the panel. */
function normalizeSeverity(severity: string): WarningSeverity {
  return severity === "warn" ? "warn" : "info";
}

/** spec.warnings + derived notes (heuristic labels, mostly-low confidence). Renders nothing when clean. */
export function collectWarnings(spec: MotionSpec): Item[] {
  const labels = WARNING_LABELS as Partial<Record<string, string>>;
  const items: Item[] = spec.warnings.map((w, i) => ({
    key: `${w.code}-${i}`,
    severity: normalizeSeverity(w.severity),
    title: labels[w.code] ?? humanizeCode(String(w.code)),
    message: w.message?.trim() || "No further detail from the server.",
    aboutLabels: w.code === "interpretation_fallback",
  }));

  const interp = spec.interpretation;
  if (interp.status !== "ok" && !spec.warnings.some((w) => w.code === "interpretation_fallback")) {
    items.push({
      key: "labels_heuristic",
      severity: "info",
      title: "Labels are heuristic",
      message:
        interp.status === "disabled"
          ? "AI labeling was off, so elements are named by role (e.g. “Image (e2)”). Measurements are unaffected."
          : `${INTERPRETATION_STATUS_LABELS[interp.status] ?? "Heuristic labels"}. Measurements are unaffected.`,
      aboutLabels: true,
    });
  }

  // Transition results rate transitions; continuous results rate phases (empty `transitions`).
  const phases = continuousOf(spec)?.phases ?? [];
  const rated = phases.length > 0 ? phases.map((p) => p.confidence.band) : spec.transitions.map((t) => t.confidence.band);
  const total = rated.length;
  const low = rated.filter((b) => b === "low").length;
  if (total > 0 && low / total >= 0.5) {
    items.unshift({
      key: "mostly_low",
      severity: "warn",
      title: "Mostly low confidence",
      message:
        phases.length > 0
          ? `${low} of ${total} phases are low confidence. Treat the numbers as rough; re-recording at 60 fps with the cursor visible and slower drags usually helps.`
          : `${low} of ${total} measurements are low confidence. Treat the numbers as rough; re-recording at 60 fps with the cursor visible usually helps.`,
    });
  }
  return items;
}

/** `labelAction` renders under the first item about element labels (the relabel flow). */
export function WarningsPanel({ spec, labelAction }: { spec: MotionSpec; labelAction?: ReactNode }) {
  const items = collectWarnings(spec);
  if (items.length === 0) return null;
  const warnCount = items.filter((i) => i.severity === "warn").length;
  const actionKey = labelAction ? items.find((i) => i.aboutLabels)?.key : undefined;

  return (
    <Panel
      id="warnings"
      title="Notes on this analysis"
      meta={<span className="nums text-2xs text-ink-3">{warnCount > 0 ? `${warnCount} warn · ${items.length - warnCount} info` : `${items.length} info`}</span>}
      bodyClassName="divide-y divide-line"
    >
      <ul className="divide-y divide-line">
        {items.map((i) => (
          <li key={i.key} className="flex gap-3 px-4 py-2.5">
            {i.severity === "warn" ? (
              <IconAlert className="mt-0.5 shrink-0 text-warn" />
            ) : (
              <IconInfo className="mt-0.5 shrink-0 text-ink-3" />
            )}
            <div className="min-w-0 flex-1 text-sm">
              <p className="font-medium text-ink">
                <span className="sr-only">{i.severity === "warn" ? "Warning: " : "Note: "}</span>
                {i.title}
              </p>
              <p className="break-words text-ink-2">{i.message}</p>
              {i.key === actionKey ? <div className="mt-2.5">{labelAction}</div> : null}
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
