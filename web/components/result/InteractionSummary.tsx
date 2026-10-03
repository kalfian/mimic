import type { ReactNode } from "react";

import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { LabelSourceTag } from "@/components/ui/LabelSourceTag";
import { Panel } from "@/components/ui/Panel";
import {
  DIRECTION_LABELS,
  formatMs,
  formatPixelRatio,
  INTERACTION_TYPE_LABELS,
  PATTERN_LABELS,
  TRIGGER_LABELS,
} from "@/lib/format";
import type { MotionSpec, ReverseTriggerKind } from "@/lib/types";

const REVERSE_PHRASE: Record<ReverseTriggerKind, string | null> = {
  pointer_leave: "reverses when the pointer leaves",
  click: "reverses on the next click",
  none: null,
  unknown: "reverse trigger unknown",
};

/** Label | value (+ optional right-aligned badge that never wraps under the value). */
function Row({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 border-b border-line py-2.5 last:border-b-0 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-x-4">
      <dt className="text-sm text-ink-3">{label}</dt>
      <dd className={`min-w-0 text-sm text-ink ${aside ? "grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3" : ""}`}>
        {aside ? (
          <>
            <div className="min-w-0">{children}</div>
            <div className="pt-0.5">{aside}</div>
          </>
        ) : (
          children
        )}
      </dd>
    </div>
  );
}

/** "Detected Interaction" (PRD §17): what was triggered, on what, for how long. */
export function InteractionSummary({ spec }: { spec: MotionSpec }) {
  const { interaction: ix, source, cursor } = spec;
  const target = spec.elements.find((e) => e.id === ix.target_element_id);
  const typeUncertain = ix.type === "unknown" || ix.type_confidence.band === "low";
  const triggerUnknown = ix.trigger.kind === "unknown";
  const { forward, reverse } = ix.total_duration_ms;

  return (
    <Panel id="interaction" title="Detected interaction" bodyClassName="px-4 py-1">
      <dl>
        <Row label="Type" aside={<ConfidenceBadge confidence={ix.type_confidence} />}>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium">{INTERACTION_TYPE_LABELS[ix.type]}</span>
            <span className="text-ink-3">· {PATTERN_LABELS[ix.pattern].toLowerCase()} pattern</span>
            <LabelSourceTag source={ix.type_source} />
          </span>
          {typeUncertain ? <span className="mt-1 block text-xs text-warn">Low confidence: the interaction type is a best guess.</span> : null}
        </Row>

        <Row label="Trigger" aside={<ConfidenceBadge confidence={ix.trigger.confidence} />}>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className={`font-medium ${triggerUnknown ? "text-ink-2" : ""}`}>{TRIGGER_LABELS[ix.trigger.kind]}</span>
            {REVERSE_PHRASE[ix.trigger.reverse_kind] ? <span className="text-ink-3">· {REVERSE_PHRASE[ix.trigger.reverse_kind]}</span> : null}
          </span>
          <span className="mt-1 block text-xs text-ink-2">{ix.trigger.description}</span>
          {!cursor.visible ? <span className="mt-1 block text-xs text-warn">Cursor not visible in the recording.</span> : null}
        </Row>

        <Row label="Target">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium">{ix.target_label}</span>
            {target ? <LabelSourceTag source={target.label_source} /> : null}
          </span>
          {ix.target_description ? <span className="mt-1 block text-xs text-ink-2">{ix.target_description}</span> : null}
          {spec.structure.length > 0 ? (
            <span className="mt-1 block text-xs text-ink-3">Contains: {spec.structure.join(", ")}</span>
          ) : null}
        </Row>

        <Row label="Duration">
          <span className="nums flex flex-wrap gap-x-4 text-sm">
            <span>
              <span className="text-ink-3">fwd </span>
              {formatMs(forward, { approx: true })}
            </span>
            <span>
              <span className="text-ink-3">rev </span>
              {reverse != null ? formatMs(reverse, { approx: true }) : <span className="font-sans text-ink-3">not recorded</span>}
            </span>
          </span>
        </Row>

        <Row label="Direction">{DIRECTION_LABELS[ix.direction]}</Row>

        <Row label="Display scale">
          <span className="nums">{formatPixelRatio(source.pixel_ratio)}</span>{" "}
          <span className="text-ink-3">{source.pixel_ratio_source === "auto" ? "· auto-detected" : "· set on upload"}</span>
        </Row>
      </dl>
    </Panel>
  );
}
