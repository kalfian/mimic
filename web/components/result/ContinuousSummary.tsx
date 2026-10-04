import type { ReactNode } from "react";

import { AppearanceFactList, ColorSwatch } from "@/components/result/AppearanceFacts";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { LabelSourceTag } from "@/components/ui/LabelSourceTag";
import { Panel } from "@/components/ui/Panel";
import {
  appearanceFacts,
  AXIS_LABELS,
  BAND_LABELS,
  formatColor,
  formatPercent,
  formatPixelRatio,
  formatPx,
  INTERACTION_TYPE_LABELS,
  PATTERN_LABELS,
  PHASE_EVIDENCE_LABELS,
} from "@/lib/format";
import type { ContinuousSpec } from "@/lib/spec";
import type { MeasuredNumber } from "@/lib/types";
import type { BehaviorFlag, BehaviorRow, VelocityModel } from "@/lib/velocity";

/**
 * Label | value (+ optional right-aligned badge that never wraps under the value). Same grid as
 * InteractionSummary. `footer` (jump links, value details) runs under value *and* badge, so it gets
 * the full value-column width instead of the space left beside the badge.
 */
function Row({ label, aside, footer, muted = false, children }: { label: string; aside?: ReactNode; footer?: ReactNode; muted?: boolean; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 border-b border-line py-2.5 last:border-b-0 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-x-4">
      <dt className="text-sm text-ink-3">{label}</dt>
      <dd className={`min-w-0 text-sm ${muted ? "text-ink-3" : "text-ink"} ${aside ? "grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3" : ""}`}>
        {aside ? (
          <>
            <div className="min-w-0">{children}</div>
            <div className="pt-0.5">{aside}</div>
            {footer ? <div className="col-span-2 min-w-0">{footer}</div> : null}
          </>
        ) : (
          <>
            {children}
            {footer}
          </>
        )}
      </dd>
    </div>
  );
}

const FLAG_TEXT: Record<BehaviorFlag, string> = {
  trigger_unknown: "trigger not visible",
  loop_not_observed: "loop length unknown",
  snap_ambiguous: "ambiguous",
  overshoot: "overshoots",
  pointer_unknown: "pointer not visible",
  direction_changed: "direction reversed",
};

const CHIP = "rounded-sm border px-1 font-mono text-2xs whitespace-nowrap";

/** Low-confidence measured numbers read as estimates: "≈216 px?" style hedge, band in the tooltip. */
function Hedged({ n, text }: { n: MeasuredNumber; text: string }) {
  const low = n.confidence.band === "low";
  return (
    <span className={low ? "text-ink-3" : undefined} title={`${BAND_LABELS[n.confidence.band]} confidence, ${formatPercent(n.confidence.value)}`}>
      {text}
      {low ? <span aria-hidden="true">?</span> : null}
      <span className="sr-only"> ({BAND_LABELS[n.confidence.band].toLowerCase()} confidence)</span>
    </span>
  );
}

/**
 * "Detected behaviour" for continuous results (replaces InteractionSummary): what moves, the seven
 * behaviours with their confidences, the trigger source and the measured appearance. Phase times
 * in a row jump the video there and select the phase on the chart.
 */
export function ContinuousSummary({
  spec,
  rows,
  model,
  onPhase,
}: {
  spec: ContinuousSpec;
  rows: BehaviorRow[];
  model: VelocityModel;
  /** Seek + select a phase (shared with the chart). */
  onPhase: (phaseId: string) => void;
}) {
  const { interaction: ix, continuous: c, source, cursor } = spec;
  const scroller = spec.elements.find((e) => e.id === c.element_id);
  const evidence = rows.find((r) => r.evidence === "velocity+cursor")?.evidence ?? rows.find((r) => r.evidence === "cursor")?.evidence ?? "velocity";
  // Anything beyond autoplay + loop (+ card scaling, which is driven by position, not by the user)
  // means someone interacted (pause, drag, momentum, snap, resume).
  const interacted = rows.some((r) => r.observed && r.key !== "autoplay" && r.key !== "loop" && r.key !== "card_scale");
  const spanS = ((c.span_ms.end_ms - c.span_ms.start_ms) / 1000).toFixed(2);

  return (
    <Panel id="behaviour" title="Detected behaviour" bodyClassName="px-4 py-1">
      <dl>
        <Row label="Type" aside={<ConfidenceBadge confidence={ix.type_confidence} />}>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium">{INTERACTION_TYPE_LABELS[ix.type]}</span>
            <span className="text-ink-3">· {PATTERN_LABELS[ix.pattern].toLowerCase()} pattern</span>
            <LabelSourceTag source={ix.type_source} />
          </span>
          {ix.trigger.description ? <span className="mt-1 block text-xs text-ink-2">{ix.trigger.description}</span> : null}
        </Row>

        <Row label="Scroller" aside={<ConfidenceBadge confidence={c.region_confidence} />}>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium">{ix.target_label}</span>
            {scroller ? <LabelSourceTag source={scroller.label_source} /> : null}
          </span>
          <span className="nums mt-1 block text-xs text-ink-2">
            <span className="font-sans">{AXIS_LABELS[c.axis]}</span> · {formatPx(c.region.w)} × {formatPx(c.region.h)}
            {c.pitch_px ? (
              <>
                {" "}
                · <span className="font-sans">cards every</span> <Hedged n={c.pitch_px} text={`≈${formatPx(c.pitch_px.value)}`} />
                {c.card_scale ? <span className="font-sans"> at the centre</span> : null}
              </>
            ) : null}
            {c.gap_px ? (
              <>
                {" "}
                <span className="font-sans">(gap</span> <Hedged n={c.gap_px} text={formatPx(c.gap_px.value)} />
                <span className="font-sans">)</span>
              </>
            ) : null}
          </span>
        </Row>

        <Row label="Analysed">
          <span className="nums">{spanS} s</span>
          <span className="text-ink-3">
            {" "}
            · {formatPixelRatio(source.pixel_ratio)} {source.pixel_ratio_source === "auto" ? "auto-detected" : "set on upload"}
          </span>
        </Row>
      </dl>

      <h3 className="caption mt-3 border-t border-line pt-3 pb-1">Behaviour</h3>
      <dl>
        {rows.map((r) => (
          <BehaviorItem key={r.key} row={r} model={model} onPhase={onPhase} />
        ))}
        <Row label="Trigger source">
          <span className="font-medium">{PHASE_EVIDENCE_LABELS[evidence]}</span>
          <span className={`mt-1 block text-xs ${cursor.visible || !interacted ? "text-ink-2" : "text-warn"}`}>
            {cursor.visible
              ? "The cursor is visible, so hover, press and release are read from the pointer."
              : interacted
                ? "Cursor not visible in the recording: what starts and stops the motion is inferred from the motion only."
                : "No interaction in the recording: the content moves on its own."}
          </span>
        </Row>
      </dl>

      <AppearanceSection spec={spec} />
    </Panel>
  );
}

function BehaviorItem({ row, model, onPhase }: { row: BehaviorRow; model: VelocityModel; onPhase: (phaseId: string) => void }) {
  if (!row.observed) {
    return (
      <Row label={row.label} muted>
        {row.value}
      </Row>
    );
  }
  // The loop row shares the autoplay phases; one set of jump links (on Autoplay) is enough.
  const phases = row.key === "loop" ? [] : row.phase_ids.map((id) => model.bands.find((b) => b.phase_id === id)).filter((b) => b != null);
  const footer =
    phases.length > 0 || row.facts.length > 0 ? (
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
        {phases.length > 0 ? (
          <span className="inline-flex flex-wrap items-center gap-1">
            <span className="text-2xs text-ink-3">at</span>
            {phases.map((b, i) => {
              const chip = (
                <button
                  key={b.phase_id}
                  type="button"
                  onClick={() => onPhase(b.phase_id)}
                  aria-label={`${b.label} at ${(b.start_ms / 1000).toFixed(2)} s: move the video there and select it on the chart`}
                  className="focus-ring nums rounded-sm border border-line px-1 text-2xs text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
                >
                  {(b.start_ms / 1000).toFixed(1)}
                </button>
              );
              // Keep the unit on the same line as the last time.
              return i < phases.length - 1 ? (
                chip
              ) : (
                <span key={b.phase_id} className="inline-flex items-center gap-1 whitespace-nowrap">
                  {chip}
                  <span className="text-2xs text-ink-3">s</span>
                </span>
              );
            })}
          </span>
        ) : null}
        {row.facts.length > 0 ? (
          <details className="group text-xs [&[open]]:basis-full">
            <summary className="focus-ring inline-flex cursor-pointer list-none items-center gap-1 rounded-sm font-mono text-2xs text-ink-3 select-none hover:text-ink [&::-webkit-details-marker]:hidden">
              <span aria-hidden="true" className="inline-block transition-transform duration-150 group-open:rotate-90">
                ▸
              </span>
              {row.facts.length} value{row.facts.length === 1 ? "" : "s"}
            </summary>
            {/* Narrow (the usual side column, phones): label + badge on one line, the value under them
                at full width. From 26rem: one table row per value. */}
            <div className="@container mt-1.5">
              <dl className="grid gap-y-2 rounded-sm border border-line bg-canvas px-2.5 py-2 @[26rem]:grid-cols-[minmax(0,auto)_minmax(0,1fr)_auto] @[26rem]:gap-x-3 @[26rem]:gap-y-1">
                {row.facts.map((f) => (
                  <div key={f.label} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 @[26rem]:contents">
                    <dt className="text-2xs text-ink-3 @[26rem]:text-xs">{f.label}</dt>
                    <dd className="nums col-span-2 row-start-2 min-w-0 text-ink @[26rem]:col-span-1 @[26rem]:row-start-auto">{f.value}</dd>
                    <dd className="col-start-2 row-start-1 justify-self-end @[26rem]:col-start-auto @[26rem]:row-start-auto @[26rem]:self-start">
                      {f.confidence ? <ConfidenceBadge confidence={f.confidence} /> : null}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </details>
        ) : null}
      </div>
    ) : null;

  return (
    <Row label={row.label} aside={row.confidence ? <ConfidenceBadge confidence={row.confidence} /> : undefined} footer={footer}>
      <span className={row.uncertain ? "text-ink-2" : undefined}>{row.value}</span>
      {row.uncertain || row.flags.length > 0 ? (
        <span className="mt-1 flex flex-wrap gap-1">
          {row.uncertain ? <span className={`${CHIP} border-dashed border-ink-3 text-ink-2`}>uncertain</span> : null}
          {row.flags.map((f) => (
            <span key={f} className={`${CHIP} border-line text-ink-2`}>
              {FLAG_TEXT[f]}
            </span>
          ))}
        </span>
      ) : null}
      {row.notes.length > 0 ? (
        <ul className="mt-1 space-y-0.5 text-xs text-ink-2">
          {row.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : null}

    </Row>
  );
}

/** Measured appearance (PLAN-continuous §13), quiet on purpose: it's context for the prompt, not the result. */
function AppearanceSection({ spec }: { spec: ContinuousSpec }) {
  const items = spec.elements.map((e) => ({ e, facts: appearanceFacts(e.static) })).filter((x) => x.facts.length > 0);
  const page = spec.scene;
  if (items.length === 0 && !page) return null;
  const depth = (id: string | null): number => {
    let d = 0;
    let cur = spec.elements.find((x) => x.id === id);
    while (cur?.parent_id && d < 6) {
      d += 1;
      cur = spec.elements.find((x) => x.id === cur?.parent_id);
    }
    return d;
  };

  return (
    <div className="mt-1 border-t border-line pt-3 pb-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="caption">Appearance · measured</h3>
        <span className="font-mono text-2xs text-ink-3">? = low confidence</span>
      </div>
      <ul className="space-y-1.5">
        {page ? (
          <li className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-baseline gap-x-3 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-x-4">
            <span className="truncate text-xs text-ink-3">Page</span>
            <span className="nums flex flex-wrap items-center gap-x-3.5 gap-y-1 text-xs text-ink-2">
              <span>
                {formatPx(page.viewport_css.w)} × {formatPx(page.viewport_css.h)}
              </span>
              {page.page_background ? (
                <span className="inline-flex items-center gap-1.5" title={`Page background, ${BAND_LABELS[page.page_background.confidence.band].toLowerCase()} confidence`}>
                  <span className="font-mono text-2xs text-ink-3">bg</span>
                  <ColorSwatch hex={formatColor(page.page_background.value)} />
                  {formatColor(page.page_background.value)}
                  {page.page_background.confidence.band === "low" ? "?" : ""}
                </span>
              ) : null}
            </span>
          </li>
        ) : null}
        {items.map(({ e, facts }) => (
          <li key={e.id} className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-baseline gap-x-3 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-x-4">
            <span className="truncate text-xs text-ink-3" title={e.label} style={{ paddingLeft: `${depth(e.id) * 8}px` }}>
              {e.label}
            </span>
            <AppearanceFactList facts={facts} />
          </li>
        ))}
      </ul>
    </div>
  );
}
