import type { ReactNode } from "react";

import { AppearanceFactList } from "@/components/result/AppearanceFacts";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { LabelSourceTag } from "@/components/ui/LabelSourceTag";
import {
  formatCubicBezier,
  formatEasing,
  formatMs,
  formatNumber,
  formatPercent,
  formatPx,
  formatScaleChange,
  formatValue,
  SEGMENT_LABELS,
  type AppearanceFact,
} from "@/lib/format";
import type { TimelineRow } from "@/lib/timeline";
import type { Easing, LabelSource, Transition, Value } from "@/lib/types";

function Swatch({ value }: { value: Value }) {
  if (value.kind !== "color") return null;
  return <span className="inline-block size-3 rounded-[2px] border border-line-strong align-[-1px]" style={{ background: value.color }} aria-hidden="true" />;
}

function deltaText(row: TimelineRow): string | null {
  if (row.delta == null) return null;
  if (row.from.kind === "px") return formatPx(row.delta, { signed: true });
  if (row.from.kind === "ratio" && row.to.kind === "ratio") {
    if (row.property === "opacity") return `${row.delta > 0 ? "+" : ""}${formatNumber(row.delta, 2)}`;
    return row.from.number !== 0 ? formatScaleChange(row.from.number, row.to.number) : null;
  }
  return null;
}

function Fact({ label, children, wide = false }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : ""}>
      <dt className="caption">{label}</dt>
      <dd className="mt-0.5 text-sm text-ink">{children}</dd>
    </div>
  );
}

/** Readout for the hovered / focused / selected bar. */
export function TimelineInspector({
  row,
  transition,
  labelSource,
  source,
  appearance = [],
}: {
  row: TimelineRow | null;
  transition: Transition | null;
  labelSource: LabelSource | null;
  /** Measured static appearance of the row's element (empty = nothing measured, nothing shown). */
  appearance?: AppearanceFact[];
  /** "preview" = hovered or keyboard-focused bar; only a committed selection is announced. */
  source: "preview" | "selected";
}) {
  if (!row) {
    return <p className="px-4 py-4 text-sm text-ink-3">Select a bar to inspect its measurement.</p>;
  }
  const delta = deltaText(row);
  const c = transition?.confidence;

  return (
    <div className="grid gap-6 px-4 py-4 md:grid-cols-[minmax(0,1fr)_10.5rem]">
      <div className="min-w-0 space-y-4">
        {/* Badge in its own column so it never wraps under the title on narrow screens. */}
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <h3 className="text-md font-medium text-ink">
              {row.element_label} <span className="text-ink-3">·</span> {row.property_label}
            </h3>
            {labelSource ? <LabelSourceTag source={labelSource} /> : null}
            <span className="rounded-sm border border-line px-1.5 font-mono text-2xs text-ink-2">{SEGMENT_LABELS[row.segment_id]}</span>
            {row.uncertain ? <span className="rounded-sm border border-dashed border-ink-3 px-1.5 font-mono text-2xs text-ink-2">uncertain</span> : null}
          </div>
          <ConfidenceBadge confidence={{ value: row.confidence, band: row.confidence_band }} showPercent className="mt-1" />
        </div>
        <span className="sr-only" aria-live="polite">
          {source === "selected" ? `Selected ${row.element_label} ${row.property_label}, ${SEGMENT_LABELS[row.segment_id].toLowerCase()}` : ""}
        </span>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
          <Fact label="Change" wide>
            <span className="nums inline-flex flex-wrap items-center gap-1.5 break-all">
              <Swatch value={row.from} />
              {formatValue(row.from, row.property)}
              <span className="text-ink-3">→</span>
              <Swatch value={row.to} />
              {formatValue(row.to, row.property)}
              {delta ? <span className="text-ink-3">({delta})</span> : null}
            </span>
          </Fact>
          <Fact label="Duration">
            <span className="nums">{formatMs(row.duration_ms, { approx: true })}</span>
          </Fact>
          <Fact label="Delay">
            <span className="nums">{row.delay_ms > 0 ? `+${formatMs(row.delay_ms)}` : "0 ms"}</span>
          </Fact>
          <Fact label="Easing" wide>
            <span className="nums block">{formatEasing(row.easing)}</span>
            <span className="nums block text-xs text-ink-3">
              {row.easing.keyword ? formatCubicBezier(row.easing.cubic_bezier) : `close to ${row.easing.nearest_named}`}
              {row.easing.flags.includes("overshoot") ? " · overshoots" : ""}
            </span>
          </Fact>
          <Fact label="Video time">
            <span className="nums">
              {Math.round(row.start_ms)}–{Math.round(row.end_ms)}
            </span>
            <span className="nums text-xs text-ink-3"> ms</span>
          </Fact>
          <Fact label="Origin">
            <span className={row.transform_origin ? "font-mono text-xs" : "text-ink-3"}>{row.transform_origin ?? "—"}</span>
          </Fact>
          {c ? (
            <Fact label="Confidence breakdown" wide>
              <span className="nums flex flex-wrap gap-x-4 text-xs">
                <span>
                  <span className="text-ink-3">value </span>
                  {formatPercent(c.value)}
                </span>
                <span>
                  <span className="text-ink-3">timing </span>
                  {formatPercent(c.timing)}
                </span>
                <span>
                  <span className="text-ink-3">easing </span>
                  {formatPercent(c.easing)}
                </span>
              </span>
            </Fact>
          ) : null}
          {appearance.length > 0 ? (
            <Fact label="Appearance · measured" wide>
              <AppearanceFactList facts={appearance} className="mt-0.5" />
            </Fact>
          ) : null}
          {row.notes.length > 0 ? (
            <Fact label="Notes" wide>
              <ul className="space-y-0.5 text-xs text-ink-2">
                {row.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Fact>
          ) : null}
        </dl>
      </div>

      <EasingPlot easing={row.easing} samples={row.samples} start={row.start_ms} duration={row.duration_ms} />
    </div>
  );
}

/* ---------- fitted curve vs measured samples ---------- */

const SIZE = 120;
const PAD = 10;
const Y_MIN = -0.15;
const Y_MAX = 1.2;

const px = (x: number) => PAD + x * (SIZE - 2 * PAD);
const py = (y: number) => PAD + ((Y_MAX - Math.min(Y_MAX, Math.max(Y_MIN, y))) / (Y_MAX - Y_MIN)) * (SIZE - 2 * PAD);

function bezierPath([x1, y1, x2, y2]: Easing["cubic_bezier"]): string {
  const pts: string[] = [];
  for (let i = 0; i <= 48; i++) {
    const t = i / 48;
    const u = 1 - t;
    const x = 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t;
    const y = 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t;
    pts.push(`${i === 0 ? "M" : "L"}${px(x).toFixed(2)},${py(y).toFixed(2)}`);
  }
  return pts.join(" ");
}

function EasingPlot({ easing, samples, start, duration }: { easing: Easing; samples: [number, number][]; start: number; duration: number }) {
  const points = duration > 0 ? samples.map(([t, p]) => [(t - start) / duration, p] as const).filter(([x]) => x >= -0.02 && x <= 1.02) : [];
  return (
    <figure className="w-[10.5rem] justify-self-start md:justify-self-end">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="h-auto w-full rounded-sm border border-line bg-canvas"
        role="img"
        aria-label={`Easing: fitted ${formatEasing(easing)} curve plotted against ${points.length} measured samples`}
      >
        <line x1={px(0)} x2={px(1)} y1={py(1)} y2={py(1)} className="stroke-line-strong" strokeDasharray="2 3" strokeWidth={1} />
        <line x1={px(0)} x2={px(1)} y1={py(0)} y2={py(0)} className="stroke-line-strong" strokeWidth={1} />
        <line x1={px(0)} x2={px(0)} y1={py(0)} y2={py(1)} className="stroke-line" strokeWidth={1} />
        <line x1={px(0)} x2={px(1)} y1={py(0)} y2={py(1)} className="stroke-line" strokeWidth={1} strokeDasharray="1 3" />
        <path d={bezierPath(easing.cubic_bezier)} fill="none" className="stroke-signal-fill" strokeWidth={1.75} strokeLinecap="round" />
        {points.map(([x, p], i) => (
          <circle key={i} cx={px(x)} cy={py(p)} r={1.7} className="fill-ink" />
        ))}
      </svg>
      <figcaption className="mt-1.5 flex items-center justify-between font-mono text-2xs text-ink-3">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-0.5 w-3 bg-signal-fill" aria-hidden="true" />
          fit
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-1.5 rounded-full bg-ink" aria-hidden="true" />
          measured
        </span>
      </figcaption>
    </figure>
  );
}
