"use client";

import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import { PhaseInspector } from "@/components/result/PhaseInspector";
import { BandChips, MarkerGlyph } from "@/components/result/VelocityParts";
import type { Playback } from "@/components/result/usePlayback";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { Panel } from "@/components/ui/Panel";
import { Segmented } from "@/components/ui/Segmented";
import { BAND_LABELS, formatApproxMs, formatPercent, formatSpeed, formatVelocity } from "@/lib/format";
import {
  axisTicks,
  bandAt,
  findBand,
  fractionToTime,
  timeToFraction,
  velocityAt,
  velocityTicks,
  velocityToFraction,
  type VelocityBand,
  type VelocityMarker,
  type VelocityMarkerKind,
  type VelocityModel,
  type VelocityRun,
  type VelocityScale,
} from "@/lib/velocity";

/*
 * Continuous-mode replacement for the motion timeline: signed velocity over the analysed span.
 *
 * Layout: a sticky y-axis gutter + one time column made of three strips that share the same
 * x mapping: the phase strip (keyboard grid), the plot, and the time-axis slider. Everything is
 * positioned in % of the analysed window, so the strips line up without measuring the DOM.
 *
 * Semantics: the phase strip is a one-row role="grid" (same conventions as MotionTimeline:
 * roving tab stop, ←/→ between cells, Home/End, Enter/Space selects = seek + details). The plot
 * is decorative (aria-hidden); its text alternative is the table view, which is also the
 * fallback when the result has no samples. The time axis is a slider, a separate tab stop.
 *
 * Encoding never relies on colour alone: degraded tracking = dashed line without area fill,
 * uncertain phases = hatched + "?", markers = distinct glyph shapes.
 */

const GUTTER = "w-14 shrink-0";
const STRIP = "h-12";
const PLOT = "h-56";
const AXIS = "h-7";
/** Shift / PageUp / PageDown step on the time axis. */
const AXIS_COARSE_MS = 100;
/** SVG user space for the plot (stretched with preserveAspectRatio="none", strokes non-scaling). */
const VB = 1000;

const pct = (f: number) => `${(f * 100).toFixed(3)}%`;
const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
const sec = (ms: number, digits = 2) => (ms / 1000).toFixed(digits);

type View = "chart" | "table";

export function VelocityChart({
  model,
  playback,
  selectedId,
  onSelect,
}: {
  model: VelocityModel;
  playback: Playback;
  /** Committed phase selection (lifted: the behaviour summary and keyframes select too). */
  selectedId: string | null;
  onSelect: (phaseId: string) => void;
}) {
  const [scale, setScale] = useState<VelocityScale>(model.suggestCompressedScale ? "compressed" : "linear");
  const [view, setView] = useState<View>("chart");
  const [hoverId, setHoverId] = useState<string | null>(null);
  /** Cell under keyboard focus, previewed in the inspector until Enter commits it. */
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [focusIndex, setFocusIndex] = useState<number | null>(null);
  const cellRefs = useRef<(HTMLDivElement | null)[]>([]);

  const { bands, window: w } = model;
  const t = playback.currentMs;
  const playBand = bandAt(model, t);
  const selected = selectedId ? findBand(model, selectedId) : null;
  const hovered = hoverId ? findBand(model, hoverId) : null;
  const previewed = previewId ? findBand(model, previewId) : null;
  // Without a selection the details follow the playhead, so playing the video narrates the phases.
  const inspected = hovered ?? previewed ?? selected ?? playBand ?? bands[0] ?? null;

  const showChart = model.hasSamples && view === "chart";
  const showTable = !model.hasSamples || view === "table";
  const showStrip = !model.hasSamples || view === "chart";
  const wide = bands.length > 3;

  const anchor = focusIndex ?? (selected ? bands.indexOf(selected) : -1);
  const stop = clamp(anchor >= 0 ? anchor : playBand ? bands.indexOf(playBand) : 0, 0, Math.max(0, bands.length - 1));

  const select = (band: VelocityBand) => {
    onSelect(band.phase_id);
    playback.seek(band.start_ms);
  };

  const onGridKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.altKey || bands.length === 0) return;
    let i = stop;
    switch (e.key) {
      case "ArrowLeft":
        i -= 1;
        break;
      case "ArrowRight":
        i += 1;
        break;
      case "Home":
        i = 0;
        break;
      case "End":
        i = bands.length - 1;
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        select(bands[stop]);
        return;
      default:
        return;
    }
    e.preventDefault();
    cellRefs.current[clamp(i, 0, bands.length - 1)]?.focus();
  };

  const meta = (
    <>
      {model.hasSamples ? (
        <Segmented
          label="View"
          value={view}
          onChange={setView}
          options={[
            { value: "chart", label: "Chart" },
            { value: "table", label: "Table" },
          ]}
        />
      ) : null}
      {model.suggestCompressedScale && showChart ? (
        <Segmented
          label="Velocity scale"
          value={scale}
          onChange={setScale}
          options={[
            { value: "linear", label: "Linear", title: "Proportional: slow autoplay sits close to zero" },
            { value: "compressed", label: "Compressed", title: "asinh scale: slow autoplay and fast drags both stay readable" },
          ]}
        />
      ) : null}
    </>
  );

  return (
    <Panel id="velocity" title="Velocity profile" meta={meta} bodyClassName="">
      <p id="velocity-grid-help" className="sr-only">
        {bands.length} phase{bands.length === 1 ? "" : "s"} in order. Left and right arrows move between phases, Home and End go to the first and
        last. Enter or Space selects a phase: the video jumps to its start and the details below show its values.
        {model.hasSamples ? " The table view lists the same phases with their values." : ""}
      </p>
      <p id="velocity-axis-help" className="sr-only">
        Left and right arrows move the playhead one frame, Shift plus arrow or Page Up and Page Down move it {AXIS_COARSE_MS} milliseconds, Home and
        End go to the start and end of the analysed span.
      </p>

      {showStrip ? (
        <div className="overflow-x-auto scroll-pl-14">
          <div className={`flex ${wide ? "min-w-[46rem]" : ""}`}>
            <div className={`${GUTTER} sticky left-0 z-20 border-r border-line bg-surface`}>
              <div className={`${STRIP} flex items-end border-b border-line px-2 pb-1.5`}>
                <span className="caption">Phase</span>
              </div>
              {showChart ? <YAxis model={model} scale={scale} /> : <div className="h-16 border-b border-line" />}
              <div className={AXIS} />
            </div>

            <div className="relative min-w-0 flex-1">
              <div
                role="grid"
                aria-labelledby="velocity-title"
                aria-describedby="velocity-grid-help"
                onKeyDown={onGridKeyDown}
                className={`${STRIP} relative border-b border-line`}
              >
                <div role="row" className="absolute inset-0">
                  {bands.map((b, i) => (
                    <div
                      key={b.phase_id}
                      ref={(el) => {
                        cellRefs.current[i] = el;
                      }}
                      role="gridcell"
                      tabIndex={i === stop ? 0 : -1}
                      aria-selected={b.phase_id === selected?.phase_id}
                      aria-label={bandName(b)}
                      onFocus={(e) => {
                        setFocusIndex(i);
                        setPreviewId(e.currentTarget.matches(":focus-visible") ? b.phase_id : null);
                      }}
                      onBlur={() => setPreviewId(null)}
                      onClick={() => select(b)}
                      onMouseEnter={() => setHoverId(b.phase_id)}
                      onMouseLeave={() => setHoverId(null)}
                      className={`@container absolute inset-y-0 cursor-pointer overflow-hidden border-line px-1.5 transition-colors duration-100 focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-solid focus-visible:-outline-offset-2 focus-visible:outline-[var(--focus)] ${
                        i > 0 ? "border-l" : ""
                      } ${b.uncertain ? "hatch" : ""} ${
                        b.phase_id === selected?.phase_id
                          ? "bg-surface-2 shadow-[inset_0_-2px_0_var(--ink)]"
                          : b.phase_id === hoverId || b === playBand
                            ? "bg-surface-2/60"
                            : ""
                      }`}
                      style={{ left: pct(timeToFraction(b.start_ms, w)), width: pct(timeToFraction(b.end_ms, w) - timeToFraction(b.start_ms, w)) }}
                    >
                      <span className="flex h-full flex-col justify-center" aria-hidden="true">
                        <span
                          className={`hidden truncate text-xs leading-4 font-medium @[2.75rem]:block ${b.uncertain ? "text-ink-3 italic" : b === playBand ? "text-ink" : "text-ink-2"}`}
                        >
                          {b.label}
                          {b.uncertain ? "?" : ""}
                        </span>
                        <span className="nums hidden truncate text-2xs leading-4 text-ink-3 @[5.5rem]:block">{b.headline}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {showChart ? (
                <Plot
                  model={model}
                  scale={scale}
                  playheadMs={t}
                  selectedId={selected?.phase_id ?? null}
                  hoverId={hoverId}
                  onHover={setHoverId}
                  onSelect={select}
                />
              ) : (
                <div className="hatch flex h-16 items-center border-b border-line px-3">
                  <p className="rounded-sm bg-surface/90 px-2 py-1 text-xs text-ink-2">
                    No velocity samples were stored for this result, so there is no curve. Every phase and its values are listed below.
                  </p>
                </div>
              )}

              <TimeAxis model={model} playheadMs={t} frameMs={playback.frameMs} onSeek={playback.seek} />
            </div>
          </div>
        </div>
      ) : null}

      {showChart ? <Legend model={model} scale={scale} /> : null}
      {showTable ? <PhaseTable model={model} selectedId={selected?.phase_id ?? null} playBand={playBand} onSelect={select} /> : null}

      <StatusBar model={model} playheadMs={t} playBand={playBand} keys={showStrip} />

      <div className="border-t border-line">
        <PhaseInspector
          band={inspected}
          model={model}
          announce={selected != null && inspected?.phase_id === selected.phase_id}
          following={!hovered && !previewed && !selected && inspected === playBand}
        />
      </div>
    </Panel>
  );
}

function bandName(b: VelocityBand): string {
  return (
    `${b.label}${b.uncertain ? " (uncertain)" : ""}, ${sec(b.start_ms)} to ${sec(b.end_ms)} seconds, ${b.headline}, ` +
    `${BAND_LABELS[b.band].toLowerCase()} confidence ${formatPercent(b.confidence)}` +
    (b.interrupted ? ", cut short" : "") +
    (b.degraded ? ", tracking less precise" : "")
  );
}

/* ---------- y axis ---------- */

function tickText(v: number): string {
  const a = Math.abs(v);
  return a >= 10000 ? `${a / 1000}k` : String(a);
}

function YAxis({ model, scale }: { model: VelocityModel; scale: VelocityScale }) {
  const ticks = velocityTicks(model, scale);
  return (
    <div className={`relative ${PLOT} border-b border-line`} aria-hidden="true">
      <div className="absolute inset-x-0 top-3 bottom-3">
        {ticks.map((v) => {
          const f = velocityToFraction(v, model, scale);
          // Edge ticks sit inside the plot; the rest are centred on their gridline.
          const align = f > 0.97 || f < 0.03 ? "" : "translate-y-1/2";
          return (
            <span
              key={v}
              className={`nums absolute right-2 text-2xs ${v === 0 ? "text-ink-2" : "text-ink-3"} ${align}`}
              style={{ bottom: f > 0.97 ? undefined : pct(f), top: f > 0.97 ? 0 : undefined }}
            >
              {tickText(v)}
            </span>
          );
        })}
      </div>
      <span className="absolute bottom-0.5 left-2 font-mono text-2xs text-ink-3">px/s</span>
    </div>
  );
}

/* ---------- plot ---------- */

function runPoints(run: VelocityRun, model: VelocityModel, scale: VelocityScale): string {
  return run.points
    .map((p) => `${(timeToFraction(p.t_ms, model.window) * VB).toFixed(2)},${((1 - velocityToFraction(p.v, model, scale)) * VB).toFixed(2)}`)
    .join(" ");
}

function Plot({
  model,
  scale,
  playheadMs,
  selectedId,
  hoverId,
  onHover,
  onSelect,
}: {
  model: VelocityModel;
  scale: VelocityScale;
  playheadMs: number;
  selectedId: string | null;
  hoverId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (band: VelocityBand) => void;
}) {
  const w = model.window;
  const ticks = velocityTicks(model, scale);
  const zeroY = (1 - velocityToFraction(0, model, scale)) * VB;
  const ref = model.autoplayRef;
  const refF = ref ? velocityToFraction(ref.v, model, scale) : null;
  const playIn = playheadMs >= w.start_ms && playheadMs <= w.end_ms;
  const at = playIn ? velocityAt(model, playheadMs) : null;
  const dots = model.runs.filter((r) => r.points.length === 1);

  return (
    <div className={`relative ${PLOT} border-b border-line`}>
      {/* Phase columns: hover preview + click to select (keyboard path is the strip above). */}
      {model.bands.map((b, i) => (
        <div
          key={b.phase_id}
          aria-hidden="true"
          onClick={() => onSelect(b)}
          onMouseEnter={() => onHover(b.phase_id)}
          onMouseLeave={() => onHover(null)}
          className={`absolute inset-y-0 cursor-pointer border-line/70 transition-colors duration-100 ${i > 0 ? "border-l" : ""} ${b.uncertain ? "hatch" : ""} ${
            b.phase_id === selectedId ? "bg-surface-2" : b.phase_id === hoverId ? "bg-surface-2/50" : ""
          }`}
          style={{ left: pct(timeToFraction(b.start_ms, w)), width: pct(timeToFraction(b.end_ms, w) - timeToFraction(b.start_ms, w)) }}
        />
      ))}

      <div className="pointer-events-none absolute inset-x-0 top-3 bottom-3" aria-hidden="true">
        {ticks.map((v) =>
          v === 0 ? null : <span key={v} className="absolute inset-x-0 h-px bg-line/70" style={{ bottom: pct(velocityToFraction(v, model, scale)) }} />,
        )}
        <span className="absolute inset-x-0 h-px bg-line-strong" style={{ bottom: pct(velocityToFraction(0, model, scale)) }} />

        {ref && refF != null ? (
          <>
            <span className="absolute inset-x-0 border-t border-dashed border-ink-3" style={{ bottom: pct(refF) }} />
            <span
              className={`nums absolute right-1.5 rounded-sm bg-surface/85 px-1 text-2xs text-ink-2 ${refF > 0.85 ? "translate-y-full pt-0.5" : "-translate-y-0.5"}`}
              style={{ bottom: pct(refF) }}
            >
              autoplay {formatSpeed(ref.v)}
            </span>
          </>
        ) : null}

        <svg viewBox={`0 0 ${VB} ${VB}`} preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible">
          {model.runs.map((run, i) =>
            run.points.length > 1 ? (
              <g key={i}>
                {!run.degraded ? (
                  <polygon
                    points={`${(timeToFraction(run.points[0].t_ms, w) * VB).toFixed(2)},${zeroY.toFixed(2)} ${runPoints(run, model, scale)} ${(
                      timeToFraction(run.points[run.points.length - 1].t_ms, w) * VB
                    ).toFixed(2)},${zeroY.toFixed(2)}`}
                    className="fill-signal-fill"
                    fillOpacity={0.1}
                  />
                ) : null}
                <polyline
                  points={runPoints(run, model, scale)}
                  fill="none"
                  className="stroke-signal-fill"
                  strokeWidth={run.degraded ? 1.5 : 1.75}
                  strokeDasharray={run.degraded ? "4 3" : undefined}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                />
              </g>
            ) : null,
          )}
        </svg>
        {dots.map((r) => (
          <span
            key={r.points[0].t_ms}
            className={`absolute size-1.5 -translate-x-1/2 translate-y-1/2 rounded-full ${r.degraded ? "border border-signal-fill bg-surface" : "bg-signal-fill"}`}
            style={{ left: pct(timeToFraction(r.points[0].t_ms, w)), bottom: pct(velocityToFraction(r.points[0].v, model, scale)) }}
          />
        ))}

        {/* Momentum that blends back into autoplay: ring where the decay meets the autoplay line. */}
        {model.markers.map((m) => {
          if (m.kind !== "autoplay_join") return null;
          const fit = model.bands.find((b) => b.phase_id === m.phase_id)?.fit;
          if (fit?.model !== "exponential") return null;
          return (
            <span
              key={m.id}
              className="absolute size-2 -translate-x-1/2 translate-y-1/2 rounded-full border border-ink-2 bg-surface"
              style={{ left: pct(timeToFraction(m.t_ms, w)), bottom: pct(velocityToFraction(fit.v_inf_px_s, model, scale)) }}
            />
          );
        })}

        {at ? (
          <span
            className="absolute z-10 size-2 -translate-x-1/2 translate-y-1/2 rounded-full border-2 border-surface bg-ink"
            style={{ left: pct(timeToFraction(playheadMs, w)), bottom: pct(velocityToFraction(at.v, model, scale)) }}
          />
        ) : null}
      </div>

      {/* Direction of the sign convention, inside the plot so it never fights the tick labels. */}
      <span className="pointer-events-none absolute top-1 left-1.5 rounded-sm bg-surface/85 px-1 font-mono text-2xs text-ink-3" aria-hidden="true">
        moving {model.positiveLabel}
      </span>
      <span className="pointer-events-none absolute bottom-1 left-1.5 rounded-sm bg-surface/85 px-1 font-mono text-2xs text-ink-3" aria-hidden="true">
        moving {model.negativeLabel}
      </span>

      {model.markers.map((m) => (
        <MarkerLine key={m.id} marker={m} f={timeToFraction(m.t_ms, w)} />
      ))}

      {playIn ? <span className="pointer-events-none absolute inset-y-0 z-10 w-px bg-ink" style={{ left: pct(timeToFraction(playheadMs, w)) }} aria-hidden="true" /> : null}
    </div>
  );
}

/* ---------- markers ---------- */

const MARKER_LINE: Record<VelocityMarkerKind, string> = {
  release: "border-dashed border-ink-3",
  rest: "border-dotted border-ink-3",
  resume_start: "border-dotted border-ink-3",
  autoplay_join: "border-dotted border-ink-3",
  cursor_enter: "border-dotted border-line-strong",
  cursor_leave: "border-dotted border-line-strong",
};

const MARKER_LEGEND: Record<VelocityMarkerKind, string> = {
  release: "released",
  rest: "comes to rest",
  resume_start: "autoplay resumes",
  autoplay_join: "blends into autoplay",
  cursor_enter: "cursor enters",
  cursor_leave: "cursor leaves",
};

function MarkerLine({ marker, f }: { marker: VelocityMarker; f: number }) {
  return (
    <span className="pointer-events-none absolute inset-y-0" style={{ left: pct(f) }} aria-hidden="true">
      <span className={`absolute inset-y-0 border-l ${MARKER_LINE[marker.kind]}`} />
      <span className="absolute top-1 -translate-x-1/2 text-ink-2" title={`${marker.label} at ${sec(marker.t_ms)} s`}>
        <MarkerGlyph kind={marker.kind} className="block" />
      </span>
    </span>
  );
}

/* ---------- time axis (slider) ---------- */

function tickLabel(ms: number, step: number): string {
  const digits = step >= 1000 ? 0 : step >= 100 ? 1 : 2;
  return `${(ms / 1000).toFixed(digits)}`;
}

function TimeAxis({ model, playheadMs, frameMs, onSeek }: { model: VelocityModel; playheadMs: number; frameMs: number; onSeek: (ms: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const w = model.window;
  const ticks = axisTicks(w, 6);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1000;
  const inside = playheadMs >= w.start_ms && playheadMs <= w.end_ms;
  const value = clamp(playheadMs, w.start_ms, w.end_ms);
  const band = bandAt(model, value);

  const seekTo = (ms: number) => onSeek(clamp(ms, w.start_ms, w.end_ms));
  const fromPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (rect && rect.width > 0) seekTo(fractionToTime((clientX - rect.left) / rect.width, w));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const s = e.shiftKey ? AXIS_COARSE_MS : frameMs;
    const next: Record<string, number> = {
      ArrowRight: value + s,
      ArrowUp: value + s,
      ArrowLeft: value - s,
      ArrowDown: value - s,
      PageUp: value + AXIS_COARSE_MS,
      PageDown: value - AXIS_COARSE_MS,
      Home: w.start_ms,
      End: w.end_ms,
    };
    if (!(e.key in next) || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    seekTo(next[e.key]);
  };

  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label="Velocity chart time axis"
      aria-describedby="velocity-axis-help"
      aria-orientation="horizontal"
      aria-valuemin={Math.round(w.start_ms)}
      aria-valuemax={Math.round(w.end_ms)}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${sec(playheadMs)} seconds${inside ? (band ? `, ${band.label}` : "") : ", outside the analysed span"}`}
      onKeyDown={onKeyDown}
      onPointerDown={(e: PointerEvent<HTMLDivElement>) => {
        if (e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        fromPointer(e.clientX);
      }}
      onPointerMove={(e: PointerEvent<HTMLDivElement>) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) fromPointer(e.clientX);
      }}
      className={`${AXIS} relative cursor-col-resize touch-none select-none transition-colors duration-100 hover:bg-surface-2/60 focus-visible:outline-2 focus-visible:outline-solid focus-visible:-outline-offset-2 focus-visible:outline-[var(--focus)]`}
    >
      <div ref={trackRef} className="absolute inset-0" aria-hidden="true">
        {ticks.map((tk, i) => {
          const f = timeToFraction(tk, w);
          const align = f < 0.04 ? "translate-x-1" : f > 0.96 ? "-translate-x-full pr-1" : "-translate-x-1/2";
          return (
            <span key={tk} className="absolute top-0 h-full" style={{ left: pct(f) }}>
              <span className="absolute top-0 h-1.5 w-px bg-line-strong" />
              <span className={`nums absolute top-1.5 text-2xs whitespace-nowrap text-ink-3 ${align}`}>
                {tickLabel(tk, step)}
                {i === ticks.length - 1 ? " s" : ""}
              </span>
            </span>
          );
        })}
        {inside ? <span className="absolute top-0 h-2 w-[3px] -translate-x-px rounded-b-[1px] bg-ink" style={{ left: pct(timeToFraction(playheadMs, w)) }} /> : null}
      </div>
    </div>
  );
}

/* ---------- legend ---------- */

function Legend({ model, scale }: { model: VelocityModel; scale: VelocityScale }) {
  const kinds = [...new Set(model.markers.map((m) => m.kind))];
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-line px-3 py-2 text-xs text-ink-2" aria-label="Chart legend">
      <li className="inline-flex items-center gap-1.5">
        <svg viewBox="0 0 24 10" className="h-2.5 w-6" aria-hidden="true">
          <path d="M0 9 L24 9 L24 4 L0 4Z" className="fill-signal-fill" fillOpacity={0.12} />
          <path d="M0 4 H24" className="stroke-signal-fill" strokeWidth={1.75} />
        </svg>
        measured velocity
      </li>
      {model.degradedCount > 0 ? (
        <li className="inline-flex items-center gap-1.5">
          <svg viewBox="0 0 24 10" className="h-2.5 w-6" aria-hidden="true">
            <path d="M0 5 H24" className="stroke-signal-fill" strokeWidth={1.5} strokeDasharray="4 3" />
          </svg>
          less precise tracking
        </li>
      ) : null}
      {model.autoplayRef ? (
        <li className="inline-flex items-center gap-1.5">
          <span className="inline-block w-6 border-t border-dashed border-ink-3" aria-hidden="true" />
          autoplay speed
        </li>
      ) : null}
      {model.uncertainCount > 0 ? (
        <li className="inline-flex items-center gap-1.5">
          <span className="hatch inline-block h-2.5 w-6 rounded-[2px] border border-line" aria-hidden="true" />
          uncertain phase
        </li>
      ) : null}
      {kinds.map((k) => (
        <li key={k} className="inline-flex items-center gap-1.5">
          <MarkerGlyph kind={k} />
          {MARKER_LEGEND[k]}
        </li>
      ))}
      {scale === "compressed" ? <li className="font-mono text-2xs text-ink-3 sm:ml-auto">compressed scale: equal steps are ×10</li> : null}
    </ul>
  );
}

/* ---------- table view (text alternative; the only view without samples) ---------- */

function PhaseTable({
  model,
  selectedId,
  playBand,
  onSelect,
}: {
  model: VelocityModel;
  selectedId: string | null;
  playBand: VelocityBand | null;
  onSelect: (band: VelocityBand) => void;
}) {
  const th = "caption px-3 py-2 text-left font-normal whitespace-nowrap";
  return (
    <div className="border-t border-line">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[44rem] text-sm">
          <caption className="sr-only">
            Phases of the motion in order: time, duration, velocity at start and end, key value and label confidence. Activate a phase to move the
            video to its start.
          </caption>
          <thead>
            <tr className="border-b border-line">
              <th scope="col" className={`${th} w-10`}>
                #
              </th>
              <th scope="col" className={th}>
                Phase
              </th>
              <th scope="col" className={th}>
                Video time
              </th>
              <th scope="col" className={th}>
                Velocity
              </th>
              <th scope="col" className={th}>
                Key value
              </th>
              <th scope="col" className={th}>
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {model.bands.map((b, i) => {
              const sel = b.phase_id === selectedId;
              return (
                <tr key={b.phase_id} className={`border-b border-line last:border-b-0 ${sel ? "bg-surface-2" : b === playBand ? "bg-surface-2/50" : ""}`}>
                  <td className="nums px-3 py-2 align-top text-2xs text-ink-3">{String(i + 1).padStart(2, "0")}</td>
                  <th scope="row" className="px-3 py-2 text-left align-top font-normal">
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <button
                        type="button"
                        onClick={() => onSelect(b)}
                        aria-current={sel ? "true" : undefined}
                        className={`focus-ring rounded-sm font-medium underline-offset-4 hover:underline ${b.uncertain ? "text-ink-2 italic" : "text-ink"}`}
                      >
                        {b.label}
                        {b.uncertain ? "?" : ""}
                        <span className="sr-only">, move the video to {sec(b.start_ms)} s</span>
                      </button>
                      <BandChips band={b} />
                    </span>
                  </th>
                  <td className="nums px-3 py-2 align-top whitespace-nowrap text-ink-2">
                    {sec(b.start_ms)}–{sec(b.end_ms)} s<span className="block text-2xs text-ink-3">{formatApproxMs(b.duration_ms)}</span>
                  </td>
                  <td className="nums px-3 py-2 align-top text-xs whitespace-nowrap text-ink-2">
                    {formatVelocity(b.v_start, model.axis)}
                    <span className="text-ink-3"> → </span>
                    {formatVelocity(b.v_end, model.axis)}
                  </td>
                  <td className="nums px-3 py-2 align-top text-xs text-ink">{b.headline}</td>
                  <td className="px-3 py-2 align-top">
                    <ConfidenceBadge confidence={{ value: b.confidence, band: b.band }} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {model.markers.length > 0 ? (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-line px-3 py-2 text-xs text-ink-2">
          <h3 className="caption">Events</h3>
          <ul className="flex flex-wrap gap-x-4 gap-y-1">
            {model.markers.map((m) => (
              <li key={m.id} className="inline-flex items-center gap-1.5">
                <MarkerGlyph kind={m.kind} className="text-ink-2" />
                {m.label}
                <span className="nums text-ink-3">{sec(m.t_ms)} s</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/* ---------- status bar ---------- */

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-[2px] border border-line px-1 font-mono text-2xs text-ink-2">{children}</kbd>;
}

function StatusBar({ model, playheadMs, playBand, keys }: { model: VelocityModel; playheadMs: number; playBand: VelocityBand | null; keys: boolean }) {
  const at = useMemo(() => velocityAt(model, playheadMs), [model, playheadMs]);
  const inside = playheadMs >= model.window.start_ms && playheadMs <= model.window.end_ms;
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-line px-3 py-1.5 font-mono text-2xs text-ink-3">
      <span className="nums" aria-hidden="true">
        playhead <span className="text-ink-2">{sec(playheadMs, 3)} s</span>
        {!inside ? " · outside the analysed span" : null}
        {playBand ? (
          <>
            {" · "}
            <span className="text-ink-2">{playBand.label}</span>
          </>
        ) : null}
        {at ? (
          <>
            {" · "}
            <span className="text-ink-2">{formatVelocity(at.v, model.axis)}</span>
            {at.degraded ? " (less precise)" : ""}
          </>
        ) : null}
      </span>
      {/* Key hints only while the strip and axis they describe are on screen (not in the table view). */}
      <span className={`hidden items-center gap-1.5 ${keys ? "md:inline-flex" : ""}`} aria-hidden="true">
        phases <Kbd>←→</Kbd> move <Kbd>Enter</Kbd> select
        <span className="px-1 text-line-strong">|</span>
        axis <Kbd>←→</Kbd> 1 frame <Kbd>Shift</Kbd> {AXIS_COARSE_MS} ms
      </span>
    </div>
  );
}
