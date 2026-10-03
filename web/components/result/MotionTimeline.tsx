"use client";

import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import { TimelineInspector } from "@/components/result/TimelineInspector";
import type { Playback } from "@/components/result/usePlayback";
import { LabelSourceTag } from "@/components/ui/LabelSourceTag";
import { Panel } from "@/components/ui/Panel";
import { BAND_LABELS, formatChange, formatEasing, formatMs, formatPercent, REVERSE_TRIGGER_LABELS, TRIGGER_LABELS } from "@/lib/format";
import {
  activeRowsAt,
  axisTicks,
  buildTimeline,
  findRow,
  fractionToTime,
  rowSpan,
  timeToFraction,
  type TimelineLane,
  type TimelineModel,
  type TimelineRow,
  type TimelineTrack,
} from "@/lib/timeline";
import type { LabelSource, MotionSpec } from "@/lib/types";

/*
 * Layout: row-major so every ARIA row owns its cells. Each horizontal strip (header, overlay,
 * grid rows) repeats the same column classes, so the lanes line up without a <table>. The label
 * column is sticky inside the horizontal scroller.
 *
 * Semantics: role="grid". The data is two-dimensional (element × property rows, one column per
 * motion segment), navigation is 2-D (↑/↓ rows, ←/→ lanes) and a cell can be empty when a lane
 * lacks that track, which a listbox (1-D, options only) cannot express. The time axes are
 * sliders, so they sit outside the grid and are separate tab stops.
 */
const HEAD = "h-12";
const AXIS = "h-7";
const ROW = "h-9";
const LABEL_COL = "w-32 shrink-0 sm:w-64";
const LANE_COL = "min-w-[17rem] flex-1 sm:min-w-[20rem]";
const MISSING_COL = "min-w-[13rem] flex-[0.6]";
/** Shift / PageUp / PageDown step on the time axis. */
const AXIS_COARSE_MS = 50;

const pct = (f: number) => `${(f * 100).toFixed(3)}%`;
const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

interface Pos {
  r: number;
  c: number;
}

export function MotionTimeline({ spec, playback }: { spec: MotionSpec; playback: Playback }) {
  const [showUncertain, setShowUncertain] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  /** Bar under keyboard focus (previewed in the inspector until Enter commits it). */
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [focusPos, setFocusPos] = useState<Pos | null>(null);
  const cellRefs = useRef(new Map<string, HTMLDivElement>());

  const model = useMemo(() => buildTimeline(spec, { includeUncertain: showUncertain }), [spec, showUncertain]);
  const { tracks, lanes } = model;
  const matrix = useMemo(() => tracks.map((tr) => lanes.map((l) => l.rows.find((r) => r.key === tr.key) ?? null)), [tracks, lanes]);
  const groupStart = useMemo(() => tracks.map((tr, i) => i === 0 || tracks[i - 1].element_id !== tr.element_id), [tracks]);
  const labelSources = useMemo(() => new Map(spec.elements.map((e) => [e.id, e.label_source])), [spec.elements]);
  const uncertainTracks = useMemo(() => uncertainTrackKeys(model), [model]);

  const firstRow = lanes.find((l) => l.rows.length > 0)?.rows[0] ?? null;
  const selected = (selectedId ? findRow(model, selectedId) : null) ?? firstRow;
  const hovered = hoverId ? findRow(model, hoverId) : null;
  const previewed = previewId ? findRow(model, previewId) : null;
  const inspected = hovered ?? previewed ?? selected;

  const t = playback.currentMs;
  const activeRows = activeRowsAt(model, t);
  const activeIds = new Set(activeRows.map((r) => r.transition_id));
  const activeKeys = new Set(activeRows.map((r) => r.key));
  const forwardOnly = lanes.length === 1 && lanes[0].kind === "forward";

  // Roving tab stop: the last focused cell, else the selected bar, else the first cell.
  const R = tracks.length;
  const C = lanes.length;
  const anchor = focusPos ?? (selected ? positionOf(matrix, selected.transition_id) : null) ?? { r: 0, c: 0 };
  const stop: Pos = { r: clamp(anchor.r, 0, Math.max(0, R - 1)), c: clamp(anchor.c, 0, Math.max(0, C - 1)) };

  const select = (row: TimelineRow) => {
    setSelectedId(row.transition_id);
    playback.seek(row.start_ms);
  };

  const focusCell = (p: Pos) => cellRefs.current.get(`${p.r}:${p.c}`)?.focus();

  const onGridKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.altKey || R === 0) return;
    let { r, c } = stop;
    const toEdge = e.ctrlKey || e.metaKey;
    switch (e.key) {
      case "ArrowUp":
        r -= 1;
        break;
      case "ArrowDown":
        r += 1;
        break;
      case "ArrowLeft":
        c -= 1;
        break;
      case "ArrowRight":
        c += 1;
        break;
      case "Home":
        c = 0;
        if (toEdge) r = 0;
        break;
      case "End":
        c = C - 1;
        if (toEdge) r = R - 1;
        break;
      case "PageUp": {
        let i = r - 1;
        while (i > 0 && !groupStart[i]) i -= 1;
        r = Math.max(0, i);
        break;
      }
      case "PageDown": {
        let i = r + 1;
        while (i < R && !groupStart[i]) i += 1;
        r = Math.min(R - 1, i);
        break;
      }
      case "Enter":
      case " ": {
        e.preventDefault();
        const row = matrix[r]?.[c];
        if (row) select(row);
        return;
      }
      default:
        return;
    }
    e.preventDefault();
    focusCell({ r: clamp(r, 0, R - 1), c: clamp(c, 0, C - 1) });
  };

  const legend = (
    <>
      <Legend />
      {model.uncertainCount > 0 ? (
        <label className="ml-1 inline-flex cursor-pointer items-center gap-1.5 text-xs text-ink-2">
          <input
            type="checkbox"
            checked={showUncertain}
            onChange={(e) => setShowUncertain(e.target.checked)}
            className="focus-ring size-3.5 accent-[var(--ink)]"
          />
          Show uncertain <span className="nums text-ink-3">({model.uncertainCount})</span>
        </label>
      ) : null}
    </>
  );

  return (
    <Panel id="timeline" title="Motion timeline" meta={legend} bodyClassName="">
      {model.isEmpty ? (
        <EmptyTimeline hiddenUncertain={!showUncertain ? model.uncertainCount : 0} onShow={() => setShowUncertain(true)} />
      ) : (
        <>
          <p id="timeline-grid-help" className="sr-only">
            {model.rowCount} measured transitions, one row per element property, one column per {lanes.map((l) => l.label.toLowerCase()).join(" and ")}{" "}
            motion. Arrow keys move between cells, Page Up and Page Down jump between elements, Home and End go to the first and last column. Enter
            or Space selects a transition: the inspector below shows it and the playhead moves to its start.
          </p>
          <p id="timeline-axis-help" className="sr-only">
            Left and right arrows move the playhead one frame, Shift plus arrow or Page Up and Page Down move it {AXIS_COARSE_MS} milliseconds, Home and
            End go to the start and end of this lane.
          </p>

          <div className="overflow-x-auto scroll-pl-32 sm:scroll-pl-64">
            <div className="min-w-max">
              {/* Lane titles + time axes. Outside the grid: the axes are sliders, not cells. */}
              <div className="flex">
                <div className={`${LABEL_COL} sticky left-0 z-20 border-r border-line bg-surface`}>
                  <div className={`${HEAD} flex items-end border-b border-line px-3 pb-1.5`}>
                    <span className="caption">Element · property</span>
                  </div>
                  <div className={`${AXIS} border-b border-line`} />
                </div>
                {lanes.map((lane, c) => (
                  <div key={lane.segment} className={`${LANE_COL} border-line ${c < C - 1 || forwardOnly ? "border-r" : ""}`}>
                    <LaneHeader lane={lane} spec={spec} />
                    <AxisSlider lane={lane} spec={spec} playheadMs={t} frameMs={playback.frameMs} onSeek={playback.seek} />
                  </div>
                ))}
                {forwardOnly ? <MissingReverseHeader /> : null}
              </div>

              <div className="relative">
                <div className="pointer-events-none absolute inset-0 flex">
                  <div className={LABEL_COL} aria-hidden="true" />
                  {lanes.map((lane, c) => (
                    <LaneOverlay
                      key={lane.segment}
                      lane={lane}
                      model={model}
                      playheadMs={t}
                      className={`${LANE_COL} border-line ${c < C - 1 || forwardOnly ? "border-r" : ""}`}
                    />
                  ))}
                  {forwardOnly ? <MissingReverseBody /> : null}
                </div>

                <div role="grid" aria-labelledby="timeline-title" aria-describedby="timeline-grid-help" onKeyDown={onGridKeyDown}>
                  <div role="row" className="sr-only">
                    <span role="columnheader">Element and property</span>
                    {lanes.map((lane) => (
                      <span key={lane.segment} role="columnheader">
                        {lane.label} motion
                      </span>
                    ))}
                  </div>

                  {tracks.map((track, r) => (
                    <div key={track.key} role="row" className={`${ROW} flex ${groupStart[r] && r > 0 ? "border-t border-line" : ""}`}>
                      <TrackHeader
                        track={track}
                        first={groupStart[r]}
                        active={activeKeys.has(track.key)}
                        uncertain={uncertainTracks.has(track.key)}
                        labelSource={labelSources.get(track.element_id) ?? null}
                      />
                      {lanes.map((lane, c) => {
                        const row = matrix[r][c];
                        const isStop = stop.r === r && stop.c === c;
                        return (
                          <div
                            key={lane.segment}
                            ref={(el) => {
                              if (el) cellRefs.current.set(`${r}:${c}`, el);
                              else cellRefs.current.delete(`${r}:${c}`);
                            }}
                            role="gridcell"
                            tabIndex={isStop ? 0 : -1}
                            aria-selected={row ? row.transition_id === selected?.transition_id : undefined}
                            aria-label={row ? barName(row, lane) : `${lane.label}: no ${track.property_label.toLowerCase()} change measured for ${track.element_label}`}
                            onFocus={(e) => {
                              setFocusPos({ r, c });
                              setPreviewId(row && e.currentTarget.matches(":focus-visible") ? row.transition_id : null);
                            }}
                            onBlur={() => setPreviewId(null)}
                            className={`${LANE_COL} relative border-line focus-visible:outline-2 focus-visible:outline-solid focus-visible:-outline-offset-2 focus-visible:outline-[var(--focus)] ${
                              c < C - 1 || forwardOnly ? "border-r" : ""
                            } ${row && activeIds.has(row.transition_id) ? "bg-surface-2/60" : ""}`}
                          >
                            {row ? (
                              <div className="absolute inset-x-3 inset-y-0">
                                <Bar row={row} lane={lane} selected={row.transition_id === selected?.transition_id} onSelect={select} onHover={setHoverId} />
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                      {forwardOnly ? <div className={MISSING_COL} aria-hidden="true" /> : null}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <StatusBar playheadMs={t} moving={activeIds.size} />

          <div className="border-t border-line">
            <TimelineInspector
              row={inspected}
              transition={inspected ? (spec.transitions.find((x) => x.id === inspected.transition_id) ?? null) : null}
              labelSource={inspected ? (labelSources.get(inspected.element_id) ?? null) : null}
              source={inspected && inspected.transition_id !== selected?.transition_id ? "preview" : "selected"}
            />
          </div>
        </>
      )}
    </Panel>
  );
}

function positionOf(matrix: (TimelineRow | null)[][], transitionId: string): Pos | null {
  for (let r = 0; r < matrix.length; r++) {
    const c = matrix[r].findIndex((x) => x?.transition_id === transitionId);
    if (c >= 0) return { r, c };
  }
  return null;
}

function uncertainTrackKeys(model: TimelineModel): Set<string> {
  const all = new Map<string, boolean>();
  for (const lane of model.lanes) for (const r of lane.rows) all.set(r.key, (all.get(r.key) ?? true) && r.uncertain);
  return new Set([...all].filter(([, u]) => u).map(([k]) => k));
}

function barName(row: TimelineRow, lane: TimelineLane): string {
  return (
    `${row.element_label} ${row.property_label}, ${lane.label.toLowerCase()}: ${formatChange(row)}, ` +
    `starts at ${Math.round(row.start_ms)} ms, lasts ${formatMs(row.duration_ms)}, delay ${formatMs(row.delay_ms)}, ` +
    `easing ${formatEasing(row.easing)}, ${BAND_LABELS[row.confidence_band].toLowerCase()} confidence ${formatPercent(row.confidence)}` +
    (row.uncertain ? ", uncertain observation" : "")
  );
}

/* ---------- label column ---------- */

function TrackHeader({
  track,
  first,
  active,
  uncertain,
  labelSource,
}: {
  track: TimelineTrack;
  first: boolean;
  active: boolean;
  uncertain: boolean;
  labelSource: LabelSource | null;
}) {
  return (
    <div
      role="rowheader"
      className={`${LABEL_COL} sticky left-0 z-20 flex flex-col items-start justify-center border-r border-line pr-2 leading-4 transition-colors duration-100 sm:flex-row sm:items-center sm:gap-2 sm:pr-3 ${
        active ? "bg-surface-2" : "bg-surface"
      }`}
      style={{ paddingLeft: `${12 + track.depth * 10}px` }}
    >
      <span className="flex min-w-0 max-w-full flex-1 items-baseline gap-1.5">
        {first ? (
          <>
            <span className={`truncate text-xs font-medium sm:text-sm ${uncertain ? "text-ink-3" : "text-ink"}`} title={track.element_label}>
              {track.element_label}
            </span>
            {track.is_target ? <span className="hidden shrink-0 font-mono text-2xs text-signal sm:inline">target</span> : null}
            {labelSource === "interpreter" ? <LabelSourceTag source={labelSource} variant="inline" className="hidden sm:inline" /> : null}
          </>
        ) : (
          <span className="sr-only">{track.element_label}</span>
        )}
      </span>
      <span className={`shrink-0 font-mono text-2xs ${uncertain ? "text-ink-3 italic" : "text-ink-2"}`}>
        {track.property}
        {uncertain ? (
          <>
            <span aria-hidden="true">?</span>
            <span className="sr-only">, uncertain</span>
          </>
        ) : null}
      </span>
    </div>
  );
}

/* ---------- lane header + time axis ---------- */

function laneTrigger(lane: TimelineLane, spec: MotionSpec): string {
  return lane.kind === "forward" ? TRIGGER_LABELS[spec.interaction.trigger.kind] : REVERSE_TRIGGER_LABELS[spec.interaction.trigger.reverse_kind];
}

function LaneHeader({ lane, spec }: { lane: TimelineLane; spec: MotionSpec }) {
  const isFwd = lane.kind === "forward";
  return (
    <div className={`${HEAD} flex min-w-0 flex-col justify-center border-b border-line px-3`}>
      <p className="flex min-w-0 items-center gap-2 text-sm font-medium text-ink">
        <span className={`inline-block h-3 w-1.5 shrink-0 rounded-[2px] ${isFwd ? "bg-signal-fill" : "bg-rev-fill"}`} aria-hidden="true" />
        {lane.label}
        <span className="truncate font-normal text-ink-3">· {laneTrigger(lane, spec)}</span>
      </p>
      <p className="nums text-2xs text-ink-3">
        onset {Math.round(lane.onset_ms)} · settles {formatMs(lane.settle_ms - lane.onset_ms, { approx: true })} later
      </p>
    </div>
  );
}

/** Per-lane time axis. Pointer: click or drag to seek. Keyboard: a slider (←/→ one frame). */
function AxisSlider({
  lane,
  spec,
  playheadMs,
  frameMs,
  onSeek,
}: {
  lane: TimelineLane;
  spec: MotionSpec;
  playheadMs: number;
  frameMs: number;
  onSeek: (ms: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const w = lane.window;
  const ticks = axisTicks(w, 5);
  const inside = playheadMs >= w.start_ms && playheadMs <= w.end_ms;
  const value = clamp(playheadMs, w.start_ms, w.end_ms);
  const triggerF = lane.trigger_event_ms != null ? timeToFraction(lane.trigger_event_ms, w) : null;
  const trigger = laneTrigger(lane, spec);

  const seekTo = (ms: number) => onSeek(clamp(ms, w.start_ms, w.end_ms));
  const fromPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (rect && rect.width > 0) seekTo(fractionToTime((clientX - rect.left) / rect.width, w));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? AXIS_COARSE_MS : frameMs;
    const next: Record<string, number> = {
      ArrowRight: value + step,
      ArrowUp: value + step,
      ArrowLeft: value - step,
      ArrowDown: value - step,
      PageUp: value + AXIS_COARSE_MS,
      PageDown: value - AXIS_COARSE_MS,
      Home: w.start_ms,
      End: w.end_ms,
    };
    if (!(e.key in next) || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    seekTo(next[e.key]);
  };

  const where = inside ? "" : playheadMs < w.start_ms ? ", playhead is before this lane" : ", playhead is after this lane";

  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={`${lane.label} time axis`}
      aria-describedby="timeline-axis-help"
      aria-orientation="horizontal"
      aria-valuemin={Math.round(w.start_ms)}
      aria-valuemax={Math.round(w.end_ms)}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${Math.round(playheadMs)} ms${where}`}
      onKeyDown={onKeyDown}
      onPointerDown={(e: PointerEvent<HTMLDivElement>) => {
        if (e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        fromPointer(e.clientX);
      }}
      onPointerMove={(e: PointerEvent<HTMLDivElement>) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) fromPointer(e.clientX);
      }}
      className={`${AXIS} relative cursor-col-resize touch-none border-b border-line select-none transition-colors duration-100 hover:bg-surface-2/60 focus-visible:outline-2 focus-visible:outline-solid focus-visible:-outline-offset-2 focus-visible:outline-[var(--focus)]`}
    >
      <div ref={trackRef} className="absolute inset-x-3 inset-y-0" aria-hidden="true">
        {ticks.map((tk) => {
          const f = timeToFraction(tk, w);
          const align = f < 0.06 ? "translate-x-0" : f > 0.94 ? "-translate-x-full" : "-translate-x-1/2";
          return (
            <span key={tk} className="absolute top-0 h-full" style={{ left: pct(f) }}>
              <span className="absolute bottom-0 h-1.5 w-px bg-line-strong" />
              <span className={`nums absolute top-1 text-2xs text-ink-3 ${align}`}>{tk}</span>
            </span>
          );
        })}
        {triggerF != null ? (
          <span
            className="absolute bottom-0 size-2 translate-x-[-4px] translate-y-1/2 rotate-45 border border-ink bg-surface"
            style={{ left: pct(triggerF) }}
            title={`${trigger} at ${lane.trigger_event_ms} ms`}
          />
        ) : null}
        {inside ? <span className="absolute bottom-0 h-2 w-[3px] -translate-x-px rounded-t-[1px] bg-ink" style={{ left: pct(timeToFraction(playheadMs, w)) }} /> : null}
      </div>
    </div>
  );
}

/* ---------- lane body overlay (motion window, gridlines, markers, playhead) ---------- */

function LaneOverlay({ lane, model, playheadMs, className }: { lane: TimelineLane; model: TimelineModel; playheadMs: number; className: string }) {
  const w = lane.window;
  const ticks = axisTicks(w, 5);
  const playIn = playheadMs >= w.start_ms && playheadMs <= w.end_ms;
  const onsetF = timeToFraction(lane.onset_ms, w);
  const settleF = timeToFraction(lane.settle_ms, w);
  const triggerF = lane.trigger_event_ms != null ? timeToFraction(lane.trigger_event_ms, w) : null;
  const cursorMarkers = model.markers.filter(
    (m) => m.t_ms >= w.start_ms && m.t_ms <= w.end_ms && (lane.trigger_event_ms == null || Math.abs(m.t_ms - lane.trigger_event_ms) > 10),
  );
  return (
    <div className={`relative ${className}`} aria-hidden="true">
      <div className="absolute inset-x-3 inset-y-0">
        <span className="absolute inset-y-0 bg-surface-2/70" style={{ left: pct(onsetF), width: pct(Math.max(0, settleF - onsetF)) }} />
        {ticks.map((tk) => (
          <span key={tk} className="absolute inset-y-0 w-px bg-line/70" style={{ left: pct(timeToFraction(tk, w)) }} />
        ))}
        {triggerF != null ? <span className="absolute inset-y-0 border-l border-dashed border-ink-3" style={{ left: pct(triggerF) }} /> : null}
        {cursorMarkers.map((m) => (
          <span key={m.id} className="absolute inset-y-0 border-l border-dotted border-line-strong" style={{ left: pct(timeToFraction(m.t_ms, w)) }} title={m.label} />
        ))}
        {playIn ? <span className="absolute inset-y-0 z-10 w-px bg-ink" style={{ left: pct(timeToFraction(playheadMs, w)) }} /> : null}
      </div>
    </div>
  );
}

/* ---------- bar (visual only: the gridcell around it is the focusable, labelled element) ---------- */

function Bar({
  row,
  lane,
  selected,
  onSelect,
  onHover,
}: {
  row: TimelineRow;
  lane: TimelineLane;
  selected: boolean;
  onSelect: (row: TimelineRow) => void;
  onHover: (id: string | null) => void;
}) {
  const { left, width } = rowSpan(row, lane.window);
  const labelOutsideLeft = left + width > 0.8;
  const label = `${formatMs(row.duration_ms)}${row.uncertain ? " ?" : ""}`;
  const spark = row.samples
    .map(([ts, p]) => [(ts - row.start_ms) / Math.max(1, row.duration_ms), p] as const)
    .filter(([x]) => x >= 0 && x <= 1)
    .map(([x, p]) => `${(x * 100).toFixed(2)},${((1 - Math.min(1.1, Math.max(-0.1, p))) * 80 + 10).toFixed(2)}`)
    .join(" ");

  return (
    <>
      <div
        className="tl-bar absolute top-2 bottom-2 cursor-pointer overflow-hidden transition-[filter] duration-100 hover:brightness-110"
        data-band={row.confidence_band}
        data-uncertain={row.uncertain}
        data-lane={lane.kind}
        data-selected={selected}
        style={{ left: pct(left), width: `max(6px, ${pct(width)})` }}
        onClick={() => onSelect(row)}
        onMouseEnter={() => onHover(row.transition_id)}
        onMouseLeave={() => onHover(null)}
        aria-hidden="true"
      >
        {spark ? (
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full opacity-70">
            <polyline points={spark} fill="none" stroke="currentColor" strokeWidth={1.25} vectorEffect="non-scaling-stroke" />
          </svg>
        ) : null}
      </div>
      <span
        className={`nums pointer-events-none absolute top-1/2 -translate-y-1/2 text-2xs whitespace-nowrap ${row.uncertain ? "text-ink-3" : "text-ink-2"} ${
          labelOutsideLeft ? "-translate-x-full pr-1.5" : "pl-1.5"
        }`}
        style={{ left: labelOutsideLeft ? pct(left) : `calc(${pct(left)} + max(6px, ${pct(width)}))` }}
        aria-hidden="true"
      >
        {label}
      </span>
    </>
  );
}

/* ---------- status bar ---------- */

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-[2px] border border-line px-1 font-mono text-2xs text-ink-2">{children}</kbd>;
}

function StatusBar({ playheadMs, moving }: { playheadMs: number; moving: number }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-line px-3 py-1.5 font-mono text-2xs text-ink-3">
      <span className="nums" aria-hidden="true">
        playhead <span className="text-ink-2">{Math.round(playheadMs)} ms</span>
        {moving > 0 ? ` · ${moving} moving` : ""}
      </span>
      <span className="hidden items-center gap-1.5 md:inline-flex" aria-hidden="true">
        bars <Kbd>←↑↓→</Kbd> move <Kbd>Enter</Kbd> select <Kbd>PgUp/PgDn</Kbd> element
        <span className="px-1 text-line-strong">|</span>
        axis <Kbd>←→</Kbd> 1 frame <Kbd>Shift</Kbd> {AXIS_COARSE_MS} ms
      </span>
    </div>
  );
}

/* ---------- states ---------- */

function MissingReverseHeader() {
  return (
    <div className={MISSING_COL}>
      <div className={`${HEAD} flex flex-col justify-center border-b border-line px-3`} role="note" aria-label="Reverse motion not recorded">
        <p className="flex items-center gap-2 text-sm font-medium text-ink-3">
          <span className="inline-block h-3 w-1.5 rounded-[2px] border border-dashed border-ink-3" aria-hidden="true" />
          Reverse
        </p>
        <p className="font-mono text-2xs text-ink-3">not recorded</p>
      </div>
      <div className={`${AXIS} hatch border-b border-line`} aria-hidden="true" />
    </div>
  );
}

function MissingReverseBody() {
  return (
    // Padding lives on the inner box: on the flex item it would change the column's base size and
    // break alignment with the row spacers.
    <div className={`${MISSING_COL} hatch flex items-center`}>
      <div className="px-4 py-6">
        <p className="max-w-[16rem] rounded-sm bg-surface/90 px-2 py-1 text-xs text-ink-2" role="note">
          Only the forward motion is in the recording. The prompt assumes the reverse mirrors it. Record the UI returning to its initial state to
          measure it.
        </p>
      </div>
    </div>
  );
}

function EmptyTimeline({ hiddenUncertain, onShow }: { hiddenUncertain: number; onShow: () => void }) {
  return (
    <div className="flex flex-col items-start gap-3 px-4 py-10">
      <p className="text-md font-medium text-ink">No transitions above the confidence threshold</p>
      <p className="max-w-prose text-sm text-ink-2">
        Motion was detected, but no property could be measured reliably. Recording at 60 fps with a pause before and after the interaction usually
        fixes this.
      </p>
      {hiddenUncertain > 0 ? (
        <button type="button" onClick={onShow} className="focus-ring rounded-sm text-sm text-ink underline underline-offset-4">
          Show {hiddenUncertain} uncertain observation{hiddenUncertain === 1 ? "" : "s"}
        </button>
      ) : null}
    </div>
  );
}

function Legend() {
  const items: { band: string; uncertain?: boolean; label: string }[] = [
    { band: "high", label: "High" },
    { band: "medium", label: "Medium" },
    { band: "low", label: "Low" },
    { band: "low", uncertain: true, label: "Uncertain" },
  ];
  return (
    <ul className="flex flex-wrap items-center gap-x-3 gap-y-1" aria-label="Bar fill shows confidence">
      <li className="caption normal-case tracking-normal">confidence</li>
      {items.map((i) => (
        <li key={i.label} className="inline-flex items-center gap-1.5 text-xs text-ink-2">
          <span className="tl-bar tl-swatch" data-band={i.band} data-uncertain={i.uncertain ? "true" : undefined} aria-hidden="true" />
          {i.label}
        </li>
      ))}
    </ul>
  );
}
