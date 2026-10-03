/**
 * IR → timeline model for `MotionTimeline` (PLAN §10.2). Pure and deterministic.
 *
 * Shape:
 *   window   absolute ms range covering every lane + cursor marker (for a shared axis / playhead)
 *   tracks   one entry per (element × property), in a stable order shared by all lanes, so the
 *            forward and reverse lanes can be drawn as aligned grids (a lane may lack a track)
 *   lanes    one per segment (fwd, rev | rt_in, rt_out), each with its own padded window, rows
 *            (one per transition, sorted by track order) and segment markers (trigger/onset/settle)
 *   markers  cursor events (enter/leave/stationary/move_start), absolute ms
 *
 * All times are absolute video ms (same clock as `<video>.currentTime * 1000`), so seeking a row is
 * `video.currentTime = row.start_ms / 1000`.
 */

import { PROPERTY_LABELS, SEGMENT_LABELS } from "./format";
import type {
  ConfidenceBand,
  CursorEventKind,
  Easing,
  MotionElement,
  MotionSpec,
  Property,
  Role,
  SegmentId,
  SegmentKind,
  Transition,
  Value,
} from "./types";
import { UNCERTAIN_BELOW } from "./types";

export interface TimeWindow {
  start_ms: number;
  end_ms: number;
}

/** One (element × property) line; identical across lanes. */
export interface TimelineTrack {
  /** `${element_id}:${property}` */
  key: string;
  element_id: string;
  element_label: string;
  element_role: Role | null;
  /** 0 for root elements, 1 for children, ... (for indentation). */
  depth: number;
  parent_id: string | null;
  /** True for the interaction target element. */
  is_target: boolean;
  property: Property;
  property_label: string;
}

/** One transition bar. */
export interface TimelineRow extends TimelineTrack {
  transition_id: string;
  segment_id: SegmentId;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  /** Relative to the segment onset (what CSS `transition-delay` would be). */
  delay_ms: number;
  confidence: number;
  confidence_band: ConfidenceBand;
  /** overall < UNCERTAIN_BELOW (0.3): render distinctly, and not by color alone. */
  uncertain: boolean;
  easing: Easing;
  from: Value;
  to: Value;
  delta: number | null;
  transform_origin: string | null;
  notes: string[];
  /** [t_ms, progress 0..1] pairs (absolute ms), for an optional progress sparkline. */
  samples: [number, number][];
}

export type TimelineMarkerKind =
  | "cursor_enter"
  | "cursor_leave"
  | "cursor_stationary"
  | "cursor_move_start"
  | "trigger"
  | "onset"
  | "settle";

export interface TimelineMarker {
  /** Stable React key. */
  id: string;
  kind: TimelineMarkerKind;
  t_ms: number;
  segment_id: SegmentId | null;
  element_id: string | null;
  label: string;
}

export interface TimelineLane {
  segment: SegmentId;
  kind: SegmentKind;
  label: string;
  /** Padded range of this lane's rows + markers. */
  window: TimeWindow;
  onset_ms: number;
  settle_ms: number;
  trigger_event_ms: number | null;
  rows: TimelineRow[];
  /** trigger / onset / settle for this segment. */
  markers: TimelineMarker[];
}

export interface TimelineModel {
  window: TimeWindow;
  tracks: TimelineTrack[];
  lanes: TimelineLane[];
  /** Cursor events (absolute ms). */
  markers: TimelineMarker[];
  /** Rows across all lanes (after the uncertain filter). */
  rowCount: number;
  /** Rows flagged `uncertain` (counted even when hidden). */
  uncertainCount: number;
  /** No rows to draw → render the empty state. */
  isEmpty: boolean;
}

export interface BuildTimelineOptions {
  /** Keep transitions with overall < 0.3 (flagged `uncertain`). Default true. */
  includeUncertain?: boolean;
  /** Fixed lane padding in ms; default max(40, 10% of the lane span), rounded to 10 ms. */
  paddingMs?: number;
}

/** Row order within an element (geometry first, then paint). */
const PROPERTY_ORDER: readonly Property[] = [
  "translateX",
  "translateY",
  "scale",
  "scaleX",
  "scaleY",
  "height",
  "opacity",
  "background-color",
  "color",
  "box-shadow",
  "border-radius",
  "content",
];

const CURSOR_MARKER: Record<CursorEventKind, { kind: TimelineMarkerKind; label: string }> = {
  enter: { kind: "cursor_enter", label: "Cursor enters" },
  leave: { kind: "cursor_leave", label: "Cursor leaves" },
  stationary: { kind: "cursor_stationary", label: "Cursor stops" },
  move_start: { kind: "cursor_move_start", label: "Cursor starts moving" },
};

/** Elements in hierarchy order (DFS, parents before children, siblings in spec order) + depth. */
function orderElements(elements: MotionElement[]): Map<string, { index: number; depth: number; el: MotionElement }> {
  const ids = new Set(elements.map((e) => e.id));
  const children = new Map<string | null, MotionElement[]>();
  for (const el of elements) {
    const parent = el.parent_id && ids.has(el.parent_id) && el.parent_id !== el.id ? el.parent_id : null;
    const list = children.get(parent) ?? [];
    list.push(el);
    children.set(parent, list);
  }
  const out = new Map<string, { index: number; depth: number; el: MotionElement }>();
  const visit = (parent: string | null, depth: number) => {
    for (const el of children.get(parent) ?? []) {
      if (out.has(el.id)) continue; // cycle guard
      out.set(el.id, { index: out.size, depth, el });
      visit(el.id, depth + 1);
    }
  };
  visit(null, 0);
  // Anything unreachable (malformed cycles) goes last at depth 0.
  for (const el of elements) if (!out.has(el.id)) out.set(el.id, { index: out.size, depth: 0, el });
  return out;
}

function propertyRank(p: Property): number {
  const i = PROPERTY_ORDER.indexOf(p);
  return i === -1 ? PROPERTY_ORDER.length : i;
}

function defaultPadding(span: number): number {
  return Math.max(40, Math.round((span * 0.1) / 10) * 10);
}

function padWindow(min: number, max: number, paddingMs: number | undefined, clampEnd: number | null): TimeWindow {
  const pad = paddingMs ?? defaultPadding(max - min);
  const start = Math.max(0, min - pad);
  const end = clampEnd != null && clampEnd > 0 ? Math.min(clampEnd, max + pad) : max + pad;
  return { start_ms: start, end_ms: Math.max(end, start + 1) };
}

export function buildTimeline(spec: MotionSpec, options: BuildTimelineOptions = {}): TimelineModel {
  const { includeUncertain = true, paddingMs } = options;
  const order = orderElements(spec.elements);
  const targetId = spec.interaction.target_element_id;
  const durationEnd = spec.source.duration_ms > 0 ? spec.source.duration_ms : null;

  const trackFor = (t: Transition): TimelineTrack => {
    const info = order.get(t.element_id);
    return {
      key: `${t.element_id}:${t.property}`,
      element_id: t.element_id,
      element_label: info?.el.label ?? t.element_id,
      element_role: info?.el.role ?? null,
      depth: info?.depth ?? 0,
      parent_id: info?.el.parent_id ?? null,
      is_target: t.element_id === targetId,
      property: t.property,
      property_label: PROPERTY_LABELS[t.property] ?? t.property,
    };
  };

  const compareTracks = (a: TimelineTrack, b: TimelineTrack) =>
    (order.get(a.element_id)?.index ?? Infinity) - (order.get(b.element_id)?.index ?? Infinity) ||
    propertyRank(a.property) - propertyRank(b.property);

  // Tracks: union over the included transitions (hidden uncertain ones get no track).
  const trackMap = new Map<string, TimelineTrack>();
  let uncertainCount = 0;
  const rowsBySegment = new Map<SegmentId, TimelineRow[]>();

  for (const t of spec.transitions) {
    const uncertain = t.confidence.overall < UNCERTAIN_BELOW;
    if (uncertain) uncertainCount += 1;
    if (uncertain && !includeUncertain) continue;
    const track = trackFor(t);
    if (!trackMap.has(track.key)) trackMap.set(track.key, track);
    const row: TimelineRow = {
      ...track,
      transition_id: t.id,
      segment_id: t.segment_id,
      start_ms: t.start_ms,
      end_ms: t.start_ms + t.duration_ms,
      duration_ms: t.duration_ms,
      delay_ms: t.delay_ms,
      confidence: t.confidence.overall,
      confidence_band: t.confidence.band,
      uncertain,
      easing: t.easing,
      from: t.from,
      to: t.to,
      delta: t.delta,
      transform_origin: t.transform_origin,
      notes: t.notes,
      samples: t.samples,
    };
    const list = rowsBySegment.get(t.segment_id) ?? [];
    list.push(row);
    rowsBySegment.set(t.segment_id, list);
  }

  const tracks = [...trackMap.values()].sort(compareTracks);

  const lanes: TimelineLane[] = spec.segments.map((seg) => {
    const rows = (rowsBySegment.get(seg.id) ?? []).sort(
      (a, b) => compareTracks(a, b) || a.start_ms - b.start_ms || a.transition_id.localeCompare(b.transition_id),
    );
    const markers: TimelineMarker[] = [];
    if (seg.trigger_event_ms != null) {
      markers.push({ id: `${seg.id}:trigger`, kind: "trigger", t_ms: seg.trigger_event_ms, segment_id: seg.id, element_id: targetId, label: "Trigger" });
    }
    markers.push(
      { id: `${seg.id}:onset`, kind: "onset", t_ms: seg.onset_ms, segment_id: seg.id, element_id: null, label: "Motion starts" },
      { id: `${seg.id}:settle`, kind: "settle", t_ms: seg.settle_ms, segment_id: seg.id, element_id: null, label: "Motion settles" },
    );
    const times = [...rows.flatMap((r) => [r.start_ms, r.end_ms]), ...markers.map((m) => m.t_ms)];
    const window = padWindow(Math.min(...times), Math.max(...times), paddingMs, durationEnd);
    return {
      segment: seg.id,
      kind: seg.kind,
      label: SEGMENT_LABELS[seg.id] ?? seg.id,
      window,
      onset_ms: seg.onset_ms,
      settle_ms: seg.settle_ms,
      trigger_event_ms: seg.trigger_event_ms,
      rows,
      markers,
    };
  });

  const markers: TimelineMarker[] = spec.cursor.events
    .map((e, i) => ({
      id: `cursor:${i}`,
      kind: CURSOR_MARKER[e.kind].kind,
      t_ms: e.t_ms,
      segment_id: null,
      element_id: e.element_id,
      label: CURSOR_MARKER[e.kind].label,
    }))
    .sort((a, b) => a.t_ms - b.t_ms);

  const bounds = [...lanes.flatMap((l) => [l.window.start_ms, l.window.end_ms]), ...markers.map((m) => m.t_ms)];
  const window: TimeWindow =
    bounds.length > 0
      ? padWindow(Math.min(...bounds), Math.max(...bounds), lanes.length > 0 ? 0 : paddingMs, durationEnd)
      : { start_ms: 0, end_ms: Math.max(1, spec.source.duration_ms) };

  const rowCount = lanes.reduce((n, l) => n + l.rows.length, 0);
  return { window, tracks, lanes, markers, rowCount, uncertainCount, isEmpty: rowCount === 0 };
}

/* ---------- geometry helpers for rendering ---------- */

/** Position of `t_ms` inside `window` as 0..1 (clamped). */
export function timeToFraction(t_ms: number, window: TimeWindow): number {
  const span = window.end_ms - window.start_ms;
  if (span <= 0) return 0;
  return Math.min(1, Math.max(0, (t_ms - window.start_ms) / span));
}

/** Inverse of `timeToFraction` (for click/drag-to-seek on the axis). */
export function fractionToTime(fraction: number, window: TimeWindow): number {
  const f = Math.min(1, Math.max(0, fraction));
  return window.start_ms + f * (window.end_ms - window.start_ms);
}

/** Bar geometry as fractions of the window: `left`/`width` in 0..1 (multiply by 100 for %). */
export function rowSpan(row: Pick<TimelineRow, "start_ms" | "end_ms">, window: TimeWindow): { left: number; width: number } {
  const left = timeToFraction(row.start_ms, window);
  return { left, width: Math.max(0, timeToFraction(row.end_ms, window) - left) };
}

/**
 * "Nice" axis ticks (1/2/5 × 10^n ms) inside `window`, roughly `approxCount` of them.
 * E.g. window 1100–1600 → [1100, 1200, 1300, 1400, 1500, 1600].
 */
export function axisTicks(window: TimeWindow, approxCount = 6): number[] {
  const span = window.end_ms - window.start_ms;
  if (span <= 0 || approxCount < 1) return [];
  const raw = span / approxCount;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks: number[] = [];
  for (let t = Math.ceil(window.start_ms / step) * step; t <= window.end_ms + 1e-9; t += step) {
    ticks.push(Number(t.toFixed(6)));
  }
  return ticks;
}

/** Rows whose bar contains `t_ms` (playhead highlight). */
export function activeRowsAt(model: TimelineModel, t_ms: number): TimelineRow[] {
  return model.lanes.flatMap((l) => l.rows.filter((r) => t_ms >= r.start_ms && t_ms <= r.end_ms));
}

/** Lane whose window contains `t_ms`, or null (playhead outside any analysed segment). */
export function laneAt(model: TimelineModel, t_ms: number): TimelineLane | null {
  return model.lanes.find((l) => t_ms >= l.window.start_ms && t_ms <= l.window.end_ms) ?? null;
}

/** Look up a row by transition id (e.g. from a URL hash or a selection). */
export function findRow(model: TimelineModel, transitionId: string): TimelineRow | null {
  for (const lane of model.lanes) {
    const row = lane.rows.find((r) => r.transition_id === transitionId);
    if (row) return row;
  }
  return null;
}
