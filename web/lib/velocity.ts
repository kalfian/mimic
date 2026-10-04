/**
 * Continuous-mode IR → velocity chart model + behaviour summary (PLAN-continuous §7.1). Pure and
 * deterministic, no React.
 *
 * Shape:
 *   window   analysed span (absolute video ms, same clock as `<video>.currentTime * 1000`)
 *   points   `continuous.samples` as objects, `degraded` when tracking quality < 0.5
 *   runs     points split into polylines: a new run starts where `degraded` flips (the boundary
 *            point is shared so the line stays connected) or at a time gap (> 3× the median step)
 *   bands    one per phase (chronological), with label, headline, hover/focus facts, flags
 *   markers  cursor enter/leave + derived release / rest / resume_start instants
 *   yDomain  symmetric velocity range, padded to a "nice" number (px/s)
 *
 * Sign convention (`SIGN_CONVENTION`): positive = content moves right (x) / down (y), so the
 * chart's upper half is "right"/"down". Two y scales: `linear`, and `compressed` (asinh around
 * `compressedPivot`) for recordings where drags are > 10× the autoplay speed.
 */

import {
  AUTOPLAY_DIRECTION_LABELS,
  INERTIA_MODEL_LABELS,
  PAUSE_TRIGGER_LABELS,
  PHASE_EVIDENCE_LABELS,
  PHASE_KIND_LABELS,
  SNAP_KIND_LABELS,
  axisDirection,
  formatApproxMs,
  formatCardScale,
  formatDistance,
  formatEasing,
  formatEasingName,
  formatPx,
  formatScale,
  formatSpeed,
  formatSpeedChange,
  formatTau,
  formatVelocity,
  roundSpeed,
} from "./format";
import { continuousOf } from "./spec";
import type { TimeWindow } from "./timeline";
import type {
  AutoplayDirection,
  Axis,
  Confidence,
  ConfidenceBand,
  ContinuousMotion,
  Cursor,
  ExponentialFit,
  MeasuredNumber,
  MotionSpec,
  Phase,
  PhaseEvidence,
  PhaseKind,
} from "./types";
import { SIGN_CONVENTION, UNCERTAIN_BELOW } from "./types";

export { axisTicks, fractionToTime, timeToFraction } from "./timeline";
export type { TimeWindow } from "./timeline";

/* ---------- thresholds (PLAN-continuous §4.3 / §7.1) ---------- */

/** A sample with tracking quality below this is "degraded" (low-quality match). */
export const DEGRADED_QUALITY_BELOW = 0.5;
/** A phase is `degraded` when more than this fraction of its samples is degraded. */
export const DEGRADED_PHASE_FRACTION = 0.1;
/** Offer the compressed scale when max |v| / |autoplay v| exceeds this. */
export const COMPRESSED_SCALE_RATIO = 10;
/** A time step larger than this × the median step breaks the line (missing samples). */
const GAP_FACTOR = 3;

/* ---------- model types ---------- */

export interface VelocityPoint {
  t_ms: number;
  /** Signed velocity, px/s. */
  v: number;
  /** Signed integrated position, px (0 at the first sample). */
  pos: number;
  /** Tracking quality 0..1. */
  quality: number;
  /** quality < DEGRADED_QUALITY_BELOW: draw distinctly, and not by color alone (dashed / gap). */
  degraded: boolean;
}

/** A polyline to draw. Consecutive runs that differ only in `degraded` share their boundary point. */
export interface VelocityRun {
  degraded: boolean;
  points: VelocityPoint[];
}

/** A labelled value for the hover/focus details and the behaviour rows. */
export interface VelocityFact {
  label: string;
  value: string;
  /** Own confidence when the value is measured (`MeasuredNumber`), else null. */
  confidence: Confidence | null;
}

/** One phase as a background band on the chart. */
export interface VelocityBand {
  phase_id: string;
  kind: PhaseKind;
  /** `PHASE_KIND_LABELS[kind]`, e.g. "Momentum". */
  label: string;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  /** Label confidence (0..1) and its band. */
  confidence: number;
  band: ConfidenceBand;
  /** confidence < 0.3: hatched + labelled, not by color alone. */
  uncertain: boolean;
  /** Cut short (decelerate cut by a drag, inertia re-grabbed). */
  interrupted: boolean;
  /** More than 10 % of the samples inside are degraded. */
  degraded: boolean;
  degradedFraction: number;
  evidence: PhaseEvidence;
  /** Short key value, e.g. "τ ≈ 200 ms", "≈1500 px/s peak · left", "≈39 px/s right". */
  headline: string;
  v_start: number;
  v_end: number;
  v_peak: number;
  displacement_px: number;
  fit: Phase["fit"];
  /** Hover/focus details: duration, velocities, distance, fit values with their confidences. */
  facts: VelocityFact[];
  notes: string[];
}

export type VelocityMarkerKind = "cursor_enter" | "cursor_leave" | "release" | "rest" | "autoplay_join" | "resume_start";

export interface VelocityMarker {
  /** Stable React key. */
  id: string;
  kind: VelocityMarkerKind;
  t_ms: number;
  label: string;
  /** Phase that starts/ends at the marker (derived markers), null for cursor events. */
  phase_id: string | null;
}

export interface AutoplayReference {
  /** Signed autoplay velocity (draw a horizontal reference line here). */
  v: number;
  confidence: number;
  band: ConfidenceBand;
  /** e.g. "Autoplay ≈39 px/s right". */
  label: string;
}

export type VelocityScale = "linear" | "compressed";

export interface VelocityModel {
  window: TimeWindow;
  axis: Axis;
  unit: "px/s";
  /** Word for v > 0 / v < 0 (axis labels: "↑ right", "↓ left"). */
  positiveLabel: "right" | "down";
  negativeLabel: "left" | "up";
  signConvention: typeof SIGN_CONVENTION;
  points: VelocityPoint[];
  runs: VelocityRun[];
  /** false → no chart line; fall back to the phase list (bands still exist). */
  hasSamples: boolean;
  degradedCount: number;
  /** Symmetric, padded: { min: -M, max: M }. */
  yDomain: { min: number; max: number };
  /** Largest |v| over samples, phase peaks and autoplay. */
  maxAbsV: number;
  /** max|v| / |v_auto| > 10 → offer the compressed (asinh) scale; false without autoplay. */
  suggestCompressedScale: boolean;
  /** asinh pivot for the compressed scale (≈ linear below it, logarithmic above), px/s. */
  compressedPivot: number;
  autoplayRef: AutoplayReference | null;
  bands: VelocityBand[];
  markers: VelocityMarker[];
  /** Bands with confidence < 0.3. */
  uncertainCount: number;
  /** Bands flagged `degraded`. */
  degradedBandCount: number;
}

export interface BuildVelocityOptions {
  /** `spec.cursor`, for cursor enter/leave markers. */
  cursor?: Cursor | null;
}

/* ---------- small helpers ---------- */

function isUncertainValue(c: number): boolean {
  return c < UNCERTAIN_BELOW;
}

function measured(label: string, n: MeasuredNumber | null | undefined, text: (v: number) => string): VelocityFact[] {
  return n ? [{ label, value: text(n.value), confidence: n.confidence }] : [];
}

function plain(label: string, value: string): VelocityFact {
  return { label, value, confidence: null };
}

/** 1 / 2 / 2.5 / 5 × 10^n at or above `x` (x > 0). */
function niceCeil(x: number): number {
  const mag = 10 ** Math.floor(Math.log10(x));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= x - 1e-9);
  return Number((step ?? 10 * mag).toPrecision(12));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/* ---------- bands ---------- */

/**
 * Exponential momentum that levels off at a non-zero speed (the autoplay velocity, `v_inf`) instead
 * of coming to rest: the content glides and blends back into autoplay. Same rounding as the
 * generators (`roundSpeed`), so "≈0 px/s" counts as at rest.
 */
export function decaysToAutoplay(fit: Phase["fit"]): boolean {
  return fit?.model === "exponential" && roundSpeed(fit.v_inf_px_s) !== 0;
}

/** Band title: momentum fitted as a fixed-length eased tween is a "Glide", not an exponential coast. */
function bandLabel(p: Phase): string {
  if (p.kind === "inertia" && p.fit?.model === "tween") return "Glide";
  return PHASE_KIND_LABELS[p.kind] ?? p.kind;
}

function phaseHeadline(p: Phase, axis: Axis, duration: number): string {
  const fit = p.fit;
  switch (p.kind) {
    case "autoplay":
      return formatVelocity(fit?.model === "constant" ? fit.velocity_px_s.value : p.v_peak_px_s, axis);
    case "decelerate":
    case "resume":
      if (fit?.model === "ramp") return `${formatSpeedChange(fit.from_px_s, fit.to_px_s)} over ${formatApproxMs(fit.duration_ms.value)}`;
      return `${formatSpeedChange(p.v_start_px_s, p.v_end_px_s)} over ${formatApproxMs(duration)}`;
    case "drag":
      return `${formatSpeed(p.v_peak_px_s)} peak · ${axisDirection(p.v_peak_px_s, axis)}`;
    case "inertia":
      if (fit?.model === "exponential") return `${formatTau(fit.tau_ms.value)}${decaysToAutoplay(fit) ? " → autoplay" : ""}`;
      if (fit?.model === "tween") return `${formatApproxMs(fit.duration_ms.value)} · ${formatEasingName(fit.easing)}`;
      return `from ${formatSpeed(p.v_start_px_s)}`;
    case "snap":
      if (fit?.model === "tween") return `${formatPx(Math.abs(fit.distance_px.value))} in ${formatApproxMs(fit.duration_ms.value)}`;
      return formatApproxMs(duration);
    case "stop":
      return `${formatSpeedChange(p.v_start_px_s, p.v_end_px_s)} in ${formatApproxMs(duration)}`;
    case "paused":
    case "unknown":
      return formatApproxMs(duration);
  }
}

/** Where an exponential decay goes: rest (below `stop_px_s`) or back into autoplay (`v_inf`). */
function exponentialEnd(fit: ExponentialFit, interrupted: boolean, axis: Axis): string {
  if (decaysToAutoplay(fit)) {
    const auto = formatVelocity(fit.v_inf_px_s, axis);
    return interrupted ? `Heading into autoplay (${auto}), cut short` : `Blends into autoplay (${auto})`;
  }
  if (interrupted) return "Cut short before coming to rest";
  return fit.stop_px_s != null ? `At rest below ${formatSpeed(fit.stop_px_s)}` : "At rest";
}

function fitFacts(p: Phase, axis: Axis): VelocityFact[] {
  const fit = p.fit;
  if (!fit) return [];
  switch (fit.model) {
    case "constant":
      return measured("Speed", fit.velocity_px_s, (v) => formatVelocity(v, axis));
    case "exponential":
      return [
        ...(p.kind === "inertia" ? [plain("Model", INERTIA_MODEL_LABELS.exponential)] : []),
        ...measured("Time constant", fit.tau_ms, formatTau),
        ...measured("Release speed", fit.v0_px_s, (v) => formatVelocity(v, axis)),
        plain("Ends", exponentialEnd(fit, p.interrupted, axis)),
      ];
    case "ramp":
      return [
        plain("Ramp", formatSpeedChange(fit.from_px_s, fit.to_px_s)),
        ...measured("Ramp duration", fit.duration_ms, formatApproxMs),
        plain("Easing", formatEasing(fit.easing, { withNearest: true })),
      ];
    case "tween":
      return [
        ...(p.kind === "inertia" ? [plain("Model", INERTIA_MODEL_LABELS.tween)] : []),
        ...measured("Distance", fit.distance_px, (v) => formatDistance(v, axis)),
        ...measured("Duration", fit.duration_ms, formatApproxMs),
        plain("Easing", formatEasing(fit.easing, { withNearest: true })),
      ];
  }
}

/** "≈1400 px/s left → 0 px/s"; one value when start and end read the same ("≈40 px/s right, steady"). */
function velocitySpan(from: number, to: number, axis: Axis): string {
  const a = formatVelocity(from, axis);
  const b = formatVelocity(to, axis);
  return a === b ? `${a}, steady` : `${a} → ${b}`;
}

function buildBand(p: Phase, axis: Axis, points: VelocityPoint[]): VelocityBand {
  const duration = p.end_ms - p.start_ms;
  const inside = points.filter((pt) => pt.t_ms >= p.start_ms && pt.t_ms <= p.end_ms);
  const degradedFraction = inside.length ? inside.filter((pt) => pt.degraded).length / inside.length : 0;
  // A tween fit carries its own distance and duration (with confidence): don't list them twice.
  const tween = p.fit?.model === "tween";
  const facts: VelocityFact[] = [
    ...(tween ? [] : [plain("Duration", formatApproxMs(duration))]),
    plain("Velocity", velocitySpan(p.v_start_px_s, p.v_end_px_s, axis)),
  ];
  if (p.kind === "drag" || p.kind === "inertia" || p.kind === "stop") facts.push(plain("Peak", formatVelocity(p.v_peak_px_s, axis)));
  facts.push(...(tween ? [] : [plain("Distance", formatDistance(p.displacement_px, axis))]), ...fitFacts(p, axis), plain("Based on", PHASE_EVIDENCE_LABELS[p.evidence]));
  return {
    phase_id: p.id,
    kind: p.kind,
    label: bandLabel(p),
    start_ms: p.start_ms,
    end_ms: p.end_ms,
    duration_ms: duration,
    confidence: p.confidence.value,
    band: p.confidence.band,
    uncertain: isUncertainValue(p.confidence.value),
    interrupted: p.interrupted,
    degraded: degradedFraction > DEGRADED_PHASE_FRACTION,
    degradedFraction: Number(degradedFraction.toFixed(4)),
    evidence: p.evidence,
    headline: phaseHeadline(p, axis, duration),
    v_start: p.v_start_px_s,
    v_end: p.v_end_px_s,
    v_peak: p.v_peak_px_s,
    displacement_px: p.displacement_px,
    fit: p.fit,
    facts,
    notes: p.notes,
  };
}

/* ---------- markers ---------- */

const RELEASE_KINDS: ReadonlySet<PhaseKind> = new Set(["inertia", "snap", "stop"]);
const MOTION_ENDING_KINDS: ReadonlySet<PhaseKind> = new Set(["inertia", "snap", "stop", "decelerate"]);

function buildMarkers(phases: Phase[], cursor: Cursor | null | undefined): VelocityMarker[] {
  const markers: VelocityMarker[] = [];
  phases.forEach((p, i) => {
    const prev = phases[i - 1];
    const next = phases[i + 1];
    // Release: a drag hands over to momentum / snap / stop.
    if (prev?.kind === "drag" && RELEASE_KINDS.has(p.kind)) {
      markers.push({ id: `release:${p.id}`, kind: "release", t_ms: p.start_ms, label: "Released", phase_id: p.id });
    }
    // Rest: motion comes to rest (a paused phase, or a rest-reaching phase straight into resume).
    if (p.kind === "paused") {
      markers.push({ id: `rest:${p.id}`, kind: "rest", t_ms: p.start_ms, label: "Comes to rest", phase_id: p.id });
    } else if (MOTION_ENDING_KINDS.has(p.kind) && !p.interrupted && next?.kind === "resume" && !decaysToAutoplay(p.fit)) {
      markers.push({ id: `rest:${p.id}`, kind: "rest", t_ms: p.end_ms, label: "Comes to rest", phase_id: p.id });
    }
    // Momentum that levels off at the autoplay velocity: it never rests, it blends back into autoplay.
    if (p.kind === "inertia" && !p.interrupted && decaysToAutoplay(p.fit)) {
      markers.push({ id: `join:${p.id}`, kind: "autoplay_join", t_ms: p.end_ms, label: "Blends into autoplay", phase_id: p.id });
    }
    if (p.kind === "resume") {
      markers.push({ id: `resume:${p.id}`, kind: "resume_start", t_ms: p.start_ms, label: "Autoplay resumes", phase_id: p.id });
    }
  });
  (cursor?.events ?? []).forEach((e, i) => {
    if (e.kind === "enter") markers.push({ id: `cursor:${i}`, kind: "cursor_enter", t_ms: e.t_ms, label: "Cursor enters", phase_id: null });
    if (e.kind === "leave") markers.push({ id: `cursor:${i}`, kind: "cursor_leave", t_ms: e.t_ms, label: "Cursor leaves", phase_id: null });
  });
  return markers.sort((a, b) => a.t_ms - b.t_ms || a.id.localeCompare(b.id));
}

/* ---------- points / runs ---------- */

function toPoints(samples: ContinuousMotion["samples"]): VelocityPoint[] {
  return samples.map(([t_ms, v, pos, quality]) => ({ t_ms, v, pos, quality, degraded: quality < DEGRADED_QUALITY_BELOW }));
}

function buildRuns(points: VelocityPoint[]): VelocityRun[] {
  if (points.length === 0) return [];
  const steps = points.slice(1).map((p, i) => p.t_ms - points[i].t_ms);
  const gapAbove = GAP_FACTOR * median(steps);
  const runs: VelocityRun[] = [{ degraded: points[0].degraded, points: [points[0]] }];
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    const run = runs[runs.length - 1];
    const gap = steps.length > 0 && p.t_ms - points[i - 1].t_ms > gapAbove;
    if (gap) runs.push({ degraded: p.degraded, points: [p] });
    else if (p.degraded !== run.degraded) runs.push({ degraded: p.degraded, points: [points[i - 1], p] });
    else run.points.push(p);
  }
  // A one-point run whose point also starts the next run (gap, then an immediate flip) is redundant.
  // Isolated one-point runs stay (draw them as a dot).
  return runs.filter((r, i) => r.points.length > 1 || runs[i + 1]?.points[0] !== r.points[0]);
}

/* ---------- model ---------- */

/** The chart model for a `continuous` section. Pass `{ cursor: spec.cursor }` for cursor markers. */
export function buildVelocityModel(c: ContinuousMotion, options: BuildVelocityOptions = {}): VelocityModel {
  const axis = c.axis;
  const points = toPoints(c.samples);
  const bands = c.phases.map((p) => buildBand(p, axis, points));

  const start = c.span_ms.start_ms;
  const end = c.span_ms.end_ms > start ? c.span_ms.end_ms : Math.max(start + 1, ...c.phases.map((p) => p.end_ms), ...points.map((p) => p.t_ms));

  const autoV = c.autoplay?.velocity_px_s ?? null;
  const maxAbsV = Math.max(0, ...points.map((p) => Math.abs(p.v)), ...c.phases.map((p) => Math.abs(p.v_peak_px_s)), Math.abs(autoV ?? 0));
  const M = niceCeil(Math.max(10, maxAbsV * 1.1));
  const autoAbs = autoV != null ? Math.abs(autoV) : 0;

  return {
    window: { start_ms: start, end_ms: end },
    axis,
    unit: "px/s",
    positiveLabel: axis === "x" ? "right" : "down",
    negativeLabel: axis === "x" ? "left" : "up",
    signConvention: SIGN_CONVENTION,
    points,
    runs: buildRuns(points),
    hasSamples: points.length > 0,
    degradedCount: points.filter((p) => p.degraded).length,
    yDomain: { min: -M, max: M },
    maxAbsV,
    suggestCompressedScale: autoAbs > 0 && maxAbsV / autoAbs > COMPRESSED_SCALE_RATIO,
    compressedPivot: Math.max(5, autoAbs > 0 ? autoAbs : maxAbsV / 50),
    autoplayRef:
      c.autoplay && autoV != null
        ? {
            v: autoV,
            confidence: c.autoplay.speed_px_s.confidence.value,
            band: c.autoplay.speed_px_s.confidence.band,
            label: `Autoplay ${formatVelocity(autoV, axis)}`,
          }
        : null,
    bands,
    markers: buildMarkers(c.phases, options.cursor),
    uncertainCount: bands.filter((b) => b.uncertain).length,
    degradedBandCount: bands.filter((b) => b.degraded).length,
  };
}

/** `buildVelocityModel` for a whole spec; null in transition mode (incl. stored 0.1 results). */
export function velocityModelOf(spec: MotionSpec): VelocityModel | null {
  const c = continuousOf(spec);
  return c ? buildVelocityModel(c, { cursor: spec.cursor }) : null;
}

/* ---------- geometry / lookup helpers ---------- */

/**
 * Vertical position of a velocity as 0..1 (0 = `yDomain.min` at the bottom, 1 = `yDomain.max`,
 * 0.5 = zero). `compressed` uses asinh(v / pivot), so slow autoplay stays readable next to fast
 * drags. Values outside the domain are clamped.
 */
export function velocityToFraction(v: number, model: Pick<VelocityModel, "yDomain" | "compressedPivot">, scale: VelocityScale = "linear"): number {
  const { min, max } = model.yDomain;
  if (max <= min) return 0.5;
  const f = scale === "compressed" ? (x: number) => Math.asinh(x / model.compressedPivot) : (x: number) => x;
  const lo = f(min);
  const hi = f(max);
  return Math.min(1, Math.max(0, (f(v) - lo) / (hi - lo)));
}

/**
 * Y-axis ticks (px/s, signed, ascending). Linear: about `approxPerSide` nice steps each side of 0.
 * Compressed: 0, ±the pivot-ish decade(s) and ±powers of ten up to the domain edge.
 */
export function velocityTicks(model: Pick<VelocityModel, "yDomain" | "compressedPivot">, scale: VelocityScale = "linear", approxPerSide = 2): number[] {
  const M = model.yDomain.max;
  if (M <= 0) return [0];
  let positive: number[];
  if (scale === "linear") {
    const step = niceCeil(M / Math.max(1, approxPerSide));
    positive = [];
    for (let t = step; t <= M + 1e-9; t += step) positive.push(Number(t.toPrecision(12)));
  } else {
    positive = [];
    const firstDecade = 10 ** Math.max(1, Math.round(Math.log10(model.compressedPivot)));
    for (let t = firstDecade; t <= M + 1e-9; t *= 10) positive.push(t);
    if (positive.length === 0) positive.push(M);
  }
  return [...positive.map((t) => -t).reverse(), 0, ...positive];
}

/** Band (phase) containing `t_ms`; at a shared boundary the later phase wins. Null outside all phases. */
export function bandAt(model: Pick<VelocityModel, "bands">, t_ms: number): VelocityBand | null {
  for (let i = model.bands.length - 1; i >= 0; i--) {
    const b = model.bands[i];
    if (t_ms >= b.start_ms && t_ms <= b.end_ms) return b;
  }
  return null;
}

export function findBand(model: Pick<VelocityModel, "bands">, phaseId: string): VelocityBand | null {
  return model.bands.find((b) => b.phase_id === phaseId) ?? null;
}

/**
 * Phase label for a keyframe (`KeyframeArtifact.kind === "phase"`): the band at its `t_ms`, or null
 * (composite images have `t_ms: null`).
 */
export function bandForKeyframe(model: Pick<VelocityModel, "bands">, t_ms: number | null): VelocityBand | null {
  return t_ms == null ? null : bandAt(model, t_ms);
}

/**
 * Velocity / position under the playhead, linearly interpolated between samples (null without
 * samples or outside them). `degraded` if either neighbouring sample is degraded.
 */
export function velocityAt(model: Pick<VelocityModel, "points">, t_ms: number): VelocityPoint | null {
  const pts = model.points;
  if (pts.length === 0 || t_ms < pts[0].t_ms || t_ms > pts[pts.length - 1].t_ms) return null;
  let lo = 0;
  let hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].t_ms <= t_ms) lo = mid;
    else hi = mid;
  }
  const a = pts[lo];
  const b = pts[hi];
  if (b.t_ms === a.t_ms || t_ms === a.t_ms) return { ...a };
  const k = (t_ms - a.t_ms) / (b.t_ms - a.t_ms);
  return {
    t_ms,
    v: a.v + (b.v - a.v) * k,
    pos: a.pos + (b.pos - a.pos) * k,
    quality: Math.min(a.quality, b.quality),
    degraded: a.degraded || b.degraded,
  };
}

/* ---------- behaviour summary (ContinuousSummary) ---------- */

export type BehaviorKey = "autoplay" | "loop" | "card_scale" | "pause" | "drag" | "inertia" | "snap" | "resume";

export type BehaviorFlag =
  /** Pause trigger not visible (no cursor): hover or press. */
  | "trigger_unknown"
  /** Autoplay present but the content never repeated: loop length unknown. */
  | "loop_not_observed"
  /** Abrupt stops without grid evidence: a snap, or the pointer stopping before release. */
  | "snap_ambiguous"
  /** Spring-like overshoot seen; spring parameters not reconstructed. */
  | "overshoot"
  /** No cursor visible during drags: "follows the pointer" can't be checked. */
  | "pointer_unknown"
  /** Autoplay resumed in the other direction. */
  | "direction_changed";

export interface BehaviorRow {
  key: BehaviorKey;
  /** Row title, e.g. "Autoplay", "Momentum". */
  label: string;
  /** false → "Not observed" (hide the row or show it muted). */
  observed: boolean;
  /** One-line summary, e.g. "≈39 px/s right, constant speed". */
  value: string;
  /** Confidence of the row's primary value (see README), null when not observed / not measured. */
  confidence: Confidence | null;
  /** confidence < 0.3 → show under "uncertain", not as a plain fact. */
  uncertain: boolean;
  /** Every value of the behaviour with its own confidence. */
  facts: VelocityFact[];
  flags: BehaviorFlag[];
  notes: string[];
  /** What the related phases are based on (trigger source: cursor vs motion only); null if none. */
  evidence: PhaseEvidence | null;
  /** Related phases (highlight / seek on the chart). */
  phase_ids: string[];
}

const BEHAVIOR_LABELS: Record<BehaviorKey, string> = {
  autoplay: "Autoplay",
  loop: "Loop",
  card_scale: "Card scaling",
  pause: "Pause",
  drag: "Drag",
  inertia: "Momentum",
  snap: "Snap",
  resume: "Resume",
};

const EVIDENCE_RANK: Record<PhaseEvidence, number> = { velocity: 0, cursor: 1, "velocity+cursor": 2 };

function relatedPhases(c: ContinuousMotion, kinds: PhaseKind[]): Phase[] {
  return c.phases.filter((p) => kinds.includes(p.kind));
}

/** Strongest evidence among the phases (velocity+cursor > cursor > velocity). */
function bestEvidence(phases: Phase[]): PhaseEvidence | null {
  return phases.reduce<PhaseEvidence | null>((best, p) => (best == null || EVIDENCE_RANK[p.evidence] > EVIDENCE_RANK[best] ? p.evidence : best), null);
}

function row(
  key: BehaviorKey,
  phases: Phase[],
  data: { value: string; confidence?: Confidence | null; facts?: VelocityFact[]; flags?: BehaviorFlag[]; notes?: string[] } | null,
  notObserved: string,
): BehaviorRow {
  if (!data) {
    return { key, label: BEHAVIOR_LABELS[key], observed: false, value: notObserved, confidence: null, uncertain: false, facts: [], flags: [], notes: [], evidence: null, phase_ids: [] };
  }
  const confidence = data.confidence ?? null;
  return {
    key,
    label: BEHAVIOR_LABELS[key],
    observed: true,
    value: data.value,
    confidence,
    uncertain: confidence != null && isUncertainValue(confidence.value),
    facts: data.facts ?? [],
    flags: data.flags ?? [],
    notes: data.notes ?? [],
    evidence: bestEvidence(phases),
    phase_ids: phases.map((p) => p.id),
  };
}

/**
 * Momentum row. Two models, worded apart: an exponential coast (`τ`) that either comes to rest
 * (below `stop_px_s`) or blends back into autoplay (`v_inf` ≠ 0), and a fixed-length eased glide
 * (tween). Releases fitted with the other model than the aggregate are counted, not hidden.
 */
function inertiaRow(
  inertia: NonNullable<ContinuousMotion["behavior"]["inertia"]>,
  phases: Phase[],
  axis: Axis,
): { value: string; confidence: Confidence | null; facts: VelocityFact[] } {
  const exp = phases.map((p) => p.fit).filter((f): f is ExponentialFit => f?.model === "exponential");
  const tweens = phases.filter((p) => p.fit?.model === "tween").length;
  const joins = exp.filter((f) => decaysToAutoplay(f));
  // Where exponential releases end: all at rest, all into autoplay, or both.
  const towards = joins.length === 0 ? "rest" : joins.length === exp.length ? "autoplay" : "mixed";
  const fitted = exp.length + tweens;
  const stop = inertia.stop_px_s ?? null;

  let value: string;
  if (inertia.model === "exponential") {
    const tau = inertia.tau_ms ? ` (${formatTau(inertia.tau_ms.value)})` : "";
    if (towards === "autoplay") value = `Coasts after release, then blends into autoplay${tau}`;
    else if (towards === "mixed") value = `Coasts after release, slowing exponentially${tau}, to rest or into autoplay`;
    else value = `Coasts after release, slowing exponentially${tau}${stop ? `; stops below ${formatSpeed(stop.value)}` : ""}`;
    if (tweens > 0) value += `; ${tweens} of ${fitted} releases glide instead`;
  } else {
    value =
      `Glides to rest after release${inertia.duration_ms ? ` over ${formatApproxMs(inertia.duration_ms.value)}` : ""}` +
      (inertia.easing ? ` (${formatEasingName(inertia.easing)})` : "");
    if (exp.length > 0) value += `; ${exp.length} of ${fitted} releases coast exponentially instead`;
  }

  const ends: VelocityFact[] =
    towards === "autoplay"
      ? [plain("Ends", `Blends into autoplay (${formatVelocity(joins[0].v_inf_px_s, axis)})`)]
      : towards === "mixed"
        ? [plain("Ends", `At rest (${exp.length - joins.length}×) or in autoplay (${joins.length}×)`)]
        : inertia.model === "exponential"
          ? [plain("Ends", "At rest"), ...measured("Stops below", stop, (v) => formatSpeed(v))]
          : [plain("Ends", "At rest")];

  return {
    value,
    confidence: inertia.tau_ms?.confidence ?? inertia.duration_ms?.confidence ?? null,
    facts: [
      plain("Model", INERTIA_MODEL_LABELS[inertia.model]),
      ...(exp.length > 0 && tweens > 0 ? [plain("Releases", `${exp.length} exponential, ${tweens} eased glide${tweens === 1 ? "" : "s"}`)] : []),
      ...measured("Time constant", inertia.tau_ms, formatTau),
      ...measured("Duration", inertia.duration_ms, formatApproxMs),
      ...(inertia.easing ? [plain("Easing", formatEasing(inertia.easing, { withNearest: true }))] : []),
      ...ends,
      plain(
        "Release speeds",
        inertia.release_speed_min_px_s === inertia.release_speed_max_px_s
          ? formatSpeed(inertia.release_speed_min_px_s)
          : `${formatSpeed(inertia.release_speed_min_px_s, { unit: false })}–${formatSpeed(inertia.release_speed_max_px_s, { approx: false, unit: false })} px/s`,
      ),
      plain("Seen", `${inertia.instances}×`),
    ],
  };
}

/** Where autoplay stops without slowing down: a drag / rest / stop straight after autoplay or a resume. */
function instantStopPhases(c: ContinuousMotion): Phase[] {
  return c.phases.filter((p, i) => {
    const prev = c.phases[i - 1];
    return (p.kind === "drag" || p.kind === "paused" || p.kind === "stop") && (prev?.kind === "autoplay" || prev?.kind === "resume");
  });
}

function directionWord(d: AutoplayDirection): string {
  return AUTOPLAY_DIRECTION_LABELS[d].toLowerCase();
}

/**
 * One row per behaviour, always in the order autoplay, loop, pause, drag, inertia, snap, resume.
 * Behaviours that were not seen come back with `observed: false` and a "Not observed …" value, so
 * the UI decides whether to hide or mute them. Exception: the "Card scaling" row (P2c, cards that
 * grow with their distance from the scroller centre) sits after loop and exists only when
 * `card_scale` was measured — rigid cards are the norm, not a missing behaviour. Primary confidence per row: autoplay → speed, loop
 * → period, pause → deceleration time (else trigger), drag → peak speed, inertia → τ (else tween
 * duration), snap → step (else duration; null for ambiguous stops), resume → ramp duration.
 */
export function buildBehaviorSummary(c: ContinuousMotion): BehaviorRow[] {
  const axis = c.axis;
  const b = c.behavior;
  const a = c.autoplay;
  const autoplayPhases = relatedPhases(c, ["autoplay"]);
  const rows: BehaviorRow[] = [];

  rows.push(
    row(
      "autoplay",
      autoplayPhases,
      a && {
        value: `${formatVelocity(a.velocity_px_s, axis)}, constant speed`,
        confidence: a.speed_px_s.confidence,
        facts: [
          ...measured("Speed", a.speed_px_s, (v) => formatSpeed(v)),
          plain("Direction", AUTOPLAY_DIRECTION_LABELS[a.direction]),
          plain("Easing", "linear"),
        ],
      },
      "No autoplay before the first interaction",
    ),
  );

  rows.push(
    row(
      "loop",
      autoplayPhases,
      a &&
        (a.loop.observed && a.loop.period_px && a.loop.duration_ms
          ? {
              value: `Repeats every ${formatPx(a.loop.period_px.value)} (${formatApproxMs(a.loop.duration_ms.value)} per loop)`,
              confidence: a.loop.period_px.confidence,
              facts: [...measured("Loop length", a.loop.period_px, (v) => formatPx(v)), ...measured("Loop duration", a.loop.duration_ms, formatApproxMs)],
            }
          : {
              value: "Not observed: the content never repeated within the recording",
              flags: ["loop_not_observed"],
            }),
      "No autoplay, so no loop",
    ),
  );

  const cs = c.card_scale ?? null;
  if (cs) {
    rows.push(
      row(
        "card_scale",
        [],
        {
          value: formatCardScale(cs),
          confidence: cs.confidence,
          facts: [
            plain("Model", "Grows with the squared distance from the scroller centre"),
            { label: `Scale at ${formatPx(cs.reference_distance_px)}`, value: `×${formatScale(cs.scale_at_reference)}`, confidence: cs.confidence },
            plain("Layout", "Cards stay packed: the gaps keep their size"),
            plain("On-screen speed", `×${formatScale(cs.mean_scale)} the unscaled track (mean magnification)`),
          ],
        },
        "Not measured",
      ),
    );
  }

  const pause = b.pause;
  const decelPhases = relatedPhases(c, ["decelerate"]);
  // No slow-down at all (e.g. the press that starts a drag): autoplay stops at once. Same test as the
  // technical output ("Slowdown: none (autoplay stops at once)").
  const stopsAtOnce = pause != null && pause.decel_ms == null && decelPhases.length === 0;
  const trigger = pause == null ? "" : pause.on === "unknown" ? " (trigger not visible: hover or press)" : ` when the pointer ${pause.on === "hover" ? "hovers" : "presses"}`;
  rows.push(
    row(
      "pause",
      stopsAtOnce ? instantStopPhases(c) : decelPhases,
      pause && {
        value: stopsAtOnce
          ? `Stops at once${trigger}`
          : `${pause.stops_completely ? "Slows to a stop" : "Slows down"}${pause.decel_ms ? ` over ${formatApproxMs(pause.decel_ms.value)}` : ""}${trigger}`,
        confidence: pause.decel_ms?.confidence ?? pause.on_confidence,
        facts: [
          { label: "Trigger", value: PAUSE_TRIGGER_LABELS[pause.on], confidence: pause.on_confidence },
          ...(stopsAtOnce ? [plain("Slow-down", "None, stops at once")] : pause.decel_ms ? measured("Slow-down", pause.decel_ms, formatApproxMs) : [plain("Slow-down", "Not measured")]),
          ...(pause.easing ? [plain("Easing", formatEasing(pause.easing, { withNearest: true }))] : []),
          plain("Stops completely", pause.stops_completely ? "Yes" : "No (interrupted first)"),
        ],
        flags: pause.on === "unknown" ? ["trigger_unknown"] : [],
      },
      "Not observed",
    ),
  );

  const drag = b.drag;
  rows.push(
    row(
      "drag",
      relatedPhases(c, ["drag"]),
      drag && {
        value:
          `${drag.count} drag${drag.count === 1 ? "" : "s"}, peak ${formatSpeed(drag.peak_speed_px_s.value)}` +
          (drag.follows_pointer === true ? ", follows the pointer" : ""),
        confidence: drag.peak_speed_px_s.confidence,
        facts: [
          plain("Drags", String(drag.count)),
          ...measured("Peak speed", drag.peak_speed_px_s, (v) => formatSpeed(v)),
          drag.follows_pointer == null
            ? plain("Follows the pointer", "Assumed 1:1 (pointer not visible during drags)")
            : { label: "Follows the pointer", value: drag.follows_pointer ? "Yes" : "No", confidence: drag.pointer_ratio?.confidence ?? null },
          ...measured("Content / pointer speed", drag.pointer_ratio, (v) => `${v.toFixed(2)}×`),
        ],
        flags: drag.follows_pointer == null ? ["pointer_unknown"] : [],
      },
      "Not observed",
    ),
  );

  const inertia = b.inertia;
  const inertiaPhases = relatedPhases(c, ["inertia"]);
  rows.push(row("inertia", inertiaPhases, inertia && inertiaRow(inertia, inertiaPhases, axis), "Not observed"));

  const snap = b.snap;
  rows.push(
    row(
      "snap",
      relatedPhases(c, [snap?.kind === "abrupt_ambiguous" ? "stop" : "snap"]),
      snap && {
        value:
          snap.kind === "grid"
            ? `Settles on the nearest card${snap.step_px ? ` (step ${formatPx(snap.step_px.value)})` : ""}${snap.duration_ms ? ` over ${formatApproxMs(snap.duration_ms.value)}` : ""}`
            : "Stops abruptly: a snap, or the pointer stopping before release (ambiguous without a cursor)",
        confidence: snap.step_px?.confidence ?? snap.duration_ms?.confidence ?? null,
        facts: [
          plain("Kind", SNAP_KIND_LABELS[snap.kind]),
          ...measured("Step", snap.step_px, (v) => formatPx(v)),
          ...measured("Duration", snap.duration_ms, formatApproxMs),
          ...(snap.easing ? [plain("Easing", formatEasing(snap.easing, { withNearest: true }))] : []),
          ...(snap.overshoot ? [plain("Overshoot", "Yes (spring-like; spring not reconstructed)")] : []),
        ],
        flags: [...(snap.kind === "abrupt_ambiguous" ? (["snap_ambiguous"] as const) : []), ...(snap.overshoot ? (["overshoot"] as const) : [])],
      },
      "Not observed",
    ),
  );

  const resume = b.resume;
  // Hover pauses: the resume clock that matters starts when the pointer leaves (same lead as the
  // generated prompt); otherwise it starts when the motion came to rest.
  const leave = resume?.delay_after_leave_ms ?? null;
  rows.push(
    row(
      "resume",
      relatedPhases(c, ["resume"]),
      resume && {
        value:
          `Resumes ${leave ? `${formatApproxMs(leave.value)} after the pointer leaves` : `${formatApproxMs(resume.delay_after_rest_ms.value)} after coming to rest`}, ` +
          `ramping to ${formatSpeed(resume.to_speed_px_s)} over ${formatApproxMs(resume.ramp_ms.value)}${resume.easing ? ` (${formatEasingName(resume.easing)})` : ""}`,
        confidence: resume.ramp_ms.confidence,
        facts: [
          ...measured("Delay after pointer leaves", leave, formatApproxMs),
          ...measured("Delay after rest", resume.delay_after_rest_ms, formatApproxMs),
          ...measured("Delay after release", resume.delay_after_release_ms, formatApproxMs),
          ...measured("Ramp", resume.ramp_ms, formatApproxMs),
          ...(resume.easing ? [plain("Easing", formatEasing(resume.easing, { withNearest: true }))] : []),
          plain("Target speed", formatSpeed(resume.to_speed_px_s)),
          plain(
            "Direction",
            resume.direction_preserved ? "Same as before" : a ? `Reversed (now ${directionWord(oppositeDirection(a.direction))})` : "Reversed",
          ),
        ],
        flags: resume.direction_preserved ? [] : ["direction_changed"],
      },
      "Not observed",
    ),
  );

  return rows;
}

function oppositeDirection(d: AutoplayDirection): AutoplayDirection {
  return ({ left: "right", right: "left", up: "down", down: "up" } as const)[d];
}
