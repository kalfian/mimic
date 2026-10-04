/**
 * Mock mode: continuous-mode results (PLAN-continuous §7.2 states) built from the contract fixture
 * `sample-continuous-result.json` (synced via `pnpm sync:fixture`). Pure transforms, no I/O.
 *
 * Every variant stays a valid 0.2 IR (same rules as `api/app/models/ir.py`), so the UI never sees
 * a shape the backend can't produce. Numbers are synthetic.
 *
 * | Variant                  | What it exercises                                                     |
 * |--------------------------|-----------------------------------------------------------------------|
 * | `continuous`             | fixture as-is: drag carousel, no cursor, loop not observed, pause      |
 * |                          | trigger unknown, abrupt-ambiguous stop, compressed scale suggested     |
 * | `continuous_degraded`    | fast drag samples with tracking quality < 0.5 (dashed/gap line),      |
 * |                          | two `degraded` bands, lowered confidences                             |
 * | `continuous_uncertain`   | several phases and behaviour values < 0.3 (hatched bands, uncertain)  |
 * | `continuous_autoplay`    | autoplay-only marquee (single band), loop observed, no behaviours     |
 * | `continuous_vertical`    | axis y (up/down wording), cursor visible (hover pause, markers,       |
 * |                          | "Motion + cursor" evidence, follows pointer), grid snap               |
 * | `continuous_no_samples`  | `samples: []` → no chart line, fall back to the phase list            |
 * | `continuous_glide`       | P2 shapes: autoplay stops at once on press (no slow-down), momentum  |
 * |                          | that blends back into autoplay (`v_inf` ≠ 0), a tween "glide" release |
 * |                          | + P2c cards that grow towards the edges (`card_scale`, ×1 → ×1.17)     |
 */

import { bandFor } from "../format";
import type { Confidence, Easing, MeasuredNumber, Phase, ResultEnvelope } from "../types";
import sampleContinuousJson from "./sample-continuous-result.json";

export const CONTINUOUS_VARIANTS = [
  "continuous",
  "continuous_degraded",
  "continuous_uncertain",
  "continuous_autoplay",
  "continuous_vertical",
  "continuous_no_samples",
  "continuous_glide",
] as const;

export type ContinuousVariant = (typeof CONTINUOUS_VARIANTS)[number];

export function isContinuousVariant(value: unknown): value is ContinuousVariant {
  return typeof value === "string" && (CONTINUOUS_VARIANTS as readonly string[]).includes(value);
}

/** The contract fixture (do not mutate; `continuousResult()` returns a fresh copy). */
export const CONTINUOUS_SAMPLE = sampleContinuousJson as unknown as ResultEnvelope;

/** Job id of the continuous fixture (`/jobs/<id>` in mock mode shows it directly). */
export const MOCK_CONTINUOUS_SAMPLE_JOB_ID: string = CONTINUOUS_SAMPLE.job_id;

const conf = (value: number): Confidence => ({ value, band: bandFor(value) });
const num = (value: number, c: number): MeasuredNumber => ({ value, confidence: conf(c) });

const EASE_OUT_QUAD: Easing = { keyword: null, cubic_bezier: [0.25, 0.46, 0.45, 0.94], nearest_named: "easeOutQuad", family: "ease-out", rmse: 0.012, flags: [] };
const EASE_IN_OUT: Easing = { keyword: "ease-in-out", cubic_bezier: [0.42, 0, 0.58, 1], nearest_named: "ease-in-out", family: "ease-in-out", rmse: 0.02, flags: [] };

const EASE_OUT: Easing = { keyword: "ease-out", cubic_bezier: [0, 0, 0.58, 1], nearest_named: "ease-out", family: "ease-out", rmse: 0.03, flags: [] };

function phase(result: ResultEnvelope, id: string): Phase {
  const p = result.spec.continuous?.phases.find((x) => x.id === id);
  if (!p) throw new Error(`continuous fixture has no phase ${id}`);
  return p;
}

function setConfidence(target: { confidence: Confidence }, value: number) {
  target.confidence = conf(value);
}

/* ---------- variants ---------- */

/** Fast drag frames lose tracking quality (motion blur): p3 and p9 become `degraded` bands. */
function applyDegraded(result: ResultEnvelope) {
  const c = result.spec.continuous!;
  // Strictly inside the drags p3 (2000–2500) and p9 (7000–7600): boundary samples are shared.
  const inFastDrag = (t: number) => (t > 2000 && t < 2500) || (t > 7000 && t < 7600);
  c.samples = c.samples.map(([t, v, pos, q]) => {
    if (inFastDrag(t) && Math.abs(v) >= 1000) return [t, v, pos, 0.35];
    if (inFastDrag(t) && Math.abs(v) >= 400) return [t, v, pos, 0.55];
    return [t, v, pos, q];
  });
  // §4.3: phases with > 10 % low-quality frames get their value confidences × 0.7.
  for (const id of ["p3", "p9"]) setConfidence(phase(result, id), 0.56);
  const drag = c.behavior.drag!;
  drag.peak_speed_px_s = num(drag.peak_speed_px_s.value, 0.53);
}

/** Several labels and behaviour values below 0.3 (uncertain observations). */
function applyUncertain(result: ResultEnvelope) {
  const c = result.spec.continuous!;
  const p2 = phase(result, "p2");
  setConfidence(p2, 0.2);
  if (p2.fit?.model === "ramp") p2.fit.duration_ms = num(p2.fit.duration_ms.value, 0.2);
  const p6 = phase(result, "p6");
  setConfidence(p6, 0.25);
  if (p6.fit?.model === "exponential") p6.fit.tau_ms = num(p6.fit.tau_ms.value, 0.25);
  setConfidence(phase(result, "p10"), 0.15);
  setConfidence(phase(result, "p11"), 0.22);

  const b = c.behavior;
  b.pause!.on_confidence = conf(0.15);
  b.pause!.decel_ms = num(b.pause!.decel_ms!.value, 0.2);
  b.inertia!.tau_ms = num(b.inertia!.tau_ms!.value, 0.25);
  b.resume!.delay_after_rest_ms = num(b.resume!.delay_after_rest_ms.value, 0.2);
  b.resume!.delay_after_release_ms = num(b.resume!.delay_after_release_ms!.value, 0.18);
  result.spec.interaction.type_confidence = conf(0.35);
}

/** Autoplay-only marquee (C2-like): 240 px/s left for 8 s, 4-card loop (864 px) observed. */
function applyAutoplayOnly(result: ResultEnvelope) {
  const spec = result.spec;
  const c = spec.continuous!;
  const span = 8000;
  const v = -240;
  const samples: [number, number, number, number][] = [];
  for (let t = 0; t <= span; t += 50) {
    const jitter = Number((1.2 * Math.sin(t / 170)).toFixed(1)); // deterministic tracking noise
    samples.push([t, v + jitter, Number(((v * t) / 1000).toFixed(2)), 0.95]);
  }
  c.samples = samples;
  c.span_ms = { start_ms: 0, end_ms: span };
  c.autoplay = {
    direction: "left",
    speed_px_s: num(240, 0.92),
    velocity_px_s: v,
    easing: "linear",
    loop: { observed: true, period_px: num(864, 0.85), duration_ms: num(3600, 0.85) },
  };
  c.phases = [
    {
      id: "p1",
      kind: "autoplay",
      start_ms: 0,
      end_ms: span,
      v_start_px_s: v,
      v_end_px_s: v,
      v_peak_px_s: v,
      displacement_px: (v * span) / 1000,
      fit: { model: "constant", velocity_px_s: num(v, 0.92) },
      interrupted: false,
      evidence: "velocity",
      confidence: conf(0.92),
      notes: [],
    },
  ];
  c.behavior = { pause: null, drag: null, inertia: null, snap: null, resume: null };

  spec.interaction = {
    ...spec.interaction,
    type: "continuous",
    type_confidence: conf(0.85),
    pattern: "marquee",
    trigger: {
      kind: "autoplay",
      reverse_kind: "none",
      description: "The content scrolls on its own; no interaction was recorded.",
      confidence: conf(0.85),
    },
    total_duration_ms: { forward: span, reverse: null },
  };
  spec.source = { ...spec.source, filename: "marquee.mp4", duration_ms: span, fps_effective: 60, is_vfr: false, timing_resolution_ms: 17 };
  spec.warnings = [];
  result.artifacts.keyframes = result.artifacts.keyframes.slice(0, 1).map((k) => ({ ...k, t_ms: span / 2 }));
}

/** Flip every signed continuous value (velocities, positions, distances). */
function negateSigns(result: ResultEnvelope) {
  const c = result.spec.continuous!;
  const neg = (x: number) => (x === 0 ? 0 : -x);
  c.samples = c.samples.map(([t, v, pos, q]) => [t, neg(v), neg(pos), q]);
  for (const p of c.phases) {
    p.v_start_px_s = neg(p.v_start_px_s);
    p.v_end_px_s = neg(p.v_end_px_s);
    p.v_peak_px_s = neg(p.v_peak_px_s);
    p.displacement_px = neg(p.displacement_px);
    const f = p.fit;
    if (f?.model === "constant") f.velocity_px_s.value = neg(f.velocity_px_s.value);
    if (f?.model === "exponential") {
      f.v0_px_s.value = neg(f.v0_px_s.value);
      f.v_inf_px_s = neg(f.v_inf_px_s);
    }
    if (f?.model === "ramp") {
      f.from_px_s = neg(f.from_px_s);
      f.to_px_s = neg(f.to_px_s);
    }
    if (f?.model === "tween") f.distance_px.value = neg(f.distance_px.value);
  }
  if (c.autoplay) c.autoplay.velocity_px_s = neg(c.autoplay.velocity_px_s);
}

/**
 * Vertical ticker (C10-like): content moves up, cursor visible (hover pause, drags follow the
 * pointer), release ends in a grid snap.
 */
function applyVertical(result: ResultEnvelope) {
  const spec = result.spec;
  const c = spec.continuous!;
  negateSigns(result);
  c.axis = "y";
  c.autoplay!.direction = "up";
  c.region = { x: 520, y: 100, w: 240, h: 600 };
  c.pitch_px = num(176, 0.7);
  const boxes: Record<string, { x: number; y: number; w: number; h: number }> = {
    e1: c.region,
    e2: { x: 528, y: 112, w: 224, h: 160 },
    e3: { x: 544, y: 232, w: 120, h: 20 },
  };
  for (const el of spec.elements) {
    const box = boxes[el.id];
    if (box) {
      el.bbox_initial = { ...box };
      el.bbox_active = { ...box };
    }
  }

  // Cursor visible: hover starts the slow-down, the drags follow it, leaving starts the resume.
  spec.cursor = {
    visible: true,
    confidence: 0.9,
    events: [
      { t_ms: 1450, kind: "enter", element_id: "e1" },
      { t_ms: 7800, kind: "leave", element_id: "e1" },
    ],
  };
  const p2 = phase(result, "p2");
  p2.evidence = "velocity+cursor";
  setConfidence(p2, 0.85);
  for (const p of c.phases) if (p.kind === "drag") p.evidence = "velocity+cursor";
  c.behavior.pause = { ...c.behavior.pause!, on: "hover", on_confidence: conf(0.85) };
  c.behavior.drag = { ...c.behavior.drag!, follows_pointer: true, pointer_ratio: num(1.0, 0.8) };

  // The abrupt stop becomes a snap tween onto the card grid.
  const p10 = phase(result, "p10");
  p10.kind = "snap";
  p10.fit = { model: "tween", distance_px: num(p10.displacement_px, 0.6), duration_ms: num(150, 0.6), easing: EASE_OUT };
  p10.notes = [];
  setConfidence(p10, 0.7);
  c.behavior.snap = { kind: "grid", step_px: num(176, 0.7), duration_ms: num(150, 0.6), easing: EASE_OUT, overshoot: false };

  spec.interaction.trigger = {
    ...spec.interaction.trigger,
    description: "A vertical ticker that slows down on hover and is dragged by the pointer; after release it coasts and snaps to the nearest card.",
    confidence: conf(0.8),
  };
  spec.source = { ...spec.source, filename: "ticker-vertical.mp4" };
  spec.warnings = spec.warnings.filter((w) => w.code !== "cursor_not_visible");
}

/** No velocity samples (e.g. trimmed by an older backend): the UI falls back to the phase list. */
function applyNoSamples(result: ResultEnvelope) {
  result.spec.continuous!.samples = [];
}

/** Velocity at `t` inside a phase, from its fit (drags: straight line to the release speed). */
function fitVelocity(p: Phase, t: number): number {
  const f = p.fit;
  const u = Math.min(1, Math.max(0, (t - p.start_ms) / Math.max(1, p.end_ms - p.start_ms)));
  if (f?.model === "constant") return f.velocity_px_s.value;
  if (f?.model === "exponential") return f.v_inf_px_s + (f.v0_px_s.value - f.v_inf_px_s) * Math.exp(-(t - p.start_ms) / f.tau_ms.value);
  if (f?.model === "ramp" || f?.model === "tween") {
    // Solve the bezier parameter for time progress u, then read the eased value / slope there.
    const [x1, y1, x2, y2] = f.easing.cubic_bezier;
    const bx = (s: number) => 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s * s * x2 + s ** 3;
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      if (bx(mid) < u) lo = mid;
      else hi = mid;
    }
    const s = (lo + hi) / 2;
    if (f.model === "ramp") return f.from_px_s + (f.to_px_s - f.from_px_s) * (3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s ** 3);
    const dx = 3 * (1 - s) ** 2 * x1 + 6 * (1 - s) * s * (x2 - x1) + 3 * s * s * (1 - x2);
    const dy = 3 * (1 - s) ** 2 * y1 + 6 * (1 - s) * s * (y2 - y1) + 3 * s * s * (1 - y2);
    return dx > 1e-6 ? (f.distance_px.value / (f.duration_ms.value / 1000)) * (dy / dx) : 0;
  }
  return p.v_start_px_s + (p.v_end_px_s - p.v_start_px_s) * u * u; // drag: accelerates into the release
}

/**
 * The shapes P2 found in a real recording (numbers synthetic): autoplay +35 px/s; a press stops it
 * at once (no slow-down) and drags left; the momentum decays towards the autoplay velocity and
 * blends back into autoplay without resting; a second drag ends in a 300 ms eased glide (tween,
 * close to easeOutQuad) to rest; after a short rest autoplay ramps back. No cursor visible.
 */
function applyGlide(result: ResultEnvelope) {
  const spec = result.spec;
  const c = spec.continuous!;
  const auto = 35;
  const span = 9300;
  const glideMs = 300;
  const glidePx = Number(((1200 * glideMs) / 1000 / (0.46 / 0.25)).toFixed(1)); // release speed = D/T × E'(0)
  const base = { interrupted: false, evidence: "velocity" as const, notes: [] as string[] };
  const raw: Omit<Phase, "id" | "displacement_px" | "interrupted" | "evidence" | "notes">[] = [
    { kind: "autoplay", start_ms: 0, end_ms: 1500, v_start_px_s: auto, v_end_px_s: auto, v_peak_px_s: auto, fit: { model: "constant", velocity_px_s: num(auto, 0.9) }, confidence: conf(0.9) },
    { kind: "drag", start_ms: 1500, end_ms: 2000, v_start_px_s: auto, v_end_px_s: -1400, v_peak_px_s: -1400, fit: null, confidence: conf(0.8) },
    {
      kind: "inertia",
      start_ms: 2000,
      end_ms: 3400,
      v_start_px_s: -1400,
      v_end_px_s: 28.4,
      v_peak_px_s: -1400,
      fit: { model: "exponential", tau_ms: num(260, 0.85), v0_px_s: num(-1400, 0.7), v_inf_px_s: auto },
      confidence: conf(0.8),
    },
    { kind: "autoplay", start_ms: 3400, end_ms: 4400, v_start_px_s: auto, v_end_px_s: auto, v_peak_px_s: auto, fit: { model: "constant", velocity_px_s: num(auto, 0.88) }, confidence: conf(0.88) },
    { kind: "drag", start_ms: 4400, end_ms: 4900, v_start_px_s: auto, v_end_px_s: 1200, v_peak_px_s: 1200, fit: null, confidence: conf(0.8) },
    {
      kind: "inertia",
      start_ms: 4900,
      end_ms: 4900 + glideMs,
      v_start_px_s: 1200,
      v_end_px_s: 0,
      v_peak_px_s: 1200,
      fit: { model: "tween", distance_px: num(glidePx, 0.7), duration_ms: num(glideMs, 0.75), easing: EASE_OUT_QUAD },
      confidence: conf(0.75),
    },
    { kind: "paused", start_ms: 5200, end_ms: 5400, v_start_px_s: 0, v_end_px_s: 0, v_peak_px_s: 0, fit: null, confidence: conf(0.7) },
    {
      kind: "resume",
      start_ms: 5400,
      end_ms: 6200,
      v_start_px_s: 0,
      v_end_px_s: auto,
      v_peak_px_s: auto,
      fit: { model: "ramp", from_px_s: 0, to_px_s: auto, duration_ms: num(800, 0.8), easing: EASE_IN_OUT },
      confidence: conf(0.8),
    },
    { kind: "autoplay", start_ms: 6200, end_ms: span, v_start_px_s: auto, v_end_px_s: auto, v_peak_px_s: auto, fit: { model: "constant", velocity_px_s: num(auto, 0.9) }, confidence: conf(0.9) },
  ];
  const phases: Phase[] = raw.map((p, i) => ({ ...base, ...p, id: `p${i + 1}`, displacement_px: 0 }));

  // Samples at 60 fps from the fits, integrated to positions; per-phase displacement from them.
  const samples: [number, number, number, number][] = [];
  let pos = 0;
  let prev: [number, number] | null = null;
  for (let i = 0; i * (1000 / 60) <= span; i++) {
    const t = Number((i * (1000 / 60)).toFixed(1));
    const ph = phases.find((p) => t >= p.start_ms && t < p.end_ms) ?? phases[phases.length - 1];
    const v = Number(fitVelocity(ph, t).toFixed(1));
    if (prev) pos += ((prev[1] + v) / 2) * ((t - prev[0]) / 1000);
    samples.push([t, v, Number(pos.toFixed(2)), 0.93]);
    prev = [t, v];
  }
  for (const p of phases) {
    const at = (t: number) => samples.reduce((best, s) => (Math.abs(s[0] - t) < Math.abs(best[0] - t) ? s : best))[2];
    p.displacement_px = Number((at(p.end_ms) - at(p.start_ms)).toFixed(1));
  }

  c.samples = samples;
  c.span_ms = { start_ms: 0, end_ms: span };
  c.phases = phases;
  c.autoplay = { direction: "right", speed_px_s: num(auto, 0.9), velocity_px_s: auto, easing: "linear", loop: { observed: false, period_px: null, duration_ms: null } };
  c.behavior = {
    pause: { on: "unknown", on_confidence: conf(0.35), decel_ms: null, easing: null, stops_completely: true },
    drag: { count: 2, follows_pointer: null, pointer_ratio: null, peak_speed_px_s: num(1400, 0.75) },
    inertia: { model: "exponential", tau_ms: num(260, 0.85), duration_ms: null, easing: null, release_speed_min_px_s: 1200, release_speed_max_px_s: 1400, instances: 2 },
    snap: null,
    resume: {
      delay_after_rest_ms: num(200, 0.6),
      delay_after_release_ms: num(500, 0.55),
      ramp_ms: num(800, 0.8),
      easing: EASE_IN_OUT,
      to_speed_px_s: auto,
      direction_preserved: true,
    },
  };
  spec.interaction.trigger = {
    ...spec.interaction.trigger,
    description: "Autoplaying scroller that stops at once when grabbed; one release coasts back into autoplay, another glides to rest before autoplay resumes.",
  };
  spec.interaction.total_duration_ms = { forward: span, reverse: null };
  spec.source = { ...spec.source, filename: "carousel-glide.mov", duration_ms: span };
  spec.warnings = spec.warnings.filter((w) => w.code !== "tracking_degraded");
  result.artifacts.keyframes = result.artifacts.keyframes.map((k, i) => (k.t_ms == null ? k : { ...k, t_ms: [750, 1750, 2700, 4650, 5050, 5800][i % 6] }));
  // P2c: the motivating recording's cards grow with their distance from the scroller centre.
  // mean_scale = 1 + (1.17 − 1)·(half / 270)² / 3 with half = 380 (760 px region), as ir.py checks.
  const half = c.region.w / 2;
  c.card_scale = {
    model: "quadratic",
    origin: "scroller_center",
    reference_distance_px: 270,
    scale_at_reference: 1.17,
    mean_scale: Number((1 + (0.17 * (half / 270) ** 2) / 3).toFixed(4)),
    confidence: conf(0.6),
  };
}

const APPLY: Record<ContinuousVariant, ((r: ResultEnvelope) => void) | null> = {
  continuous: null,
  continuous_degraded: applyDegraded,
  continuous_uncertain: applyUncertain,
  continuous_autoplay: applyAutoplayOnly,
  continuous_vertical: applyVertical,
  continuous_no_samples: applyNoSamples,
  continuous_glide: applyGlide,
};

/** A fresh copy of the continuous fixture with `variant` applied. */
export function continuousResult(variant: ContinuousVariant = "continuous"): ResultEnvelope {
  const result = structuredClone(CONTINUOUS_SAMPLE);
  APPLY[variant]?.(result);
  return result;
}
