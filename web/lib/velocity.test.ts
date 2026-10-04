import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { CONTINUOUS_VARIANTS, continuousResult } from "./mock/continuousVariants";
import sampleResult from "./mock/sample-result.json";
import type { ContinuousMotion, MotionSpec, ResultEnvelope } from "./types";
import { CONTINUOUS_SAMPLE_MAX, PHASE_KINDS } from "./types";
import {
  bandAt,
  bandForKeyframe,
  buildBehaviorSummary,
  buildVelocityModel,
  DEGRADED_QUALITY_BELOW,
  findBand,
  velocityAt,
  velocityModelOf,
  velocityTicks,
  velocityToFraction,
} from "./velocity";

const fixture = () => continuousResult("continuous");
const model = (variant: (typeof CONTINUOUS_VARIANTS)[number] = "continuous") => velocityModelOf(continuousResult(variant).spec)!;

/** Minimal continuous section for edge cases. */
function section(patch: Partial<ContinuousMotion> = {}): ContinuousMotion {
  return { ...structuredClone(fixture().spec.continuous!), ...patch };
}

describe("buildVelocityModel on the contract fixture", () => {
  const m = model();
  const c = fixture().spec.continuous!;

  it("covers the analysed span with one point per sample", () => {
    assert.deepEqual(m.window, { start_ms: c.span_ms.start_ms, end_ms: c.span_ms.end_ms });
    assert.equal(m.points.length, c.samples.length);
    assert.ok(m.points.length <= CONTINUOUS_SAMPLE_MAX);
    assert.deepEqual(m.points[1], { t_ms: c.samples[1][0], v: c.samples[1][1], pos: c.samples[1][2], quality: c.samples[1][3], degraded: false });
    assert.equal(m.hasSamples, true);
    assert.equal(m.degradedCount, 0);
    assert.equal(m.runs.length, 1, "no degraded samples, no gaps → one polyline");
  });

  it("axis words follow the sign convention", () => {
    assert.equal(m.axis, "x");
    assert.equal(m.unit, "px/s");
    assert.equal(m.positiveLabel, "right");
    assert.equal(m.negativeLabel, "left");
    assert.equal(m.signConvention, c.sign_convention);
  });

  it("y domain is symmetric, padded and nice; compressed scale suggested (drags ≫ autoplay)", () => {
    assert.equal(m.yDomain.min, -m.yDomain.max);
    assert.ok(m.yDomain.max >= m.maxAbsV * 1.1);
    assert.equal(m.yDomain.max, 2000);
    assert.equal(m.maxAbsV, 1720);
    assert.equal(m.suggestCompressedScale, true); // 1720 / 39 > 10
    assert.equal(m.compressedPivot, 39);
    assert.deepEqual(m.autoplayRef, { v: 39, confidence: 0.9, band: "high", label: "Autoplay ≈39 px/s right" });
  });

  it("one band per phase, chronological, with labels, flags and headlines", () => {
    assert.deepEqual(
      m.bands.map((b) => b.phase_id),
      c.phases.map((p) => p.id),
    );
    for (const b of m.bands) {
      assert.ok(PHASE_KINDS.includes(b.kind));
      assert.ok(b.label.length > 0 && b.headline.length > 0, b.phase_id);
      assert.equal(b.duration_ms, b.end_ms - b.start_ms);
      assert.equal(b.uncertain, b.confidence < 0.3);
    }
    const byId = (id: string) => findBand(m, id)!;
    assert.equal(byId("p1").headline, "≈39 px/s right");
    assert.equal(byId("p2").headline, "≈39 → 0 px/s over ≈700 ms");
    assert.equal(byId("p2").interrupted, true);
    assert.equal(byId("p3").headline, "≈1500 px/s peak · left");
    assert.equal(byId("p4").headline, "τ\u00a0≈\u00a0200\u00a0ms");
    assert.equal(byId("p4").label, "Momentum");
    assert.equal(byId("p7").headline, "≈990 px/s peak · right");
    assert.equal(byId("p10").kind, "stop");
    assert.equal(byId("p10").band, "low");
    assert.equal(byId("p12").headline, "0 → ≈39 px/s over ≈750 ms");
    assert.equal(m.uncertainCount, 0);
  });

  it("band facts carry fit values with their own confidence", () => {
    const facts = findBand(m, "p4")!.facts;
    const tau = facts.find((f) => f.label === "Time constant");
    assert.deepEqual(tau, { label: "Time constant", value: "τ\u00a0≈\u00a0200\u00a0ms", confidence: { value: 0.82, band: "high" } });
    assert.equal(facts.find((f) => f.label === "Release speed")?.value, "≈1400 px/s left");
    assert.equal(facts.find((f) => f.label === "Distance")?.value, "280px left");
    assert.equal(facts.find((f) => f.label === "Based on")?.value, "Motion only");
  });

  it("derives release / rest / resume markers (no cursor events without a cursor)", () => {
    assert.deepEqual(
      m.markers.map((x) => `${x.kind}@${x.t_ms}`),
      ["release@2500", "release@5000", "release@6300", "release@7600", "rest@7750", "resume_start@7850"],
    );
    assert.equal(new Set(m.markers.map((x) => x.id)).size, m.markers.length, "stable unique ids");
  });
});

describe("velocity model states (mock variants)", () => {
  it("every variant builds (and stays within the sample cap)", () => {
    for (const v of CONTINUOUS_VARIANTS) {
      const m = model(v);
      assert.ok(m.bands.length >= 1, v);
      assert.ok(m.points.length <= CONTINUOUS_SAMPLE_MAX, v);
    }
  });

  it("degraded: low-quality samples split the line into dashed runs and flag their bands", () => {
    const m = model("continuous_degraded");
    assert.ok(m.degradedCount > 0);
    assert.ok(m.points.filter((p) => p.degraded).every((p) => p.quality < DEGRADED_QUALITY_BELOW));
    assert.deepEqual(
      m.runs.map((r) => r.degraded),
      [false, true, false, true, false],
    );
    // Neighbouring runs share their boundary point, so the drawn line has no holes.
    for (let i = 1; i < m.runs.length; i++) assert.equal(m.runs[i].points[0], m.runs[i - 1].points.at(-1));
    assert.deepEqual(
      m.bands.filter((b) => b.degraded).map((b) => b.phase_id),
      ["p3", "p9"],
    );
    assert.equal(m.degradedBandCount, 2);
  });

  it("uncertain: bands below 0.3 are flagged", () => {
    const m = model("continuous_uncertain");
    assert.deepEqual(
      m.bands.filter((b) => b.uncertain).map((b) => b.phase_id),
      ["p2", "p6", "p10", "p11"],
    );
    assert.equal(m.uncertainCount, 4);
    assert.ok(m.bands.filter((b) => b.uncertain).every((b) => b.band === "low"));
  });

  it("autoplay-only: a single band, no markers, linear scale is enough", () => {
    const m = model("continuous_autoplay");
    assert.equal(m.bands.length, 1);
    assert.equal(m.bands[0].kind, "autoplay");
    assert.equal(m.bands[0].headline, "≈240 px/s left");
    assert.deepEqual(m.markers, []);
    assert.equal(m.suggestCompressedScale, false);
    assert.equal(m.autoplayRef?.v, -240);
  });

  it("vertical: up/down wording and cursor markers", () => {
    const m = model("continuous_vertical");
    assert.equal(m.axis, "y");
    assert.equal(m.positiveLabel, "down");
    assert.equal(m.negativeLabel, "up");
    assert.equal(m.autoplayRef?.label, "Autoplay ≈39 px/s up");
    assert.equal(findBand(m, "p3")!.headline, "≈1500 px/s peak · down");
    assert.equal(findBand(m, "p10")!.kind, "snap");
    assert.equal(findBand(m, "p10")!.headline, "27px in ≈150 ms");
    assert.equal(findBand(m, "p2")!.evidence, "velocity+cursor");
    const kinds = m.markers.map((x) => x.kind);
    assert.ok(kinds.includes("cursor_enter") && kinds.includes("cursor_leave"));
    assert.deepEqual(
      m.markers.map((x) => x.t_ms),
      [...m.markers.map((x) => x.t_ms)].sort((a, b) => a - b),
    );
  });

  it("no samples: bands only, no line", () => {
    const m = model("continuous_no_samples");
    assert.equal(m.hasSamples, false);
    assert.deepEqual(m.points, []);
    assert.deepEqual(m.runs, []);
    assert.equal(m.bands.length, 13);
    assert.equal(m.maxAbsV, 1719.7); // from the phase peaks
    assert.equal(velocityAt(m, 1000), null);
  });
});

describe("transition / legacy specs", () => {
  it("velocityModelOf is null for a transition spec and for a stored 0.1 spec without mode", () => {
    const transition = (sampleResult as unknown as ResultEnvelope).spec;
    assert.equal(velocityModelOf(transition), null);
    const legacy = structuredClone(transition) as Partial<MotionSpec>;
    delete legacy.mode;
    legacy.schema_version = "0.1";
    assert.equal(velocityModelOf(legacy as MotionSpec), null);
  });
});

describe("geometry and lookup helpers", () => {
  const m = model();

  it("velocityToFraction: 0 in the middle, clamped, compressed lifts slow speeds", () => {
    assert.equal(velocityToFraction(0, m), 0.5);
    assert.equal(velocityToFraction(m.yDomain.max, m), 1);
    assert.equal(velocityToFraction(-1e9, m), 0);
    const linear = velocityToFraction(39, m, "linear");
    const compressed = velocityToFraction(39, m, "compressed");
    assert.ok(Math.abs(linear - 0.51) < 0.001);
    // asinh(1) / asinh(2000 / 39) ≈ 0.19 of the half height (vs ≈ 0.02 linear).
    assert.ok(compressed > 0.59 && compressed < 0.6, `asinh should spread the autoplay line, got ${compressed}`);
    assert.equal(velocityToFraction(0, m, "compressed"), 0.5);
    assert.ok(Math.abs(velocityToFraction(500, m, "compressed") + velocityToFraction(-500, m, "compressed") - 1) < 1e-12, "symmetric");
  });

  it("velocityTicks: symmetric, ascending, include 0", () => {
    assert.deepEqual(velocityTicks(m), [-2000, -1000, 0, 1000, 2000]);
    assert.deepEqual(velocityTicks(m, "compressed"), [-1000, -100, 0, 100, 1000]);
    assert.deepEqual(velocityTicks({ yDomain: { min: -500, max: 500 }, compressedPivot: 240 }, "linear", 2), [-500, -250, 0, 250, 500]);
  });

  it("bandAt: later phase wins at a shared boundary; null outside", () => {
    assert.equal(bandAt(m, 0)?.phase_id, "p1");
    assert.equal(bandAt(m, 1500)?.phase_id, "p2");
    assert.equal(bandAt(m, 2600)?.phase_id, "p4");
    assert.equal(bandAt(m, 9300)?.phase_id, "p13");
    assert.equal(bandAt(m, 9301), null);
  });

  it("bandForKeyframe labels phase keyframes by their time", () => {
    const keyframes = fixture().artifacts.keyframes;
    assert.ok(keyframes.every((k) => k.kind === "phase"));
    assert.deepEqual(
      keyframes.map((k) => bandForKeyframe(m, k.t_ms)?.label),
      ["Autoplay", "Drag", "Momentum", "Drag", "Abrupt stop", "Resumes"],
    );
    assert.equal(bandForKeyframe(m, null), null);
  });

  it("velocityAt interpolates between samples", () => {
    const at = velocityAt(m, 2025)!; // between 2000 (11.1) and 2050 (-492.6)
    assert.ok(Math.abs(at.v - (11.1 + -492.6) / 2) < 1e-9);
    assert.equal(velocityAt(m, 0)?.v, 39);
    assert.equal(velocityAt(m, -1), null);
  });

  it("gaps in the samples break the line", () => {
    const samples = fixture().spec.continuous!.samples.filter(([t]) => t < 1000 || t > 3000);
    const gapped = buildVelocityModel(section({ samples }));
    assert.equal(gapped.runs.length, 2);
    assert.equal(gapped.runs[0].points.at(-1)!.t_ms, 950);
    assert.equal(gapped.runs[1].points[0].t_ms, 3050);
  });
});

describe("buildBehaviorSummary", () => {
  const keys = ["autoplay", "loop", "pause", "drag", "inertia", "snap", "resume"];

  it("fixture: every behaviour, in a fixed order, with the §7.2 states", () => {
    const rows = buildBehaviorSummary(fixture().spec.continuous!);
    assert.deepEqual(
      rows.map((r) => r.key),
      keys,
    );
    const r = Object.fromEntries(rows.map((x) => [x.key, x]));
    assert.equal(r.autoplay.value, "≈39 px/s right, constant speed");
    assert.equal(r.autoplay.confidence?.band, "high");
    assert.deepEqual(r.loop.flags, ["loop_not_observed"]);
    assert.equal(r.loop.confidence, null);
    assert.deepEqual(r.pause.flags, ["trigger_unknown"]);
    assert.equal(r.pause.value, "Slows down over ≈700 ms (trigger not visible: hover or press)");
    assert.equal(r.pause.evidence, "velocity");
    assert.deepEqual(r.pause.phase_ids, ["p2"]);
    assert.deepEqual(r.drag.flags, ["pointer_unknown"]);
    assert.equal(r.drag.value, "4 drags, peak ≈1720 px/s");
    assert.deepEqual(r.drag.phase_ids, ["p3", "p5", "p7", "p9"]);
    assert.equal(r.inertia.value, "Coasts after release, slowing exponentially (τ\u00a0≈\u00a0200\u00a0ms)");
    assert.equal(r.inertia.facts.find((f) => f.label === "Release speeds")?.value, "≈750–1400 px/s");
    assert.deepEqual(r.snap.flags, ["snap_ambiguous"]);
    assert.deepEqual(r.snap.phase_ids, ["p10"]);
    assert.equal(r.resume.value, "Resumes ≈100 ms after coming to rest, ramping to ≈39 px/s over ≈750 ms (ease-in-out)");
    assert.ok(rows.every((x) => x.observed));
  });

  it("autoplay-only: loop observed, interaction behaviours not observed", () => {
    const rows = buildBehaviorSummary(continuousResult("continuous_autoplay").spec.continuous!);
    const r = Object.fromEntries(rows.map((x) => [x.key, x]));
    assert.equal(r.loop.value, "Repeats every 864px (≈3600 ms per loop)");
    assert.deepEqual(r.loop.flags, []);
    for (const k of ["pause", "drag", "inertia", "snap", "resume"]) {
      assert.equal(r[k].observed, false, k);
      assert.equal(r[k].value, "Not observed");
      assert.equal(r[k].confidence, null);
      assert.deepEqual(r[k].facts, []);
    }
  });

  it("vertical with cursor: hover trigger, follows pointer, grid snap", () => {
    const r = Object.fromEntries(buildBehaviorSummary(continuousResult("continuous_vertical").spec.continuous!).map((x) => [x.key, x]));
    assert.equal(r.autoplay.value, "≈39 px/s up, constant speed");
    assert.equal(r.pause.value, "Slows down over ≈700 ms when the pointer hovers");
    assert.deepEqual(r.pause.flags, []);
    assert.equal(r.pause.evidence, "velocity+cursor");
    assert.equal(r.drag.value, "4 drags, peak ≈1720 px/s, follows the pointer");
    assert.equal(r.drag.facts.find((f) => f.label === "Content / pointer speed")?.value, "1.00×");
    assert.equal(r.snap.value, "Settles on the nearest card (step 176px) over ≈150 ms");
    assert.deepEqual(r.snap.phase_ids, ["p10"]);
  });

  it("uncertain values mark their rows", () => {
    const r = Object.fromEntries(buildBehaviorSummary(continuousResult("continuous_uncertain").spec.continuous!).map((x) => [x.key, x]));
    assert.equal(r.pause.uncertain, true);
    assert.equal(r.inertia.uncertain, true);
    assert.equal(r.autoplay.uncertain, false);
  });

  it("no autoplay: autoplay and loop rows not observed", () => {
    const rows = buildBehaviorSummary(section({ autoplay: null }));
    assert.equal(rows[0].observed, false);
    assert.equal(rows[0].value, "No autoplay before the first interaction");
    assert.equal(rows[1].observed, false);
    assert.equal(buildVelocityModel(section({ autoplay: null })).suggestCompressedScale, false);
  });

  it("resume in the other direction is flagged", () => {
    const c = section();
    c.behavior.resume = { ...c.behavior.resume!, direction_preserved: false };
    const resume = buildBehaviorSummary(c).find((x) => x.key === "resume")!;
    assert.deepEqual(resume.flags, ["direction_changed"]);
    assert.equal(resume.facts.find((f) => f.label === "Direction")?.value, "Reversed (now left)");
  });
});

describe("P2c card scaling", () => {
  it("a measured card_scale adds one Card scaling row after Loop (glide variant)", () => {
    const rows = buildBehaviorSummary(continuousResult("continuous_glide").spec.continuous!);
    assert.deepEqual(
      rows.map((r) => r.key),
      ["autoplay", "loop", "card_scale", "pause", "drag", "inertia", "snap", "resume"],
    );
    const r = rows.find((x) => x.key === "card_scale")!;
    assert.equal(r.label, "Card scaling");
    assert.equal(r.value, "×1 at the centre → ≈×1.17 at 270px");
    assert.equal(r.confidence?.band, "medium");
    assert.equal(r.uncertain, false);
    assert.deepEqual(r.phase_ids, []);
    assert.equal(r.facts.find((f) => f.label === "Scale at 270px")?.value, "×1.17");
    assert.equal(r.facts.find((f) => f.label === "On-screen speed")?.value, "×1.11 the unscaled track (mean magnification)");
  });

  it("rigid cards (no card_scale): no row at all", () => {
    for (const variant of CONTINUOUS_VARIANTS.filter((v) => v !== "continuous_glide")) {
      const keys = buildBehaviorSummary(continuousResult(variant).spec.continuous!).map((r) => r.key);
      assert.ok(!keys.includes("card_scale"), variant);
    }
  });

  it("an uncertain scale is marked uncertain", () => {
    const c = continuousResult("continuous_glide").spec.continuous!;
    c.card_scale = { ...c.card_scale!, confidence: { value: 0.2, band: "low" } };
    assert.equal(buildBehaviorSummary(c).find((x) => x.key === "card_scale")?.uncertain, true);
  });
});

describe("P2 contract fields and wording", () => {
  const rowsOf = (c: ContinuousMotion) => Object.fromEntries(buildBehaviorSummary(c).map((x) => [x.key, x]));

  it("glide variant: press-stop at once, momentum blends into autoplay, tween release is a Glide", () => {
    const c = continuousResult("continuous_glide").spec.continuous!;
    const r = rowsOf(c);
    assert.equal(r.pause.value, "Stops at once (trigger not visible: hover or press)");
    assert.equal(r.pause.facts.find((f) => f.label === "Slow-down")?.value, "None, stops at once");
    assert.deepEqual(r.pause.phase_ids, ["p2", "p5"], "the drags that took over straight from autoplay");
    assert.equal(r.inertia.value, "Coasts after release, then blends into autoplay (τ\u00a0≈\u00a0260\u00a0ms); 1 of 2 releases glide instead");
    assert.ok(!r.inertia.value.includes("rest"));
    assert.equal(r.inertia.facts.find((f) => f.label === "Ends")?.value, "Blends into autoplay (≈35 px/s right)");
    assert.equal(r.inertia.facts.find((f) => f.label === "Releases")?.value, "1 exponential, 1 eased glide");

    const m = velocityModelOf(continuousResult("continuous_glide").spec)!;
    const p3 = findBand(m, "p3")!;
    assert.equal(p3.label, "Momentum");
    assert.equal(p3.headline, "τ\u00a0≈\u00a0260\u00a0ms → autoplay");
    assert.equal(p3.facts.find((f) => f.label === "Ends")?.value, "Blends into autoplay (≈35 px/s right)");
    const p6 = findBand(m, "p6")!;
    assert.equal(p6.label, "Glide");
    assert.equal(p6.headline, "≈300 ms · close to easeOutQuad");
    assert.equal(p6.facts.find((f) => f.label === "Model")?.value, "Eased glide (tween)");
    // No "comes to rest" where the momentum joined autoplay; a join marker instead.
    assert.deepEqual(
      m.markers.map((x) => `${x.kind}@${x.t_ms}`),
      ["release@2000", "autoplay_join@3400", "release@4900", "rest@5200", "resume_start@5400"],
    );
  });

  it("press that stops autoplay at once (no decelerate phase) is not a slow-down", () => {
    const c = section();
    c.phases = c.phases.filter((p) => p.kind !== "decelerate").map((p, i) => ({ ...p, id: `p${i + 1}` }));
    c.behavior.pause = { on: "press", on_confidence: { value: 0.7, band: "medium" }, decel_ms: null, easing: null, stops_completely: true };
    const r = rowsOf(c);
    assert.equal(r.pause.value, "Stops at once when the pointer presses");
    assert.equal(r.pause.confidence?.value, 0.7);
    // decel_ms missing but a slow-down phase exists → still a slow-down, duration not measured.
    const d = section();
    d.behavior.pause = { ...d.behavior.pause!, on: "hover", decel_ms: null, stops_completely: true };
    assert.equal(rowsOf(d).pause.value, "Slows to a stop when the pointer hovers");
    assert.equal(rowsOf(d).pause.facts.find((f) => f.label === "Slow-down")?.value, "Not measured");
  });

  it("stop_px_s: momentum to rest says where it counts as stopped", () => {
    const c = section();
    c.behavior.inertia = { ...c.behavior.inertia!, stop_px_s: { value: 10, confidence: { value: 0.9, band: "high" } } };
    const p4 = c.phases.find((p) => p.id === "p4")!;
    if (p4.fit?.model === "exponential") p4.fit.stop_px_s = 10;
    const r = rowsOf(c);
    assert.equal(r.inertia.value, "Coasts after release, slowing exponentially (τ\u00a0≈\u00a0200\u00a0ms); stops below ≈10 px/s");
    assert.equal(r.inertia.facts.find((f) => f.label === "Stops below")?.confidence?.band, "high");
    const m = buildVelocityModel(c);
    assert.equal(findBand(m, "p4")!.facts.find((f) => f.label === "Ends")?.value, "At rest below ≈10 px/s");
    assert.equal(findBand(m, "p6")!.facts.find((f) => f.label === "Ends")?.value, "Cut short before coming to rest");
  });

  it("delay_after_leave_ms leads the resume row (hover pause)", () => {
    const c = section();
    c.behavior.resume = { ...c.behavior.resume!, delay_after_leave_ms: { value: 1480, confidence: { value: 0.85, band: "high" } } };
    const r = rowsOf(c);
    assert.equal(r.resume.value, "Resumes ≈1480 ms after the pointer leaves, ramping to ≈39 px/s over ≈750 ms (ease-in-out)");
    assert.deepEqual(
      r.resume.facts.slice(0, 3).map((f) => f.label),
      ["Delay after pointer leaves", "Delay after rest", "Delay after release"],
    );
    // Absent (0.1/older results, press pauses) → the rest clock, as before.
    assert.ok(!rowsOf(section()).resume.facts.some((f) => f.label === "Delay after pointer leaves"));
  });

  it("pointer ratio not measured → assumed 1:1", () => {
    const r = rowsOf(section());
    assert.equal(r.drag.facts.find((f) => f.label === "Follows the pointer")?.value, "Assumed 1:1 (pointer not visible during drags)");
    assert.deepEqual(r.drag.flags, ["pointer_unknown"]);
  });
});

describe("band facts", () => {
  it("labels are unique per band (React keys), tween values not listed twice", () => {
    for (const v of CONTINUOUS_VARIANTS) {
      for (const b of model(v).bands) {
        const labels = b.facts.map((f) => f.label);
        assert.equal(new Set(labels).size, labels.length, `${v} ${b.phase_id}: ${labels.join(", ")}`);
      }
    }
  });
});
