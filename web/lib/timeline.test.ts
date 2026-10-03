import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import { activeRowsAt, axisTicks, buildTimeline, findRow, fractionToTime, laneAt, rowSpan, timeToFraction } from "./timeline";
import type { MotionSpec, ResultEnvelope } from "./types";

const fixture = JSON.parse(readFileSync(new URL("./mock/sample-result.json", import.meta.url), "utf8")) as ResultEnvelope;
const spec = (): MotionSpec => structuredClone(fixture.spec);

describe("buildTimeline (fixture: card hover fwd + rev)", () => {
  const model = buildTimeline(spec());

  it("has one lane per segment, in segment order", () => {
    assert.deepEqual(
      model.lanes.map((l) => [l.segment, l.kind, l.label]),
      [
        ["fwd", "forward", "Forward"],
        ["rev", "reverse", "Reverse"],
      ],
    );
    assert.equal(model.rowCount, 12);
    assert.equal(model.isEmpty, false);
  });

  it("orders tracks by element hierarchy, then property order", () => {
    assert.deepEqual(
      model.tracks.map((t) => t.key),
      ["e1:translateY", "e1:box-shadow", "e2:scale", "e3:color", "e4:translateX", "e4:opacity"],
    );
    assert.deepEqual(
      model.tracks.map((t) => t.depth),
      [0, 0, 1, 1, 1, 1],
    );
    assert.equal(model.tracks[0].is_target, true);
    assert.equal(model.tracks[2].is_target, false);
    // Both lanes use the shared track order.
    for (const lane of model.lanes) {
      assert.deepEqual(
        lane.rows.map((r) => r.key),
        model.tracks.map((t) => t.key),
      );
    }
  });

  it("maps transition timing and confidence onto rows", () => {
    const t3 = findRow(model, "t3");
    assert.ok(t3);
    assert.equal(t3.element_label, "Product image");
    assert.equal(t3.property_label, "Scale");
    assert.equal(t3.start_ms, 1210);
    assert.equal(t3.end_ms, 1210 + 320);
    assert.equal(t3.delay_ms, 30);
    assert.equal(t3.confidence_band, "medium");
    assert.equal(t3.uncertain, false);
    assert.equal(t3.segment_id, "fwd");
  });

  it("pads lane windows around rows and segment markers", () => {
    const fwd = model.lanes[0];
    // rows 1180..1530, trigger at 1120 → span 410 → pad max(40, 40) = 40
    assert.deepEqual(fwd.window, { start_ms: 1080, end_ms: 1570 });
    assert.deepEqual(
      fwd.markers.map((m) => [m.kind, m.t_ms]),
      [
        ["trigger", 1120],
        ["onset", 1180],
        ["settle", 1530],
      ],
    );
  });

  it("global window covers lanes and cursor events, within the video duration", () => {
    assert.equal(model.window.start_ms, 620); // cursor move_start
    assert.equal(model.window.end_ms, model.lanes[1].window.end_ms);
    assert.ok(model.window.end_ms <= fixture.spec.source.duration_ms);
    assert.deepEqual(
      model.markers.map((m) => m.kind),
      ["cursor_move_start", "cursor_enter", "cursor_stationary", "cursor_move_start", "cursor_leave"],
    );
  });
});

describe("buildTimeline edge cases", () => {
  it("flags transitions below 0.3 overall as uncertain, and can hide them", () => {
    const s = spec();
    s.transitions[1].confidence = { ...s.transitions[1].confidence, overall: 0.2, band: "low" }; // t2 box-shadow fwd
    const all = buildTimeline(s);
    assert.equal(findRow(all, "t2")?.uncertain, true);
    assert.equal(all.uncertainCount, 1);

    const hidden = buildTimeline(s, { includeUncertain: false });
    assert.equal(findRow(hidden, "t2"), null);
    assert.equal(hidden.rowCount, 11);
    assert.equal(hidden.uncertainCount, 1);
    // e1:box-shadow still has a track because the reverse transition t8 is not uncertain.
    assert.ok(hidden.tracks.some((t) => t.key === "e1:box-shadow"));
  });

  it("is empty when nothing is above threshold", () => {
    const s = spec();
    for (const t of s.transitions) t.confidence = { ...t.confidence, overall: 0.1, band: "low" };
    const model = buildTimeline(s, { includeUncertain: false });
    assert.equal(model.isEmpty, true);
    assert.equal(model.tracks.length, 0);
    assert.equal(model.lanes.length, 2);
  });

  it("handles a forward-only spec", () => {
    const s = spec();
    s.segments = s.segments.filter((x) => x.id === "fwd");
    s.transitions = s.transitions.filter((t) => t.segment_id === "fwd");
    const model = buildTimeline(s);
    assert.equal(model.lanes.length, 1);
    assert.equal(model.rowCount, 6);
  });

  it("respects an explicit padding", () => {
    const model = buildTimeline(spec(), { paddingMs: 0 });
    assert.deepEqual(model.lanes[0].window, { start_ms: 1120, end_ms: 1530 });
  });

  it("is deterministic", () => {
    assert.deepEqual(buildTimeline(spec()), buildTimeline(spec()));
  });
});

describe("timeline geometry helpers", () => {
  const w = { start_ms: 1000, end_ms: 2000 };

  it("converts between time and fraction (clamped)", () => {
    assert.equal(timeToFraction(1250, w), 0.25);
    assert.equal(timeToFraction(500, w), 0);
    assert.equal(timeToFraction(3000, w), 1);
    assert.equal(fractionToTime(0.5, w), 1500);
    const span = rowSpan({ start_ms: 1200, end_ms: 1500 }, w);
    assert.equal(span.left, 0.2);
    assert.ok(Math.abs(span.width - 0.3) < 1e-9);
  });

  it("produces nice axis ticks", () => {
    assert.deepEqual(axisTicks({ start_ms: 1100, end_ms: 1600 }, 5), [1100, 1200, 1300, 1400, 1500, 1600]);
    assert.deepEqual(axisTicks({ start_ms: 1080, end_ms: 1570 }, 5), [1100, 1200, 1300, 1400, 1500]);
    assert.deepEqual(axisTicks({ start_ms: 0, end_ms: 0 }), []);
  });

  it("finds active rows and the lane under the playhead", () => {
    const model = buildTimeline(spec());
    assert.deepEqual(
      activeRowsAt(model, 1450).map((r) => r.transition_id),
      ["t1", "t2", "t3", "t5", "t6"],
    );
    assert.equal(laneAt(model, 2900)?.segment, "rev");
    assert.equal(laneAt(model, 2000), null);
  });
});
