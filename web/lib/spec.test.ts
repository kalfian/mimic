import assert from "node:assert/strict";
import { test } from "node:test";

import { continuousResult } from "./mock/continuousVariants";
import sampleResult from "./mock/sample-result.json";
import { continuousOf, isContinuousSpec, jsOutput, normalizeResult, specMode } from "./spec";
import type { MotionSpec, ResultEnvelope } from "./types";

const transition = () => structuredClone(sampleResult as unknown as ResultEnvelope);

test("specMode: missing mode (stored 0.1 results) is transition", () => {
  assert.equal(specMode({}), "transition");
  assert.equal(specMode({ mode: null }), "transition");
  assert.equal(specMode({ mode: "transition" }), "transition");
  assert.equal(specMode({ mode: "continuous" }), "continuous");

  const legacy = transition().spec as Partial<MotionSpec>;
  delete legacy.mode;
  legacy.schema_version = "0.1";
  assert.equal(specMode(legacy), "transition");
  assert.equal(isContinuousSpec(legacy as MotionSpec), false);
});

test("isContinuousSpec / continuousOf need both mode and the section", () => {
  const c = continuousResult().spec;
  assert.equal(specMode(c), "continuous");
  assert.equal(isContinuousSpec(c), true);
  assert.equal(continuousOf(c)?.axis, "x");

  const t = transition().spec;
  assert.equal(isContinuousSpec(t), false);
  assert.equal(continuousOf(t), null);

  const malformed = structuredClone(c);
  malformed.continuous = null;
  assert.equal(isContinuousSpec(malformed), false, "mode without a section falls back to transition view");
});

test("jsOutput: JS tab only when outputs.js is a string", () => {
  assert.equal(typeof jsOutput(continuousResult().outputs), "string");
  const t = transition().outputs;
  assert.equal("js" in t, false, "transition fixture has no js key");
  assert.equal(jsOutput(t), null);
  assert.equal(jsOutput({ ...t, js: null }), null);
});

test("normalizeResult fills a missing mode and keeps present ones", () => {
  const legacy = transition();
  delete (legacy.spec as Partial<MotionSpec>).mode;
  const fixed = normalizeResult(legacy);
  assert.equal(fixed.spec.mode, "transition");
  assert.equal("mode" in legacy.spec, false); // input untouched

  const t = transition();
  assert.equal(normalizeResult(t), t);
  const c = continuousResult();
  assert.equal(normalizeResult(c), c);
  assert.equal(normalizeResult(c).spec.mode, "continuous");
});
