/**
 * Rounding parity with the backend generators (api/app/generate/phrasing.py: Decimal ROUND_HALF_UP
 * on the shortest repr). Expected strings were produced by phrasing.px / scale_num / opacity_num
 * / ms_int, so the UI and the generated spec text never disagree on a rounded number.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { formatMs, formatOpacity, formatPx, formatScale } from "./format";

test("formatPx rounds half away from zero like the backend", () => {
  const cases: [number, string][] = [
    [2.25, "2.5px"], [-2.25, "-2.5px"], [3.5, "4px"], [-3.5, "-4px"], [-8, "-8px"],
    [0.24, "0px"], [-0.26, "-0.5px"], [7.4999, "7px"],
  ];
  for (const [v, want] of cases) assert.equal(formatPx(v), want, `formatPx(${v})`);
});

test("formatScale / formatOpacity / formatMs match the backend", () => {
  for (const [v, want] of [[1.005, "1.01"], [0.995, "1.00"], [1.06, "1.06"], [0.955, "0.96"]] as const) {
    assert.equal(formatScale(v), want, `formatScale(${v})`);
  }
  for (const [v, want] of [[0.425, "0.45"], [0.475, "0.5"], [0.97, "0.95"]] as const) {
    assert.equal(formatOpacity(v), want, `formatOpacity(${v})`);
  }
  for (const [v, want] of [[275, "~280 ms"], [285, "~290 ms"], [-15, "~-20 ms"]] as const) {
    assert.equal(formatMs(v, { approx: true }), want, `formatMs(${v})`);
  }
});
