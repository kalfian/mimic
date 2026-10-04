/**
 * Mode helpers for the IR root (PLAN-continuous §5, P0 notes). Pure, no React.
 *
 * Always read the mode through `specMode(spec)`, never `spec.mode` directly: results stored by
 * schema 0.1 are served verbatim by `GET /api/jobs/{id}/result` and have **no** `mode` key (a
 * re-run of the interpretation re-dumps them as 0.2 and adds it). Missing / null mode means
 * "transition".
 */

import type { ContinuousMotion, MotionSpec, Outputs, ResultEnvelope, SpecMode } from "./types";

/** Anything with an optional `mode` (a full `MotionSpec`, or a stored 0.1 spec without the key). */
export type SpecLike = { mode?: SpecMode | null };

/** The spec's mode; a missing `mode` (stored 0.1 results) is `"transition"`. */
export function specMode(spec: SpecLike): SpecMode {
  return spec.mode === "continuous" ? "continuous" : "transition";
}

/**
 * The envelope with `spec.mode` filled in (`specMode`), so code that reads `spec.mode` directly
 * also sees `"transition"` for stored 0.1 results. Returns the same object when nothing changes.
 */
export function normalizeResult(env: ResultEnvelope): ResultEnvelope {
  const spec = env.spec as MotionSpec & SpecLike;
  if (spec.mode === "transition" || spec.mode === "continuous") return env;
  return { ...env, spec: { ...env.spec, mode: specMode(spec) } };
}

/** A spec in continuous mode with its `continuous` section present. */
export type ContinuousSpec = MotionSpec & { mode: "continuous"; continuous: ContinuousMotion };

/**
 * True when the spec should render as continuous motion (velocity chart + behaviour summary).
 * Requires both `mode === "continuous"` and a `continuous` section; the backend guarantees they
 * go together, so a mismatch is a malformed result and falls back to the transition view.
 */
export function isContinuousSpec(spec: MotionSpec): spec is ContinuousSpec {
  return specMode(spec) === "continuous" && spec.continuous != null;
}

/** The `continuous` section, or null in transition mode. */
export function continuousOf(spec: MotionSpec): ContinuousMotion | null {
  return isContinuousSpec(spec) ? spec.continuous : null;
}

/**
 * The JS driver text, or null when there is none. Show the **JS** tab only when this is non-null
 * (continuous results only; the key is absent in transition results and in stored 0.1 results).
 */
export function jsOutput(outputs: Outputs): string | null {
  return typeof outputs.js === "string" ? outputs.js : null;
}
