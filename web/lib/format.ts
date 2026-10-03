/**
 * Display formatting for IR values, confidences, stages and error codes. Pure, no React.
 *
 * Rounding follows the backend phrasing rules (PLAN §9.1) so numbers on screen match the
 * generated Technical / Prompt text: px |v|>=3 → 1 px, else 0.5 px; scale 2 decimals;
 * opacity nearest 0.05; ms rounded to 10 when shown as "~".
 */

import type { ApiErrorCode } from "./errors";
import type { InterpreterCheck, InterpreterCheckErrorCode, InterpreterMode, StructuredOutputMode } from "./api";
import type {
  Confidence,
  ConfidenceBand,
  Direction,
  Easing,
  ElementKind,
  InteractionType,
  InterpretationStatus,
  JobStatus,
  Pattern,
  Property,
  ReverseTriggerKind,
  Role,
  SegmentId,
  Shadow,
  Stage,
  TriggerKind,
  Value,
  WarningCode,
} from "./types";
import { BAND_HIGH_MIN, BAND_MEDIUM_MIN, STAGE_LABELS, STAGE_ORDER, UNCERTAIN_BELOW } from "./types";

/* ---------- numbers ---------- */

/**
 * Round to a multiple of `step`, half away from zero, without float noise (0.1 + 0.2 style).
 * Matches the backend generators (`phrasing.round_step`: Decimal ROUND_HALF_UP on the value's
 * shortest repr), so the UI never shows -3px where the generated text says -2px or vice versa.
 */
export function roundTo(value: number, step: number): number {
  const q = Number((value / step).toPrecision(12)); // drop binary noise: 2.4999999999 → 2.5
  const r = Math.sign(q) * Math.round(Math.abs(q)) * step;
  return Number(r.toFixed(6)) || 0; // no -0
}

/** Trim to at most `digits` decimals, no trailing zeros: 1.50 → "1.5", 2.00 → "2". Avoids "-0". */
export function formatNumber(value: number, digits = 2): string {
  const s = Number(value.toFixed(digits)).toString();
  return s === "-0" ? "0" : s;
}

/** Milliseconds. `approx` rounds to 10 ms and prefixes "~" (UI convention for estimates). */
export function formatMs(ms: number, options: { approx?: boolean; unit?: boolean } = {}): string {
  const { approx = false, unit = true } = options;
  const v = approx ? roundTo(ms, 10) : Math.round(ms);
  return `${approx ? "~" : ""}${v}${unit ? " ms" : ""}`;
}

/** Video time for axes / playhead: 1180 → "1.18 s", 0 → "0 s". */
export function formatSeconds(ms: number, digits = 2): string {
  return `${formatNumber(ms / 1000, digits)} s`;
}

/** Elapsed wall time for the processing screen: 75_000 → "1:15". */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** CSS px per PLAN §9.1: |v| >= 3 → whole px, else nearest 0.5. `signed` adds "+" for positives. */
export function formatPx(value: number, options: { signed?: boolean; unit?: boolean } = {}): string {
  const { signed = false, unit = true } = options;
  const v = Math.abs(value) >= 3 ? roundTo(value, 1) : roundTo(value, 0.5);
  const s = formatNumber(v, 1);
  return `${signed && v > 0 ? "+" : ""}${s}${unit ? "px" : ""}`;
}

/** Scale factor, 2 decimals: 1.06 → "1.06", 1 → "1.00". */
export function formatScale(value: number): string {
  return roundTo(value, 0.01).toFixed(2);
}

/** Opacity, nearest 0.05: 0.47 → "0.45". */
export function formatOpacity(value: number): string {
  return formatNumber(roundTo(value, 0.05), 2);
}

/** 0..1 fraction → percent: 0.853 → "85%". */
export function formatPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Scale delta as a percent change: 1.06 → "+6%", 0.96 → "-4%". */
export function formatScaleChange(from: number, to: number): string {
  const pct = Math.round((to / from - 1) * 100);
  return `${pct > 0 ? "+" : ""}${pct}%`;
}

export function formatColor(hex: string): string {
  return hex.toUpperCase();
}

/** `Shadow` → CSS box-shadow text; `null` → "none". Alpha keeps 2 decimals. */
export function formatShadow(shadow: Shadow | null): string {
  if (!shadow) return "none";
  const [r, g, b, a] = shadow.rgba;
  const px = (n: number) => (n === 0 ? "0" : `${formatNumber(n, 1)}px`);
  return `${px(shadow.x)} ${px(shadow.y)} ${px(shadow.blur)} ${px(shadow.spread)} rgba(${r}, ${g}, ${b}, ${formatNumber(a, 2)})`;
}

/**
 * Any IR `Value` as display text. `property` refines ratio values (scale vs opacity).
 * px → "-8px", scale → "1.06", opacity → "0.45", color → "#5252FF", shadow → CSS text, text → as-is.
 */
export function formatValue(value: Value, property?: Property): string {
  switch (value.kind) {
    case "px":
      return formatPx(value.number);
    case "ratio":
      return property === "opacity" ? formatOpacity(value.number) : formatScale(value.number);
    case "color":
      return formatColor(value.color);
    case "shadow":
      return formatShadow(value.shadow);
    case "text":
      return value.text;
  }
}

/** "0px → -8px" style change text for a transition. */
export function formatChange(t: { from: Value; to: Value; property: Property }): string {
  return `${formatValue(t.from, t.property)} → ${formatValue(t.to, t.property)}`;
}

/* ---------- easing ---------- */

/** `cubic-bezier(0.16, 1, 0.3, 1)` */
export function formatCubicBezier(b: Easing["cubic_bezier"]): string {
  return `cubic-bezier(${b.map((n) => formatNumber(n, 3)).join(", ")})`;
}

/**
 * Short easing text: the CSS keyword when the fit is one, else the bezier.
 * `withNearest` appends "(close to expoOut)" for non-keyword fits.
 */
export function formatEasing(easing: Easing, options: { withNearest?: boolean } = {}): string {
  if (easing.keyword) return easing.keyword;
  const base = formatCubicBezier(easing.cubic_bezier);
  return options.withNearest ? `${base} (close to ${easing.nearest_named})` : base;
}

/** CSS `transition-timing-function` value for a fit (keyword or cubic-bezier). */
export function easingToCss(easing: Easing): string {
  return easing.keyword ?? formatCubicBezier(easing.cubic_bezier);
}

/* ---------- confidence ---------- */

/** Band for a 0..1 score; same thresholds as the backend `band_for` (0.8 / 0.5). */
export function bandFor(value: number): ConfidenceBand {
  if (value >= BAND_HIGH_MIN) return "high";
  if (value >= BAND_MEDIUM_MIN) return "medium";
  return "low";
}

export const BAND_LABELS: Record<ConfidenceBand, string> = { high: "High", medium: "Medium", low: "Low" };

/** Transitions below 0.3 overall are "uncertain observations" (excluded from the main text). */
export function isUncertain(overall: number): boolean {
  return overall < UNCERTAIN_BELOW;
}

/** `{value: 0.85, band: "high"}` or `0.85` → "High · 85%" (`withPercent: false` → "High"). */
export function formatConfidence(c: Confidence | number, options: { withPercent?: boolean } = {}): string {
  const { withPercent = true } = options;
  const value = typeof c === "number" ? c : c.value;
  const band = typeof c === "number" ? bandFor(c) : c.band;
  return withPercent ? `${BAND_LABELS[band]} · ${formatPercent(value)}` : BAND_LABELS[band];
}

/* ---------- stages ---------- */

/** Stages shown as steps in the processing list (excludes `queued` and `done`). */
export const PIPELINE_STAGES: readonly Stage[] = STAGE_ORDER.filter((s) => s !== "queued" && s !== "done");

/**
 * Stage label. Without AI labeling the `interpreting` stage still runs (deterministic fallback),
 * so pass `useInterpreter: false` to drop the "(AI)" suffix.
 */
export function stageLabel(stage: Stage, options: { useInterpreter?: boolean } = {}): string {
  if (stage === "interpreting" && options.useInterpreter === false) return "Labeling elements";
  return STAGE_LABELS[stage];
}

export type StageStepState = "done" | "current" | "pending" | "failed";

export interface StageStep {
  stage: Stage;
  label: string;
  state: StageStepState;
}

/**
 * Processing checklist from a job: stages before the current one are `done`, the current one is
 * `current` (or `failed` when the job failed there), later ones `pending`. All `done` on success.
 */
export function stageSteps(job: Pick<JobStatus, "status" | "stage" | "options"> | null): StageStep[] {
  const useInterpreter = job?.options.use_interpreter;
  const current = job ? STAGE_ORDER.indexOf(job.stage) : 0;
  return PIPELINE_STAGES.map((stage) => {
    const idx = STAGE_ORDER.indexOf(stage);
    let state: StageStepState;
    if (job?.status === "succeeded") state = "done";
    else if (idx < current) state = "done";
    else if (idx === current) state = job?.status === "failed" ? "failed" : "current";
    else state = "pending";
    return { stage, label: stageLabel(stage, { useInterpreter }), state };
  });
}

/* ---------- enum labels ---------- */

export const PROPERTY_LABELS: Record<Property, string> = {
  translateX: "Translate X",
  translateY: "Translate Y",
  scale: "Scale",
  scaleX: "Scale X",
  scaleY: "Scale Y",
  opacity: "Opacity",
  color: "Text color",
  "background-color": "Background",
  "box-shadow": "Shadow",
  "border-radius": "Radius",
  height: "Height",
  content: "Content",
};

export const SEGMENT_LABELS: Record<SegmentId, string> = {
  fwd: "Forward",
  rev: "Reverse",
  rt_in: "Press",
  rt_out: "Release",
};

export const INTERACTION_TYPE_LABELS: Record<InteractionType, string> = {
  hover: "Hover",
  click: "Click",
  press: "Press",
  expand_collapse: "Expand / collapse",
  dropdown: "Dropdown",
  modal: "Modal",
  unknown: "Unknown",
};

export const TRIGGER_LABELS: Record<TriggerKind, string> = {
  pointer_enter: "Pointer enters",
  click: "Click",
  unknown: "Unknown",
};

export const REVERSE_TRIGGER_LABELS: Record<ReverseTriggerKind, string> = {
  pointer_leave: "Pointer leaves",
  click: "Click again",
  none: "None",
  unknown: "Unknown",
};

export const DIRECTION_LABELS: Record<Direction, string> = {
  forward: "Forward only",
  forward_reverse: "Forward + reverse",
  round_trip: "Round trip",
};

export const PATTERN_LABELS: Record<Pattern, string> = {
  card: "Card",
  button: "Button",
  menu: "Menu",
  accordion: "Accordion",
  modal: "Modal",
  generic: "Generic",
};

export const ELEMENT_KIND_LABELS: Record<ElementKind, string> = {
  transform: "Moves / scales",
  photometric: "Color / opacity",
  appear: "Appears",
  disappear: "Disappears",
  backdrop: "Backdrop",
  resize: "Resizes",
  content_change: "Content changes",
};

export const ROLE_LABELS: Record<Role, string> = {
  card: "Card",
  button: "Button",
  image: "Image",
  icon: "Icon",
  text: "Text",
  title: "Title",
  label: "Label",
  container: "Container",
  dropdown_menu: "Dropdown menu",
  menu_item: "Menu item",
  modal_panel: "Modal panel",
  backdrop: "Backdrop",
  list_item: "List item",
  link: "Link",
  input: "Input",
  badge: "Badge",
  accordion_panel: "Accordion panel",
  other: "Element",
};

export const INTERPRETATION_STATUS_LABELS: Record<InterpretationStatus, string> = {
  ok: "AI labels",
  fallback: "Heuristic labels (AI unavailable)",
  timeout: "Heuristic labels (AI timed out)",
  error: "Heuristic labels (AI failed)",
  disabled: "Heuristic labels (AI off)",
};

export const WARNING_LABELS: Record<WarningCode, string> = {
  pixel_ratio_assumed: "Display scale assumed",
  timestamps_estimated: "Frame timestamps estimated",
  vfr_source: "Variable frame rate",
  cursor_not_visible: "Cursor not visible",
  extra_segments_ignored: "Extra interactions ignored",
  elements_truncated: "Too many elements",
  short_stable_state: "Short pause around the interaction",
  not_settled: "Motion did not settle",
  rotation_detected: "Rotation detected",
  interpretation_fallback: "AI labeling fell back",
  interpretation_disagrees: "AI disagrees with measurement",
  low_fps_source: "Low frame rate",
  preview_unavailable: "Preview unavailable",
  reverse_not_recorded: "Reverse not recorded",
  frames_subsampled: "Reduced frame rate (long motion)",
};

/** Pixel ratio as shown to users: 2 → "2x (Retina)". */
export function formatPixelRatio(ratio: number): string {
  return ratio === 2 ? "2x (Retina)" : `${ratio}x`;
}

/* ---------- interpreter (Layer B) ---------- */

export const INTERPRETER_MODE_LABELS: Record<InterpreterMode, string> = {
  claude_cli: "Claude Code (local CLI)",
  openai_compat: "OpenAI-compatible endpoint",
  none: "Disabled",
};

export const STRUCTURED_OUTPUT_LABELS: Record<StructuredOutputMode, string> = {
  json_schema: "JSON schema",
  json_object: "JSON mode",
  prompt_only: "Prompt only (less reliable)",
};

export const INTERPRETER_CHECK_ERROR_MESSAGES: Record<InterpreterCheckErrorCode, string> = {
  not_configured: "No AI interpreter is configured on the server.",
  unauthorized: "The server's API key was rejected by the endpoint.",
  unreachable: "The AI endpoint could not be reached.",
  timeout: "The AI endpoint did not answer in time.",
  model_not_found: "The configured model was not found on the endpoint.",
  no_image_support: "The configured model does not accept images.",
  bad_response: "The AI endpoint returned an unexpected response.",
};

/** One-line summary of a connection check, e.g. "Connected · gpt-4o · 420 ms". */
export function formatInterpreterCheck(check: InterpreterCheck): string {
  if (!check.ok) {
    const code = check.error?.code;
    return code ? INTERPRETER_CHECK_ERROR_MESSAGES[code] : (check.error?.message ?? "Connection check failed.");
  }
  const parts = ["Connected"];
  if (check.model) parts.push(check.model);
  if (check.latency_ms != null) parts.push(formatMs(check.latency_ms));
  return parts.join(" · ");
}

/* ---------- errors ---------- */

export interface ErrorDescription {
  title: string;
  message: string;
  /** Actionable tips (recording guidelines etc.); may be empty. */
  guidance: string[];
}

/** PRD §27 recording guidelines, phrased as fixes. */
const RECORDING_TIPS = [
  "Start recording before you interact.",
  "Wait about one second, then perform one interaction.",
  "Keep the cursor visible.",
  "Wait until the animation has finished before you stop recording.",
];

const RE_EXPORT = "Re-export it as an H.264 MP4 and upload it again.";

/**
 * One entry per code. Tone: title says what happened, message gives the reason in one sentence,
 * guidance lists what to do. The backend's `DEFAULT_MESSAGES` (api/app/core/errors.py) follow the
 * same wording; `describeError(code, serverMessage)` prefers the server's more specific message.
 */
const ERROR_DESCRIPTIONS: Record<ApiErrorCode, ErrorDescription> = {
  unsupported_format: {
    title: "Unsupported file type",
    message: "Upload an MP4, MOV, M4V or WebM video.",
    guidance: ["Screen recorders usually save MP4 or MOV; export to one of those if yours does not."],
  },
  file_too_large: {
    title: "File too large",
    message: "The file is larger than the upload limit.",
    guidance: ["Trim the recording to just the interaction.", "Or export it at a lower resolution (keep 60 fps if you can)."],
  },
  too_long: {
    title: "Recording too long",
    message: "The recording is longer than the maximum length.",
    guidance: ["Trim it to about one second before the interaction until the animation has settled."],
  },
  too_short: {
    title: "Recording too short",
    message: "The recording is too short to analyze.",
    guidance: [RECORDING_TIPS[0], RECORDING_TIPS[1], RECORDING_TIPS[3]],
  },
  decode_failed: { title: "Video could not be read", message: "The video could not be decoded.", guidance: [RE_EXPORT] },
  invalid_request: { title: "Request rejected", message: "The server rejected the request.", guidance: ["Reload the page and try again."] },
  no_motion_detected: {
    title: "No motion found",
    message: "No UI motion was found in the recording.",
    guidance: ["Make sure the interaction visibly changes the UI on screen.", ...RECORDING_TIPS],
  },
  unsupported_motion: {
    title: "Page motion is not supported",
    message: "Scrolling and page-level transitions are not supported yet.",
    guidance: ["Record a single component interaction (hover, click, dropdown, modal…).", "Keep the page still: no scrolling, no navigation."],
  },
  no_stable_state: {
    title: "The UI is moving from the start",
    message: "The UI is already moving when the recording starts.",
    guidance: [
      "Start recording, then wait about one second without touching anything.",
      "Pause looping animations, carousels or videos on the page.",
      "Then perform one interaction and wait until it has finished.",
    ],
  },
  internal_error: {
    title: "Analysis failed",
    message: "Something went wrong while analyzing the recording.",
    guidance: ["Try again.", "If it keeps failing, try another recording and check the API log."],
  },
  interrupted: { title: "Processing interrupted", message: "The server restarted while processing this recording.", guidance: ["Upload the video again."] },
  not_found: {
    title: "Analysis not found",
    message: "This analysis does not exist; it may have been deleted.",
    guidance: ["Upload the video again."],
  },
  not_ready: { title: "Result not ready", message: "The result is not ready yet.", guidance: ["Wait until processing has finished."] },
  already_running: { title: "Still processing", message: "This analysis is still running.", guidance: ["Wait until it finishes, then try again."] },
  network_error: {
    title: "Server unreachable",
    message: "Could not reach the analysis server.",
    guidance: ["Check that the API is running (make dev-api) and that NEXT_PUBLIC_API_BASE_URL points to it."],
  },
  aborted: { title: "Cancelled", message: "The request was cancelled.", guidance: [] },
  invalid_response: {
    title: "Unexpected response",
    message: "The server returned an unexpected response.",
    guidance: ["Check that NEXT_PUBLIC_API_BASE_URL points to the Mimic API, then reload."],
  },
};

/**
 * Human title/message/guidance for an error code. Pass the server `message` as `serverMessage`
 * to prefer its specifics (e.g. "Display scale ...", actual duration) over the generic text.
 */
export function describeError(code: ApiErrorCode, serverMessage?: string | null): ErrorDescription {
  const base = ERROR_DESCRIPTIONS[code] ?? ERROR_DESCRIPTIONS.internal_error;
  return serverMessage ? { ...base, message: serverMessage } : base;
}
