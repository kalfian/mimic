/**
 * In-browser fake of the Mimic API (used when `NEXT_PUBLIC_API_MOCK=1`, see lib/README.md).
 *
 * Lifecycle is computed from wall time since upload (no background timers), and the scenario,
 * AI flag, pixel ratio and creation time are encoded in the job id, so `/jobs/<id>` survives a
 * page reload. Default run: queued → probing → … → generating → succeeded in ~6 s, then the
 * contract fixture (`sample-result.json`, synced via `pnpm sync:fixture`).
 *
 * Scenario selection (first match wins):
 *   1. `uploadVideo(..., { mockScenario })`
 *   2. page URL `?fail=<error code>` (upload-time or pipeline-time code)
 *   3. page URL `?scenario=<name>` (success | low_confidence | no_preview | forward_only | slow | flaky)
 * Interpreter status / connection check: page URL `?interpreter=<MockInterpreterScenario>`.
 * Re-run interpretation (`mockRerunInterpretation`): queued → interpreting → generating → succeeded
 * in ~2 s (AI on), then the fixture with the new labels. Kept in memory only (a reload during or
 * after a re-run shows the original result again).
 * Jump straight to a finished result: `/jobs/<fixture job_id>` (MOCK_SAMPLE_JOB_ID).
 *
 * Accounts (PLAN-auth §8.3, see mockAuth.ts): every job route needs a mock session (401 / 403
 * `password_change_required` otherwise). Jobs have owners (mockJobsRegistry.ts): users only see
 * their own jobs (another user's job → 404 `not_found`), admins see all. Unknown-but-well-formed
 * ids (e.g. after clearing storage) count as owned by the current user. `getHealth` returns
 * `interpreter: null, limits: null` without a full session; `checkInterpreter` is admin-only.
 */

import type { InterpreterCheck, InterpreterMode, UploadOptions, UploadProgress, UploadRequestOptions } from "../api";
import { ApiError } from "../errors";
import { bandFor, ROLE_LABELS } from "../format";
import type {
  ErrorCode,
  Health,
  JobCreated,
  JobList,
  JobOwner,
  JobState,
  JobStatus,
  KeyframeArtifact,
  MotionSpec,
  PixelRatioOption,
  ResultEnvelope,
  SpecWarning,
  Stage,
} from "../types";
import { STAGE_LABELS, STAGE_WINDOWS } from "../types";
import { dropMockSession, mockFailOnce, mockFailureFor, mockUserRef, optionalMockUser, requireMockAdmin, requireMockUser, resetMockAuth } from "./mockAuth";
import {
  allJobEntries,
  isDeletedJob,
  registerJob,
  registryEntry,
  resetMockJobsRegistry,
  unregisterJob,
} from "./mockJobsRegistry";
import { currentSearch, mockLatency, mockTiming, sleep } from "./mockStore";
import sampleResultJson from "./sample-result.json";

/* ---------- scenarios ---------- */

/** Rejected by `uploadVideo` itself (sync validation, HTTP 413/415/422). */
export const MOCK_UPLOAD_FAILURES = [
  "unsupported_format",
  "file_too_large",
  "too_long",
  "too_short",
  "decode_failed",
  "invalid_request",
] as const satisfies readonly ErrorCode[];

/** Job is accepted, then fails during processing (surfaced via `JobStatus.error`). */
export const MOCK_PIPELINE_FAILURES = [
  "no_motion_detected",
  "unsupported_motion",
  "no_stable_state",
  "internal_error",
  "interrupted",
] as const satisfies readonly ErrorCode[];

/**
 * Successful runs:
 * - `success`: the fixture as-is (card hover, fwd + rev, mixed confidence bands)
 * - `low_confidence`: low/uncertain transitions, unknown trigger, no cursor, AI fallback, VFR warnings
 * - `no_preview`: `artifacts.video_url = null` + `preview_unavailable` warning
 * - `forward_only`: reverse not recorded (single lane, `reverse_not_recorded` warning)
 * - `slow`: AI labeling stage takes ~25 s (long-running processing state)
 * - `flaky`: every 3rd status poll fails with a network error (exercises backoff / "reconnecting")
 */
export const MOCK_RESULT_SCENARIOS = ["success", "low_confidence", "no_preview", "forward_only", "slow", "flaky"] as const;

export type MockUploadFailure = (typeof MOCK_UPLOAD_FAILURES)[number];
export type MockPipelineFailure = (typeof MOCK_PIPELINE_FAILURES)[number];
export type MockResultScenario = (typeof MOCK_RESULT_SCENARIOS)[number];
export type MockScenario = MockResultScenario | MockUploadFailure | MockPipelineFailure;

export const MOCK_SCENARIOS: readonly MockScenario[] = [...MOCK_RESULT_SCENARIOS, ...MOCK_PIPELINE_FAILURES, ...MOCK_UPLOAD_FAILURES];

/** `?interpreter=` values for `getHealth` / `checkInterpreter`. */
export const MOCK_INTERPRETER_SCENARIOS = [
  "claude_cli",
  "ok",
  "ok_prompt_only",
  "unauthorized",
  "unreachable",
  "model_not_found",
  "no_image_support",
  "timeout",
  "not_configured",
] as const;
export type MockInterpreterScenario = (typeof MOCK_INTERPRETER_SCENARIOS)[number];

export function isMockScenario(value: unknown): value is MockScenario {
  return typeof value === "string" && (MOCK_SCENARIOS as readonly string[]).includes(value);
}

export function isMockInterpreterScenario(value: unknown): value is MockInterpreterScenario {
  return typeof value === "string" && (MOCK_INTERPRETER_SCENARIOS as readonly string[]).includes(value);
}

/** `?fail=<code>` or `?scenario=<name>` from a query string (default: the current page URL). */
export function readMockScenarioFromLocation(search: string = currentSearch()): MockScenario | null {
  const params = new URLSearchParams(search);
  const fail = params.get("fail");
  if (isMockScenario(fail)) return fail;
  const scenario = params.get("scenario");
  return isMockScenario(scenario) ? scenario : null;
}

/** `?interpreter=<scenario>` (default `claude_cli`: configured + available). */
export function readMockInterpreterScenario(search: string = currentSearch()): MockInterpreterScenario {
  const value = new URLSearchParams(search).get("interpreter");
  return isMockInterpreterScenario(value) ? value : "claude_cli";
}

/* ---------- config ---------- */

export interface MockConfig {
  /** Multiplies every simulated duration (stages, upload, latency). 0.01 makes tests fast. */
  timeScale: number;
  /** Simulated network latency per request, ms (before timeScale). */
  latencyMs: number;
}

// Shared with mockAuth.ts (same latency for every mock endpoint).
const config: MockConfig = mockTiming;

export function configureMock(patch: Partial<MockConfig>): MockConfig {
  Object.assign(config, patch);
  return { ...config };
}

/**
 * Forget all in-memory job state (tests: "simulate a reload"). Ids still decode and the persisted
 * registry (owners, deletions) and accounts stay; only blob video URLs and re-runs are lost.
 */
export function resetMock(): void {
  jobs.clear();
  results.clear();
}

/** Back to a fresh mock: seeded accounts (signed out), seeded jobs, no in-memory state. */
export function resetMockData(): void {
  resetMock();
  resetMockJobsRegistry();
  resetMockAuth();
}

/* ---------- helpers ---------- */

const SAMPLE = sampleResultJson as unknown as ResultEnvelope;

/** Job id of the fixture: `getJob` reports it as already succeeded. */
export const MOCK_SAMPLE_JOB_ID: string = SAMPLE.job_id;

const PIPELINE_MESSAGES: Record<MockPipelineFailure | MockUploadFailure, { status: number; message: string }> = {
  // Same wording as the backend (api/app/core/errors.py DEFAULT_MESSAGES / probe.py).
  unsupported_format: { status: 415, message: "Unsupported file type. Upload an MP4, MOV, M4V or WebM video." },
  file_too_large: { status: 413, message: "The file is larger than 200 MB. Trim it or export it at a lower resolution." },
  too_long: { status: 422, message: "The recording is 18.4 s long. Trim it to 15 seconds or less around the interaction." },
  too_short: {
    status: 422,
    message: "The recording is 0.32 s long; it must be at least 0.5 s. Wait about 1 second before interacting and keep recording until the animation has finished.",
  },
  decode_failed: { status: 422, message: "The video could not be decoded. Re-export it as an H.264 MP4 and upload it again." },
  invalid_request: { status: 422, message: "The request is invalid." },
  no_motion_detected: {
    status: 422,
    message: "No UI motion was found in the recording. Make sure the interaction and its animation happen on screen while recording.",
  },
  unsupported_motion: {
    status: 422,
    message: "The whole page moves (scrolling or a page transition), which is not supported yet. Record a single component interaction without scrolling.",
  },
  no_stable_state: {
    status: 422,
    message: "The UI is already moving when the recording starts. Start recording, wait about 1 second without touching anything, then interact.",
  },
  internal_error: { status: 500, message: "Something went wrong while analyzing the recording. Try again; if it keeps failing, try another recording." },
  interrupted: { status: 500, message: "Processing stopped because the server restarted. Upload the video again." },
};

/** Stage where each pipeline failure happens (the job keeps that stage when failed). */
const FAILURE_STAGE: Record<MockPipelineFailure, Stage> = {
  no_motion_detected: "scanning",
  unsupported_motion: "scanning",
  no_stable_state: "scanning",
  internal_error: "measuring",
  interrupted: "decoding",
};

interface MockJob {
  id: string;
  scenario: MockScenario;
  useInterpreter: boolean;
  pixelRatio: PixelRatioOption;
  createdAt: number;
  filename: string;
  videoUrl: string | null;
  polls: number;
  /** Wall time of the last `mockRerunInterpretation` (in memory only), null if never re-run. */
  rerunAt?: number | null;
  /** Reported as `JobStatus.owner` (null = legacy job). Omitted in pure tests → null. */
  owner?: JobOwner | null;
}

const jobs = new Map<string, MockJob>();
const results = new Map<string, ResultEnvelope>();

const ID_RE = /^mock-([a-z_]+)-([01])-(auto|[123])-([0-9a-z]+)$/;

function encodeId(job: Pick<MockJob, "scenario" | "useInterpreter" | "pixelRatio" | "createdAt">): string {
  return `mock-${job.scenario}-${job.useInterpreter ? 1 : 0}-${job.pixelRatio}-${job.createdAt.toString(36)}`;
}

function lookupJob(id: string): MockJob | null {
  if (isDeletedJob(id)) return null;
  const known = jobs.get(id);
  if (known) return known;
  const entry = registryEntry(id);
  let job: MockJob | null = null;
  if (id === MOCK_SAMPLE_JOB_ID) {
    job = {
      id,
      scenario: "success",
      useInterpreter: true,
      pixelRatio: "auto",
      createdAt: entry?.createdAt ?? 0,
      filename: entry?.filename ?? SAMPLE.spec.source.filename,
      videoUrl: null,
      polls: 0,
    };
  } else {
    const m = ID_RE.exec(id);
    if (m && isMockScenario(m[1])) {
      job = {
        id,
        scenario: m[1],
        useInterpreter: m[2] === "1",
        pixelRatio: m[3] as PixelRatioOption,
        createdAt: parseInt(m[4], 36),
        filename: entry?.filename ?? SAMPLE.spec.source.filename,
        videoUrl: null,
        polls: 0,
      };
    }
  }
  if (job) jobs.set(id, job);
  return job;
}

const latency = mockLatency;

function httpError(code: ErrorCode, status: number, message: string): ApiError {
  return ApiError.fromResponse(status, { error: { code, message } });
}

/* ---------- lifecycle ---------- */

function stagePlan(job: Pick<MockJob, "scenario" | "useInterpreter">): [Stage, number][] {
  const interpretMs = job.scenario === "slow" ? 25_000 : job.useInterpreter ? 1500 : 250;
  const plan: [Stage, number][] = [
    ["queued", 300],
    ["probing", 400],
    ["preview", 500],
    ["scanning", 900],
    ["decoding", 600],
    ["detecting_elements", 500],
    ["measuring", 1000],
    ["fitting", 400],
    ["interpreting", interpretMs],
    ["generating", 300],
  ];
  return plan.map(([s, ms]) => [s, ms * config.timeScale]);
}

function overallProgress(stage: Stage, fraction: number): number {
  const [a, b] = STAGE_WINDOWS[stage];
  return Number((a + (b - a) * Math.min(1, Math.max(0, fraction))).toFixed(4));
}

function rerunPlan(job: Pick<MockJob, "scenario" | "useInterpreter">): [Stage, number][] {
  const interpretMs = job.scenario === "slow" ? 25_000 : job.useInterpreter ? 1500 : 250;
  const plan: [Stage, number][] = [
    ["queued", 200],
    ["interpreting", interpretMs],
    ["generating", 300],
  ];
  return plan.map(([s, ms]) => [s, ms * config.timeScale]);
}

/** Pure: the status a job reports at wall time `now`. */
export function mockJobStatusAt(job: Omit<MockJob, "polls" | "videoUrl">, now: number): JobStatus {
  if (job.rerunAt != null && now >= job.rerunAt) return rerunStatusAt(job, job.rerunAt, now);
  const elapsed = Math.max(0, now - job.createdAt);
  const failure = (MOCK_PIPELINE_FAILURES as readonly string[]).includes(job.scenario) ? (job.scenario as MockPipelineFailure) : null;
  const base = {
    id: job.id,
    created_at: new Date(job.createdAt).toISOString(),
    options: { pixel_ratio: job.pixelRatio, use_interpreter: job.useInterpreter },
    owner: job.owner ?? null,
  };
  const src = SAMPLE.spec.source;
  const source = { filename: job.filename, width: src.width, height: src.height, fps: src.fps_nominal, duration_s: src.duration_ms / 1000 };

  let t = 0;
  for (const [stage, ms] of stagePlan(job)) {
    const failsHere = failure !== null && FAILURE_STAGE[failure] === stage;
    const end = failsHere ? t + ms * 0.5 : t + ms;
    if (failsHere && elapsed >= end) {
      return {
        ...base,
        status: "failed",
        stage,
        stage_label: STAGE_LABELS[stage],
        progress: overallProgress(stage, 0.5),
        error: { code: failure, message: PIPELINE_MESSAGES[failure].message },
        updated_at: new Date(job.createdAt + end).toISOString(),
        source,
      };
    }
    if (elapsed < end) {
      const fraction = ms > 0 ? (elapsed - t) / ms : 1;
      return {
        ...base,
        status: stage === "queued" ? "queued" : "processing",
        stage,
        stage_label: STAGE_LABELS[stage],
        progress: overallProgress(stage, fraction),
        error: null,
        updated_at: new Date(job.createdAt + elapsed).toISOString(),
        source: stage === "queued" || stage === "probing" ? null : source,
      };
    }
    t = end;
  }
  return {
    ...base,
    status: "succeeded",
    stage: "done",
    stage_label: STAGE_LABELS.done,
    progress: 1,
    error: null,
    updated_at: new Date(job.createdAt + t).toISOString(),
    source,
  };
}

/** Status during / after an interpretation re-run that started at `rerunAt`. */
function rerunStatusAt(job: Omit<MockJob, "polls" | "videoUrl">, rerunAt: number, now: number): JobStatus {
  const src = SAMPLE.spec.source;
  const base = {
    id: job.id,
    created_at: new Date(job.createdAt).toISOString(),
    options: { pixel_ratio: job.pixelRatio, use_interpreter: job.useInterpreter },
    owner: job.owner ?? null,
    error: null,
    source: { filename: job.filename, width: src.width, height: src.height, fps: src.fps_nominal, duration_s: src.duration_ms / 1000 },
  };
  const elapsed = now - rerunAt;
  let t = 0;
  for (const [stage, ms] of rerunPlan(job)) {
    if (elapsed < t + ms) {
      // queued → 0, then the backend's real windows (interpreting starts at 0.78)
      const progress = stage === "queued" ? 0 : overallProgress(stage, ms > 0 ? (elapsed - t) / ms : 1);
      return {
        ...base,
        status: stage === "queued" ? "queued" : "processing",
        stage,
        stage_label: STAGE_LABELS[stage],
        progress,
        updated_at: new Date(now).toISOString(),
      };
    }
    t += ms;
  }
  return { ...base, status: "succeeded", stage: "done", stage_label: STAGE_LABELS.done, progress: 1, updated_at: new Date(rerunAt + t).toISOString() };
}

/* ---------- result building ---------- */

function svgDataUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string);
}

/** Placeholder keyframe: neutral canvas, element boxes interpolated between state A (0) and B (1). */
function keyframeSvg(spec: MotionSpec, title: string, progress: number, annotate: boolean): string {
  const w = Math.round(spec.source.width / spec.source.pixel_ratio);
  const h = Math.round(spec.source.height / spec.source.pixel_ratio);
  const lerp = (a: number, b: number) => a + (b - a) * progress;
  const boxes = spec.elements
    .map((el) => {
      const a = el.bbox_initial;
      const b = el.bbox_active;
      const x = lerp(a.x, b.x), y = lerp(a.y, b.y), bw = lerp(a.w, b.w), bh = lerp(a.h, b.h);
      const label = annotate
        ? `<text x="${x + 6}" y="${y + 20}" font-size="16" font-family="monospace" fill="#C2410C">${esc(el.id)} ${esc(el.label)}</text>`
        : "";
      const stroke = annotate ? `stroke="#C2410C" stroke-width="2"` : `stroke="#A1A1AA" stroke-width="1"`;
      return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="6" fill="#FFFFFF" fill-opacity="0.6" ${stroke}/>${label}`;
    })
    .join("");
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
      `<rect width="100%" height="100%" fill="#F4F4F5"/>${boxes}` +
      `<text x="24" y="${h - 24}" font-size="20" font-family="monospace" fill="#52525B">mock · ${esc(title)}</text></svg>`,
  );
}

/** Placeholder element crop: A | B side by side. */
function elementSvg(spec: MotionSpec, elementId: string): string {
  const el = spec.elements.find((e) => e.id === elementId);
  const label = el ? `${el.id} ${el.label}` : elementId;
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">` +
      `<rect width="640" height="320" fill="#F4F4F5"/><line x1="320" y1="0" x2="320" y2="320" stroke="#D4D4D8"/>` +
      `<rect x="60" y="80" width="200" height="160" rx="8" fill="#FFFFFF" stroke="#A1A1AA"/>` +
      `<rect x="380" y="66" width="200" height="160" rx="8" fill="#FFFFFF" stroke="#C2410C"/>` +
      `<text x="16" y="300" font-size="16" font-family="monospace" fill="#52525B">mock · ${esc(label)} · A | B</text></svg>`,
  );
}

function placeholderKeyframes(spec: MotionSpec, keyframes: KeyframeArtifact[]): KeyframeArtifact[] {
  const progressFor: Partial<Record<KeyframeArtifact["kind"], number>> = {
    state_a: 0,
    annotated_a: 0,
    mid_25: 0.25,
    mid_50: 0.5,
    mid_75: 0.75,
    state_b: 1,
    annotated_b: 1,
  };
  return keyframes.map((k) => ({
    ...k,
    url:
      k.kind === "element"
        ? elementSvg(spec, k.element_id ?? "")
        : keyframeSvg(spec, k.name, progressFor[k.kind] ?? 0, k.kind === "annotated_a" || k.kind === "annotated_b"),
  }));
}

function warn(spec: MotionSpec, w: SpecWarning) {
  if (!spec.warnings.some((x) => x.code === w.code)) spec.warnings.push(w);
}

function applyHeuristicLabels(spec: MotionSpec) {
  for (const el of spec.elements) {
    el.label = `${ROLE_LABELS[el.role]} (${el.id})`;
    el.label_source = "heuristic";
  }
  const target = spec.elements.find((e) => e.id === spec.interaction.target_element_id);
  if (target) spec.interaction.target_label = target.label;
  spec.interaction.target_description = null;
  spec.structure = spec.elements.filter((e) => e.parent_id === spec.interaction.target_element_id).map((e) => e.role);
}

function applyLowConfidence(spec: MotionSpec) {
  const scaleConf = (v: number) => Number((v * 0.55).toFixed(2));
  for (const t of spec.transitions) {
    const c = t.confidence;
    const overall = scaleConf(c.overall);
    t.confidence = { overall, value: scaleConf(c.value), timing: scaleConf(c.timing), easing: scaleConf(c.easing), band: bandFor(overall) };
  }
  spec.interaction.type_confidence = { value: 0.4, band: bandFor(0.4) };
  spec.interaction.trigger = {
    kind: "unknown",
    reverse_kind: "unknown",
    description: "The trigger could not be determined because the cursor is not visible.",
    confidence: { value: 0.3, band: bandFor(0.3) },
  };
  spec.cursor = { visible: false, confidence: 0, events: [] };
  for (const s of spec.segments) s.trigger_event_ms = null;
  spec.source.fps_nominal = 30;
  spec.source.fps_effective = 29.4;
  spec.source.is_vfr = true;
  spec.source.timing_resolution_ms = 34;
  spec.interpretation = { provider: "claude_cli", model: "sonnet", status: "timeout", duration_ms: 120_000 };
  applyHeuristicLabels(spec);
  warn(spec, { code: "cursor_not_visible", severity: "warn", message: "No cursor was found, so the trigger is a guess." });
  warn(spec, { code: "vfr_source", severity: "info", message: "Variable frame rate recording; timing is less precise." });
  warn(spec, { code: "timestamps_estimated", severity: "warn", message: "Frame timestamps were estimated; timing confidence lowered." });
  warn(spec, { code: "interpretation_fallback", severity: "info", message: "AI labeling timed out; labels are heuristic." });
}

function applyForwardOnly(spec: MotionSpec) {
  const fwd = spec.segments.find((s) => s.kind === "forward");
  spec.segments = spec.segments.filter((s) => s.kind === "forward");
  const keep = new Set(spec.segments.map((s) => s.id));
  spec.transitions = spec.transitions.filter((t) => keep.has(t.segment_id));
  spec.relationships = spec.relationships.filter((r) => keep.has(r.segment_id));
  spec.interaction.direction = "forward";
  spec.interaction.total_duration_ms = { forward: spec.interaction.total_duration_ms.forward, reverse: null };
  spec.interaction.trigger.reverse_kind = "unknown";
  if (fwd) spec.cursor.events = spec.cursor.events.filter((e) => e.t_ms <= fwd.end_ms);
  warn(spec, { code: "reverse_not_recorded", severity: "info", message: "Only the forward motion was recorded; assume the reverse mirrors it." });
}

const MOCK_VIDEO_URL: string | null = process.env.NEXT_PUBLIC_API_MOCK_VIDEO_URL || null;

/** Pure-ish: the fixture adapted to a mock job (exported for tests / storybook-style use). */
export function buildMockResult(
  job: Pick<MockJob, "id" | "scenario" | "useInterpreter" | "pixelRatio" | "filename" | "videoUrl">,
): ResultEnvelope {
  const result = structuredClone(SAMPLE);
  const spec = result.spec;
  result.job_id = job.id;
  spec.job_id = job.id;
  spec.source.filename = job.filename;

  if (job.pixelRatio !== "auto") {
    spec.source.pixel_ratio = Number(job.pixelRatio) as 1 | 2 | 3;
    spec.source.pixel_ratio_source = "user";
    spec.warnings = spec.warnings.filter((w) => w.code !== "pixel_ratio_assumed");
  }
  if (!job.useInterpreter) {
    spec.interpretation = { provider: "none", model: null, status: "disabled", duration_ms: null };
    applyHeuristicLabels(spec);
  }
  if (job.scenario === "low_confidence") applyLowConfidence(spec);
  if (job.scenario === "forward_only") applyForwardOnly(spec);

  result.artifacts = {
    video_url: job.scenario === "no_preview" ? null : (job.videoUrl ?? MOCK_VIDEO_URL),
    keyframes: placeholderKeyframes(spec, result.artifacts.keyframes),
  };
  if (job.scenario === "no_preview") {
    warn(spec, { code: "preview_unavailable", severity: "warn", message: "The browser preview could not be created; showing keyframes instead." });
  }
  return result;
}

/* ---------- ownership (PLAN-auth U3) ---------- */

const NOT_FOUND_MESSAGE = "This job does not exist. It may have been deleted.";

/**
 * The job if the signed-in mock user may see it (admin: any; user: own), with `owner` filled.
 * Throws 401 / 403 `password_change_required` / 404 `not_found` like the real `get_job`.
 */
function visibleJob(id: string): MockJob {
  const me = requireMockUser();
  const job = lookupJob(id);
  if (!job) throw httpError("not_found", 404, NOT_FOUND_MESSAGE);
  const entry = registryEntry(id);
  // Unregistered but well-formed ids (lenient, e.g. storage cleared) belong to the current user.
  const ownerId = entry ? entry.ownerId : me.id;
  if (me.role !== "admin" && ownerId !== me.id) throw httpError("not_found", 404, NOT_FOUND_MESSAGE);
  job.owner = mockUserRef(ownerId);
  return job;
}

/* ---------- endpoints (same signatures as lib/api.ts) ---------- */

export async function mockGetHealth(signal?: AbortSignal): Promise<Health> {
  await latency(signal);
  const check = interpreterCheckFor(readMockInterpreterScenario());
  // Anonymous / forced password change: public subset only (PLAN-auth A12).
  if (!optionalMockUser()) return { status: "ok", version: "0.1.0-mock", ffmpeg: true, interpreter: null, limits: null };
  return {
    status: "ok",
    version: "0.1.0-mock",
    ffmpeg: true,
    interpreter: {
      mode: check.mode,
      available: check.mode !== "none",
      model: check.model,
    },
    limits: { max_upload_mb: 200, max_duration_s: 15, min_duration_s: 0.5, duration_tolerance_s: 0.5 },
  };
}

function interpreterCheckFor(scenario: MockInterpreterScenario): InterpreterCheck {
  const base = {
    mode: "openai_compat" as InterpreterMode,
    ok: false,
    latency_ms: null as number | null,
    model: "gpt-4o-mini" as string | null,
    model_listed: null as boolean | null,
    supports_images: null as boolean | null,
    structured_output: null as InterpreterCheck["structured_output"],
    error: null as InterpreterCheck["error"],
    checked_at: new Date().toISOString(),
  };
  switch (scenario) {
    case "claude_cli":
      return { ...base, mode: "claude_cli", ok: true, latency_ms: 2400, model: "sonnet", supports_images: true, structured_output: "json_schema" };
    case "ok":
      return { ...base, ok: true, latency_ms: 420, model_listed: true, supports_images: true, structured_output: "json_schema" };
    case "ok_prompt_only":
      return { ...base, ok: true, latency_ms: 910, model_listed: true, supports_images: true, structured_output: "prompt_only" };
    case "unauthorized":
      return { ...base, latency_ms: 180, error: { code: "unauthorized", message: "The endpoint returned HTTP 401: the API key was rejected." } };
    case "unreachable":
      return { ...base, error: { code: "unreachable", message: "Connection refused by the configured endpoint." } };
    case "model_not_found":
      return { ...base, latency_ms: 240, model_listed: false, error: { code: "model_not_found", message: "Model 'gpt-4o-mini' is not in the endpoint's model list." } };
    case "no_image_support":
      return { ...base, latency_ms: 650, model_listed: true, supports_images: false, error: { code: "no_image_support", message: "The model rejected an image input." } };
    case "timeout":
      return { ...base, latency_ms: 15_000, error: { code: "timeout", message: "No answer within 15 s." } };
    case "not_configured":
      return { ...base, mode: "none", model: null, error: { code: "not_configured", message: "MIMIC_INTERPRETER is 'none'." } };
  }
}

/** Admin only, like the real route (403 `forbidden` for users). */
export async function mockCheckInterpreter(signal?: AbortSignal, scenario: MockInterpreterScenario = readMockInterpreterScenario()): Promise<InterpreterCheck> {
  requireMockAdmin();
  const check = interpreterCheckFor(scenario);
  // Simulate the round trip (capped so the "timeout" scenario doesn't actually wait 15 s).
  await sleep(Math.min(2000, Math.max(300, check.latency_ms ?? 800)) * config.timeScale, signal);
  return { ...check, checked_at: new Date().toISOString() };
}

export async function mockGetJob(id: string, signal?: AbortSignal): Promise<JobStatus> {
  await latency(signal);
  const job = visibleJob(id);
  job.polls += 1;
  if (job.scenario === "flaky" && job.polls % 3 === 0) throw ApiError.network();
  return mockJobStatusAt(job, Date.now());
}

export async function mockGetResult(id: string, signal?: AbortSignal): Promise<ResultEnvelope> {
  await latency(signal);
  const job = visibleJob(id);
  if (mockJobStatusAt(job, Date.now()).status !== "succeeded") throw httpError("not_ready", 409, "The result is not ready yet.");
  let result = results.get(id);
  if (!result) {
    result = buildMockResult(job);
    results.set(id, result);
  }
  return structuredClone(result);
}

/** Same contract as `rerunInterpretation` in lib/api.ts (409 while running / for failed jobs). */
export async function mockRerunInterpretation(id: string, useInterpreter = true, signal?: AbortSignal): Promise<JobStatus> {
  await latency(signal);
  const job = visibleJob(id);
  const now = Date.now();
  const current = mockJobStatusAt(job, now);
  if (current.status === "queued" || current.status === "processing") {
    throw httpError("already_running", 409, "This job is still being processed. Wait until it finishes.");
  }
  if (current.status === "failed") throw httpError("not_ready", 409, "The job failed; there is no result to re-label.");
  job.useInterpreter = useInterpreter;
  job.rerunAt = now;
  results.delete(id);
  return mockJobStatusAt(job, now);
}

export async function mockUploadVideo(
  file: File,
  options: UploadOptions,
  onProgress?: (progress: UploadProgress) => void,
  request: UploadRequestOptions = {},
): Promise<JobCreated> {
  const { signal } = request;
  const me = requireMockUser();
  const scenario: MockScenario = request.mockScenario ?? readMockScenarioFromLocation() ?? "success";
  const uploadFailure = (MOCK_UPLOAD_FAILURES as readonly string[]).includes(scenario) ? (scenario as MockUploadFailure) : null;

  // Simulated upload: ~20 MB/s, clamped to 0.6–2.5 s, in 10 progress events.
  const total = file.size;
  const uploadMs = Math.min(2500, Math.max(600, (total / (20 * 1024 * 1024)) * 1000)) * config.timeScale;
  const steps = 10;
  // The server aborts an oversized stream early; other validation happens after the body arrived.
  const stopAt = uploadFailure === "file_too_large" ? 4 : steps;
  for (let i = 1; i <= stopAt; i++) {
    await sleep(uploadMs / steps, signal);
    const loaded = Math.round((total * i) / steps);
    onProgress?.({ loaded, total, fraction: total > 0 ? loaded / total : 1 });
  }
  if (uploadFailure) {
    const { status, message } = PIPELINE_MESSAGES[uploadFailure];
    throw httpError(uploadFailure, status, message);
  }

  const createdAt = Date.now();
  const job: MockJob = {
    id: "",
    scenario,
    useInterpreter: options.useInterpreter,
    pixelRatio: options.pixelRatio,
    createdAt,
    filename: file.name,
    videoUrl: typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : null,
    polls: 0,
    owner: mockUserRef(me.id),
  };
  job.id = encodeId(job);
  jobs.set(job.id, job);
  registerJob({ id: job.id, ownerId: me.id, filename: file.name, createdAt });
  return { id: job.id, status: "queued", created_at: new Date(createdAt).toISOString() };
}

export interface MockListJobsQuery {
  owner?: string;
  status?: JobState;
  limit?: number;
  cursor?: string | null;
}

const CURSOR_RE = /^o:(\d+)$/;

/**
 * Same contract as `listJobs` in lib/api.ts: newest first, `next_cursor` (opaque here: an offset),
 * users see only their own jobs, admins all (`owner` filters; ids other than self need admin).
 * `?fail=network_error` / `?fail=unauthenticated` fail the first call per page load.
 */
export async function mockListJobs(query: MockListJobsQuery = {}, signal?: AbortSignal): Promise<JobList> {
  await latency(signal);
  const scenario = mockFailOnce("listJobs");
  if (scenario === "unauthenticated") throw dropMockSession();
  const me = requireMockUser();
  if (scenario === "network_error") throw ApiError.network();

  const limit = query.limit ?? 50;
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) throw httpError("invalid_request", 422, "limit must be between 1 and 200.");
  let offset = 0;
  if (query.cursor) {
    const m = CURSOR_RE.exec(query.cursor);
    if (!m) throw httpError("invalid_request", 422, "The cursor is invalid.");
    offset = Number(m[1]);
  }
  let ownerFilter: string | null | undefined; // undefined = no filter
  if (query.owner === "me") ownerFilter = me.id;
  else if (query.owner) {
    if (me.role !== "admin" && query.owner !== me.id) throw httpError("forbidden", 403, "You don't have permission to do this.");
    ownerFilter = query.owner;
  } else if (me.role !== "admin") ownerFilter = me.id;

  const now = Date.now();
  const rows = allJobEntries()
    .filter((e) => ownerFilter === undefined || e.ownerId === ownerFilter)
    .sort((a, b) => b.createdAt - a.createdAt || (a.id < b.id ? 1 : a.id > b.id ? -1 : 0))
    .flatMap((e) => {
      const job = lookupJob(e.id);
      if (!job) return [];
      job.owner = mockUserRef(e.ownerId);
      return [mockJobStatusAt(job, now)];
    })
    .filter((j) => !query.status || j.status === query.status);
  const items = rows.slice(offset, offset + limit);
  const next = offset + limit < rows.length ? `o:${offset + limit}` : null;
  return { items, next_cursor: next };
}

/** Same contract as `deleteJob` in lib/api.ts (204; other user's job → 404). Works while running. */
export async function mockDeleteJob(id: string, signal?: AbortSignal): Promise<void> {
  await latency(signal);
  if (mockFailureFor("deleteJob") === "not_found") {
    requireMockUser();
    throw httpError("not_found", 404, NOT_FOUND_MESSAGE);
  }
  const job = visibleJob(id);
  if (job.videoUrl?.startsWith("blob:") && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(job.videoUrl);
  jobs.delete(id);
  results.delete(id);
  unregisterJob(id);
}
