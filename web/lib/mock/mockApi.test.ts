// Proves mock mode drives the full job lifecycle through the real poller (DoD for Track W-lib).
import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

import type { JobClient } from "../api";
import { ApiError } from "../errors";
import { pollJob, type JobPollState } from "../jobPoller";
import { buildTimeline } from "../timeline";
import type { Stage } from "../types";
import { specMode, isContinuousSpec, jsOutput } from "../spec";
import { velocityModelOf } from "../velocity";
import { CONTINUOUS_FIXTURE_JOB_ID } from "./mockJobsRegistry";
import {
  buildMockResult,
  CONTINUOUS_VARIANTS,
  configureMock,
  MOCK_CONTINUOUS_SAMPLE_JOB_ID,
  MOCK_RESULT_SCENARIOS,
  MOCK_SAMPLE_JOB_ID,
  mockCheckInterpreter,
  mockGetJob,
  mockGetResult,
  mockJobStatusAt,
  mockRerunInterpretation,
  mockUploadVideo,
  readMockScenarioFromLocation,
  resetMock,
  resetMockData,
  type MockScenario,
} from "./mockApi";
import { mockSignInAs } from "./mockAuth";

const client: JobClient = { getJob: mockGetJob, getResult: mockGetResult };
const fast = { intervalMs: 5, slowIntervalMs: 10, backoffBaseMs: 5, backoffMaxMs: 20 };
const video = () => new File([new Uint8Array(1024)], "hover.mov", { type: "video/quicktime" });

async function runScenario(scenario: MockScenario, useInterpreter = false) {
  const progress: number[] = [];
  const created = await mockUploadVideo(video(), { pixelRatio: "auto", useInterpreter }, (p) => progress.push(p.fraction ?? -1), {
    mockScenario: scenario,
  });
  const updates: JobPollState[] = [];
  const final = await pollJob(created.id, { ...fast, client, onUpdate: (s) => updates.push(s) });
  return { created, progress, updates, final };
}

beforeEach(() => {
  resetMockData();
  mockSignInAs("admin"); // job routes need a session; admin also may run the interpreter check
  configureMock({ timeScale: 0.01, latencyMs: 20 });
});

describe("mock lifecycle", () => {
  it("success: queued → every stage → succeeded with the fixture result", async () => {
    const { created, progress, updates, final } = await runScenario("success", true);

    assert.equal(created.status, "queued");
    assert.equal(progress.at(-1), 1);
    assert.equal(final.phase, "succeeded");
    assert.equal(final.error, null);
    assert.ok(final.result);
    assert.equal(final.result.job_id, created.id);
    assert.equal(final.result.spec.interpretation.status, "ok");
    assert.equal(final.result.spec.source.filename, "hover.mov");
    assert.ok(final.result.artifacts.keyframes.every((k) => k.url.startsWith("data:image/svg+xml")));
    assert.ok(final.result.artifacts.video_url?.startsWith("blob:"));

    // Phases are monotonic and progress never goes backwards.
    const phases = [...new Set(updates.map((u) => u.phase))];
    assert.deepEqual(phases, ["polling", "fetching_result", "succeeded"]);
    const progresses = updates.flatMap((u) => (u.job ? [u.job.progress] : []));
    for (let i = 1; i < progresses.length; i++) assert.ok(progresses[i] >= progresses[i - 1]);
    assert.equal(updates.at(-1)?.job?.stage, "done");

    // Polling every ~25 ms over ~65 ms of simulated stages can skip short ones (the exact stage
    // sequence is asserted on mockJobStatusAt below); several distinct stages must be observed.
    const seen = new Set(updates.map((u) => u.job?.stage));
    assert.ok(seen.size >= 4, `expected several stages, saw ${[...seen].join(",")}`);
  });

  it("status function walks through all stages in order", () => {
    const job = { id: "x", scenario: "success" as const, useInterpreter: true, pixelRatio: "auto" as const, createdAt: 0, filename: "a.mp4" };
    configureMock({ timeScale: 1 });
    const stages: Stage[] = [];
    for (let t = 0; t <= 8000; t += 50) {
      const s = mockJobStatusAt(job, t);
      if (stages.at(-1) !== s.stage) stages.push(s.stage);
    }
    assert.deepEqual(stages, [
      "queued",
      "probing",
      "preview",
      "scanning",
      "decoding",
      "detecting_elements",
      "measuring",
      "fitting",
      "interpreting",
      "generating",
      "done",
    ]);
    assert.equal(mockJobStatusAt(job, 0).status, "queued");
    assert.equal(mockJobStatusAt(job, 1000).status, "processing");
    assert.equal(mockJobStatusAt(job, 100).source, null); // not probed yet
    assert.equal(mockJobStatusAt(job, 1000).source?.filename, "a.mp4");
    assert.equal(mockJobStatusAt(job, 8000).status, "succeeded");
  });

  it("AI off: heuristic labels and interpretation disabled", async () => {
    const { final } = await runScenario("success", false);
    const spec = final.result!.spec;
    assert.equal(spec.interpretation.status, "disabled");
    assert.equal(spec.interpretation.provider, "none");
    assert.deepEqual(
      spec.elements.map((e) => e.label),
      ["Card (e1)", "Image (e2)", "Title (e3)", "Icon (e4)"],
    );
  });

  for (const code of ["no_motion_detected", "unsupported_motion", "no_stable_state", "continuous_motion_unsupported", "internal_error", "interrupted"] as const) {
    it(`pipeline failure: ${code}`, async () => {
      const { final } = await runScenario(code);
      assert.equal(final.phase, "failed");
      assert.equal(final.job?.status, "failed");
      assert.ok(final.error instanceof ApiError);
      assert.equal(final.error.code, code);
      assert.equal(final.error.origin, "job");
      assert.equal(final.result, null);
    });
  }

  for (const [code, status] of [
    ["unsupported_format", 415],
    ["file_too_large", 413],
    ["too_long", 422],
    ["decode_failed", 422],
  ] as const) {
    it(`upload failure: ${code} → HTTP ${status}`, async () => {
      await assert.rejects(
        mockUploadVideo(video(), { pixelRatio: "auto", useInterpreter: false }, undefined, { mockScenario: code }),
        (err: unknown) => err instanceof ApiError && err.code === code && err.status === status && err.origin === "http",
      );
    });
  }

  it("upload can be aborted", async () => {
    const controller = new AbortController();
    const p = mockUploadVideo(video(), { pixelRatio: "auto", useInterpreter: false }, () => controller.abort(), {
      signal: controller.signal,
    });
    await assert.rejects(p, (err: unknown) => err instanceof ApiError && err.code === "aborted");
  });

  it("low_confidence: low bands, uncertain rows, unknown trigger, fallback labels", async () => {
    const { final } = await runScenario("low_confidence", true);
    const spec = final.result!.spec;
    assert.ok(spec.transitions.every((t) => t.confidence.band !== "high"));
    for (const t of spec.transitions) {
      const expected = t.confidence.overall >= 0.8 ? "high" : t.confidence.overall >= 0.5 ? "medium" : "low";
      assert.equal(t.confidence.band, expected);
    }
    assert.equal(spec.interaction.trigger.kind, "unknown");
    assert.equal(spec.cursor.visible, false);
    assert.equal(spec.interpretation.status, "timeout");
    assert.ok(spec.warnings.some((w) => w.code === "cursor_not_visible"));
    assert.ok(buildTimeline(spec).uncertainCount >= 2);
  });

  it("no_preview and forward_only variants", async () => {
    const a = await runScenario("no_preview");
    assert.equal(a.final.result!.artifacts.video_url, null);
    assert.ok(a.final.result!.spec.warnings.some((w) => w.code === "preview_unavailable"));

    const b = await runScenario("forward_only");
    const spec = b.final.result!.spec;
    assert.equal(spec.interaction.direction, "forward");
    assert.deepEqual(
      spec.segments.map((s) => s.id),
      ["fwd"],
    );
    assert.equal(buildTimeline(spec).lanes.length, 1);
  });

  it("flaky: transient network errors are retried and the job still succeeds", async () => {
    const { updates, final } = await runScenario("flaky");
    assert.equal(final.phase, "succeeded");
    assert.ok(updates.some((u) => u.retrying > 0 && u.lastTransientError?.code === "network_error"));
  });

  it("unknown id → not_found (final, no retries); fixture id → already succeeded", async () => {
    const missing = await pollJob("0123456789abcdef0123456789abcdef", { ...fast, client });
    assert.equal(missing.phase, "failed");
    assert.equal(missing.error?.code, "not_found");
    assert.equal(missing.error?.status, 404);

    const sample = await pollJob(MOCK_SAMPLE_JOB_ID, { ...fast, client });
    assert.equal(sample.phase, "succeeded");
    assert.equal(sample.result?.spec.interpretation.status, "ok");
  });

  it("result before success is 409 not_ready", async () => {
    configureMock({ timeScale: 1, latencyMs: 0 });
    const id = `mock-success-0-2-${Date.now().toString(36)}`; // freshly "created" job, decoded from its id
    await assert.rejects(mockGetResult(id), (err: unknown) => err instanceof ApiError && err.code === "not_ready" && err.status === 409);
  });

  it("job ids survive a reload (decoded without in-memory state)", async () => {
    const { created } = await runScenario("success");
    resetMock();
    const job = await mockGetJob(created.id);
    assert.equal(job.status, "succeeded");
    assert.equal(job.options.use_interpreter, false);
  });

  it("reads ?fail= / ?scenario= from the URL", () => {
    assert.equal(readMockScenarioFromLocation("?fail=too_long"), "too_long");
    assert.equal(readMockScenarioFromLocation("?scenario=low_confidence"), "low_confidence");
    assert.equal(readMockScenarioFromLocation("?fail=bogus"), null);
    assert.equal(readMockScenarioFromLocation(""), null);
  });

  it("interpreter check scenarios", async () => {
    const ok = await mockCheckInterpreter(undefined, "ok");
    assert.equal(ok.ok, true);
    assert.equal(ok.structured_output, "json_schema");
    const prompt = await mockCheckInterpreter(undefined, "ok_prompt_only");
    assert.equal(prompt.structured_output, "prompt_only");
    for (const code of ["unauthorized", "unreachable", "model_not_found", "not_configured"] as const) {
      const r = await mockCheckInterpreter(undefined, code);
      assert.equal(r.ok, false);
      assert.equal(r.error?.code, code);
    }
  });
  it("re-run interpretation: only interpreting + generating, then the AI-labeled result", () => {
    configureMock({ timeScale: 1 });
    const job = {
      id: "x", scenario: "success" as const, useInterpreter: true, pixelRatio: "auto" as const,
      createdAt: 0, filename: "a.mp4", rerunAt: 10_000,
    };
    assert.equal(mockJobStatusAt(job, 9_000).status, "succeeded"); // before the re-run
    const stages: Stage[] = [];
    for (let t = 10_000; t <= 13_000; t += 25) {
      const s = mockJobStatusAt(job, t);
      if (stages.at(-1) !== s.stage) stages.push(s.stage);
    }
    assert.deepEqual(stages, ["queued", "interpreting", "generating", "done"]);
  });

  it("re-run rejects running and failed jobs", async () => {
    const running = await mockUploadVideo(video(), { pixelRatio: "auto", useInterpreter: false }, undefined, { mockScenario: "slow" });
    await assert.rejects(mockRerunInterpretation(running.id), (e: unknown) => e instanceof ApiError && e.code === "already_running");
    const { created } = await runScenario("no_motion_detected");
    await assert.rejects(mockRerunInterpretation(created.id), (e: unknown) => e instanceof ApiError && e.code === "not_ready");
    await assert.rejects(mockRerunInterpretation("nope"), (e: unknown) => e instanceof ApiError && e.code === "not_found");
  });
});

describe("mock continuous mode (PLAN-continuous §7.2)", () => {
  it("?scenario=continuous runs the full lifecycle and ends with the continuous fixture", async () => {
    assert.equal(readMockScenarioFromLocation("?scenario=continuous"), "continuous");
    const { created, final } = await runScenario("continuous", true);
    assert.equal(final.phase, "succeeded");
    const result = final.result!;
    assert.equal(result.job_id, created.id);
    assert.equal(specMode(result.spec), "continuous");
    assert.ok(isContinuousSpec(result.spec));
    assert.equal(typeof jsOutput(result.outputs), "string", "JS tab present");
    assert.equal(result.spec.source.filename, "hover.mov");
    assert.ok(result.artifacts.keyframes.length > 0 && result.artifacts.keyframes.every((k) => k.kind === "phase" && k.url.startsWith("data:image/svg+xml")));
    assert.equal(buildTimeline(result.spec).isEmpty, true, "no transition rows in continuous mode");
  });

  it("every continuous variant is a result scenario and builds a velocity model", () => {
    for (const v of CONTINUOUS_VARIANTS) {
      assert.ok((MOCK_RESULT_SCENARIOS as readonly string[]).includes(v), v);
      assert.equal(readMockScenarioFromLocation(`?scenario=${v}`), v);
      for (const useInterpreter of [true, false]) {
        const r = buildMockResult({ id: `mock-${v}-0-auto-1`, scenario: v, useInterpreter, pixelRatio: "auto", filename: "c.mov", videoUrl: null });
        assert.equal(specMode(r.spec), "continuous", v);
        assert.ok(velocityModelOf(r.spec), v);
        assert.ok(jsOutput(r.outputs) !== null, v);
      }
    }
    const t = buildMockResult({ id: "mock-success-0-auto-1", scenario: "success", useInterpreter: true, pixelRatio: "auto", filename: "a.mov", videoUrl: null });
    assert.equal(specMode(t.spec), "transition");
    assert.equal(jsOutput(t.outputs), null, "no JS tab in transition mode");
  });

  it("variants reach each chart state", () => {
    const model = (v: (typeof CONTINUOUS_VARIANTS)[number]) =>
      velocityModelOf(buildMockResult({ id: "x", scenario: v, useInterpreter: false, pixelRatio: "auto", filename: "c.mov", videoUrl: null }).spec)!;
    assert.ok(model("continuous_degraded").degradedCount > 0);
    assert.ok(model("continuous_uncertain").uncertainCount >= 2);
    assert.equal(model("continuous_autoplay").bands.length, 1);
    assert.equal(model("continuous_vertical").axis, "y");
    assert.equal(model("continuous_no_samples").hasSamples, false);
  });

  it("continuous fixture id is seeded (owned by `user`) and already succeeded", async () => {
    assert.equal(CONTINUOUS_FIXTURE_JOB_ID, MOCK_CONTINUOUS_SAMPLE_JOB_ID, "registry seed must match the fixture");
    const done = await pollJob(MOCK_CONTINUOUS_SAMPLE_JOB_ID, { ...fast, client });
    assert.equal(done.phase, "succeeded");
    assert.equal(done.job?.owner?.username, "user");
    assert.equal(specMode(done.result!.spec), "continuous");
    assert.equal(done.result!.spec.interpretation.status, "disabled", "fixture was made with AI labeling off");
  });

  it("re-run works on a continuous job", async () => {
    const { created } = await runScenario("continuous_vertical");
    const status = await mockRerunInterpretation(created.id, true);
    assert.equal(status.status, "queued");
    const final = await pollJob(created.id, { ...fast, client });
    assert.equal(final.phase, "succeeded");
    assert.equal(final.result!.spec.continuous?.axis, "y");
  });
});
