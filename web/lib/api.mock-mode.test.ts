// End-to-end through the public client with NEXT_PUBLIC_API_MOCK=1: upload → poll → result.
// node --test runs each file in its own process, so setting the env before importing is safe.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

process.env.NEXT_PUBLIC_API_MOCK = "1";

describe("api.ts in mock mode", () => {
  it("drives the full lifecycle via uploadVideo + pollJob (default client)", async () => {
    const api = await import("./api");
    const { pollJob } = await import("./jobPoller");
    const { configureMock } = await import("./mock/mockApi");
    configureMock({ timeScale: 0.01, latencyMs: 10 });

    assert.equal(api.IS_MOCK_API, true);
    assert.deepEqual(api.DEFAULT_UPLOAD_OPTIONS, { pixelRatio: "auto", useInterpreter: false });

    const health = await api.getHealth();
    assert.equal(health.status, "ok");

    const fractions: number[] = [];
    const file = new File([new Uint8Array(2048)], "card.mp4", { type: "video/mp4" });
    const created = await api.uploadVideo(file, {}, (p) => fractions.push(p.fraction ?? -1));
    assert.equal(created.status, "queued");
    assert.equal(fractions.at(-1), 1);

    const stages: string[] = [];
    const final = await pollJob(created.id, {
      intervalMs: 5,
      onUpdate: (s) => {
        if (s.job && stages.at(-1) !== s.job.stage) stages.push(s.job.stage);
      },
    });
    assert.equal(final.phase, "succeeded");
    assert.equal(final.result?.job_id, created.id);
    assert.equal(final.result?.spec.interpretation.status, "disabled"); // AI is opt-in
    assert.equal(stages.at(-1), "done");

    // Mock asset URLs are already browser-loadable (data:/blob:), assetUrl leaves them alone.
    const kf = final.result!.artifacts.keyframes[0];
    assert.equal(api.assetUrl(kf.url), kf.url);
    assert.equal(api.assetUrl(null), null);

    const check = await api.checkInterpreter();
    assert.equal(check.ok, true);

    // Re-run interpretation: AI labels replace the heuristic ones, the job goes back through
    // interpreting → generating → done, and a second click while it runs is rejected.
    const queued = await api.rerunInterpretation(created.id);
    assert.equal(queued.status, "queued");
    assert.equal(queued.options.use_interpreter, true);
    await assert.rejects(api.rerunInterpretation(created.id), (e: unknown) => e instanceof api.ApiError && e.code === "already_running" && e.status === 409);
    const rerunStages: string[] = [];
    const again = await pollJob(created.id, {
      intervalMs: 2,
      onUpdate: (s) => {
        if (s.job && rerunStages.at(-1) !== s.job.stage) rerunStages.push(s.job.stage);
      },
    });
    assert.equal(again.phase, "succeeded");
    assert.equal(again.result?.spec.interpretation.status, "ok");
    assert.ok(!rerunStages.includes("scanning"), `re-run must not measure again: ${rerunStages.join(",")}`);
    assert.equal(rerunStages.at(-1), "done");
  });
});
