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
    const { mockSignInAs } = await import("./mock/mockAuth");
    configureMock({ timeScale: 0.01, latencyMs: 10 });
    mockSignInAs("admin");

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

  it("U3 ownership: users see only their own jobs, admins see all; delete; health subset; admin-only check", async () => {
    const api = await import("./api");
    const { configureMock, MOCK_SAMPLE_JOB_ID, resetMockData } = await import("./mock/mockApi");
    const { mockSignInAs } = await import("./mock/mockAuth");
    const { FIXTURE_JOB_ID, MOCK_USER_IDS } = await import("./mock/mockJobsRegistry");
    assert.equal(FIXTURE_JOB_ID, MOCK_SAMPLE_JOB_ID, "registry seed must match the synced fixture (pnpm sync:fixture)");
    resetMockData();
    configureMock({ timeScale: 0.01, latencyMs: 0 });
    const notFound = (e: unknown) => e instanceof api.ApiError && e.code === "not_found" && e.status === 404;

    // anonymous: health has no details, job routes are 401
    const anon = await api.getHealth();
    assert.equal(anon.interpreter, null);
    assert.equal(anon.limits, null);
    await assert.rejects(api.listJobs(), (e: unknown) => e instanceof api.ApiError && e.code === "unauthenticated");

    // `user` owns the fixture job; alice doesn't see it
    mockSignInAs("user");
    assert.ok((await api.getHealth()).interpreter);
    const own = await api.listJobs();
    assert.ok(own.items.every((j) => j.owner?.id === MOCK_USER_IDS.user));
    assert.ok(own.items.some((j) => j.id === MOCK_SAMPLE_JOB_ID));
    assert.equal((await api.getJob(MOCK_SAMPLE_JOB_ID)).owner?.username, "user");
    await assert.rejects(api.checkInterpreter(), (e: unknown) => e instanceof api.ApiError && e.code === "forbidden" && e.status === 403);
    await assert.rejects(api.listJobs({ owner: MOCK_USER_IDS.alice }), (e: unknown) => e instanceof api.ApiError && e.code === "forbidden");

    mockSignInAs("alice");
    await assert.rejects(api.getJob(MOCK_SAMPLE_JOB_ID), notFound);
    await assert.rejects(api.getResult(MOCK_SAMPLE_JOB_ID), notFound);
    await assert.rejects(api.deleteJob(MOCK_SAMPLE_JOB_ID), notFound);
    const file = new File([new Uint8Array(64)], "mine.mp4", { type: "video/mp4" });
    const created = await api.uploadVideo(file);
    const mine = await api.listJobs({ owner: "me" });
    assert.equal(mine.items[0].id, created.id); // newest first
    assert.equal(mine.items[0].owner?.username, "alice");
    assert.equal(mine.items[0].source === null || mine.items[0].source.filename === "mine.mp4", true);

    // admin: everything incl. the legacy (owner: null) job, owner filter, pagination
    mockSignInAs("admin");
    const all = await api.listJobs();
    assert.ok(all.items.some((j) => j.owner === null), "legacy job visible to admins");
    assert.ok(all.items.some((j) => j.id === created.id));
    const byAlice = await api.listJobs({ owner: MOCK_USER_IDS.alice });
    assert.ok(byAlice.items.every((j) => j.owner?.id === MOCK_USER_IDS.alice));
    assert.equal(byAlice.items.length, 3);
    const failed = await api.listJobs({ status: "failed" });
    assert.ok(failed.items.length >= 1 && failed.items.every((j) => j.status === "failed"));
    const p1 = await api.listJobs({ limit: 2 });
    assert.equal(p1.items.length, 2);
    assert.ok(p1.next_cursor);
    const p2 = await api.listJobs({ limit: 2, cursor: p1.next_cursor });
    assert.notEqual(p2.items[0].id, p1.items[0].id);
    await assert.rejects(api.listJobs({ cursor: "garbage" }), (e: unknown) => e instanceof api.ApiError && e.code === "invalid_request");
    assert.equal((await api.checkInterpreter()).ok, true);

    // delete (admin may delete anyone's job, even while it is running); then it is 404 everywhere
    await api.deleteJob(created.id);
    await assert.rejects(api.getJob(created.id), notFound);
    assert.ok(!(await api.listJobs()).items.some((j) => j.id === created.id));

    // forced password change blocks job routes
    mockSignInAs("newbie");
    await assert.rejects(api.listJobs(), (e: unknown) => e instanceof api.ApiError && e.code === "password_change_required");
    assert.equal((await api.getHealth()).interpreter, null);
  });
});
