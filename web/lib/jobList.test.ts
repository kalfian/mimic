// JobListController with a fake client: pagination, refresh of the rows on screen, polling while
// jobs run, deletion. (No React: the hook only wraps this.)
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ListJobsOptions } from "./api";
import { ApiError } from "./errors";
import { JobListController, mergeJobPages, type JobListClient } from "./jobList";
import type { JobState, JobStatus } from "./types";

function job(id: string, status: JobState = "succeeded"): JobStatus {
  return {
    id,
    status,
    stage: status === "succeeded" ? "done" : "measuring",
    stage_label: "",
    progress: status === "succeeded" ? 1 : 0.5,
    error: null,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
    options: { pixel_ratio: "auto", use_interpreter: false },
    source: null,
    owner: null,
  };
}

/** Server with `rows` (newest first) and offset cursors. */
function fakeServer(rows: JobStatus[]) {
  const calls: ListJobsOptions[] = [];
  const deleted: string[] = [];
  let failNext: ApiError | null = null;
  const client: JobListClient = {
    async listJobs(o) {
      calls.push(o);
      if (failNext) {
        const e = failNext;
        failNext = null;
        throw e;
      }
      const start = o.cursor ? Number(o.cursor) : 0;
      const limit = o.limit ?? 50;
      const items = rows.slice(start, start + limit);
      return { items, next_cursor: start + limit < rows.length ? String(start + limit) : null };
    },
    async deleteJob(id) {
      if (id === "gone") throw new ApiError("not_found", "x", { status: 404 });
      if (id === "offline") throw ApiError.network();
      deleted.push(id);
      rows = rows.filter((r) => r.id !== id);
    },
  };
  return {
    client,
    calls,
    deleted,
    setRows: (r: JobStatus[]) => (rows = r),
    failOnce: (e: ApiError) => (failNext = e),
  };
}

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

describe("JobListController", () => {
  it("loads, paginates, and refreshes everything on screen in one request", async () => {
    const rows = Array.from({ length: 5 }, (_, i) => job(`j${i}`));
    const server = fakeServer(rows);
    const c = new JobListController({ owner: "me", pageSize: 2 }, { client: server.client, refreshOnFocus: false });
    assert.equal(c.getSnapshot().isLoading, true);
    c.start();
    await tick();
    let s = c.getSnapshot();
    assert.deepEqual(s.items.map((j) => j.id), ["j0", "j1"]);
    assert.equal(s.hasMore, true);
    assert.equal(server.calls[0].owner, "me");
    await c.loadMore();
    await c.loadMore();
    s = c.getSnapshot();
    assert.deepEqual(s.items.map((j) => j.id), ["j0", "j1", "j2", "j3", "j4"]);
    assert.equal(s.hasMore, false);
    await c.loadMore(); // no-op on the last page
    assert.equal(server.calls.length, 3);
    await c.refresh();
    assert.equal(server.calls.at(-1)?.limit, 5);
    assert.equal(c.getSnapshot().items.length, 5);
    c.stop();
  });

  it("keeps rows on a failed refresh and clears the error on the next success", async () => {
    const server = fakeServer([job("a")]);
    const c = new JobListController({}, { client: server.client, refreshOnFocus: false });
    c.start();
    await tick();
    server.failOnce(ApiError.network());
    await c.refresh();
    assert.equal(c.getSnapshot().error?.code, "network_error");
    assert.equal(c.getSnapshot().items.length, 1);
    await c.refresh();
    assert.equal(c.getSnapshot().error, null);
    c.stop();
  });

  it("first-load error: isLoading false, error set, Try again works", async () => {
    const server = fakeServer([job("a")]);
    server.failOnce(new ApiError("forbidden", "no", { status: 403 }));
    const c = new JobListController({}, { client: server.client, refreshOnFocus: false });
    c.start();
    await tick();
    assert.equal(c.getSnapshot().isLoading, false);
    assert.equal(c.getSnapshot().error?.code, "forbidden");
    await c.refresh();
    assert.equal(c.getSnapshot().items.length, 1);
    c.stop();
  });

  it("polls while a job is running and stops once all are terminal", async () => {
    const server = fakeServer([job("a", "processing"), job("b")]);
    const c = new JobListController({}, { client: server.client, refreshIntervalMs: 10, refreshOnFocus: false });
    c.start();
    await tick();
    assert.equal(c.getSnapshot().hasActiveJobs, true);
    await tick(25);
    assert.ok(server.calls.length >= 2, `expected polling, got ${server.calls.length} calls`);
    server.setRows([job("a"), job("b")]);
    await tick(25);
    assert.equal(c.getSnapshot().hasActiveJobs, false);
    const n = server.calls.length;
    await tick(40);
    assert.equal(server.calls.length, n, "no more polling once everything is terminal");
    c.stop();
  });

  it("stop() cancels polling; start() again reloads (StrictMode)", async () => {
    const server = fakeServer([job("a", "queued")]);
    const c = new JobListController({}, { client: server.client, refreshIntervalMs: 10, refreshOnFocus: false });
    c.start();
    c.stop();
    c.start();
    await tick();
    c.stop();
    const n = server.calls.length;
    await tick(30);
    assert.equal(server.calls.length, n);
  });

  it("remove: drops the row; not_found counts as done; other errors are returned", async () => {
    const server = fakeServer([job("a"), job("gone"), job("offline")]);
    const c = new JobListController({}, { client: server.client, refreshOnFocus: false });
    c.start();
    await tick();
    assert.equal(await c.remove("a"), null);
    assert.equal(await c.remove("gone"), null);
    const e = await c.remove("offline");
    assert.equal(e?.code, "network_error");
    assert.deepEqual(c.getSnapshot().items.map((j) => j.id), ["offline"]);
    assert.deepEqual(server.deleted, ["a"]);
    c.stop();
  });

  it("mergeJobPages skips duplicates", () => {
    assert.deepEqual(mergeJobPages([job("a"), job("b")], [job("b"), job("c")]).map((j) => j.id), ["a", "b", "c"]);
  });
});
