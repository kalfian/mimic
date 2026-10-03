/**
 * Framework-free job list ("My jobs" / "All jobs", PLAN-auth §8.1): first page, "Load more",
 * background refresh every 5 s while any row is still queued/processing, refresh on window focus,
 * and deleting rows. `useJobList` is a thin React wrapper (useSyncExternalStore).
 *
 * A refresh re-reads everything currently shown in one request (`limit` = rows on screen, ≤ 200)
 * so "Load more" pages don't disappear and running rows update in place.
 */

import { deleteJob, listJobs, type ListJobsOptions } from "./api";
import { ApiError } from "./errors";
import type { JobList, JobState, JobStatus } from "./types";

export interface JobListQuery {
  /** `"me"`, a user id (admins), or omitted (users: own jobs; admins: all jobs). */
  owner?: ListJobsOptions["owner"];
  status?: JobState;
  /** Rows per page, default 50 (1..200). */
  pageSize?: number;
}

export interface JobListClient {
  listJobs(options: ListJobsOptions): Promise<JobList>;
  deleteJob(jobId: string): Promise<void>;
}

export interface JobListOptions {
  client?: JobListClient;
  /** Refresh cadence while any row is non-terminal. Default 5000 ms. */
  refreshIntervalMs?: number;
  /** Refresh when the window regains focus / becomes visible. Default true (browser only). */
  refreshOnFocus?: boolean;
}

export interface JobListSnapshot {
  items: JobStatus[];
  /** First load in progress (no rows yet). */
  isLoading: boolean;
  /** A background refresh is running (rows stay on screen). */
  isRefreshing: boolean;
  isLoadingMore: boolean;
  /** More rows on the server ("Load more"). */
  hasMore: boolean;
  /** Last first-load/refresh failure (rows, if any, are kept). Cleared by the next success. */
  error: ApiError | null;
  /** Last "Load more" failure. */
  loadMoreError: ApiError | null;
  /** Some row is queued/processing (the list refreshes itself while true). */
  hasActiveJobs: boolean;
}

export const DEFAULT_PAGE_SIZE = 50;
export const JOB_LIST_REFRESH_MS = 5000;
const MAX_LIMIT = 200;
const FOCUS_THROTTLE_MS = 1000;

const defaultClient: JobListClient = { listJobs, deleteJob };

export function isActiveJob(job: Pick<JobStatus, "status">): boolean {
  return job.status === "queued" || job.status === "processing";
}

/** Append `more` to `items`, skipping ids already shown (a row can shift pages between requests). */
export function mergeJobPages(items: JobStatus[], more: JobStatus[]): JobStatus[] {
  const seen = new Set(items.map((j) => j.id));
  return [...items, ...more.filter((j) => !seen.has(j.id))];
}

const INITIAL: JobListSnapshot = Object.freeze({
  items: [],
  isLoading: true,
  isRefreshing: false,
  isLoadingMore: false,
  hasMore: false,
  error: null,
  loadMoreError: null,
  hasActiveJobs: false,
}) as JobListSnapshot;

export class JobListController {
  private snapshot: JobListSnapshot = INITIAL;
  private readonly listeners = new Set<() => void>();
  private readonly client: JobListClient;
  private readonly refreshIntervalMs: number;
  private readonly refreshOnFocus: boolean;
  private readonly pageSize: number;
  private nextCursor: string | null = null;
  private running = false;
  private refreshController: AbortController | null = null;
  private moreController: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  /** Bumped by every committed refresh; a "Load more" started before it is dropped. */
  private version = 0;
  private lastRefreshAt = 0;

  private readonly query: JobListQuery;

  constructor(query: JobListQuery = {}, options: JobListOptions = {}) {
    this.query = query;
    this.client = options.client ?? defaultClient;
    this.refreshIntervalMs = options.refreshIntervalMs ?? JOB_LIST_REFRESH_MS;
    this.refreshOnFocus = options.refreshOnFocus ?? true;
    this.pageSize = Math.min(MAX_LIMIT, Math.max(1, Math.floor(query.pageSize ?? DEFAULT_PAGE_SIZE)));
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): JobListSnapshot => this.snapshot;

  /** Load (or reload) and start auto-refresh. Safe to call again after `stop()` (React StrictMode). */
  start = (): void => {
    if (this.running) return;
    this.running = true;
    if (this.refreshOnFocus && typeof window !== "undefined") {
      window.addEventListener("focus", this.onFocus);
      document.addEventListener("visibilitychange", this.onFocus);
    }
    void this.refresh();
  };

  /** Abort requests, stop timers and listeners. Rows stay in the snapshot. */
  stop = (): void => {
    this.running = false;
    this.clearTimer();
    this.refreshController?.abort();
    this.moreController?.abort();
    this.refreshController = this.moreController = null;
    if (typeof window !== "undefined") {
      window.removeEventListener("focus", this.onFocus);
      document.removeEventListener("visibilitychange", this.onFocus);
    }
    if (this.snapshot.isRefreshing || this.snapshot.isLoadingMore) this.set({ isRefreshing: false, isLoadingMore: false });
  };

  /** Re-read the rows on screen (or the first page). Resolves when done; never rejects. */
  refresh = async (): Promise<void> => {
    this.clearTimer();
    this.refreshController?.abort();
    const controller = new AbortController();
    this.refreshController = controller;
    this.lastRefreshAt = Date.now();
    const limit = Math.min(MAX_LIMIT, Math.max(this.pageSize, this.snapshot.items.length));
    if (!this.snapshot.isLoading) this.set({ isRefreshing: true });
    try {
      const page = await this.client.listJobs({ ...this.listQuery(), limit, signal: controller.signal });
      if (controller.signal.aborted) return;
      this.version++;
      this.nextCursor = page.next_cursor;
      this.moreController?.abort(); // its cursor belongs to the old rows
      this.set({
        items: page.items,
        isLoading: false,
        isRefreshing: false,
        isLoadingMore: false,
        hasMore: page.next_cursor !== null,
        error: null,
        hasActiveJobs: page.items.some(isActiveJob),
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      const e = ApiError.from(err);
      if (e.code === "aborted") return;
      this.set({ isLoading: false, isRefreshing: false, error: e });
    } finally {
      if (this.refreshController === controller) {
        this.refreshController = null;
        this.schedule();
      }
    }
  };

  /** Fetch the next page and append it. No-op without more rows or while one is loading. */
  loadMore = async (): Promise<void> => {
    if (!this.nextCursor || this.moreController || this.snapshot.isLoading) return;
    const controller = new AbortController();
    this.moreController = controller;
    const version = this.version;
    this.set({ isLoadingMore: true, loadMoreError: null });
    try {
      const page = await this.client.listJobs({ ...this.listQuery(), limit: this.pageSize, cursor: this.nextCursor, signal: controller.signal });
      if (controller.signal.aborted || version !== this.version) return;
      this.nextCursor = page.next_cursor;
      const items = mergeJobPages(this.snapshot.items, page.items);
      this.set({ items, isLoadingMore: false, hasMore: page.next_cursor !== null, hasActiveJobs: items.some(isActiveJob) });
      this.schedule();
    } catch (err) {
      if (controller.signal.aborted) return;
      this.set({ isLoadingMore: false, loadMoreError: ApiError.from(err) });
    } finally {
      if (this.moreController === controller) this.moreController = null;
    }
  };

  /**
   * Delete a job (`DELETE /api/jobs/{id}`) and drop its row. Resolves null on success — also for
   * `not_found` (already gone, or no longer visible), which removes the row too — or the
   * `ApiError` to show (e.g. `network_error`, `unauthenticated`).
   */
  remove = async (jobId: string): Promise<ApiError | null> => {
    try {
      await this.client.deleteJob(jobId);
    } catch (err) {
      const failure = ApiError.from(err);
      if (failure.code !== "not_found") return failure;
    }
    const items = this.snapshot.items.filter((j) => j.id !== jobId);
    this.set({ items, hasActiveJobs: items.some(isActiveJob) });
    return null;
  };

  private listQuery(): ListJobsOptions {
    const q: ListJobsOptions = {};
    if (this.query.owner) q.owner = this.query.owner;
    if (this.query.status) q.status = this.query.status;
    return q;
  }

  private onFocus = (): void => {
    if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
    if (!this.running || Date.now() - this.lastRefreshAt < FOCUS_THROTTLE_MS) return;
    void this.refresh();
  };

  private schedule(): void {
    this.clearTimer();
    if (!this.running || !this.snapshot.hasActiveJobs) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.refresh();
    }, this.refreshIntervalMs);
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private set(patch: Partial<JobListSnapshot>): void {
    this.snapshot = Object.freeze({ ...this.snapshot, ...patch }) as JobListSnapshot;
    for (const l of [...this.listeners]) l();
  }
}
