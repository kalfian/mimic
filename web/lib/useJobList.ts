/**
 * React hook for the job list page (client components only), see `jobList.ts`.
 *
 *   const { items, isLoading, error, hasMore, isLoadingMore, loadMore, refresh, remove } =
 *     useJobList({ owner: "me" });
 *
 * Changing `owner` / `status` / `pageSize` starts a fresh list. Unmount stops polling.
 */

import { useEffect, useMemo, useSyncExternalStore } from "react";

import type { ApiError } from "./errors";
import { JobListController, type JobListQuery, type JobListSnapshot } from "./jobList";

export interface UseJobListResult extends JobListSnapshot {
  /** Re-read the rows on screen ("Try again" in the error state). */
  refresh: () => Promise<void>;
  loadMore: () => Promise<void>;
  /** Delete a job and drop its row; resolves the `ApiError` to show, or null on success. */
  remove: (jobId: string) => Promise<ApiError | null>;
}

export function useJobList(query: JobListQuery = {}): UseJobListResult {
  const { owner, status, pageSize } = query;
  const controller = useMemo(() => new JobListController({ owner, status, pageSize }), [owner, status, pageSize]);

  useEffect(() => {
    controller.start();
    return controller.stop;
  }, [controller]);

  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
  return { ...snapshot, refresh: controller.refresh, loadMore: controller.loadMore, remove: controller.remove };
}
