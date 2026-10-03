"use client";

import Link from "next/link";
import { useRef, useState } from "react";

import { IconPlus, IconRefresh, IconTrash } from "@/components/icons";
import { useNow } from "@/components/job/useNow";
import { DeleteJobDialog, jobDisplayName } from "@/components/jobs/DeleteJobDialog";
import { JobStatusCell } from "@/components/jobs/JobStatusCell";
import { OwnerFilter } from "@/components/jobs/OwnerFilter";
import { Button, buttonClass } from "@/components/ui/Button";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import type { ApiError } from "@/lib/errors";
import { formatJobOwner, formatRelativeTime } from "@/lib/format";
import { JOBS_PATH } from "@/lib/routes";
import { isAdminSession, useSession } from "@/lib/session";
import type { JobStatus } from "@/lib/types";
import { useJobList } from "@/lib/useJobList";

const DATE_TIME = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" });

function formatDuration(job: JobStatus): string {
  return job.source ? `${job.source.duration_s.toFixed(2)} s` : "—";
}

/**
 * `/jobs`: "My jobs" for users, "All jobs" for admins (Owner column + owner filter, "Legacy" for
 * jobs from before accounts). Deleting is optimistic: the row disappears at once and comes back
 * with an error if the server refuses.
 */
export function JobsScreen({ initialOwner }: { initialOwner: string | null }) {
  const session = useSession();
  const isAdmin = isAdminSession(session);
  const meId = session.me?.id ?? "";
  const [owner, setOwner] = useState<string>(isAdmin && initialOwner ? initialOwner : "");
  const list = useJobList({ owner: isAdmin && owner ? owner : undefined });
  const now = useNow(30_000);

  const [pendingDelete, setPendingDelete] = useState<JobStatus | null>(null);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const [deleteError, setDeleteError] = useState<{ job: JobStatus; error: ApiError } | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);

  const items = list.items.filter((j) => !hidden.has(j.id));
  const showOwner = isAdmin;
  const filtered = isAdmin && owner !== "";

  const changeOwner = (next: string) => {
    setOwner(next);
    const url = next ? `${JOBS_PATH}?owner=${encodeURIComponent(next)}` : JOBS_PATH;
    window.history.replaceState(null, "", url);
  };

  const confirmDelete = async () => {
    const job = pendingDelete;
    if (!job) return;
    setPendingDelete(null);
    setDeleteError(null);
    setHidden((h) => new Set(h).add(job.id));
    const error = await list.remove(job.id);
    setHidden((h) => {
      const n = new Set(h);
      n.delete(job.id);
      return n;
    });
    if (error) {
      setDeleteError({ job, error });
      setAnnouncement("");
    } else {
      setAnnouncement(`Deleted ${jobDisplayName(job)}.`);
    }
  };

  const ownerName = filtered && owner !== "me" ? (items.find((j) => j.owner?.id === owner)?.owner?.username ?? null) : null;
  const title = isAdmin ? (filtered ? (owner === "me" ? "My jobs" : ownerName ? `Jobs by ${ownerName}` : "Jobs by owner") : "All jobs") : "My jobs";

  return (
    <div className="mx-auto max-w-[1240px] space-y-6 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0 space-y-1.5">
          <p className="caption">{isAdmin ? "Admin view · every account" : "Your analyses"}</p>
          <h1 ref={headingRef} tabIndex={-1} className="text-xl font-semibold tracking-tight text-ink outline-none">
            {title}
          </h1>
          <p className="text-sm text-ink-3" aria-live="polite">
            <ListMeta
              count={items.length}
              hasMore={list.hasMore}
              loading={list.isLoading}
              failed={list.error !== null}
              refreshing={list.isRefreshing}
              active={list.hasActiveJobs}
            />
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {isAdmin ? <OwnerFilter value={owner} onChange={changeOwner} meId={meId} /> : null}
          <Link href="/" className={buttonClass("primary", "sm")}>
            <IconPlus />
            New analysis
          </Link>
        </div>
      </header>

      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>

      {deleteError ? (
        <Notice
          tone="danger"
          live="assertive"
          title={`Couldn’t delete ${jobDisplayName(deleteError.job)}`}
          onDismiss={() => setDeleteError(null)}
          actions={
            <Button size="sm" icon={<IconRefresh />} onClick={() => setPendingDelete(deleteError.job)}>
              Try again
            </Button>
          }
        >
          {errorCopy(deleteError.error).message} The job is still there.
        </Notice>
      ) : null}

      {list.error && items.length > 0 ? (
        <Notice
          tone="warn"
          title="Couldn’t refresh the list"
          actions={
            <Button size="sm" icon={<IconRefresh />} loading={list.isRefreshing} onClick={() => void list.refresh()}>
              Try again
            </Button>
          }
        >
          {errorCopy(list.error).message} Showing the last rows that loaded.
        </Notice>
      ) : null}

      {list.isLoading ? (
        <ListSkeleton />
      ) : list.error && items.length === 0 ? (
        <Notice
          tone="danger"
          live="assertive"
          title={errorCopy(list.error).title}
          guidance={errorCopy(list.error).guidance}
          actions={
            <Button size="sm" icon={<IconRefresh />} loading={list.isRefreshing} onClick={() => void list.refresh()}>
              Try again
            </Button>
          }
        >
          {errorCopy(list.error).message}
        </Notice>
      ) : items.length === 0 ? (
        <EmptyState isAdmin={isAdmin} filtered={filtered} onShowAll={() => changeOwner("")} />
      ) : (
        <>
          <JobTable items={items} showOwner={showOwner} meId={meId} now={now} onDelete={setPendingDelete} />
          <JobCards items={items} showOwner={showOwner} meId={meId} now={now} onDelete={setPendingDelete} />
        </>
      )}

      {list.hasMore && items.length > 0 ? (
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => void list.loadMore()} loading={list.isLoadingMore}>
            {list.isLoadingMore ? "Loading…" : "Load more"}
          </Button>
          {list.loadMoreError ? (
            <p role="alert" className="text-sm text-danger">
              {errorCopy(list.loadMoreError).message} Try again.
            </p>
          ) : null}
        </div>
      ) : null}

      <DeleteJobDialog
        open={pendingDelete !== null}
        job={pendingDelete}
        othersJob={pendingDelete !== null && isAdmin && pendingDelete.owner?.id !== meId}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => void confirmDelete()}
        returnFocusRef={headingRef}
      />
    </div>
  );
}

function ListMeta({
  count,
  hasMore,
  loading,
  failed,
  refreshing,
  active,
}: {
  count: number;
  hasMore: boolean;
  loading: boolean;
  failed: boolean;
  refreshing: boolean;
  active: boolean;
}) {
  if (loading) return <>Loading…</>;
  if (failed && count === 0) return <>Couldn’t load the list</>;
  if (count === 0) return <>Nothing here yet</>;
  const shown = (
    <>
      <span className="nums">{count}</span> {count === 1 ? "job" : "jobs"}
      {hasMore ? " shown, more below" : ""}
    </>
  );
  return (
    <>
      {shown}
      {refreshing ? " · updating…" : active ? " · refreshes every 5 s while jobs run" : ""}
    </>
  );
}

function OwnerLabel({ job, meId }: { job: JobStatus; meId: string }) {
  if (!job.owner) {
    return (
      <span
        className="rounded-sm border border-line px-1 font-mono text-2xs text-ink-3"
        title="Uploaded before accounts existed. Only admins can see it."
      >
        {formatJobOwner(null)}
      </span>
    );
  }
  return (
    <span className="font-mono text-xs text-ink-2">
      {job.owner.username}
      {job.owner.id === meId ? <span className="ml-1 font-sans text-ink-3">(you)</span> : null}
    </span>
  );
}

function Created({ iso, now }: { iso: string; now: number | null }) {
  const at = Date.parse(iso);
  return (
    <time dateTime={iso} title={Number.isNaN(at) ? undefined : DATE_TIME.format(at)}>
      {now === null ? "…" : formatRelativeTime(iso, now)}
    </time>
  );
}

function FileLink({ job }: { job: JobStatus }) {
  return (
    <Link
      href={`${JOBS_PATH}/${encodeURIComponent(job.id)}`}
      className={`focus-ring rounded-sm font-medium break-all underline-offset-4 hover:underline ${job.source ? "text-ink" : "text-ink-2"}`}
    >
      {jobDisplayName(job)}
    </Link>
  );
}

function DeleteButton({ job, onDelete }: { job: JobStatus; onDelete: (job: JobStatus) => void }) {
  return (
    <button
      type="button"
      onClick={() => onDelete(job)}
      aria-label={`Delete ${jobDisplayName(job)}`}
      title="Delete job"
      className="focus-ring inline-flex size-8 items-center justify-center rounded-sm text-ink-3 transition-colors duration-150 hover:bg-danger-soft hover:text-danger"
    >
      <IconTrash />
    </button>
  );
}

interface ListProps {
  items: JobStatus[];
  showOwner: boolean;
  meId: string;
  now: number | null;
  onDelete: (job: JobStatus) => void;
}

const TH = "caption px-3 py-2 text-left font-normal";

function JobTable({ items, showOwner, meId, now, onDelete }: ListProps) {
  return (
    <div className="hidden overflow-x-auto rounded-md border border-line bg-surface md:block">
      <table className="w-full min-w-[44rem] border-collapse text-sm">
        <caption className="sr-only">Jobs, newest first</caption>
        <thead className="border-b border-line">
          <tr>
            <th scope="col" className={`${TH} pl-4`}>
              File
            </th>
            <th scope="col" className={`${TH} w-44`}>
              Status
            </th>
            {showOwner ? (
              <th scope="col" className={`${TH} w-36`}>
                Owner
              </th>
            ) : null}
            <th scope="col" className={`${TH} w-32`}>
              Created
            </th>
            <th scope="col" className={`${TH} w-24 text-right`}>
              Duration
            </th>
            <th scope="col" className={`${TH} w-14`}>
              AI
            </th>
            <th scope="col" className={`${TH} w-14 pr-4`}>
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((job) => (
            <tr key={job.id} className="border-b border-line align-top transition-colors duration-150 last:border-b-0 hover:bg-surface-2/50">
              <th scope="row" className="py-3 pr-3 pl-4 text-left font-normal">
                <FileLink job={job} />
              </th>
              <td className="px-3 py-3">
                <JobStatusCell job={job} />
              </td>
              {showOwner ? (
                <td className="px-3 py-3">
                  <OwnerLabel job={job} meId={meId} />
                </td>
              ) : null}
              <td className="px-3 py-3 text-ink-2">
                <Created iso={job.created_at} now={now} />
              </td>
              <td className="nums px-3 py-3 text-right text-xs text-ink-2">{formatDuration(job)}</td>
              <td className="px-3 py-3 font-mono text-2xs text-ink-3">
                {job.options.use_interpreter ? <span className="text-ink-2">on</span> : "off"}
              </td>
              <td className="py-2 pr-4 pl-3 text-right">
                <DeleteButton job={job} onDelete={onDelete} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** < md: one card per job (a 6-column table doesn't fit 390 px). */
function JobCards({ items, showOwner, meId, now, onDelete }: ListProps) {
  return (
    <ul className="divide-y divide-line rounded-md border border-line bg-surface md:hidden" aria-label="Jobs, newest first">
      {items.map((job) => (
        <li key={job.id} className="flex items-start gap-3 py-3 pr-2 pl-4">
          <div className="min-w-0 flex-1 space-y-1.5">
            <FileLink job={job} />
            <JobStatusCell job={job} />
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
              <Created iso={job.created_at} now={now} />
              <span aria-hidden="true">·</span>
              <span className="nums">{formatDuration(job)}</span>
              <span aria-hidden="true">·</span>
              <span>AI {job.options.use_interpreter ? "on" : "off"}</span>
              {showOwner ? (
                <>
                  <span aria-hidden="true">·</span>
                  <span>
                    {job.owner ? "by " : null}
                    <OwnerLabel job={job} meId={meId} />
                  </span>
                </>
              ) : null}
            </p>
          </div>
          <DeleteButton job={job} onDelete={onDelete} />
        </li>
      ))}
    </ul>
  );
}

function ListSkeleton() {
  return (
    <div aria-busy="true" className="space-y-px overflow-hidden rounded-md border border-line">
      <p role="status" className="sr-only">
        Loading jobs
      </p>
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="hatch h-14" />
      ))}
    </div>
  );
}

function EmptyState({ isAdmin, filtered, onShowAll }: { isAdmin: boolean; filtered: boolean; onShowAll: () => void }) {
  return (
    <section aria-labelledby="jobs-empty" className="rounded-md border border-dashed border-line-strong px-6 py-10">
      <div className="max-w-md space-y-3">
        <h2 id="jobs-empty" className="text-md font-medium text-ink">
          {filtered ? "No jobs for this owner" : "No jobs yet"}
        </h2>
        <p className="text-sm text-ink-2">
          {filtered
            ? "This account hasn’t analyzed a recording, or its jobs were deleted."
            : isAdmin
              ? "Nobody has analyzed a recording yet. Every upload from any account shows up here."
              : "Recordings you analyze show up here, newest first. Only you and admins can see them."}
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          {filtered ? (
            <Button size="sm" onClick={onShowAll}>
              Show everyone’s jobs
            </Button>
          ) : null}
          <Link href="/" className={buttonClass(filtered ? "ghost" : "primary", "sm")}>
            Analyze a recording
          </Link>
        </div>
      </div>
    </section>
  );
}
