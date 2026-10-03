"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { IconArrowLeft, IconTrash } from "@/components/icons";
import { JobErrorState } from "@/components/job/JobErrorState";
import { DeleteJobDialog } from "@/components/jobs/DeleteJobDialog";
import { ProcessingStatus } from "@/components/job/ProcessingStatus";
import { RelabelAction, type RelabelPhase } from "@/components/result/RelabelAction";
import { ResultView } from "@/components/result/ResultView";
import { Button } from "@/components/ui/Button";
import { deleteJob, rerunInterpretation } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import { describeError, formatJobOwner } from "@/lib/format";
import { JOBS_PATH } from "@/lib/routes";
import { isAdminSession, useSession } from "@/lib/session";
import type { JobStatus, ResultEnvelope } from "@/lib/types";
import { useHealth } from "@/lib/useHealth";
import { useJobPolling } from "@/lib/useJobPolling";

function BackLink() {
  return (
    <Link href="/" className="focus-ring -ml-1 inline-flex items-center gap-1.5 rounded-sm px-1 text-sm text-ink-3 hover:text-ink">
      <IconArrowLeft />
      New analysis
    </Link>
  );
}

/**
 * Re-run of the AI labeling step. `kept` is the result on screen when it started: it stays
 * visible while the job goes back through interpreting → generating.
 */
type Relabel =
  { kind: "idle" } | { kind: "requesting" | "running"; kept: ResultEnvelope } | { kind: "rejected"; kept: ResultEnvelope; message: string };

function errorText(err: ApiError): string {
  return describeError(err.code, err.message).message;
}

/** Switches Processing → Result | Error from `useJobPolling` (PLAN §10.1). Reload-safe. */
export function JobView({ id }: { id: string }) {
  const poll = useJobPolling(id);
  const { health } = useHealth();
  const [relabel, setRelabel] = useState<Relabel>({ kind: "idle" });
  const del = useDeleteJob(id);
  const session = useSession();
  const isAdmin = isAdminSession(session);
  const job = poll.job;
  // Admins see whose job this is when it isn't theirs (incl. legacy jobs without an owner).
  const owner = isAdmin && job && job.owner?.id !== session.me?.id ? <OwnerLine job={job} /> : null;
  const deleteButton = job ? (
    <Button size="sm" variant="ghost" icon={<IconTrash />} onClick={del.ask} className="text-ink-3 hover:text-danger">
      Delete
    </Button>
  ) : null;
  const dialog = (
    <DeleteJobDialog
      open={del.open}
      job={job}
      busy={del.busy}
      error={del.error}
      othersJob={owner !== null}
      onCancel={del.cancel}
      onConfirm={() => void del.confirm()}
    />
  );

  if (del.deleted) {
    return (
      <div className="mx-auto max-w-[1240px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <p role="status" className="caption">
          Deleted · back to jobs…
        </p>
      </div>
    );
  }

  const fresh = poll.phase === "succeeded" ? poll.result : null;

  // Derive what to show and the relabel phase from the request state + the poll (no effects).
  let shown: ResultEnvelope | null = fresh;
  let phase: RelabelPhase = "idle";
  let relabelError: string | null = null;
  if (relabel.kind === "requesting") {
    shown = relabel.kept;
    phase = "requesting";
  } else if (relabel.kind === "rejected") {
    shown = relabel.kept;
    phase = "failed";
    relabelError = relabel.message;
  } else if (relabel.kind === "running") {
    if (fresh && poll.job?.error) {
      // A failed re-run puts the job back to succeeded with the previous result + `error`.
      phase = "failed";
      relabelError = errorText(ApiError.fromJob(poll.job.error));
    } else if (fresh) {
      phase = "done";
    } else if (poll.phase === "failed" && poll.error) {
      shown = relabel.kept;
      phase = "failed";
      relabelError = errorText(poll.error);
    } else {
      shown = relabel.kept;
      phase = "running";
    }
  }

  const startRelabel = async () => {
    if (!shown) return;
    const kept = shown;
    setRelabel({ kind: "requesting", kept });
    try {
      await rerunInterpretation(id);
      setRelabel({ kind: "running", kept });
      poll.restart();
    } catch (err) {
      setRelabel({ kind: "rejected", kept, message: errorText(ApiError.from(err)) });
    }
  };

  if (shown) {
    const interpreter = health?.interpreter;
    const status = shown.spec.interpretation.status;
    const busy = phase === "requesting" || phase === "running";
    const offerRelabel = Boolean(interpreter?.available) && (status !== "ok" || busy);
    return (
      <>
        <ResultView
          result={shown}
          owner={owner}
          actions={deleteButton}
          labelAction={
            offerRelabel && interpreter ? (
              <RelabelAction status={status} interpreter={interpreter} phase={phase} stage={poll.stage} error={relabelError} onStart={startRelabel} />
            ) : null
          }
          relabeled={phase === "done" && status === "ok"}
        />
        {dialog}
      </>
    );
  }

  return (
    <div className="mx-auto max-w-[1240px] space-y-6 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <BackLink />
          {owner}
        </div>
        {deleteButton}
      </div>
      {dialog}
      {poll.phase === "failed" && poll.error ? (
        <JobErrorState error={poll.error} job={poll.job} onRetry={poll.retry} />
      ) : poll.job?.status === "succeeded" ? (
        <ResultLoading />
      ) : (
        <ProcessingStatus job={poll.job} progress={poll.progress} retrying={poll.retrying} />
      )}
    </div>
  );
}

function OwnerLine({ job }: { job: JobStatus }) {
  return (
    <p className="text-xs text-ink-3">
      Owner{" "}
      {job.owner ? (
        <span className="font-mono text-ink-2">{job.owner.username}</span>
      ) : (
        <span title="Uploaded before accounts existed. Only admins can see it." className="rounded-sm border border-line px-1 font-mono text-2xs">
          {formatJobOwner(null)}
        </span>
      )}
    </p>
  );
}

/** Delete from the job page: confirm → `DELETE` → `/jobs`. `not_found` counts as done (already gone). */
function useDeleteJob(id: string) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [deleted, setDeleted] = useState(false);

  const confirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteJob(id);
    } catch (err) {
      const e = ApiError.from(err);
      if (e.code !== "not_found") {
        setError(e);
        setBusy(false);
        return;
      }
    }
    setDeleted(true);
    setOpen(false);
    setBusy(false);
    router.push(JOBS_PATH);
  };

  return {
    open,
    busy,
    error,
    deleted,
    ask: () => {
      setError(null);
      setOpen(true);
    },
    cancel: () => setOpen(false),
    confirm,
  };
}

function ResultLoading() {
  return (
    <div role="status" aria-live="polite" className="space-y-4">
      <p className="caption">Loading result</p>
      <div className="grid gap-4 lg:grid-cols-12" aria-hidden="true">
        <div className="hatch aspect-video rounded-md border border-line lg:col-span-7" />
        <div className="hatch h-64 rounded-md border border-line lg:col-span-5" />
        <div className="hatch h-56 rounded-md border border-line lg:col-span-12" />
      </div>
    </div>
  );
}
