"use client";

import type { RefObject } from "react";

import { IconTrash } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import { formatJobOwner } from "@/lib/format";
import type { ApiError } from "@/lib/errors";
import type { JobStatus } from "@/lib/types";

export function jobDisplayName(job: Pick<JobStatus, "id" | "source">): string {
  return job.source?.filename ?? `Recording ${job.id.slice(0, 8)}`;
}

/**
 * Confirmation for deleting one job (PLAN-auth §14): names the file, says what goes, and that a
 * running analysis is stopped. `ownerNote` is shown when an admin deletes someone else's job.
 */
export function DeleteJobDialog({
  job,
  open,
  busy = false,
  error = null,
  othersJob = false,
  onCancel,
  onConfirm,
  returnFocusRef,
}: {
  job: JobStatus | null;
  open: boolean;
  busy?: boolean;
  error?: ApiError | null;
  /** An admin deleting another account's (or a legacy) job. */
  othersJob?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  returnFocusRef?: RefObject<HTMLElement | null>;
}) {
  if (!job) return null;
  const name = jobDisplayName(job);
  const running = job.status === "queued" || job.status === "processing";
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      role="alertdialog"
      caption="Delete job"
      title={
        <>
          Delete <span className="font-mono text-base break-all">{name}</span>?
        </>
      }
      busy={busy}
      returnFocusRef={returnFocusRef}
      description={
        <>
          The recording, its result, keyframes and generated spec are removed for good.
          {running ? " The analysis is still running and stops now." : ""}
          {othersJob ? ` This job belongs to ${job.owner ? job.owner.username : `no account (${formatJobOwner(null)})`}.` : ""}
        </>
      }
      footer={
        <>
          <Button onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant="danger" icon={<IconTrash />} loading={busy} onClick={onConfirm}>
            {busy ? "Deleting…" : "Delete job"}
          </Button>
        </>
      }
    >
      {error ? (
        <Notice tone="danger" live="assertive" title={errorCopy(error).title}>
          {errorCopy(error).message}
        </Notice>
      ) : null}
    </Dialog>
  );
}
