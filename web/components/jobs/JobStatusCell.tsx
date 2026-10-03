import { describeError, formatPercent } from "@/lib/format";
import type { JobStatus } from "@/lib/types";

const LABEL: Record<JobStatus["status"], string> = { queued: "Queued", processing: "Processing", succeeded: "Done", failed: "Failed" };
const DOT: Record<JobStatus["status"], string> = {
  queued: "bg-line-strong",
  processing: "bg-signal-fill pulse",
  succeeded: "bg-ok",
  failed: "bg-danger",
};

/** Status word (never colour alone) + one detail line: stage + progress, or why it failed. */
export function JobStatusCell({ job, compact = false }: { job: JobStatus; compact?: boolean }) {
  const running = job.status === "processing";
  const detail = running ? job.stage_label : job.status === "failed" && job.error ? describeError(job.error.code).title : null;
  return (
    <div className="min-w-0">
      <p className="flex items-center gap-2 text-sm text-ink">
        <span aria-hidden="true" className={`size-1.5 shrink-0 rounded-full ${DOT[job.status]}`} />
        <span className={job.status === "failed" ? "text-danger" : ""}>{LABEL[job.status]}</span>
        {running ? <span className="nums text-xs text-ink-3">{formatPercent(job.progress)}</span> : null}
      </p>
      {!compact && detail ? (
        <p className="mt-0.5 truncate pl-3.5 text-xs text-ink-3" title={job.error?.message ?? detail}>
          {detail}
        </p>
      ) : null}
      {!compact && running ? (
        <div aria-hidden="true" className="mt-1 ml-3.5 h-0.5 w-20 overflow-hidden rounded-full bg-surface-2">
          <div className="h-full bg-signal-fill transition-[width] duration-500 ease-out" style={{ width: `${Math.round(job.progress * 100)}%` }} />
        </div>
      ) : null}
    </div>
  );
}
