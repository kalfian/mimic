"use client";

import { IconAlert, IconInfo } from "@/components/icons";
import { StageList } from "@/components/job/StageList";
import { useNow } from "@/components/job/useNow";
import { formatElapsed, formatPercent, formatPixelRatio, stageLabel } from "@/lib/format";
import type { JobStatus } from "@/lib/types";

/** After this much wall time without finishing, say so (the slow mock runs ~30 s). */
const LONG_RUNNING_MS = 20_000;

export function ProcessingStatus({ job, progress, retrying }: { job: JobStatus | null; progress: number; retrying: number }) {
  const now = useNow(1000, true);
  const createdMs = job ? Date.parse(job.created_at) : NaN;
  const elapsedMs = now != null && Number.isFinite(createdMs) ? Math.max(0, now - createdMs) : null;
  const useInterpreter = job?.options.use_interpreter ?? false;
  const interpreting = job?.stage === "interpreting";
  const queued = job?.status === "queued";
  const pct = Math.round(progress * 100);

  const current = job ? stageLabel(job.stage, { useInterpreter }) : "Connecting";
  const src = job?.source;

  return (
    <div className="max-w-2xl space-y-6">
      <header className="space-y-2">
        <p className="caption">{queued ? "Queued" : "Analyzing"}</p>
        <h1 className="text-lg font-semibold tracking-tight break-all text-ink sm:text-xl">{src?.filename ?? "Your recording"}</h1>
        <p className="nums text-xs text-ink-3">
          {src ? `${src.width} × ${src.height} · ${src.fps.toFixed(0)} fps · ${src.duration_s.toFixed(2)} s` : "reading video…"}
          {job ? (
            <>
              {" · "}scale {job.options.pixel_ratio === "auto" ? "auto" : formatPixelRatio(Number(job.options.pixel_ratio))}
              {" · "}AI labeling {useInterpreter ? "on" : "off"}
            </>
          ) : null}
        </p>
      </header>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between gap-4 text-sm">
          <span className="font-medium text-ink" aria-live="polite">
            {current}
            {queued ? "…" : ""}
          </span>
          <span className="nums text-xs text-ink-3">
            {formatPercent(progress)}
            {elapsedMs != null ? <span className="ml-3">{formatElapsed(elapsedMs)}</span> : null}
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="Analysis progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-valuetext={`${pct}%, ${current}`}
          className="h-1 overflow-hidden rounded-full bg-surface-2"
        >
          <div className="h-full bg-signal-fill transition-[width] duration-500 ease-out" style={{ width: `${pct}%` }} />
        </div>
      </div>

      {retrying > 0 ? (
        <p role="status" className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn-soft px-3 py-2 text-sm text-ink">
          <IconAlert className="mt-0.5 shrink-0 text-warn" />
          Lost contact with the API server. Reconnecting (attempt {retrying} of 5)…
        </p>
      ) : null}

      {interpreting && useInterpreter ? (
        <p className="flex items-start gap-2 rounded-md border border-line px-3 py-2 text-sm text-ink-2">
          <IconInfo className="mt-0.5 shrink-0 text-ink-3" />
          Measurements are done. AI labeling can take up to about 2 minutes; if it times out, heuristic labels are used.
        </p>
      ) : elapsedMs != null && elapsedMs > LONG_RUNNING_MS ? (
        <p className="flex items-start gap-2 rounded-md border border-line px-3 py-2 text-sm text-ink-2">
          <IconInfo className="mt-0.5 shrink-0 text-ink-3" />
          Taking longer than usual. High-resolution or high-frame-rate recordings need more time. You can leave this page open.
        </p>
      ) : null}

      <StageList job={job} />
    </div>
  );
}
