import Link from "next/link";

import { IconAlert, IconRefresh } from "@/components/icons";
import { StageList } from "@/components/job/StageList";
import { Button, buttonClass } from "@/components/ui/Button";
import type { ApiError } from "@/lib/api";
import { describeError, stageLabel } from "@/lib/format";
import type { JobStatus } from "@/lib/types";

/** Request failures worth retrying from here (a failed pipeline needs a new upload instead). */
function canRetry(error: ApiError): boolean {
  return error.origin !== "job" && error.code !== "not_found";
}

export function JobErrorState({ error, job, onRetry }: { error: ApiError; job: JobStatus | null; onRetry: () => void }) {
  const pipeline = error.origin === "job";
  // Pipeline messages are specific ("…at 1.2 s"); HTTP/transport ones mostly repeat the title.
  const d = describeError(error.code, pipeline ? error.message : null);
  const failedAt = pipeline && job ? stageLabel(job.stage, { useInterpreter: job.options.use_interpreter }) : null;

  return (
    <div className="grid max-w-4xl gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <section aria-labelledby="error-title" className="space-y-5">
        <header className="space-y-2">
          <p className="caption text-danger">{pipeline ? "Analysis failed" : "Could not load this analysis"}</p>
          <h1 id="error-title" className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
            <IconAlert size={20} className="shrink-0 text-danger" />
            {d.title}
          </h1>
          <p className="max-w-prose text-md text-ink-2">{d.message}</p>
          <p className="font-mono text-2xs text-ink-3">
            {error.code}
            {failedAt ? ` · during “${failedAt}”` : ""}
            {error.status ? ` · HTTP ${error.status}` : ""}
          </p>
        </header>

        {d.guidance.length > 0 ? (
          <div className="rounded-md border border-line bg-surface p-4">
            <h2 className="caption mb-2">{pipeline ? "For the next recording" : "What to check"}</h2>
            <ol className="space-y-1.5">
              {d.guidance.map((g, i) => (
                <li key={g} className="flex gap-3 text-sm text-ink-2">
                  <span className="nums w-5 shrink-0 pt-px text-2xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                  {g}
                </li>
              ))}
            </ol>
          </div>
        ) : null}

        {error.code === "continuous_motion_unsupported" ? <MeasurableMotion /> : null}

        <div className="flex flex-wrap gap-3">
          <Link href="/" className={buttonClass("primary")}>
            Upload another recording
          </Link>
          {canRetry(error) ? (
            <Button icon={<IconRefresh />} onClick={onRetry}>
              Try again
            </Button>
          ) : null}
        </div>
      </section>

      {pipeline && job ? (
        <aside aria-label="Where it stopped">
          <StageList job={job} />
        </aside>
      ) : null}
    </div>
  );
}

/** What the analyser supports, for failures caused by motion it can't model (PRD §25/§26). */
function MeasurableMotion() {
  const rows: { title: string; body: string; ok: boolean }[] = [
    {
      title: "One interaction",
      body: "A component at rest, one hover, click or press, then rest again. A looping animation elsewhere on the page is usually ignored.",
      ok: true,
    },
    {
      title: "One scroller",
      body: "A marquee or auto-scrolling carousel moving along one axis: autoplay, pause, drag, momentum, snap and resume are measured.",
      ok: true,
    },
    {
      title: "Not measurable",
      body: "Two-axis or rotating motion, pulsing or morphing loops, several scrollers moving in opposite directions, page scrolling.",
      ok: false,
    },
  ];
  return (
    <div className="rounded-md border border-line bg-surface">
      <h2 className="caption border-b border-line px-4 py-2">What Mimic can measure</h2>
      <dl className="divide-y divide-line">
        {rows.map((r) => (
          <div key={r.title} className="grid gap-x-4 gap-y-0.5 px-4 py-2.5 sm:grid-cols-[9rem_minmax(0,1fr)]">
            <dt className={`flex items-center gap-2 text-sm font-medium ${r.ok ? "text-ink" : "text-ink-3"}`}>
              <span className={`inline-block h-3 w-1.5 shrink-0 rounded-[2px] ${r.ok ? "bg-signal-fill" : "border border-dashed border-ink-3"}`} aria-hidden="true" />
              {r.title}
            </dt>
            <dd className="text-sm text-ink-2">{r.body}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
