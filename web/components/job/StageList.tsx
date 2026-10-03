import { IconCheck, IconCross, IconSpinner } from "@/components/icons";
import { stageSteps, type StageStepState } from "@/lib/format";
import type { JobStatus } from "@/lib/types";

const STATE_TEXT: Record<StageStepState, string> = {
  done: "done",
  current: "running",
  pending: "",
  failed: "failed",
};

function StepIcon({ state }: { state: StageStepState }) {
  switch (state) {
    case "done":
      return <IconCheck className="text-ok" />;
    case "current":
      return <IconSpinner className="text-signal" />;
    case "failed":
      return <IconCross className="text-danger" />;
    case "pending":
      return <span className="m-[5px] size-1.5 rounded-full border border-line-strong" aria-hidden="true" />;
  }
}

/**
 * Pipeline checklist. The upload itself is shown as step 00 (it finished before the job existed),
 * so the sequence reads continuously from the upload page.
 */
export function StageList({ job }: { job: Pick<JobStatus, "status" | "stage" | "options"> | null }) {
  const steps = stageSteps(job);
  return (
    <ol aria-label="Analysis stages" className="divide-y divide-line rounded-md border border-line bg-surface">
      <Row index={0} label="Upload" state={job ? "done" : "current"} />
      {steps.map((s, i) => (
        <Row key={s.stage} index={i + 1} label={s.label} state={s.state} />
      ))}
    </ol>
  );
}

function Row({ index, label, state }: { index: number; label: string; state: StageStepState }) {
  return (
    <li
      aria-current={state === "current" ? "step" : undefined}
      className={`flex items-center gap-3 px-4 py-2 text-sm ${
        state === "current" ? "bg-signal-soft/70" : state === "failed" ? "bg-danger-soft" : ""
      }`}
    >
      <span className="flex size-4 items-center justify-center">
        <StepIcon state={state} />
      </span>
      <span className="nums w-5 text-2xs text-ink-3">{String(index).padStart(2, "0")}</span>
      <span
        className={`flex-1 ${
          state === "pending" ? "text-ink-3" : state === "failed" ? "font-medium text-danger" : state === "current" ? "font-medium text-ink" : "text-ink-2"
        }`}
      >
        {label}
      </span>
      <span className={`font-mono text-2xs ${state === "failed" ? "text-danger" : "text-ink-3"}`}>
        {STATE_TEXT[state]}
        {state === "pending" ? <span className="sr-only">pending</span> : null}
      </span>
    </li>
  );
}
