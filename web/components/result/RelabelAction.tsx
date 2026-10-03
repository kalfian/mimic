"use client";

import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";

import { IconCheck, IconRefresh, IconSpinner } from "@/components/icons";
import { CONFIDENTIAL_WARNING, interpreterDestination, KEYFRAMES_SENT } from "@/components/interpreterCopy";
import { Button } from "@/components/ui/Button";
import { INTERPRETATION_STATUS_LABELS } from "@/lib/format";
import type { HealthInterpreter, InterpretationStatus, Stage } from "@/lib/types";

/**
 * idle       nothing in flight
 * requesting POST sent, waiting for the server to accept
 * running    polling the job through interpreting → generating
 * failed     the request or the poll failed (`error` set); the old result is still shown
 * done       a new result arrived (its status may still be non-ok, e.g. timed out again)
 */
export type RelabelPhase = "idle" | "requesting" | "running" | "failed" | "done";

const STEPS: { key: "interpreting" | "generating"; label: string }[] = [
  { key: "interpreting", label: "Labeling elements with AI" },
  { key: "generating", label: "Regenerating prompt, JSON and CSS" },
];

function stepState(step: "interpreting" | "generating", phase: RelabelPhase, stage: Stage | null): "done" | "current" | "pending" {
  if (phase === "done") return "done";
  const atGenerating = stage === "generating" || stage === "done";
  if (step === "interpreting") return atGenerating ? "done" : "current";
  return atGenerating ? "current" : "pending";
}

/**
 * "Label with AI" for a result whose labels are heuristic. Two steps on purpose: the confirm step
 * repeats the upload toggle's disclosure, because this sends keyframes to an LLM the user may not
 * have opted into for this recording.
 */
export function RelabelAction({
  status,
  interpreter,
  phase,
  stage,
  error,
  onStart,
}: {
  status: InterpretationStatus;
  interpreter: HealthInterpreter;
  phase: RelabelPhase;
  stage: Stage | null;
  error: string | null;
  onStart: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef(false);
  const id = useId();
  const busy = phase === "requesting" || phase === "running";

  // Keep keyboard focus on something that still exists as the panel changes shape.
  useEffect(() => {
    if (confirming) confirmRef.current?.focus();
    else if (restoreFocus.current) {
      restoreFocus.current = false;
      triggerRef.current?.focus();
    }
  }, [confirming]);
  const wasBusy = useRef(busy);
  useEffect(() => {
    // Only on a busy ⇄ idle transition (never on mount): the control that had focus unmounts
    // when the panel swaps, so land somewhere sensible.
    if (wasBusy.current === busy) return;
    wasBusy.current = busy;
    const lost = !document.activeElement || document.activeElement === document.body;
    if (!lost) return;
    if (busy) progressRef.current?.focus();
    else triggerRef.current?.focus();
  }, [busy]);

  const cancel = () => {
    restoreFocus.current = true;
    setConfirming(false);
  };

  if (busy) {
    return (
      <div ref={progressRef} tabIndex={-1} role="status" aria-live="polite" className="space-y-1.5 rounded-md border border-line bg-surface-2/50 px-3 py-2.5 outline-none">
        <ol className="space-y-1">
          {STEPS.map((s) => {
            const state = phase === "requesting" ? (s.key === "interpreting" ? "current" : "pending") : stepState(s.key, phase, stage);
            return (
              <li key={s.key} className={`flex items-center gap-2 text-xs ${state === "pending" ? "text-ink-3" : "text-ink"}`}>
                {state === "done" ? (
                  <IconCheck size={14} className="text-ok" />
                ) : state === "current" ? (
                  <IconSpinner size={14} className="text-ink-2" />
                ) : (
                  <span className="inline-flex size-3.5 items-center justify-center" aria-hidden="true">
                    <span className="size-1.5 rounded-full border border-ink-3" />
                  </span>
                )}
                <span>
                  {s.label}
                  <span className="sr-only">{state === "done" ? " (done)" : state === "current" ? " (in progress)" : " (waiting)"}</span>
                </span>
              </li>
            );
          })}
        </ol>
        <p className="text-xs text-ink-3">This result stays usable; the new labels replace it when they arrive.</p>
      </div>
    );
  }

  if (confirming) {
    return (
      <div
        role="group"
        aria-labelledby={`${id}-title`}
        onKeyDown={(e: KeyboardEvent) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            cancel();
          }
        }}
        className="space-y-2.5 rounded-md border border-signal/40 bg-signal-soft px-3 py-2.5"
      >
        <div className="space-y-0.5">
          <p id={`${id}-title`} className="font-medium text-ink">
            Send keyframes to the AI?
          </p>
          <p id={`${id}-desc`} className="text-ink-2">
            The {KEYFRAMES_SENT} are sent to {interpreterDestination(interpreter)}. Only element names and descriptions change; every measured number
            stays as it is. {CONFIDENTIAL_WARNING}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            ref={confirmRef}
            variant="primary"
            size="sm"
            aria-describedby={`${id}-desc`}
            onClick={() => {
              setConfirming(false);
              onStart();
            }}
          >
            Send and label
          </Button>
          <Button variant="ghost" size="sm" onClick={cancel}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {phase === "failed" && error ? (
        <p className="text-xs text-danger" role="alert">
          Couldn’t relabel: {error}
        </p>
      ) : null}
      {phase === "done" && status !== "ok" ? (
        <p className="text-xs text-ink-2" role="status">
          Tried again: {INTERPRETATION_STATUS_LABELS[status] ?? "labels are still heuristic"}.
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <Button ref={triggerRef} size="sm" icon={<IconRefresh />} onClick={() => (phase === "failed" ? onStart() : setConfirming(true))}>
          {phase === "failed" ? "Try again" : "Label with AI"}
        </Button>
        <span className="text-xs text-ink-3">Re-runs only the naming step. Measurements stay the same.</span>
      </div>
    </div>
  );
}
