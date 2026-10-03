/** PRD §27 recording guidelines + the hover example as a recording-timeline diagram. Static. */

const GUIDELINES = [
  "Keep the recording short (2–15 s).",
  "Record at 60 fps when you can.",
  "Keep the cursor visible.",
  "Start recording before you interact.",
  "Wait about one second before interacting.",
  "Perform one interaction.",
  "Wait until the animation finishes.",
  "Return the UI to its initial state if possible.",
];

type StepKind = "idle" | "event" | "forward" | "reverse";

const HOVER_SEQUENCE: { label: string; hint: string; kind: StepKind }[] = [
  { label: "Initial state", hint: "wait ~1 s", kind: "idle" },
  { label: "Cursor enters", hint: "trigger", kind: "event" },
  { label: "Hover animation", hint: "measured: forward", kind: "forward" },
  { label: "Hold", hint: "wait ~1 s", kind: "idle" },
  { label: "Cursor leaves", hint: "trigger", kind: "event" },
  { label: "Return animation", hint: "measured: reverse", kind: "reverse" },
  { label: "Initial state", hint: "wait", kind: "idle" },
];

function Marker({ kind }: { kind: StepKind }) {
  switch (kind) {
    case "forward":
      return <span className="h-3.5 w-1.5 shrink-0 rounded-[2px] bg-signal-fill" />;
    case "reverse":
      return <span className="h-3.5 w-1.5 shrink-0 rounded-[2px] bg-rev-fill" />;
    case "event":
      return <span className="size-2 shrink-0 rotate-45 border border-ink bg-surface" />;
    case "idle":
      return <span className="size-1.5 shrink-0 rounded-full bg-line-strong" />;
  }
}

export function RecordingGuidelines() {
  return (
    <section aria-labelledby="guide-title" className="rounded-md border border-line bg-surface">
      <header className="border-b border-line px-4 py-2">
        <h2 id="guide-title" className="caption">
          Recording guide
        </h2>
      </header>
      <div className="space-y-6 p-4">
        <ol className="space-y-2">
          {GUIDELINES.map((g, i) => (
            <li key={g} className="flex gap-3 text-sm text-ink-2">
              <span className="nums w-5 shrink-0 pt-px text-2xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
              <span>{g}</span>
            </li>
          ))}
        </ol>

        <figure>
          <figcaption className="mb-3 text-sm font-medium text-ink">A hover, recorded well</figcaption>
          <ol className="relative ml-1 border-l border-line pl-4">
            {HOVER_SEQUENCE.map((s, i) => (
              <li key={i} className="relative flex items-baseline justify-between gap-3 py-1">
                <span className="absolute top-1/2 -left-4 flex w-0 -translate-y-1/2 items-center justify-center">
                  <Marker kind={s.kind} />
                </span>
                <span className={`text-sm ${s.kind === "idle" ? "text-ink-3" : "text-ink"}`}>{s.label}</span>
                <span className="font-mono text-2xs text-ink-3">{s.hint}</span>
              </li>
            ))}
          </ol>
        </figure>

        <ul className="space-y-2 border-t border-line pt-4 text-sm text-ink-2">
          <li>Record the browser at 100% zoom.</li>
          <li>Recorded a Retina screen? Set display scale to 2x.</li>
          <li>One component per recording. Scroll and page transitions are not supported yet.</li>
        </ul>
      </div>
    </section>
  );
}
