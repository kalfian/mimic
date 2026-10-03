"use client";

import { useState } from "react";

import { Panel } from "@/components/ui/Panel";
import { assetUrl } from "@/lib/api";
import type { KeyframeArtifact, KeyframeKind, MotionElement, Source } from "@/lib/types";

const SEQUENCE: { kind: KeyframeKind; annotated?: KeyframeKind; label: string }[] = [
  { kind: "state_a", annotated: "annotated_a", label: "A · start" },
  { kind: "mid_25", label: "25%" },
  { kind: "mid_50", label: "50%" },
  { kind: "mid_75", label: "75%" },
  { kind: "state_b", annotated: "annotated_b", label: "B · end" },
];

function Thumb({
  frame,
  label,
  aspect,
  failed,
  onFail,
  onSeek,
  active,
}: {
  frame: KeyframeArtifact;
  label: string;
  aspect: string;
  failed: boolean;
  onFail: () => void;
  onSeek: ((ms: number) => void) | null;
  active: boolean;
}) {
  const body = (
    <>
      <span className="relative block overflow-hidden rounded-sm border border-line bg-canvas" style={{ aspectRatio: aspect }}>
        {failed ? (
          <span className="hatch flex size-full items-center justify-center font-mono text-2xs text-ink-3">image unavailable</span>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element -- API-served PNG (or mock SVG data URI); no resizing wanted
          <img src={assetUrl(frame.url)} alt="" loading="lazy" className="size-full object-contain" onError={onFail} />
        )}
      </span>
      <span className="mt-1.5 flex items-baseline justify-between gap-2">
        <span className={`text-xs ${active ? "font-medium text-ink" : "text-ink-2"}`}>{label}</span>
        {frame.t_ms != null ? <span className="nums text-2xs text-ink-3">{frame.t_ms} ms</span> : null}
      </span>
    </>
  );
  if (!onSeek || frame.t_ms == null) {
    return (
      <figure className="min-w-0" aria-label={`Keyframe ${label}`}>
        {body}
      </figure>
    );
  }
  const t = frame.t_ms;
  return (
    <button
      type="button"
      onClick={() => onSeek(t)}
      aria-label={`Keyframe ${label} at ${t} ms. Move the playhead here.`}
      className={`focus-ring group min-w-0 rounded-sm text-left ${active ? "[&>span:first-child]:border-ink" : "[&>span:first-child]:hover:border-ink-3"}`}
    >
      {body}
    </button>
  );
}

export function KeyframeStrip({
  keyframes,
  elements,
  source,
  playheadMs,
  onSeek,
}: {
  keyframes: KeyframeArtifact[];
  elements: MotionElement[];
  source: Source;
  playheadMs: number;
  onSeek: (ms: number) => void;
}) {
  const [annotated, setAnnotated] = useState(false);
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const byKind = new Map(keyframes.map((k) => [k.kind, k]));
  const hasAnnotated = byKind.has("annotated_a") || byKind.has("annotated_b");
  const crops = keyframes.filter((k) => k.kind === "element");
  const aspect = `${source.width} / ${source.height}`;
  const fail = (name: string) => () => setFailed((f) => ({ ...f, [name]: true }));

  const frames = SEQUENCE.flatMap((s) => {
    const k = (annotated && s.annotated ? byKind.get(s.annotated) : undefined) ?? byKind.get(s.kind);
    return k ? [{ ...s, frame: k }] : [];
  });
  // Highlight the keyframe the playhead is closest to (within two frames).
  const near = frames.reduce<{ name: string; d: number } | null>((best, f) => {
    if (f.frame.t_ms == null) return best;
    const d = Math.abs(f.frame.t_ms - playheadMs);
    return d < 2 * source.timing_resolution_ms && (!best || d < best.d) ? { name: f.frame.name, d } : best;
  }, null);

  const meta = hasAnnotated ? (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-ink-2">
      <input type="checkbox" checked={annotated} onChange={(e) => setAnnotated(e.target.checked)} className="focus-ring size-3.5 accent-[var(--ink)]" />
      Show element boxes
    </label>
  ) : null;

  return (
    <Panel id="keyframes" title="Keyframes" meta={meta}>
      {frames.length === 0 && crops.length === 0 ? (
        <p className="text-sm text-ink-3">No keyframes were produced for this analysis.</p>
      ) : (
        <div className="space-y-5">
          {frames.length > 0 ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {frames.map((f) => (
                <Thumb
                  key={f.kind}
                  frame={f.frame}
                  label={f.label}
                  aspect={aspect}
                  failed={Boolean(failed[f.frame.name])}
                  onFail={fail(f.frame.name)}
                  onSeek={onSeek}
                  active={near?.name === f.frame.name}
                />
              ))}
            </div>
          ) : null}
          {crops.length > 0 ? (
            <div>
              <h3 className="caption mb-2">Changed elements · A | B</h3>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
                {crops.map((k) => (
                  <Thumb
                    key={k.name}
                    frame={k}
                    label={elements.find((e) => e.id === k.element_id)?.label ?? k.element_id ?? k.name}
                    aspect="2 / 1"
                    failed={Boolean(failed[k.name])}
                    onFail={fail(k.name)}
                    onSeek={null}
                    active={false}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </Panel>
  );
}
