"use client";

import Link from "next/link";

import { useEffect, useRef, type ReactNode } from "react";

import { IconArrowLeft, IconCheck } from "@/components/icons";
import { InteractionSummary } from "@/components/result/InteractionSummary";
import { KeyframeStrip } from "@/components/result/KeyframeStrip";
import { MotionTimeline } from "@/components/result/MotionTimeline";
import { SpecTabs } from "@/components/result/SpecTabs";
import { usePlayback } from "@/components/result/usePlayback";
import { VideoPlayer } from "@/components/result/VideoPlayer";
import { WarningsPanel } from "@/components/result/WarningsPanel";
import { CopyButton } from "@/components/ui/CopyButton";
import { assetUrl } from "@/lib/api";
import { formatPixelRatio, INTERACTION_TYPE_LABELS, INTERPRETATION_STATUS_LABELS } from "@/lib/format";
import type { ResultEnvelope } from "@/lib/types";

export function ResultView({
  result,
  labelAction = null,
  relabeled = false,
  owner = null,
  actions = null,
}: {
  result: ResultEnvelope;
  /** "Label with AI" flow, rendered inside the notes about heuristic labels. */
  labelAction?: ReactNode;
  /** A relabel just finished with AI labels: confirm it next to the status line. */
  relabeled?: boolean;
  /** "Owner: alice" line (admins viewing another account's job). */
  owner?: ReactNode;
  /** Job-level actions (delete), next to the format links. */
  actions?: ReactNode;
}) {
  const { spec, outputs, artifacts } = result;
  const { source, interaction, interpretation } = spec;
  const src = assetUrl(artifacts.video_url);
  const fps = source.fps_effective > 0 ? source.fps_effective : 60;
  const playback = usePlayback(src, 1000 / fps, source.duration_ms);
  const relabeledRef = useRef<HTMLParagraphElement>(null);

  // The relabel controls unmount when AI labels arrive; give keyboard focus a place to land.
  useEffect(() => {
    if (relabeled && (!document.activeElement || document.activeElement === document.body)) relabeledRef.current?.focus();
  }, [relabeled]);

  return (
    <div className="mx-auto max-w-[1240px] space-y-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0 space-y-1.5">
          <Link href="/" className="focus-ring -ml-1 inline-flex items-center gap-1.5 rounded-sm px-1 text-sm text-ink-3 hover:text-ink">
            <IconArrowLeft />
            New analysis
          </Link>
          <p className="caption">
            {INTERACTION_TYPE_LABELS[interaction.type]} · {interaction.target_label}
          </p>
          <h1 className="text-lg font-semibold tracking-tight break-all text-ink sm:text-xl">{source.filename}</h1>
          <p className="nums text-xs text-ink-3">
            {source.width} × {source.height} · {formatPixelRatio(source.pixel_ratio)} · {formatFps(source.fps_effective)} fps
            {source.is_vfr ? " (variable)" : ""} · {(source.duration_ms / 1000).toFixed(2)} s · {source.codec}
            <span className="font-sans"> · {INTERPRETATION_STATUS_LABELS[interpretation.status]}</span>
            {interpretation.status === "ok" && interpretation.model ? ` (${interpretation.model})` : ""}
          </p>
          {owner}
          <p ref={relabeledRef} tabIndex={-1} className="inline-flex items-center gap-1.5 rounded-sm text-xs text-ok outline-none" role="status">
            {relabeled ? (
              <>
                <IconCheck size={14} />
                Labels updated by AI. Prompt, JSON and CSS were regenerated; measurements are unchanged.
              </>
            ) : null}
          </p>
        </div>
        <div className="flex flex-col items-start gap-1.5 sm:items-end">
          <CopyButton text={outputs.llm_prompt} label="Copy LLM prompt" variant="primary" size="md" />
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 sm:justify-end">
            <a href="#spec" className="focus-ring rounded-sm text-xs text-ink-3 underline-offset-4 hover:text-ink hover:underline">
              All formats: technical, JSON, CSS
            </a>
            {actions}
          </div>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <VideoPlayer src={src} playback={playback} source={source} segments={spec.segments} keyframes={artifacts.keyframes} />
          <p className="mt-2 text-xs text-ink-3">{spec.meta.disclaimer}</p>
        </div>
        <div className="min-w-0 space-y-6 lg:col-span-5">
          <InteractionSummary spec={spec} />
          <WarningsPanel spec={spec} labelAction={labelAction} />
        </div>
      </div>

      <MotionTimeline spec={spec} playback={playback} />

      <KeyframeStrip
        keyframes={artifacts.keyframes}
        elements={spec.elements}
        source={source}
        playheadMs={playback.currentMs}
        onSeek={playback.seek}
      />

      <SpecTabs outputs={outputs} disclaimer={spec.meta.disclaimer} />
    </div>
  );
}

function formatFps(fps: number): string {
  return Number.isInteger(fps) ? String(fps) : fps.toFixed(1);
}
