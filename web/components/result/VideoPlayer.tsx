"use client";

import { useState, type KeyboardEvent } from "react";

import { IconInfo, IconPause, IconPlay, IconStepBack, IconStepForward } from "@/components/icons";
import { PLAYBACK_RATES, type Playback } from "@/components/result/usePlayback";
import { assetUrl } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { KeyframeArtifact, Segment, Source } from "@/lib/types";

const ctrl =
  "focus-ring inline-flex size-8 items-center justify-center rounded-md text-ink-2 transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent";

function secs(ms: number) {
  return (ms / 1000).toFixed(3);
}

export function VideoPlayer({
  src,
  playback,
  source,
  segments,
  keyframes,
}: {
  src: string | null;
  playback: Playback;
  source: Source;
  segments: Segment[];
  keyframes: KeyframeArtifact[];
}) {
  const { videoRef, videoProps, status, playing, currentMs, durationMs, frameMs, rate } = playback;
  const ready = status === "ready";
  const unavailable = !src || status === "error";
  const aspect = `${source.width} / ${source.height}`;
  const frame = Math.round(currentMs / frameMs);

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === ",") playback.step(-1);
    else if (e.key === ".") playback.step(1);
  };

  return (
    <figure aria-label="Original recording" className="overflow-hidden rounded-md border border-line bg-surface">
      {unavailable ? (
        <KeyframeFallback keyframes={keyframes} width={source.width} height={source.height} />
      ) : (
        <div className="relative bg-[#0c0c0d]" style={{ aspectRatio: aspect }}>
          {src ? (
            <video
              ref={videoRef}
              src={src}
              playsInline
              muted
              preload="auto"
              onClick={playback.toggle}
              className="absolute inset-0 size-full object-contain"
              {...videoProps}
            />
          ) : null}
          {status === "loading" ? (
            <div className="absolute inset-0 flex items-center justify-center font-mono text-xs text-[#a1a19a]" role="status">
              loading preview…
            </div>
          ) : null}
        </div>
      )}

      {unavailable ? (
        <figcaption className="flex items-start gap-2 border-t border-line px-4 py-2.5 text-sm text-ink-2">
          <IconInfo className="mt-0.5 shrink-0 text-ink-3" />
          {src ? "The preview could not be played in this browser." : "No browser preview for this recording."} Showing the start and end
          keyframes instead. Timeline bars still move the playhead.
        </figcaption>
      ) : (
        <div role="group" aria-label="Playback controls" onKeyDown={onKeyDown} className="border-t border-line">
          <Scrubber playback={playback} segments={segments} disabled={!ready} />
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-2 pb-2">
            <div className="flex items-center">
              <button type="button" className={ctrl} onClick={playback.toggle} disabled={!ready} aria-label={playing ? "Pause" : "Play"}>
                {playing ? <IconPause /> : <IconPlay />}
              </button>
              <button type="button" className={ctrl} onClick={() => playback.step(-1)} disabled={!ready} aria-label="Previous frame" title="Previous frame (,)">
                <IconStepBack />
              </button>
              <button type="button" className={ctrl} onClick={() => playback.step(1)} disabled={!ready} aria-label="Next frame" title="Next frame (.)">
                <IconStepForward />
              </button>
            </div>
            <p className="nums text-xs text-ink-2" aria-label={`Time ${secs(currentMs)} of ${secs(durationMs)} seconds, frame ${frame}`}>
              <span className="text-ink">{secs(currentMs)}</span>
              <span className="text-ink-3"> / {secs(durationMs)} s</span>
              <span className="ml-3 text-ink-3">f {String(frame).padStart(3, "0")}</span>
            </p>
            <div className="ml-auto inline-flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Playback speed">
              {PLAYBACK_RATES.map((r) => (
                <button
                  key={r}
                  type="button"
                  role="radio"
                  aria-checked={rate === r}
                  disabled={!ready}
                  onClick={() => playback.setRate(r)}
                  className="focus-ring nums rounded-[4px] px-2 py-0.5 text-xs text-ink-2 transition-colors duration-150 hover:text-ink disabled:opacity-40 aria-checked:bg-ink aria-checked:text-on-ink"
                >
                  {formatNumber(r, 2)}×
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </figure>
  );
}

/** Seek bar with the analysed segments drawn underneath (forward = signal, reverse = graphite). */
function Scrubber({ playback, segments, disabled }: { playback: Playback; segments: Segment[]; disabled: boolean }) {
  const { durationMs, currentMs, frameMs } = playback;
  const pct = (ms: number) => `${Math.min(100, Math.max(0, (ms / durationMs) * 100))}%`;
  return (
    <div className="relative px-3 pt-3 pb-1">
      <div className="pointer-events-none absolute inset-x-3 top-[1.375rem] h-1" aria-hidden="true">
        {segments.map((s) => (
          <span
            key={s.id}
            className={`absolute top-0 h-1 rounded-[1px] ${s.kind === "forward" ? "bg-signal-fill" : "bg-rev-fill"}`}
            style={{ left: pct(s.onset_ms), width: `calc(${pct(s.settle_ms)} - ${pct(s.onset_ms)})` }}
          />
        ))}
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(1, Math.round(durationMs))}
        step={Math.max(1, Math.round(frameMs))}
        value={Math.round(currentMs)}
        disabled={disabled}
        onChange={(e) => playback.seek(Number(e.target.value))}
        aria-label="Seek"
        aria-valuetext={`${(currentMs / 1000).toFixed(2)} seconds`}
        className="focus-ring relative h-5 w-full cursor-pointer accent-[var(--ink)] disabled:cursor-not-allowed"
      />
    </div>
  );
}

/**
 * Start/end keyframes, used when there is no playable preview. Each frame keeps the recording's
 * own aspect ratio (no letterboxing): landscape recordings sit side by side from `sm` up and stack
 * on phones; portrait recordings always sit side by side, capped in height.
 */
function KeyframeFallback({ keyframes, width, height }: { keyframes: KeyframeArtifact[]; width: number; height: number }) {
  const a = keyframes.find((k) => k.kind === "state_a");
  const b = keyframes.find((k) => k.kind === "state_b");
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const frames = [a, b].filter((k): k is KeyframeArtifact => Boolean(k));
  const w = width > 0 ? width : 16;
  const h = height > 0 ? height : 10;
  const portrait = h > w;

  if (frames.length === 0) {
    return (
      <div className="hatch flex items-center justify-center font-mono text-xs text-ink-3" style={{ aspectRatio: `${w} / ${h}`, maxHeight: "24rem" }}>
        no preview or keyframes
      </div>
    );
  }
  const cols = frames.length === 1 ? "grid-cols-1" : portrait ? "grid-cols-2" : "grid-cols-1 sm:grid-cols-2";
  return (
    <div className={`grid gap-px bg-[#2a2a2d] ${cols}`}>
      {frames.map((k) => (
        <div key={k.name} className="relative bg-[#0c0c0d]">
          <div className="mx-auto w-full" style={{ aspectRatio: `${w} / ${h}`, maxWidth: portrait ? `${(28 * w) / h}rem` : undefined }}>
            {failed[k.name] ? (
              <div className="flex size-full items-center justify-center font-mono text-xs text-[#a1a19a]">image unavailable</div>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element -- API-served keyframe; dimensions come from the recording
              <img
                src={assetUrl(k.url)}
                alt={k.kind === "state_a" ? "Start state keyframe" : "End state keyframe"}
                width={w}
                height={h}
                className="size-full object-contain"
                onError={() => setFailed((f) => ({ ...f, [k.name]: true }))}
              />
            )}
          </div>
          <span className="absolute top-2 left-2 rounded-sm bg-black/70 px-1.5 font-mono text-2xs text-white">
            {k.kind === "state_a" ? "A · start" : "B · end"}
          </span>
        </div>
      ))}
    </div>
  );
}
