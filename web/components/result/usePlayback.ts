import { useCallback, useEffect, useMemo, useRef, useState, type SyntheticEvent } from "react";

export type PlaybackStatus = "none" | "loading" | "ready" | "error";

export const PLAYBACK_RATES = [0.25, 0.5, 1] as const;

export interface Playback {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Playhead in absolute video ms. Moves even without a video (bars and keyframes set it). */
  currentMs: number;
  durationMs: number;
  playing: boolean;
  rate: number;
  status: PlaybackStatus;
  frameMs: number;
  seek: (ms: number) => void;
  toggle: () => void;
  step: (frames: number) => void;
  setRate: (rate: number) => void;
  /** Spread onto the <video>. */
  videoProps: {
    onLoadedMetadata: (e: SyntheticEvent<HTMLVideoElement>) => void;
    onTimeUpdate: (e: SyntheticEvent<HTMLVideoElement>) => void;
    onSeeked: (e: SyntheticEvent<HTMLVideoElement>) => void;
    onPlay: () => void;
    onPause: () => void;
    onEnded: () => void;
    onError: () => void;
  };
}

/**
 * Shared clock between the video, the timeline playhead and the keyframe strip.
 * While playing, the playhead follows `requestAnimationFrame` (timeupdate alone is ~4 Hz).
 */
export function usePlayback(src: string | null, frameMs: number, specDurationMs: number): Playback {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [durationMs, setDurationMs] = useState(specDurationMs);
  const [playing, setPlaying] = useState(false);
  const [rate, setRateState] = useState(1);
  const [status, setStatus] = useState<PlaybackStatus>(src ? "loading" : "none");

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) setCurrentMs(v.currentTime * 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const clamp = useCallback((ms: number) => Math.min(Math.max(0, ms), durationMs), [durationMs]);

  const seek = useCallback(
    (ms: number) => {
      const t = clamp(ms);
      const v = videoRef.current;
      if (v && status === "ready") v.currentTime = t / 1000;
      setCurrentMs(t);
    },
    [clamp, status],
  );

  const toggle = useCallback(() => {
    const v = videoRef.current;
    if (!v || status !== "ready") return;
    if (v.paused) {
      if (v.ended || v.currentTime * 1000 >= durationMs - 1) v.currentTime = 0;
      void v.play().catch(() => setPlaying(false));
    } else {
      v.pause();
    }
  }, [status, durationMs]);

  const step = useCallback(
    (frames: number) => {
      const v = videoRef.current;
      if (v && !v.paused) v.pause();
      const base = v && status === "ready" ? v.currentTime * 1000 : currentMs;
      seek(base + frames * frameMs);
    },
    [currentMs, frameMs, seek, status],
  );

  const setRate = useCallback((r: number) => {
    const v = videoRef.current;
    if (v) v.playbackRate = r;
    setRateState(r);
  }, []);

  const videoProps = useMemo<Playback["videoProps"]>(
    () => ({
      onLoadedMetadata: (e) => {
        const v = e.currentTarget;
        if (Number.isFinite(v.duration) && v.duration > 0) setDurationMs(v.duration * 1000);
        v.playbackRate = rate;
        setStatus("ready");
      },
      onTimeUpdate: (e) => setCurrentMs(e.currentTarget.currentTime * 1000),
      onSeeked: (e) => setCurrentMs(e.currentTarget.currentTime * 1000),
      onPlay: () => setPlaying(true),
      onPause: () => setPlaying(false),
      onEnded: () => setPlaying(false),
      onError: () => {
        setPlaying(false);
        setStatus("error");
      },
    }),
    [rate],
  );

  return { videoRef, currentMs, durationMs, playing, rate, status, frameMs, seek, toggle, step, setRate, videoProps };
}
