/**
 * Client-side pre-checks for the upload dropzone (PLAN §10.3 UploadDropzone).
 *
 * Limits come from `GET /api/health` (`limitsFromHealth`), defaulting to the backend defaults
 * (`MIMIC_MAX_UPLOAD_MB`, `MIMIC_MAX_DURATION_S`, ...) so users get instant feedback. The server stays authoritative: when a check can't run (e.g. HEVC .mov the
 * browser can't decode), skip it and let the upload's error code decide.
 */

import type { ErrorCode, Health } from "./types";

/** Extensions accepted by `POST /api/jobs` (api/schemas.py ALLOWED_EXTENSIONS). */
export const ACCEPTED_EXTENSIONS = ["mp4", "mov", "m4v", "webm"] as const;

/** Value for `<input type="file" accept=…>`. */
export const ACCEPT_ATTRIBUTE = ".mp4,.mov,.m4v,.webm,video/mp4,video/quicktime,video/x-m4v,video/webm";

/** Upload limits; the server's real values come from `GET /api/health` (`health.limits`). */
export interface UploadLimits {
  maxUploadMb: number;
  maxDurationS: number;
  minDurationS: number;
  /** Extra seconds tolerated over `maxDurationS` (MIMIC_DURATION_TOLERANCE_S). */
  durationToleranceS: number;
}

/** Backend defaults, used until (or if) health is unavailable. */
export const DEFAULT_UPLOAD_LIMITS: Readonly<UploadLimits> = Object.freeze({
  maxUploadMb: 200,
  maxDurationS: 15,
  minDurationS: 0.5,
  durationToleranceS: 0.5,
});

export const MAX_UPLOAD_MB = DEFAULT_UPLOAD_LIMITS.maxUploadMb;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
export const MAX_DURATION_S = DEFAULT_UPLOAD_LIMITS.maxDurationS;
export const MIN_DURATION_S = DEFAULT_UPLOAD_LIMITS.minDurationS;
/** Extra seconds tolerated over MAX_DURATION_S (MIMIC_DURATION_TOLERANCE_S). */
export const DURATION_TOLERANCE_S = DEFAULT_UPLOAD_LIMITS.durationToleranceS;

/** Limits reported by the server (`Health.limits`), falling back to the defaults. */
export function limitsFromHealth(health: Pick<Health, "limits"> | null | undefined): UploadLimits {
  const l = health?.limits;
  if (!l) return { ...DEFAULT_UPLOAD_LIMITS };
  return {
    maxUploadMb: l.max_upload_mb,
    maxDurationS: l.max_duration_s,
    minDurationS: l.min_duration_s,
    durationToleranceS: l.duration_tolerance_s,
  };
}

export type FileCheck =
  | { ok: true }
  | { ok: false; code: Extract<ErrorCode, "unsupported_format" | "file_too_large" | "too_long" | "too_short">; message: string };

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Extension + size check. MIME type is not trusted (browsers report `""` for .mov often). */
export function checkVideoFile(
  file: Pick<File, "name" | "size">,
  limits: UploadLimits = DEFAULT_UPLOAD_LIMITS,
): FileCheck {
  const ext = fileExtension(file.name);
  if (!(ACCEPTED_EXTENSIONS as readonly string[]).includes(ext)) {
    return { ok: false, code: "unsupported_format", message: "Upload an MP4, MOV, M4V or WebM video." };
  }
  if (file.size > limits.maxUploadMb * 1024 * 1024) {
    return { ok: false, code: "file_too_large", message: `The file is larger than ${limits.maxUploadMb} MB.` };
  }
  return { ok: true };
}

/** Duration check; pass the value from `probeVideoDuration`. `null` (unknown) always passes. */
export function checkVideoDuration(
  durationS: number | null,
  limits: UploadLimits = DEFAULT_UPLOAD_LIMITS,
): FileCheck {
  if (durationS == null || !Number.isFinite(durationS)) return { ok: true };
  if (durationS > limits.maxDurationS + limits.durationToleranceS) {
    return { ok: false, code: "too_long", message: `The recording is ${durationS.toFixed(1)} s. Trim it to ${limits.maxDurationS} s or less.` };
  }
  if (durationS < limits.minDurationS) {
    return { ok: false, code: "too_short", message: `The recording is shorter than ${limits.minDurationS} s.` };
  }
  return { ok: true };
}

/**
 * Read a video's duration (seconds) from `<video>` metadata. Resolves `null` if the browser can't
 * decode it (e.g. HEVC .mov in Chrome) or it takes longer than `timeoutMs`. Browser-only.
 */
export function probeVideoDuration(file: Blob, timeoutMs = 5000): Promise<number | null> {
  if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") return Promise.resolve(null);
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    let settled = false;
    const finish = (value: number | null) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
      resolve(value);
    };
    const timer = window.setTimeout(() => finish(null), timeoutMs);
    video.preload = "metadata";
    video.muted = true;
    video.onloadedmetadata = () => finish(Number.isFinite(video.duration) ? video.duration : null);
    video.onerror = () => finish(null);
    video.src = url;
  });
}
