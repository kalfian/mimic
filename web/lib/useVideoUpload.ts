/**
 * React hook wrapping `uploadVideo` for the upload page (client components only).
 *
 *   const { state, progress, error, job, upload, cancel, reset } = useVideoUpload();
 *   const created = await upload(file, { pixelRatio: "auto", useInterpreter: false });
 *   if (created) router.push(`/jobs/${created.id}`);
 *
 * `upload` resolves with `JobCreated` on success or `null` on failure/cancel (the error is in
 * state), so callers never need try/catch. Starting a new upload cancels the previous one.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { uploadVideo, type UploadOptions, type UploadProgress, type UploadRequestOptions } from "./api";
import { ApiError } from "./errors";
import type { JobCreated } from "./types";

export type UploadState = "idle" | "uploading" | "done" | "error";

export interface UseVideoUploadResult {
  state: UploadState;
  /** Latest progress while uploading (null before the first progress event). */
  progress: UploadProgress | null;
  error: ApiError | null;
  job: JobCreated | null;
  upload: (file: File, options?: Partial<UploadOptions>, request?: Omit<UploadRequestOptions, "signal">) => Promise<JobCreated | null>;
  cancel: () => void;
  reset: () => void;
}

export function useVideoUpload(): UseVideoUploadResult {
  const [state, setState] = useState<UploadState>("idle");
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [job, setJob] = useState<JobCreated | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const upload = useCallback<UseVideoUploadResult["upload"]>(async (file, options, request) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState("uploading");
    setProgress(null);
    setError(null);
    setJob(null);
    try {
      const created = await uploadVideo(
        file,
        options,
        (p) => {
          if (!controller.signal.aborted) setProgress(p);
        },
        { ...request, signal: controller.signal },
      );
      if (controller.signal.aborted) return null;
      setJob(created);
      setState("done");
      return created;
    } catch (err) {
      const e = ApiError.from(err);
      if (controller.signal.aborted || e.code === "aborted") {
        // Cancelled by cancel()/reset()/a newer upload: those set their own state.
        return null;
      }
      setError(e);
      setState("error");
      return null;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState("idle");
    setProgress(null);
    setError(null);
    setJob(null);
  }, []);

  return { state, progress, error, job, upload, cancel: reset, reset };
}
