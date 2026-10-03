"use client";

import { useRef, useState, type DragEvent } from "react";

import { IconAlert, IconCheck, IconFilm, IconInfo, IconUpload } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import type { ApiError, UploadProgress } from "@/lib/api";
import { describeError, formatPercent } from "@/lib/format";
import { ACCEPT_ATTRIBUTE, DEFAULT_UPLOAD_LIMITS, fileExtension, type FileCheck, type UploadLimits } from "@/lib/upload";
import type { UploadState } from "@/lib/useVideoUpload";

export interface SelectedFile {
  file: File;
  /** null = the browser could not read it (e.g. HEVC .mov); the server decides. */
  durationS: number | null;
  probing: boolean;
  check: FileCheck;
}

/** Below this the recording is accepted but probably lacks the pauses the pipeline needs. */
const RECOMMENDED_MIN_S = 2;

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadDropzone({
  selected,
  onFile,
  onClear,
  uploadState,
  progress,
  uploadError,
  limits = DEFAULT_UPLOAD_LIMITS,
}: {
  selected: SelectedFile | null;
  onFile: (file: File) => void;
  onClear: () => void;
  uploadState: UploadState;
  progress: UploadProgress | null;
  uploadError: ApiError | null;
  limits?: UploadLimits;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const busy = uploadState === "uploading" || uploadState === "done";

  const openPicker = () => inputRef.current?.click();

  const onDragOver = (e: DragEvent) => {
    if (busy) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (!dragOver) setDragOver(true);
  };
  const onDragLeave = (e: DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
    setDragOver(false);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (busy) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  const invalid = selected && !selected.check.ok ? selected.check : null;
  const ready = selected && selected.check.ok ? selected : null;

  return (
    <div onDragOver={onDragOver} onDragEnter={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) onFile(file);
        }}
      />

      {ready ? (
        <SelectedFileCard
          selected={ready}
          dragOver={dragOver}
          busy={busy}
          uploadState={uploadState}
          progress={progress}
          onReplace={openPicker}
          onClear={onClear}
          rejected={uploadError != null}
        />
      ) : (
        <button
          type="button"
          onClick={openPicker}
          aria-describedby={invalid ? "dropzone-error" : "dropzone-hint"}
          className={`focus-ring group flex w-full flex-col items-start gap-4 rounded-md border border-dashed px-5 py-8 text-left transition-colors duration-150 sm:flex-row sm:items-center sm:px-6 sm:py-10 ${
            dragOver
              ? "border-signal bg-signal-soft"
              : invalid
                ? "border-danger/60 bg-surface hover:border-danger"
                : "border-line-strong bg-surface hover:border-ink-3 hover:bg-surface-2/60"
          }`}
        >
          <span
            className={`flex size-10 shrink-0 items-center justify-center rounded-md border ${
              dragOver ? "border-signal text-signal" : "border-line text-ink-2 group-hover:text-ink"
            }`}
          >
            <IconUpload size={18} />
          </span>
          <span className="flex flex-col gap-1">
            <span className="text-md font-medium text-ink">
              {dragOver ? "Release to use this recording" : (
                <>
                  Drop a screen recording, or <span className="underline decoration-line-strong underline-offset-4 group-hover:decoration-ink">choose a file</span>
                </>
              )}
            </span>
            <span id="dropzone-hint" className="font-mono text-xs text-ink-3">
              MP4 · MOV · WebM — up to {limits.maxDurationS} s (2–{limits.maxDurationS} s works best) — max {limits.maxUploadMb} MB
            </span>
          </span>
        </button>
      )}

      {invalid ? (
        <p id="dropzone-error" role="alert" className="mt-3 flex items-start gap-2 text-sm text-danger">
          <IconAlert className="mt-0.5 shrink-0" />
          <span>
            <span className="font-medium">{selected?.file.name}:</span> {invalid.message}
          </span>
        </p>
      ) : null}

      {uploadError ? <ServerRejection error={uploadError} /> : null}
    </div>
  );
}

function SelectedFileCard({
  selected,
  dragOver,
  busy,
  uploadState,
  progress,
  onReplace,
  onClear,
  rejected,
}: {
  rejected: boolean;
  selected: SelectedFile;
  dragOver: boolean;
  busy: boolean;
  uploadState: UploadState;
  progress: UploadProgress | null;
  onReplace: () => void;
  onClear: () => void;
}) {
  const { file, durationS, probing } = selected;
  const ext = fileExtension(file.name).toUpperCase();
  const short = durationS != null && durationS < RECOMMENDED_MIN_S;
  const fraction = progress?.fraction ?? null;

  return (
    <div
      className={`rounded-md border bg-surface transition-colors duration-150 ${dragOver ? "border-signal bg-signal-soft" : "border-line-strong"}`}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-4 sm:px-5">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-md border border-line text-ink-2">
          <IconFilm size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-ink" title={file.name}>
            {file.name}
          </p>
          <p className="nums text-xs text-ink-3">
            {probing ? "reading duration…" : durationS != null ? `${durationS.toFixed(2)} s` : "duration unknown"} · {formatBytes(file.size)} · {ext}
          </p>
        </div>
        {busy ? null : (
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={onReplace}>
              Replace
            </Button>
            <Button size="sm" variant="ghost" onClick={onClear}>
              Remove
            </Button>
          </div>
        )}
      </div>

      {uploadState === "uploading" || uploadState === "done" ? (
        <div className="border-t border-line px-4 py-3 sm:px-5">
          <div className="mb-2 flex items-baseline justify-between text-sm">
            <span className="text-ink-2" role="status" aria-live="polite">
              {uploadState === "done" ? "Uploaded. Opening the analysis…" : "Uploading"}
            </span>
            <span className="nums text-xs text-ink-3">{fraction != null ? formatPercent(fraction) : "—"}</span>
          </div>
          <div
            role="progressbar"
            aria-label="Upload progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={fraction != null ? Math.round(fraction * 100) : undefined}
            className="h-1 overflow-hidden rounded-full bg-surface-2"
          >
            <div
              className="h-full bg-ink transition-[width] duration-200 ease-out"
              style={{ width: `${(uploadState === "done" ? 1 : (fraction ?? 0)) * 100}%` }}
            />
          </div>
        </div>
      ) : (
        <p className="flex items-start gap-2 border-t border-line px-4 py-2.5 text-sm sm:px-5">
          {rejected ? (
            <>
              <IconAlert className="mt-0.5 shrink-0 text-danger" />
              <span className="text-ink-2">The server rejected this file. Replace it, or try again if the problem was temporary.</span>
            </>
          ) : probing ? (
            <span className="text-ink-3">Checking the file…</span>
          ) : durationS == null ? (
            <>
              <IconInfo className="mt-0.5 shrink-0 text-ink-3" />
              <span className="text-ink-2">This browser can’t read the duration (common for HEVC .mov). The server will check it.</span>
            </>
          ) : short ? (
            <>
              <IconInfo className="mt-0.5 shrink-0 text-warn" />
              <span className="text-ink-2">
                Shorter than {RECOMMENDED_MIN_S} s. It will be analyzed, but leave a pause before and after the interaction for reliable results.
              </span>
            </>
          ) : (
            <>
              <IconCheck className="mt-0.5 shrink-0 text-ok" />
              <span className="text-ink-2">Ready to analyze.</span>
            </>
          )}
        </p>
      )}
    </div>
  );
}

function ServerRejection({ error }: { error: ApiError }) {
  const d = describeError(error.code, error.message);
  return (
    <div role="alert" className="mt-3 rounded-md border border-danger/40 bg-danger-soft px-4 py-3">
      <p className="flex items-center gap-2 font-medium text-danger">
        <IconAlert className="shrink-0" />
        {d.title}
        <span className="font-mono text-2xs font-normal text-ink-3">{error.code}</span>
      </p>
      <p className="mt-1 text-sm text-ink-2">{d.message}</p>
      {d.guidance.length > 0 ? (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-sm text-ink-2 marker:text-ink-3">
          {d.guidance.map((g) => (
            <li key={g}>{g}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
