"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { Suspense, useRef, useState, type FormEvent } from "react";

import { Button } from "@/components/ui/Button";
import { RecordingGuidelines } from "@/components/upload/RecordingGuidelines";
import { UploadDropzone, type SelectedFile } from "@/components/upload/UploadDropzone";
import { UploadOptions } from "@/components/upload/UploadOptions";
import { DEFAULT_UPLOAD_OPTIONS, IS_MOCK_API, type UploadOptions as Options } from "@/lib/api";
import { isAdminSession, useSession } from "@/lib/session";
import { checkVideoDuration, checkVideoFile, limitsFromHealth, probeVideoDuration } from "@/lib/upload";
import { useHealth, useInterpreterCheck } from "@/lib/useHealth";
import { useVideoUpload } from "@/lib/useVideoUpload";

// Loaded only in mock mode, so the fixture never ships in a real build's initial bundle.
const MockScenarioNote = dynamic(() => import("@/components/upload/MockScenarioNote"), { ssr: false });

export function UploadScreen() {
  const router = useRouter();
  const health = useHealth();
  const check = useInterpreterCheck();
  const upload = useVideoUpload();
  const isAdmin = isAdminSession(useSession());

  const [selected, setSelected] = useState<SelectedFile | null>(null);
  const [options, setOptions] = useState<Options>({ ...DEFAULT_UPLOAD_OPTIONS });
  const probeToken = useRef(0);

  const aiAvailable = health.health?.interpreter?.available === true;
  const limits = limitsFromHealth(health.health);
  const busy = upload.state === "uploading" || upload.state === "done";
  const canSubmit = Boolean(selected?.check.ok) && !selected?.probing && !busy;

  const handleFile = async (file: File) => {
    upload.reset();
    const token = ++probeToken.current;
    const basic = checkVideoFile(file, limits);
    if (!basic.ok) {
      setSelected({ file, durationS: null, probing: false, check: basic });
      return;
    }
    setSelected({ file, durationS: null, probing: true, check: basic });
    const durationS = await probeVideoDuration(file);
    if (token !== probeToken.current) return;
    setSelected({ file, durationS, probing: false, check: checkVideoDuration(durationS, limits) });
  };

  const clear = () => {
    probeToken.current += 1;
    upload.reset();
    setSelected(null);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!selected || !canSubmit) return;
    const created = await upload.upload(selected.file, {
      pixelRatio: options.pixelRatio,
      // Opt-in, and only when the server actually has an interpreter.
      useInterpreter: options.useInterpreter && aiAvailable,
    });
    if (created) router.push(`/jobs/${encodeURIComponent(created.id)}`);
  };

  return (
    <div className="mx-auto grid max-w-[1240px] gap-10 px-4 py-8 sm:px-6 lg:grid-cols-[minmax(0,1fr)_21rem] lg:gap-12 lg:px-8 lg:py-12">
      <form onSubmit={onSubmit} noValidate className="min-w-0 space-y-10">
        <header className="max-w-2xl space-y-3">
          <h1 className="text-xl font-semibold tracking-tight text-ink">Analyze a UI recording</h1>
          <p className="text-md text-ink-2">
            Mimic measures what moved, by how much, when and with which easing, then writes it up as a spec and as a prompt for a coding LLM.
            Numbers are estimated from pixels, not read from the original source.
          </p>
        </header>

        <section aria-labelledby="recording-legend" className="space-y-3">
          <h2 id="recording-legend" className="flex items-baseline gap-3 text-md font-medium text-ink">
            <span className="nums text-2xs text-ink-3">01</span>
            Recording
          </h2>
          <div className="sm:ml-8">
            <UploadDropzone
              selected={selected}
              onFile={(f) => void handleFile(f)}
              onClear={clear}
              uploadState={upload.state}
              progress={upload.progress}
              uploadError={upload.error}
              limits={limits}
            />
          </div>
        </section>

        <UploadOptions options={options} onChange={setOptions} health={health} check={check} disabled={busy} canTestInterpreter={isAdmin} />

        <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-line pt-6">
          <Button type="submit" variant="primary" disabled={!canSubmit} loading={upload.state === "uploading"}>
            {upload.state === "uploading" ? "Uploading…" : "Analyze recording"}
          </Button>
          {upload.state === "uploading" ? (
            <Button variant="ghost" onClick={upload.cancel}>
              Cancel
            </Button>
          ) : null}
          <p className="text-sm text-ink-3">
            {selected?.check.ok
              ? options.useInterpreter && aiAvailable
                ? "Usually 5–15 s. AI labeling can add up to ~2 min."
                : "Usually 5–15 s."
              : "Choose a recording to continue."}
          </p>
        </div>

        {IS_MOCK_API ? (
          <Suspense fallback={null}>
            <MockScenarioNote />
          </Suspense>
        ) : null}
      </form>

      <aside className="min-w-0 lg:pt-1">
        <div className="lg:sticky lg:top-6">
          <RecordingGuidelines />
        </div>
      </aside>
    </div>
  );
}
