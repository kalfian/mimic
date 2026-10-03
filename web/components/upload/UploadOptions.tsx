"use client";

import { CONFIDENTIAL_WARNING, interpreterDestination, KEYFRAMES_SENT } from "@/components/interpreterCopy";
import { InterpreterStatus } from "@/components/upload/InterpreterStatus";
import type { UploadOptions as Options } from "@/lib/api";
import type { UseHealthResult, UseInterpreterCheckResult } from "@/lib/useHealth";
import type { PixelRatioOption } from "@/lib/types";

const PIXEL_RATIOS: { value: PixelRatioOption; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "guess" },
  { value: "1", label: "1x", hint: "standard" },
  { value: "2", label: "2x", hint: "Retina" },
  { value: "3", label: "3x", hint: "phone" },
];

export function UploadOptions({
  options,
  onChange,
  health,
  check,
  disabled,
  canTestInterpreter,
}: {
  options: Options;
  onChange: (next: Options) => void;
  health: UseHealthResult;
  check: UseInterpreterCheckResult;
  disabled: boolean;
  /** Admins only: "Test connection" (POST /api/interpreter/check is admin-only). */
  canTestInterpreter: boolean;
}) {
  const available = health.health?.interpreter?.available === true;
  const aiOn = options.useInterpreter && available;

  return (
    <div className="space-y-8">
      <fieldset disabled={disabled} className="min-w-0">
        <legend className="mb-1 flex items-baseline gap-3 text-md font-medium text-ink">
          <span className="nums text-2xs text-ink-3">02</span>
          Display scale
        </legend>
        <p id="ratio-help" className="mb-3 max-w-prose text-sm text-ink-2 sm:ml-8">
          Video pixels ÷ scale = CSS px. Auto infers it from the resolution (a 2880 × 1800 capture reads as 2x). Set it if you know the screen: a
          wrong scale halves or doubles every distance.
        </p>
        <div className="inline-grid grid-cols-4 sm:ml-8 rounded-md border border-line-strong bg-surface p-0.5" aria-describedby="ratio-help">
          {PIXEL_RATIOS.map((r) => (
            <label
              key={r.value}
              className="relative flex min-w-[4rem] cursor-pointer flex-col items-center rounded-[4px] px-3 py-1.5 text-center transition-colors duration-150 hover:bg-surface-2 has-[input:checked]:bg-ink has-[input:checked]:text-on-ink has-[input:disabled]:cursor-not-allowed has-[input:focus-visible]:outline-2 has-[input:focus-visible]:outline-offset-2 has-[input:focus-visible]:outline-[var(--focus)]"
            >
              <input
                type="radio"
                name="pixel_ratio"
                value={r.value}
                checked={options.pixelRatio === r.value}
                onChange={() => onChange({ ...options, pixelRatio: r.value })}
                className="sr-only"
              />
              <span className="text-sm font-medium">{r.label}</span>
              <span className="font-mono text-2xs opacity-70">{r.hint}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset disabled={disabled} className="min-w-0">
        <legend className="mb-3 flex items-baseline gap-3 text-md font-medium text-ink">
          <span className="nums text-2xs text-ink-3">03</span>
          AI labeling
          <span className="rounded-sm border border-line px-1.5 font-mono text-2xs font-normal text-ink-3">opt-in</span>
        </legend>

        <div className="space-y-3 sm:ml-8">
          <div className="flex items-start gap-3">
            <button
              type="button"
              role="switch"
              id="ai-switch"
              aria-checked={aiOn}
              aria-describedby="ai-help ai-disclosure"
              disabled={disabled || !available}
              onClick={() => onChange({ ...options, useInterpreter: !options.useInterpreter })}
              className={`focus-ring relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45 ${
                aiOn ? "border-ink bg-ink" : "border-line-strong bg-surface-2"
              }`}
            >
              <span
                aria-hidden="true"
                className={`size-3.5 rounded-full shadow-[0_0_0_1px_var(--line-strong)] transition-transform duration-150 ease-out ${
                  aiOn ? "translate-x-[18px] bg-on-ink" : "translate-x-[2px] bg-surface"
                }`}
              />
            </button>
            <div className="min-w-0">
              <label htmlFor="ai-switch" className={`font-medium ${available ? "cursor-pointer text-ink" : "text-ink-3"}`}>
                Use AI labeling
                <span className="ml-2 font-mono text-2xs font-normal text-ink-3">{aiOn ? "on" : "off"}</span>
              </label>
              <p id="ai-help" className="max-w-prose text-sm text-ink-2">
                Names elements (“Product image” instead of “Image (e2)”) and describes the layout. Every measured number comes from the video, never
                from the AI.
              </p>
            </div>
          </div>

          <p
            id="ai-disclosure"
            className={`max-w-prose rounded-md border px-3 py-2 text-sm ${
              aiOn ? "border-signal/40 bg-signal-soft text-ink" : "border-line text-ink-2"
            }`}
          >
            {!available ? (
              "Unavailable on this server, so nothing from your recording leaves the API."
            ) : (
              <>
                {aiOn ? "On: " : "If you turn this on, "}
                {KEYFRAMES_SENT} are sent to {interpreterDestination(health.health?.interpreter)}. {CONFIDENTIAL_WARNING}
              </>
            )}
          </p>

          <InterpreterStatus health={health} check={check} canTest={canTestInterpreter} />
        </div>
      </fieldset>
    </div>
  );
}
