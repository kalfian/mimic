"use client";

import { IconAlert, IconCheck, IconCross, IconRefresh } from "@/components/icons";
import { Button } from "@/components/ui/Button";
import type { InterpreterCheck, InterpreterCheckErrorCode, InterpreterMode } from "@/lib/api";
import {
  describeError,
  formatMs,
  INTERPRETER_CHECK_ERROR_MESSAGES,
  INTERPRETER_MODE_LABELS,
  STRUCTURED_OUTPUT_LABELS,
} from "@/lib/format";
import type { UseHealthResult, UseInterpreterCheckResult } from "@/lib/useHealth";

/** What to do on the server for each failed check. Mentions env var names only, never values. */
function guidanceFor(code: InterpreterCheckErrorCode, mode: InterpreterMode): string {
  switch (code) {
    case "not_configured":
      return mode === "claude_cli"
        ? "Install Claude Code on the API host and make sure the claude binary is on its PATH (or set MIMIC_CLAUDE_BIN)."
        : "Set MIMIC_INTERPRETER to claude_cli or openai_compat in the API’s .env, then restart the API.";
    case "unauthorized":
      return mode === "claude_cli"
        ? "Run claude once on the API host to sign in, then test again."
        : "Check MIMIC_LLM_API_KEY in the API’s environment, then restart the API.";
    case "unreachable":
      return "Check MIMIC_LLM_BASE_URL and that the gateway is running and reachable from the API host.";
    case "timeout":
      return "The endpoint is slow or overloaded. Test again, or raise MIMIC_LLM_TIMEOUT_S.";
    case "model_not_found":
      return "Check MIMIC_LLM_MODEL against the endpoint’s model list.";
    case "no_image_support":
      return "Labeling sends keyframe images. Set MIMIC_LLM_MODEL to a vision-capable model.";
    case "bad_response":
      return "The endpoint may not be fully OpenAI-compatible. The API log has the raw response.";
  }
}

function unavailableReason(mode: InterpreterMode): string {
  switch (mode) {
    case "claude_cli":
      return "The claude binary was not found on the API host.";
    case "openai_compat":
      return "MIMIC_LLM_BASE_URL, MIMIC_LLM_API_KEY or MIMIC_LLM_MODEL is missing on the server.";
    case "none":
      return "AI labeling is turned off on the server (MIMIC_INTERPRETER=none).";
  }
}

const yesNo = (v: boolean | null) => (v == null ? "not checked" : v ? "yes" : "no");

export function InterpreterStatus({ health, check }: { health: UseHealthResult; check: UseInterpreterCheckResult }) {
  if (health.isLoading) {
    return (
      <div className="rounded-md border border-line px-4 py-3 text-sm text-ink-3" role="status">
        Checking the API server…
      </div>
    );
  }

  if (health.error || !health.health) {
    const d = describeError(health.error?.code ?? "network_error", null);
    return (
      <div className="flex flex-wrap items-start justify-between gap-3 rounded-md border border-danger/40 bg-danger-soft px-4 py-3" role="alert">
        <div className="text-sm">
          <p className="font-medium text-danger">{d.title}</p>
          <p className="text-ink-2">
            {d.message} AI labeling is unavailable until it responds.{d.guidance[0] ? ` ${d.guidance[0]}` : ""}
          </p>
        </div>
        <Button size="sm" icon={<IconRefresh />} onClick={health.refresh}>
          Retry
        </Button>
      </div>
    );
  }

  const { mode, available, model } = health.health.interpreter;
  const result = check.check;

  return (
    <div className="rounded-md border border-line">
      <dl className="grid grid-cols-[7rem_minmax(0,1fr)] gap-x-4 gap-y-1.5 px-4 py-3 text-sm">
        <dt className="text-ink-3">Interpreter</dt>
        <dd className="text-ink">{INTERPRETER_MODE_LABELS[mode]}</dd>
        <dt className="text-ink-3">Model</dt>
        <dd className="truncate font-mono text-xs leading-5 text-ink">{model ?? "—"}</dd>
        <dt className="text-ink-3">Status</dt>
        <dd className="flex items-start gap-1.5">
          {available ? (
            <>
              <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-ok" aria-hidden="true" />
              <span className="text-ink">Configured</span>
            </>
          ) : (
            <>
              <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-line-strong" aria-hidden="true" />
              <span className="text-ink-2">Not available. {unavailableReason(mode)}</span>
            </>
          )}
        </dd>
      </dl>

      <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-3">
        <Button size="sm" onClick={() => void check.run()} loading={check.isChecking}>
          {check.isChecking ? "Testing…" : result ? "Test again" : "Test connection"}
        </Button>
        <span className="text-xs text-ink-3">
          {check.isChecking ? "Sends one tiny test request from the server. Can take a few seconds." : "Runs from the API server; nothing from your recording is sent."}
        </span>
      </div>

      <div aria-live="polite">
        {check.error ? (
          <CheckFailure title="Could not run the check" message={describeError(check.error.code, check.error.message).message} />
        ) : result && !check.isChecking ? (
          result.ok ? (
            <CheckSuccess check={result} />
          ) : (
            <CheckFailure
              title={result.error ? INTERPRETER_CHECK_ERROR_MESSAGES[result.error.code] : "Connection check failed."}
              code={result.error?.code}
              message={result.error?.message}
              guidance={result.error ? guidanceFor(result.error.code, result.mode) : undefined}
              check={result}
            />
          )
        ) : null}
      </div>

      <p className="border-t border-line px-4 py-2.5 text-xs text-ink-3">
        Configured through the API server’s environment. This page never asks for, stores or sends API keys.
      </p>
    </div>
  );
}

function CheckFacts({ check }: { check: InterpreterCheck }) {
  const facts: [string, string, boolean?][] = [
    ["Latency", check.latency_ms != null ? formatMs(check.latency_ms) : "—"],
    ["Model", check.model ?? "—"],
    ["Model listed", yesNo(check.model_listed), check.model_listed === false],
    ["Image input", yesNo(check.supports_images), check.supports_images === false],
    [
      "Structured output",
      check.structured_output ? STRUCTURED_OUTPUT_LABELS[check.structured_output] : "—",
      check.structured_output === "prompt_only",
    ],
  ];
  return (
    <dl className="grid grid-cols-[7rem_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
      {facts.map(([k, v, bad]) => (
        <div key={k} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className={`truncate font-mono text-xs leading-5 ${bad ? "text-warn" : "text-ink"}`}>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function CheckSuccess({ check }: { check: InterpreterCheck }) {
  return (
    <div className="space-y-2 border-t border-line bg-ok-soft/60 px-4 py-3">
      <p className="flex items-center gap-2 text-sm font-medium text-ok">
        <IconCheck className="shrink-0" />
        Connected
        <span className="font-mono text-2xs font-normal text-ink-3">{new Date(check.checked_at).toLocaleTimeString()}</span>
      </p>
      <CheckFacts check={check} />
      {check.structured_output === "prompt_only" ? (
        <p className="flex items-start gap-2 text-xs text-ink-2">
          <IconAlert className="mt-px shrink-0 text-warn" />
          The model has no JSON mode, so labels are parsed from plain text and fall back to heuristics more often.
        </p>
      ) : null}
    </div>
  );
}

function CheckFailure({
  title,
  code,
  message,
  guidance,
  check,
}: {
  title: string;
  code?: string;
  message?: string;
  guidance?: string;
  check?: InterpreterCheck;
}) {
  return (
    <div className="space-y-2 border-t border-line bg-danger-soft/70 px-4 py-3" role="alert">
      <p className="flex flex-wrap items-center gap-2 text-sm font-medium text-danger">
        <IconCross className="shrink-0" />
        {title}
        {code ? <span className="font-mono text-2xs font-normal text-ink-3">{code}</span> : null}
      </p>
      {message ? <p className="font-mono text-xs break-words text-ink-2">{message}</p> : null}
      {guidance ? <p className="text-sm text-ink-2">{guidance}</p> : null}
      {check && (check.latency_ms != null || check.model_listed != null || check.supports_images != null) ? <CheckFacts check={check} /> : null}
    </div>
  );
}
