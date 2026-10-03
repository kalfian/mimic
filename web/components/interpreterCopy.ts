import type { HealthInterpreter } from "@/lib/types";

/**
 * Where keyframes go when AI labeling runs. Shared by the upload toggle and the result page's
 * "Label with AI" action so both disclose the same destination in the same words.
 */
export function interpreterDestination(interpreter: HealthInterpreter | null | undefined): string {
  if (!interpreter) return "the LLM configured on the API server";
  const model = interpreter.model ? ` (model ${interpreter.model})` : "";
  if (interpreter.mode === "claude_cli") return `Claude through the Claude Code CLI on the API server${model}`;
  if (interpreter.mode === "openai_compat") return `the OpenAI-compatible endpoint configured on the API server${model}`;
  return "the LLM configured on the API server";
}

/** What is sent. Completed by "… are sent to {destination}." */
export const KEYFRAMES_SENT = "keyframe images from this recording (stills of the start, middle and end states)";
export const CONFIDENTIAL_WARNING = "Don’t use it on recordings of confidential screens.";
