import type { LabelSource } from "@/lib/types";

/**
 * Who named an element: the AI interpreter or the heuristics. `chip` is the bordered tag used in
 * prose (summary, inspector); `inline` is the borderless mark used inside dense rows (timeline).
 */
export function LabelSourceTag({ source, variant = "chip", className = "" }: { source: LabelSource; variant?: "chip" | "inline"; className?: string }) {
  const ai = source === "interpreter";
  const title = ai ? "Named by the AI interpreter" : "Named by heuristics (no AI)";
  const look = variant === "chip" ? "rounded-sm border border-line px-1" : "";
  return (
    <span className={`shrink-0 font-mono text-2xs text-ink-3 ${look} ${className}`} title={title}>
      <span aria-hidden="true">{ai ? "AI" : "heuristic"}</span>
      <span className="sr-only">{ai ? "(named by AI)" : "(named by heuristics)"}</span>
    </span>
  );
}
