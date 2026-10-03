import { BAND_LABELS, bandFor, formatPercent } from "@/lib/format";
import type { Confidence, ConfidenceBand } from "@/lib/types";

const LEVEL: Record<ConfidenceBand, number> = { high: 3, medium: 2, low: 1 };
const TONE: Record<ConfidenceBand, string> = { high: "text-ok", medium: "text-ink-2", low: "text-warn" };

/**
 * Band as a 3-step meter + word (never colour alone); the exact % is in the tooltip and the
 * accessible name, or shown inline with `showPercent`.
 */
export function ConfidenceBadge({
  confidence,
  showPercent = false,
  className = "",
}: {
  confidence: Confidence | number;
  showPercent?: boolean;
  className?: string;
}) {
  const value = typeof confidence === "number" ? confidence : confidence.value;
  const band = typeof confidence === "number" ? bandFor(confidence) : confidence.band;
  const pct = formatPercent(value);
  const label = `${BAND_LABELS[band]} confidence, ${pct}`;
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 text-xs font-medium whitespace-nowrap ${TONE[band]} ${className}`}
      title={label}
      aria-label={label}
      role="img"
    >
      <span className="inline-flex h-2.5 items-end gap-px" aria-hidden="true">
        {[1, 2, 3].map((i) => (
          <span
            key={i}
            className={`w-[3px] rounded-[1px] ${i <= LEVEL[band] ? "bg-current" : "bg-line-strong"}`}
            style={{ height: `${4 + i * 2}px` }}
          />
        ))}
      </span>
      <span aria-hidden="true">
        {BAND_LABELS[band]}
        {showPercent ? <span className="nums ml-1 font-normal text-ink-3">{pct}</span> : null}
      </span>
    </span>
  );
}
