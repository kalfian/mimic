import { BAND_LABELS, formatPercent, type AppearanceFact } from "@/lib/format";

/** Short labels for dense rows; the full label (`fact.label`) goes into the tooltip / accessible text. */
const SHORT: Record<AppearanceFact["key"], string> = {
  background_color: "bg",
  text_color: "text",
  font_size_px: "font",
  border_radius_px: "radius",
  shadow: "shadow",
};

export function ColorSwatch({ hex, className = "" }: { hex: string; className?: string }) {
  return <span className={`inline-block size-3 shrink-0 rounded-[2px] border border-line-strong ${className}`} style={{ background: hex }} aria-hidden="true" />;
}

/**
 * Measured static appearance (PLAN-continuous §13), deliberately quiet: mono values, colour as a
 * swatch + hex. Hedged per confidence without relying on colour: low-confidence values get a "?"
 * and are muted; the band and % are in the tooltip and the accessible text.
 */
export function AppearanceFactList({ facts, className = "" }: { facts: AppearanceFact[]; className?: string }) {
  if (facts.length === 0) return null;
  return (
    <ul className={`flex flex-wrap gap-x-3.5 gap-y-1 text-xs ${className}`}>
      {facts.map((f) => {
        const low = f.confidence.band === "low";
        const conf = `${BAND_LABELS[f.confidence.band].toLowerCase()} confidence, ${formatPercent(f.confidence.value)}`;
        return (
          <li key={f.key} className="inline-flex min-w-0 items-center gap-1.5" title={`${f.label}: ${f.value} (${conf})`}>
            <span className="font-mono text-2xs text-ink-3" aria-hidden="true">
              {SHORT[f.key]}
            </span>
            <span className="sr-only">{f.label}:</span>
            {f.swatch ? <ColorSwatch hex={f.swatch} /> : null}
            <span className={`nums break-all ${low ? "text-ink-3" : "text-ink-2"}`}>
              {f.value}
              {low ? <span aria-hidden="true">?</span> : null}
            </span>
            <span className="sr-only">({conf})</span>
          </li>
        );
      })}
    </ul>
  );
}
