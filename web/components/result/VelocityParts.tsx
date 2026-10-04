import type { ReactNode } from "react";

import { formatPercent } from "@/lib/format";
import type { VelocityBand, VelocityMarkerKind } from "@/lib/velocity";

/* Small pieces shared by the velocity chart, its table view and the phase inspector. */

/** One shape per marker kind, so markers stay distinguishable without colour. */
export function MarkerGlyph({ kind, className = "" }: { kind: VelocityMarkerKind; className?: string }) {
  const shape: Record<VelocityMarkerKind, ReactNode> = {
    release: <path d="M5 1 9 5 5 9 1 5Z" className="fill-surface stroke-current" strokeWidth={1.25} />,
    rest: <rect x={1.75} y={1.75} width={6.5} height={6.5} className="fill-current" />,
    resume_start: <path d="M2 1.25 9 5 2 8.75Z" className="fill-current" />,
    // Hollow twin of resume_start: back to autoplay speed, but straight from momentum (no rest, no ramp).
    autoplay_join: <path d="M2 1.5 8.75 5 2 8.5Z" className="fill-surface stroke-current" strokeWidth={1.25} strokeLinejoin="round" />,
    cursor_enter: <circle cx={5} cy={5} r={3.5} className="fill-current" />,
    cursor_leave: <circle cx={5} cy={5} r={3.25} className="fill-surface stroke-current" strokeWidth={1.25} />,
  };
  return (
    <svg viewBox="0 0 10 10" className={`size-2.5 shrink-0 overflow-visible ${className}`} aria-hidden="true" focusable="false">
      {shape[kind]}
    </svg>
  );
}

const CHIP = "rounded-sm border px-1 font-mono text-2xs whitespace-nowrap";

export function BandChips({ band }: { band: VelocityBand }) {
  if (!band.uncertain && !band.interrupted && !band.degraded) return null;
  return (
    <span className="inline-flex flex-wrap gap-1">
      {band.uncertain ? <span className={`${CHIP} border-dashed border-ink-3 text-ink-2`}>uncertain</span> : null}
      {band.interrupted ? <span className={`${CHIP} border-line text-ink-2`}>cut short</span> : null}
      {band.degraded ? (
        <span className={`${CHIP} border-dashed border-line-strong text-ink-2`} title={`${formatPercent(band.degradedFraction)} of its samples tracked less precisely`}>
          less precise
        </span>
      ) : null}
    </span>
  );
}

