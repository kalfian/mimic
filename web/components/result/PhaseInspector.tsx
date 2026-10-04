import type { ReactNode } from "react";

import { BandChips } from "@/components/result/VelocityParts";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { decaysToAutoplay, type VelocityBand, type VelocityModel, type VelocityPoint } from "@/lib/velocity";
import type { Easing } from "@/lib/types";

function Fact({ label, children, wide = false }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : ""}>
      <dt className="caption">{label}</dt>
      <dd className="mt-0.5 text-sm text-ink">{children}</dd>
    </div>
  );
}

/** Readout for the hovered / focused / selected phase (or the phase under the playhead). */
export function PhaseInspector({
  band,
  model,
  announce,
  following,
}: {
  band: VelocityBand | null;
  model: VelocityModel;
  /** Only a committed selection is announced (hover / focus previews are not). */
  announce: boolean;
  /** Nothing selected: the details follow the playhead. */
  following: boolean;
}) {
  if (!band) {
    return <p className="px-4 py-4 text-sm text-ink-3">No phases were measured.</p>;
  }
  const points = model.points.filter((p) => p.t_ms >= band.start_ms && p.t_ms <= band.end_ms);

  return (
    <div className="grid gap-6 px-4 py-4 md:grid-cols-[minmax(0,1fr)_12rem]">
      <div className="min-w-0 space-y-4">
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <h3 className="text-md font-medium text-ink">{band.label}</h3>
            <span className="rounded-sm border border-line px-1.5 font-mono text-2xs text-ink-2">{band.phase_id}</span>
            <BandChips band={band} />
            {following ? <span className="font-mono text-2xs text-ink-3">at the playhead</span> : null}
          </div>
          <ConfidenceBadge confidence={{ value: band.confidence, band: band.band }} showPercent className="mt-1" />
        </div>
        <span className="sr-only" aria-live="polite">
          {announce ? `Selected ${band.label}, ${(band.start_ms / 1000).toFixed(2)} to ${(band.end_ms / 1000).toFixed(2)} seconds` : ""}
        </span>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
          {band.facts.map((f) => (
            <Fact key={f.label} label={f.label}>
              <span className="nums block">{f.value}</span>
              {f.confidence ? <ConfidenceBadge confidence={f.confidence} className="mt-0.5" /> : null}
            </Fact>
          ))}
          <Fact label="Video time">
            <span className="nums">
              {Math.round(band.start_ms)}–{Math.round(band.end_ms)}
            </span>
            <span className="nums text-xs text-ink-3"> ms</span>
          </Fact>
          {band.notes.length > 0 ? (
            <Fact label="Notes" wide>
              <ul className="space-y-0.5 text-xs text-ink-2">
                {band.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Fact>
          ) : null}
        </dl>
      </div>

      <PhasePlot band={band} points={points} />
    </div>
  );
}

/* ---------- fitted velocity vs measured samples, for one phase ---------- */

const W = 160;
const H = 100;
const PAD = 8;

function bezierAt([x1, y1, x2, y2]: Easing["cubic_bezier"], s: number) {
  const u = 1 - s;
  return {
    x: 3 * u * u * s * x1 + 3 * u * s * s * x2 + s * s * s,
    y: 3 * u * u * s * y1 + 3 * u * s * s * y2 + s * s * s,
    dx: 3 * u * u * x1 + 6 * u * s * (x2 - x1) + 3 * s * s * (1 - x2),
    dy: 3 * u * u * y1 + 6 * u * s * (y2 - y1) + 3 * s * s * (1 - y2),
  };
}

/** The fitted model as [t_ms, v] points, clipped to the phase. Null when the phase has no fit. */
function fitCurve(band: VelocityBand): [number, number][] | null {
  const fit = band.fit;
  if (!fit) return null;
  const t0 = band.start_ms;
  const t1 = band.end_ms;
  const out: [number, number][] = [];
  switch (fit.model) {
    case "constant":
      return [
        [t0, fit.velocity_px_s.value],
        [t1, fit.velocity_px_s.value],
      ];
    case "exponential": {
      const tau = Math.max(1, fit.tau_ms.value);
      for (let i = 0; i <= 48; i++) {
        const t = t0 + ((t1 - t0) * i) / 48;
        out.push([t, fit.v_inf_px_s + (fit.v0_px_s.value - fit.v_inf_px_s) * Math.exp(-(t - t0) / tau)]);
      }
      return out;
    }
    case "ramp": {
      const d = Math.max(1, fit.duration_ms.value);
      for (let i = 0; i <= 48; i++) {
        const b = bezierAt(fit.easing.cubic_bezier, i / 48);
        const t = t0 + b.x * d;
        if (t > t1) break;
        out.push([t, fit.from_px_s + (fit.to_px_s - fit.from_px_s) * b.y]);
      }
      return out.length > 1 ? out : null;
    }
    case "tween": {
      // Velocity of an eased position tween: D / T · E'(x) (slope of the easing curve).
      const d = Math.max(1, fit.duration_ms.value);
      const scale = fit.distance_px.value / (d / 1000);
      for (let i = 0; i <= 48; i++) {
        const b = bezierAt(fit.easing.cubic_bezier, i / 48);
        const t = t0 + b.x * d;
        if (t > t1) break;
        out.push([t, b.dx > 1e-6 ? scale * (b.dy / b.dx) : 0]);
      }
      return out.length > 1 ? out : null;
    }
  }
}

function PhasePlot({ band, points }: { band: VelocityBand; points: VelocityPoint[] }) {
  const curve = fitCurve(band);
  if (points.length === 0 && !curve) return null;
  // Momentum that blends into autoplay levels off at v_inf: show that target, not just zero.
  const target = band.fit?.model === "exponential" && decaysToAutoplay(band.fit) ? band.fit.v_inf_px_s : null;
  const values = [0, ...(target != null ? [target] : []), ...points.map((p) => p.v), ...(curve ?? []).map(([, v]) => v)];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const padV = Math.max(1, (hi - lo) * 0.12);
  lo -= padV;
  hi += padV;
  const span = Math.max(1, band.end_ms - band.start_ms);
  const x = (t: number) => PAD + ((t - band.start_ms) / span) * (W - 2 * PAD);
  const y = (v: number) => PAD + ((hi - v) / (hi - lo)) * (H - 2 * PAD);
  const path = curve ? curve.map(([t, v], i) => `${i === 0 ? "M" : "L"}${x(t).toFixed(2)},${y(v).toFixed(2)}`).join(" ") : null;
  const degraded = points.some((p) => p.degraded);

  return (
    <figure className="w-48 justify-self-start md:justify-self-end">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full rounded-sm border border-line bg-canvas"
        role="img"
        aria-label={`Velocity during ${band.label}: ${points.length} measured samples${curve ? ", with the fitted curve" : ""}${target != null ? ", levelling off at the autoplay speed" : ""}`}
      >
        <line x1={PAD} x2={W - PAD} y1={y(0)} y2={y(0)} className="stroke-line-strong" strokeWidth={1} />
        {target != null ? <line x1={PAD} x2={W - PAD} y1={y(target)} y2={y(target)} className="stroke-ink-3" strokeWidth={1} strokeDasharray="3 2" /> : null}
        {points.map((p) =>
          p.degraded ? (
            <circle key={p.t_ms} cx={x(p.t_ms)} cy={y(p.v)} r={1.9} className="fill-canvas stroke-ink" strokeWidth={0.9} />
          ) : (
            <circle key={p.t_ms} cx={x(p.t_ms)} cy={y(p.v)} r={1.6} className="fill-ink" />
          ),
        )}
        {/* Fit drawn last: for constant phases it coincides with the samples and would vanish under them. */}
        {path ? <path d={path} fill="none" className="stroke-signal-fill" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" /> : null}
      </svg>
      <figcaption className="mt-1.5 flex flex-wrap items-center justify-between gap-x-3 font-mono text-2xs text-ink-3">
        {curve ? (
          <span className="inline-flex items-center gap-1">
            <span className="inline-block h-0.5 w-3 bg-signal-fill" aria-hidden="true" />
            fit
          </span>
        ) : (
          <span>no fit</span>
        )}
        {points.length > 0 ? (
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-1.5 rounded-full bg-ink" aria-hidden="true" />
            measured
          </span>
        ) : null}
        {degraded ? (
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-1.5 rounded-full border border-ink" aria-hidden="true" />
            less precise
          </span>
        ) : null}
        {target != null ? (
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 border-t border-dashed border-ink-3" aria-hidden="true" />
            autoplay
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}
