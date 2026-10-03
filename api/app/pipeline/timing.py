"""Progress normalization, significance gates and the joint (t0, D, curve) fit (PLAN §6.7).

Pure numpy. Times are float **seconds** (absolute video time), as in ``models/measure.py``.

Model (PLAN §6.7)::

    p̂(t) = 0                    t < t0
           E((t - t0) / D)       t0 ≤ t ≤ t0 + D
           1                    t > t0 + D

For every candidate curve ``E`` (Appendix B; back/overshoot curves only if the data overshoots)
``t0`` and ``D`` are found by grid search minimizing the SSE over samples in
``[t5 - fit_pre_s, t95 + fit_post_s]``. Fitting t0/D jointly with the curve removes the
threshold-crossing biases (ease-in looks late, ease-out looks short).

Implementation notes (deviations from the literal PLAN text, same intent):

* Grid search is coarse-to-fine: a coarse grid over the PLAN ranges (≤ 25 intervals per axis),
  then step/4 rounds around each candidate's optimum down to the PLAN resolution
  (``t0_step_ms`` / ``d_step_ms``) and ``_POLISH_ROUNDS`` more below it. Without the sub-grid
  polish the 2/4 ms quantization alone leaves a ~0.002 RMSE floor on the *true* curve, which is
  a sizeable part of the gap between close candidates (ease vs easeOutQuad ≈ 0.008).
* If a competitive optimum sits on any edge of the PLAN range, the range is extended. Needed
  for expo-out and back curves: their 5 %→95 % span is only ~0.3–0.4·D, so ``2.5·span`` can be
  shorter than ``D``; also guards against a noisy ``t95``.
* ``t5``/``t95`` and the overshoot test use a running median of ``p`` (5 / 3 samples). A running
  median leaves a monotonic transition untouched but stops single noisy samples from moving
  ``t95`` by a second or switching on the back/overshoot candidates.
* Free bezier: shape ↔ (t0, D) alternate up to ``_FREE_ALTERNATIONS`` times (PLAN: once).

Accuracy at σ = 0.02 progress noise is sample-count limited (PLAN R3): family recovery pooled
over Appendix B is ≥ 95 % from ~200 ms at 60 fps and ~400 ms at 30 fps; ``ease`` vs
``easeOutQuad`` stays the least separable pair (best-fit curves differ by < 0.01 RMSE).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.models.ir import ColorValue, Easing, EasingFamily, PxValue, RatioValue
from app.models.measure import FittedTransition, PropertySeries
from app.pipeline.easing import (
    Bezier,
    EasingCurve,
    candidate_set,
    detect_overshoot,
    eval_tables,
    family_of,
    fit_free_bezier,
    make_easing,
    tables_for,
)
from app.pipeline.params import DEFAULT_PARAMS, FitParams

#: Coarse grid: at most this many intervals per axis (step never below 4× the fine step).
_COARSE_INTERVALS = 24
#: Coarse step as a multiple of the fine (PLAN) step.
_COARSE_STEP_MULT = 4.0
#: Range extensions when an optimum sits on the search edge.
_MAX_EXTENSIONS = 4
#: A candidate counts as "competitive" for edge extension if rmse ≤ k·best + abs.
_EXTEND_COMPETITIVE_K = 1.5
_EXTEND_COMPETITIVE_ABS = 0.02
#: Running-median widths: coarse crossings / overshoot detection (noise-spike robustness).
_CROSS_MEDIAN = 5
_OVERSHOOT_MEDIAN = 3
#: Extra step/4 rounds below the PLAN grid resolution (2 ms / 4 ms → 0.5 / 1 ms → …).
_POLISH_ROUNDS = 2
#: Local refinement grid half-width (points per side) and a hard cap on refinement rounds.
_REFINE_HALF_POINTS = 4
_MAX_REFINE_ROUNDS = 40
#: Free bezier: max shape <-> (t0, D) alternations and the RMSE gain needed to continue.
_FREE_ALTERNATIONS = 4
_FREE_MIN_GAIN = 1e-4
#: Minimum samples to attempt a fit at all.
_MIN_SAMPLES = 4

COLOR_PROPS = frozenset({"color", "background-color"})


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def normalize_progress(values: np.ndarray, v_start: float, v_end: float) -> np.ndarray:
    """``p(t) = (v(t) - v_start) / (v_end - v_start)``. Works for reverse segments as-is.

    For a reverse segment ``v_start`` is state B and ``v_end`` state A, which is exactly the
    PLAN's ``p_rev = (v - v_B) / (v_A - v_B)``.
    """
    denom = float(v_end) - float(v_start)
    if denom == 0.0 or not np.isfinite(denom):
        raise ValueError("v_end == v_start: no change to normalize")
    return (np.asarray(values, dtype=np.float64) - float(v_start)) / denom


def frame_interval(times: np.ndarray) -> float:
    """Median sample interval in seconds (0 if fewer than 2 samples)."""
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2:
        return 0.0
    return float(np.median(np.diff(t)))


def timing_resolution_ms(times: np.ndarray) -> int:
    """IR ``source.timing_resolution_ms``: median frame interval, rounded, ≥ 1."""
    return max(1, round(frame_interval(times) * 1000.0))


def progress_samples(
    times: np.ndarray, progress: np.ndarray, max_points: int = 60
) -> list[tuple[int, float]]:
    """IR ``Transition.samples``: ``[(t_ms, progress)]`` uniformly decimated to ≤ ``max_points``."""
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(progress, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size == 0:
        return []
    if t.size > max_points:
        idx = np.unique(np.round(np.linspace(0, t.size - 1, max_points)).astype(np.int64))
        t, p = t[idx], p[idx]
    return [(max(0, round(ti * 1000.0)), round(float(pi), 4)) for ti, pi in zip(t, p, strict=True)]


def running_median(p: np.ndarray, k: int = 5) -> np.ndarray:
    """Centered running median of odd width ``k`` (edges use a shrinking window).

    A running median leaves monotonic runs untouched, so it removes isolated noise spikes without
    distorting the transition itself.
    """
    p = np.asarray(p, dtype=np.float64)
    h = k // 2
    if p.size < 3 or h == 0:
        return p.copy()
    padded = np.concatenate([np.full(h, np.nan), p, np.full(h, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.nanmedian(windows, axis=1)


def _cross(t: np.ndarray, p: np.ndarray, i: int, level: float) -> float:
    """Linear-interp time where p crosses ``level`` between samples i and i+1."""
    p0, p1 = p[i], p[i + 1]
    if p1 == p0:
        return float(t[i + 1])
    f = float(np.clip((level - p0) / (p1 - p0), 0.0, 1.0))
    return float(t[i] + f * (t[i + 1] - t[i]))


def coarse_crossings(
    times: np.ndarray,
    progress: np.ndarray,
    params: FitParams = DEFAULT_PARAMS.fit,
    *,
    overshoot: bool = False,
) -> tuple[float, float] | None:
    """Coarse ``(t5, t95)`` (PLAN §6.7) on a 5-sample running median of ``p``.

    * ``t5``: the 0.05 crossing immediately preceding the first 0.5 crossing (robust to
      single-sample noise spikes before the motion).
    * ``t95``: first time after which ``p`` stays ≥ 0.95, within ±0.05 unless ``overshoot``.
      If the series never settles, the last sample time.

    Returns None when ``p`` never reaches 0.5 (no motion in this series).
    """
    t = np.asarray(times, dtype=np.float64)
    ps = running_median(np.asarray(progress, dtype=np.float64), _CROSS_MEDIAN)
    lo, hi = params.t_lo, params.t_hi
    above_half = np.nonzero(ps >= 0.5)[0]
    if above_half.size == 0:
        return None
    i50 = int(above_half[0])

    below = np.nonzero(ps[:i50] < lo)[0]
    t5 = float(t[0]) if below.size == 0 else _cross(t, ps, int(below[-1]), lo)

    band_hi = np.inf if overshoot else 1.0 + (1.0 - hi)
    ok = (ps >= hi) & (ps <= band_hi)
    bad = np.nonzero(~ok[i50:])[0]
    if bad.size == 0:
        j = i50 - 1
        t95 = float(t[i50]) if j < 0 or ps[j] >= hi else _cross(t, ps, j, hi)
    else:
        j = i50 + int(bad[-1])
        if j >= t.size - 1:
            t95 = float(t[-1])
        elif ps[j] < hi <= ps[j + 1]:
            t95 = _cross(t, ps, j, hi)
        else:
            t95 = float(t[j + 1])
    return t5, max(t95, t5)


# --------------------------------------------------------------------------------------------
# Significance gate (PLAN §6.7)
# --------------------------------------------------------------------------------------------


def hex_to_lab(hex_color: str) -> np.ndarray:
    """``#RRGGBB`` (sRGB, D65) → CIE L*a*b*."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def delta_e76(a: str, b: str) -> float:
    """CIE76 ΔE between two ``#RRGGBB`` colors."""
    return float(np.linalg.norm(hex_to_lab(a) - hex_to_lab(b)))


def series_delta(series: PropertySeries) -> float:
    """Magnitude of change in the property's own unit (ΔE76 for colors)."""
    fv, tv = series.from_value, series.to_value
    if isinstance(fv, ColorValue) and isinstance(tv, ColorValue):
        return delta_e76(fv.color, tv.color)
    if isinstance(fv, PxValue | RatioValue) and isinstance(tv, PxValue | RatioValue):
        return abs(tv.number - fv.number)
    return abs(float(series.v_end) - float(series.v_start))


def is_significant(series: PropertySeries, params: FitParams = DEFAULT_PARAMS.fit) -> bool:
    """PLAN §6.7 gate: drop the property when its change is below the measurable floor.

    ``box-shadow`` and ``content`` have no numeric gate here (presence is decided in M1).
    """
    prop = series.property
    delta = series_delta(series)
    gates = {
        "translateX": params.gate_translate_css,
        "translateY": params.gate_translate_css,
        "scale": params.gate_scale,
        "scaleX": params.gate_scale,
        "scaleY": params.gate_scale,
        "opacity": params.gate_opacity,
        "border-radius": params.gate_radius_css,
        "height": params.gate_height_css,
        "color": params.gate_delta_e,
        "background-color": params.gate_delta_e,
    }
    gate = gates.get(prop)
    if gate is None:
        return float(series.v_end) != float(series.v_start)
    return delta >= gate


# --------------------------------------------------------------------------------------------
# Joint fit
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class CandidateFit:
    """Best (t0, D) for one curve and its error over the fit window."""

    name: str  # Appendix B name, or "free"
    family: EasingFamily
    bezier: Bezier
    t0_s: float
    duration_s: float
    sse: float
    rmse: float
    curve: EasingCurve | None = None  # None for the free fit


@dataclass(slots=True)
class TimingFit:
    """Result of :func:`fit_timing`. ``rmse`` is the progress RMSE over the fit window."""

    t0_s: float
    duration_s: float
    easing: Easing
    rmse: float
    rmse_second_family: float  # best rmse among curves of another family (inf if none)
    n_window: int  # samples in the fit window
    n_in: int  # samples strictly inside (t0, t0 + D)
    dt_s: float  # median sample interval
    t5_s: float
    t95_s: float
    overshoot_detected: bool
    used_free: bool
    candidates: list[CandidateFit] = field(default_factory=list)  # named, sorted by rmse

    @property
    def end_s(self) -> float:
        return self.t0_s + self.duration_s


def _grid_sse(
    tables: np.ndarray, t: np.ndarray, p: np.ndarray, t0: np.ndarray, dur: np.ndarray
) -> np.ndarray:
    """SSE for every (curve c, t0[c, a], D[c, b]) → ``(C, A, B)``."""
    x = (t[None, None, None, :] - t0[:, :, None, None]) / dur[:, None, :, None]
    r = eval_tables(tables, x) - p[None, None, None, :]
    return np.einsum("cabs,cabs->cab", r, r)


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    n = max(1, int(np.floor((hi - lo) / step + 1e-9)))
    return lo + step * np.arange(n + 1)


def _search(
    tables: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    t0_range: tuple[float, float],
    d_range: tuple[float, float],
    fine: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse grid + edge extension + per-curve refinement. Returns (t0[C], D[C], sse[C])."""
    c = tables.shape[0]
    t0_lo, t0_hi = t0_range
    d_lo, d_hi = d_range
    st0_f, sd_f = fine
    for _ in range(_MAX_EXTENSIONS + 1):
        st0 = max(_COARSE_STEP_MULT * st0_f, (t0_hi - t0_lo) / _COARSE_INTERVALS)
        sd = max(_COARSE_STEP_MULT * sd_f, (d_hi - d_lo) / _COARSE_INTERVALS)
        g_t0 = _axis(t0_lo, t0_hi, st0)
        g_d = _axis(d_lo, d_hi, sd)
        sse = _grid_sse(
            tables, t, p, np.broadcast_to(g_t0, (c, g_t0.size)), np.broadcast_to(g_d, (c, g_d.size))
        )
        flat = sse.reshape(c, -1)
        best = np.argmin(flat, axis=1)
        ia, ib = np.unravel_index(best, (g_t0.size, g_d.size))
        best_sse = flat[np.arange(c), best]
        competitive = (
            best_sse
            <= (
                _EXTEND_COMPETITIVE_K * np.sqrt(best_sse.min())
                + _EXTEND_COMPETITIVE_ABS * np.sqrt(t.size)
            )
            ** 2
        )
        hit_t0_lo = bool(np.any(competitive & (ia == 0)))
        hit_t0_hi = bool(np.any(competitive & (ia == g_t0.size - 1)))
        hit_d_lo = bool(np.any(competitive & (ib == 0))) and d_lo > sd_f + 1e-12
        hit_d_hi = bool(np.any(competitive & (ib == g_d.size - 1)))
        if not (hit_t0_lo or hit_t0_hi or hit_d_lo or hit_d_hi):
            break
        t0_ext = 0.5 * (t0_hi - t0_lo)
        d_ext = 0.5 * (d_hi - d_lo)
        if hit_t0_lo:
            t0_lo -= t0_ext
        if hit_t0_hi:
            t0_hi += t0_ext
        if hit_d_lo:
            d_lo = max(d_lo - d_ext, sd_f)
        if hit_d_hi:
            d_hi += d_ext

    t0_best = g_t0[ia].astype(np.float64)
    d_best = g_d[ib].astype(np.float64)
    offs = np.arange(-_REFINE_HALF_POINTS, _REFINE_HALF_POINTS + 1, dtype=np.float64)
    edge = (0, offs.size - 1)
    # Refine to the PLAN resolution, then polish below it (sub-grid) so grid quantization does
    # not add a curve-independent error floor that blurs close candidates. A candidate whose
    # optimum lands on the edge of its local grid is re-centred at the same step (the SSE valley
    # in (t0, D) is diagonal for out-curves; shrinking there would stall short of the optimum).
    st0_min = st0_f / 4.0**_POLISH_ROUNDS
    sd_min = sd_f / 4.0**_POLISH_ROUNDS
    st0_c = np.full(c, max(st0 / 4.0, st0_min))
    sd_c = np.full(c, max(sd / 4.0, sd_min))
    rows = np.arange(c)
    for _ in range(_MAX_REFINE_ROUNDS):
        g_t0c = t0_best[:, None] + offs[None, :] * st0_c[:, None]
        g_dc = np.maximum(d_best[:, None] + offs[None, :] * sd_c[:, None], sd_min)
        sse = _grid_sse(tables, t, p, g_t0c, g_dc).reshape(c, -1)
        best = np.argmin(sse, axis=1)
        ia, ib = np.unravel_index(best, (offs.size, offs.size))
        t0_best = g_t0c[rows, ia]
        d_best = g_dc[rows, ib]
        best_sse = sse[rows, best]
        on_edge = np.isin(ia, edge) | np.isin(ib, edge)
        done = (~on_edge) & (st0_c <= st0_min + 1e-12) & (sd_c <= sd_min + 1e-12)
        if bool(np.all(done)):
            break
        st0_c = np.where(on_edge, st0_c, np.maximum(st0_c / 4.0, st0_min))
        sd_c = np.where(on_edge, sd_c, np.maximum(sd_c / 4.0, sd_min))
    return t0_best, d_best, best_sse


def fit_timing(
    times: np.ndarray,
    progress: np.ndarray,
    params: FitParams = DEFAULT_PARAMS.fit,
    *,
    candidates: Sequence[EasingCurve] | None = None,
    allow_free: bool = True,
) -> TimingFit | None:
    """Joint (t0, D, easing) fit of a normalized progress series (PLAN §6.7).

    ``times`` are seconds (non-duplicate frames), ``progress`` the normalized ``p(t)``. Returns
    None if the series has too few samples or never reaches 50 % progress.
    """
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(progress, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size < _MIN_SAMPLES:
        return None
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]

    overshoot = detect_overshoot(running_median(p, _OVERSHOOT_MEDIAN), params)
    cross = coarse_crossings(t, p, params, overshoot=overshoot)
    if cross is None:
        return None
    t5, t95 = cross
    dt = max(frame_interval(t), 1e-4)
    span = max(t95 - t5, dt)

    win = (t >= t5 - params.fit_pre_s) & (t <= t95 + params.fit_post_s)
    tw, pw = t[win], p[win]
    if tw.size < _MIN_SAMPLES:
        return None

    pool = tuple(candidates) if candidates is not None else candidate_set(overshoot)
    tables = tables_for((cv.bezier for cv in pool), params.lut_points)
    fine = (params.t0_step_ms / 1000.0, params.d_step_ms / 1000.0)
    t0_range = (t5 - params.t0_back_frac * span - dt, t5 + dt)
    d_range = (max(params.d_min_frac * span, fine[1]), params.d_max_frac * span + 2.0 * dt)
    t0s, ds, sses = _search(tables, tw, pw, t0_range, d_range, fine)

    n = tw.size
    fits = [
        CandidateFit(
            name=cv.name,
            family=cv.family,
            bezier=cv.bezier,
            t0_s=float(t0s[k]),
            duration_s=float(ds[k]),
            sse=float(sses[k]),
            rmse=float(np.sqrt(max(sses[k], 0.0) / n)),
            curve=cv,
        )
        for k, cv in enumerate(pool)
    ]
    fits.sort(key=lambda f: f.rmse)
    best = fits[0]
    winner = best

    used_free = False
    if allow_free:
        n_in = _count_inside(tw, best.t0_s, best.duration_s)
        if n_in >= params.free_min_samples:
            free = _fit_free(tw, pw, best, params, fine, t0_range, d_range)
            if best.rmse > free.rmse + params.occam_margin:
                winner = free
                used_free = True

    # Final error of the reported curve on the high-resolution LUT.
    final_tables = tables_for([winner.bezier], params.lut_points_final)
    x = (tw - winner.t0_s) / winner.duration_s
    rmse = float(np.sqrt(np.mean((eval_tables(final_tables, x[None, :])[0] - pw) ** 2)))

    win_family = winner.family
    others = [f.rmse for f in fits if f.family != win_family]
    rmse_2nd = min(others) if others else float("inf")
    easing = make_easing(winner.bezier, rmse, named=winner.curve)
    return TimingFit(
        t0_s=winner.t0_s,
        duration_s=winner.duration_s,
        easing=easing,
        rmse=rmse,
        rmse_second_family=rmse_2nd,
        n_window=n,
        n_in=_count_inside(tw, winner.t0_s, winner.duration_s),
        dt_s=dt,
        t5_s=t5,
        t95_s=t95,
        overshoot_detected=overshoot,
        used_free=used_free,
        candidates=fits,
    )


def _count_inside(t: np.ndarray, t0: float, dur: float) -> int:
    return int(np.count_nonzero((t > t0) & (t < t0 + dur)))


def _fit_free(
    tw: np.ndarray,
    pw: np.ndarray,
    best: CandidateFit,
    params: FitParams,
    fine: tuple[float, float],
    t0_range: tuple[float, float],
    d_range: tuple[float, float],
) -> CandidateFit:
    """Free bezier refine (PLAN §6.7), started from the best named t0/D.

    PLAN: shape grid at fixed t0/D, then re-optimize t0/D once. Here the shape ↔ t0/D step is
    repeated up to ``_FREE_ALTERNATIONS`` times while it still improves, because a single pass
    stays stuck at the named curve's timing for curves far from the catalog.
    """
    t0, dur = best.t0_s, best.duration_s
    result: CandidateFit | None = None
    for _ in range(_FREE_ALTERNATIONS):
        x = (tw - t0) / dur
        inside = (x > 0.0) & (x < 1.0)
        if np.count_nonzero(inside) < 4:
            break
        free = fit_free_bezier(x[inside], pw[inside], params)
        tables = tables_for([free.bezier], params.lut_points)
        t0s, ds, sses = _search(tables, tw, pw, t0_range, d_range, fine)
        cand = CandidateFit(
            name="free",
            family=family_of(free.bezier),
            bezier=free.bezier,
            t0_s=float(t0s[0]),
            duration_s=float(ds[0]),
            sse=float(sses[0]),
            rmse=float(np.sqrt(max(sses[0], 0.0) / tw.size)),
            curve=None,
        )
        improved = result is None or cand.rmse < result.rmse - _FREE_MIN_GAIN
        if result is None or cand.rmse < result.rmse:
            result = cand
        if not improved:
            break
        t0, dur = cand.t0_s, cand.duration_s
    if result is None:  # degenerate timing: report the named fit as-is
        return best
    return result


def fit_series(
    series: PropertySeries,
    params: FitParams = DEFAULT_PARAMS.fit,
    *,
    allow_free: bool = True,
) -> TimingFit | None:
    """Normalize ``series`` with its ``v_start``/``v_end`` and run :func:`fit_timing`."""
    if float(series.v_end) == float(series.v_start):
        return None
    p = normalize_progress(series.values, series.v_start, series.v_end)
    return fit_timing(series.times, p, params, allow_free=allow_free)


# --------------------------------------------------------------------------------------------
# Segment-level timing
# --------------------------------------------------------------------------------------------


def segment_onset_settle(
    fits: Iterable[TimingFit | FittedTransition],
) -> tuple[float, float]:
    """``(onset, settle)`` = (min t0, max t0 + D) over a segment's transitions (PLAN §6.7)."""
    starts: list[float] = []
    ends: list[float] = []
    for f in fits:
        starts.append(f.t0_s)
        ends.append(f.t0_s + f.duration_s)
    if not starts:
        raise ValueError("segment has no fitted transitions")
    return min(starts), max(ends)


def delay_s(t0_s: float, onset_s: float) -> float:
    """``delay = t0 - segment onset`` (≥ 0)."""
    return max(0.0, t0_s - onset_s)


__all__ = [
    "CandidateFit",
    "TimingFit",
    "coarse_crossings",
    "delay_s",
    "delta_e76",
    "fit_series",
    "fit_timing",
    "frame_interval",
    "hex_to_lab",
    "is_significant",
    "normalize_progress",
    "progress_samples",
    "segment_onset_settle",
    "series_delta",
    "timing_resolution_ms",
]
