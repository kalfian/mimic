"""Position-domain kinematic fits + behaviour aggregation (PLAN-continuous §4.5 steps 3, 6, 9).

Pure numpy. Every model is fitted to **positions** ``x(t)`` (CSS px, signed), never to a
differentiated velocity (rejected alternative: differentiation amplifies tracking noise and VFR
jitter). Weights ``w`` are inverse variances (``1 / σ_i²``); with them ``sse`` is a χ² and the
BIC-style costs of ``phases.py`` are comparable across models.

Models
------
* constant      ``x = x0 + v (t − t_ref)``
* exponential   ``x = x_r + v_∞ s + (v0 − v_∞) τ (1 − e^{−s/τ})``, ``s = t − t_r`` (inertia;
                ``v_∞ ∈ {0, v_auto}`` fixed by the caller). For fixed τ the model is linear in
                ``(x_r, v0)``; τ is searched on a log grid then golden-section refined.
* tween         ``x = x_r + D · E((t − t_r) / T)`` (snap / tween-style inertia). Linear in
                ``(x_r, D)`` for fixed ``(E, T)``.
* ramp          ``v = v_a + (v_b − v_a) · E((t − t0) / D)`` integrated:
                ``x = x0 + v_a (t − t0) + (v_b − v_a) · D · IE((t − t0) / D)`` with
                ``IE(u) = ∫_0^u E``, ``IE(u > 1) = IE(1) + u − 1``, ``IE(u < 0) = 0`` (decelerate /
                resume; joint t0 / D grid like PLAN §6.7, same Bezier LUTs as ``easing.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from app.models.ir import (
    Behavior,
    DragBehavior,
    Easing,
    ExponentialFit,
    InertiaBehavior,
    PauseBehavior,
    PauseTrigger,
    ResumeBehavior,
    SnapBehavior,
    TweenFit,
)
from app.models.measure import PhaseCandidate
from app.pipeline.continuous import confidence as cconf
from app.pipeline.easing import (
    CANDIDATES,
    EasingCurve,
    candidate_set,
    eval_tables,
    make_easing,
    tables_for,
)
from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams

#: LUT resolution for the integrated easing tables.
LUT_N = 513
#: τ log-grid size and golden-section iterations.
TAU_GRID = 64
#: Candidate stop times (samples) for the exponential's momentum stop threshold.
CUT_GRID = 60
GOLDEN_ITERS = 30
#: Tween / ramp duration grids (log spaced) and t0 grid density (samples per frame interval).
DUR_GRID = 36
RAMP_DUR_GRID = 72
T0_PER_FRAME = 2
#: ``fit_ramp`` polish: pattern-search starts per curve (best coarse cells), iterations and
#: initial log-duration step.
POLISH_STARTS = 3
POLISH_ITERS = 60
POLISH_LOG_STEP = 0.06
#: Easing families tried for snap tweens (ease-out like curves; overshoot curves added when the
#: data overshoots) and for velocity ramps (all non-overshoot Appendix B curves).
#: (No ``linear`` / ease-in: a tween that does not end at rest is indistinguishable from a
#: constant-velocity drag followed by an abrupt stop.)
TWEEN_CURVES: tuple[str, ...] = (
    "ease-out", "ease", "ease-in-out", "easeOutQuad", "easeOutCubic", "easeOutQuart",
    "easeOutExpo", "expoOut", "material-decelerate", "material-standard",
)  # fmt: skip
RAMP_CURVES: tuple[str, ...] = tuple(c.name for c in candidate_set(include_overshoot=False))
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Fit:
    """One fitted model on samples ``t, x`` (``pred`` aligned with them)."""

    model: str  # "constant" | "exponential" | "tween" | "ramp" | "rest"
    params: dict[str, float]
    sse: float  # weighted (χ² when w = 1/σ²)
    n: int
    k: int  # free parameters (BIC penalty)
    rmse: float  # unweighted, CSS px
    pred: np.ndarray
    se: dict[str, float] = field(default_factory=dict)
    curve: EasingCurve | None = None
    progress_rmse: float = 0.0  # tween / ramp: rmse in progress units
    at_bound: bool = False  # τ / duration hit the search bound

    def bic(self) -> float:
        return float(self.sse + self.k * math.log(max(self.n, 2)))


def _w(w: np.ndarray | None, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    return np.asarray(w, dtype=np.float64)


def _lstsq(cols: Sequence[np.ndarray], y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted least squares ``y ≈ Σ c_j cols_j``; ``(coef, weighted sse)``."""
    a = np.stack(cols, axis=1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)
    r = y - a @ coef
    return coef, float(np.sum(w * r * r))


def _batched_lstsq2(
    c1: np.ndarray, c2: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 2-parameter WLS ``y ≈ a c1 + b c2`` over a leading batch axis.

    ``c1``, ``c2``: ``(B, n)`` (or broadcastable); returns ``(a (B,), b (B,), sse (B,))``.
    """
    c1, c2 = np.broadcast_arrays(c1, c2)
    s11 = np.sum(w * c1 * c1, axis=-1)
    s12 = np.sum(w * c1 * c2, axis=-1)
    s22 = np.sum(w * c2 * c2, axis=-1)
    s1y = np.sum(w * c1 * y, axis=-1)
    s2y = np.sum(w * c2 * y, axis=-1)
    det = s11 * s22 - s12 * s12
    ok = np.abs(det) > 1e-12 * np.maximum(s11 * s22, 1e-300)
    det = np.where(ok, det, 1.0)
    a = np.where(ok, (s22 * s1y - s12 * s2y) / det, 0.0)
    b = np.where(ok, (s11 * s2y - s12 * s1y) / det, 0.0)
    r = y - a[..., None] * c1 - b[..., None] * c2
    sse = np.sum(w * r * r, axis=-1)
    sse = np.where(ok, sse, np.inf)
    return a, b, sse


def _rmse(x: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - pred) ** 2))) if len(x) else 0.0


def _golden(f, lo: float, hi: float, iters: int = GOLDEN_ITERS) -> float:  # type: ignore[no-untyped-def]
    a, b = lo, hi
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    return (a + b) / 2.0


# --------------------------------------------------------------------------------------------
# Constant / rest
# --------------------------------------------------------------------------------------------


def fit_constant(t: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> Fit:
    """``x = x0 + v (t − t̄)``; ``se['v']`` from the residual scale."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = _w(w, len(t))
    tm = float(np.sum(w * t) / np.sum(w))
    tt = t - tm
    coef, sse = _lstsq([np.ones_like(t), tt], x, w)
    pred = coef[0] + coef[1] * tt
    n = len(t)
    s_tt = float(np.sum(w * tt * tt))
    resid_scale = sse / max(n - 2, 1)
    se_v = math.sqrt(resid_scale / s_tt) if s_tt > 0 else math.inf
    return Fit(
        model="constant",
        params={"x0": float(coef[0]), "v": float(coef[1]), "t_ref": tm},
        sse=sse,
        n=n,
        k=2,
        rmse=_rmse(x, pred),
        pred=pred,
        se={"v": se_v},
    )


def fit_rest(t: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> Fit:
    """``x = x0`` (zero velocity)."""
    x = np.asarray(x, dtype=np.float64)
    w = _w(w, len(x))
    x0 = float(np.sum(w * x) / np.sum(w)) if len(x) else 0.0
    pred = np.full(len(x), x0)
    return Fit("rest", {"x0": x0}, float(np.sum(w * (x - x0) ** 2)), len(x), 1, _rmse(x, pred),
               pred)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Exponential decay (inertia)
# --------------------------------------------------------------------------------------------


def exp_position(
    s: np.ndarray,
    x_r: float,
    v0: float,
    tau: float,
    v_inf: float,
    s_stop: float = math.inf,
) -> np.ndarray:
    """Exponential decay position; frozen after ``s_stop`` (momentum stop threshold)."""
    s = np.minimum(np.maximum(np.asarray(s, dtype=np.float64), 0.0), s_stop)
    return x_r + v_inf * s + (v0 - v_inf) * tau * (1.0 - np.exp(-s / tau))


def exp_velocity(
    s: np.ndarray | float, v0: float, tau: float, v_inf: float, s_stop: float = math.inf
) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    v = v_inf + (v0 - v_inf) * np.exp(-s / tau)
    return np.where(s > s_stop, 0.0, v)


def fit_exponential(
    t: np.ndarray,
    x: np.ndarray,
    t_r: float,
    v_inf: float = 0.0,
    w: np.ndarray | None = None,
    *,
    tau_min: float = DEFAULT_PARAMS.continuous.tau_min_s,
    tau_max: float = DEFAULT_PARAMS.continuous.tau_max_s,
    cutoff: bool = False,
) -> Fit | None:
    """Fit ``x_r, v0, τ`` with ``v_∞`` fixed. None with fewer than 3 samples.

    ``cutoff`` (only with ``v_∞ = 0``): also consider a **stop threshold** — momentum
    implementations commonly stop the decay once |v| falls below a minimum speed, which freezes
    the position abruptly. The stop time ``s_c`` becomes a 4th parameter (``params['s_stop']``,
    ``inf`` when the plain decay wins by BIC; ``params['v_stop']`` = speed at the stop).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(t)
    if n < 3:
        return None
    w = _w(w, n)
    s = np.maximum(t - t_r, 0.0)
    y = x - v_inf * s
    use_cut = cutoff and v_inf == 0.0 and n >= 8
    # x - v_inf s = x_r + (v0 - v_inf) τ g,  g = 1 - e^{-min(s, s_c)/τ}
    taus = np.geomspace(tau_min, tau_max, TAU_GRID)
    cuts = [math.inf]
    if use_cut:
        idx = np.arange(3, n - 1)
        if len(idx) > CUT_GRID:
            idx = np.unique(np.linspace(3, n - 2, CUT_GRID).astype(int))
        cuts += [float(s[i]) for i in idx]
    cut_arr = np.asarray(cuts)
    sm = np.minimum(s[None, :], cut_arr[:, None])  # (C, n)
    basis = taus[:, None, None] * (1.0 - np.exp(-sm[None, :, :] / taus[:, None, None]))
    _, _, sse = _batched_lstsq2(np.ones((1, 1, n)), basis, y[None, None, :], w)
    # BIC-compare: the cutoff costs one extra parameter
    pen = np.where(np.isinf(cut_arr), 0.0, math.log(max(n, 2)))
    total = sse + pen[None, :]
    ti, ci = np.unravel_index(int(np.argmin(total)), total.shape)
    if not math.isfinite(float(sse[ti, ci])):
        return None
    s_c = float(cut_arr[ci])

    def sse_at(tau: float, sc: float) -> float:
        g = tau * (1.0 - np.exp(-np.minimum(s, sc) / tau))
        _, _, e = _batched_lstsq2(np.ones((1, n)), g[None, :], y[None, :], w)
        return float(e[0])

    lo = math.log(taus[max(ti - 1, 0)])
    hi = math.log(taus[min(ti + 1, TAU_GRID - 1)])
    tau = (
        math.exp(_golden(lambda lt: sse_at(math.exp(lt), s_c), lo, hi))
        if hi > lo
        else float(taus[ti])
    )
    if math.isfinite(s_c):
        k_ = int(np.searchsorted(s, s_c))
        c_lo = float(s[max(k_ - 1, 1)])
        c_hi = float(s[min(k_ + 1, n - 1)])
        if c_hi > c_lo:
            s_c = _golden(lambda sc: sse_at(tau, sc), c_lo, c_hi)
        lo, hi = math.log(tau) - 0.1, math.log(tau) + 0.1
        tau = math.exp(_golden(lambda lt: sse_at(math.exp(lt), s_c), lo, hi))
    sm1 = np.minimum(s, s_c)
    g = tau * (1.0 - np.exp(-sm1 / tau))
    aa, bb, ee = _batched_lstsq2(np.ones((1, n)), g[None, :], y[None, :], w)
    x_r, dv = float(aa[0]), float(bb[0])
    v0 = dv + v_inf
    pred = exp_position(s, x_r, v0, tau, v_inf, s_c)
    sse_f = float(ee[0])
    # standard errors from the Jacobian (residual-scaled covariance; s_c held fixed)
    e_s = np.exp(-sm1 / tau)
    j_tau = dv * ((1.0 - e_s) - (sm1 / tau) * e_s)
    jac = np.stack([np.ones(n), tau * (1.0 - e_s), j_tau], axis=1)
    jw = jac * w[:, None]
    se: dict[str, float] = {"tau": math.inf, "v0": math.inf}
    k_par = 3 if math.isinf(s_c) else 4
    try:
        cov = np.linalg.inv(jac.T @ jw) * (sse_f / max(n - k_par, 1))
        se = {"tau": float(math.sqrt(max(cov[2, 2], 0.0))),
              "v0": float(math.sqrt(max(cov[1, 1], 0.0)))}  # fmt: skip
    except np.linalg.LinAlgError:
        pass
    at_bound = tau <= tau_min * 1.02 or tau >= tau_max * 0.98
    v_stop = abs(dv) * math.exp(-s_c / tau) if math.isfinite(s_c) else 0.0
    return Fit(
        model="exponential",
        params={
            "x_r": x_r,
            "v0": v0,
            "tau": tau,
            "v_inf": float(v_inf),
            "t_r": float(t_r),
            "s_stop": s_c,
            "v_stop": v_stop,
        },  # fmt: skip
        sse=sse_f,
        n=n,
        k=k_par,
        rmse=_rmse(x, pred),
        pred=pred,
        se=se,
        at_bound=at_bound,
    )


# --------------------------------------------------------------------------------------------
# Tween (snap / tween-style inertia)
# --------------------------------------------------------------------------------------------


def _curves(names: Sequence[str]) -> list[EasingCurve]:
    by = {c.name: c for c in CANDIDATES}
    return [by[n] for n in names if n in by]


def fit_tween(
    t: np.ndarray,
    x: np.ndarray,
    t_r: float,
    w: np.ndarray | None = None,
    *,
    dur_min: float,
    dur_max: float,
    curve_names: Sequence[str] = TWEEN_CURVES,
    include_overshoot: bool = False,
) -> Fit | None:
    """``x = x_r + D · E((t − t_r)/T)``: best curve + ``T`` (log grid + golden refine)."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(t)
    if n < 4 or dur_max <= dur_min:
        return None
    w = _w(w, n)
    names = list(curve_names)
    if include_overshoot:
        names += [c.name for c in CANDIDATES if c.overshoot_only]
    curves = _curves(names)
    tables = tables_for([c.bezier for c in curves], LUT_N)
    s = np.maximum(t - t_r, 0.0)
    durs = np.geomspace(dur_min, dur_max, DUR_GRID)
    # (D, C, n)
    u = np.clip(s[None, :] / durs[:, None], 0.0, 1.0)
    e = eval_tables(tables, np.broadcast_to(u[None, :, :], (len(curves), len(durs), n)))
    e = np.transpose(e, (1, 0, 2))
    a, b, sse = _batched_lstsq2(np.ones((1, 1, n)), e, x[None, None, :], w)
    di, ci = np.unravel_index(int(np.argmin(sse)), sse.shape)
    if not math.isfinite(float(sse[di, ci])):
        return None
    table = tables[ci : ci + 1]

    def cost(log_d: float) -> float:
        uu = np.clip(s / math.exp(log_d), 0.0, 1.0)
        ee = eval_tables(table, uu[None, :])
        _, _, sse_ = _batched_lstsq2(np.ones((1, n)), ee, x[None, :], w)
        return float(sse_[0])

    lo = math.log(durs[max(di - 1, 0)])
    hi = math.log(durs[min(di + 1, DUR_GRID - 1)])
    dur = math.exp(_golden(cost, lo, hi))
    uu = np.clip(s / dur, 0.0, 1.0)
    ee = eval_tables(table, uu[None, :])
    aa, bb, ss = _batched_lstsq2(np.ones((1, n)), ee, x[None, :], w)
    x_r, dist = float(aa[0]), float(bb[0])
    pred = x_r + dist * ee[0]
    rmse = _rmse(x, pred)
    return Fit(
        model="tween",
        params={"x_r": x_r, "distance": dist, "duration": dur, "t_r": float(t_r)},
        sse=float(ss[0]),
        n=n,
        k=4,  # x_r, D, T + the discrete curve choice (keeps BIC ties with exponential honest)
        rmse=rmse,
        pred=pred,
        curve=curves[ci],
        progress_rmse=rmse / abs(dist) if abs(dist) > 1e-9 else 1.0,
        at_bound=dur <= dur_min * 1.02 or dur >= dur_max * 0.98,
    )


# --------------------------------------------------------------------------------------------
# Eased velocity ramp (decelerate / resume)
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _integral_table(bezier: tuple[float, float, float, float]) -> np.ndarray:
    """``IE(u) = ∫_0^u E`` sampled on the uniform LUT grid (trapezoid)."""
    tab = tables_for([bezier], LUT_N)[0]
    du = 1.0 / (LUT_N - 1)
    ie = np.concatenate([[0.0], np.cumsum((tab[1:] + tab[:-1]) * 0.5 * du)])
    ie.setflags(write=False)
    return ie


def ramp_integral(bezier: Sequence[float], u: np.ndarray) -> np.ndarray:
    """``IE`` extended: 0 below 0, ``IE(1) + (u − 1)`` above 1."""
    ie = _integral_table(tuple(float(v) for v in bezier))  # type: ignore[arg-type]
    u = np.asarray(u, dtype=np.float64)
    inner = eval_tables(ie[None, :], np.clip(u, 0.0, 1.0)[None, ...])[0]
    return np.where(u <= 0.0, 0.0, np.where(u >= 1.0, ie[-1] + (u - 1.0), inner))


def ramp_position(
    t: np.ndarray, x0: float, t0: float, dur: float, v_a: float, v_b: float, bezier: Sequence[float]
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return x0 + v_a * (t - t0) + (v_b - v_a) * dur * ramp_integral(bezier, (t - t0) / dur)


def fit_ramp(
    t: np.ndarray,
    x: np.ndarray,
    v_a: float,
    v_b: float,
    w: np.ndarray | None = None,
    *,
    t0_range: tuple[float, float],
    dur_range: tuple[float, float],
    frame_dt: float,
    curve_names: Sequence[str] = RAMP_CURVES,
) -> Fit | None:
    """Joint ``t0 / D / E`` grid for a velocity ramp ``v_a → v_b`` (both fixed by the caller).

    Only ``x0`` is free in closed form; ``t0`` on a grid of ``frame_dt / T0_PER_FRAME`` steps
    within ``t0_range``, ``D`` log-spaced in ``dur_range``; then a local refinement round.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(t)
    if n < 5 or dur_range[1] <= dur_range[0] or t0_range[1] < t0_range[0]:
        return None
    w = _w(w, n)
    curves = _curves(curve_names)
    ies = np.stack([_integral_table(c.bezier) for c in curves])  # (C, LUT)
    ie1 = ies[:, -1][:, None, None, None]
    step = max(frame_dt / T0_PER_FRAME, 1e-3)
    t0s = np.arange(t0_range[0], t0_range[1] + step * 0.5, step)
    if len(t0s) > 160:
        t0s = np.linspace(t0_range[0], t0_range[1], 160)
    durs = np.geomspace(dur_range[0], dur_range[1], RAMP_DUR_GRID)
    dv = v_b - v_a

    def evaluate(t0_arr: np.ndarray, d_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # u: (T0, D, n)
        u = (t[None, None, :] - t0_arr[:, None, None]) / d_arr[None, :, None]
        uc = np.clip(u, 0.0, 1.0)
        inner = eval_tables(ies, np.broadcast_to(uc[None], (len(curves), *uc.shape)))
        ie = np.where(u[None] <= 0, 0.0, np.where(u[None] >= 1.0, ie1 + (u[None] - 1.0), inner))
        base = v_a * (t[None, None, :] - t0_arr[:, None, None])  # (T0, 1, n)
        model_wo_x0 = base[None] + dv * d_arr[None, None, :, None] * ie  # (C, T0, D, n)
        r = x - model_wo_x0
        sw = np.sum(w)
        x0 = np.sum(w * r, axis=-1) / sw
        sse = np.sum(w * (r - x0[..., None]) ** 2, axis=-1)
        return sse, x0

    def evaluate_pairs(
        ci_: int, t0_v: np.ndarray, d_v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u = (t[None, :] - t0_v[:, None]) / d_v[:, None]
        ie = ramp_integral(curves[ci_].bezier, u)
        model_wo_x0 = v_a * (t[None, :] - t0_v[:, None]) + dv * d_v[:, None] * ie
        r = x - model_wo_x0
        x0_ = np.sum(w * r, axis=-1) / np.sum(w)
        sse_ = np.sum(w * (r - x0_[:, None]) ** 2, axis=-1)
        bad = (t0_v < t0_range[0] - 1e-9) | (t0_v > t0_range[1] + 1e-9)
        bad |= (d_v < dur_range[0] * 0.999) | (d_v > dur_range[1] * 1.001)
        return np.where(bad, np.inf, sse_), x0_

    def polish(ci_: int, t0_: float, d_: float) -> tuple[float, float, float, float]:
        """Pattern search in (centre time, log D): the χ² valley runs along D at a fixed ramp
        centre, so these coordinates are nearly separable (t0 / D are strongly coupled)."""
        tc, ls = t0_ + d_ / 2.0, math.log(d_)
        h_tc, h_ls = max(frame_dt / 2.0, 1e-3), POLISH_LOG_STEP
        offs = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=np.float64)
        cur_sse, cur_x0 = math.inf, 0.0
        for _ in range(POLISH_ITERS):
            tcs = tc + offs[:, 0] * h_tc
            dd = np.exp(ls + offs[:, 1] * h_ls)
            ss, xx = evaluate_pairs(ci_, tcs - dd / 2.0, dd)
            k_ = int(np.argmin(ss))
            if ss[k_] < cur_sse - 1e-12 and k_ != 4:
                tc, ls = float(tcs[k_]), float(np.log(dd[k_]))
                cur_sse, cur_x0 = float(ss[k_]), float(xx[k_])
            else:
                if ss[4] < cur_sse:
                    cur_sse, cur_x0 = float(ss[4]), float(xx[4])
                h_tc, h_ls = h_tc / 2.0, h_ls / 2.0
                if h_tc < 1e-4 and h_ls < 1e-3:
                    break
        d_f = math.exp(ls)
        return cur_sse, tc - d_f / 2.0, d_f, cur_x0

    sse, x0 = evaluate(t0s, durs)
    # polish every curve from its best coarse cells (the χ² valley is narrow when σ is small;
    # refining only the global coarse winner can stay in the wrong basin)
    best = (math.inf, 0, 0.0, 0.0, 0.0)  # sse, curve, t0, dur, x0
    refined: list[float] = []
    for ci_ in range(len(curves)):
        flat = np.argsort(sse[ci_], axis=None)[:POLISH_STARTS]
        cand = (math.inf, ci_, 0.0, 0.0, 0.0)
        for f_idx in flat:
            ti, di = np.unravel_index(int(f_idx), sse[ci_].shape)
            ps, pt0, pd, px0 = polish(ci_, float(t0s[ti]), float(durs[di]))
            if ps < cand[0]:
                cand = (ps, ci_, pt0, pd, px0)
        refined.append(cand[0])
        if cand[0] < best[0]:
            best = cand
    best_sse, ci, t0, dur, x0v = best
    curve = curves[ci]
    pred = ramp_position(t, x0v, t0, dur, v_a, v_b, curve.bezier)
    rmse = _rmse(x, pred)
    # progress rmse: position error relative to the ramp's displacement deficit |dv|·D/2
    scale = abs(dv) * dur * 0.5
    # second-best family for the easing separation factor
    fam_best = curve.family
    other = [refined[i] for i, c in enumerate(curves) if c.family != fam_best]
    fit = Fit(
        model="ramp",
        params={
            "x0": x0v,
            "t0": t0,
            "duration": dur,
            "v_a": float(v_a),
            "v_b": float(v_b),
            "sse_second_family": min(other) if other else math.inf,
        },  # fmt: skip
        sse=best_sse,
        n=n,
        k=4,
        rmse=rmse,
        pred=pred,
        curve=curve,
        progress_rmse=rmse / scale if scale > 1e-9 else 1.0,
        at_bound=dur >= dur_range[1] * 0.98 or dur <= dur_range[0] * 1.02,
    )
    return fit


def fit_easing(fit: Fit) -> Easing | None:
    """IR ``Easing`` for a tween / ramp fit (named candidate)."""
    if fit.curve is None:
        return None
    return make_easing(fit.curve.bezier, min(fit.progress_rmse, 1.0), named=fit.curve)


# --------------------------------------------------------------------------------------------
# §4.5 step 9: aggregation into behaviour
# --------------------------------------------------------------------------------------------


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


def agreement_factor(
    values: Sequence[float], params: ContinuousParams = DEFAULT_PARAMS.continuous
) -> float:
    """``clip(1 − IQR/median, agreement_min_factor, 1)`` for ≥ 2 instances, else 1."""
    if len(values) < 2:
        return 1.0
    v = np.abs(np.asarray(values, dtype=np.float64))
    med = float(np.median(v))
    if med <= 0:
        return params.agreement_min_factor
    q75, q25 = np.percentile(v, [75, 25])
    spread = float(q75 - q25)
    return float(np.clip(1.0 - spread / med, params.agreement_min_factor, 1.0))


@dataclass(slots=True)
class BehaviorContext:
    """Facts beyond the phases that aggregation needs."""

    autoplay_velocity: float | None
    pitch_px: float | None
    pitch_confidence: float
    snap_grid: bool  # rests on the pitch grid (phases already relabelled snap)
    snap_grid_confidence: float
    pause_on: PauseTrigger = "unknown"
    pause_on_confidence: float = 0.0
    follows_pointer: bool | None = None
    pointer_ratio: float | None = None
    pointer_ratio_confidence: float = 0.0
    overshoot: bool = False
    resume_delays: list[tuple[float, float | None, float]] = field(default_factory=list)
    #: (delay_after_rest_s, delay_after_release_s | None, confidence) per resume phase
    resume_leave_delays: list[tuple[float, float]] = field(default_factory=list)
    #: Confidence of the drag peak speed (tracking around the peak); None = the phase's.
    drag_peak_confidence: float | None = None
    #: (delay_after_pointer_leave_s, confidence) per resume phase that followed a hover pause


def _mn(value: float, conf: float, cap: float | None = None, nd: int = 1) -> dict:
    return {"value": round(float(value), nd), "confidence": cconf.ir_confidence(conf, cap)}


def aggregate(
    phases: Sequence[PhaseCandidate],
    ctx: BehaviorContext,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
) -> Behavior:
    """Behaviour values over phase instances (weighted median, agreement factor)."""
    from app.models.ir import MeasuredNumber  # local: keep module import list short

    def mnum(value: float, conf: float, cap: float | None = None, nd: int = 1) -> MeasuredNumber:
        return MeasuredNumber.model_validate(_mn(value, conf, cap, nd))

    kinds = [p.kind for p in phases]

    # ---- pause (decelerate phases, or autoplay stopped at once by a press: autoplay -> drag) --
    pause = None
    decs = [p for p in phases if p.kind == "decelerate"]
    instant = any(
        a.kind == "autoplay" and b.kind == "drag" for a, b in zip(phases, phases[1:], strict=False)
    )
    if not decs and instant:
        pause = PauseBehavior(
            on=ctx.pause_on,
            on_confidence=cconf.ir_confidence(
                ctx.pause_on_confidence,
                params.cap_trigger_no_cursor if ctx.pause_on == "unknown" else None,
            ),
            decel_ms=None,
            easing=None,
            stops_completely=True,
        )
    if decs:
        durs = [(p.fit.duration_ms.value, p.fit.duration_ms.confidence.value, p)  # type: ignore[union-attr]
                for p in decs if p.fit is not None and p.fit.model == "ramp"]  # fmt: skip
        decel_ms = None
        easing = None
        if durs:
            vals = [d[0] for d in durs]
            confs = [d[1] for d in durs]
            agree = agreement_factor(vals, params)
            decel_ms = mnum(weighted_median(vals, confs), float(np.median(confs)) * agree, nd=0)
            best = max(durs, key=lambda d: d[1])[2]
            easing = best.fit.easing  # type: ignore[union-attr]
        pause = PauseBehavior(
            on=ctx.pause_on,
            on_confidence=cconf.ir_confidence(
                ctx.pause_on_confidence,
                params.cap_trigger_no_cursor if ctx.pause_on == "unknown" else None,
            ),
            decel_ms=decel_ms,
            easing=easing,
            stops_completely=any(not p.interrupted for p in decs),
        )

    # ---- drag ---------------------------------------------------------------------------------
    drag = None
    drags = [p for p in phases if p.kind == "drag"]
    if drags:
        peak = max(drags, key=lambda p: abs(p.v_peak))
        ratio = None
        if ctx.pointer_ratio is not None:
            ratio = mnum(ctx.pointer_ratio, ctx.pointer_ratio_confidence, nd=2)
        drag = DragBehavior(
            count=len(drags),
            follows_pointer=ctx.follows_pointer,
            pointer_ratio=ratio,
            peak_speed_px_s=mnum(
                abs(peak.v_peak),
                peak.confidence if ctx.drag_peak_confidence is None else ctx.drag_peak_confidence,
                nd=0,
            ),
        )

    # ---- inertia ------------------------------------------------------------------------------
    inertia = None
    inert = [p for p in phases if p.kind == "inertia"]
    if inert:
        exps = [p for p in inert if isinstance(p.fit, ExponentialFit)]
        tws = [p for p in inert if isinstance(p.fit, TweenFit)]
        model = "exponential" if len(exps) >= len(tws) else "tween"
        speeds = [abs(p.v_start) for p in inert]
        tau_ms = dur_ms = None
        easing = None
        if model == "exponential":
            vals = [p.fit.tau_ms.value for p in exps]  # type: ignore[union-attr]
            confs = [p.fit.tau_ms.confidence.value for p in exps]  # type: ignore[union-attr]
            agree = agreement_factor(vals, params)
            tau_ms = mnum(weighted_median(vals, confs), float(np.median(confs)) * agree, nd=0)
            speeds = [abs(p.fit.v0_px_s.value) for p in exps]  # type: ignore[union-attr]
        else:
            vals = [p.fit.duration_ms.value for p in tws]  # type: ignore[union-attr]
            confs = [p.fit.duration_ms.confidence.value for p in tws]  # type: ignore[union-attr]
            agree = agreement_factor(vals, params)
            dur_ms = mnum(weighted_median(vals, confs), float(np.median(confs)) * agree, nd=0)
            easing = max(tws, key=lambda p: p.confidence).fit.easing  # type: ignore[union-attr]
        stop = None
        stops = [p for p in exps if p.fit.stop_px_s is not None] if model == "exponential" else []  # type: ignore[union-attr]
        if stops:
            vals = [p.fit.stop_px_s for p in stops]  # type: ignore[union-attr]
            confs = [p.fit.tau_ms.confidence.value for p in stops]  # type: ignore[union-attr]
            # the stop speed is read off the fitted decay at the observed rest: as good as τ,
            # loosened by the instance spread
            stop = mnum(weighted_median(vals, confs),
                        float(np.median(confs)) * agreement_factor(vals, params), nd=1)  # fmt: skip
        inertia = InertiaBehavior(
            model=model,  # type: ignore[arg-type]
            tau_ms=tau_ms,
            duration_ms=dur_ms,
            easing=easing,
            release_speed_min_px_s=round(min(speeds), 0),
            release_speed_max_px_s=round(max(speeds), 0),
            instances=len(inert),
            stop_px_s=stop,
        )

    # ---- snap ---------------------------------------------------------------------------------
    snap = None
    if "snap" in kinds and ctx.snap_grid and ctx.pitch_px is not None:
        snaps = [p for p in phases if p.kind == "snap"]
        tw = [p for p in snaps if isinstance(p.fit, TweenFit)]
        dur_ms = None
        easing = None
        if tw:
            vals = [p.fit.duration_ms.value for p in tw]  # type: ignore[union-attr]
            confs = [p.fit.duration_ms.confidence.value for p in tw]  # type: ignore[union-attr]
            dur_ms = mnum(weighted_median(vals, confs), float(np.median(confs)), nd=0)
            easing = max(tw, key=lambda p: p.confidence).fit.easing  # type: ignore[union-attr]
        snap = SnapBehavior(
            kind="grid",
            step_px=mnum(
                ctx.pitch_px,
                min(ctx.pitch_confidence, ctx.snap_grid_confidence),
                params.cap_snap_step,
            ),
            duration_ms=dur_ms,
            easing=easing,
            overshoot=ctx.overshoot,
        )
    elif "stop" in kinds:
        snap = SnapBehavior(
            kind="abrupt_ambiguous", step_px=None, duration_ms=None, easing=None,
            overshoot=ctx.overshoot,
        )  # fmt: skip

    # ---- resume -------------------------------------------------------------------------------
    resume = None
    res = [p for p in phases if p.kind == "resume" and p.fit is not None and p.fit.model == "ramp"]
    if res:
        vals = [p.fit.duration_ms.value for p in res]  # type: ignore[union-attr]
        confs = [p.fit.duration_ms.confidence.value for p in res]  # type: ignore[union-attr]
        agree = agreement_factor(vals, params)
        best = max(res, key=lambda p: p.confidence)
        to_speed = abs(best.fit.to_px_s)  # type: ignore[union-attr]
        delays = ctx.resume_delays or [(0.0, None, 0.0)]
        d_rest = [d[0] * 1000.0 for d in delays]
        d_conf = [d[2] for d in delays]
        rel = [d[1] * 1000.0 for d in delays if d[1] is not None]
        rel_conf = [d[2] for d in delays if d[1] is not None]
        av = ctx.autoplay_velocity
        to_v = best.fit.to_px_s  # type: ignore[union-attr]
        resume = ResumeBehavior(
            delay_after_rest_ms=mnum(
                max(weighted_median(d_rest, d_conf), 0.0), float(np.median(d_conf)), nd=0
            ),  # fmt: skip
            delay_after_release_ms=(
                mnum(
                    max(weighted_median(rel, rel_conf), 0.0), float(np.median(rel_conf)) * 0.9, nd=0
                )  # fmt: skip
                if rel
                else None
            ),
            delay_after_leave_ms=(
                mnum(
                    max(
                        weighted_median(
                            [d * 1000.0 for d, _ in ctx.resume_leave_delays],
                            [cf for _, cf in ctx.resume_leave_delays],
                        ),
                        0.0,
                    ),
                    float(np.median([cf for _, cf in ctx.resume_leave_delays])),
                    nd=0,
                )  # fmt: skip
                if ctx.resume_leave_delays
                else None
            ),
            ramp_ms=mnum(weighted_median(vals, confs), float(np.median(confs)) * agree, nd=0),
            easing=best.fit.easing,  # type: ignore[union-attr]
            to_speed_px_s=round(to_speed, 1),
            direction_preserved=bool(av is None or (to_v > 0) == (av > 0)),
        )

    return Behavior(pause=pause, drag=drag, inertia=inertia, snap=snap, resume=resume)


__all__ = [
    "RAMP_CURVES",
    "TWEEN_CURVES",
    "BehaviorContext",
    "Fit",
    "agreement_factor",
    "aggregate",
    "exp_position",
    "exp_velocity",
    "fit_constant",
    "fit_easing",
    "fit_exponential",
    "fit_ramp",
    "fit_rest",
    "fit_tween",
    "ramp_integral",
    "ramp_position",
    "weighted_median",
]
