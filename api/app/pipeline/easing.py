"""Cubic-bezier evaluation, the named candidate set and curve-level fitting (PLAN §6.7, App. B).

Pure numpy, no I/O. Time handling (t0 / duration) lives in ``timing.py``; this module only
deals with normalized curves ``y = E(x)`` on ``x ∈ [0, 1]``.

Evaluation strategy
-------------------
A CSS ``cubic-bezier(x1, y1, x2, y2)`` is the parametric curve
``B(u) = 3(1-u)²u·P1 + 3(1-u)u²·P2 + u³`` with ``P0 = (0, 0)``, ``P3 = (1, 1)``. For
``x1, x2 ∈ [0, 1]`` the x-component is monotonic, so ``E(x)`` is a function.

* :func:`solve_bezier` inverts ``x(u)`` by vectorized bisection (exact to ~1e-12). It is used to
  build LUTs and for the free-bezier search.
* :func:`curve_table` is the "LUT" of PLAN §6.7: ``E`` sampled on a **uniform x grid** of
  ``lut_points`` (513) points, built with the exact solver. Uniform x makes batched evaluation of
  many curves a gather + lerp (:func:`eval_tables`) instead of one ``np.interp`` per curve.
  Final RMSE is reported with ``lut_points_final`` (2001).

Free bezier search
------------------
For fixed ``(x1, x2)`` the curve value at a given ``x`` is *linear* in ``(y1, y2)``:
``y = a(u)·y1 + b(u)·y2 + u³`` where ``u = u(x; x1, x2)``. So the SSE over all ``(y1, y2)``
grid points of a given ``(x1, x2)`` is a quadratic form computed from six sums. This makes the
PLAN's full grid (``x ∈ {0, .1, …, 1}²``, ``y ∈ {-.5, -.25, …, 1.5}²``) plus two step/4
refinement rounds cheap (a few ms) without changing what is searched.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.models.ir import CSS_KEYWORD_BEZIER, CssEasingKeyword, Easing, EasingFamily
from app.pipeline.params import DEFAULT_PARAMS, FitParams

Bezier = tuple[float, float, float, float]

#: Bisection iterations for ``x(u) = x``; 2^-45 ≈ 3e-14 (LUTs / exact evaluation).
_BISECT_ITERS = 44
#: Bisection iterations inside the free-bezier grid search; 2^-27 ≈ 7e-9 is ample there.
_BISECT_ITERS_SEARCH = 26
#: A curve whose peak exceeds this is reported as overshooting (family ``overshoot``).
_OVERSHOOT_CURVE_PEAK = 1.02
#: Free-bezier coarse grid (PLAN §6.7).
_FREE_X_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 10)
_FREE_Y_GRID = np.round(np.arange(-0.5, 1.5 + 1e-9, 0.25), 10)
_FREE_Y_BOUNDS = (-0.5, 1.5)
#: Points per side around the incumbent in a refinement round (grid = 2·k+1 per dimension).
_REFINE_HALF_POINTS = 4
#: Free-fit bezier coordinates are reported with this many decimals.
_FREE_DECIMALS = 3


# --------------------------------------------------------------------------------------------
# Candidates (Appendix B)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EasingCurve:
    """A named easing candidate."""

    name: str
    bezier: Bezier
    family: EasingFamily
    keyword: CssEasingKeyword | None = None
    overshoot_only: bool = False  # only considered when the data overshoots


def _kw(name: CssEasingKeyword, family: EasingFamily) -> EasingCurve:
    return EasingCurve(name, CSS_KEYWORD_BEZIER[name], family, keyword=name)


#: PLAN Appendix B, in table order. Names are the ``Easing.nearest_named`` vocabulary.
CANDIDATES: tuple[EasingCurve, ...] = (
    _kw("linear", "linear"),
    _kw("ease", "ease"),
    _kw("ease-in", "ease-in"),
    _kw("ease-out", "ease-out"),
    _kw("ease-in-out", "ease-in-out"),
    EasingCurve("easeOutQuad", (0.25, 0.46, 0.45, 0.94), "ease-out"),
    EasingCurve("easeOutCubic", (0.215, 0.61, 0.355, 1.0), "ease-out"),
    EasingCurve("easeOutQuart", (0.165, 0.84, 0.44, 1.0), "ease-out"),
    EasingCurve("easeOutExpo", (0.19, 1.0, 0.22, 1.0), "ease-out"),
    EasingCurve("expoOut", (0.16, 1.0, 0.3, 1.0), "ease-out"),
    EasingCurve("easeInCubic", (0.55, 0.055, 0.675, 0.19), "ease-in"),
    EasingCurve("easeInOutCubic", (0.645, 0.045, 0.355, 1.0), "ease-in-out"),
    EasingCurve("material-standard", (0.4, 0.0, 0.2, 1.0), "ease-in-out"),
    EasingCurve("material-decelerate", (0.0, 0.0, 0.2, 1.0), "ease-out"),
    EasingCurve("material-accelerate", (0.4, 0.0, 1.0, 1.0), "ease-in"),
    EasingCurve("backOut", (0.34, 1.56, 0.64, 1.0), "overshoot", overshoot_only=True),
    EasingCurve("easeOutBack", (0.175, 0.885, 0.32, 1.275), "overshoot", overshoot_only=True),
)
CANDIDATES_BY_NAME: dict[str, EasingCurve] = {c.name: c for c in CANDIDATES}

#: Families scored as interchangeable by the "family correct" metric (PLAN App. B note).
_LENIENT_FAMILY_PAIRS = frozenset({frozenset({"ease", "ease-in-out"})})


def candidate_set(include_overshoot: bool) -> tuple[EasingCurve, ...]:
    """Candidates for a fit; back/overshoot curves only when the data overshoots."""
    return tuple(c for c in CANDIDATES if include_overshoot or not c.overshoot_only)


def families_match(a: str, b: str) -> bool:
    """Family equality for accuracy scoring: ``ease`` and ``ease-in-out`` are not penalized."""
    return a == b or frozenset({a, b}) in _LENIENT_FAMILY_PAIRS


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def bezier_point(bezier: Sequence[float], u: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ``(x(u), y(u))`` of the curve at parameter ``u``."""
    x1, y1, x2, y2 = bezier
    u = np.asarray(u, dtype=np.float64)
    a = 3.0 * (1.0 - u) ** 2 * u
    b = 3.0 * (1.0 - u) * u**2
    c = u**3
    return a * x1 + b * x2 + c, a * y1 + b * y2 + c


def solve_u(
    x1: np.ndarray | float,
    x2: np.ndarray | float,
    x: np.ndarray,
    iters: int = _BISECT_ITERS,
) -> np.ndarray:
    """Parameter ``u`` with ``x(u) = x`` for monotonic curves (vectorized bisection).

    ``x1``, ``x2`` and ``x`` broadcast together; ``x`` is clipped to ``[0, 1]``. The error is
    ``2^-(iters+1)`` in ``u``.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    shape = np.broadcast_shapes(x1.shape, x2.shape, x.shape)
    u = np.full(shape, 0.5)
    step = 0.25
    for _ in range(iters):
        om = 1.0 - u
        xm = 3.0 * om * u * (om * x1 + u * x2) + u * u * u
        u += np.where(xm < x, step, -step)
        step *= 0.5
    return u


def solve_bezier(bezier: Sequence[float], x: np.ndarray | float) -> np.ndarray:
    """Exact ``E(x)`` (to ~1e-12) for one curve. ``x`` outside ``[0, 1]`` is clamped."""
    x1, _, x2, _ = bezier
    u = solve_u(x1, x2, np.asarray(x, dtype=np.float64))
    return bezier_point(bezier, u)[1]


def curve_table(bezier: Sequence[float], n: int = DEFAULT_PARAMS.fit.lut_points) -> np.ndarray:
    """LUT: ``E`` sampled at ``n`` uniform x positions in ``[0, 1]`` (float64, shape ``(n,)``)."""
    return _curve_table_cached(tuple(float(v) for v in bezier), int(n)).copy()


@lru_cache(maxsize=256)
def _curve_table_cached(bezier: Bezier, n: int) -> np.ndarray:
    table = solve_bezier(bezier, np.linspace(0.0, 1.0, n))
    table.setflags(write=False)
    return table


def tables_for(curves: Iterable[Sequence[float]], n: int) -> np.ndarray:
    """Stack LUTs of several curves → ``(C, n)``."""
    return np.stack([_curve_table_cached(tuple(float(v) for v in b), int(n)) for b in curves])


def eval_tables(tables: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate LUTs at ``x`` (clamped to ``[0, 1]``).

    ``tables`` is ``(C, n)``; ``x`` must have a leading axis of size ``C`` (per-curve positions)
    or 1 (positions shared by all curves). The result has shape ``(C, *x.shape[1:])``.
    """
    c, n = tables.shape
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if x.ndim == 0 or x.shape[0] not in (1, c):
        raise ValueError(f"x needs a leading axis of size 1 or {c}, got shape {x.shape}")
    x = np.broadcast_to(x, (c, *x.shape[1:]))
    pos = x * (n - 1)
    i = np.minimum(pos.astype(np.int64), n - 2)
    f = pos - i
    row = (np.arange(c) * n).reshape((c,) + (1,) * (x.ndim - 1))
    flat = tables.reshape(-1)
    lo = flat[i + row]
    hi = flat[i + 1 + row]
    return lo + (hi - lo) * f


def eval_curve(bezier: Sequence[float], x: np.ndarray | float, n: int | None = None) -> np.ndarray:
    """``E(x)`` through the LUT (``n`` defaults to ``lut_points``)."""
    table = _curve_table_cached(
        tuple(float(v) for v in bezier), int(n or DEFAULT_PARAMS.fit.lut_points)
    )
    return eval_tables(table[None, :], np.asarray(x, dtype=np.float64)[None, ...])[0]


def curve_distance(a: Sequence[float], b: Sequence[float], n: int = 257) -> float:
    """RMS difference of two curves over uniform ``x ∈ [0, 1]`` (progress units)."""
    ta = _curve_table_cached(tuple(float(v) for v in a), n)
    tb = _curve_table_cached(tuple(float(v) for v in b), n)
    return float(np.sqrt(np.mean((ta - tb) ** 2)))


def curve_peak(bezier: Sequence[float]) -> float:
    """Maximum of ``y(u)`` over ``u ∈ [0, 1]`` (overshoot magnitude when > 1)."""
    _, y = bezier_point(bezier, np.linspace(0.0, 1.0, 257))
    return float(np.max(y))


def nearest_named(bezier: Sequence[float], include_overshoot: bool = True) -> EasingCurve:
    """Closest Appendix B candidate by :func:`curve_distance`."""
    pool = candidate_set(include_overshoot)
    dists = [curve_distance(bezier, c.bezier) for c in pool]
    return pool[int(np.argmin(dists))]


def family_of(bezier: Sequence[float]) -> EasingFamily:
    """Family of an arbitrary curve: ``overshoot`` if it peaks above 1, else nearest named's."""
    if curve_peak(bezier) > _OVERSHOOT_CURVE_PEAK:
        return "overshoot"
    return nearest_named(bezier, include_overshoot=False).family


def keyword_for(bezier: Sequence[float]) -> CssEasingKeyword | None:
    """The CSS keyword whose bezier equals ``bezier`` exactly, if any."""
    for kw, ref in CSS_KEYWORD_BEZIER.items():
        if all(abs(float(a) - b) <= 1e-9 for a, b in zip(bezier, ref, strict=True)):
            return kw  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------------------------
# Overshoot detection
# --------------------------------------------------------------------------------------------


def detect_overshoot(progress: np.ndarray, params: FitParams = DEFAULT_PARAMS.fit) -> bool:
    """PLAN §6.7: overshoot if progress exceeds ``overshoot_p`` on ≥ ``overshoot_min_samples``."""
    p = np.asarray(progress, dtype=np.float64)
    p = p[np.isfinite(p)]
    return int(np.count_nonzero(p > params.overshoot_p)) >= params.overshoot_min_samples


# --------------------------------------------------------------------------------------------
# Free bezier (curve shape only, at fixed t0 / D)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FreeFit:
    bezier: Bezier
    sse: float  # over the given in-motion samples only


def fit_free_bezier(
    x: np.ndarray, p: np.ndarray, params: FitParams = DEFAULT_PARAMS.fit
) -> FreeFit:
    """Best ``cubic-bezier`` for samples ``(x_i, p_i)`` with ``x_i ∈ (0, 1)``.

    Coarse grid ``x1, x2 ∈ {0, 0.1, …, 1}``, ``y1, y2 ∈ {-0.5, -0.25, …, 1.5}`` followed by
    ``params.refine_rounds`` rounds at step/4 around the incumbent (PLAN §6.7). The SSE is exact
    on each grid point (closed-form quadratic in ``y1, y2``).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if x.size == 0:
        raise ValueError("fit_free_bezier needs at least one sample")

    xs1, xs2 = np.meshgrid(_FREE_X_GRID, _FREE_X_GRID, indexing="ij")
    ys1, ys2 = np.meshgrid(_FREE_Y_GRID, _FREE_Y_GRID, indexing="ij")
    best = _free_grid(x, p, xs1.ravel(), xs2.ravel(), ys1.ravel(), ys2.ravel())
    x_step, y_step = 0.1, 0.25
    for _ in range(params.refine_rounds):
        x_step /= 4.0
        y_step /= 4.0
        offs = np.arange(-_REFINE_HALF_POINTS, _REFINE_HALF_POINTS + 1)
        gx1 = np.unique(np.clip(best[0] + offs * x_step, 0.0, 1.0))
        gx2 = np.unique(np.clip(best[2] + offs * x_step, 0.0, 1.0))
        gy1 = np.unique(np.clip(best[1] + offs * y_step, *_FREE_Y_BOUNDS))
        gy2 = np.unique(np.clip(best[3] + offs * y_step, *_FREE_Y_BOUNDS))
        a1, a2 = np.meshgrid(gx1, gx2, indexing="ij")
        b1, b2 = np.meshgrid(gy1, gy2, indexing="ij")
        best = _free_grid(x, p, a1.ravel(), a2.ravel(), b1.ravel(), b2.ravel())
    bez = tuple(round(float(v), _FREE_DECIMALS) for v in best[:4])
    return FreeFit(bezier=bez, sse=float(best[4]))  # type: ignore[arg-type]


def _free_grid(
    x: np.ndarray,
    p: np.ndarray,
    gx1: np.ndarray,
    gx2: np.ndarray,
    gy1: np.ndarray,
    gy2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Evaluate SSE on the product grid ``(gx1[i], gx2[i]) × (gy1[j], gy2[j])``; return argmin."""
    u = solve_u(gx1[:, None], gx2[:, None], x[None, :], _BISECT_ITERS_SEARCH)  # (X, S)
    om = 1.0 - u
    a = 3.0 * om * om * u
    b = 3.0 * om * u * u
    r = p[None, :] - u**3
    s_rr = np.sum(r * r, axis=1)[:, None]
    s_ra = np.sum(r * a, axis=1)[:, None]
    s_rb = np.sum(r * b, axis=1)[:, None]
    s_aa = np.sum(a * a, axis=1)[:, None]
    s_ab = np.sum(a * b, axis=1)[:, None]
    s_bb = np.sum(b * b, axis=1)[:, None]
    y1 = gy1[None, :]
    y2 = gy2[None, :]
    sse = (
        s_rr
        - 2.0 * y1 * s_ra
        - 2.0 * y2 * s_rb
        + y1 * y1 * s_aa
        + 2.0 * y1 * y2 * s_ab
        + y2 * y2 * s_bb
    )
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    return (
        float(gx1[i]),
        float(gy1[j]),
        float(gx2[i]),
        float(gy2[j]),
        float(max(sse[i, j], 0.0)),
    )


# --------------------------------------------------------------------------------------------
# IR construction
# --------------------------------------------------------------------------------------------


def make_easing(
    bezier: Sequence[float],
    rmse: float,
    *,
    named: EasingCurve | None = None,
) -> Easing:
    """Build the IR ``Easing`` for a fitted curve.

    ``named`` = the winning Appendix B candidate (its exact bezier is used); ``None`` = a free
    fit, whose family/nearest name are derived from the shape. The ``overshoot`` flag is set when
    the reported curve overshoots.
    """
    if named is not None:
        bez = named.bezier
        family: EasingFamily = named.family
        nearest = named.name
        keyword = named.keyword
    else:
        bez = tuple(float(v) for v in bezier)  # type: ignore[assignment]
        family = family_of(bez)
        nearest = nearest_named(bez).name
        keyword = keyword_for(bez)
    flags = (
        ["overshoot"] if family == "overshoot" or curve_peak(bez) > _OVERSHOOT_CURVE_PEAK else []
    )
    return Easing(
        keyword=keyword,
        cubic_bezier=bez,
        nearest_named=nearest,
        family=family,
        rmse=round(max(float(rmse), 0.0), 4),
        flags=flags,  # type: ignore[arg-type]
    )


__all__ = [
    "CANDIDATES",
    "CANDIDATES_BY_NAME",
    "Bezier",
    "EasingCurve",
    "FreeFit",
    "bezier_point",
    "candidate_set",
    "curve_distance",
    "curve_peak",
    "curve_table",
    "detect_overshoot",
    "eval_curve",
    "eval_tables",
    "families_match",
    "family_of",
    "fit_free_bezier",
    "keyword_for",
    "make_easing",
    "nearest_named",
    "solve_bezier",
    "solve_u",
    "tables_for",
]
