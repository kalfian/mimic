"""Confidence formulas for continuous-scroller values (PLAN-continuous §4.6).

Starting formulas, to be calibrated in P2 (like ``make calibrate``) without relaxing §8.4.
All thresholds come from ``ContinuousParams``; the shared VFR / estimated-timestamp penalties
from ``ConfidenceParams``; bands from ``models.ir.band_for`` (via
``pipeline.confidence.confidence``)::

    q_track      = clip((mean q in phase − q_track_lo) / q_track_span, 0, 1)
    speed        = q_track · clip(T_obs / speed_full_s, 0, 1) · clip(1 − cv / speed_cv_zero, 0, 1)
                   cap cap_speed
    tau          = q_track · clip(1 − tau_rel_se_k · rel_se, 0, 1)
                   · clip((n − n_min) / n_span, 0, 1)                   cap cap_tau
    release v0   = q_track · clip(1 − dt / τ, 0, 1)                     cap cap_release
    phase label  = clip(label_sep_base + ΔBIC / label_sep_bic_span, 0, 1)
    duration     = clip(1 − dt / D, 0, 1) · fit factor − VFR / timestamp penalties

``cv`` of a speed is its relative standard error (position-domain fit), not a velocity spread.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.models.ir import Confidence
from app.pipeline.confidence import clip01
from app.pipeline.confidence import confidence as make_confidence
from app.pipeline.params import DEFAULT_PARAMS, ConfidenceParams, ContinuousParams

_P = DEFAULT_PARAMS.continuous
_C = DEFAULT_PARAMS.confidence
#: Fit factor: progress rmse at which a tween / ramp fit stops counting (same idea as
#: ``ConfidenceParams.fit_rmse_zero``; ramps are fitted in position, so their progress rmse is
#: looser). TODO(calibrate P2): module-local, candidate for ``ContinuousParams``.
FIT_RMSE_ZERO = 0.25


def ir_confidence(score: float, cap: float | None = None) -> Confidence:
    """IR ``Confidence`` (rounded, banded), optionally capped."""
    return make_confidence(score if math.isfinite(score) else 0.0, cap)


def q_track(quality: Sequence[float] | np.ndarray, params: ContinuousParams = _P) -> float:
    q = np.asarray(quality, dtype=np.float64)
    q = q[np.isfinite(q)]
    if q.size == 0:
        return 0.0
    return clip01((float(np.mean(q)) - params.q_track_lo) / params.q_track_span)


def timing_penalty(
    is_vfr: bool, timestamps_estimated: bool, params: ConfidenceParams = _C
) -> float:
    return (params.pen_vfr if is_vfr else 0.0) + (
        params.pen_timestamps_estimated if timestamps_estimated else 0.0
    )


def speed_confidence(
    qt: float, t_obs: float, rel_se: float, params: ContinuousParams = _P
) -> float:
    cv_f = clip01(1.0 - rel_se / params.speed_cv_zero) if math.isfinite(rel_se) else 0.0
    return min(qt * clip01(t_obs / params.speed_full_s) * cv_f, params.cap_speed)


def tau_confidence(qt: float, rel_se: float, n: int, params: ContinuousParams = _P) -> float:
    se_f = clip01(1.0 - params.tau_rel_se_k * rel_se) if math.isfinite(rel_se) else 0.0
    n_f = clip01((n - params.tau_n_min) / params.tau_n_span)
    return min(qt * se_f * n_f, params.cap_tau)


def release_confidence(qt: float, dt: float, tau: float, params: ContinuousParams = _P) -> float:
    return min(qt * clip01(1.0 - dt / tau) if tau > 0 else 0.0, params.cap_release)


def label_confidence(delta_bic: float, params: ContinuousParams = _P) -> float:
    if not math.isfinite(delta_bic):
        delta_bic = params.label_sep_bic_span
    return clip01(params.label_sep_base + delta_bic / params.label_sep_bic_span)


def fit_factor(progress_rmse: float) -> float:
    return clip01(1.0 - progress_rmse / FIT_RMSE_ZERO)


def duration_confidence(
    dt: float,
    duration: float,
    fit: float,
    *,
    is_vfr: bool = False,
    timestamps_estimated: bool = False,
) -> float:
    res = clip01(1.0 - dt / duration) if duration > 0 else 0.0
    return clip01(res * fit - timing_penalty(is_vfr, timestamps_estimated))


def degraded_fraction(quality: np.ndarray, params: ContinuousParams = _P) -> float:
    q = np.asarray(quality, dtype=np.float64)
    return float(np.mean(q < params.degraded_q)) if q.size else 0.0


__all__ = [
    "FIT_RMSE_ZERO",
    "degraded_fraction",
    "duration_confidence",
    "fit_factor",
    "ir_confidence",
    "label_confidence",
    "q_track",
    "release_confidence",
    "speed_confidence",
    "tau_confidence",
    "timing_penalty",
]
