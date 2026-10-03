"""Confidence scoring helpers, penalties, caps and bands (PLAN §6.8).

Starting formulas, to be calibrated in Phase 4 against the synthetic suite. All parameters come
from ``params.confidence`` (``ConfidenceParams``); bands from ``models.ir.band_for``.

Per transition::

    q_track      = clip((mean_rho - rho_lo) / rho_span, 0, 1)      geometric properties
                 = clip(mean R², 0, 1)                              photometric properties
    snr_factor   = clip(|Δ| / max(σ_stable, floor) / snr_full, 0, 1)
    value        = q_track · snr_factor                 (− cursor-over-element penalty)
    fit_factor   = clip(1 - rmse_x / fit_rmse_zero, 0, 1)
                   rmse_x = sqrt(max(rmse² − σ_p², 0)), σ_p = σ_stable / |Δ| (Phase 4: the
                   part of the fit error that measurement noise does not explain)
    timing       = fit_factor · clip(1 - dt / D, 0, 1) · q_track   (− timestamp / VFR penalties)
    easing       = clip(n_factor · fit_factor · sep_factor, 0, easing_max)
                   n_factor   = clip((n_in - n_min) / n_span, 0, 1)
                   sep_factor = clip(sep_base + (rmse_2nd_family - rmse_best) / sep_span, 0, 1)
    overall      = (value · timing · easing)^(1/3)

Caps (property / element kind) bound ``value`` and ``overall``; every score is ≤ ``cap_all``.
Scores are rounded to 3 decimals and the band is derived from the rounded ``overall`` so the IR
validator (``band == band_for(overall)``) always holds.
"""

from __future__ import annotations

import math

import numpy as np

from app.models.ir import Confidence, ConfidenceBand, ElementKind, TransitionConfidence, band_for
from app.models.measure import PropertySeries
from app.pipeline.params import DEFAULT_PARAMS, ConfidenceParams
from app.pipeline.timing import COLOR_PROPS, TimingFit, series_delta

#: Properties whose ``quality`` is an ECC rho (tracked geometrically). Others carry a fit R².
GEOMETRIC_PROPS = frozenset({"translateX", "translateY", "scale", "scaleX", "scaleY", "height"})
#: Properties whose ``values`` are the α blend progress (0 → 1), not the property's own unit.
ALPHA_PROPS = frozenset({"color", "background-color", "box-shadow", "content"})
_DECIMALS = 3


def clip01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0)) if math.isfinite(v) else 0.0


def band(score: float) -> ConfidenceBand:
    """Band for a score (high ≥ 0.8, medium ≥ 0.5, low otherwise)."""
    return band_for(score)


def confidence(score: float, cap: float | None = None) -> Confidence:
    """IR ``Confidence`` from a score, optionally capped; rounded with a matching band."""
    s = clip01(score)
    if cap is not None:
        s = min(s, cap)
    s = round(s, _DECIMALS)
    return Confidence(value=s, band=band_for(s))


# --------------------------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------------------------


def q_track_geometric(
    mean_rho: float, params: ConfidenceParams = DEFAULT_PARAMS.confidence
) -> float:
    return clip01((mean_rho - params.rho_lo) / params.rho_span)


def q_track_photometric(r2: float) -> float:
    return clip01(r2)


def series_q_track(
    series: PropertySeries, params: ConfidenceParams = DEFAULT_PARAMS.confidence
) -> float:
    """``q_track`` from the per-frame ``quality`` (ECC rho or blend R²) of a series."""
    q = np.asarray(series.quality, dtype=np.float64)
    q = q[np.isfinite(q)]
    if q.size == 0:
        return params.q_track_unknown
    mean_q = float(np.mean(q))
    if series.property in GEOMETRIC_PROPS:
        return q_track_geometric(mean_q, params)
    return q_track_photometric(mean_q)


def snr_factor(
    delta: float, sigma: float, floor: float, params: ConfidenceParams = DEFAULT_PARAMS.confidence
) -> float:
    """``clip(|Δ| / max(σ, floor) / snr_full, 0, 1)``."""
    denom = max(float(sigma) if math.isfinite(sigma) else 0.0, floor)
    if denom <= 0.0:
        return 0.0
    return clip01(abs(delta) / denom / params.snr_full)


def snr_floor(prop: str, params: ConfidenceParams = DEFAULT_PARAMS.confidence) -> float:
    """σ floor per property unit (PLAN §6.8: 0.05 px / 0.002 scale / 1 ΔE).

    Opacity and α-progress series have no PLAN floor; they use the ratio floor (0.002).
    """
    if prop in COLOR_PROPS:
        return params.snr_floor_delta_e
    if prop in ("translateX", "translateY", "height", "border-radius"):
        return params.snr_floor_px
    return params.snr_floor_scale


def series_snr_factor(
    series: PropertySeries, params: ConfidenceParams = DEFAULT_PARAMS.confidence
) -> float:
    """SNR factor of a series in the property's own unit.

    For colors, ``values``/``stable_std`` are in α units; both Δ and σ are converted to ΔE76
    (σ_ΔE = σ_α · ΔE) so the ΔE floor applies.
    """
    prop = series.property
    delta = series_delta(series)
    sigma = float(series.stable_std)
    if prop in COLOR_PROPS:
        sigma *= delta
    elif prop in ALPHA_PROPS:
        delta = abs(float(series.v_end) - float(series.v_start))
    return snr_factor(delta, sigma, snr_floor(prop, params), params)


def noise_progress(series: PropertySeries) -> float:
    """Measurement noise in progress units: σ of the stable frames / |Δ| (0 if unknown)."""
    span = abs(float(series.v_end) - float(series.v_start))
    sigma = float(series.stable_std)
    if span <= 0 or not math.isfinite(sigma) or sigma <= 0:
        return 0.0
    return sigma / span


def excess_rmse(rmse: float, noise: float) -> float:
    """Fit RMSE beyond what the measurement noise alone explains (Phase 4 calibration).

    A perfect curve on noisy samples still has ``rmse ≈ σ``; only the excess says the curve or
    timing is wrong. ``sqrt(max(rmse² − σ², 0))``.
    """
    if not math.isfinite(rmse):
        return rmse
    return math.sqrt(max(rmse * rmse - noise * noise, 0.0))


def fit_factor(rmse: float, params: ConfidenceParams = DEFAULT_PARAMS.confidence) -> float:
    return clip01(1.0 - rmse / params.fit_rmse_zero)


def value_confidence(
    q_track: float,
    snr: float,
    *,
    cursor_over_element: bool = False,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> float:
    v = q_track * snr
    if cursor_over_element:
        v -= params.pen_cursor_over_element
    return clip01(v)


def timing_confidence(
    rmse: float,
    dt_s: float,
    duration_s: float,
    q_track: float,
    *,
    timestamps_estimated: bool = False,
    is_vfr: bool = False,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> float:
    res = clip01(1.0 - dt_s / duration_s) if duration_s > 0 else 0.0
    v = fit_factor(rmse, params) * res * q_track
    if timestamps_estimated:
        v -= params.pen_timestamps_estimated
    if is_vfr:
        v -= params.pen_vfr
    return clip01(v)


def easing_confidence(
    n_in: int,
    rmse_best: float,
    rmse_second_family: float,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> float:
    n_factor = clip01((n_in - params.n_min) / params.n_span)
    gap = rmse_second_family - rmse_best if math.isfinite(rmse_second_family) else params.sep_span
    sep = clip01(params.sep_base + gap / params.sep_span)
    return min(clip01(n_factor * fit_factor(rmse_best, params) * sep), params.easing_max)


def overall_confidence(value: float, timing: float, easing: float) -> float:
    prod = clip01(value) * clip01(timing) * clip01(easing)
    return float(prod ** (1.0 / 3.0)) if prod > 0 else 0.0


# --------------------------------------------------------------------------------------------
# Caps
# --------------------------------------------------------------------------------------------


def cap_for(
    prop: str,
    element_kind: ElementKind | None = None,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> float:
    """Confidence cap for a property on an element of ``element_kind`` (PLAN §6.8)."""
    if element_kind == "content_change" or prop == "content":
        return params.cap_content_change
    if element_kind == "backdrop":
        return params.cap_backdrop
    caps = {
        "color": params.cap_color,
        "background-color": params.cap_color,
        "box-shadow": params.cap_shadow,
        "border-radius": params.cap_radius,
        "height": params.cap_height,
    }
    if prop == "opacity":
        appear = element_kind in ("appear", "disappear")
        return params.cap_opacity_appear if appear else params.cap_opacity
    return min(caps.get(prop, params.cap_all), params.cap_all)


def origin_confidence(score: float, params: ConfidenceParams = DEFAULT_PARAMS.confidence) -> float:
    """Transform-origin confidence (cap 0.7)."""
    return min(clip01(score), params.cap_origin, params.cap_all)


def static_confidence(
    score: float, prop: str, params: ConfidenceParams = DEFAULT_PARAMS.confidence
) -> Confidence:
    """Static element measurement (``border-radius`` / ``box-shadow``) as an IR ``Confidence``."""
    return confidence(score, cap=cap_for(prop, None, params))


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------


def make_transition_confidence(
    value: float,
    timing: float,
    easing: float,
    *,
    cap: float,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> TransitionConfidence:
    """Apply caps, compute ``overall``, round, derive the band."""
    hard = params.cap_all
    v = round(min(clip01(value), cap, hard), _DECIMALS)
    t = round(min(clip01(timing), hard), _DECIMALS)
    e = round(min(clip01(easing), params.easing_max, hard), _DECIMALS)
    o = round(min(overall_confidence(v, t, e), cap, hard), _DECIMALS)
    return TransitionConfidence(overall=o, value=v, timing=t, easing=e, band=band_for(o))


def transition_confidence(
    series: PropertySeries,
    fit: TimingFit,
    *,
    element_kind: ElementKind | None = None,
    timestamps_estimated: bool = False,
    is_vfr: bool = False,
    cursor_over_element: bool = False,
    params: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> TransitionConfidence:
    """Full §6.8 confidence for a fitted series."""
    q = series_q_track(series, params)
    value = value_confidence(
        q, series_snr_factor(series, params), cursor_over_element=cursor_over_element, params=params
    )
    noise = noise_progress(series)
    rmse = excess_rmse(fit.rmse, noise)
    rmse_2nd = excess_rmse(fit.rmse_second_family, noise)
    timing = timing_confidence(
        rmse,
        fit.dt_s,
        fit.duration_s,
        q,
        timestamps_estimated=timestamps_estimated,
        is_vfr=is_vfr,
        params=params,
    )
    easing = easing_confidence(fit.n_in, rmse, rmse_2nd, params)
    return make_transition_confidence(
        value, timing, easing, cap=cap_for(series.property, element_kind, params), params=params
    )


__all__ = [
    "ALPHA_PROPS",
    "GEOMETRIC_PROPS",
    "band",
    "cap_for",
    "clip01",
    "confidence",
    "easing_confidence",
    "fit_factor",
    "make_transition_confidence",
    "origin_confidence",
    "overall_confidence",
    "q_track_geometric",
    "q_track_photometric",
    "series_q_track",
    "series_snr_factor",
    "snr_factor",
    "snr_floor",
    "static_confidence",
    "timing_confidence",
    "transition_confidence",
    "value_confidence",
]
