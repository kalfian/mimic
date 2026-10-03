"""Track M2 — timing.py: normalization, gates, joint (t0, D, curve) fit (PLAN §6.7, §11.1).

The noisy-recovery DoD (Appendix B curves, 60/30 fps, σ = 0.02) lives at the bottom. Synthetic
samples come from an independent scalar bezier implementation (``ref_bezier``), not from
``pipeline/easing.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import ColorValue, PxValue, RatioValue, ShadowValue
from app.models.measure import PropertySeries
from app.pipeline import timing as tm
from app.pipeline.easing import CANDIDATES, CANDIDATES_BY_NAME, families_match


def ref_bezier(bezier: tuple[float, ...], x: float) -> float:
    """Independent scalar evaluation (bisection on x(u))."""
    x1, y1, x2, y2 = bezier
    x = min(max(x, 0.0), 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        u = (lo + hi) / 2
        if 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u**3 < x:
            lo = u
        else:
            hi = u
    u = (lo + hi) / 2
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u**3


_REF_GRID = np.linspace(0.0, 1.0, 4001)
_REF_CACHE: dict[tuple[float, ...], np.ndarray] = {}


def truth(bezier: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Vectorized wrapper over ``ref_bezier`` (dense cached reference grid + interp)."""
    key = tuple(bezier)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = np.array([ref_bezier(key, float(v)) for v in _REF_GRID])
    return np.interp(np.clip(x, 0.0, 1.0), _REF_GRID, _REF_CACHE[key])


def synth(
    bezier: tuple[float, ...],
    *,
    fps: float,
    t0: float,
    dur: float,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    start: float = 0.5,
    end: float = 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(start, end, 1.0 / fps)
    p = truth(bezier, (t - t0) / dur)
    if sigma:
        assert rng is not None
        p = p + rng.normal(0.0, sigma, t.size)
    return t, p


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def test_normalize_progress_forward_and_reverse() -> None:
    v = np.array([0.0, -4.0, -8.0])
    assert tm.normalize_progress(v, 0.0, -8.0) == pytest.approx([0, 0.5, 1])
    # reverse segment: v_start = state B, v_end = state A
    assert tm.normalize_progress(v[::-1], -8.0, 0.0) == pytest.approx([0, 0.5, 1])
    with pytest.raises(ValueError):
        tm.normalize_progress(v, 1.0, 1.0)


def test_frame_interval_and_resolution() -> None:
    t = np.arange(0, 1, 1 / 60)
    assert tm.frame_interval(t) == pytest.approx(1 / 60)
    assert tm.timing_resolution_ms(t) == 17
    assert tm.timing_resolution_ms(np.arange(0, 1, 1 / 30)) == 33
    assert tm.frame_interval(np.array([1.0])) == 0.0


def test_progress_samples_decimates() -> None:
    t = np.arange(0, 3, 1 / 60)
    s = tm.progress_samples(t, np.linspace(0, 1, t.size))
    assert len(s) <= 60 and s[0] == (0, 0.0) and s[-1][1] == 1.0
    assert all(isinstance(ms, int) for ms, _ in s)
    assert tm.progress_samples(t[:10], np.zeros(10)) == [(round(x * 1000), 0.0) for x in t[:10]]


def test_coarse_crossings_linear_and_spike_robustness() -> None:
    t, p = synth((0, 0, 1, 1), fps=60, t0=1.0, dur=0.3)
    t5, t95 = tm.coarse_crossings(t, p)
    assert t5 == pytest.approx(1.015, abs=0.002)
    assert t95 == pytest.approx(1.285, abs=0.002)
    spiked = p.copy()
    spiked[10] = 0.4  # isolated pre-motion spike
    spiked[-20] = 0.6  # isolated post-settle dropout
    assert tm.coarse_crossings(t, spiked) == pytest.approx((t5, t95), abs=1e-9)
    assert tm.coarse_crossings(t, np.zeros_like(t)) is None


def test_coarse_crossings_not_settled_uses_last_sample() -> None:
    t, p = synth((0, 0, 1, 1), fps=60, t0=1.0, dur=1.5, end=2.0)
    _, t95 = tm.coarse_crossings(t, p)
    assert t95 == pytest.approx(t[-1])


# --------------------------------------------------------------------------------------------
# Significance gate
# --------------------------------------------------------------------------------------------


def _series(prop: str, fv, tv, v_start=0.0, v_end=1.0, unit="px") -> PropertySeries:  # type: ignore[no-untyped-def]
    return PropertySeries(
        element_id="e1",
        property=prop,  # type: ignore[arg-type]
        segment="fwd",
        times=np.zeros(1),
        values=np.zeros(1),
        quality=np.ones(1),
        v_start=v_start,
        v_end=v_end,
        unit=unit,  # type: ignore[arg-type]
        from_value=fv,
        to_value=tv,
    )


def test_delta_e76_reference_values() -> None:
    assert tm.delta_e76("#FFFFFF", "#FFFFFF") == 0.0
    assert tm.delta_e76("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.01)
    lab = tm.hex_to_lab("#FF0000")
    assert lab == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def test_significance_gates() -> None:
    sig = tm.is_significant
    assert sig(_series("translateY", PxValue(number=0), PxValue(number=-0.8), 0, -0.8))
    assert not sig(_series("translateY", PxValue(number=0), PxValue(number=-0.5), 0, -0.5))
    assert sig(_series("scale", RatioValue(number=1), RatioValue(number=1.01), 1, 1.01, "ratio"))
    assert not sig(
        _series("scale", RatioValue(number=1), RatioValue(number=1.005), 1, 1.005, "ratio")
    )
    assert not sig(
        _series("opacity", RatioValue(number=1), RatioValue(number=0.97), 1, 0.97, "ratio")
    )
    assert sig(_series("height", PxValue(number=0), PxValue(number=120), 0, 120))
    assert not sig(_series("border-radius", PxValue(number=8), PxValue(number=9), 8, 9))
    near = _series("color", ColorValue(color="#111111"), ColorValue(color="#121212"), unit="color")
    far = _series("color", ColorValue(color="#111111"), ColorValue(color="#5252FF"), unit="color")
    assert not sig(near) and sig(far)
    shadow = _series(
        "box-shadow", ShadowValue(shadow=None), ShadowValue(shadow=None), unit="shadow"
    )
    assert sig(shadow)


# --------------------------------------------------------------------------------------------
# Joint fit — noise-free
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [60, 30])
@pytest.mark.parametrize("name", [c.name for c in CANDIDATES])
def test_noise_free_fit_recovers_curve_and_timing(fps: int, name: str) -> None:
    c = CANDIDATES_BY_NAME[name]
    t0, dur = 1.0 + 0.37 / fps, 0.3
    t, p = synth(c.bezier, fps=fps, t0=t0, dur=dur)
    fit = tm.fit_timing(t, p)
    assert fit is not None
    assert fit.easing.nearest_named == name
    assert fit.easing.family == c.family
    assert fit.rmse < 0.002
    assert abs(fit.t0_s - t0) < 0.002
    assert abs(fit.duration_s - dur) < 0.004
    assert ("overshoot" in fit.easing.flags) == (c.family == "overshoot")


@pytest.mark.parametrize("name", ["ease-in", "ease-out"])
@pytest.mark.parametrize("fps", [60, 30])
def test_joint_fit_corrects_threshold_bias(name: str, fps: int) -> None:
    """§6.7: ease-in looks late, ease-out looks short — the joint fit removes both biases."""
    c = CANDIDATES_BY_NAME[name]
    t0, dur = 1.0, 0.25
    t, p = synth(c.bezier, fps=fps, t0=t0, dur=dur)
    fit = tm.fit_timing(t, p)
    assert fit is not None
    if name == "ease-in":
        assert fit.t5_s - t0 > 1.0 / fps  # threshold onset is late by > 1 frame
    else:
        assert dur - (fit.t95_s - fit.t5_s) > 1.0 / fps  # threshold duration is short
    assert abs(fit.t0_s - t0) < 0.002 and abs(fit.duration_s - dur) < 0.004


def test_irregular_timestamps_vfr() -> None:
    rng = np.random.default_rng(5)
    base = np.arange(0.5, 2.2, 1 / 60)
    t = np.sort(base + rng.uniform(-0.004, 0.004, base.size))
    t = np.delete(t, rng.choice(t.size, 15, replace=False))  # dropped (duplicate) frames
    bez = CANDIDATES_BY_NAME["ease-out"].bezier
    p = truth(bez, (t - 1.0) / 0.28)
    fit = tm.fit_timing(t, p)
    assert fit is not None and fit.easing.family == "ease-out"
    assert abs(fit.t0_s - 1.0) < 0.003 and abs(fit.duration_s - 0.28) < 0.006


def test_fit_series_reverse_segment_and_none_cases() -> None:
    bez = CANDIDATES_BY_NAME["ease-in-out"].bezier
    t, p = synth(bez, fps=60, t0=2.0, dur=0.22, start=1.6, end=2.8)
    values = -8.0 + 8.0 * p  # reverse: translateY -8 -> 0
    s = PropertySeries(
        element_id="e1",
        property="translateY",
        segment="rev",
        times=t,
        values=values,
        quality=np.ones_like(t),
        v_start=-8.0,
        v_end=0.0,
        unit="px",
        from_value=PxValue(number=-8),
        to_value=PxValue(number=0),
    )
    fit = tm.fit_series(s)
    assert fit is not None and fit.easing.keyword == "ease-in-out"
    assert abs(fit.t0_s - 2.0) < 0.002 and abs(fit.duration_s - 0.22) < 0.004
    flat = PropertySeries(**{**{f: getattr(s, f) for f in s.__slots__}, "v_end": -8.0})
    assert tm.fit_series(flat) is None
    assert tm.fit_timing(t[:3], p[:3]) is None
    assert tm.fit_timing(t, np.zeros_like(t)) is None


def test_nan_samples_are_ignored() -> None:
    bez = CANDIDATES_BY_NAME["linear"].bezier
    t, p = synth(bez, fps=60, t0=1.0, dur=0.3)
    p[[5, 40, 41]] = np.nan
    fit = tm.fit_timing(t, p)
    assert fit is not None and fit.easing.keyword == "linear"


def test_free_bezier_used_only_when_clearly_better() -> None:
    off_catalog = (0.0, 0.6, 0.9, 0.4)  # rise, plateau, rise: far from every candidate
    t, p = synth(off_catalog, fps=60, t0=1.0, dur=0.5)
    fit = tm.fit_timing(t, p)
    assert fit is not None and fit.used_free
    assert fit.easing.keyword is None
    best_named = fit.candidates[0].rmse
    assert fit.rmse < best_named - 0.01  # Occam margin exceeded
    # A catalog curve stays named (Occam), even though the free fit could match it too.
    t, p = synth(CANDIDATES_BY_NAME["easeOutCubic"].bezier, fps=60, t0=1.0, dur=0.5)
    fit = tm.fit_timing(t, p)
    assert fit is not None and not fit.used_free and fit.easing.nearest_named == "easeOutCubic"
    # Too few in-motion samples (< free_min_samples): no free refine at all.
    t, p = synth(off_catalog, fps=30, t0=1.0, dur=0.15)
    fit = tm.fit_timing(t, p)
    assert fit is not None and not fit.used_free
    t, p = synth(off_catalog, fps=60, t0=1.0, dur=0.5)
    assert not tm.fit_timing(t, p, allow_free=False).used_free  # type: ignore[union-attr]


def test_family_separation_noise_free() -> None:
    """Distinct families fit each other clearly worse than the true family."""
    for name in ["linear", "ease-in", "ease-out", "ease-in-out", "backOut"]:
        t, p = synth(CANDIDATES_BY_NAME[name].bezier, fps=60, t0=1.0, dur=0.3)
        fit = tm.fit_timing(t, p)
        assert fit is not None
        assert fit.rmse_second_family - fit.rmse > 0.01, name


def test_overshoot_candidates_only_when_data_overshoots() -> None:
    t, p = synth(CANDIDATES_BY_NAME["ease-out"].bezier, fps=60, t0=1.0, dur=0.3)
    fit = tm.fit_timing(t, p)
    assert fit is not None and not fit.overshoot_detected
    assert all(c.family != "overshoot" for c in fit.candidates)


def test_segment_onset_settle_and_delay() -> None:
    fits = []
    for t0, dur in [(1.0, 0.28), (1.03, 0.32)]:
        t, p = synth(CANDIDATES_BY_NAME["ease-out"].bezier, fps=60, t0=t0, dur=dur)
        f = tm.fit_timing(t, p)
        assert f is not None
        fits.append(f)
    onset, settle = tm.segment_onset_settle(fits)
    assert onset == pytest.approx(1.0, abs=0.002)
    assert settle == pytest.approx(1.35, abs=0.005)
    assert tm.delay_s(fits[1].t0_s, onset) == pytest.approx(0.03, abs=0.003)
    assert tm.delay_s(0.9, 1.0) == 0.0
    with pytest.raises(ValueError):
        tm.segment_onset_settle([])


# --------------------------------------------------------------------------------------------
# Joint fit — σ = 0.02 noise (§11.1: t0/D within 1 sample for ease-in/out)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [60, 30])
@pytest.mark.parametrize("name", ["ease-in", "ease-out"])
def test_noisy_t0_duration_within_one_sample(fps: int, name: str) -> None:
    rng = np.random.default_rng(11)
    bez = CANDIDATES_BY_NAME[name].bezier
    dt = 1.0 / fps
    t0_err, d_err = [], []
    for _ in range(30):
        t0 = 1.0 + rng.uniform(0, dt)
        t, p = synth(bez, fps=fps, t0=t0, dur=0.3, sigma=0.02, rng=rng)
        fit = tm.fit_timing(t, p)
        assert fit is not None
        t0_err.append(abs(fit.t0_s - t0))
        d_err.append(abs(fit.duration_s - 0.3))
    assert np.median(t0_err) <= dt and np.median(d_err) <= dt
    assert np.percentile(t0_err, 80) <= dt


# --------------------------------------------------------------------------------------------
# DoD: family recovery for every Appendix B curve at σ = 0.02 (noise-only, no video)
# --------------------------------------------------------------------------------------------

#: Duration used by the DoD check. At σ = 0.02 the family rate is sample-count limited (PLAN R3);
#: see the module docstring of ``timing.py`` / Track M2 report for the full duration sweep.
DOD_DURATION_S = 0.4
DOD_TRIALS = 25


@pytest.mark.parametrize("fps", [60, 30])
def test_dod_family_recovery_sigma_002(fps: int) -> None:
    rng = np.random.default_rng(2026 + fps)
    rates: dict[str, float] = {}
    for c in CANDIDATES:
        ok = 0
        for _ in range(DOD_TRIALS):
            t0 = 1.0 + rng.uniform(0, 1.0 / fps)
            t, p = synth(c.bezier, fps=fps, t0=t0, dur=DOD_DURATION_S, sigma=0.02, rng=rng)
            fit = tm.fit_timing(t, p)
            assert fit is not None
            ok += families_match(fit.easing.family, c.family)
        rates[c.name] = ok / DOD_TRIALS
    pooled = float(np.mean(list(rates.values())))
    assert pooled >= 0.95, rates
    # Per-curve floor: `ease` vs `easeOutQuad` is the one pair the data cannot separate reliably.
    assert min(rates.values()) >= 0.6, rates
