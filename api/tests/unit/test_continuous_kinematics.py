"""Position-domain kinematic fits + aggregation (PLAN-continuous §4.5, §8.4 targets).

Also hosts the synthetic-profile helpers shared by the other ``test_continuous_*`` modules:
a velocity profile ``v(t)`` is integrated on a 0.1 ms grid and sampled at (optionally jittered)
frame times with position noise — numbers only, no frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from app.models.ir import Behavior, ExponentialFit, MeasuredNumber, RampFit
from app.models.measure import DisplacementSeries, PhaseCandidate, Rect
from app.pipeline.continuous import kinematics as kin
from app.pipeline.continuous.confidence import ir_confidence
from app.pipeline.easing import CANDIDATES_BY_NAME, solve_bezier

VelocityFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------------
# Shared helpers (imported by test_continuous_phases / test_continuous_assemble)
# --------------------------------------------------------------------------------------------


def ease(name: str, u: np.ndarray | float) -> np.ndarray:
    return solve_bezier(CANDIDATES_BY_NAME[name].bezier, np.clip(u, 0.0, 1.0))


def integrate(vel: VelocityFn, dur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine grid ``(t, v, x)`` with ``x`` = trapezoid integral of ``v`` (0.1 ms steps)."""
    tt = np.arange(0.0, dur + 0.05, 1e-4)
    vv = vel(tt)
    xx = np.concatenate([[0.0], np.cumsum((vv[1:] + vv[:-1]) * 0.5 * 1e-4)])
    return tt, vv, xx


def make_series(
    vel: VelocityFn,
    *,
    fps: float = 60.0,
    dur: float = 8.0,
    sigma: float = 0.05,
    fast_sigma: float | None = None,
    jitter_s: float = 0.0,
    seed: int = 0,
    region: Rect | None = None,
    fast_quality: float = 0.98,
) -> DisplacementSeries:
    """Sampled displacement series of ``vel`` (positions at true frame times; recorded times
    jittered by ``±jitter_s`` like a VFR recorder)."""
    rng = np.random.default_rng(seed)
    tt, vv, xx = integrate(vel, dur)
    n = int(dur * fps)
    t_true = np.arange(n) / fps
    t_rec = t_true + (rng.uniform(-jitter_s, jitter_s, n) if jitter_s else 0.0)
    t_rec[0] = 0.0
    t_rec = np.maximum.accumulate(t_rec + 1e-6 * np.arange(n))
    x = np.interp(t_true, tt, xx)
    v = np.interp(t_true, tt, vv)
    fs = sigma if fast_sigma is None else fast_sigma
    s = np.where(np.abs(v) > 1000, fs, sigma)
    x = x + rng.normal(0.0, 1.0, n) * s
    x -= x[0]
    q = np.where(np.abs(v) > 1000, fast_quality, 0.99)
    return DisplacementSeries(
        times=t_rec,
        pos=x,
        quality=q,
        axis="x",
        region=region or Rect(260, 300, 760, 200),
        region_confidence=0.7,
        dt_median=1.0 / fps,
    )


def make_grid_series(
    vel: VelocityFn,
    *,
    render_fps: float = 60.0,
    grid_hz: float = 75.0,
    dur: float = 8.0,
    sigma: float = 0.05,
    fast_sigma: float | None = None,
    seed: int = 0,
    fast_quality: float = 0.98,
) -> DisplacementSeries:
    """The motivating recording's clock (PLAN-continuous P2): the page renders at
    ``render_fps``; the recorder captures on a ``1 / grid_hz`` grid keeping 3 of every 4 slots
    (13.3 / 13.3 / 26.7 ms stamps at 75 Hz) and each capture shows the latest rendered frame
    while carrying its capture time. Captures of an already captured render frame are
    duplicates and are dropped, like the tracker does."""
    rng = np.random.default_rng(seed)
    tt, vv, xx = integrate(vel, dur)
    slots = []
    n = 0
    while True:
        slot = 4 * (n // 3) + n % 3
        if slot / grid_hz > dur:
            break
        slots.append(slot)
        n += 1
    stamps = np.asarray(slots, dtype=np.float64) / grid_hz
    render = np.floor(stamps * render_fps + 1e-9) / render_fps
    keep = np.concatenate([[True], np.diff(render) > 0])
    stamps, render = stamps[keep], render[keep]
    x = np.interp(render, tt, xx)
    v = np.interp(render, tt, vv)
    fs = sigma if fast_sigma is None else fast_sigma
    x = x + rng.normal(0.0, 1.0, len(x)) * np.where(np.abs(v) > 1000, fs, sigma)
    x -= x[0]
    q = np.where(np.abs(v) > 1000, fast_quality, 0.99)
    return DisplacementSeries(
        times=stamps - stamps[0],
        pos=x,
        quality=q,
        axis="x",
        region=Rect(260, 300, 760, 200),
        region_confidence=0.7,
        dt_median=float(np.median(np.diff(stamps))),
    )


def vel_c4(t: np.ndarray) -> np.ndarray:
    """C4-like: 40 px/s autoplay; press at 2.0 (instant stop) + drag left reaching −1100 px/s;
    release 2.45 (τ 200 ms); rest; resume at 4.45 (750 ms ease-in-out); drag right at 6.0,
    release 6.4 at +900 px/s (τ 200 ms)."""
    v = np.zeros_like(t)
    v[t < 2.0] = 40.0
    m = (t >= 2.0) & (t < 2.45)
    v[m] = -1100.0 * np.sin(np.pi / 2 * np.clip((t[m] - 2.0) / 0.135, 0, 1))
    m = (t >= 2.45) & (t < 4.45)
    v[m] = -1100.0 * np.exp(-(t[m] - 2.45) / 0.2)
    m = (t >= 4.45) & (t < 5.2)
    v[m] = 40.0 * ease("ease-in-out", (t[m] - 4.45) / 0.75)
    v[(t >= 5.2) & (t < 6.0)] = 40.0
    m = (t >= 6.0) & (t < 6.4)
    v[m] = 900.0 * np.clip((t[m] - 6.0) / 0.08, 0, 1)
    m = t >= 6.4
    v[m] = 900.0 * np.exp(-(t[m] - 6.4) / 0.2)
    return v


# §0.1-derived profile (numbers only): autoplay +39, decelerate cut by a drag, 4 drags with 3
# inertia phases (τ 200 / 210 / 190 ms, the last two re-grabbed), abrupt stop, rest, resume.
def vel_section01(tt: np.ndarray) -> np.ndarray:
    v = np.zeros_like(tt)

    def seg(a: float, b: float) -> np.ndarray:
        return (tt >= a) & (tt < b)

    def drag(a: float, b: float, v_in: float, v_mid: float, v_rel: float) -> None:
        m = seg(a, b)
        s = tt[m] - a
        base = v_mid + (v_rel - v_mid) * np.clip((s - 0.05) / (b - a - 0.05), 0, 1)
        v[m] = v_in + (base - v_in) * np.clip(s / 0.05, 0, 1)

    def inertia(a: float, b: float, v0: float, tau: float) -> None:
        m = seg(a, b)
        v[m] = v0 * np.exp(-(tt[m] - a) / tau)

    v[seg(0, 1.5)] = 39.0
    m = seg(1.5, 2.0)
    v[m] = 39.0 * (1 - (tt[m] - 1.5) / 0.7)
    drag(2.0, 2.5, 39 * (1 - 0.5 / 0.7), -1500, -1400)
    inertia(2.5, 3.75, -1400, 0.2)
    drag(3.75, 5.0, -1400 * math.exp(-1.25 / 0.2), -850, -750)
    inertia(5.0, 5.5, -750, 0.21)
    drag(5.5, 6.3, -750 * math.exp(-0.5 / 0.21), 990, 900)
    inertia(6.3, 7.0, 900, 0.19)
    drag(7.0, 7.6, 900 * math.exp(-0.7 / 0.19), -1720, -1650)
    v[seg(7.6, 7.85)] = 0.0
    m = seg(7.85, 8.6)
    v[m] = 39.0 * ease("ease-in-out", (tt[m] - 7.85) / 0.75)
    v[seg(8.6, 9.4)] = 39.0
    return v


def section01_series(seed: int = 0) -> DisplacementSeries:
    """§0.1-shaped VFR series: 55.7 fps, ±2 ms timestamp jitter, noisier fast frames."""
    return make_series(
        vel_section01, fps=55.7, dur=9.3, sigma=0.1, fast_sigma=0.3, jitter_s=0.002,
        seed=seed, region=Rect(260, 300, 760, 200), fast_quality=0.62,
    )  # fmt: skip


# --------------------------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fps, tol", [(60.0, 0.15), (30.0, 0.25)])
def test_exponential_recovers_tau_and_v0(fps: float, tol: float) -> None:
    """τ and v0 within §8.4 (≤ 15 % at 60 fps, ≤ 25 % at 30 fps) over a noisy Monte Carlo."""
    rng = np.random.default_rng(7)
    worst_tau = worst_v0 = 0.0
    for _ in range(25):
        tau = float(rng.uniform(0.12, 0.6))
        v0 = float(rng.choice([-1, 1]) * rng.uniform(300, 2500))
        v_inf = float(rng.choice([0.0, 40.0]))
        t = np.arange(0, 3.5 * tau + 0.1, 1 / fps)
        x = kin.exp_position(t, 0.0, v0, tau, v_inf) + rng.normal(0, 0.1, len(t))
        f = kin.fit_exponential(t, x, 0.0, v_inf)
        assert f is not None and f.model == "exponential"
        worst_tau = max(worst_tau, abs(f.params["tau"] / tau - 1))
        worst_v0 = max(worst_v0, abs(f.params["v0"] / v0 - 1))
    assert worst_tau <= tol
    assert worst_v0 <= tol


def test_exponential_standard_error_is_small_for_clean_data() -> None:
    t = np.arange(0, 1.0, 1 / 60)
    x = kin.exp_position(t, 5.0, -900.0, 0.2, 0.0) + np.random.default_rng(1).normal(
        0, 0.05, len(t)
    )
    f = kin.fit_exponential(t, x, 0.0, 0.0)
    assert f is not None
    assert f.se["tau"] / f.params["tau"] < 0.02
    assert not f.at_bound


def test_exponential_with_momentum_stop_threshold() -> None:
    """Momentum libraries stop the decay below a minimum speed: the position freezes. The
    cutoff variant recovers τ / v0 and the stop; the plain decay would misfit by v_stop · τ."""
    tau, v0, v_stop = 0.2, -1100.0, 10.0
    s_stop = tau * math.log(abs(v0) / v_stop)
    t = np.arange(0, 2.0, 1 / 60)
    x = kin.exp_position(t, 0.0, v0, tau, 0.0, s_stop)
    x = x + np.random.default_rng(4).normal(0, 0.03, len(t))
    f = kin.fit_exponential(t, x, 0.0, 0.0, cutoff=True)
    assert f is not None and math.isfinite(f.params["s_stop"])
    assert f.params["tau"] == pytest.approx(tau, rel=0.03)
    assert f.params["v0"] == pytest.approx(v0, rel=0.03)
    assert f.params["s_stop"] == pytest.approx(s_stop, abs=0.03)
    assert f.params["v_stop"] == pytest.approx(v_stop, abs=4.0)
    plain = kin.fit_exponential(t, x, 0.0, 0.0)
    assert plain is not None and plain.sse > 10 * f.sse
    # without a stop the cutoff is not used (BIC)
    x2 = kin.exp_position(t, 0.0, v0, tau, 0.0) + np.random.default_rng(5).normal(0, 0.03, len(t))
    f2 = kin.fit_exponential(t, x2, 0.0, 0.0, cutoff=True)
    assert f2 is not None and (math.isinf(f2.params["s_stop"]) or f2.params["v_stop"] < 1.0)


def test_constant_fit_speed() -> None:
    t = np.arange(0, 1.5, 1 / 60)
    x = 39.0 * t + np.random.default_rng(2).normal(0, 0.1, len(t))
    f = kin.fit_constant(t, x)
    assert f.params["v"] == pytest.approx(39.0, rel=0.02)
    assert f.se["v"] < 0.5


@pytest.mark.parametrize(
    "fps, kind, v_a, v_b, name, dur",
    [
        (60.0, "decel", 40.0, 0.0, "ease-out", 0.4),
        (30.0, "decel", 40.0, 0.0, "ease-out", 0.4),
        (60.0, "resume", 0.0, 39.0, "ease-in-out", 0.75),
        (30.0, "resume", 0.0, 39.0, "ease-in-out", 0.75),
        (60.0, "resume", 0.0, 40.0, "ease-in", 0.6),
    ],
)
def test_ramp_recovers_t0_and_duration(
    fps: float, kind: str, v_a: float, v_b: float, name: str, dur: float
) -> None:
    """Soft boundary ≤ max(60 ms, 15 %) (30 fps: max(90 ms, 20 %)); duration ≤ max(60 ms, 20 %)
    (30 fps: max(90 ms, 25 %))."""
    del kind
    t0_tol = max(0.06, 0.15 * dur) if fps >= 60 else max(0.09, 0.20 * dur)
    d_tol = max(0.06, 0.20 * dur) if fps >= 60 else max(0.09, 0.25 * dur)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        t = np.arange(0, 2.5, 1 / fps)
        bez = CANDIDATES_BY_NAME[name].bezier
        x = kin.ramp_position(t, 0.0, 1.0, dur, v_a, v_b, bez) + rng.normal(0, 0.05, len(t))
        f = kin.fit_ramp(t, x, v_a, v_b, t0_range=(0.5, 1.6), dur_range=(2 / fps, 2.0),
                         frame_dt=1 / fps)  # fmt: skip
        assert f is not None
        assert abs(f.params["t0"] - 1.0) <= t0_tol, f.params
        assert abs(f.params["duration"] - dur) <= d_tol, f.params


def test_tween_recovers_snap() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(0, 0.7, 1 / 60)
    x = 5.0 + 80.0 * ease("ease-out", t / 0.3) + rng.normal(0, 0.1, len(t))
    f = kin.fit_tween(t, x, 0.0, dur_min=0.04, dur_max=1.0)
    assert f is not None
    assert f.params["duration"] == pytest.approx(0.3, abs=0.06)
    assert f.params["distance"] == pytest.approx(80.0, abs=2.0)
    assert f.curve is not None and f.curve.family == "ease-out"
    assert kin.fit_easing(f) is not None


def test_tween_curves_exclude_constant_velocity_shapes() -> None:
    # a linear "tween" is a constant-velocity drag followed by an abrupt stop
    assert "linear" not in kin.TWEEN_CURVES
    assert "ease-in" not in kin.TWEEN_CURVES


def test_ramp_integral_extends_linearly() -> None:
    bez = CANDIDATES_BY_NAME["linear"].bezier
    u = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(kin.ramp_integral(bez, u), [0.0, 0.0, 0.125, 0.5, 1.5], atol=1e-3)


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------


def _mn(v: float, c: float) -> MeasuredNumber:
    return MeasuredNumber(value=v, confidence=ir_confidence(c))


def test_weighted_median_and_agreement() -> None:
    assert kin.weighted_median([1, 2, 100], [1, 1, 0.01]) == 2
    assert kin.agreement_factor([200]) == 1.0
    assert kin.agreement_factor([200, 210, 190]) > 0.85
    assert kin.agreement_factor([100, 400]) == pytest.approx(0.6)


def test_aggregate_builds_valid_behavior() -> None:
    def inertia(tau: float, v0: float) -> PhaseCandidate:
        return PhaseCandidate(
            kind="inertia", start_s=0, end_s=1, v_start=v0, v_end=0, v_peak=v0, displacement=0,
            fit=ExponentialFit(tau_ms=_mn(tau, 0.8), v0_px_s=_mn(v0, 0.7), v_inf_px_s=0.0),
            confidence=0.8,
        )  # fmt: skip

    ramp = RampFit(from_px_s=0.0, to_px_s=39.0, duration_ms=_mn(750, 0.7),
                   easing=kin.fit_easing(_ramp_fit()))  # type: ignore[arg-type]  # fmt: skip
    phases = [
        PhaseCandidate("drag", 0, 1, 0, -1400, -1500, -500, confidence=0.8),
        inertia(200, -1400),
        PhaseCandidate("drag", 1, 2, 0, -750, -850, -500, confidence=0.8),
        inertia(210, -750),
        PhaseCandidate("stop", 2, 2.1, -1650, 0, -1650, -20, confidence=0.45),
        PhaseCandidate("resume", 2.2, 3, 0, 39, 39, 15, fit=ramp, confidence=0.7),
    ]
    ctx = kin.BehaviorContext(
        autoplay_velocity=39.0, pitch_px=None, pitch_confidence=0.0, snap_grid=False,
        snap_grid_confidence=0.0, resume_delays=[(0.1, 0.25, 0.5)],
    )  # fmt: skip
    b = kin.aggregate(phases, ctx)
    assert isinstance(b, Behavior)
    assert b.drag is not None and b.drag.count == 2 and b.drag.peak_speed_px_s.value == 1500
    assert b.inertia is not None and b.inertia.instances == 2 and b.inertia.model == "exponential"
    assert b.inertia.tau_ms is not None and 200 <= b.inertia.tau_ms.value <= 210
    assert b.inertia.release_speed_min_px_s == 750 and b.inertia.release_speed_max_px_s == 1400
    assert b.snap is not None and b.snap.kind == "abrupt_ambiguous" and b.snap.step_px is None
    assert b.resume is not None and b.resume.delay_after_rest_ms.value == 100
    assert b.resume.direction_preserved
    assert b.pause is None


def _ramp_fit() -> kin.Fit:
    t = np.arange(0, 2.0, 1 / 60)
    bez = CANDIDATES_BY_NAME["ease-in-out"].bezier
    x = kin.ramp_position(t, 0.0, 0.5, 0.75, 0.0, 39.0, bez)
    f = kin.fit_ramp(t, x, 0.0, 39.0, t0_range=(0.2, 0.8), dur_range=(0.04, 2.0), frame_dt=1 / 60)
    assert f is not None
    return f
