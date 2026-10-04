"""Phase segmentation on synthetic displacement profiles (PLAN-continuous §4.5, §8.3, §8.4).

Profiles are numbers only (velocity / position functions integrated and sampled with noise);
no frames. Targets: §8.4 (60 fps; 30 fps / VFR relaxed values in parentheses there).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import ConstantFit, ExponentialFit, RampFit
from app.models.measure import CursorSample, DisplacementSeries, PhaseCandidate, Rect
from app.pipeline.continuous import phases as ph
from tests.unit.test_continuous_kinematics import (
    ease,
    make_grid_series,
    make_series,
    section01_series,
    vel_c4,
    vel_section01,
)


def kinds(phases: list[PhaseCandidate], *, drop: tuple[str, ...] = ()) -> list[str]:
    """Phase kinds with same-kind neighbours merged and ``drop`` kinds removed."""
    out: list[str] = []
    for p in phases:
        if p.kind in drop:
            continue
        if not out or out[-1] != p.kind:
            out.append(p.kind)
    return out


def first(phases: list[PhaseCandidate], kind: str, after: float = -1.0) -> PhaseCandidate:
    return next(p for p in phases if p.kind == kind and p.start_s > after)


def contiguous(phases: list[PhaseCandidate], series: DisplacementSeries) -> None:
    assert phases[0].start_s == pytest.approx(series.times[0])
    assert phases[-1].end_s == pytest.approx(series.times[-1])
    for a, b in zip(phases, phases[1:], strict=False):
        assert a.end_s == pytest.approx(b.start_s)
        assert b.end_s > b.start_s


# --------------------------------------------------------------------------------------------
# §0.1-derived profile (the motivating recording, numbers only)
# --------------------------------------------------------------------------------------------

#: §8.3: autoplay → decelerate(interrupted) → drag → inertia → drag → inertia → drag → inertia →
#: drag → stop|snap → resume → autoplay. ``paused`` (rest) phases are allowed in between: the
#: grammar needs one between stop and resume, and a decay may come to rest before a re-grab.
EXPECTED_01 = [
    "autoplay", "decelerate", "drag", "inertia", "drag", "inertia", "drag", "inertia", "drag",
    "stop", "resume", "autoplay",
]  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_section01_profile_phase_sequence(seed: int) -> None:
    s = section01_series(seed)
    seg = ph.segment(s, is_vfr=True)
    contiguous(seg.phases, s)
    assert kinds(seg.phases, drop=("paused",)) == EXPECTED_01
    assert seg.autoplay_velocity == pytest.approx(39.0, rel=0.03)

    dec = first(seg.phases, "decelerate")
    assert dec.interrupted and isinstance(dec.fit, RampFit)
    inertia = [p for p in seg.phases if p.kind == "inertia"]
    truth = [(2.5, -1400.0, 200.0), (5.0, -750.0, 210.0), (6.3, 900.0, 190.0)]
    for p, (t_rel, v0, tau) in zip(inertia, truth, strict=True):
        assert isinstance(p.fit, ExponentialFit)
        # VFR + degraded fast frames: relaxed §8.4 targets (τ ≤ 25 %, v0 ≤ 25 %, 2 frames)
        assert p.fit.tau_ms.value == pytest.approx(tau, rel=0.25)
        assert p.fit.v0_px_s.value == pytest.approx(v0, rel=0.25)
        assert abs(p.start_s - t_rel) <= 2 * 0.018 + 0.002
    assert inertia[1].interrupted and inertia[2].interrupted
    stop = first(seg.phases, "stop")
    assert abs(stop.end_s - 7.6) <= 0.05
    resume = first(seg.phases, "resume")
    assert isinstance(resume.fit, RampFit)
    assert abs(resume.start_s - 7.85) <= max(0.09, 0.2 * 0.75)
    assert abs(resume.fit.duration_ms.value - 750) <= max(90, 0.25 * 750)

    b = seg.behavior
    assert b.drag is not None and b.drag.count == 4 and b.drag.follows_pointer is None
    assert b.inertia is not None and b.inertia.instances == 3 and b.inertia.model == "exponential"
    assert b.inertia.tau_ms is not None and b.inertia.tau_ms.value == pytest.approx(200, rel=0.15)
    assert b.snap is not None and b.snap.kind == "abrupt_ambiguous"
    assert b.pause is not None and b.pause.on == "unknown"
    assert b.pause.on_confidence.value <= 0.4  # trigger without cursor is capped
    assert b.resume is not None and b.resume.direction_preserved


# --------------------------------------------------------------------------------------------
# Capture-grid timestamps (the motivating .mov: 13.3 ms stamp grid, 60 Hz page render)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c4_profile_on_capture_grid_within_vfr_targets(seed: int) -> None:
    """Positions lag their timestamps by 0..1 render interval, irregularly: §8.4 VFR targets
    still hold, and no phase is shorter than two frames (no spurious 1-frame drags / rests)."""
    s = make_grid_series(vel_c4, dur=8.0, sigma=0.05, seed=seed)
    d = np.diff(s.times) * 1000
    assert set(np.round(d, 2)) <= {13.33, 26.67, 40.0}  # the grid really is irregular
    seg = ph.segment(s, is_vfr=True)
    contiguous(seg.phases, s)
    assert kinds(seg.phases) == [
        "autoplay", "drag", "inertia", "paused", "resume", "autoplay", "drag", "inertia",
        "paused",
    ]  # fmt: skip
    frame = float(np.median(np.diff(s.times)))
    for p in seg.phases[:-1]:
        assert p.end_s - p.start_s >= 2 * frame - 1e-6, p
    for p, (t_rel, v0) in zip(
        [p for p in seg.phases if p.kind == "inertia"], [(2.45, -1100.0), (6.4, 900.0)],
        strict=True,
    ):  # fmt: skip
        assert isinstance(p.fit, ExponentialFit)
        assert p.fit.tau_ms.value == pytest.approx(200, rel=0.25)
        assert p.fit.v0_px_s.value == pytest.approx(v0, rel=0.25)
        assert abs(p.start_s - t_rel) <= 0.067
    drags = [p for p in seg.phases if p.kind == "drag"]
    assert abs(drags[0].v_peak) == pytest.approx(1100, rel=0.15)  # not inflated by the jitter
    b = seg.behavior
    assert b.drag is not None and b.drag.peak_speed_px_s.value == pytest.approx(1100, rel=0.15)
    assert b.resume is not None and b.resume.ramp_ms.value == pytest.approx(750, abs=187.5)
    assert b.resume.delay_after_rest_ms.value == pytest.approx(4450 - 3390, abs=80)


def test_section01_profile_on_capture_grid_is_clean() -> None:
    """§0.1-shaped profile on the capture grid: the §8.3 sequence, nothing shorter than two
    frames, peak speed close to the true ≈1720 px/s."""
    s = make_grid_series(vel_section01, dur=9.3, sigma=0.1, fast_sigma=0.3, seed=4,
                         fast_quality=0.62)  # fmt: skip
    seg = ph.segment(s, is_vfr=True)
    contiguous(seg.phases, s)
    assert kinds(seg.phases, drop=("paused",)) == EXPECTED_01
    frame = float(np.median(np.diff(s.times)))
    # (an abrupt "stop" is an event of ≤ 2 frames by definition)
    assert all(
        p.end_s - p.start_s >= 2 * frame - 1e-6 for p in seg.phases[1:-1] if p.kind != "stop"
    )
    b = seg.behavior
    assert b.drag is not None
    assert b.drag.peak_speed_px_s.value == pytest.approx(1720, rel=0.15)
    assert b.inertia is not None and b.inertia.tau_ms is not None
    assert b.inertia.tau_ms.value == pytest.approx(200, rel=0.25)


# --------------------------------------------------------------------------------------------
# C4-like carousel (CFR): drag + inertia, rest, resume, second drag
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [60.0, 30.0])
def test_c4_profile_values_within_targets(fps: float) -> None:
    hi = fps >= 60
    s = make_series(vel_c4, fps=fps, dur=8.0, sigma=0.05, seed=3)
    seg = ph.segment(s)
    contiguous(seg.phases, s)
    assert kinds(seg.phases) == [
        "autoplay", "drag", "inertia", "paused", "resume", "autoplay", "drag", "inertia",
        "paused",
    ]  # fmt: skip
    frame = 1.0 / fps
    assert seg.autoplay_velocity == pytest.approx(40.0, rel=0.02 if hi else 0.03)
    tol = 0.15 if hi else 0.25
    for p, (t_rel, v0) in zip(
        [p for p in seg.phases if p.kind == "inertia"], [(2.45, -1100.0), (6.4, 900.0)],
        strict=True,
    ):  # fmt: skip
        assert isinstance(p.fit, ExponentialFit)
        assert p.fit.tau_ms.value == pytest.approx(200, rel=tol)
        assert p.fit.v0_px_s.value == pytest.approx(v0, rel=tol)
        assert abs(p.start_s - t_rel) <= 2 * frame  # sharp boundary: release
    drags = [p for p in seg.phases if p.kind == "drag"]
    assert abs(drags[0].start_s - 2.0) <= 2 * frame and abs(drags[1].start_s - 6.0) <= 2 * frame
    assert abs(drags[0].v_peak) == pytest.approx(1100, rel=0.10 if hi else 0.15)
    res = first(seg.phases, "resume")
    assert isinstance(res.fit, RampFit)
    assert abs(res.start_s - 4.45) <= (max(0.06, 0.15 * 0.75) if hi else max(0.09, 0.2 * 0.75))
    assert abs(res.fit.duration_ms.value - 750) <= (max(60, 150) if hi else max(90, 187.5))
    assert res.fit.easing.family in ("ease-in-out", "ease")
    b = seg.behavior
    assert b.resume is not None
    # P2 inertia-end definition (phases._inertia_end): a decay fading to rest ends at 10 px/s,
    # τ ln(1100/10) ≈ 0.94 s after release (same convention as the synthetic truth)
    assert b.resume.delay_after_rest_ms.value == pytest.approx(4450 - 3390, abs=50 if hi else 80)
    inertia = [p for p in seg.phases if p.kind == "inertia"]
    assert inertia[0].fit is not None and inertia[0].fit.stop_px_s == pytest.approx(10.0)
    assert b.inertia is not None and b.inertia.stop_px_s is not None
    assert b.inertia.stop_px_s.value == pytest.approx(10.0)
    assert b.snap is None  # no grid / abrupt stop
    # press stops autoplay at once (autoplay -> drag): a pause without slowdown; no cursor
    assert b.pause is not None and b.pause.on == "unknown" and b.pause.decel_ms is None
    assert b.pause.stops_completely


def test_c4_profile_with_vfr_jitter() -> None:
    s = make_series(vel_c4, fps=55.7, dur=8.0, sigma=0.08, jitter_s=0.002, seed=5)
    seg = ph.segment(s, is_vfr=True)
    assert kinds(seg.phases, drop=("paused",)) == [
        "autoplay", "drag", "inertia", "resume", "autoplay", "drag", "inertia",
    ]  # fmt: skip
    taus = [p.fit.tau_ms.value for p in seg.phases if isinstance(p.fit, ExponentialFit)]
    assert all(abs(t - 200) <= 50 for t in taus)


# --------------------------------------------------------------------------------------------
# Marquee, hover pause, snap, no autoplay
# --------------------------------------------------------------------------------------------


def test_marquee_autoplay_only() -> None:
    s = make_series(lambda t: np.full_like(t, -60.0), fps=60.0, dur=8.0, sigma=0.05)
    seg = ph.segment(s)
    assert [p.kind for p in seg.phases] == ["autoplay"]
    assert seg.autoplay_velocity == pytest.approx(-60.0, rel=0.02)
    fit = seg.phases[0].fit
    assert isinstance(fit, ConstantFit) and fit.velocity_px_s.confidence.band == "high"
    b = seg.behavior
    assert (b.pause, b.drag, b.inertia, b.snap, b.resume) == (None, None, None, None, None)


def _vel_hover(t: np.ndarray) -> np.ndarray:
    v = np.full_like(t, 40.0)
    m = (t >= 2.0) & (t < 2.4)
    v[m] = 40.0 * (1 - ease("ease-out", (t[m] - 2.0) / 0.4))
    v[(t >= 2.4) & (t < 4.3)] = 0.0
    m = (t >= 4.3) & (t < 4.9)
    v[m] = 40.0 * ease("ease-in", (t[m] - 4.3) / 0.6)
    return v


def _hover_cursor(region: Rect) -> list[CursorSample]:
    out: list[CursorSample] = []
    for t in np.arange(0.0, 8.0, 1 / 30):
        inside = 1.95 <= t <= 4.0
        x = region.x + region.w / 2 if inside else region.x - 120
        y = region.y + region.h / 2
        out.append(CursorSample(float(t), float(x), float(y), "stationary" if inside else "moving"))
    return out


def test_hover_pause_with_cursor() -> None:
    s = make_series(_vel_hover, fps=60.0, dur=8.0, sigma=0.05, seed=4)
    seg = ph.segment(s, cursor=_hover_cursor(s.region))
    assert kinds(seg.phases) == ["autoplay", "decelerate", "paused", "resume", "autoplay"]
    dec = first(seg.phases, "decelerate")
    assert isinstance(dec.fit, RampFit) and not dec.interrupted
    assert abs(dec.start_s - 2.0) <= 0.06
    assert abs(dec.fit.duration_ms.value - 400) <= max(60, 80)
    assert dec.evidence == "velocity+cursor"
    res = first(seg.phases, "resume")
    assert abs(res.start_s - 4.3) <= max(0.06, 0.15 * 0.6)
    assert isinstance(res.fit, RampFit) and abs(res.fit.duration_ms.value - 600) <= 120
    b = seg.behavior
    assert b.pause is not None and b.pause.on == "hover" and b.pause.stops_completely
    assert b.pause.on_confidence.value == pytest.approx(0.85)
    assert b.drag is None and b.snap is None


def test_hover_pause_without_cursor_is_unknown() -> None:
    s = make_series(_vel_hover, fps=60.0, dur=8.0, sigma=0.05, seed=4)
    b = ph.segment(s).behavior
    assert b.pause is not None and b.pause.on == "unknown"
    assert b.pause.on_confidence.value <= 0.4


PITCH = 216.0


def _pos_snap(t: np.ndarray) -> np.ndarray:
    """Autoplay 40 px/s → press 2.0 → drag −700 px/s → release 2.4 → snap 300 ms ease-out to
    the grid (multiples of PITCH) → rest → drag +600 px/s at 4.0 → release 4.35 → snap → rest."""
    x = np.where(t < 2.0, 40.0 * t, 80.0)

    def drag(t0: float, t1: float, x0: float, v: float) -> float:
        m = (t >= t0) & (t < t1)
        ramp = np.clip((t[m] - t0) / 0.06, 0, 1)
        # velocity ramps up over 60 ms then holds
        s = np.where(t[m] - t0 < 0.06, 0.5 * ramp * (t[m] - t0), (t[m] - t0) - 0.03)
        x[m] = x0 + v * s
        return x0 + v * ((t1 - t0) - 0.03)

    def snap(t0: float, x0: float, target: float, t_end: float) -> None:
        m = (t >= t0) & (t < t_end)
        x[m] = x0 + (target - x0) * ease("ease-out", (t[m] - t0) / 0.3)

    x1 = drag(2.0, 2.4, 80.0, -700.0)
    g1 = round(x1 / PITCH) * PITCH
    snap(2.4, x1, g1, 4.0)
    x2 = drag(4.0, 4.35, g1, 600.0)
    g2 = round(x2 / PITCH) * PITCH
    snap(4.35, x2, g2, 99.0)
    return x


def test_snap_to_grid() -> None:
    rng = np.random.default_rng(9)
    t = np.arange(0, 6.5, 1 / 60)
    x = _pos_snap(t) + rng.normal(0, 0.05, len(t))
    s = DisplacementSeries(
        times=t, pos=x - x[0], quality=np.full(len(t), 0.99), axis="x",
        region=Rect(260, 300, 760, 200), region_confidence=0.7, dt_median=1 / 60,
    )  # fmt: skip
    seg = ph.segment(s, pitch_px=PITCH, pitch_confidence=0.8)
    assert kinds(seg.phases) == ["autoplay", "drag", "snap", "paused", "drag", "snap", "paused"]
    assert seg.snap_resultant is not None and seg.snap_resultant >= 0.9
    b = seg.behavior
    assert b.snap is not None and b.snap.kind == "grid"
    assert b.snap.step_px is not None and b.snap.step_px.value == pytest.approx(PITCH, abs=2)
    assert b.snap.duration_ms is not None and abs(b.snap.duration_ms.value - 300) <= 60
    assert b.inertia is None


def test_no_false_snap_without_grid_evidence() -> None:
    s = make_series(vel_c4, fps=60.0, dur=8.0, sigma=0.05, seed=3)
    seg = ph.segment(s, pitch_px=PITCH, pitch_confidence=0.8)
    assert "snap" not in kinds(seg.phases)


def test_drag_from_rest_has_no_autoplay() -> None:
    def vel(t: np.ndarray) -> np.ndarray:
        v = np.zeros_like(t)
        m = (t >= 1.0) & (t < 1.5)
        v[m] = -900.0 * np.clip((t[m] - 1.0) / 0.08, 0, 1)
        m = t >= 1.5
        v[m] = -900.0 * np.exp(-(t[m] - 1.5) / 0.25)
        return v

    s = make_series(vel, fps=60.0, dur=4.0, sigma=0.05)
    seg = ph.segment(s)
    assert seg.autoplay_velocity is None
    assert kinds(seg.phases) == ["paused", "drag", "inertia", "paused"]
    inertia = first(seg.phases, "inertia")
    assert isinstance(inertia.fit, ExponentialFit)
    assert inertia.fit.tau_ms.value == pytest.approx(250, rel=0.15)


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------


def test_coarse_labels_and_absorb_short() -> None:
    v = np.array([40, 41, 39, 120, 40, 40, 40, 2, 1, 0, 300, 900, 700])
    labels = ph.coarse_labels(v, 40.0, 4.8, 6.0)
    assert labels == ["A", "A", "A", "M", "A", "A", "A", "Z", "Z", "Z", "M", "M", "M"]
    t = np.arange(len(v)) / 10.0
    runs = ph.absorb_short(ph._runs(labels), t, 3)
    assert [r[0] for r in runs] == ["A", "Z", "M"]  # the 1-sample blip is absorbed


def test_noise_sigma_scales_with_speed() -> None:
    s = make_series(vel_c4, fps=60.0, dur=8.0, sigma=0.05, fast_sigma=0.5, seed=1)
    from app.pipeline.continuous.displacement import smoothed_velocity

    v = smoothed_velocity(s.times, s.pos)
    sig = ph.noise_sigma(s.times, s.pos, v, s.quality, is_vfr=False)
    slow = np.abs(v) < 60
    fast = np.abs(v) > 1000
    assert np.median(sig[slow]) == pytest.approx(0.05, rel=0.6)
    assert np.median(sig[fast]) > 2 * np.median(sig[slow])


def test_tracking_degraded_warning_and_confidence() -> None:
    s = section01_series(0)
    q = s.quality.copy()
    v = np.gradient(s.pos, s.times)
    q[np.abs(v) > 1000] = 0.3  # most fast frames below quality
    s.quality = q
    seg = ph.segment(s, is_vfr=True)
    assert any(w.code == "tracking_degraded" for w in seg.warnings)
    degraded = [p for p in seg.phases if ph.NOTE_DEGRADED in p.notes]
    assert degraded and all(p.confidence <= 0.7 for p in degraded)


# --------------------------------------------------------------------------------------------
# P2b: slow autoplay on the capture grid, momentum that blends back into autoplay (C13)
# --------------------------------------------------------------------------------------------


def vel_blend(t: np.ndarray) -> np.ndarray:
    """35 px/s autoplay; press at 2.0 + fling left to −1400 px/s; release 2.4; the momentum
    decays towards +35 px/s (τ 260 ms), crossing zero on the way, and autoplay goes on."""
    v = np.full_like(t, 35.0)
    m = (t >= 2.0) & (t < 2.4)
    v[m] = -1400.0 * np.sin(np.pi / 2 * np.clip((t[m] - 2.0) / 0.15, 0, 1))
    m = t >= 2.4
    v[m] = 35.0 + (-1400.0 - 35.0) * np.exp(-(t[m] - 2.4) / 0.26)
    return v


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_momentum_blending_into_autoplay_on_capture_grid(seed: int) -> None:
    s = make_grid_series(vel_blend, dur=6.0, sigma=0.05, seed=seed)
    seg = ph.segment(s, is_vfr=True)
    contiguous(seg.phases, s)
    assert kinds(seg.phases) == ["autoplay", "drag", "inertia", "autoplay"]
    drag = first(seg.phases, "drag")
    assert abs(drag.start_s - 2.0) <= 0.067  # the 75/60 Hz velocity wobble is not a slowdown
    inertia = first(seg.phases, "inertia")
    assert isinstance(inertia.fit, ExponentialFit)
    assert inertia.fit.v_inf_px_s == pytest.approx(35.0, rel=0.05)
    assert inertia.fit.tau_ms.value == pytest.approx(260, rel=0.25)
    assert ph.NOTE_BLEND in inertia.notes
    assert seg.behavior.resume is None


def test_lead_velocity_noise_widens_the_autoplay_band() -> None:
    t = np.arange(60) / 60.0
    v = 35.0 + np.random.default_rng(0).normal(0.0, 3.0, len(t))  # capture-grid wobble
    assert ph.lead_velocity_noise(t, v, 35.0) == pytest.approx(3.0, rel=0.35)
    assert ph.lead_velocity_noise(t, v, None) == 0.0


def test_absorb_short_takes_the_shortest_run_first() -> None:
    """``A … M(1) A(0.2 s) M …``: the one-sample blip joins the autoplay before the short
    autoplay run is judged (it used to be judged next to the not-yet-merged blip and lost)."""
    t = np.arange(200) / 75.0
    runs = [["A", 0, 130], ["M", 130, 131], ["A", 131, 145], ["M", 145, 200]]
    assert ph.absorb_short(runs, t, 3) == [["A", 0, 145], ["M", 145, 200]]


# --------------------------------------------------------------------------------------------
# Acceptance run (iteration 1): drags that end with the pointer stopped
# --------------------------------------------------------------------------------------------


def _exp(start: float, end: float, tau_ms: float, v0: float, v_inf: float = 35.0) -> ph._Ph:
    from app.models.ir import MeasuredNumber  # noqa: PLC0415

    conf = {"value": 0.9, "band": "high"}
    fit = ExponentialFit(
        tau_ms=MeasuredNumber(value=tau_ms, confidence=conf),
        v0_px_s=MeasuredNumber(value=v0, confidence=conf),
        v_inf_px_s=v_inf,
    )
    return ph._Ph("inertia", start, end, fit=fit, v_start=v0, v_end=v_inf)


def _glide(start: float, end: float, v0: float) -> ph._Ph:
    from app.models.ir import Easing, MeasuredNumber, TweenFit  # noqa: PLC0415

    conf = {"value": 0.8, "band": "high"}
    fit = TweenFit(
        distance_px=MeasuredNumber(value=v0 * (end - start) / 2, confidence=conf),
        duration_ms=MeasuredNumber(value=(end - start) * 1000, confidence=conf),
        easing=Easing(cubic_bezier=(0.25, 0.46, 0.45, 0.94), nearest_named="easeOutQuad",
                      family="ease-out", rmse=0.02),
    )  # fmt: skip
    return ph._Ph("inertia", start, end, fit=fit, v_start=v0, v_end=0.0)


def test_glide_far_shorter_than_the_momentum_is_the_pointer_stopping() -> None:
    """The recording's 244 / 303 ms "glides to rest" after drags released at 1570–2350 px/s,
    while its momentum (τ ≈ 260 ms, 2 instances) would need ≈1.3–1.4 s to slow to 10 px/s: the
    pointer stopped while pressed; the glide merges into the drag, which then ends at rest."""
    phs = [
        ph._Ph("drag", 2.15, 2.55, v_end=-1409.0),
        _exp(2.55, 3.59, 262, -1409.0),
        ph._Ph("drag", 5.54, 6.16, v_end=1456.0),
        _glide(6.16, 6.46, 1570.0),
        ph._Ph("paused", 6.46, 6.58),
        ph._Ph("drag", 6.58, 6.68, v_end=117.0),
        _exp(6.68, 7.04, 258, 117.0),
        ph._Ph("drag", 7.04, 7.44, v_end=-1598.0),
        _glide(7.44, 7.69, -2347.0),
        ph._Ph("paused", 7.69, 7.88),
    ]
    out = ph._merge_pointer_stops(phs)
    assert [p.kind for p in out] == ["drag", "inertia", "drag", "paused", "drag", "inertia",
                                     "drag", "paused"]  # fmt: skip
    stops = [p for p in out if ph.NOTE_POINTER_STOP in p.notes]
    assert [(p.start, p.end, p.v_end) for p in stops] == [(5.54, 6.46, 0.0), (7.04, 7.69, 0.0)]


def test_glide_kept_without_a_consistent_exponential_momentum() -> None:
    base = [ph._Ph("drag", 1.0, 1.4, v_end=1500.0), _glide(1.4, 1.7, 1500.0)]
    # one exponential instance is no reference
    assert len(ph._merge_pointer_stops([*base, _exp(3.0, 4.0, 260, -900.0)])) == 3
    # two that disagree by more than 2x neither
    assert len(ph._merge_pointer_stops([*base, _exp(3.0, 4.0, 260, -900.0),
                                        _exp(5.0, 6.0, 700, -900.0)])) == 4  # fmt: skip
    # a long glide (≥ half the momentum's time to rest) is a real glide
    long_glide = [ph._Ph("drag", 1.0, 1.4, v_end=1500.0), _glide(1.4, 2.2, 1500.0)]
    refs = [_exp(3.0, 4.0, 260, -900.0), _exp(5.0, 6.0, 250, 900.0)]
    assert len(ph._merge_pointer_stops([*long_glide, *refs])) == 4


def vel_held_stop(t: np.ndarray) -> np.ndarray:
    """35 px/s autoplay; fling left (release −1400 px/s at 2.4 s) blending into autoplay;
    fling right (release +1100 px/s at 4.9 s) blending into autoplay; a drag left 7.0–7.7 s
    whose speed falls to zero with the pointer (half-sine); held still 200 ms; resume ramp
    815 ms (material-decelerate) back to 35 px/s."""
    v = np.full_like(t, 35.0)
    m = (t >= 2.0) & (t < 2.4)
    v[m] = -1400.0 * np.sin(np.pi / 2 * np.clip((t[m] - 2.0) / 0.15, 0, 1))
    m = (t >= 2.4) & (t < 4.5)
    v[m] = 35.0 + (-1400.0 - 35.0) * np.exp(-(t[m] - 2.4) / 0.26)
    m = (t >= 4.5) & (t < 4.9)
    v[m] = 1100.0 * np.sin(np.pi / 2 * np.clip((t[m] - 4.5) / 0.15, 0, 1))
    m = (t >= 4.9) & (t < 7.0)
    v[m] = 35.0 + (1100.0 - 35.0) * np.exp(-(t[m] - 4.9) / 0.26)
    m = (t >= 7.0) & (t < 7.7)
    v[m] = -np.pi * 600.0 / (2 * 0.7) * np.sin(np.pi * (t[m] - 7.0) / 0.7)
    m = (t >= 7.7) & (t < 7.9)
    v[m] = 0.0
    m = (t >= 7.9) & (t < 8.715)
    v[m] = 35.0 * ease("material-decelerate", (t[m] - 7.9) / 0.815)
    return v


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_drag_ending_at_rest_then_resume(seed: int) -> None:
    s = make_series(vel_held_stop, dur=10.0, sigma=0.05, seed=seed)
    seg = ph.segment(s)
    contiguous(seg.phases, s)
    assert kinds(seg.phases) == ["autoplay", "drag", "inertia", "autoplay", "drag", "inertia",
                                 "autoplay", "drag", "paused", "resume", "autoplay"]  # fmt: skip
    held = [p for p in seg.phases if p.kind == "drag"][-1]
    assert abs(held.start_s - 7.0) <= 0.034 and abs(held.end_s - 7.7) <= 0.05
    assert held.v_end == pytest.approx(0.0, abs=15.0)
    r = seg.behavior.resume
    assert r is not None
    assert r.delay_after_rest_ms.value == pytest.approx(200, abs=60)
    assert r.delay_after_release_ms is None  # the release itself is not visible
    assert r.ramp_ms.value == pytest.approx(815, abs=163)
    for p in seg.phases:
        if p.kind == "inertia":
            assert isinstance(p.fit, ExponentialFit) and p.fit.v_inf_px_s == pytest.approx(
                35.0, 0.05
            )


def _series(t: np.ndarray, x: np.ndarray) -> DisplacementSeries:
    return DisplacementSeries(times=t, pos=x, quality=np.ones_like(t), axis="x",
                              region=Rect(0, 0, 400, 100), region_confidence=0.9,
                              dt_median=float(np.median(np.diff(t))))  # fmt: skip


def test_short_noise_blip_and_trailing_wobble_stay_autoplay() -> None:
    """Tracking wobble of ≲ 1.5 px on low-texture content (in the middle and at the very end)
    is not a slowdown or a drag; a 133 ms pause (4.7 px off the line) still is."""
    t = np.arange(0, 6.0, 1 / 60)
    x = 35.0 * t
    x = x + np.where((t > 2.0) & (t < 2.6), 1.4 * np.sin(np.pi * (t - 2.0) / 0.3), 0.0)
    x = x + np.where(t > 5.7, -1.2 * np.sin(np.pi * (t - 5.7) / 0.3), 0.0)
    s = _series(t, x - x[0])
    assert kinds(ph.segment(s).phases) == ["autoplay"]
    v = np.full_like(t, 35.0)
    v[(t >= 3.0) & (t < 3.133)] = 0.0
    xp = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(t))])
    s2 = _series(t, xp)
    assert "autoplay" in kinds(ph.segment(s2).phases) and len(kinds(ph.segment(s2).phases)) > 1


def vel_short_momentum(t: np.ndarray) -> np.ndarray:
    """A fling blending into autoplay, then a slow flick (+146 px/s) whose momentum is
    re-grabbed after 360 ms — too short to tell "towards autoplay" from "towards zero with a
    longer τ" on its own (the recording's p9/p10) — then a fling left blending into autoplay."""
    v = np.full_like(t, 35.0)
    m = (t >= 2.0) & (t < 2.4)
    v[m] = -1400.0 * np.sin(np.pi / 2 * np.clip((t[m] - 2.0) / 0.15, 0, 1))
    m = (t >= 2.4) & (t < 4.4)
    v[m] = 35.0 + (-1400.0 - 35.0) * np.exp(-(t[m] - 2.4) / 0.26)
    m = (t >= 4.4) & (t < 4.5)
    v[m] = 146.0 * np.clip((t[m] - 4.4) / 0.03, 0, 1)
    m = (t >= 4.5) & (t < 4.86)
    v[m] = 35.0 + (146.0 - 35.0) * np.exp(-(t[m] - 4.5) / 0.26)
    m = (t >= 4.86) & (t < 5.26)
    v[m] = -1200.0 * np.sin(np.pi / 2 * np.clip((t[m] - 4.86) / 0.15, 0, 1))
    m = t >= 5.26
    v[m] = 35.0 + (-1200.0 - 35.0) * np.exp(-(t[m] - 5.26) / 0.26)
    return v


def test_unify_refits_an_interrupted_decay_towards_autoplay() -> None:
    """A 340 ms momentum from ~100 px/s decaying towards +35 px/s (τ 260 ms) fits a decay to
    zero with τ ≈ 500 ms just as well; with another release visibly decaying towards autoplay,
    the interrupted one is refitted towards autoplay (one scroller, one momentum law)."""
    from app.models.ir import MeasuredNumber  # noqa: PLC0415
    from app.pipeline.continuous import kinematics as kin  # noqa: PLC0415
    from app.pipeline.params import DEFAULT_PARAMS  # noqa: PLC0415

    p = DEFAULT_PARAMS.continuous
    t = 4.5 + np.arange(0, 0.345, 1 / 60)
    u = t - 4.5
    x = 35.0 * u + (100.0 - 35.0) * 0.26 * (1 - np.exp(-u / 0.26))
    x = x + np.random.default_rng(0).normal(0, 0.05, len(t))
    sig = np.full_like(t, 0.05)
    c = ph._Ctx(t=t, x=x, q=np.ones_like(t), v=np.gradient(x, t), sigma=sig, w=1 / sig**2,
                dt=1 / 60, v_auto=35.0, tol_a=4.0, tol_z=5.0, params=p, is_vfr=False,
                ts_est=False)  # fmt: skip
    raw0 = kin.fit_exponential(t, x, 4.5, 0.0, 1 / sig**2, tau_min=p.tau_min_s, tau_max=p.tau_max_s)
    assert raw0 is not None
    conf = {"value": 0.9, "band": "high"}
    tau0 = round(raw0.params["tau"] * 1000)
    fit0 = ExponentialFit(
        tau_ms=MeasuredNumber(value=tau0, confidence=conf),
        v0_px_s=MeasuredNumber(value=100.0, confidence=conf),
        v_inf_px_s=0.0,
    )
    short = ph._Ph(
        "inertia", 4.5, float(t[-1]), fit=fit0, raw=raw0, v_start=100.0, interrupted=True
    )
    ref = _exp(2.4, 4.4, 260, -1400.0)
    ref.raw = kin.Fit("exponential", {"v_inf": 35.0, "tau": 0.26}, 0.0, 10, 3, 0.0, np.zeros(10))
    out = ph._unify_momentum_target(c, [ref, short])
    fit = out[1].fit
    assert isinstance(fit, ExponentialFit) and fit.v_inf_px_s == pytest.approx(35.0)
    assert fit.tau_ms.value == pytest.approx(260, rel=0.25)
    # without a reference decaying towards autoplay nothing changes
    short2 = ph._Ph("inertia", 4.5, float(t[-1]), fit=fit0, raw=raw0, v_start=100.0,
                    interrupted=True)  # fmt: skip
    assert ph._unify_momentum_target(c, [short2])[0].fit.v_inf_px_s == 0.0  # type: ignore[union-attr]


@pytest.mark.parametrize("seed", [0, 1])
def test_reggrabbed_short_momentum_follows_the_scroller_law(seed: int) -> None:
    s = make_series(vel_short_momentum, dur=8.0, sigma=0.05, seed=seed)
    seg = ph.segment(s)
    exps = [p for p in seg.phases if p.kind == "inertia" and isinstance(p.fit, ExponentialFit)]
    assert len(exps) >= 3
    for p in exps:
        assert p.fit.v_inf_px_s == pytest.approx(35.0, rel=0.05)  # type: ignore[union-attr]


def vel_wavy_autoplay_then_press(t: np.ndarray) -> np.ndarray:
    """Autoplay whose on-screen speed drifts slowly (35 ± 1.5 px/s, cards scaling with their
    position), then a press at 3.0 s and a fling left."""
    v = 35.0 + 1.5 * np.sin(2 * np.pi * t / 1.6)
    m = t >= 3.0
    v[m] = -1300.0 * np.sin(np.pi / 2 * np.clip((t[m] - 3.0) / 0.2, 0, 1))
    m = t >= 3.4
    v[m] = 35.0 + (-1300.0 - 35.0) * np.exp(-(t[m] - 3.4) / 0.26)
    return v


def test_drag_onset_is_not_pulled_early_by_a_drifting_autoplay() -> None:
    s = make_series(vel_wavy_autoplay_then_press, dur=6.0, sigma=0.03, seed=0)
    seg = ph.segment(s)
    drag = first(seg.phases, "drag")
    assert abs(drag.start_s - 3.0) <= 2 / 60 + 1e-6
