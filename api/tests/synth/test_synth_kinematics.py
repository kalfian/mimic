"""Kinematic drivers (PLAN-continuous §8.1): closed forms, phase rules, wrap, drag cursor, and
rendered strip shifts that agree with the profile (no codec)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from . import scenarios as S
from .kinematics import (
    Autoplay,
    Drag,
    DragCursor,
    Hold,
    Inertia,
    Ramp,
    SnapTween,
    Stop,
    VelocityProfile,
    wrap_translate,
)
from .scene import Renderer, hex_to_rgb


def _numeric_position(prof: VelocityProfile, t: float, dt: float = 1e-4) -> float:
    """Independent check: integrate the profile's velocity (midpoint rule; phase boundaries are
    whole ms, so no midpoint straddles a velocity jump)."""
    n = int(round((t - prof.start_ms / 1000.0) / dt))
    ts = prof.start_ms / 1000.0 + (np.arange(n) + 0.5) * dt
    v = np.array([prof.velocity(float(x)) for x in ts])
    return prof.segments[0].x0 + float(v.sum() * dt)


PROFILE = VelocityProfile([
    Autoplay(40.0, 0.5),
    Ramp(0.0, 0.3, "ease-out"),
    Hold(0.2),
    Drag(-300.0, 0.4, -1000.0),
    Inertia(0.2),
    Hold(0.1),
    Ramp(40.0, 0.5, "ease-in-out"),
    Autoplay(40.0, 0.2),
    Drag(200.0, 0.3, 900.0),
    SnapTween(0.3, "ease-out", step=216.0, min_px=54.0),
    Hold(0.2),
    Drag(-100.0, 0.2, -700.0),
    Stop(0.03),
])  # fmt: skip


def test_kinds_and_whole_ms_boundaries() -> None:
    kinds = [s.kind for s in PROFILE.segments]
    assert kinds == ["autoplay", "decelerate", "paused", "drag", "inertia", "paused", "resume",
                     "autoplay", "drag", "snap", "paused", "drag", "stop"]  # fmt: skip
    for a, b in zip(PROFILE.segments, PROFILE.segments[1:], strict=False):
        assert a.t1_ms == b.t0_ms
        assert a.position(a.t1) == pytest.approx(b.x0, abs=1e-9)  # position is continuous


@pytest.mark.parametrize("t", [0.3, 0.65, 0.9, 1.25, 1.6, 2.1, 2.5, 3.0, 3.3, 3.6, 3.9])
def test_closed_form_position_matches_integrated_velocity(t: float) -> None:
    assert PROFILE.position(t) == pytest.approx(_numeric_position(PROFILE, t), abs=2e-3)


def test_drag_inertia_ramp_values() -> None:
    drag = PROFILE.segments[3]
    assert drag.v_start == 0.0  # the press stops the content instantly
    assert drag.v_end == pytest.approx(-1000.0)  # analytic release velocity
    assert drag.displacement == pytest.approx(-300.0, abs=1e-9)
    inertia = PROFILE.segments[4]
    assert inertia.v_a == pytest.approx(-1000.0)
    assert inertia.t1_ms - inertia.t0_ms == round(200 * math.log(1000 / 10))  # v_stop cut-off
    assert abs(inertia.v_end) == pytest.approx(10.0, rel=0.01)
    assert inertia.displacement == pytest.approx(-1000 * 0.2 * (1 - 10 / 1000), rel=1e-3)
    decel = PROFILE.segments[1]
    assert decel.v_start == pytest.approx(40.0) and decel.v_end == pytest.approx(0.0)
    # an ease-out *velocity* ramp sheds speed early: less travel than linear (40 * 0.3 / 2)
    assert 0.0 < decel.displacement < 6.0
    resume = PROFILE.segments[6]
    # symmetric ease-in-out: exactly half of the constant-velocity distance
    assert resume.displacement == pytest.approx(40.0 * 0.5 / 2, abs=1e-4)


def test_snap_lands_on_grid_in_release_direction() -> None:
    drag, snap = PROFILE.segments[8], PROFILE.segments[9]
    x_rel = drag.position(drag.t1)
    end = snap.position(snap.t1)
    assert end % 216.0 == pytest.approx(0.0, abs=1e-9)
    assert end - x_rel >= 54.0  # forward (release velocity > 0), at least min_px
    assert snap.v_end == pytest.approx(0.0, abs=1.0)


def test_stop_is_abrupt() -> None:
    stop = PROFILE.segments[-1]
    assert stop.v_start == pytest.approx(-700.0) and stop.v_end == pytest.approx(0.0, abs=1e-9)
    assert stop.t1_ms - stop.t0_ms == 30


def test_invalid_phases_rejected() -> None:
    with pytest.raises(ValueError, match="not reachable"):
        VelocityProfile([Drag(-100.0, 0.4, -2000.0)])  # > 2 |d| / T
    with pytest.raises(ValueError, match="same sign"):
        VelocityProfile([Drag(-100.0, 0.4, 400.0)])
    with pytest.raises(ValueError, match="release velocity"):
        VelocityProfile([Hold(0.1), Inertia(0.2)])
    with pytest.raises(ValueError, match="positive duration"):
        VelocityProfile([Autoplay(10.0, 0.0)])


def test_wrap_translate() -> None:
    L = 1728.0
    assert wrap_translate(0.0, L) == -L
    assert wrap_translate(10.0, L) == pytest.approx(-L + 10.0)
    assert wrap_translate(-10.0, L) == pytest.approx(-10.0)
    for x in np.linspace(-5000, 5000, 41):
        tr = wrap_translate(float(x), L)
        assert -L <= tr < 0.0
        assert (tr - x) % L == pytest.approx(0.0, abs=1e-6) or (tr - x) % L == pytest.approx(L)


def test_drag_cursor_follows_content_1_to_1() -> None:
    defn = S.get("carousel_drag_inertia").build()
    cur = defn.scene.cursor
    prof = defn.continuous.profile
    assert isinstance(cur, DragCursor)
    for press, release in prof.drag_windows():
        p0 = cur.position(press)
        for t in np.linspace(press, release, 9):
            x, y = cur.position(float(t))
            assert x - p0[0] == pytest.approx(prof.position(float(t)) - prof.position(press))
            assert y == p0[1]
        assert cur.position(release + 0.2) == pytest.approx(cur.position(release))  # rests
    assert cur.clicks == prof.drag_windows()


def _card_centres(frame: np.ndarray, strip: S.Strip) -> list[float]:
    """Sub-pixel centres (CSS px, along the axis) of the cards fully visible in the viewport,
    from the whiteness centroid on a scan line that crosses only the cards' plain white area
    (between image and title for x strips, right margin for y strips)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bg = float(np.mean(hex_to_rgb(S.STRIP_BG)))  # flat grey: R≈G≈B
    if strip.axis == "x":
        line = g[int(strip.y + strip.cross + 110), int(strip.x) : int(strip.x + strip.w)]
        origin = strip.x
    else:
        line = g[int(strip.y) : int(strip.y + strip.h), int(strip.x + strip.cross + 196)]
        origin = strip.y
    w = np.clip((line - bg) / (255.0 - bg), 0.0, 1.0)
    on = w > 0.02
    centres, i = [], 0
    while i < on.size:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < on.size and on[j]:
            j += 1
        if i > 0 and j < on.size:  # fully inside the viewport
            idx = np.arange(i, j)
            centres.append(origin + float((idx * w[i:j]).sum() / w[i:j].sum()) + 0.5)
        i = j
    return centres


@pytest.mark.parametrize(
    "name,times",
    [
        ("marquee_autoplay", [0.0, 1.0, 3.37, 7.9]),
        ("carousel_drag_inertia", [2.0, 2.31, 2.45, 2.7, 4.6, 6.3]),
        ("carousel_drag_snap", [2.75, 6.7]),  # snapped rests
        ("ticker_vertical_autoplay", [0.5, 2.2, 4.6]),
    ],
)
def test_rendered_strip_position_matches_profile(name: str, times: list[float]) -> None:
    """Absolute check: every visible card centre sits at ``lead + x(t) + k * pitch`` (mod)."""
    defn = S.get(name).build()
    strip, prof = defn.continuous.strip, defn.continuous.profile
    along = strip.card_w if strip.axis == "x" else strip.card_h
    origin = strip.x if strip.axis == "x" else strip.y
    r = Renderer(defn.scene)
    for t in times:
        centres = _card_centres(r.frame(t), strip)
        assert len(centres) >= 2, (name, t)
        want = origin + strip.lead + along / 2.0 + prof.position(t)
        for cx in centres:
            off = (cx - want) % strip.pitch
            err = min(off, strip.pitch - off)
            assert err < 0.05, (name, t, cx, err)


def test_wrap_is_invisible() -> None:
    """Positions one period apart render identically (duplicated strip, seamless loop)."""
    defn = S.get("marquee_fast_loop").build()
    strip, prof = defn.continuous.strip, defn.continuous.profile
    t0 = 0.5
    t1 = t0 + strip.period / abs(prof.velocity(t0))  # exactly one period later
    r = Renderer(defn.scene)
    x, y, w, h = (int(v) for v in strip.box)
    a = r.frame(t0)[y : y + h, x : x + w].astype(int)
    b = r.frame(t1)[y : y + h, x : x + w].astype(int)
    assert np.abs(a - b).max() <= 1
