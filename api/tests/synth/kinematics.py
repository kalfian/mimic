"""Velocity-integrated kinematics for continuous synthetic scenarios (PLAN-continuous §8.1).

A :class:`VelocityProfile` is a chain of phases that drive the signed content position ``x(t)``
of one scroller (CSS px; **+ = content moves right (x) / down (y)**, the IR sign convention):

* :class:`Autoplay` — constant velocity.
* :class:`Ramp` — eased velocity ramp ``v = v_a + (v_b - v_a) E((t - t0) / D)`` (decelerate /
  resume); ``v_a`` is the velocity the previous phase ended with.
* :class:`Hold` — at rest (``paused``).
* :class:`Drag` — pointer-driven: the press stops the content instantly, the pointer accelerates
  with a smoothstep ramp to ``v_release`` and keeps that speed until release. The content
  follows the pointer 1:1, so ``v_release`` is the analytic release velocity.
* :class:`Inertia` — ``v = v_inf + (v0 - v_inf) e^(-(t - t0) / tau)`` with ``v0`` = the release
  velocity; ends when ``|v|`` drops to ``v_stop`` (the JS "minimum velocity" cut-off) or after
  an explicit duration.
* :class:`SnapTween` — eased position tween onto the next grid point (card pitch) in the release
  direction.
* :class:`Stop` — abrupt linear stop over a few ms (pointer held still, or a hard snap).

Like ``animate``, this module is deliberately **independent** of ``app.pipeline`` (no shared
code with ``pipeline/continuous/*``): it is the ground truth those modules are measured against.
Positions use closed forms; the bezier velocity ramp is integrated numerically on a 0.1 ms grid
(trapezoid, error < 1e-6 px). All phase boundaries are whole milliseconds (truth times are
integer ms), so durations are rounded to 1 ms when a profile is compiled.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .animate import CubicBezier, CursorPath

Axis = Literal["x", "y"]
PhaseKind = Literal["autoplay", "decelerate", "paused", "drag", "inertia", "snap", "stop", "resume"]

#: Integration step for eased velocity ramps (s).
RAMP_GRID_S = 1e-4


# --------------------------------------------------------------------------------------------
# Phase specs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Autoplay:
    v: float
    dur: float


@dataclass(frozen=True)
class Ramp:
    """Eased velocity ramp from the current velocity to ``v_to``. ``kind`` defaults to
    ``decelerate`` when ``|v_to|`` is smaller than the starting speed, else ``resume``."""

    v_to: float
    dur: float
    easing: str = "ease-in-out"
    kind: Literal["decelerate", "resume"] | None = None


@dataclass(frozen=True)
class Hold:
    dur: float


@dataclass(frozen=True)
class Drag:
    """Pointer drag over ``distance`` px in ``dur`` s, released at ``v_release`` px/s.

    Velocity: ``v_release * smoothstep(u / (a D))`` for ``u < a D``, then ``v_release``, with
    ``a = 2 (1 - distance / (v_release D))`` so the integral equals ``distance``. Needs
    ``sign(distance) == sign(v_release)`` and ``|distance| / D <= |v_release| <= 2 |distance| / D``.
    """

    distance: float
    dur: float
    v_release: float


@dataclass(frozen=True)
class HeldDrag:
    """Pointer drag over ``distance`` px in ``dur`` s that slows to a stop **while pressed**
    (acceptance run on the motivating recording): velocity ``π·distance/(2·dur) ·
    sin(π u/dur)``, zero at both ends with a finite slope (the hand starts and stops
    briskly). The content stops with the pointer; letting go then has no velocity, so no
    momentum follows (a ``Hold`` = paused, then the resume ramp)."""

    distance: float
    dur: float


@dataclass(frozen=True)
class Inertia:
    tau: float
    v_inf: float = 0.0
    v_stop: float = 10.0
    dur: float | None = None


@dataclass(frozen=True)
class SnapTween:
    """Eased tween to the first grid point (``k * step``) at least ``min_px`` ahead of the
    release position in the release direction."""

    dur: float
    easing: str = "ease-out"
    step: float = 216.0
    min_px: float = 54.0


@dataclass(frozen=True)
class Stop:
    dur: float = 0.033


PhaseSpec = Autoplay | Ramp | Hold | Drag | HeldDrag | Inertia | SnapTween | Stop


# --------------------------------------------------------------------------------------------
# Compiled segments
# --------------------------------------------------------------------------------------------


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass
class Segment:
    """One compiled phase on ``[t0, t1]`` (seconds; whole ms)."""

    kind: PhaseKind
    t0_ms: int
    t1_ms: int
    x0: float
    v_in: float  # velocity of the previous phase at t0 (before any jump)
    spec: PhaseSpec
    # derived parameters
    v_a: float = 0.0  # ramp start / inertia v0 / stop start
    v_b: float = 0.0  # ramp end / inertia v_inf
    accel_frac: float = 0.0  # drag
    target: float = 0.0  # snap
    _cum: np.ndarray | None = field(default=None, repr=False)  # ramp: integral of E on the grid

    @property
    def t0(self) -> float:
        return self.t0_ms / 1000.0

    @property
    def t1(self) -> float:
        return self.t1_ms / 1000.0

    @property
    def dur(self) -> float:
        return (self.t1_ms - self.t0_ms) / 1000.0

    # ---- evaluation (u = t - t0, clamped to the segment) --------------------------------

    def _u(self, t: float) -> float:
        return min(max(t - self.t0, 0.0), self.dur)

    def position(self, t: float) -> float:
        u, D, s = self._u(t), self.dur, self.spec
        if isinstance(s, Autoplay):
            return self.x0 + s.v * u
        if isinstance(s, Hold):
            return self.x0
        if isinstance(s, Ramp):
            cum = self._cum
            assert cum is not None
            q = u / RAMP_GRID_S
            i = min(int(q), cum.size - 2)
            integral = float(cum[i] + (cum[i + 1] - cum[i]) * (q - i))
            return self.x0 + self.v_a * u + (self.v_b - self.v_a) * integral
        if isinstance(s, Drag):
            aD = self.accel_frac * D
            vr = s.v_release
            if u < aD:
                q = u / aD
                return self.x0 + vr * aD * (q**3 - q**4 / 2.0)
            return self.x0 + vr * (aD / 2.0 + (u - aD))
        if isinstance(s, HeldDrag):
            return self.x0 + s.distance * (1.0 - math.cos(math.pi * u / D)) / 2.0
        if isinstance(s, Inertia):
            v0, vi, tau = self.v_a, self.v_b, s.tau
            return self.x0 + vi * u + (v0 - vi) * tau * (1.0 - math.exp(-u / tau))
        if isinstance(s, SnapTween):
            p = float(CubicBezier.named(s.easing)(u / D))
            return self.x0 + (self.target - self.x0) * p
        if isinstance(s, Stop):
            return self.x0 + self.v_a * (u - u * u / (2.0 * D))
        raise TypeError(s)  # pragma: no cover

    def velocity(self, t: float) -> float:
        """Velocity at ``t`` inside the segment (right-continuous at ``t0``)."""
        u, D, s = self._u(t), self.dur, self.spec
        if isinstance(s, Autoplay):
            return s.v
        if isinstance(s, Hold):
            return 0.0
        if isinstance(s, Ramp):
            e = float(CubicBezier.named(s.easing)(u / D))
            return self.v_a + (self.v_b - self.v_a) * e
        if isinstance(s, Drag):
            return s.v_release * smoothstep(u / (self.accel_frac * D))
        if isinstance(s, HeldDrag):
            return math.pi * s.distance / (2.0 * D) * math.sin(math.pi * u / D)
        if isinstance(s, Inertia):
            return self.v_b + (self.v_a - self.v_b) * math.exp(-u / s.tau)
        if isinstance(s, SnapTween):
            # central difference of the (closed-form) position; one-sided at the ends
            h = 1e-5
            a, b = max(u - h, 0.0), min(u + h, D)
            return (self.position(self.t0 + b) - self.position(self.t0 + a)) / (b - a)
        if isinstance(s, Stop):
            return self.v_a * (1.0 - u / D)
        raise TypeError(s)  # pragma: no cover

    @property
    def v_start(self) -> float:
        return self.velocity(self.t0)

    @property
    def v_end(self) -> float:
        return self.velocity(self.t1)

    def v_peak(self) -> float:
        """Signed velocity with the largest magnitude inside the segment (1 ms grid)."""
        ts = np.linspace(self.t0, self.t1, max(self.t1_ms - self.t0_ms, 1) + 1)
        vs = [self.velocity(float(t)) for t in ts]
        return max(vs, key=abs)

    @property
    def displacement(self) -> float:
        return self.position(self.t1) - self.x0


# --------------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------------


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def _ramp_integral(easing: str, dur: float) -> np.ndarray:
    """Cumulative ``∫_0^u E(s / D) ds`` on the :data:`RAMP_GRID_S` grid (trapezoid)."""
    n = max(int(round(dur / RAMP_GRID_S)), 1)
    u = np.arange(n + 1) * RAMP_GRID_S
    e = CubicBezier.named(easing)(np.clip(u / dur, 0.0, 1.0))
    cum = np.concatenate([[0.0], np.cumsum((e[1:] + e[:-1]) * 0.5 * RAMP_GRID_S)])
    return cum


class VelocityProfile:
    """Chain of phases → ``position(t)`` / ``velocity(t)`` for a scroller's content.

    Before the first phase the content rests at ``x0``; after the last phase it continues with
    the last phase's end velocity (scenarios end on a phase boundary = video end).
    """

    def __init__(self, phases: Sequence[PhaseSpec], *, x0: float = 0.0, t0: float = 0.0) -> None:
        if not phases:
            raise ValueError("a velocity profile needs at least one phase")
        self.segments: list[Segment] = []
        t_ms, x, v = _ms(t0), float(x0), 0.0
        for spec in phases:
            seg = self._compile(spec, t_ms, x, v)
            self.segments.append(seg)
            t_ms, x, v = seg.t1_ms, seg.position(seg.t1), seg.v_end

    # ---- compile ----------------------------------------------------------------------

    @staticmethod
    def _compile(spec: PhaseSpec, t_ms: int, x: float, v_in: float) -> Segment:
        def seg(kind: PhaseKind, dur_s: float, **kw) -> Segment:
            d = _ms(dur_s)
            if d <= 0:
                raise ValueError(f"{kind} phase needs a positive duration (>= 1 ms), got {dur_s}")
            return Segment(kind=kind, t0_ms=t_ms, t1_ms=t_ms + d, x0=x, v_in=v_in, spec=spec, **kw)

        if isinstance(spec, Autoplay):
            return seg("autoplay", spec.dur)
        if isinstance(spec, Hold):
            return seg("paused", spec.dur)
        if isinstance(spec, Ramp):
            kind = spec.kind or ("decelerate" if abs(spec.v_to) < abs(v_in) else "resume")
            s = seg(kind, spec.dur, v_a=v_in, v_b=spec.v_to)
            s._cum = _ramp_integral(spec.easing, s.dur)
            return s
        if isinstance(spec, Drag):
            if spec.distance == 0 or np.sign(spec.distance) != np.sign(spec.v_release):
                raise ValueError("drag distance and release velocity need the same sign")
            d = _ms(spec.dur) / 1000.0
            a = 2.0 * (1.0 - spec.distance / (spec.v_release * d))
            if not 0.0 < a <= 1.0:
                raise ValueError(
                    f"drag release velocity {spec.v_release} not reachable: needs "
                    f"|distance|/dur <= |v| <= 2|distance|/dur (accel fraction {a:.3f})"
                )
            return seg("drag", spec.dur, accel_frac=a)
        if isinstance(spec, HeldDrag):
            if spec.distance == 0:
                raise ValueError("a held drag needs a distance")
            return seg("drag", spec.dur)
        if isinstance(spec, Inertia):
            if abs(v_in) <= spec.v_stop:
                raise ValueError("inertia needs a release velocity above v_stop")
            if spec.dur is not None:
                dur = spec.dur
            else:
                if spec.v_inf != 0.0:
                    raise ValueError("inertia towards a non-zero v_inf needs an explicit dur")
                dur = spec.tau * math.log(abs(v_in) / spec.v_stop)
            return seg("inertia", dur, v_a=v_in, v_b=spec.v_inf)
        if isinstance(spec, SnapTween):
            sign = 1.0 if v_in > 0 else -1.0
            if v_in == 0:
                raise ValueError("snap needs a release velocity (direction)")
            k = (
                math.ceil((x + spec.min_px) / spec.step)
                if sign > 0
                else math.floor((x - spec.min_px) / spec.step)
            )
            return seg("snap", spec.dur, target=k * spec.step)
        if isinstance(spec, Stop):
            return seg("stop", spec.dur, v_a=v_in)
        raise TypeError(spec)  # pragma: no cover

    # ---- evaluation -------------------------------------------------------------------

    @property
    def start_ms(self) -> int:
        return self.segments[0].t0_ms

    @property
    def end_ms(self) -> int:
        return self.segments[-1].t1_ms

    def segment_at(self, t: float) -> Segment | None:
        t_ms = t * 1000.0
        for s in self.segments:
            if s.t0_ms <= t_ms < s.t1_ms:
                return s
        return None

    def position(self, t: float) -> float:
        first, last = self.segments[0], self.segments[-1]
        if t <= first.t0:
            return first.x0
        if t >= last.t1:
            return last.position(last.t1) + last.v_end * (t - last.t1)
        s = self.segment_at(t)
        assert s is not None
        return s.position(t)

    def velocity(self, t: float) -> float:
        first, last = self.segments[0], self.segments[-1]
        if t < first.t0:
            return 0.0
        if t >= last.t1:
            return last.v_end
        s = self.segment_at(t)
        assert s is not None
        return s.velocity(t)

    def drag_windows(self) -> list[tuple[float, float]]:
        """``(press_s, release_s)`` of every drag phase."""
        return [(s.t0, s.t1) for s in self.segments if s.kind == "drag"]

    def travel_range(self, step_ms: int = 5) -> float:
        """``max x - min x`` over the profile (content positions covered by the recording)."""
        ts = np.arange(self.start_ms, self.end_ms + 1, step_ms) / 1000.0
        xs = [self.position(float(t)) for t in ts]
        return float(max(xs) - min(xs))


# --------------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------------


def wrap_translate(x: float, period: float) -> float:
    """Track translate for content position ``x`` on a strip duplicated once.

    The track holds two identical copies of length ``period``; ``translate`` stays in
    ``[-period, 0)`` and jumps by exactly one period, which is invisible because the copies are
    identical. ``x = 0`` shows the initial layout (second copy at the viewport start)."""
    return (x % period) - period


def strip_driver(
    profile: VelocityProfile, axis: Axis, period: float
) -> Callable[[float], tuple[float, float]]:
    """``Scene.drivers`` entry for a scroller track: ``t -> (tx, ty)``."""

    def drive(t: float) -> tuple[float, float]:
        tr = wrap_translate(profile.position(t), period)
        return (tr, 0.0) if axis == "x" else (0.0, tr)

    return drive


@dataclass(frozen=True)
class EdgeZoom:
    """Magnifier along a scroller's axis (P2b: cards grow towards the viewport edges).

    Screen offset ``x`` from the viewport centre ↔ layout offset ``u`` (both CSS px) with the
    local magnification ``s(x) = 1 + a x²``: 1 at the centre, ``edge`` at ``x = ±half``. Then
    ``dx/du = s`` integrates to ``x = tan(√a u) / √a``. A card whose layout centre sits at ``u``
    is drawn at ``x(u)`` scaled by ``s(x(u))`` about its centre, so the gaps scale with the cards
    (the motivating recording: 164 px cards at the centre, ≈192 px near the edges, 4 px gaps).
    Beyond ``x = ±1.5 half`` (clipped anyway) the mapping continues linearly.
    """

    edge: float
    half: float

    @property
    def a(self) -> float:
        return (self.edge - 1.0) / (self.half * self.half)

    @property
    def _u_max(self) -> float:
        r = math.sqrt(self.a)
        return math.atan(r * 1.5 * self.half) / r

    def screen(self, u: float) -> float:
        """Screen offset of layout offset ``u`` (odd function)."""
        if self.a <= 0.0:
            return u
        r, um = math.sqrt(self.a), self._u_max
        if abs(u) <= um:
            return math.tan(r * u) / r
        xm = 1.5 * self.half
        return math.copysign(xm + (1.0 + self.a * xm * xm) * (abs(u) - um), u)

    def scale(self, u: float) -> float:
        """Magnification at layout offset ``u``."""
        x = min(abs(self.screen(u)), 1.5 * self.half)
        return 1.0 + self.a * x * x

    def mean_scale(self) -> float:
        """Mean magnification over the viewport, ``1 + a half² / 3``: a rigid strip moving at
        ``v · mean_scale`` covers the same screen distance as this one at layout speed ``v``."""
        return 1.0 + self.a * self.half * self.half / 3.0


def card_zoom_driver(
    track: Callable[[float], tuple[float, float]],
    zoom: EdgeZoom,
    axis: Axis,
    centre: float,
) -> Callable[[float], tuple[float, float, float]]:
    """``Scene.drivers`` entry for one card of a zoomed strip: ``t -> (tx, ty, scale)``.

    ``track``: the strip's track driver; ``centre``: the card's layout centre along the axis
    relative to the viewport centre at track translate 0 (CSS px)."""

    def drive(t: float) -> tuple[float, float, float]:
        tr = track(t)[0 if axis == "x" else 1]
        u = centre + tr
        d = zoom.screen(u) - u
        s = zoom.scale(u)
        return (d, 0.0, s) if axis == "x" else (0.0, d, s)

    return drive


@dataclass
class DragCursor(CursorPath):
    """Cursor path that follows the content 1:1 during the profile's drag phases.

    Outside drag windows the ordinary keyed path applies; inside, the hotspot is the keyed
    position at the press plus the content displacement since the press along ``axis``. Keys
    must place the cursor at the press point at press time and at ``press + displacement`` at
    release time (``drag_keys`` builds those)."""

    profile: VelocityProfile | None = None
    axis: Axis = "x"

    def position(self, t: float) -> tuple[float, float]:
        if self.profile is not None:
            for press, release in self.profile.drag_windows():
                if press <= t <= release:
                    px, py = CursorPath.position(self, press)
                    d = self.profile.position(t) - self.profile.position(press)
                    return (px + d, py) if self.axis == "x" else (px, py + d)
        return CursorPath.position(self, t)


__all__ = [
    "Autoplay",
    "Axis",
    "DragCursor",
    "Drag",
    "HeldDrag",
    "Hold",
    "Inertia",
    "PhaseKind",
    "PhaseSpec",
    "Ramp",
    "Segment",
    "SnapTween",
    "Stop",
    "VelocityProfile",
    "smoothstep",
    "strip_driver",
    "wrap_translate",
]
