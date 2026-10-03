"""Keyframed properties + an independent cubic-bezier evaluator (PLAN §11.2).

This module deliberately does NOT import ``app.pipeline.easing``. The synthetic suite is the
ground truth for the measurement pipeline, so it carries its own implementation of the CSS
timing function: a shared bug must not be able to hide itself.

Evaluation solves ``x(u) = x`` per query with Newton's method and finishes with bisection for
any query Newton did not converge on, then returns ``y(u)``. That is the same contract browsers
implement (WebKit/Blink ``UnitBezier``), written from the math rather than copied.

Times here are float seconds; colors are ``(r, g, b)`` floats in sRGB 0..255; shadows are
``(x, y, blur, spread, alpha)`` floats in CSS px. Interpolation is linear per component, which
matches CSS transitions for numbers, sRGB colors (the default interpolation space for
``transition``) and box-shadow layers of identical shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Bezier = tuple[float, float, float, float]
Family = Literal["linear", "ease", "ease-in", "ease-out", "ease-in-out", "overshoot"]
Color = tuple[float, float, float]
ShadowTuple = tuple[float, float, float, float, float]
AnimValue = float | Color | ShadowTuple

# --------------------------------------------------------------------------------------------
# Named curves (PLAN Appendix B). Kept as a local table on purpose (independence, see above).
# --------------------------------------------------------------------------------------------

#: name -> (cubic-bezier, family)
EASINGS: dict[str, tuple[Bezier, Family]] = {
    "linear": ((0.0, 0.0, 1.0, 1.0), "linear"),
    "ease": ((0.25, 0.1, 0.25, 1.0), "ease"),
    "ease-in": ((0.42, 0.0, 1.0, 1.0), "ease-in"),
    "ease-out": ((0.0, 0.0, 0.58, 1.0), "ease-out"),
    "ease-in-out": ((0.42, 0.0, 0.58, 1.0), "ease-in-out"),
    "easeOutQuad": ((0.25, 0.46, 0.45, 0.94), "ease-out"),
    "easeOutCubic": ((0.215, 0.61, 0.355, 1.0), "ease-out"),
    "easeOutQuart": ((0.165, 0.84, 0.44, 1.0), "ease-out"),
    "easeOutExpo": ((0.19, 1.0, 0.22, 1.0), "ease-out"),
    "expoOut": ((0.16, 1.0, 0.3, 1.0), "ease-out"),
    "easeInCubic": ((0.55, 0.055, 0.675, 0.19), "ease-in"),
    "easeInOutCubic": ((0.645, 0.045, 0.355, 1.0), "ease-in-out"),
    "material-standard": ((0.4, 0.0, 0.2, 1.0), "ease-in-out"),
    "material-decelerate": ((0.0, 0.0, 0.2, 1.0), "ease-out"),
    "material-accelerate": ((0.4, 0.0, 1.0, 1.0), "ease-in"),
    "backOut": ((0.34, 1.56, 0.64, 1.0), "overshoot"),
    "easeOutBack": ((0.175, 0.885, 0.32, 1.275), "overshoot"),
}

#: The five CSS keywords (an IR ``easing.keyword`` may only be one of these).
CSS_KEYWORDS: frozenset[str] = frozenset({"linear", "ease", "ease-in", "ease-out", "ease-in-out"})


class CubicBezier:
    """CSS ``cubic-bezier(x1, y1, x2, y2)`` timing function, vectorized over numpy arrays."""

    __slots__ = ("x1", "y1", "x2", "y2", "_ax", "_bx", "_cx", "_ay", "_by", "_cy")

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0):
            raise ValueError(f"cubic-bezier x1/x2 must be in [0, 1], got {x1}, {x2}")
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        # Power-basis coefficients of B(u) = 3(1-u)^2 u P1 + 3(1-u) u^2 P2 + u^3 (P0=0, P3=1).
        self._cx = 3.0 * self.x1
        self._bx = 3.0 * (self.x2 - self.x1) - self._cx
        self._ax = 1.0 - self._cx - self._bx
        self._cy = 3.0 * self.y1
        self._by = 3.0 * (self.y2 - self.y1) - self._cy
        self._ay = 1.0 - self._cy - self._by

    @classmethod
    def named(cls, name: str) -> CubicBezier:
        return cls(*EASINGS[name][0])

    @property
    def params(self) -> Bezier:
        return (self.x1, self.y1, self.x2, self.y2)

    def _x(self, u: np.ndarray) -> np.ndarray:
        return ((self._ax * u + self._bx) * u + self._cx) * u

    def _y(self, u: np.ndarray) -> np.ndarray:
        return ((self._ay * u + self._by) * u + self._cy) * u

    def _dx(self, u: np.ndarray) -> np.ndarray:
        return (3.0 * self._ax * u + 2.0 * self._bx) * u + self._cx

    def solve_u(self, x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Parameter ``u`` with ``x(u) = x`` for ``x`` in [0, 1] (Newton, then bisection)."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        u = x.copy()
        for _ in range(8):
            err = self._x(u) - x
            d = self._dx(u)
            ok = np.abs(d) > 1e-6
            u = np.where(ok, u - np.where(ok, err / np.where(ok, d, 1.0), 0.0), u)
            u = np.clip(u, 0.0, 1.0)
        bad = np.abs(self._x(u) - x) > eps
        if np.any(bad):
            lo = np.zeros(int(bad.sum()))
            hi = np.ones_like(lo)
            xb = x[bad]
            mid = (lo + hi) / 2.0
            for _ in range(64):  # x(u) is monotonic for x1, x2 in [0, 1]
                mid = (lo + hi) / 2.0
                gt = self._x(mid) > xb
                hi = np.where(gt, mid, hi)
                lo = np.where(gt, lo, mid)
            u[bad] = mid
        return u

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        """Output progress for input time fraction(s) ``x`` (clamped to [0, 1])."""
        arr = np.asarray(x, dtype=np.float64)
        out = self._y(self.solve_u(arr.reshape(-1))).reshape(arr.shape)
        # Endpoints are exact by definition (avoids 0.9999999 drifting into the truth).
        out = np.where(arr <= 0.0, 0.0, np.where(arr >= 1.0, 1.0, out))
        return out


def easing_family(name: str) -> Family:
    return EASINGS[name][1]


def lerp(a: AnimValue, b: AnimValue, p: float) -> AnimValue:
    """Linear interpolation for scalars and equal-length tuples."""
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(x + (y - x) * p for x, y in zip(a, b, strict=True))  # type: ignore[return-value]
    assert not isinstance(b, tuple)
    return a + (b - a) * p


# --------------------------------------------------------------------------------------------
# Tweens / tracks
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tween:
    """One CSS-like transition of one property of one node.

    ``prop`` uses the IR property vocabulary (``translateY``, ``background-color``, ...).
    ``segment`` is the IR segment id the transition belongs to (``fwd``, ``rev``, ``rt_in``,
    ``rt_out``). ``start_s`` is the absolute video time where progress leaves 0.
    """

    node: str
    prop: str
    segment: str
    start_s: float
    duration_s: float
    from_value: AnimValue
    to_value: AnimValue
    easing: str
    notes: tuple[str, ...] = ()

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def bezier(self) -> CubicBezier:
        return CubicBezier.named(self.easing)

    def progress(self, t: float) -> float:
        if t <= self.start_s:
            return 0.0
        if t >= self.end_s:
            return 1.0
        return float(self.bezier((t - self.start_s) / self.duration_s))

    def value_at(self, t: float) -> AnimValue:
        return lerp(self.from_value, self.to_value, self.progress(t))


@dataclass
class Timeline:
    """All tweens of a scene. A property holds its base value until its first tween starts,
    then each tween's value from its start until the next tween of the same property starts.
    Tweens of one (node, prop) must not overlap (no interrupted transitions in the suite)."""

    tweens: list[Tween] = field(default_factory=list)

    def add(self, *tweens: Tween) -> None:
        self.tweens.extend(tweens)

    def validate(self) -> None:
        by_key: dict[tuple[str, str], list[Tween]] = {}
        for tw in self.tweens:
            if tw.duration_s <= 0:
                raise ValueError(f"tween {tw.node}.{tw.prop} has non-positive duration")
            by_key.setdefault((tw.node, tw.prop), []).append(tw)
        for key, tws in by_key.items():
            tws.sort(key=lambda w: w.start_s)
            for a, b in zip(tws, tws[1:], strict=False):
                if b.start_s < a.end_s - 1e-9:
                    raise ValueError(f"overlapping tweens on {key}")
                if a.to_value != b.from_value:
                    raise ValueError(
                        f"discontinuous tweens on {key}: {a.to_value} -> {b.from_value}"
                    )

    def for_prop(self, node: str, prop: str) -> list[Tween]:
        return sorted(
            (w for w in self.tweens if w.node == node and w.prop == prop), key=lambda w: w.start_s
        )

    def value(self, node: str, prop: str, base: AnimValue, t: float) -> AnimValue:
        tws = self.for_prop(node, prop)
        if not tws:
            return base
        current: Tween | None = None
        for tw in tws:
            if tw.start_s <= t:
                current = tw
            else:
                break
        if current is None:
            return tws[0].from_value
        return current.value_at(t)

    def is_static(self, t: float) -> bool:
        """True if no tween is in progress at ``t``."""
        return not any(w.start_s < t < w.end_s for w in self.tweens)


# --------------------------------------------------------------------------------------------
# Cursor path
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CursorKey:
    """Cursor hotspot at time ``t_s`` (CSS px). The leg *arriving* at this key uses ``easing``."""

    t_s: float
    x: float
    y: float
    easing: str = "ease-in-out"


@dataclass
class CursorPath:
    keys: list[CursorKey]
    visible: bool = True
    #: "arrow" | "hand" | "auto" (hand while over a ``pointer`` node, arrow otherwise).
    style: str = "arrow"
    #: (mousedown_s, mouseup_s) pairs. Clicks are invisible (like macOS screen recordings).
    clicks: list[tuple[float, float]] = field(default_factory=list)

    def position(self, t: float) -> tuple[float, float]:
        ks = self.keys
        if t <= ks[0].t_s:
            return (ks[0].x, ks[0].y)
        for a, b in zip(ks, ks[1:], strict=False):
            if t <= b.t_s:
                span = b.t_s - a.t_s
                p = float(CubicBezier.named(b.easing)((t - a.t_s) / span)) if span > 0 else 1.0
                return (a.x + (b.x - a.x) * p, a.y + (b.y - a.y) * p)
        return (ks[-1].x, ks[-1].y)

    def moving(self, t: float, dt: float = 1e-3) -> bool:
        p0, p1 = self.position(t - dt), self.position(t + dt)
        return abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) > 1e-6


def solve_start_for_crossing(
    end_x: float,
    edge_x: float,
    t_start: float,
    t_end: float,
    t_cross: float,
    easing: str,
) -> float:
    """Start x of a horizontal leg ``t_start → t_end`` ending at ``end_x`` so that the hotspot
    crosses ``edge_x`` exactly at ``t_cross``. Used to place "cursor enters at 1.00 s"."""
    p = float(CubicBezier.named(easing)((t_cross - t_start) / (t_end - t_start)))
    if not 0.0 < p < 1.0:
        raise ValueError("t_cross must fall strictly inside the leg")
    # edge = x0 + p (end - x0)  =>  x0 = (edge - p end) / (1 - p)
    return (edge_x - p * end_x) / (1.0 - p)


def first_time(
    test: Callable[[float], bool],
    *,
    t_from: float,
    t_to: float,
    step: float = 1e-3,
) -> float | None:
    """First ``t`` on a ``step`` grid in ``[t_from, t_to]`` where ``test(t)`` is true.

    Coarse-to-fine (10 steps, then 1): assumes ``test`` does not flip twice within 10 steps."""
    n = int(round((t_to - t_from) / step))
    prev = 0
    for i in [*range(0, n + 1, 10), n]:
        if test(t_from + i * step):
            for j in range(prev, i + 1):
                t = t_from + j * step
                if test(t):
                    return round(t, 6)
        prev = i
    return None
