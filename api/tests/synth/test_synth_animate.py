"""The synth suite's own cubic-bezier evaluator and tween/cursor timing."""

from __future__ import annotations

import numpy as np
import pytest

from .animate import (
    EASINGS,
    CubicBezier,
    CursorKey,
    CursorPath,
    Timeline,
    Tween,
    solve_start_for_crossing,
)


def _dense_reference(b: tuple[float, float, float, float], x: np.ndarray) -> np.ndarray:
    """Brute force: sample the parametric curve densely and interpolate y(x)."""
    u = np.linspace(0.0, 1.0, 400_001)
    x1, y1, x2, y2 = b
    bx = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u**2 * x2 + u**3
    by = 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u**2 * y2 + u**3
    return np.interp(x, bx, by)


@pytest.mark.parametrize("name", sorted(EASINGS))
def test_bezier_matches_dense_reference(name: str) -> None:
    bez, _ = EASINGS[name]
    x = np.linspace(0.0, 1.0, 257)
    got = CubicBezier(*bez)(x)
    np.testing.assert_allclose(got, _dense_reference(bez, x), atol=2e-5)
    assert got[0] == 0.0 and got[-1] == 1.0


def test_linear_is_identity_and_symmetric_curves_are_symmetric() -> None:
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(CubicBezier.named("linear")(x), x, atol=1e-9)
    f = CubicBezier.named("ease-in-out")
    np.testing.assert_allclose(f(x) + f(1 - x), 1.0, atol=1e-9)
    assert abs(float(f(0.5)) - 0.5) < 1e-9


def test_known_points_and_shape() -> None:
    # ease-out is ahead of linear, ease-in behind it
    assert float(CubicBezier.named("ease-out")(0.25)) > 0.25 + 0.1
    assert float(CubicBezier.named("ease-in")(0.25)) < 0.25 - 0.1
    for name, (_, fam) in EASINGS.items():
        y = CubicBezier.named(name)(np.linspace(0, 1, 501))
        if fam == "overshoot":
            assert y.max() > 1.02, name
        else:
            assert np.all(np.diff(y) >= -1e-12), name


def test_x_outside_unit_rejected() -> None:
    with pytest.raises(ValueError):
        CubicBezier(1.2, 0, 0.5, 1)


def test_timeline_holds_values_between_tweens() -> None:
    tl = Timeline()
    tl.add(
        Tween("n", "translateY", "fwd", 1.0, 0.2, 0.0, -8.0, "linear"),
        Tween("n", "translateY", "rev", 2.0, 0.2, -8.0, 0.0, "linear"),
    )
    tl.validate()
    assert tl.value("n", "translateY", 0.0, 0.5) == 0.0
    assert tl.value("n", "translateY", 0.0, 1.1) == pytest.approx(-4.0)
    assert tl.value("n", "translateY", 0.0, 1.6) == -8.0
    assert tl.value("n", "translateY", 0.0, 2.1) == pytest.approx(-4.0)
    assert tl.value("n", "translateY", 0.0, 3.0) == 0.0
    assert tl.is_static(1.5) and not tl.is_static(1.1)


def test_timeline_rejects_overlap_and_discontinuity() -> None:
    tl = Timeline([Tween("n", "scale", "fwd", 1.0, 0.3, 1.0, 1.1, "linear"),
                   Tween("n", "scale", "rev", 1.2, 0.3, 1.1, 1.0, "linear")])  # fmt: skip
    with pytest.raises(ValueError, match="overlapping"):
        tl.validate()
    tl = Timeline([Tween("n", "scale", "fwd", 1.0, 0.1, 1.0, 1.1, "linear"),
                   Tween("n", "scale", "rev", 2.0, 0.1, 1.05, 1.0, "linear")])  # fmt: skip
    with pytest.raises(ValueError, match="discontinuous"):
        tl.validate()


def test_solve_start_for_crossing() -> None:
    x0 = solve_start_for_crossing(600, 480, 0.5, 1.3, 1.0, "ease-in-out")
    path = CursorPath([CursorKey(0.5, x0, 0), CursorKey(1.3, 600, 0, "ease-in-out")])
    assert path.position(1.0)[0] == pytest.approx(480.0, abs=1e-6)
    assert path.position(0.0) == (x0, 0)
    assert path.position(5.0) == (600, 0)
