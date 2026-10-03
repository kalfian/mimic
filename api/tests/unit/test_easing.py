"""Track M2 — easing.py: bezier evaluation, candidate set, free fit, family logic (PLAN §11.1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import CSS_KEYWORD_BEZIER
from app.pipeline import easing as ez
from app.pipeline.params import FitParams


def ref_bezier(bezier: tuple[float, float, float, float], x: float) -> float:
    """Independent scalar reference: Newton on x(u) with bisection fallback."""
    x1, y1, x2, y2 = bezier
    x = min(max(x, 0.0), 1.0)

    def bx(u: float) -> float:
        return 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u**3

    def dbx(u: float) -> float:
        return 3 * (1 - u) ** 2 * x1 + 6 * (1 - u) * u * (x2 - x1) + 3 * u * u * (1 - x2)

    u = x
    for _ in range(8):
        d = dbx(u)
        if abs(d) < 1e-6:
            break
        u = min(max(u - (bx(u) - x) / d, 0.0), 1.0)
    if abs(bx(u) - x) > 1e-10:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            u = (lo + hi) / 2
            if bx(u) < x:
                lo = u
            else:
                hi = u
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u**3


APPENDIX_B = {
    "linear": ((0, 0, 1, 1), "linear"),
    "ease": ((0.25, 0.1, 0.25, 1), "ease"),
    "ease-in": ((0.42, 0, 1, 1), "ease-in"),
    "ease-out": ((0, 0, 0.58, 1), "ease-out"),
    "ease-in-out": ((0.42, 0, 0.58, 1), "ease-in-out"),
    "easeOutQuad": ((0.25, 0.46, 0.45, 0.94), "ease-out"),
    "easeOutCubic": ((0.215, 0.61, 0.355, 1), "ease-out"),
    "easeOutQuart": ((0.165, 0.84, 0.44, 1), "ease-out"),
    "easeOutExpo": ((0.19, 1, 0.22, 1), "ease-out"),
    "expoOut": ((0.16, 1, 0.3, 1), "ease-out"),
    "easeInCubic": ((0.55, 0.055, 0.675, 0.19), "ease-in"),
    "easeInOutCubic": ((0.645, 0.045, 0.355, 1), "ease-in-out"),
    "material-standard": ((0.4, 0, 0.2, 1), "ease-in-out"),
    "material-decelerate": ((0, 0, 0.2, 1), "ease-out"),
    "material-accelerate": ((0.4, 0, 1, 1), "ease-in"),
    "backOut": ((0.34, 1.56, 0.64, 1), "overshoot"),
    "easeOutBack": ((0.175, 0.885, 0.32, 1.275), "overshoot"),
}


# --------------------------------------------------------------------------------------------
# Candidate set
# --------------------------------------------------------------------------------------------


def test_candidates_match_appendix_b() -> None:
    assert [c.name for c in ez.CANDIDATES] == list(APPENDIX_B)
    for c in ez.CANDIDATES:
        bez, fam = APPENDIX_B[c.name]
        assert c.bezier == pytest.approx(bez)
        assert c.family == fam
        assert c.overshoot_only == (fam == "overshoot")
        if c.name in CSS_KEYWORD_BEZIER:
            assert c.keyword == c.name
            assert c.bezier == CSS_KEYWORD_BEZIER[c.name]
        else:
            assert c.keyword is None


def test_candidate_set_overshoot_gating() -> None:
    plain = ez.candidate_set(include_overshoot=False)
    full = ez.candidate_set(include_overshoot=True)
    assert all(c.family != "overshoot" for c in plain)
    assert {c.name for c in full} - {c.name for c in plain} == {"backOut", "easeOutBack"}


def test_families_match_lenient_pair_only() -> None:
    assert ez.families_match("ease-out", "ease-out")
    assert ez.families_match("ease", "ease-in-out")
    assert ez.families_match("ease-in-out", "ease")
    assert not ez.families_match("ease", "ease-out")
    assert not ez.families_match("ease-in", "ease-in-out")


# --------------------------------------------------------------------------------------------
# Evaluation: LUT vs analytic points
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(APPENDIX_B))
def test_solver_hits_analytic_points(name: str) -> None:
    bez = ez.CANDIDATES_BY_NAME[name].bezier
    u = np.linspace(0.0, 1.0, 101)
    x, y = ez.bezier_point(bez, u)
    assert np.max(np.abs(ez.solve_bezier(bez, x) - y)) < 1e-9


@pytest.mark.parametrize("name", list(APPENDIX_B))
def test_lut_matches_analytic_points(name: str) -> None:
    bez = ez.CANDIDATES_BY_NAME[name].bezier
    u = np.linspace(0.0, 1.0, 257)
    x, y = ez.bezier_point(bez, u)
    assert np.max(np.abs(ez.eval_curve(bez, x, 513) - y)) < 5e-4
    assert np.max(np.abs(ez.eval_curve(bez, x, 2001) - y)) < 5e-5


@pytest.mark.parametrize("name", list(APPENDIX_B))
def test_lut_matches_independent_reference(name: str) -> None:
    bez = ez.CANDIDATES_BY_NAME[name].bezier
    xs = np.linspace(0.0, 1.0, 37)
    ref = np.array([ref_bezier(bez, float(v)) for v in xs])
    assert np.max(np.abs(ez.solve_bezier(bez, xs) - ref)) < 1e-8
    assert np.max(np.abs(ez.eval_curve(bez, xs) - ref)) < 5e-4


def test_endpoints_clamping_and_linear_identity() -> None:
    for c in ez.CANDIDATES:
        y = ez.eval_curve(c.bezier, np.array([-0.5, 0.0, 1.0, 1.5]))
        assert y == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-12)
    xs = np.linspace(0, 1, 11)
    assert ez.eval_curve((0, 0, 1, 1), xs) == pytest.approx(xs, abs=1e-12)


def test_known_values() -> None:
    # ease-in-out is symmetric: E(0.5) = 0.5, E(x) + E(1 - x) = 1
    eio = CSS_KEYWORD_BEZIER["ease-in-out"]
    assert ez.solve_bezier(eio, 0.5) == pytest.approx(0.5, abs=1e-12)
    xs = np.linspace(0, 1, 21)
    assert ez.solve_bezier(eio, xs) + ez.solve_bezier(eio, 1 - xs) == pytest.approx(1.0)
    # ease-out is above the diagonal, ease-in below
    assert np.all(ez.solve_bezier(CSS_KEYWORD_BEZIER["ease-out"], xs[1:-1]) > xs[1:-1])
    assert np.all(ez.solve_bezier(CSS_KEYWORD_BEZIER["ease-in"], xs[1:-1]) < xs[1:-1])


def test_eval_tables_batches_curves() -> None:
    curves = [c.bezier for c in ez.CANDIDATES]
    tables = ez.tables_for(curves, 513)
    x = np.linspace(0, 1, 50)
    shared = ez.eval_tables(tables, x[None, :])
    assert shared.shape == (len(curves), 50)
    for k, bez in enumerate(curves):
        assert shared[k] == pytest.approx(ez.eval_curve(bez, x))
    per_curve = ez.eval_tables(tables, np.tile(x, (len(curves), 1)))
    assert per_curve == pytest.approx(shared)
    with pytest.raises(ValueError):
        ez.eval_tables(tables, np.zeros((3, 4)))


def test_overshoot_curves_peak_above_one() -> None:
    assert ez.curve_peak(APPENDIX_B["backOut"][0]) > 1.05
    assert ez.curve_peak(APPENDIX_B["easeOutBack"][0]) > 1.05
    assert ez.curve_peak(APPENDIX_B["ease-out"][0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------
# Overshoot detection, family, IR easing
# --------------------------------------------------------------------------------------------


def test_detect_overshoot_rule() -> None:
    p = np.array([0, 0.5, 1.0, 1.03, 1.0])
    assert not ez.detect_overshoot(p)
    assert not ez.detect_overshoot(np.array([0, 1.08, 1.0, 1.0]))  # one sample only
    assert ez.detect_overshoot(np.array([0, 1.05, 1.06, 1.0]))
    assert ez.detect_overshoot(np.array([0, 1.05, 1.06]), FitParams(overshoot_min_samples=2))


def test_family_of_and_nearest_named() -> None:
    for c in ez.CANDIDATES:
        assert ez.nearest_named(c.bezier).name == c.name
        assert ez.family_of(c.bezier) == c.family
    assert ez.family_of((0.0, 0.0, 0.6, 1.0)) == "ease-out"  # near ease-out
    assert ez.family_of((0.3, 1.8, 0.6, 1.0)) == "overshoot"


def test_make_easing_named_free_and_flags() -> None:
    e = ez.make_easing((0, 0, 0.58, 1), 0.01234, named=ez.CANDIDATES_BY_NAME["ease-out"])
    assert e.keyword == "ease-out" and e.family == "ease-out" and e.rmse == 0.0123
    e = ez.make_easing((0, 0, 0, 0), 0.02, named=ez.CANDIDATES_BY_NAME["expoOut"])
    assert e.keyword is None and e.cubic_bezier == (0.16, 1.0, 0.3, 1.0)
    assert e.nearest_named == "expoOut"
    e = ez.make_easing((0.05, 0.0, 0.6, 1.0), 0.02)
    assert e.keyword is None and e.family == "ease-out" and e.flags == []
    e = ez.make_easing((0.3, 1.6, 0.6, 1.0), 0.02)
    assert e.family == "overshoot" and e.flags == ["overshoot"]
    e = ez.make_easing(
        ez.CANDIDATES_BY_NAME["backOut"].bezier, 0.02, named=ez.CANDIDATES_BY_NAME["backOut"]
    )
    assert e.flags == ["overshoot"]
    # a free fit that lands exactly on a keyword reports the keyword
    assert ez.make_easing((0.42, 0.0, 1.0, 1.0), 0.0).keyword == "ease-in"


# --------------------------------------------------------------------------------------------
# Free bezier
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bez", [(0.3, 0.2, 0.1, 0.9), (0.7, 0.0, 0.3, 1.0), (0.05, 0.8, 0.5, 1.0), (0.3, 1.4, 0.6, 1.0)]
)
def test_free_fit_recovers_off_catalog_curves(bez: tuple[float, float, float, float]) -> None:
    x = np.linspace(0.02, 0.98, 30)
    p = np.array([ref_bezier(bez, float(v)) for v in x])
    fit = ez.fit_free_bezier(x, p)
    assert ez.curve_distance(fit.bezier, bez) < 0.004
    assert fit.sse < 1e-3
    assert 0.0 <= fit.bezier[0] <= 1.0 and 0.0 <= fit.bezier[2] <= 1.0


def test_free_fit_noisy_is_close() -> None:
    rng = np.random.default_rng(3)
    bez = (0.2, 0.7, 0.3, 1.0)
    x = np.linspace(0.02, 0.98, 25)
    p = np.array([ref_bezier(bez, float(v)) for v in x]) + rng.normal(0, 0.02, x.size)
    fit = ez.fit_free_bezier(x, p)
    assert ez.curve_distance(fit.bezier, bez) < 0.03
