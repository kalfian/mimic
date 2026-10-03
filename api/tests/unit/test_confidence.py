"""Track M2 — confidence.py: factors, penalties, caps, bands (PLAN §6.8)."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ir import ColorValue, PxValue, RatioValue, ShadowValue, TextValue
from app.models.measure import PropertySeries
from app.pipeline import confidence as cf
from app.pipeline.easing import CANDIDATES_BY_NAME, solve_bezier
from app.pipeline.params import ConfidenceParams
from app.pipeline.timing import fit_series

P = ConfidenceParams()


def make_series(
    prop: str = "translateY",
    *,
    fv=None,  # type: ignore[no-untyped-def]
    tv=None,  # type: ignore[no-untyped-def]
    v_start: float = 0.0,
    v_end: float = -8.0,
    quality: float = 0.97,
    stable_std: float = 0.05,
    sigma: float = 0.0,
    fps: float = 60.0,
    dur: float = 0.28,
    unit: str = "px",
) -> PropertySeries:
    rng = np.random.default_rng(0)
    t = np.arange(0.5, 2.0, 1.0 / fps)
    p = solve_bezier(CANDIDATES_BY_NAME["ease-out"].bezier, (t - 1.0) / dur)
    values = v_start + (v_end - v_start) * p + rng.normal(0, sigma, t.size)
    return PropertySeries(
        element_id="e1",
        property=prop,  # type: ignore[arg-type]
        segment="fwd",
        times=t,
        values=values,
        quality=np.full(t.size, quality),
        v_start=v_start,
        v_end=v_end,
        unit=unit,  # type: ignore[arg-type]
        from_value=fv if fv is not None else PxValue(number=v_start),
        to_value=tv if tv is not None else PxValue(number=v_end),
        stable_std=stable_std,
    )


# --------------------------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.0, "low"), (0.49, "low"), (0.5, "medium"), (0.799, "medium"), (0.8, "high"), (1.0, "high")],
)
def test_bands(score: float, band: str) -> None:
    assert cf.band(score) == band
    assert cf.confidence(score).band == band


def test_confidence_helper_caps_and_rounds() -> None:
    c = cf.confidence(0.87654, cap=0.85)
    assert c.value == 0.85 and c.band == "high"
    c = cf.confidence(0.79996)
    assert c.value == 0.8 and c.band == "high"  # band from the rounded value
    assert cf.confidence(float("nan")).value == 0.0
    assert cf.confidence(1.7).value == 1.0


# --------------------------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------------------------


def test_q_track_formulas() -> None:
    assert cf.q_track_geometric(0.80) == 0.0
    assert cf.q_track_geometric(0.98) == pytest.approx(1.0)
    assert cf.q_track_geometric(0.89) == pytest.approx(0.5)
    assert cf.q_track_photometric(0.7) == 0.7
    assert cf.q_track_photometric(1.4) == 1.0
    s = make_series(quality=0.89)
    assert cf.series_q_track(s) == pytest.approx(0.5)
    color = make_series(
        "color",
        fv=ColorValue(color="#111111"),
        tv=ColorValue(color="#5252FF"),
        quality=0.9,
        v_end=1.0,
        unit="color",
    )
    assert cf.series_q_track(color) == pytest.approx(0.9)  # photometric: mean R² as-is
    s.quality = np.array([np.nan, np.nan])
    assert cf.series_q_track(s) == 0.5


def test_snr_factor_and_floors() -> None:
    assert cf.snr_factor(8.0, 0.1, 0.05) == pytest.approx(1.0)  # snr 80
    assert cf.snr_factor(1.0, 0.1, 0.05) == pytest.approx(0.5)  # snr 10
    assert cf.snr_factor(1.0, 0.0, 0.05) == pytest.approx(1.0)  # floor stops div-by-0
    assert cf.snr_factor(0.5, 0.0, 0.05) == pytest.approx(0.5)
    assert cf.snr_floor("translateX") == P.snr_floor_px
    assert cf.snr_floor("scale") == P.snr_floor_scale
    assert cf.snr_floor("background-color") == P.snr_floor_delta_e


def test_color_snr_uses_delta_e_units() -> None:
    def color(std: float) -> PropertySeries:
        return make_series(
            "color",
            fv=ColorValue(color="#111111"),
            tv=ColorValue(color="#5252FF"),
            v_end=1.0,
            stable_std=std,
            unit="color",
        )

    # ΔE ≈ 60: α-σ 0.01 -> σ_ΔE 0.6 < floor 1 -> snr 60 -> factor 1
    assert cf.series_snr_factor(color(0.01)) == pytest.approx(1.0)
    # α-σ 0.1 -> σ_ΔE ≈ 6 -> snr 10 -> factor 0.5
    assert cf.series_snr_factor(color(0.1)) == pytest.approx(0.5)


def test_fit_timing_easing_overall() -> None:
    assert cf.fit_factor(0.0) == 1.0 and cf.fit_factor(0.08) == 0.0
    assert cf.fit_factor(0.02) == pytest.approx(0.75)
    t = cf.timing_confidence(0.0, 1 / 60, 0.3, 1.0)
    assert t == pytest.approx(1 - (1 / 60) / 0.3)
    assert cf.timing_confidence(0.0, 0.05, 0.03, 1.0) == 0.0  # D shorter than a frame
    pen = cf.timing_confidence(0.0, 1 / 60, 0.3, 1.0, timestamps_estimated=True, is_vfr=True)
    assert pen == pytest.approx(t - 0.15 - 0.05)
    # easing: n_factor = (n-3)/12, sep = 0.4 + gap/0.03
    assert cf.easing_confidence(15, 0.0, 0.03) == pytest.approx(0.95)  # capped at easing_max
    assert cf.easing_confidence(9, 0.0, 0.0) == pytest.approx(0.5 * 0.4)
    assert cf.easing_confidence(3, 0.0, 1.0) == 0.0
    assert cf.easing_confidence(15, 0.0, float("inf")) == pytest.approx(0.95)
    assert cf.overall_confidence(0.8, 0.8, 0.8) == pytest.approx(0.8)
    assert cf.overall_confidence(1.0, 1.0, 0.0) == 0.0
    v = cf.value_confidence(0.9, 1.0, cursor_over_element=True)
    assert v == pytest.approx(0.85)


# --------------------------------------------------------------------------------------------
# Caps
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prop", "kind", "cap"),
    [
        ("translateY", "transform", 0.95),
        ("scale", None, 0.95),
        ("color", "photometric", 0.85),
        ("background-color", None, 0.85),
        ("opacity", "transform", 0.6),
        ("opacity", "appear", 0.8),
        ("opacity", "disappear", 0.8),
        ("opacity", "backdrop", 0.7),
        ("box-shadow", "transform", 0.45),
        ("border-radius", None, 0.5),
        ("height", "resize", 0.75),
        ("content", "content_change", 0.4),
        ("color", "content_change", 0.4),
    ],
)
def test_caps(prop: str, kind: str | None, cap: float) -> None:
    assert cf.cap_for(prop, kind) == cap  # type: ignore[arg-type]


def test_origin_and_static_confidence() -> None:
    assert cf.origin_confidence(0.9) == 0.7
    assert cf.origin_confidence(0.4) == 0.4
    assert cf.static_confidence(0.9, "box-shadow").value == 0.45
    assert cf.static_confidence(0.9, "border-radius").value == 0.5


def test_make_transition_confidence_applies_caps_and_band() -> None:
    tc = cf.make_transition_confidence(0.99, 0.99, 0.99, cap=0.45)
    assert tc.value == 0.45 and tc.overall == 0.45 and tc.band == "low"
    assert tc.timing <= 0.95 and tc.easing <= 0.95
    tc = cf.make_transition_confidence(0.9, 0.9, 0.9, cap=0.95)
    assert tc.overall == pytest.approx(0.9) and tc.band == "high"


# --------------------------------------------------------------------------------------------
# End-to-end on fitted series
# --------------------------------------------------------------------------------------------


def test_transition_confidence_clean_translate_is_high() -> None:
    s = make_series()
    fit = fit_series(s)
    assert fit is not None
    tc = cf.transition_confidence(s, fit, element_kind="transform")
    assert tc.band in ("high", "medium")
    assert tc.value > 0.8 and tc.timing > 0.8
    assert tc.overall == pytest.approx(
        min((tc.value * tc.timing * tc.easing) ** (1 / 3), 0.95), abs=2e-3
    )


def test_transition_confidence_degrades_with_noise_and_tracking() -> None:
    clean = make_series()
    noisy = make_series(sigma=0.6, stable_std=0.6, quality=0.86)
    fc, fn = fit_series(clean), fit_series(noisy)
    assert fc is not None and fn is not None
    c = cf.transition_confidence(clean, fc)
    n = cf.transition_confidence(noisy, fn)
    assert n.value < c.value and n.timing < c.timing and n.overall < c.overall
    pen = cf.transition_confidence(clean, fc, timestamps_estimated=True, is_vfr=True)
    assert pen.timing == pytest.approx(max(c.timing - 0.2, 0.0), abs=2e-3)


def test_transition_confidence_shadow_and_content_caps() -> None:
    shadow = make_series(
        "box-shadow",
        fv=ShadowValue(shadow=None),
        tv=ShadowValue(shadow=None),
        v_end=1.0,
        quality=0.95,
        unit="shadow",
    )
    fit = fit_series(shadow)
    assert fit is not None
    tc = cf.transition_confidence(shadow, fit, element_kind="transform")
    assert tc.overall <= 0.45 and tc.value <= 0.45 and tc.band == "low"
    content = make_series(
        "content",
        fv=TextValue(text="Add"),
        tv=TextValue(text="Added"),
        v_end=1.0,
        unit="text",
    )
    fit = fit_series(content)
    assert fit is not None
    assert cf.transition_confidence(content, fit, element_kind="content_change").overall <= 0.4
    opacity = make_series(
        "opacity",
        fv=RatioValue(number=0),
        tv=RatioValue(number=1),
        v_end=1.0,
        unit="ratio",
        stable_std=0.0,
    )
    fit = fit_series(opacity)
    assert fit is not None
    assert cf.transition_confidence(opacity, fit, element_kind="appear").overall <= 0.8
