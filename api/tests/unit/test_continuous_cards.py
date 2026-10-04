"""Card census (PLAN-continuous P2b, ``continuous/cards.py``) on numpy-drawn scroller bands.

Numbers only: dark page (= track) with low-contrast cards holding a noisy "photo", 4 px gaps,
optionally scaled by their distance from the scroller centre (the motivating recording's
layout); plus the census-vs-panorama choice and the census path of ``layout``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.measure import Rect
from app.pipeline.continuous import cards as C
from app.pipeline.continuous import measure as M
from app.pipeline.continuous import panorama as pano
from app.pipeline.continuous.layout import CensusCard, find_card_by_census

BG = 2.0
FILL = 22.0
W, H = 760, 240  # crop: the band is rows [BAND0, BAND1)
BAND0, BAND1 = 25, 215
CARD_W, CARD_H, GAP = 164.0, 150.0, 4.0


def _draw(pos: float, zoom_a: float = 0.0, seed: int = 0, n: int = 12) -> np.ndarray:
    """One crop (gray float32): cards of layout pitch ``CARD_W + GAP`` at content position
    ``pos``; with ``zoom_a`` each card is scaled by ``1 + a d²`` about its centre and the cards
    are packed outwards from the one nearest the centre (constant gaps, like the recording)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), BG, np.float64)
    centre = W / 2.0

    def scale(c: float) -> float:  # magnification, held constant beyond the viewport
        d = min(abs(c - centre), W / 2.0)
        return 1.0 + zoom_a * d * d

    p = CARD_W + GAP
    first = (pos % p) - 3 * p
    centres = [first + i * p + CARD_W / 2.0 for i in range(n)]
    if zoom_a:
        # pack outwards from the card nearest the centre with scaled widths and gaps
        k0 = int(np.argmin([abs(c - centre) for c in centres]))
        out = {k0: centres[k0]}
        for step in (1, -1):
            k = k0
            while 0 <= k + step < n:
                c = out[k]
                s_a = scale(c)
                guess = c + step * p * s_a
                s_b = scale(guess)
                out[k + step] = c + step * (CARD_W * (s_a + s_b) / 2.0 + GAP * (s_a + s_b) / 2.0)
                k += step
        centres = [out[k] for k in range(n)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    mid_y = (BAND0 + BAND1) / 2.0
    for i, c in enumerate(centres):
        s = scale(c)
        hw, hh = CARD_W * s / 2.0, CARD_H * s / 2.0
        # exact area coverage of the card rectangle per pixel (anti-aliased edges)
        cx = np.clip(np.minimum(xx + 0.5, c + hw) - np.maximum(xx - 0.5, c - hw), 0.0, 1.0)
        cy = np.clip(np.minimum(yy + 0.5, mid_y + hh) - np.maximum(yy - 0.5, mid_y - hh), 0, 1)
        cov = cx * cy
        tex = FILL + 60.0 * (rng.random((H, W)) if i % 3 else 0.3 * rng.random((H, W)))
        inner = (np.abs(xx - c) < hw - 8) & (np.abs(yy - mid_y) < hh - 8)
        val = np.where(inner, tex, FILL)
        img = img * (1.0 - cov) + val * cov
    return img.astype(np.float32)


def _census(zoom_a: float = 0.0, frames: int = 24) -> C.CensusResult | None:
    census = C.CardCensus(1.0)
    box = (0, BAND0, W, BAND1)
    for i in range(frames):
        pos = 7.3 * i
        f = _draw(pos, zoom_a, seed=i)
        census.add(f[BAND0:BAND1], C.band_margins(f, box, "x"), pos)
    return census.result((0.0, float(W)))


def test_rigid_dark_cards() -> None:
    r = _census()
    assert r is not None and not r.zoomed
    assert r.card_len == pytest.approx(CARD_W, abs=1.0)
    assert r.card_cross == pytest.approx(CARD_H, abs=1.0)
    assert r.gap == pytest.approx(GAP, abs=1.0)
    assert r.pitch == pytest.approx(CARD_W + GAP, abs=0.5)
    assert r.pitch_confidence >= 0.8 and r.gap_confidence >= 0.8
    # grid phases: a card's leading edge sits at the scroller's start every pitch
    assert len(r.grid_phases) == 2


def test_cards_scaling_with_position() -> None:
    a = 2.4e-6  # the recording: ×1.17 at 270 px from the centre
    r = _census(zoom_a=a)
    assert r is not None and r.zoomed
    assert r.card_len == pytest.approx(CARD_W, abs=1.5)
    assert r.card_cross == pytest.approx(CARD_H, abs=1.5)
    assert r.zoom_per_px2 == pytest.approx(a, rel=0.2)
    assert r.zoom_observed > 1.1
    assert r.gap == pytest.approx(GAP, abs=1.0)
    assert r.pitch == pytest.approx(r.card_len + r.gap)
    assert r.pitch_confidence <= C.ZOOM_CONF_CAP and r.grid_phases == ()
    # the scale itself (P2c): clear growth over many cards -> confident, uncapped here
    assert r.zoom_confidence > C.ZOOM_CONF_CAP
    assert r.zoom_extent > 200.0
    assert _census().zoom_confidence == 0.0  # type: ignore[union-attr]


def test_clipped_cards_at_the_scroller_ends_are_ignored() -> None:
    census = C.CardCensus(1.0)
    box = (0, BAND0, W, BAND1)
    for i in range(12):
        f = _draw(11.0 * i, seed=i)
        census.add(f[BAND0:BAND1], C.band_margins(f, box, "x"), 11.0 * i)
    # a span narrower than the crop clips cards at both ends: they never count
    r = census.result((100.0, 660.0))
    assert r is not None
    assert r.card_len == pytest.approx(CARD_W, abs=1.0)


def test_background_without_margins_uses_flat_columns() -> None:
    f = _draw(0.0)
    bg = C.background(f[BAND0:BAND1], None)
    assert bg is not None and bg[0] == pytest.approx(BG, abs=0.5)
    noisy = np.random.default_rng(1).uniform(0, 255, 400)
    assert C.background(f[BAND0:BAND1], noisy)[0] == pytest.approx(BG, abs=0.5)  # type: ignore[index]


def test_segment_needs_both_card_edges() -> None:
    f = _draw(0.0)
    cards, gaps = C.segment(f[BAND0:BAND1], BG, 3.0, 1.0)
    assert cards and all(0 < a and b < W for a, b, _, _ in cards)
    assert len(gaps) == len(cards) - 1


def _pres(pitch: float | None, loop: float | None = None) -> pano.PanoramaResult:
    return pano.PanoramaResult(loop, 0.9, 0.9, pitch, 0.7, 16.0 if pitch else None, 0.5, [1.0],
                               0.0)  # fmt: skip


def _cres(pitch: float, zoom: float | None = None, conf: float = 0.7) -> C.CensusResult:
    return C.CensusResult(pitch - 4.0, 150.0, 4.0, pitch, conf, conf, zoom, 1.0, 200.0, 40, 20)


def test_card_layout_prefers_the_panorama_for_rigid_periodic_cards() -> None:
    out = M.card_layout(_pres(216.0), _cres(216.0))
    assert out.pitch_px == 216.0 and out.card_len_px is None and out.grid_phases == [1.0]


@pytest.mark.parametrize(
    ("pano_pitch", "zoom"),
    [(None, None), (216.0, 2e-6), (336.0, None)],  # no pitch / cards scale / a multiple
)
def test_card_layout_uses_the_census(pano_pitch: float | None, zoom: float | None) -> None:
    out = M.card_layout(_pres(pano_pitch), _cres(168.0, zoom))
    assert out.pitch_px == 168.0 and out.gap_px == 4.0 and out.card_len_px == 164.0
    assert out.card_zoom_per_px2 == zoom
    # the scale's reach / confidence ride along only when the cards scale
    assert out.card_zoom_reach_px == (200.0 if zoom else None)
    assert out.card_zoom_confidence == 0.0


def test_card_layout_never_reports_a_pitch_at_the_loop_period() -> None:
    out = M.card_layout(_pres(None, loop=168.5), _cres(168.0))
    assert out.pitch_px is None


def test_layout_census_path_finds_the_centre_card() -> None:
    a = 2.4e-6
    gray = _draw(40.0, a)
    frame = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    region = Rect(0.0, float(BAND0), float(W), float(BAND1 - BAND0))
    card, viewport = find_card_by_census(frame, 1.0, region, "x", CensusCard(CARD_W, CARD_H, a))
    assert card is not None and viewport is not None
    assert (card.w, card.h) == (CARD_W, CARD_H)  # the census size, centre card
    assert abs(card.x + card.w / 2.0 - W / 2.0) <= (CARD_W + GAP) / 2.0 + 2.0
    assert card.y + card.h / 2.0 == pytest.approx((BAND0 + BAND1) / 2.0, abs=1.5)
