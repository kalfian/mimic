"""Panorama: loop period, card pitch, gap, grid phases (PLAN-continuous §4.4, §8.4).

The content strip mirrors the §8.2 synth geometry: 8 distinct procedural cards 200 px wide with
a 16 px gap (pitch 216), duplicated once for the loop (period 1728); viewport 760 px.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.pipeline.continuous import panorama as pano

CARD, GAP, N_CARDS, VIEW = 200, 16, 8, 760
PITCH = CARD + GAP
PERIOD = N_CARDS * PITCH


def strip(
    seed: int = 0, n_cards: int = N_CARDS, distinct: bool = True, layout: bool = False
) -> np.ndarray:
    """``layout=True``: same-layout cards (white body, 8 px inset image, title bar) whose images
    differ — globally they match at every multiple of the pitch (NCC > 0.9), like the synth
    suite's procedural cards."""
    rng = np.random.default_rng(seed)
    img = np.full((200, n_cards * PITCH), 229 if layout else 235, np.float32)
    shared = None
    for i in range(n_cards):
        x0 = i * PITCH + GAP // 2
        if layout:
            tex = cv2.GaussianBlur(rng.random((96, CARD - 16)).astype(np.float32), (0, 0), 4)
            tex = (tex - tex.min()) / (tex.max() - tex.min())
            card = np.full((160, CARD), 255, np.float32)
            card[8:104, 8 : CARD - 8] = 120 + 90 * tex
            card[116:126, 12:92] = 40
            img[20:180, x0 : x0 + CARD] = card
            continue
        if shared is None or distinct:
            tex = cv2.GaussianBlur(rng.random((160, CARD)).astype(np.float32), (0, 0), 4)
            tex = (tex - tex.min()) / (tex.max() - tex.min()) * 170 + 30
            shared = tex
        img[20:180, x0 : x0 + CARD] = shared
    return img


def frame(content: np.ndarray, pos: float) -> np.ndarray:
    """Viewport whose content is displaced by ``pos`` (+ = right), content repeating."""
    period = content.shape[1]
    big = np.concatenate([content] * 3, axis=1)
    off = (-pos) % period
    m = np.float32([[1, 0, -off], [0, 1, 0]])
    sh = cv2.warpAffine(big, m, (period + VIEW + 8, big.shape[0]), flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray(sh[:, :VIEW])


def run(speed: float, dur: float = 8.0, fps: float = 30.0, **kw) -> pano.PanoramaResult:
    content = strip(**kw)
    b = pano.PanoramaBuilder(1.0)
    for t in np.arange(0.0, dur, 1.0 / fps):
        pos = speed * t
        b.add(frame(content, pos), pos)
    return pano.analyse(b)


@pytest.mark.parametrize("speed", [240.0, -240.0, 500.0])
def test_loop_period_observed(speed: float) -> None:
    r = run(speed)
    assert r.loop_period_px is not None
    assert r.loop_period_px == pytest.approx(PERIOD, rel=0.01)
    assert 0 < r.loop_confidence <= 0.9
    assert r.pitch_px == pytest.approx(PITCH, abs=2.0)
    assert r.gap_px == pytest.approx(GAP, abs=2.0)


def test_loop_not_observed_with_short_travel() -> None:
    """C1-like: 60 px/s for 8 s = 480 px of travel < period → never guessed."""
    r = run(60.0)
    assert r.loop_period_px is None and r.loop_confidence == 0.0
    assert r.pitch_px == pytest.approx(PITCH, abs=2.0)
    assert r.travel_px == pytest.approx(480, abs=40)


def test_identical_cards_loop_at_pitch_and_no_pitch() -> None:
    """Identical cards: the content truly repeats every card, so the loop period is the pitch
    and no separate (smaller) pitch is reported."""
    r = run(240.0, distinct=False)
    assert r.loop_period_px == pytest.approx(PITCH, rel=0.01)
    assert r.pitch_px is None


def test_pitch_needs_several_periods() -> None:
    r = run(10.0, dur=2.0)  # 20 px of travel: one viewport ≈ 3.5 cards
    assert r.pitch_px is None or r.pitch_px == pytest.approx(PITCH, abs=2.0)
    assert r.loop_period_px is None


def test_grid_phases_locate_card_edges() -> None:
    """A card's leading edge sits at the viewport's leading edge when pos ≡ −GAP/2 (mod pitch)."""
    r = run(240.0)
    assert r.grid_phases
    lead = r.grid_phases[0]
    off = ((lead - (-GAP / 2)) + PITCH / 2) % PITCH - PITCH / 2
    assert abs(off) <= 2.0


def test_same_layout_cards_do_not_fake_a_loop() -> None:
    """Cards that share their layout match globally (NCC > 0.9 at multiples of the pitch) but
    not window by window: the loop is only reported at the true repeat."""
    r = run(240.0, layout=True)
    assert r.loop_period_px == pytest.approx(PERIOD, rel=0.01)
    r = run(60.0, layout=True)
    assert r.loop_period_px is None


def test_gap_excludes_inset_content_margins() -> None:
    """The strongest shared edges are the inset image's, 8 px inside each card; the gap is
    still the background between the cards."""
    r = run(240.0, layout=True)
    assert r.pitch_px == pytest.approx(PITCH, abs=2.0)
    assert r.gap_px is not None and r.gap_px == pytest.approx(GAP, abs=2.0)


def test_builder_keeps_at_most_max_contrib_per_column() -> None:
    b = pano.PanoramaBuilder(1.0)
    img = np.full((10, 50), 100.0, np.float32)
    for _ in range(10):
        b.add(img, 0.0)
    assert b.count is not None and int(b.count.max()) == b.c
    canvas, valid = b.canvas()
    assert canvas.shape == (10, 50) and valid.all()
