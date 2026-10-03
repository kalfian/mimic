"""Shadow + border-radius heuristics on rendered cards (Track M1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.measure import Rect
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.radius import corner_radius
from app.pipeline.shadow import estimate_shadow, is_shadow_change, shadow_change
from tests.unit.m1_helpers import composite, solid_rect_sprite, translate

BG = (238, 238, 238)


def card_with_shadow(oy: float, blur: float, alpha: float, radius: int = 0) -> np.ndarray:
    img = np.zeros((300, 360, 3), np.uint8)
    img[:] = BG
    spr = solid_rect_sprite(200, 140, (255, 255, 255), radius)
    if alpha > 0:
        sh = np.zeros((300, 360), np.float32)
        sh[80 : 80 + 140, 80 : 80 + 200] = 1.0
        sh = np.roll(sh, int(round(oy)), axis=0)
        if blur > 0:
            sh = cv2.GaussianBlur(sh, (0, 0), blur / 2)
        img = (img.astype(np.float32) * (1 - alpha * sh[..., None])).round().astype(np.uint8)
    composite(img, spr, translate(80, 80))
    return img


BOX = Rect(80, 80, 200, 140)


def test_estimate_shadow_offset_and_presence() -> None:
    sh = estimate_shadow(card_with_shadow(8, 16, 0.2), BOX, 1.0, P)
    assert sh is not None
    assert sh.y > 0 and 0 < sh.rgba[3] <= P.photometric.shadow_alpha_max
    assert estimate_shadow(card_with_shadow(0, 0, 0.0), BOX, 1.0, P) is None


def test_shadow_change_direction() -> None:
    a, b = card_with_shadow(4, 12, 0.08), card_with_shadow(12, 32, 0.25)
    sc = shadow_change(a, b, BOX, np.eye(2, 3), 1.0, P)
    assert is_shadow_change(sc, P) and sc.increases
    sc2 = shadow_change(b, a, BOX, np.eye(2, 3), 1.0, P)
    assert is_shadow_change(sc2, P) and not sc2.increases


@pytest.mark.parametrize("r", [8, 16, 24])
def test_corner_radius(r: int) -> None:
    img = np.zeros((300, 360, 3), np.uint8)
    img[:] = BG
    composite(img, solid_rect_sprite(200, 140, (90, 60, 200), r), translate(80, 80))
    est = corner_radius(img, BOX, 1.0, P)
    assert est is not None and est.corners >= 3
    assert est.radius_css == pytest.approx(r, abs=3.0)


def test_corner_radius_low_contrast_is_unknown() -> None:
    img = np.zeros((300, 360, 3), np.uint8)
    img[:] = BG
    composite(img, solid_rect_sprite(200, 140, (242, 242, 242), 12), translate(80, 80))
    assert corner_radius(img, BOX, 1.0, P) is None
