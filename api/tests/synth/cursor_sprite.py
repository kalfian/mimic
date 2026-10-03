"""Arrow / hand cursor sprites (PLAN §11.2): white fill, black outline, ~12×19 CSS px.

Sprites are premultiplied BGRA float32 at device resolution (``pixel_ratio`` ×), anti-aliased
by 4× supersampling. ``hot_x``/``hot_y`` is the hotspot in sprite edge coordinates (device px).
The path the cursor follows lives in :class:`synth.animate.CursorPath`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

_SS = 4
_PAD = 2

#: Polygons in CSS px, hotspot first.
ARROW: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 16.5),
    (4.0, 12.8),
    (6.6, 18.8),
    (9.2, 17.7),
    (6.7, 11.9),
    (11.8, 11.9),
)
ARROW_HOT = (0.0, 0.0)

HAND: tuple[tuple[float, float], ...] = (
    (5.0, 1.0),
    (6.5, 0.0),
    (8.0, 1.0),
    (8.0, 8.0),
    (10.0, 7.5),
    (12.0, 8.0),
    (14.0, 9.0),
    (15.5, 10.5),
    (15.5, 16.0),
    (13.5, 20.0),
    (5.5, 20.0),
    (1.0, 13.5),
    (1.0, 11.0),
    (3.0, 10.5),
    (5.0, 12.0),
)
HAND_HOT = (6.5, 0.0)


@dataclass(frozen=True)
class CursorSprite:
    rgba: np.ndarray  # (H, W, 4) premultiplied BGRA float32, alpha 0..1
    hot_x: float
    hot_y: float


def cursor_sprite(style: str, pixel_ratio: float) -> CursorSprite:
    if style == "hand":
        poly, hot = HAND, HAND_HOT
    elif style == "arrow":
        poly, hot = ARROW, ARROW_HOT
    else:
        raise ValueError(f"unknown cursor style {style!r}")
    r = float(pixel_ratio)
    pts = np.array(poly, dtype=np.float64)
    w = int(math.ceil(pts[:, 0].max() * r)) + 2 * _PAD
    h = int(math.ceil(pts[:, 1].max() * r)) + 2 * _PAD
    big_pts = (pts * r + _PAD) * _SS  # edge coordinates at 4x
    ys, xs = np.mgrid[0 : h * _SS, 0 : w * _SS].astype(np.float64) + 0.5
    inside = np.zeros(xs.shape, bool)
    for (xa, ya), (xb, yb) in zip(big_pts, np.roll(big_pts, -1, axis=0), strict=True):
        if ya != yb:
            inside ^= ((ya > ys) != (yb > ys)) & (xs < xa + (ys - ya) * (xb - xa) / (yb - ya))
    outer = inside.astype(np.uint8) * 255
    rad = max(1, int(round(1.0 * r * _SS)))  # 1 CSS px black outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    inner = cv2.erode(outer, kernel)
    a = cv2.resize(outer.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    white = cv2.resize(inner.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[..., :3] = (white * 255.0)[..., None]  # premultiplied: white part only, outline black
    rgba[..., 3] = a
    return CursorSprite(rgba=rgba, hot_x=_PAD + hot[0] * r, hot_y=_PAD + hot[1] * r)
