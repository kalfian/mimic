"""Scene model + compositor for synthetic UI recordings (PLAN §11.2).

A scene is a tree of :class:`Node` boxes laid out in CSS px (children relative to their
parent's box). Every frame is composited from scratch in **premultiplied float32 RGBA** at
device resolution (``CSS px * pixel_ratio``):

* Each node owns a pre-rendered sprite: a coverage mask (anti-aliased by drawing at 4×
  supersampling and area-downsampling) plus, for images, an RGB texture.
* Sprites are placed with ``cv2.warpAffine`` (``INTER_LINEAR``) using the node's full CSS
  transform chain, so sub-pixel translate and scale are exact up to bilinear resampling.
* Transforms follow CSS individual properties: ``translate`` then ``scale`` about
  ``transform-origin`` — ``M = T(pos + origin + translate) · S(scale) · T(-origin)``.
  Children inherit the parent's matrix, so child values are *relative to the parent*.
* ``opacity`` < 1 renders the subtree into its own layer first (CSS group opacity), which keeps
  the blend between state A and B linear in the opacity value.
* ``clip_children`` = ``overflow: hidden`` (children masked by the node's coverage).
* ``clip_h`` = visible height in CSS px (``height`` animation with clipped content).
* ``shadow`` = one ``box-shadow`` layer ``(x, y, blur, spread, alpha)``, black, drawn below the
  node from its own coverage (Gaussian σ = blur / 2), transformed with the node.

Colors in the public API are ``(r, g, b)``; frames are BGR ``uint8``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .animate import AnimValue, Color, CursorPath, ShadowTuple, Timeline
from .cursor_sprite import CursorSprite, cursor_sprite

Shape = Literal["rect", "text", "image", "icon", "group"]

SUPERSAMPLE = 4
SPRITE_PAD = 2  # device px of transparent border around every sprite
FONT = cv2.FONT_HERSHEY_SIMPLEX

#: IR property -> Node attribute it animates.
PROP_ATTR: dict[str, str] = {
    "translateX": "tx",
    "translateY": "ty",
    "scale": "scale",
    "opacity": "opacity",
    "color": "fill",
    "background-color": "fill",
    "box-shadow": "shadow",
    "height": "clip_h",
}


def hex_to_rgb(h: str) -> Color:
    h = h.lstrip("#")
    return (float(int(h[0:2], 16)), float(int(h[2:4], 16)), float(int(h[4:6], 16)))


def rgb_to_hex(c: Color) -> str:
    return "#" + "".join(f"{int(round(min(max(v, 0.0), 255.0))):02X}" for v in c)


def _font_scale(text: str, font_px: float) -> float:
    """Hershey ``fontScale`` per output px so that glyph height equals ``font_px``."""
    (_, h1), _ = cv2.getTextSize(text or "A", FONT, 1.0, 1)
    return font_px / h1


def _thickness_css(font_px: float, weight: float) -> float:
    return weight * 1.6 * font_px / 14.0


def text_size(text: str, font_px: float, weight: float = 1.0) -> tuple[float, float, float]:
    """``(width, ascent, descent)`` in CSS px of Hershey text scaled to glyph height ``font_px``.

    Measured at 8× so stroke thickness is included the same way the sprite draws it."""
    k = 8.0
    thick = max(1, int(round(_thickness_css(font_px, weight) * k)))
    (w, h), base = cv2.getTextSize(text, FONT, _font_scale(text, font_px) * k, thick)
    return (w / k + thick / k, h / k, base / k + thick / (2 * k))


@dataclass
class Node:
    """A box in the scene tree. ``x, y`` are relative to the parent's box (CSS px)."""

    id: str
    shape: Shape = "rect"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    fill: Color = (255.0, 255.0, 255.0)
    radius: float = 0.0
    # text
    text: str = ""
    font_px: float = 12.0
    weight: float = 1.0
    # image / icon
    seed: int = 0
    points: tuple[tuple[float, float], ...] = ()  # icon polygon, normalized to the box
    # structure / behaviour
    children: list[Node] = field(default_factory=list)
    clip_children: bool = False
    origin: tuple[float, float] = (0.5, 0.5)  # transform-origin as box fractions
    pointer: bool = False  # ``cursor: pointer`` (hand cursor in "auto" style)
    # base values of animatable properties
    tx: float = 0.0
    ty: float = 0.0
    scale: float = 1.0
    opacity: float = 1.0
    shadow: ShadowTuple | None = None
    clip_h: float | None = None

    def __post_init__(self) -> None:
        if self.shape == "text" and (self.w <= 0 or self.h <= 0):
            tw, asc, desc = text_size(self.text, self.font_px, self.weight)
            self.w, self.h = math.ceil(tw) + 1.0, math.ceil(asc + desc) + 1.0
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"node {self.id}: needs a positive size")
        if self.shadow is not None and self.shadow[3] != 0:
            raise NotImplementedError("box-shadow spread is not supported by the synth renderer")

    def walk(self) -> Iterator[Node]:
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class Scene:
    """Everything needed to render one scenario (CSS px, seconds)."""

    width: int
    height: int
    background: Color
    static_nodes: list[Node]
    nodes: list[Node]
    timeline: Timeline
    cursor: CursorPath | None
    duration_s: float

    def __post_init__(self) -> None:
        self.timeline.validate()
        self._index: dict[str, tuple[Node, tuple[Node, ...]]] = {}
        for root in [*self.static_nodes, *self.nodes]:
            self._register(root, ())
        static_ids = {n.id for r in self.static_nodes for n in r.walk()}
        for tw in self.timeline.tweens:
            if tw.node not in self._index:
                raise ValueError(f"tween targets unknown node {tw.node!r}")
            if tw.node in static_ids:
                raise ValueError(f"tween targets static node {tw.node!r}")
            if tw.prop not in PROP_ATTR:
                raise ValueError(f"unsupported animated property {tw.prop!r}")

    def _register(self, node: Node, ancestors: tuple[Node, ...]) -> None:
        if node.id in self._index:
            raise ValueError(f"duplicate node id {node.id!r}")
        self._index[node.id] = (node, ancestors)
        for c in node.children:
            self._register(c, (*ancestors, node))

    def node(self, node_id: str) -> Node:
        return self._index[node_id][0]

    # ---- animated values ----------------------------------------------------------------

    def value(self, node: Node, attr: str, t: float) -> AnimValue | None:
        props = [p for p, a in PROP_ATTR.items() if a == attr]
        base = getattr(node, attr)
        for p in props:
            if self.timeline.for_prop(node.id, p):
                return self.timeline.value(node.id, p, base, t)
        return base

    def local_matrix(self, node: Node, t: float) -> np.ndarray:
        tx = float(self.value(node, "tx", t))  # type: ignore[arg-type]
        ty = float(self.value(node, "ty", t))  # type: ignore[arg-type]
        s = float(self.value(node, "scale", t))  # type: ignore[arg-type]
        ox, oy = node.origin[0] * node.w, node.origin[1] * node.h
        return _t(node.x + ox + tx, node.y + oy + ty) @ _s(s) @ _t(-ox, -oy)

    def world_matrix(self, node_id: str, t: float) -> np.ndarray:
        node, ancestors = self._index[node_id]
        m = np.eye(3)
        for a in (*ancestors, node):
            m = m @ self.local_matrix(a, t)
        return m

    def box(self, node_id: str, t: float, *, clipped: bool = True) -> tuple[float, ...]:
        """World AABB ``(x, y, w, h)`` in CSS px of the node's (visible) box at time ``t``.

        ``clipped`` intersects with ancestor ``clip_children`` boxes and applies ``clip_h``.
        """
        node, ancestors = self._index[node_id]
        h = node.h
        if clipped:
            ch = self.value(node, "clip_h", t)
            if ch is not None:
                h = min(h, float(ch))  # type: ignore[arg-type]
        x0, y0, x1, y1 = _aabb(self.world_matrix(node_id, t), node.w, h)
        if clipped:
            for a in ancestors:
                if a.clip_children:
                    ah = a.h
                    ach = self.value(a, "clip_h", t)
                    if ach is not None:
                        ah = min(ah, float(ach))  # type: ignore[arg-type]
                    cx0, cy0, cx1, cy1 = _aabb(self.world_matrix(a.id, t), a.w, ah)
                    x0, y0, x1, y1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)
        return (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))

    def pointer_at(self, x: float, y: float, t: float) -> bool:
        for node_id, (node, _) in self._index.items():
            if node.pointer:
                bx, by, bw, bh = self.box(node_id, t)
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    return True
        return False

    def state_key(self, t: float) -> tuple:
        """Hashable snapshot of every animated value + cursor at ``t`` (frame dedup)."""
        vals = tuple(
            (w.node, w.prop, self.timeline.value(w.node, w.prop, w.from_value, t))
            for w in self.timeline.tweens
        )
        cur = None
        if self.cursor is not None and self.cursor.visible:
            x, y = self.cursor.position(t)
            cur = (round(x, 4), round(y, 4), self.cursor_style(t))
        return (vals, cur)

    def cursor_style(self, t: float) -> str:
        assert self.cursor is not None
        if self.cursor.style != "auto":
            return self.cursor.style
        x, y = self.cursor.position(t)
        return "hand" if self.pointer_at(x, y, t) else "arrow"


# --------------------------------------------------------------------------------------------
# Matrix helpers (3x3 homogeneous, CSS px)
# --------------------------------------------------------------------------------------------


def _t(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _s(sx: float, sy: float | None = None) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sx if sy is None else sy, 0.0], [0.0, 0.0, 1.0]])


def _aabb(m: np.ndarray, w: float, h: float) -> tuple[float, float, float, float]:
    pts = m @ np.array([[0.0, w, 0.0, w], [0.0, 0.0, h, h], [1.0, 1.0, 1.0, 1.0]])
    return (pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max())


# --------------------------------------------------------------------------------------------
# Sprites
# --------------------------------------------------------------------------------------------


def rounded_rect_coverage(
    W: int, H: int, x0: float, y0: float, w: float, h: float, radius: float
) -> np.ndarray:
    """Exact-geometry anti-aliased coverage of a rounded rectangle (edge coordinates, px).

    Uses the rounded-box signed distance at pixel centers, ``cov = clip(0.5 - d, 0, 1)``:
    exact area for straight edges, so sub-pixel positions/sizes are preserved. (OpenCV's
    polygon rasterizer includes both endpoints and anti-aliases asymmetrically, which biases
    positions by ~0.1 px — not acceptable for ground truth.)"""
    radius = max(0.0, min(radius, w / 2.0, h / 2.0))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    px, py = xx + 0.5 - (x0 + w / 2.0), yy + 0.5 - (y0 + h / 2.0)
    qx = np.abs(px) - (w / 2.0 - radius)
    qy = np.abs(py) - (h / 2.0 - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    d = outside + np.minimum(np.maximum(qx, qy), 0.0) - radius
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def polygon_coverage(W: int, H: int, pts: np.ndarray, ss: int = SUPERSAMPLE) -> np.ndarray:
    """Coverage of a polygon (edge coordinates, px) by ``ss``×``ss`` point sampling, even-odd."""
    n = W * ss, H * ss
    ys, xs = np.mgrid[0 : n[1], 0 : n[0]].astype(np.float64)
    xs, ys = (xs + 0.5) / ss, (ys + 0.5) / ss
    inside = np.zeros(xs.shape, bool)
    p = np.asarray(pts, np.float64)
    for (xa, ya), (xb, yb) in zip(p, np.roll(p, -1, axis=0), strict=True):
        if ya == yb:
            continue
        cond = (ya > ys) != (yb > ys)
        xcross = xa + (ys - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (xs < xcross)
    return inside.reshape(H, ss, W, ss).mean(axis=(1, 3)).astype(np.float32)


@dataclass
class Sprite:
    """Device-resolution sprite. ``cov`` (H, W) float32 in 0..1, ``rgb`` (H, W, 3) BGR float32
    straight (non-premultiplied) color or ``None`` for flat-colored nodes."""

    cov: np.ndarray
    rgb: np.ndarray | None


def build_sprite(node: Node, r: float) -> Sprite:
    ss = SUPERSAMPLE
    pad = SPRITE_PAD
    W = int(math.ceil(node.w * r)) + 2 * pad
    H = int(math.ceil(node.h * r)) + 2 * pad
    if node.shape in ("rect", "image", "group"):
        cov = rounded_rect_coverage(W, H, pad, pad, node.w * r, node.h * r, node.radius * r)
    elif node.shape == "icon":
        pts = np.array(node.points, dtype=np.float64) * np.array([node.w, node.h]) * r + pad
        cov = polygon_coverage(W, H, pts)
    else:  # text: Hershey glyphs at 4x (OpenCV 5 putText needs 8-bit images)
        big = np.zeros((H * ss, W * ss), np.uint8)
        k = r * ss
        _, asc, _ = text_size(node.text, node.font_px, node.weight)
        fs = _font_scale(node.text, node.font_px) * k
        thick = max(1, int(round(_thickness_css(node.font_px, node.weight) * k)))
        org = (int(round(pad * ss + 0.5 * k)), int(round(pad * ss + asc * k)))
        cv2.putText(big, node.text, org, FONT, fs, 255, thick, cv2.LINE_AA)
        cov = cv2.resize(big.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
        cov = np.clip(cov, 0.0, 1.0)
    if node.shape == "group":
        cov[:] = 0.0  # invisible container (layout only)
    rgb = _texture(node, W, H, r) if node.shape == "image" else None
    return Sprite(cov=cov, rgb=rgb)


def _texture(node: Node, W: int, H: int, r: float) -> np.ndarray:
    """Procedural "photo": gradient + smooth noise + a few shapes (gives ORB/ECC features)."""
    rng = np.random.default_rng(node.seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c0 = np.array(rng.uniform(60, 200, 3), np.float32)
    c1 = np.array(rng.uniform(60, 200, 3), np.float32)
    g = ((xx / max(W, 1)) * 0.6 + (yy / max(H, 1)) * 0.4)[..., None]
    img = c0 * (1 - g) + c1 * g
    noise = rng.normal(0, 1, (max(H // 8, 2), max(W // 8, 2), 3)).astype(np.float32)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    img += noise * 14.0
    ss = SUPERSAMPLE
    for _ in range(9):
        col = tuple(float(v) for v in rng.uniform(20, 240, 3))
        layer = np.zeros((H * ss, W * ss), np.uint8)
        cx, cy = rng.uniform(0, W * ss), rng.uniform(0, H * ss)
        if rng.random() < 0.5:
            rad = rng.uniform(8, 40) * r * ss
            cv2.circle(layer, (int(cx), int(cy)), int(rad), 255, -1, cv2.LINE_AA)
        else:
            w2, h2 = rng.uniform(10, 60) * r * ss, rng.uniform(10, 60) * r * ss
            ang = rng.uniform(0, 180)
            box = cv2.boxPoints(((cx, cy), (w2, h2), ang))
            cv2.fillPoly(layer, [np.round(box).astype(np.int32)], 255, cv2.LINE_AA)
        a = cv2.resize(layer.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
        a = a[..., None] * 0.85
        img = img * (1 - a) + np.array(col, np.float32) * a
    return np.clip(img, 0, 255).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------------


def _center_conv(a_edge: np.ndarray) -> np.ndarray:
    """Edge-coordinate affine (pixel i spans [i, i+1)) -> OpenCV pixel-center convention."""
    return _t(-0.5, -0.5) @ a_edge @ _t(0.5, 0.5)


class Renderer:
    """Renders a :class:`Scene` at a device pixel ratio. Output frames are BGR ``uint8``."""

    def __init__(self, scene: Scene, pixel_ratio: int = 1) -> None:
        self.scene = scene
        self.r = float(pixel_ratio)
        self.W = int(round(scene.width * self.r))
        self.H = int(round(scene.height * self.r))
        self._sprites: dict[str, Sprite] = {}
        self._cursors: dict[str, CursorSprite] = {}
        bg = np.empty((self.H, self.W, 4), np.float32)
        bg[..., :3] = np.array(scene.background[::-1], np.float32)
        bg[..., 3] = 1.0
        for root in scene.static_nodes:
            self._draw(bg, root, _s(self.r), 0.0, None)
        self._background = bg
        self._last_key: tuple | None = None
        self._last_frame: np.ndarray | None = None

    # ---- public -------------------------------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        key = self.scene.state_key(t)
        if key == self._last_key and self._last_frame is not None:
            return self._last_frame
        layer = self._background.copy()
        for root in self.scene.nodes:
            self._draw(layer, root, _s(self.r), t, None)
        cur = self.scene.cursor
        if cur is not None and cur.visible:
            self._draw_cursor(layer, t)
        out = np.clip(layer[..., :3] + 0.5, 0, 255).astype(np.uint8)
        self._last_key, self._last_frame = key, out
        return out

    def frames(self, fps: float) -> Iterator[tuple[float, np.ndarray]]:
        n = int(math.floor(self.scene.duration_s * fps + 1e-9))
        for i in range(n):
            t = i / fps
            yield t, self.frame(t)

    # ---- internals ----------------------------------------------------------------------

    def _sprite(self, node: Node) -> Sprite:
        sp = self._sprites.get(node.id)
        if sp is None:
            sp = build_sprite(node, self.r)
            self._sprites[node.id] = sp
        return sp

    def _sprite_affine(self, device_m: np.ndarray) -> np.ndarray:
        """Edge-coordinate affine sprite px -> device px for a node with device matrix."""
        return device_m @ _s(1.0 / self.r) @ _t(-SPRITE_PAD, -SPRITE_PAD)

    def _roi(self, a_edge: np.ndarray, w: int, h: int, margin: float = 0.0):
        x0, y0, x1, y1 = _aabb(a_edge, w, h)
        x0 = max(int(math.floor(x0 - margin)) - 1, 0)
        y0 = max(int(math.floor(y0 - margin)) - 1, 0)
        x1 = min(int(math.ceil(x1 + margin)) + 1, self.W)
        y1 = min(int(math.ceil(y1 + margin)) + 1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _warp(self, src: np.ndarray, a_edge: np.ndarray, roi) -> np.ndarray:
        x0, y0, x1, y1 = roi
        a = _t(-x0, -y0) @ _center_conv(a_edge)
        return cv2.warpAffine(
            src,
            a[:2],
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _coverage(self, node: Node, t: float) -> np.ndarray:
        sp = self._sprite(node)
        ch = self.scene.value(node, "clip_h", t)
        if ch is None:
            return sp.cov
        rows = np.arange(sp.cov.shape[0], dtype=np.float32)
        lim = float(ch) * self.r + SPRITE_PAD  # type: ignore[arg-type]
        mult = np.clip(lim - rows, 0.0, 1.0)[:, None]
        return sp.cov * mult

    def _full_mask(self, cov: np.ndarray, a_edge: np.ndarray) -> np.ndarray:
        m = np.zeros((self.H, self.W), np.float32)
        roi = self._roi(a_edge, cov.shape[1], cov.shape[0])
        if roi is not None:
            x0, y0, x1, y1 = roi
            m[y0:y1, x0:x1] = self._warp(cov, a_edge, roi)
        return m

    def _draw(
        self,
        layer: np.ndarray,
        node: Node,
        parent_m: np.ndarray,
        t: float,
        clip: np.ndarray | None,
    ) -> None:
        sc = self.scene
        opacity = float(sc.value(node, "opacity", t))  # type: ignore[arg-type]
        if opacity <= 1e-6:
            return
        m = parent_m @ sc.local_matrix(node, t)
        target = layer
        if opacity < 1.0 - 1e-6:
            target = np.zeros_like(layer)

        cov = self._coverage(node, t)
        a_edge = self._sprite_affine(m)

        shadow = sc.value(node, "shadow", t)
        if shadow is not None:
            self._draw_shadow(target, cov, m, shadow, clip)  # type: ignore[arg-type]

        if node.shape != "group":
            roi = self._roi(a_edge, cov.shape[1], cov.shape[0])
            if roi is not None:
                x0, y0, x1, y1 = roi
                sp = self._sprite(node)
                if sp.rgb is not None:
                    src = np.dstack([sp.rgb * cov[..., None], cov]).astype(np.float32)
                    w = self._warp(src, a_edge, roi)
                    a = w[..., 3]
                    rgb = w[..., :3]
                else:
                    a = self._warp(cov, a_edge, roi)
                    fill = sc.value(node, "fill", t)
                    rgb = a[..., None] * np.array(fill[::-1], np.float32)  # type: ignore[index]
                if clip is not None:
                    c = clip[y0:y1, x0:x1]
                    a = a * c
                    rgb = rgb * c[..., None]
                _over(target, rgb, a, roi)

        child_clip = clip
        if node.clip_children and node.children:
            mask = self._full_mask(cov, a_edge)
            child_clip = mask if clip is None else mask * clip
        for c in node.children:
            self._draw(target, c, m, t, child_clip)

        if target is not layer:
            layer[..., :3] = target[..., :3] * opacity + layer[..., :3] * (
                1.0 - target[..., 3:4] * opacity
            )
            layer[..., 3] = target[..., 3] * opacity + layer[..., 3] * (
                1.0 - target[..., 3] * opacity
            )

    def _draw_shadow(
        self,
        layer: np.ndarray,
        cov: np.ndarray,
        m: np.ndarray,
        shadow: ShadowTuple,
        clip: np.ndarray | None,
    ) -> None:
        sx, sy, blur, _spread, alpha = shadow
        if alpha <= 0:
            return
        a_edge = self._sprite_affine(m @ _t(sx, sy))
        sigma = max(blur / 2.0 * self.r, 1e-3)
        roi = self._roi(a_edge, cov.shape[1], cov.shape[0], margin=3.0 * sigma + 2.0)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        s = self._warp(cov, a_edge, roi)
        if blur > 0:
            s = cv2.GaussianBlur(s, (0, 0), sigmaX=sigma, sigmaY=sigma)
        a = s * float(alpha)
        if clip is not None:
            a = a * clip[y0:y1, x0:x1]
        _over(layer, np.zeros(a.shape + (3,), np.float32), a, roi)

    def _draw_cursor(self, layer: np.ndarray, t: float) -> None:
        cur = self.scene.cursor
        assert cur is not None
        style = self.scene.cursor_style(t)
        sp = self._cursors.get(style)
        if sp is None:
            sp = cursor_sprite(style, self.r)
            self._cursors[style] = sp
        x, y = cur.position(t)
        # sprite px (edge coords) -> device: hotspot pixel lands on the cursor position
        a_edge = _t(x * self.r - sp.hot_x, y * self.r - sp.hot_y)
        h, w = sp.rgba.shape[:2]
        roi = self._roi(a_edge, w, h)
        if roi is None:
            return
        wv = self._warp(sp.rgba, a_edge, roi)
        _over(layer, wv[..., :3], wv[..., 3], roi)


def _over(layer: np.ndarray, rgb_premul: np.ndarray, a: np.ndarray, roi) -> None:
    x0, y0, x1, y1 = roi
    dst = layer[y0:y1, x0:x1]
    inv = 1.0 - a
    dst[..., :3] = rgb_premul + dst[..., :3] * inv[..., None]
    dst[..., 3] = a + dst[..., 3] * inv
