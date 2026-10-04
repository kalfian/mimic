"""Measured appearance in state A (PLAN-continuous §13, Track A). Pure functions, no I/O.

What the coding LLM needs to make the rebuilt UI *look* like the recording (content stays
placeholder): per element the fill colour, the text colour of text-like elements and a rough font
size, plus the page background. Inputs are one state-A image and the element boxes in CSS px.

* **Background** (:func:`background_color`): dominant colour (Lab histogram mode, refined by a
  trimmed median) of an *inner ring* just inside the element border. The ring skips the outer
  ``ring_inset_css`` (anti-aliased edge / border), the four corner squares (rounded corners show
  the page there), child boxes (dilated) and text ink pixels.
* **Text colour** (:func:`analyse_text`): deterministic 2-means in Lab inside a text-like box;
  the majority cluster is the local background, the minority the ink. Anti-aliasing pulls the ink
  cluster towards the background, so the colour is the median of the ink pixels farthest from
  the background (``text_core_frac``). H.264 4:2:0 halves chroma resolution, so thin coloured
  strokes come back desaturated; the score drops with small text.
* **Font size** (:func:`font_size`): per text line, ink top to baseline (bottom of the dense
  x-height band) ≈ cap height; ``font_px ≈ cap_height / CAP_HEIGHT_EM``. All-lowercase text
  without ascenders reads ~30 % small, so the estimate is reported with low/medium confidence only
  (``FONT_SIZE_CONF_CAP``; the IR rejects a high band).
* **Page background** (:func:`page_background`): dominant colour of the pixels outside every
  element box (dilated by ``page_margin_css`` so drop shadows are skipped).

Scores are raw 0..1 values; ``pipeline.assemble.apply_appearance`` turns them into IR
confidences (colour cap = ``ConfidenceParams.cap_color``, font size cap ``FONT_SIZE_CONF_CAP``).

Geometry: the image covers the CSS rectangle starting at ``AppearanceImage.origin_css`` with
``k`` image px per CSS px (``Scale.k`` for pass-2 crops, ``css_to_img`` for a keyframe). Element
boxes are full-frame CSS px (``ElementCandidate.bbox_a``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.measure import ElementCandidate, Rect
from app.pipeline.photometric import bgr_to_lab, delta_e76, hex_from_lab

#: Cap height of common UI sans fonts in em (Inter 0.727, Roboto 0.711, Helvetica/Arial 0.716,
#: SF Pro 0.705, Segoe UI 0.70). ``font_px = cap_height_px / CAP_HEIGHT_EM``.
CAP_HEIGHT_EM = 0.72
#: Font size confidence never exceeds this (IR: ``font_size_px`` band at most medium).
FONT_SIZE_CONF_CAP = 0.6
#: Element kinds that are not visible in state A (nothing to sample).
NOT_IN_STATE_A = frozenset({"appear", "backdrop"})

_LAB_LO = np.array([0.0, -128.0, -128.0], np.float32)


@dataclass(frozen=True, slots=True)
class AppearanceParams:
    """Module-local tuning (``params.py`` is frozen in P1; P2 may move these into it)."""

    ring_inset_css: float = 2.0
    ring_width_frac: float = 0.2
    ring_width_min_css: float = 2.0
    ring_width_max_css: float = 12.0
    corner_frac: float = 0.25
    corner_max_css: float = 16.0
    child_margin_css: float = 2.0
    page_margin_css: float = 4.0
    #: Lab histogram bin edge for the colour mode, and the ΔE76 radius around it that counts as
    #: "the fill" (purity) — wide enough for H.264 chroma noise on flat fills.
    mode_bin: float = 4.0
    mode_tol_de: float = 6.0
    min_pixels: int = 30
    max_samples: int = 200_000
    bg_min_purity: float = 0.5
    page_min_purity: float = 0.35
    page_min_pixels: int = 400
    #: A child / text box whose fill matches its context within this ΔE76 is transparent.
    same_as_context_de: float = 3.0
    #: Ink must differ from the local background by at least this ΔE76.
    text_min_sep_de: float = 15.0
    text_min_frac: float = 0.01
    text_max_frac: float = 0.6
    text_core_frac: float = 0.35
    text_core_min: int = 5
    #: Cap height below this many image px is too small to measure.
    font_min_cap_px: float = 5.0
    #: Rows with at least this fraction of the line's densest row belong to the x-height band.
    font_dense_frac: float = 0.45


DEFAULT_APPEARANCE_PARAMS = AppearanceParams()


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppearanceImage:
    """A state-A image placed in CSS px.

    ``bgr`` uint8 (h, w, 3); ``origin_css`` = CSS position of pixel (0, 0); ``k`` = image px per
    CSS px; ``valid`` optional uint8/bool mask (non-zero = usable, e.g. ``StateFrames.valid``
    which excludes cursor-covered pixels).
    """

    bgr: np.ndarray
    origin_css: tuple[float, float] = (0.0, 0.0)
    k: float = 1.0
    valid: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class AppearanceElement:
    """One element to sample: state-A box (CSS px, full frame), kind, text flag, parent."""

    id: str
    box: Rect
    kind: str = "transform"
    text_like: bool = False
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class ColorEstimate:
    """A solid colour estimate. ``score`` is raw 0..1 (uncapped)."""

    hex: str
    lab: tuple[float, float, float]
    #: Fraction of the sampled pixels within ``mode_tol_de`` of the estimate.
    purity: float
    #: Median ΔE76 of those pixels to the estimate (noise / codec error).
    spread_de: float
    n: int
    score: float


@dataclass(frozen=True, slots=True)
class FontSizeEstimate:
    """Rough font size (CSS px) from the cap height of the text lines."""

    px: float
    cap_height_css: float
    lines: int
    score: float


@dataclass(frozen=True, slots=True)
class TextAnalysis:
    """Ink colour + ink mask of a text-like box (mask is box-local, image px)."""

    color: ColorEstimate
    background_lab: tuple[float, float, float]
    mask: np.ndarray  # bool (h, w): ink coverage >= 50 %
    origin_px: tuple[int, int]  # image px of mask[0, 0]


@dataclass(frozen=True, slots=True)
class ElementAppearance:
    background: ColorEstimate | None = None
    text: ColorEstimate | None = None
    font_size: FontSizeEstimate | None = None

    @property
    def empty(self) -> bool:
        return self.background is None and self.text is None and self.font_size is None


@dataclass(frozen=True, slots=True)
class AppearanceResult:
    """Output of :func:`measure_appearance` (input of ``assemble.apply_appearance``)."""

    viewport_css: tuple[float, float]
    page_background: ColorEstimate | None
    elements: dict[str, ElementAppearance] = field(default_factory=dict)


def elements_from_candidates(cands: Iterable[ElementCandidate]) -> list[AppearanceElement]:
    """Pipeline element candidates → :class:`AppearanceElement` (state-A boxes)."""
    return [
        AppearanceElement(
            id=c.id, box=c.bbox_a, kind=c.kind, text_like=c.text_like, parent_id=c.parent_id
        )
        for c in cands
    ]


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _clip01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0)) if math.isfinite(v) else 0.0


def _px_rect(r: Rect, img: AppearanceImage, grow_css: float = 0.0) -> tuple[int, int, int, int]:
    """CSS rect (optionally grown) → clipped image px ``(x0, y0, x1, y1)`` (may be empty)."""
    h, w = img.bgr.shape[:2]
    ox, oy = img.origin_css
    x0 = int(math.floor((r.x - grow_css - ox) * img.k + 1e-6))
    y0 = int(math.floor((r.y - grow_css - oy) * img.k + 1e-6))
    x1 = int(math.ceil((r.x2 + grow_css - ox) * img.k - 1e-6))
    y1 = int(math.ceil((r.y2 + grow_css - oy) * img.k - 1e-6))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _valid(img: AppearanceImage) -> np.ndarray:
    h, w = img.bgr.shape[:2]
    if img.valid is None:
        return np.ones((h, w), bool)
    return np.asarray(img.valid) > 0


def _contained(inner: Rect, outer: Rect) -> float:
    inter = inner.intersection(outer)
    return 0.0 if inter is None or inner.area <= 0 else inter.area / inner.area


def child_boxes(el: AppearanceElement, elements: Sequence[AppearanceElement]) -> list[Rect]:
    """Boxes to exclude from ``el``: its children by ``parent_id`` plus any smaller element box
    lying ≥ 90 % inside it (CV hierarchies are sometimes flat)."""
    out = []
    for o in elements:
        if o.id == el.id or o.box.area <= 0:
            continue
        if o.parent_id == el.id or (o.box.area < el.box.area and _contained(o.box, el.box) >= 0.9):
            out.append(o.box)
    return out


def _sample(lab: np.ndarray, max_n: int) -> np.ndarray:
    if len(lab) <= max_n:
        return lab
    step = int(math.ceil(len(lab) / max_n))
    return lab[::step]  # deterministic stride


# --------------------------------------------------------------------------------------------
# Colour mode
# --------------------------------------------------------------------------------------------


def _mode_center(lab: np.ndarray, bin_size: float) -> np.ndarray:
    """Centre of the most populated Lab histogram bin, counts smoothed over its 26 neighbours."""
    idx = np.floor((lab - _LAB_LO) / bin_size).astype(np.int64) + 1
    side = int(math.ceil(256.0 / bin_size)) + 3
    keys = (idx[:, 0] * side + idx[:, 1]) * side + idx[:, 2]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    smooth = np.zeros(len(uniq), np.int64)
    for dl in (-1, 0, 1):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                nb = uniq + (dl * side + da) * side + db
                pos = np.clip(np.searchsorted(uniq, nb), 0, len(uniq) - 1)
                hit = uniq[pos] == nb
                smooth += np.where(hit, counts[pos], 0)
    best = int(np.argmax(smooth))  # ties: lowest key (deterministic)
    # pixels of the best bin and its neighbours, weighted equally
    best_idx = np.array(np.unravel_index(int(uniq[best]), (side, side, side)))
    near = np.all(np.abs(idx - best_idx) <= 1, axis=1)
    return np.median(lab[near], axis=0) if near.any() else lab[inv == best].mean(axis=0)


def fill_score(purity: float, spread_de: float, n: int) -> float:
    """Raw confidence of a flat-fill colour: purity dominates, codec noise and tiny samples
    lower it."""
    p = _clip01((purity - 0.4) / 0.5)
    s = _clip01(1.0 - (spread_de - 1.5) / 6.0)
    m = _clip01(n / 100.0) ** 0.5
    return round(p * (0.6 + 0.4 * s) * m, 4)


def dominant_color(
    lab: np.ndarray, params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS
) -> ColorEstimate | None:
    """Dominant colour of Lab pixels ``(N, 3)``: histogram mode refined by a trimmed median."""
    lab = np.asarray(lab, np.float32).reshape(-1, 3)
    n_all = len(lab)
    if n_all < params.min_pixels:
        return None
    lab = _sample(lab, params.max_samples)
    c = _mode_center(lab, params.mode_bin)
    sel = np.zeros(len(lab), bool)
    for _ in range(3):
        d = np.linalg.norm(lab - c, axis=1)
        sel = d <= params.mode_tol_de
        if not sel.any():
            return None
        c = np.median(lab[sel], axis=0)
    d = np.linalg.norm(lab - c, axis=1)
    sel = d <= params.mode_tol_de
    purity = float(sel.mean())
    spread = float(np.median(d[sel])) if sel.any() else float("inf")
    return ColorEstimate(
        hex=hex_from_lab(c),
        lab=(float(c[0]), float(c[1]), float(c[2])),
        purity=round(purity, 4),
        spread_de=round(spread, 3),
        n=n_all,
        score=fill_score(purity, spread, n_all),
    )


# --------------------------------------------------------------------------------------------
# Background (inner ring)
# --------------------------------------------------------------------------------------------


def inner_ring_mask(
    el: Rect,
    img: AppearanceImage,
    children: Sequence[Rect] = (),
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
) -> np.ndarray:
    """Bool mask (image size) of the inner ring of ``el`` minus corners, children, invalid px."""
    h, w = img.bgr.shape[:2]
    mask = np.zeros((h, w), bool)
    side = min(el.w, el.h)
    if side <= 0:
        return mask
    inset = min(params.ring_inset_css, side * 0.15)
    width = min(
        max(side * params.ring_width_frac, params.ring_width_min_css), params.ring_width_max_css
    )
    width = min(width, max(side / 2 - inset, 0.5))
    outer = Rect(el.x + inset, el.y + inset, el.w - 2 * inset, el.h - 2 * inset)
    inner = Rect(outer.x + width, outer.y + width, outer.w - 2 * width, outer.h - 2 * width)
    ox0, oy0, ox1, oy1 = _px_rect(outer, img)
    if ox1 <= ox0 or oy1 <= oy0:
        return mask
    mask[oy0:oy1, ox0:ox1] = True
    if inner.w > 0 and inner.h > 0:
        ix0, iy0, ix1, iy1 = _px_rect(inner, img)
        # inner hole: shrink by one px so the ring keeps its full width after rounding
        mask[min(iy0 + 1, iy1) : max(iy1 - 1, iy0), min(ix0 + 1, ix1) : max(ix1 - 1, ix0)] = False
    corner = min(side * params.corner_frac, params.corner_max_css)
    if corner > 0:
        right, bottom = outer.x2 - corner, outer.y2 - corner
        for cx, cy in ((outer.x, outer.y), (right, outer.y), (outer.x, bottom), (right, bottom)):
            x0, y0, x1, y1 = _px_rect(Rect(cx, cy, corner, corner), img)
            mask[y0:y1, x0:x1] = False
    for c in children:
        x0, y0, x1, y1 = _px_rect(c, img, grow_css=params.child_margin_css)
        mask[y0:y1, x0:x1] = False
    return mask & _valid(img)


def background_color(
    img: AppearanceImage,
    el: Rect,
    children: Sequence[Rect] = (),
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
    *,
    lab: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> ColorEstimate | None:
    """Fill colour of the element at ``el`` (None when the ring is not a solid fill).

    ``exclude`` = extra bool mask (image size) of pixels to skip (e.g. text ink). When the ring
    is not a solid fill it is retried at half the width (down to ``ring_width_min_css``): inset
    media (a card image a few px inside the border) reach into a wide ring.
    """
    lab = bgr_to_lab(img.bgr) if lab is None else lab
    p = params
    while True:
        mask = inner_ring_mask(el, img, children, p)
        if exclude is not None:
            mask &= ~exclude
        if int(mask.sum()) >= p.min_pixels:
            est = dominant_color(lab[mask], p)
            if est is not None and est.purity >= p.bg_min_purity:
                return est
        side = min(el.w, el.h)
        width = min(max(side * p.ring_width_frac, p.ring_width_min_css), p.ring_width_max_css)
        if width <= p.ring_width_min_css + 1e-9:
            return None
        p = dataclasses.replace(p, ring_width_max_css=max(width / 2, p.ring_width_min_css))


def outer_ring_color(
    img: AppearanceImage,
    el: Rect,
    others: Sequence[Rect] = (),
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
    *,
    lab: np.ndarray | None = None,
) -> ColorEstimate | None:
    """Dominant colour just outside ``el`` (its visual context), skipping other boxes."""
    h, w = img.bgr.shape[:2]
    width = min(max(min(el.w, el.h) * params.ring_width_frac, params.ring_width_min_css),
                params.ring_width_max_css)  # fmt: skip
    mask = np.zeros((h, w), bool)
    x0, y0, x1, y1 = _px_rect(el, img, grow_css=params.ring_inset_css + width)
    mask[y0:y1, x0:x1] = True
    x0, y0, x1, y1 = _px_rect(el, img, grow_css=params.ring_inset_css)
    mask[y0:y1, x0:x1] = False
    for o in others:
        if o is el:
            continue
        ix0, iy0, ix1, iy1 = _px_rect(o, img)
        mask[iy0:iy1, ix0:ix1] = False
    mask &= _valid(img)
    if int(mask.sum()) < params.min_pixels:
        return None
    lab = bgr_to_lab(img.bgr) if lab is None else lab
    return dominant_color(lab[mask], params)


# --------------------------------------------------------------------------------------------
# Text colour + font size
# --------------------------------------------------------------------------------------------


def _two_means(px: np.ndarray, c_bg: np.ndarray, c_ink: np.ndarray, iters: int = 8):
    for _ in range(iters):
        d_bg = np.linalg.norm(px - c_bg, axis=1)
        d_ink = np.linalg.norm(px - c_ink, axis=1)
        ink = d_ink < d_bg
        if not ink.any() or ink.all():
            break
        new_bg, new_ink = np.median(px[~ink], axis=0), px[ink].mean(axis=0)
        if np.allclose(new_bg, c_bg) and np.allclose(new_ink, c_ink):
            break
        c_bg, c_ink = new_bg, new_ink
    d_bg = np.linalg.norm(px - c_bg, axis=1)
    d_ink = np.linalg.norm(px - c_ink, axis=1)
    return c_bg, c_ink, d_ink < d_bg


def text_score(sep_de: float, core_spread_de: float, n_core: int, cap_px: float | None) -> float:
    """Raw confidence of an ink colour: contrast, core consistency, enough (thick) ink."""
    sep = _clip01((sep_de - 10.0) / 30.0) ** 0.5
    cons = _clip01(1.0 - (core_spread_de - 2.0) / 12.0)
    n = _clip01(n_core / 40.0) ** 0.5
    size = 0.6 if cap_px is None else 0.4 + 0.6 * _clip01((cap_px - 5.0) / 10.0)
    return round(sep * (0.5 + 0.5 * cons) * n * size, 4)


def analyse_text(
    img: AppearanceImage,
    el: Rect,
    children: Sequence[Rect] = (),
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
    *,
    lab: np.ndarray | None = None,
) -> TextAnalysis | None:
    """Ink colour and ink mask of the text-like box ``el`` (None if no clear two-tone text)."""
    x0, y0, x1, y1 = _px_rect(el, img)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    lab = bgr_to_lab(img.bgr) if lab is None else lab
    box_lab = lab[y0:y1, x0:x1]
    use = _valid(img)[y0:y1, x0:x1].copy()
    for c in children:
        cx0, cy0, cx1, cy1 = _px_rect(c, img, grow_css=params.child_margin_css)
        use[max(cy0 - y0, 0) : max(cy1 - y0, 0), max(cx0 - x0, 0) : max(cx1 - x0, 0)] = False
    px = box_lab[use].reshape(-1, 3)
    if len(px) < params.min_pixels:
        return None
    bg = dominant_color(px, params)
    if bg is None:
        return None
    c_bg = np.asarray(bg.lab, np.float32)
    d = np.linalg.norm(px - c_bg, axis=1)
    far = d >= np.quantile(d, 0.95)
    c_ink = np.median(px[far], axis=0)
    if delta_e76(c_ink, c_bg) < params.text_min_sep_de:
        return None
    c_bg, c_ink, ink = _two_means(px, c_bg, c_ink)
    frac = float(ink.mean())
    if not (params.text_min_frac <= frac <= params.text_max_frac):
        return None
    # ink core: the ink pixels farthest from the background (AA edges lean towards it)
    d_ink = np.linalg.norm(px[ink] - c_bg, axis=1)
    n_core = max(params.text_core_min, int(math.ceil(params.text_core_frac * len(d_ink))))
    if len(d_ink) < params.text_core_min:
        return None
    core = px[ink][np.argsort(-d_ink, kind="stable")[:n_core]]
    color = np.median(core, axis=0)
    sep = delta_e76(color, c_bg)
    if sep < params.text_min_sep_de:
        return None
    spread = float(np.median(np.linalg.norm(core - color, axis=1)))
    # ink coverage >= 50 %: projection onto the background -> ink axis
    axis = (color - c_bg).astype(np.float32)
    t = ((box_lab - c_bg) @ axis) / float(axis @ axis)
    mask = (t >= 0.5) & use
    cap = _cap_height_px(mask, params)
    est = ColorEstimate(
        hex=hex_from_lab(color),
        lab=(float(color[0]), float(color[1]), float(color[2])),
        purity=round(
            float((np.linalg.norm(px[ink] - color, axis=1) <= params.mode_tol_de).mean()), 4
        ),  # fmt: skip
        spread_de=round(spread, 3),
        n=int(ink.sum()),
        score=text_score(sep, spread, len(core), None if cap is None else cap[0]),
    )
    return TextAnalysis(
        color=est,
        background_lab=(float(c_bg[0]), float(c_bg[1]), float(c_bg[2])),
        mask=mask,
        origin_px=(x0, y0),
    )


def _line_runs(rows_on: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    y = 0
    n = len(rows_on)
    while y < n:
        if not rows_on[y]:
            y += 1
            continue
        start = y
        while y < n and rows_on[y]:
            y += 1
        if runs and start - runs[-1][1] - 1 <= merge_gap:
            runs[-1] = (runs[-1][0], y - 1)
        else:
            runs.append((start, y - 1))
    return runs


def _line_caps(mask: np.ndarray, params: AppearanceParams) -> list[tuple[float, bool]]:
    """Per text line ``(cap height px, descenders seen)``."""
    prof = mask.sum(axis=1).astype(np.float64)
    if prof.max(initial=0) <= 0:
        return []
    rows_on = prof >= max(1.0, 0.02 * prof.max())
    raw = _line_runs(rows_on, 0)
    tallest = max(b - a + 1 for a, b in raw)
    # i/j dots and accents sit a pixel or two above their line: merge small gaps
    runs = _line_runs(rows_on, max(1, int(round(0.2 * tallest))))
    out = []
    for a, b in runs:
        p = prof[a : b + 1]
        dense = np.nonzero(p >= params.font_dense_frac * p.max())[0]
        if dense.size == 0:
            continue
        baseline = a + int(dense[-1])
        cap = float(baseline - a + 1)
        if cap < params.font_min_cap_px:
            continue
        out.append((cap, (b - baseline) >= 0.15 * cap))
    return out


def _cap_height_px(mask: np.ndarray, params: AppearanceParams) -> tuple[float, int, bool] | None:
    caps = _line_caps(mask, params)
    if not caps:
        return None
    return (
        float(np.median([c for c, _ in caps])),
        len(caps),
        any(desc for _, desc in caps),
    )


def font_size(
    text: TextAnalysis, k: float, params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS
) -> FontSizeEstimate | None:
    """Rough font size (CSS px) from the ink mask of :func:`analyse_text`."""
    cap = _cap_height_px(text.mask, params)
    if cap is None or k <= 0:
        return None
    cap_px, lines, descenders = cap
    # descenders below the dense band confirm where the baseline is; without them the line may
    # be all caps (right) or x-height only (~30 % small)
    base = 0.5 if descenders else 0.35
    size = _clip01((cap_px - params.font_min_cap_px) / 10.0)
    score = min(base * (0.6 + 0.4 * size) * (1.0 if lines == 1 else 1.1), FONT_SIZE_CONF_CAP)
    cap_css = cap_px / k
    return FontSizeEstimate(
        px=round(cap_css / CAP_HEIGHT_EM, 2),
        cap_height_css=round(cap_css, 2),
        lines=lines,
        score=round(score, 4),
    )


# --------------------------------------------------------------------------------------------
# Page background
# --------------------------------------------------------------------------------------------


def page_background(
    img: AppearanceImage,
    boxes: Sequence[Rect],
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
    *,
    lab: np.ndarray | None = None,
) -> ColorEstimate | None:
    """Dominant colour outside every element box (boxes grown by ``page_margin_css``)."""
    mask = _valid(img).copy()
    for b in boxes:
        x0, y0, x1, y1 = _px_rect(b, img, grow_css=params.page_margin_css)
        mask[y0:y1, x0:x1] = False
    if int(mask.sum()) < params.page_min_pixels:
        return None
    lab = bgr_to_lab(img.bgr) if lab is None else lab
    est = dominant_color(lab[mask], params)
    if est is None or est.purity < params.page_min_purity:
        return None
    return est


# --------------------------------------------------------------------------------------------
# Stage entry
# --------------------------------------------------------------------------------------------


def _depth(el: AppearanceElement, by_id: dict[str, AppearanceElement]) -> int:
    d, seen, cur = 0, {el.id}, el.parent_id
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        d += 1
        cur = by_id[cur].parent_id
    return d


def measure_appearance(
    image: AppearanceImage,
    elements: Sequence[AppearanceElement],
    viewport_css: tuple[float, float],
    *,
    page_image: AppearanceImage | None = None,
    params: AppearanceParams = DEFAULT_APPEARANCE_PARAMS,
) -> AppearanceResult:
    """Appearance of every element visible in state A, plus the page background.

    ``image`` should cover the elements at the best available resolution (the pass-2 state-A
    crop); ``page_image`` (default ``image``) should cover as much page as possible (the full
    state-A frame) for the page background.

    Transparency rule: a child or text-like element whose fill matches its context (the
    parent's fill, else the colour just outside the box) within ``same_as_context_de`` reports
    no background (it inherits). Root, non-text elements always report their fill.
    """
    lab = bgr_to_lab(image.bgr)
    page_img = image if page_image is None else page_image
    page_lab = lab if page_image is None else bgr_to_lab(page_img.bgr)
    visible = [e for e in elements if e.kind not in NOT_IN_STATE_A and e.box.area > 0]
    page = page_background(page_img, [e.box for e in elements if e.box.area > 0], params,
                           lab=page_lab)  # fmt: skip
    by_id = {e.id: e for e in visible}
    raw_bg: dict[str, ColorEstimate | None] = {}
    out: dict[str, ElementAppearance] = {}
    all_boxes = [e.box for e in visible]
    for el in sorted(visible, key=lambda e: (_depth(e, by_id), e.id)):
        children = child_boxes(el, visible)
        text = analyse_text(image, el.box, children, params, lab=lab) if el.text_like else None
        exclude = None
        if text is not None:
            exclude = np.zeros(image.bgr.shape[:2], bool)
            x0, y0 = text.origin_px
            mh, mw = text.mask.shape
            # grow the ink by one px: AA fringes are neither ink nor fill
            ink = cv2.dilate(text.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            exclude[y0 : y0 + mh, x0 : x0 + mw] = ink
        bg = background_color(image, el.box, children, params, lab=lab, exclude=exclude)
        raw_bg[el.id] = bg
        if bg is not None and (el.parent_id is not None or el.text_like):
            parent_bg = raw_bg.get(el.parent_id) if el.parent_id is not None else None
            context = parent_bg
            if context is None:
                context = outer_ring_color(image, el.box, all_boxes, params, lab=lab)
            if context is None and el.parent_id is None:
                context = page
            if context is not None and delta_e76(bg.lab, context.lab) < params.same_as_context_de:
                bg = None
        fs = font_size(text, image.k, params) if text is not None else None
        ea = ElementAppearance(
            background=bg, text=text.color if text is not None else None, font_size=fs
        )
        if not ea.empty:
            out[el.id] = ea
    ordered = {e.id: out[e.id] for e in elements if e.id in out}
    return AppearanceResult(
        viewport_css=(float(viewport_css[0]), float(viewport_css[1])),
        page_background=page,
        elements=ordered,
    )


__all__ = [
    "CAP_HEIGHT_EM",
    "DEFAULT_APPEARANCE_PARAMS",
    "FONT_SIZE_CONF_CAP",
    "AppearanceElement",
    "AppearanceImage",
    "AppearanceParams",
    "AppearanceResult",
    "ColorEstimate",
    "ElementAppearance",
    "FontSizeEstimate",
    "TextAnalysis",
    "analyse_text",
    "background_color",
    "child_boxes",
    "dominant_color",
    "elements_from_candidates",
    "font_size",
    "inner_ring_mask",
    "measure_appearance",
    "outer_ring_color",
    "page_background",
]
