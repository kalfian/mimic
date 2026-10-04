"""Card + title boxes inside a scroller, from one still frame (PLAN-continuous §13, P2).

The appearance block of a continuous spec needs the card box (size, fill, radius) and one text
line inside it (ink colour, font size). The phase pipeline already measured the pitch ``p`` and
the gap ``g`` (panorama); here one frame of the page (a rest frame when there is one) is enough:

1. **Card columns** along the axis (:func:`_gap_grid`): the track background is the colour of
   the band's flattest column; columns showing it on (nearly) every band row are gap columns;
   folded modulo the pitch, their longest run is the gap and its end the card start (the card
   length is ``p − g``). Fallbacks: the panorama's gap + flattest window, then
   :func:`panorama.gap_and_grid`. The first card lying fully inside the viewport is kept.
2. **Card extent across the axis**: rows (columns for a vertical scroller) where the card's
   inner lanes differ from the gap lanes (the track background) by more than
   :data:`CARD_ROW_DE`; the longest such run is the card.
3. **Viewport**: the moving band (changed pixels) only covers the cards; rows around it that
   still show the track background (the gap colour) across the scroller widen it to the
   ``overflow: hidden`` box. A side whose background runs into the crop margin stays at the band.
4. **Title line**: inside the card, ink = pixels far from the card fill (:data:`INK_DE`); text
   lines are ink row runs of plausible height with sparse ink (an inset image is a dense
   block and is skipped). The first line is the title.

**Census path (P2b)**: when the card census (``continuous.cards``) supplied the pitch — no
periodic pitch in the panorama, or cards that scale with their position — the pitch fold above
does not apply. :func:`find_card_by_census` segments the frame instead (gap columns = the track
background of the band's cross margins on every band row), keeps the card nearest the scroller
centre and reports it at the census size (the centre card's size when the cards scale), centred
where it is drawn.

Everything is CSS px, full frame. Pure (no I/O); the caller grabs the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.models.ir import Axis
from app.models.measure import Rect
from app.pipeline.continuous.panorama import FOLD_EPS, gap_and_grid
from app.pipeline.photometric import bgr_to_lab

#: The scroller crop is grown by this much across the axis so the track margins are visible.
CROSS_MARGIN_CSS = 48.0
#: Gap columns: at least ``GAP_ROW_QUANTILE`` of the band rows within ``GAP_DE`` ΔE76 of the
#: track background.
GAP_DE = 3.0
GAP_ROW_QUANTILE = 0.9
#: Rows within this ΔE76 of the track background (gap colour) belong to the viewport (track
#: and page backgrounds are often only a few ΔE apart: e.g. #E5E7EB on #F3F4F6 is ΔE ≈ 4.8).
VIEWPORT_DE = 3.0
#: A row belongs to the card when its card lanes differ from the gap lanes by this ΔE76.
CARD_ROW_DE = 6.0
#: Rows whose difference dips for at most this many CSS px still belong to the card.
CARD_ROW_BRIDGE_CSS = 3.0
#: Card lanes / gap lanes are sampled this far inside the card / gap edges (CSS px).
LANE_INSET_CSS = 3.0
#: Smallest card accepted (CSS px along and across the axis).
MIN_CARD_CSS = 24.0
#: Ink = ΔE76 from the card fill above this.
INK_DE = 20.0
#: Card border band ignored when looking for ink (rounded corners, edge anti-aliasing).
INK_BORDER_CSS = 3.0
#: Card fill = median of this band just inside the border (after ``INK_BORDER_CSS``).
FILL_RING_CSS = 3.0
#: Plausible text line height (CSS px) and the densest ink fraction a text line can have.
TEXT_MIN_H_CSS = 6.0
TEXT_MAX_H_CSS = 40.0
TEXT_MAX_DENSITY = 0.8
#: Ink row runs closer than this (CSS px) belong to one line (i/j dots, accents).
TEXT_ROW_BRIDGE_CSS = 2.0


@dataclass(frozen=True, slots=True)
class ScrollerLayout:
    """Boxes found in the frame (CSS px, full frame); ``None`` = not found."""

    card: Rect | None = None
    text: Rect | None = None
    #: The scroller viewport (the moving band grown over the visible track background).
    viewport: Rect | None = None


def _runs(on: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """``[start, end)`` runs of True, bridging gaps of ≤ ``bridge`` False samples."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        if out and i - out[-1][1] <= bridge:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return out


def _crop_px(r: Rect, k: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0 = max(0, int(math.floor(r.x * k)))
    y0 = max(0, int(math.floor(r.y * k)))
    x1 = min(w, int(math.ceil(r.x2 * k)))
    y1 = min(h, int(math.ceil(r.y2 * k)))
    return x0, y0, x1, y1


def _gap_grid(band_lab: np.ndarray, p_px: float) -> tuple[float, float] | None:
    """``(gap, card start)`` in px (start modulo the pitch, band column units) from colour.

    ``band_lab``: axis-major Lab of the moving band (the card rows). The track background is
    the colour of the band's flattest column (a gap); gap columns show it on (nearly) every
    band row, card columns do not (their fill differs, also in their plain margins). The
    gap is the longest circular run of pitch-folded bins that are mostly gap columns.
    """
    if band_lab.shape[0] < 4 or band_lab.shape[1] < p_px:
        return None
    std = band_lab[..., 0].std(axis=0)
    track = np.median(band_lab[:, int(np.argmin(std))], axis=0)
    de = np.linalg.norm(band_lab - track, axis=2)
    is_gap = np.quantile(de, GAP_ROW_QUANTILE, axis=0) < GAP_DE
    bins = max(2, int(round(p_px)))
    cols = np.arange(band_lab.shape[1], dtype=np.float64)
    idx = np.minimum(np.floor((cols % p_px) / p_px * bins + FOLD_EPS).astype(int), bins - 1)
    frac = np.bincount(idx, weights=is_gap.astype(np.float64), minlength=bins) / np.maximum(
        np.bincount(idx, minlength=bins), 1
    )
    on = frac > 0.5
    if not on.any() or on.all():
        return None
    # longest circular run of gap bins
    k0 = int(np.argmin(on))  # start scanning at a card bin so runs do not wrap
    rolled = np.roll(on, -k0)
    best = (0, 0)
    i = 0
    while i < bins:
        if rolled[i]:
            j = i
            while j < bins and rolled[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    g_bins = best[1] - best[0]
    g_px = g_bins * p_px / bins
    if g_bins == 0 or g_px >= 0.5 * p_px:
        return None
    start = ((best[1] + k0) % bins) * p_px / bins
    return g_px, start


def _card_start(img: np.ndarray, p_px: float, g_px: float) -> float:
    """Card start column (``img`` column units, modulo the pitch) for a known gap: the gap is
    the window of ``g_px`` columns whose (pitch-folded) variation across the axis is lowest."""
    bins = max(2, int(round(p_px)))
    cols = np.arange(img.shape[1], dtype=np.float64)
    idx = np.minimum(np.floor((cols % p_px) / p_px * bins + FOLD_EPS).astype(int), bins - 1)
    std = np.bincount(idx, weights=img.std(axis=0), minlength=bins) / np.maximum(
        np.bincount(idx, minlength=bins), 1
    )
    g = max(1, int(round(g_px * bins / p_px)))
    circ = np.concatenate([std, std[: g - 1]])
    win = np.convolve(circ, np.ones(g), mode="valid")[:bins]
    j = int(np.argmin(win))
    return ((j + g) * p_px / bins) % p_px


def _cross_rect(
    outer: Rect, axis: Axis, k: float, along: tuple[float, float], cross: tuple[float, float]
) -> Rect:
    """Axis-major px spans (along, cross) of the outer crop → CSS rect."""
    a_css = (outer.x if axis == "x" else outer.y) + along[0] / k
    c_css = (outer.y if axis == "x" else outer.x) + cross[0] / k
    la, lc = (along[1] - along[0]) / k, (cross[1] - cross[0]) / k
    return Rect(a_css, c_css, la, lc) if axis == "x" else Rect(c_css, a_css, lc, la)


def find_card_and_viewport(
    frame_bgr: np.ndarray,
    k: float,
    region_css: Rect,
    axis: Axis,
    pitch_css: float,
    gap_css: float | None,
) -> tuple[Rect | None, Rect | None]:
    """``(card, viewport)``: the first card fully inside ``region_css`` and the scroller
    viewport (``region_css`` grown across the axis over the track background), or None."""
    if pitch_css <= 0 or k <= 0:
        return None, None
    fh, fw = frame_bgr.shape[:2]
    frame_css = (fw / k, fh / k)
    grow = (0.0, CROSS_MARGIN_CSS) if axis == "x" else (CROSS_MARGIN_CSS, 0.0)
    outer = Rect(region_css.x - grow[0], region_css.y - grow[1], region_css.w + 2 * grow[0],
                 region_css.h + 2 * grow[1]).clamped(*frame_css)  # fmt: skip
    x0, y0, x1, y1 = _crop_px(outer, k, frame_bgr.shape)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    lab = bgr_to_lab(frame_bgr[y0:y1, x0:x1])
    # axis-major: columns run along the motion axis, rows across it
    if axis == "y":
        lab = np.ascontiguousarray(lab.transpose(1, 0, 2))
    gray = lab[..., 0]
    along0 = (region_css.x - outer.x) if axis == "x" else (region_css.y - outer.y)
    along_len = region_css.w if axis == "x" else region_css.h
    a0 = max(0, int(round(along0 * k)))
    a1 = min(gray.shape[1], int(round((along0 + along_len) * k)))
    p_px = pitch_css * k
    if a1 - a0 < 1.5 * p_px:
        return None, None
    cross_lo = (region_css.y - outer.y) if axis == "x" else (region_css.x - outer.x)
    cross_hi = cross_lo + (region_css.h if axis == "x" else region_css.w)
    b0 = max(0, int(round(cross_lo * k)))
    b1 = min(gray.shape[0], int(round(cross_hi * k)))
    valid = np.zeros(gray.shape[1], bool)
    valid[a0:a1] = True
    grid = _gap_grid(lab[b0:b1, a0:a1], p_px)
    if grid is not None:
        g_px, start = grid
        starts = [start + a0]
    else:
        # fallbacks: the panorama's gap + the flattest window (cross margins matter: gap columns
        # stay track background across them while the cards' own flat margins do not), else
        # the panorama's edge-peak grid on this frame
        g32 = np.ascontiguousarray(gray, dtype=np.float32)
        if gap_css is not None and gap_css * k >= 2.0:
            g_px = gap_css * k
            starts = [_card_start(g32[:, a0:a1], p_px, g_px) + a0]
        else:
            g_px, _, starts = gap_and_grid(g32, valid, p_px, 0)
            if not starts or g_px is None:
                return None, None
    card_len = p_px - g_px
    if card_len < MIN_CARD_CSS * k:
        return None, None
    s = starts[0] % p_px
    while s < a0:
        s += p_px
    if s + card_len > a1 + 1:
        return None, None
    # rows across the axis: card lanes vs gap lanes
    inset = max(1, int(round(LANE_INSET_CSS * k)))
    c_lo, c_hi = int(round(s)) + inset, int(round(s + card_len)) - inset
    if c_hi - c_lo < 2:
        return None, None
    card_med = np.median(lab[:, c_lo:c_hi], axis=1)
    # gap lanes before / after the card that lie inside the scroller (the viewport edge clips)
    lanes: list[np.ndarray] = []
    if g_px >= 2 * inset + 1:
        for g0 in (s - g_px, s + card_len):
            lo_, hi_ = int(round(g0)) + inset, int(round(g0 + g_px)) - inset
            if lo_ >= a0 and hi_ <= a1 and hi_ > lo_:
                lanes.append(lab[:, lo_:hi_])
    has_gap = bool(lanes)
    if has_gap:
        bg_med = np.median(np.concatenate(lanes, axis=1), axis=1)
    else:  # no visible gap: the track background is the cross margin's colour
        bg_med = np.repeat(np.median(lab[:2, c_lo:c_hi].reshape(-1, 3), axis=0)[None], len(lab), 0)
    diff = np.linalg.norm(card_med - bg_med, axis=1)
    runs = _runs(diff > CARD_ROW_DE, max(0, int(round(CARD_ROW_BRIDGE_CSS * k))))
    if not runs:
        return None, None
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    if (r1 - r0) / k < MIN_CARD_CSS:
        return None, None
    card = _cross_rect(outer, axis, k, (s, s + card_len), (r0, r1))
    viewport = None
    if has_gap:
        # the viewport: rows around the cards that still show the track background (the gap
        # colour) across the whole scroller; a side that runs into the crop edge is unknown
        track = np.median(bg_med[r0:r1], axis=0)
        rows = np.median(lab[:, a0:a1], axis=1)
        on = np.linalg.norm(rows - track, axis=1) < VIEWPORT_DE
        lo, hi = r0, r1
        while lo > 0 and on[lo - 1]:
            lo -= 1
        while hi < len(on) and on[hi]:
            hi += 1
        lo = r0 if lo == 0 else lo
        hi = r1 if hi == len(on) else hi
        lo, hi = min(lo, b0), max(hi, b1)
        viewport = _cross_rect(outer, axis, k, (a0, a1), (lo, hi))
    return card, viewport


@dataclass(frozen=True, slots=True)
class CensusCard:
    """Card size from the census (CSS px; the centre card when ``zoom_per_px2`` is set)."""

    length: float  # along the axis
    cross: float  # across it
    zoom_per_px2: float | None = None


#: Census path: the card nearest the centre must be within this fraction of the expected size.
CENSUS_SIZE_REL = 0.2
#: Census path: background = the band margin's median colour; ``tol`` ΔE76 = clip(4 σ, 2.5, 6).
CENSUS_TOL_MIN = 2.5
CENSUS_TOL_MAX = 6.0
#: Census path: a column is a gap when this fraction of the band rows are background.
CENSUS_GAP_ROWS = 0.95
#: Census path: margin rows sampled 1..MARGIN_ROWS CSS px outside the band.
CENSUS_MARGIN_CSS = 8.0


def find_card_by_census(
    frame_bgr: np.ndarray,
    k: float,
    region_css: Rect,
    axis: Axis,
    hint: CensusCard,
) -> tuple[Rect | None, Rect | None]:
    """``(card, viewport)`` by segmentation (census path, module docstring), or None."""
    if k <= 0 or hint.length <= 0 or hint.cross <= 0:
        return None, None
    fh, fw = frame_bgr.shape[:2]
    frame_css = (fw / k, fh / k)
    grow = (0.0, CROSS_MARGIN_CSS) if axis == "x" else (CROSS_MARGIN_CSS, 0.0)
    outer = Rect(region_css.x - grow[0], region_css.y - grow[1], region_css.w + 2 * grow[0],
                 region_css.h + 2 * grow[1]).clamped(*frame_css)  # fmt: skip
    x0, y0, x1, y1 = _crop_px(outer, k, frame_bgr.shape)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    lab = bgr_to_lab(frame_bgr[y0:y1, x0:x1])
    if axis == "y":
        lab = np.ascontiguousarray(lab.transpose(1, 0, 2))
    along0 = (region_css.x - outer.x) if axis == "x" else (region_css.y - outer.y)
    along_len = region_css.w if axis == "x" else region_css.h
    a0 = max(0, int(round(along0 * k)))
    a1 = min(lab.shape[1], int(round((along0 + along_len) * k)))
    cross_lo = (region_css.y - outer.y) if axis == "x" else (region_css.x - outer.x)
    cross_hi = cross_lo + (region_css.h if axis == "x" else region_css.w)
    b0 = max(0, int(round(cross_lo * k)))
    b1 = min(lab.shape[0], int(round(cross_hi * k)))
    if a1 - a0 < 8 or b1 - b0 < 4:
        return None, None
    m = max(2, int(round(CENSUS_MARGIN_CSS * k)))
    margin = [
        lab[max(0, b0 - m) : max(0, b0 - 1), a0:a1],
        lab[min(len(lab), b1 + 1) : b1 + m, a0:a1],
    ]
    px = np.concatenate([q.reshape(-1, 3) for q in margin]) if any(q.size for q in margin) else None
    if px is None or len(px) < 32:
        return None, None
    bg = np.median(px, axis=0)
    de_m = np.linalg.norm(px - bg, axis=1)
    tol = float(np.clip(4.0 * 1.4826 * float(np.median(de_m)), CENSUS_TOL_MIN, CENSUS_TOL_MAX))
    de = np.linalg.norm(lab - bg, axis=2)
    off = de > tol
    band = off[b0:b1, a0:a1]
    gap_col = (1.0 - band.mean(axis=0)) >= CENSUS_GAP_ROWS
    gaps = _runs(gap_col, 0)
    centre = (a1 - a0) / 2.0
    cands: list[tuple[float, int, int]] = []
    for (_, g1), (g2, _) in zip(gaps, gaps[1:], strict=False):
        if g2 - g1 >= MIN_CARD_CSS * k:
            cands.append((abs((g1 + g2) / 2.0 - centre), g1, g2))
    for dist, c0_, c1_ in sorted(cands):
        d_css = dist / k
        zoom = 1.0 + (hint.zoom_per_px2 or 0.0) * d_css * d_css
        want = hint.length * k * zoom
        if abs((c1_ - c0_) - want) > CENSUS_SIZE_REL * want:
            continue
        rows = off[:, a0 + c0_ : a0 + c1_].mean(axis=1) >= 0.5
        runs = _runs(rows, max(0, int(round(CARD_ROW_BRIDGE_CSS * k))))
        if not runs:
            continue
        r0, r1 = max(runs, key=lambda r: r[1] - r[0])
        want_c = hint.cross * k * zoom
        if abs((r1 - r0) - want_c) > CENSUS_SIZE_REL * want_c:
            continue
        # the census size, centred where this card is drawn
        ca, cc = a0 + (c0_ + c1_) / 2.0, (r0 + r1) / 2.0
        la, lc = hint.length * k, hint.cross * k
        card = _cross_rect(outer, axis, k, (ca - la / 2.0, ca + la / 2.0),
                           (cc - lc / 2.0, cc + lc / 2.0))  # fmt: skip
        # viewport: rows around the cards that still show the track background across the
        # scroller; a side that runs into the crop edge stays at the band
        rows_bg = (1.0 - off[:, a0:a1].mean(axis=1)) >= CENSUS_GAP_ROWS
        lo, hi = min(r0, b0), max(r1, b1)
        while lo > 0 and rows_bg[lo - 1]:
            lo -= 1
        while hi < len(rows_bg) and rows_bg[hi]:
            hi += 1
        lo = min(r0, b0) if lo == 0 else lo
        hi = max(r1, b1) if hi == len(rows_bg) else hi
        viewport = _cross_rect(outer, axis, k, (a0, a1), (lo, hi))
        return card, viewport
    return None, None


def find_card(
    frame_bgr: np.ndarray,
    k: float,
    region_css: Rect,
    axis: Axis,
    pitch_css: float,
    gap_css: float | None,
) -> Rect | None:
    """The first card fully inside ``region_css`` (see module docstring), or None."""
    return find_card_and_viewport(frame_bgr, k, region_css, axis, pitch_css, gap_css)[0]


def find_title(frame_bgr: np.ndarray, k: float, card: Rect) -> Rect | None:
    """First text line inside ``card`` (CSS px), or None."""
    x0, y0, x1, y1 = _crop_px(card, k, frame_bgr.shape)
    b = max(1, int(round(INK_BORDER_CSS * k)))
    if x1 - x0 <= 2 * b + 4 or y1 - y0 <= 2 * b + 4:
        return None
    lab = bgr_to_lab(frame_bgr[y0 + b : y1 - b, x0 + b : x1 - b])
    # the card fill: the thin band just inside the border (inset media can cover most of the
    # card, so the card-wide median would be the image)
    ring = max(1, int(round(FILL_RING_CSS * k)))
    band = np.concatenate(
        [lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
         lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)]
    )  # fmt: skip
    fill = np.median(band, axis=0)
    ink = np.linalg.norm(lab - fill, axis=2) > INK_DE
    rows = ink.mean(axis=1)
    lines = _runs(rows > 0, max(0, int(round(TEXT_ROW_BRIDGE_CSS * k))))
    for a, z in lines:
        h_css = (z - a) / k
        if not (TEXT_MIN_H_CSS <= h_css <= TEXT_MAX_H_CSS):
            continue
        cols = np.nonzero(ink[a:z].any(axis=0))[0]
        if cols.size == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        density = float(ink[a:z, c0:c1].mean())
        if density > TEXT_MAX_DENSITY:
            continue
        # one line: pad by a px so anti-aliased edges are inside the box
        pad = 1
        return Rect(
            (x0 + b + c0 - pad) / k, (y0 + b + a - pad) / k,
            (c1 - c0 + 2 * pad) / k, (z - a + 2 * pad) / k,
        )  # fmt: skip
    return None


def detect_layout(
    frame_bgr: np.ndarray,
    k: float,
    region_css: Rect,
    axis: Axis,
    pitch_css: float | None,
    gap_css: float | None,
    census: CensusCard | None = None,
) -> ScrollerLayout:
    """Card + title boxes (see module docstring). Needs the measured pitch, or the census card
    size (census path)."""
    if census is not None:
        card, viewport = find_card_by_census(frame_bgr, k, region_css, axis, census)
    elif pitch_css is None:
        return ScrollerLayout()
    else:
        card, viewport = find_card_and_viewport(frame_bgr, k, region_css, axis, pitch_css,
                                                gap_css)  # fmt: skip
    if card is None:
        return ScrollerLayout()
    return ScrollerLayout(card=card, text=find_title(frame_bgr, k, card), viewport=viewport)


__all__ = [
    "CensusCard",
    "ScrollerLayout",
    "detect_layout",
    "find_card",
    "find_card_and_viewport",
    "find_card_by_census",
    "find_title",
]
