"""Card census: cards segmented frame by frame (PLAN-continuous P2b).

The panorama's pitch (``panorama.pitch``: edge-energy periodicity of the stitched strip) needs
rigid, periodic content. The motivating recording breaks both assumptions: photo-filled cards
whose inner edges outweigh the 4 px gaps, and cards that **scale with their distance from the
scroller centre** (≈164 px at the centre, ≈192 px near the edges, gaps scale along), so the
stitched canvas smears and even single frames are not periodic. Here every sampled frame of
the moving band is segmented on its own:

1. **Track background** ``bg``: the band's cross-axis margins just outside the moving box (the
   scroller's own background between the cards and its edge), else the most common level of
   the flat columns.
2. **Gap columns**: within ``tol`` of ``bg`` (``tol`` from the margin noise, see
   :func:`gap_tolerance`) on ≥ :data:`GAP_ROW_FRAC` of the band rows. A **card** is a run of
   other columns between two gap runs, both of its edges inside the crop, at least
   :data:`MIN_CARD_CSS` long.
3. **Cross extent**: the longest run of rows where ≥ :data:`CROSS_FILL` of the card's columns
   differ from ``bg``.
4. **Aggregate** (:meth:`CardCensus.result`): card length / cross size against the distance
   ``d`` of the card centre from the band centre, fitted as ``L0 + b·d²`` on per-distance-bin
   medians. When the fit grows by more than :data:`ZOOM_MIN_REL` over the observed range (and
   the cross size agrees) the cards **scale with position**: the reported card is the one at
   the centre (``L0 × C0``), the pitch ``L0 + gap``. Otherwise sizes are medians and the pitch
   is the median centre spacing of neighbouring cards. The gap is the median gap run (near the
   centre when zoomed).

Pure numpy; the orchestrator (``continuous.measure``) feeds frames, ``continuous.layout`` uses
the result to find the card box in the rest frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Gap columns: within ``tol`` of the background on at least this fraction of the band rows.
GAP_ROW_FRAC = 0.95
#: ``tol`` = clip(GAP_TOL_SIGMAS · σ_margin, GAP_TOL_MIN, GAP_TOL_MAX) gray levels.
GAP_TOL_SIGMAS = 4.0
GAP_TOL_MIN = 3.0
GAP_TOL_MAX = 8.0
#: Margin rows sampled just outside the moving box (analysis px): ``[MARGIN_SKIP, MARGIN_ROWS)``
#: away from it, so anti-aliased card edges and the box padding stay out.
MARGIN_SKIP = 1
MARGIN_ROWS = 8
#: A margin is usable when it is flat: robust σ ≤ MARGIN_MAX_SIGMA gray levels.
MARGIN_MAX_SIGMA = 4.0
#: Fallback background: columns whose std across the band is ≤ FLAT_STD are flat.
FLAT_STD = 2.5
#: Smallest card accepted (CSS px along the axis and across it).
MIN_CARD_CSS = 24.0
#: A row belongs to the card when this fraction of the card's columns differ from ``bg``.
CROSS_FILL = 0.5
#: Rows whose fill dips for at most this many analysis px still belong to the card.
CROSS_BRIDGE_PX = 2
#: Cards closer than this (analysis px) to the scroller's ends along the axis are clipped.
EDGE_MARGIN_PX = 4.0
#: Aggregation needs this many cards from this many frames.
MIN_CARDS = 6
MIN_FRAMES = 3
#: Distance bins (CSS px) for the zoom fit: max(ZOOM_BIN_MIN_CSS, card / ZOOM_BIN_DIV).
ZOOM_BIN_MIN_CSS = 8.0
ZOOM_BIN_DIV = 4.0
#: Cards scale with position when the fitted size grows by more than this fraction between
#: the centre and the farthest observed card (and the cross size grows by at least half that).
ZOOM_MIN_REL = 0.03
#: Card sizes off the model by more than this fraction are outliers (split / merged cards).
OUTLIER_REL = 0.15
#: Confidence: full with ≥ CONF_FULL_CARDS cards; spread (robust σ of the size residuals over the
#: card size) of SPREAD_ZERO_REL gives 0. Zoomed cards are capped at ZOOM_CONF_CAP (one size
#: stands for a range of sizes).
CONF_FULL_CARDS = 24
SPREAD_ZERO_REL = 0.05
ZOOM_CONF_CAP = 0.6
#: Scale confidence: the growth over the observed range counts fully from
#: ``ZOOM_MIN_REL · (1 + ZOOM_GROW_SPAN)`` on (a growth just above the threshold is weak evidence).
ZOOM_GROW_SPAN = 1.0
#: Gap confidence = pitch confidence × the fraction of gap observations within this many CSS
#: px of the median (the §14 acceptance tolerance).
GAP_TARGET_PX = 2.0
#: Grid phases (snap evidence) only for rigid cards with a circular resultant at least this.
GRID_MIN_R = 0.9


@dataclass(frozen=True, slots=True)
class CardObs:
    """One card in one frame (analysis px, band image coordinates)."""

    pos: float  # content position of the frame (CSS px, ``pos`` units of the tracker)
    start: float  # leading edge column
    end: float  # trailing edge column (exclusive)
    cross0: float
    cross1: float


@dataclass(frozen=True, slots=True)
class GapObs:
    width: float  # analysis px
    spacing: float  # centre-to-centre distance of the two cards (analysis px)
    mid: float  # gap centre column


@dataclass(frozen=True, slots=True)
class CensusResult:
    """Aggregated card geometry (CSS px)."""

    card_len: float  # along the axis (the centre card when zoomed)
    card_cross: float  # across the axis
    gap: float
    pitch: float
    pitch_confidence: float
    gap_confidence: float
    #: Relative size change per CSS px² of distance from the band centre (``b / L0``); None =
    #: rigid cards (no position dependence measured).
    zoom_per_px2: float | None
    #: Size relative to the centre card at the farthest observed card centre (1.0 = rigid).
    zoom_observed: float
    #: Distance (CSS px) of the farthest observed card centre from the band centre.
    zoom_extent: float
    cards: int
    frames: int
    #: Content positions (CSS px) at which a card's leading edge / centre sits at the band's
    #: leading edge / centre, modulo the pitch (``panorama.PanoramaResult.grid_phases``).
    grid_phases: tuple[float, ...] = ()
    #: Confidence of the position-dependent scale (0 when rigid), uncapped: card count × size
    #: spread around the model × how clearly the growth clears :data:`ZOOM_MIN_REL`.
    zoom_confidence: float = 0.0

    @property
    def zoomed(self) -> bool:
        return self.zoom_per_px2 is not None


def _runs(on: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """``[start, end)`` runs of True, bridging gaps of ≤ ``bridge`` False samples."""
    out: list[tuple[int, int]] = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    for a, b in zip(idx[::2], idx[1::2], strict=True):
        if out and a - out[-1][1] <= bridge:
            out[-1] = (out[-1][0], int(b))
        else:
            out.append((int(a), int(b)))
    return out


def gap_tolerance(margin_sigma: float) -> float:
    return float(np.clip(GAP_TOL_SIGMAS * margin_sigma, GAP_TOL_MIN, GAP_TOL_MAX))


def background(img: np.ndarray, margins: np.ndarray | None) -> tuple[float, float] | None:
    """``(bg level, robust σ)`` of the track background, or None.

    ``margins``: pixels just outside the moving box across the axis (any shape). Used when
    there are enough of them and they are flat; else the most common level of the band's
    flat columns (std across the band ≤ :data:`FLAT_STD`)."""
    if margins is not None and margins.size >= 32:
        m = margins.astype(np.float64).ravel()
        med = float(np.median(m))
        sig = 1.4826 * float(np.median(np.abs(m - med)))
        if sig <= MARGIN_MAX_SIGMA:
            return med, sig
    if img.shape[0] < 4:
        return None
    std = img.std(axis=0)
    flat = std <= FLAT_STD
    if np.count_nonzero(flat) < 2:
        return None
    levels = np.median(img[:, flat], axis=0)
    hist = np.bincount(np.clip(np.round(levels / 2.0).astype(int), 0, 128), minlength=129)
    mode = float(np.argmax(hist) * 2.0)
    near = levels[np.abs(levels - mode) <= 2.0]
    return float(np.median(near)), float(np.median(std[flat]))


def segment(
    img: np.ndarray, bg: float, tol: float, k: float
) -> tuple[list[tuple[float, float, float, float]], list[GapObs]]:
    """Cards ``(start, end, cross0, cross1)`` and gaps between neighbouring cards in one
    axis-major band image (rows across the axis, columns along it; analysis px)."""
    h, w = img.shape
    if h < 4 or w < 8:
        return [], []
    off = np.abs(img - bg) > tol
    gap_col = (1.0 - off.mean(axis=0)) >= GAP_ROW_FRAC
    min_len = MIN_CARD_CSS * k
    cards: list[tuple[float, float, float, float]] = []
    gaps: list[GapObs] = []
    gap_runs = _runs(gap_col)
    prev_run = (-1, -1)
    for (_, ga1), (gb0, _) in zip(gap_runs, gap_runs[1:], strict=False):
        a, b = ga1, gb0  # card columns between two gap runs (edges seen on both sides)
        if b - a < min_len:
            continue
        rows = off[:, a:b].mean(axis=1) >= CROSS_FILL
        rr = _runs(rows, CROSS_BRIDGE_PX)
        if not rr:
            continue
        r0, r1 = max(rr, key=lambda r: r[1] - r[0])
        if r1 - r0 < min_len:
            continue
        # sub-pixel edges: an anti-aliased edge column counts as card when partly covered;
        # its coverage ≈ its deviation from bg relative to the column next to it inside the
        # card, per row (median over rows where that column clearly differs from bg)
        dev = np.abs(img[r0:r1, a:b] - bg)
        s0 = a + 1.0 - _coverage(dev[:, 0], dev[:, 1], tol)
        s1 = b - 1.0 + _coverage(dev[:, -1], dev[:, -2], tol)
        if r1 - r0 >= 4:  # same across the axis (top / bottom rows of the card)
            devc = np.abs(img[r0:r1, a:b] - bg)
            c0f = r0 + 1.0 - _coverage(devc[0], devc[1], tol)
            c1f = r1 - 1.0 + _coverage(devc[-1], devc[-2], tol)
        else:
            c0f, c1f = float(r0), float(r1)
        if cards:
            prev = cards[-1]
            # neighbours only (no skipped / rejected card in between): the gap run is theirs
            if prev_run[1] == ga0_of(gap_runs, a):
                spacing = (s0 + s1) / 2.0 - (prev[0] + prev[1]) / 2.0
                gaps.append(GapObs(width=s0 - prev[1], spacing=spacing, mid=(s0 + prev[1]) / 2.0))
        cards.append((s0, s1, c0f, c1f))
        prev_run = (a, b)
    return cards, gaps


#: Sub-pixel edges: rows count when the inner column differs from bg by ≥ this many ``tol``;
#: at least EDGE_MIN_ROWS of them, else the edge column counts as fully covered.
EDGE_REF_TOLS = 2.0
EDGE_MIN_ROWS = 4


def _coverage(edge: np.ndarray, inner: np.ndarray, tol: float) -> float:
    """Coverage of an edge column (per-row deviation ``edge``) from its inner neighbour."""
    ok = inner >= EDGE_REF_TOLS * tol
    if np.count_nonzero(ok) < EDGE_MIN_ROWS:
        return 1.0
    return float(np.clip(np.median(edge[ok] / inner[ok]), 0.0, 1.0))


def ga0_of(gap_runs: list[tuple[int, int]], start: int) -> int:
    """Start column of the gap run that ends at ``start`` (-1 when none)."""
    for g0, g1 in gap_runs:
        if g1 == start:
            return g0
    return -1


def band_margins(frame: np.ndarray, box: tuple[int, int, int, int], axis: str) -> np.ndarray | None:
    """Pixels of ``frame`` (crop, gray) just outside ``box`` across the axis, over the box's
    extent along it; None when the box touches both crop edges across the axis."""
    x0, y0, x1, y1 = box
    parts: list[np.ndarray] = []
    if axis == "x":
        lo, hi = max(0, y0 - MARGIN_ROWS), max(0, y0 - MARGIN_SKIP)
        if hi > lo:
            parts.append(frame[lo:hi, x0:x1].ravel())
        lo, hi = min(frame.shape[0], y1 + MARGIN_SKIP), min(frame.shape[0], y1 + MARGIN_ROWS)
        if hi > lo:
            parts.append(frame[lo:hi, x0:x1].ravel())
    else:
        lo, hi = max(0, x0 - MARGIN_ROWS), max(0, x0 - MARGIN_SKIP)
        if hi > lo:
            parts.append(frame[y0:y1, lo:hi].ravel())
        lo, hi = min(frame.shape[1], x1 + MARGIN_SKIP), min(frame.shape[1], x1 + MARGIN_ROWS)
        if hi > lo:
            parts.append(frame[y0:y1, lo:hi].ravel())
    if not parts:
        return None
    return np.concatenate(parts)


def _quad_fit(d2: np.ndarray, y: np.ndarray, wts: np.ndarray) -> tuple[float, float]:
    """Weighted least squares ``y = c0 + c1·d2``; ``c1 = 0`` when d2 does not vary."""
    sw = float(wts.sum())
    if sw <= 0:
        return float(np.median(y)), 0.0
    mx = float((wts * d2).sum() / sw)
    my = float((wts * y).sum() / sw)
    var = float((wts * (d2 - mx) ** 2).sum())
    if var <= 1e-9:
        return my, 0.0
    c1 = float((wts * (d2 - mx) * (y - my)).sum()) / var
    return my - c1 * mx, c1


def _binned(d: np.ndarray, y: np.ndarray, width: float) -> tuple[np.ndarray, ...]:
    """Per |d| bin: (mean d², median y, count)."""
    b = np.floor(np.abs(d) / width).astype(int)
    out_d2, out_y, out_n = [], [], []
    for j in np.unique(b):
        m = b == j
        out_d2.append(float(np.mean(d[m] ** 2)))
        out_y.append(float(np.median(y[m])))
        out_n.append(float(m.sum()))
    return np.asarray(out_d2), np.asarray(out_y), np.asarray(out_n)


class CardCensus:
    """Streaming census over band frames (see module docstring)."""

    def __init__(self, k: float) -> None:
        self.k = float(k)
        self.cards: list[CardObs] = []
        self.gaps: list[GapObs] = []
        self.band_len = 0
        self.frames = 0
        self._centre = 0.0
        self._lead = 0.0
        self._inside: list[CardObs] = []

    def add(self, img: np.ndarray, margins: np.ndarray | None, pos_css: float) -> None:
        """Segment one axis-major band image (gray, analysis px) of content position
        ``pos_css``; ``margins`` = :func:`band_margins` of the same frame."""
        bgs = background(img, margins)
        if bgs is None:
            return
        bg, sig = bgs
        cards, gaps = segment(img, bg, gap_tolerance(sig), self.k)
        self.band_len = img.shape[1]
        if not cards:
            return
        self.frames += 1
        self.cards += [CardObs(pos_css, a, b, c0, c1) for a, b, c0, c1 in cards]
        self.gaps += gaps

    def result(self, span: tuple[float, float] | None = None) -> CensusResult | None:
        """Aggregate (module docstring). ``span``: the scroller's extent along the axis in band
        columns (default the whole band). Its centre is the zoom centre; cards within
        :data:`EDGE_MARGIN_PX` of its ends are clipped by the scroller edge (where the track
        background equals the page they look like short cards) and are left out."""
        k = self.k
        if len(self.cards) < MIN_CARDS or self.frames < MIN_FRAMES or self.band_len <= 0:
            return None
        lo, hi = (0.0, float(self.band_len)) if span is None else (float(span[0]), float(span[1]))
        centre = (lo + hi) / 2.0
        self._centre = centre
        self._lead = lo
        inside = [c for c in self.cards
                  if c.start >= lo + EDGE_MARGIN_PX and c.end <= hi - EDGE_MARGIN_PX]  # fmt: skip
        if len(inside) < MIN_CARDS:
            return None
        self._inside = inside
        a = np.array([[c.start, c.end, c.cross0, c.cross1] for c in inside])
        length = (a[:, 1] - a[:, 0]) / k
        cross = (a[:, 3] - a[:, 2]) / k
        d = ((a[:, 0] + a[:, 1]) / 2.0 - centre) / k
        # outliers (split / merged cards) against the median first
        med_l = float(np.median(length))
        keep = np.abs(length - med_l) <= 0.5 * med_l
        if np.count_nonzero(keep) < MIN_CARDS:
            return None
        length, cross, d = length[keep], cross[keep], d[keep]
        width = max(ZOOM_BIN_MIN_CSS, float(np.median(length)) / ZOOM_BIN_DIV)
        bd2, bl, bn = _binned(d, length, width)
        _, bc, _ = _binned(d, cross, width)
        l0, bl1 = _quad_fit(bd2, bl, bn)
        c0, bc1 = _quad_fit(bd2, bc, bn)
        # the farthest well-populated distance (≥ 2 cards in its bin)
        reach2 = float(bd2[bn >= 2].max()) if np.any(bn >= 2) else float(bd2.max())
        grow_l = bl1 * reach2 / l0 if l0 > 0 else 0.0
        grow_c = bc1 * reach2 / c0 if c0 > 0 else 0.0
        zoomed = grow_l > ZOOM_MIN_REL and grow_c > 0.5 * ZOOM_MIN_REL
        if zoomed:
            model_l = l0 + bl1 * d * d
            card_len, card_cross = l0, c0
        else:
            model_l = np.full(len(length), float(np.median(length)))
            card_len, card_cross = float(np.median(length)), float(np.median(cross))
        inl = np.abs(length - model_l) <= OUTLIER_REL * card_len
        if np.count_nonzero(inl) < MIN_CARDS:
            return None
        if not zoomed:
            card_len = float(np.median(length[inl]))
            card_cross = float(np.median(cross[inl]))
        resid = length[inl] - model_l[inl]
        spread = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) / card_len
        n_cards = int(np.count_nonzero(inl))

        # gaps (near the centre when the cards scale with position)
        g = np.array([[x.width, x.spacing, x.mid] for x in self.gaps]) if self.gaps else None
        if g is None or len(g) < 2:
            return None
        gw, gs = g[:, 0] / k, g[:, 1] / k
        gd = (g[:, 2] - centre) / k
        ok = gw < 0.5 * card_len
        if zoomed:
            near = ok & (np.abs(gd) <= card_len)
            if np.count_nonzero(near) >= 2:
                ok = near
        if np.count_nonzero(ok) < 2:
            return None
        gap = float(np.median(gw[ok]))
        gap_in = float(np.mean(np.abs(gw[ok] - gap) <= GAP_TARGET_PX))
        if zoomed:
            pitch = card_len + gap
        else:
            sp = gs[ok & (np.abs(gs - (card_len + gap)) <= OUTLIER_REL * card_len)]
            pitch = float(np.median(sp)) if sp.size >= 2 else card_len + gap

        n_fac = min(1.0, n_cards / CONF_FULL_CARDS)
        s_fac = float(np.clip(1.0 - spread / SPREAD_ZERO_REL, 0.0, 1.0))
        p_conf = n_fac * s_fac
        z_conf = 0.0
        if zoomed:
            g_fac = float(np.clip((grow_l / ZOOM_MIN_REL - 1.0) / ZOOM_GROW_SPAN, 0.0, 1.0))
            z_conf = p_conf * g_fac
            p_conf = min(p_conf, ZOOM_CONF_CAP)
        g_conf = p_conf * gap_in

        phases: tuple[float, ...] = ()
        if not zoomed:
            phases = self._grid_phases(pitch, keep, inl)
        reach = math.sqrt(reach2)
        return CensusResult(
            card_len=card_len,
            card_cross=card_cross,
            gap=gap,
            pitch=pitch,
            pitch_confidence=float(p_conf),
            gap_confidence=float(g_conf),
            zoom_per_px2=(bl1 / l0) if zoomed else None,
            zoom_observed=(1.0 + bl1 * reach2 / l0) if zoomed else 1.0,
            zoom_extent=reach,
            cards=n_cards,
            frames=self.frames,
            grid_phases=phases,
            zoom_confidence=float(z_conf),
        )

    def _grid_phases(self, pitch: float, keep: np.ndarray, inl: np.ndarray) -> tuple[float, ...]:
        """Leading-edge / centre grid phases (``PanoramaResult.grid_phases``) from the card
        edges: a card at band column ``s`` in a frame at content position ``pos`` has its
        leading edge at the scroller's leading edge (band column ``lead``) when the content is
        at ``pos − (s − lead) / k``; likewise for the centres."""
        k = self.k
        cards = [c for c, kk in zip(self._inside, keep, strict=True) if kk]
        cards = [c for c, ii in zip(cards, inl, strict=True) if ii]
        if not cards or pitch <= 0:
            return ()
        lead = np.array([c.pos - (c.start - self._lead) / k for c in cards])
        mid = self._centre
        centre = np.array([c.pos + (mid - (c.start + c.end) / 2.0) / k for c in cards])
        out: list[float] = []
        for ph in (lead, centre):
            ang = 2.0 * math.pi * (ph % pitch) / pitch
            cs, sn = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
            if math.hypot(cs, sn) < GRID_MIN_R:
                return ()
            out.append((math.atan2(sn, cs) / (2.0 * math.pi) * pitch) % pitch)
        return tuple(out)


__all__ = [
    "CardCensus",
    "CardObs",
    "CensusResult",
    "GapObs",
    "background",
    "band_margins",
    "gap_tolerance",
    "segment",
]
