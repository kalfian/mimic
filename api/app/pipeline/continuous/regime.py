"""Regime detection (PLAN-continuous §3): ambient regions, masked scan, mode decision.

Everything here is pure (no video access):

* :func:`ambient_regions` / :func:`detect_regime` — cells of the pass-1 activity grid
  (``Pass1Stats.activity``) that are active during the leading window, cursor removed, grouped
  into connected regions (§3.2). Only motion present from the start counts as ambient.
* :func:`scan_for_regime` — the transition scan, masked when ambient regions exist (§3.4);
  returns the :class:`PipelineError` instead of raising so :func:`decide_mode` can route it.
* :func:`decide_mode` — the §3.3 decision table.
* :func:`apply_ambient_mask` — OR the ambient rects into pass-2 ``WindowFrames.cursor_masks``
  so A4/A5 ignore those pixels (§3.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.errors import ErrorCode, PipelineError
from app.models.ir import SpecMode, SpecWarning
from app.models.measure import (
    AmbientRegion,
    CursorTrack,
    Rect,
    RegimeReport,
    Scale,
    ScanResult,
    ScrollerAnalysis,
    WindowFrames,
)
from app.pipeline.cursor import build_cursor_track
from app.pipeline.params import MeasureParams
from app.pipeline.scan import Pass1Stats, build_scan_result

# --------------------------------------------------------------------------------------------
# §3.2 ambient regions
# --------------------------------------------------------------------------------------------


def cell_size_css(
    grid_shape: tuple[int, int], frame_css: tuple[float, float]
) -> tuple[float, float]:
    """``(cell_w, cell_h)`` CSS px: the grid tiles the full frame (``scan.activity_grid_shape``)."""
    gh, gw = grid_shape
    return frame_css[0] / gw, frame_css[1] / gh


def cells_rect(
    r0: int, c0: int, r1: int, c1: int, grid_shape: tuple[int, int], frame_css: tuple[float, float]
) -> Rect:
    """CSS rect of the cell block ``[r0, r1] x [c0, c1]`` (inclusive), clamped to the frame."""
    cw, ch = cell_size_css(grid_shape, frame_css)
    return Rect(c0 * cw, r0 * ch, (c1 - c0 + 1) * cw, (r1 - r0 + 1) * ch).clamped(*frame_css)


#: §3.2 step 1 refinement: a cursor box is cleared only when less than this fraction of the
#: one-cell ring around it is active in that frame. A pointer over static page is isolated
#: activity; when the ring is active the box sits inside moving content (a pointer over the
#: scroller, or content that the pass-1 cursor tracker mistook for a pointer — glyphs drifting
#: in a marquee form long, pointer-sized tracks) and clearing it would cut holes into the
#: ambient region. TODO(calibrate): candidate for ``ContinuousParams`` (params frozen in P1).
CURSOR_RING_KEEP_FRAC = 0.5


def _clear_cursor_cells(
    activity: np.ndarray, boxes: Sequence[Rect | None], frame_css: tuple[float, float]
) -> np.ndarray:
    """Copy of ``activity`` with each frame's cursor-box cells cleared (§3.2 step 1).

    Boxes surrounded by activity are kept (see :data:`CURSOR_RING_KEEP_FRAC`).
    """
    act = activity.copy()
    gh, gw = act.shape[1:]
    cw, ch = cell_size_css((gh, gw), frame_css)
    for i, b in enumerate(boxes[: len(act)]):
        if b is None:
            continue
        c0, c1 = max(0, math.floor(b.x / cw)), min(gw, math.ceil(b.x2 / cw))
        r0, r1 = max(0, math.floor(b.y / ch)), min(gh, math.ceil(b.y2 / ch))
        if c1 <= c0 or r1 <= r0:
            continue
        g_r0, g_r1, g_c0, g_c1 = max(0, r0 - 1), min(gh, r1 + 1), max(0, c0 - 1), min(gw, c1 + 1)
        ring_cells = (g_r1 - g_r0) * (g_c1 - g_c0) - (r1 - r0) * (c1 - c0)
        if ring_cells > 0:
            ring_active = int(activity[i, g_r0:g_r1, g_c0:g_c1].sum()) - int(
                activity[i, r0:r1, c0:c1].sum()
            )
            if ring_active >= CURSOR_RING_KEEP_FRAC * ring_cells:
                continue
        act[i, r0:r1, c0:c1] = False
    return act


def lead_frames(times: np.ndarray, params: MeasureParams) -> list[int]:
    """Pass-1 frame indices of the leading window (§3.2 step 2).

    Frame 0 has no diff (its activity is empty by construction) and is skipped. The window is
    ``lead_window_s`` long but holds at least ``lead_min_frames`` frames; ``[]`` when the
    recording is shorter than that (too short to call anything ambient).
    """
    c = params.continuous
    n = len(times)
    if n < 2:
        return []
    t_end = float(times[0]) + c.lead_window_s + 1e-9
    idx = [i for i in range(1, n) if float(times[i]) <= t_end]
    if len(idx) < c.lead_min_frames:
        idx = list(range(1, min(n, 1 + c.lead_min_frames)))
    return idx if len(idx) >= c.lead_min_frames else []


def ambient_regions(
    activity: np.ndarray,
    times: np.ndarray,
    frame_css: tuple[float, float],
    params: MeasureParams,
    cursor_boxes: Sequence[Rect | None] = (),
) -> RegimeReport:
    """Ambient regions from a bool ``(N, gh, gw)`` activity grid (§3.2, pure)."""
    c = params.continuous
    grid_shape = (int(activity.shape[1]), int(activity.shape[2]))
    t0 = float(times[0]) if len(times) else 0.0
    idx = lead_frames(times, params)
    if not idx:
        return RegimeReport([], c.cell_css, grid_shape, (t0, t0 + c.lead_window_s))
    window = (t0, float(times[idx[-1]]))
    act = _clear_cursor_cells(activity, cursor_boxes, frame_css) if cursor_boxes else activity
    lead_frac = act[idx].mean(axis=0)
    amb = lead_frac >= c.ambient_lead_frac
    if not amb.any():
        return RegimeReport([], c.cell_css, grid_shape, window)
    k = max(1, c.ambient_close_cells)
    closed = cv2.morphologyEx(amb.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, labels = cv2.connectedComponents(closed, connectivity=8)
    found: list[tuple[int, AmbientRegion]] = []
    for j in range(1, n_lab):
        comp = labels == j
        own = comp & amb  # the ambient cells themselves (closing only bridges gaps)
        cells = int(own.sum())
        if cells < c.ambient_min_cells:
            continue
        rows, cols = np.nonzero(own)
        rect = cells_rect(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()),
                          grid_shape, frame_css)  # fmt: skip
        found.append((int(comp.sum()), AmbientRegion(rect, cells, float(lead_frac[own].mean()))))
    found.sort(key=lambda e: (-e[0], -e[1].rect_css.area, e[1].rect_css.y, e[1].rect_css.x))
    return RegimeReport([a for _, a in found[: c.max_ambient_regions]], c.cell_css, grid_shape,
                        window)  # fmt: skip


#: Pieces of one scroller (:func:`merge_row_pieces`): cross extents overlapping by at least this
#: fraction of the smaller one, autoplay speeds within this ratio (cards that scale with their
#: position move up to ≈1.2× faster near the scroller edges than at its centre).
PIECE_CROSS_OVERLAP = 0.6
PIECE_SPEED_RATIO = 1.5


def _cross(r: Rect, axis: str) -> tuple[float, float]:
    return (r.y, r.y2) if axis == "x" else (r.x, r.x2)


def _pieces_of_one(a: ScrollerAnalysis, b: ScrollerAnalysis) -> bool:
    """Two analysed ambient regions that are parts of one scroller row: same axis, side by side
    along it in one band, moving the same way."""
    if not (a.is_scroller and b.is_scroller) or a.axis is None or a.axis != b.axis:
        return False
    ra, rb = a.region.rect_css, b.region.rect_css
    (a0, a1), (b0, b1) = _cross(ra, a.axis), _cross(rb, a.axis)
    overlap = min(a1, b1) - max(a0, b0)
    if overlap < PIECE_CROSS_OVERLAP * min(a1 - a0, b1 - b0):
        return False
    va, vb = a.autoplay_velocity, b.autoplay_velocity
    if (va is None) != (vb is None):
        return False
    if va is not None and vb is not None:
        if va * vb <= 0:
            return False
        lo, hi = sorted((abs(va), abs(vb)))
        if hi > PIECE_SPEED_RATIO * lo:
            return False
    return True


def merge_row_pieces(
    regime: RegimeReport, scrollers: Sequence[ScrollerAnalysis], frame_css: tuple[float, float]
) -> RegimeReport | None:
    """Ambient regions that are pieces of ONE scroller row → one region (acceptance run).

    Flat placeholder cards (a plain fill, a short title) only change where their edges and text
    pass during the leading window, so a single carousel can come out of §3.2 as several
    disjoint regions in one band — each then measured as its own "scroller" (a fraction of the
    real box, a local speed that varies with the card scaling). Analysed regions with the same
    axis, overlapping across it (:data:`PIECE_CROSS_OVERLAP`) and the same autoplay direction
    and similar speed (:data:`PIECE_SPEED_RATIO`) are merged. Along the axis the merged region
    spans the whole frame (the flat stretches the lead window could not see may lie beyond the
    outermost pieces, at either end); across it, the union of the pieces. The scroller box itself
    is refined later from the pixels that actually changed over the whole recording. Returns
    ``None`` when nothing merges (pure)."""
    n = len(scrollers)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _pieces_of_one(scrollers[i], scrollers[j]):
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    merged = [g for g in groups.values() if len(g) > 1]
    if not merged:
        return None
    fw, fh = frame_css
    used = {i for g in merged for i in g}
    out: list[AmbientRegion] = [s.region for i, s in enumerate(scrollers) if i not in used]
    for g in merged:
        axis = scrollers[g[0]].axis
        assert axis is not None
        rects = [scrollers[i].region.rect_css for i in g]
        lo, hi = 0.0, fw if axis == "x" else fh
        c0 = min(_cross(r, axis)[0] for r in rects)
        c1 = max(_cross(r, axis)[1] for r in rects)
        rect = Rect(lo, c0, hi - lo, c1 - c0) if axis == "x" else Rect(c0, lo, c1 - c0, hi - lo)
        cells = sum(scrollers[i].region.cells for i in g)
        lead = sum(scrollers[i].region.lead_frac * scrollers[i].region.cells for i in g) / max(
            cells, 1
        )
        out.append(AmbientRegion(rect.clamped(fw, fh), cells, lead, merged=True))
    out.sort(key=lambda a: (-a.rect_css.area, a.rect_css.y, a.rect_css.x))
    return RegimeReport(out, regime.cell_css, regime.grid_shape, regime.lead_window_s)


def detect_regime(
    stats: Pass1Stats, params: MeasureParams, cursor: CursorTrack | None = None
) -> RegimeReport:
    """§3.2 on pass-1 stats. ``cursor`` defaults to the unchanged ``build_cursor_track``."""
    c = params.continuous
    if stats.activity is None or len(stats.activity) == 0:
        t0 = float(stats.times[0]) if len(stats.times) else 0.0
        return RegimeReport([], c.cell_css, (0, 0), (t0, t0 + c.lead_window_s))
    if cursor is None:
        cursor, _ = build_cursor_track(stats.blobs, stats.times, params)
    return ambient_regions(stats.activity, stats.times, stats.frame_css, params, cursor.boxes_css)


# --------------------------------------------------------------------------------------------
# §3.4 masked transition path
# --------------------------------------------------------------------------------------------


def scan_for_regime(
    stats: Pass1Stats, scale: Scale, params: MeasureParams, regime: RegimeReport
) -> ScanResult | PipelineError:
    """Transition scan (masked by the ambient regions, if any); a raised error is returned."""
    try:
        return build_scan_result(stats, scale, params, ambient=regime.ambient or None)
    except PipelineError as exc:
        return exc


def apply_ambient_mask(
    wfs: Sequence[WindowFrames], regions: Sequence[AmbientRegion | Rect], params: MeasureParams
) -> Sequence[WindowFrames]:
    """OR the (dilated) ambient rects into every frame's ``cursor_masks`` (in place; §3.4).

    ``255`` = excluded pixel, like the cursor boxes; frames whose crop does not overlap any
    region keep their mask (``None`` stays ``None``). Returns ``wfs`` for chaining.
    """
    if not regions:
        return wfs
    pad = params.continuous.ambient_mask_dilate_css
    rects = [(a.rect_css if isinstance(a, AmbientRegion) else a).padded(pad) for a in regions]
    for wf in wfs:
        if not wf.crops:
            continue
        h, w = wf.crops[0].shape[:2]
        k = wf.scale.k
        boxes: list[tuple[int, int, int, int]] = []
        for r in rects:
            x0 = max(int(math.floor((r.x - wf.crop_css.x) * k)), 0)
            y0 = max(int(math.floor((r.y - wf.crop_css.y) * k)), 0)
            x1 = min(int(math.ceil((r.x2 - wf.crop_css.x) * k)), w)
            y1 = min(int(math.ceil((r.y2 - wf.crop_css.y) * k)), h)
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
        if not boxes:
            continue
        for i, m in enumerate(wf.cursor_masks):
            mask = np.zeros((h, w), np.uint8) if m is None else m.copy()
            for x0, y0, x1, y1 in boxes:
                mask[y0:y1, x0:x1] = 255
            wf.cursor_masks[i] = mask
    return wfs


# --------------------------------------------------------------------------------------------
# §3.3 decision
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ModeDecision:
    """Outcome of :func:`decide_mode` (errors are raised instead).

    * ``transition``: run the transition path on ``scan`` (masked when ambient regions exist;
      P2 also calls :func:`apply_ambient_mask` on the pass-2 windows then).
    * ``continuous``: analyse ``scroller``; ``scan`` is the masked scan when it had runs (its
      segments are the ignored discrete motion), else ``None``.

    ``warnings`` are the decision's own warnings; ``scan.warnings`` are not repeated here.
    """

    mode: SpecMode
    scan: ScanResult | None
    scroller: ScrollerAnalysis | None = None
    extra_scrollers: int = 0
    masked: list[AmbientRegion] = field(default_factory=list)  # regions masked out of the scan
    warnings: list[SpecWarning] = field(default_factory=list)


def describe_region(r: Rect) -> str:
    """``"around x 89, y 301, 746×194 px"`` (CSS px, rounded)."""
    return f"around x {round(r.x)}, y {round(r.y)}, {round(r.w)}×{round(r.h)} px"


def _speed_text(v: float) -> str:
    """≥ 100 px/s rounded to 10, else to 1 (PLAN-continuous §6 speed wording)."""
    s = abs(v)
    return f"{int(round(s / 10.0) * 10)}" if s >= 100 else f"{int(round(s))}"


def _direction(axis: str | None, v: float) -> str:
    if axis == "y":
        return "down" if v > 0 else "up"
    return "right" if v > 0 else "left"


def scroller_box(s: ScrollerAnalysis) -> Rect:
    """Refined viewport when measured, else the ambient region's cell box."""
    return s.series.region if s.series is not None else s.region.rect_css


def _is_page_scroll(
    s: ScrollerAnalysis, frame_css: tuple[float, float], params: MeasureParams
) -> bool:
    c = params.continuous
    fw, fh = frame_css
    b = scroller_box(s).clamped(fw, fh)
    if fw <= 0 or fh <= 0:
        return False
    span = b.w >= c.page_scroll_span_frac * fw and b.h >= c.page_scroll_span_frac * fh
    return b.area > c.page_scroll_area_frac * fw * fh or span


def _by_area(items: Sequence[ScrollerAnalysis]) -> list[ScrollerAnalysis]:
    return sorted(
        items, key=lambda s: (-scroller_box(s).area, scroller_box(s).y, scroller_box(s).x)
    )


def _masked_warning(regime: RegimeReport, scrollers: Sequence[ScrollerAnalysis]) -> SpecWarning:
    """``ambient_motion_masked``: names every masked region (+ speed when it is a scroller)."""
    parts: list[str] = []
    for a in regime.ambient:
        s = next((s for s in scrollers if s.region == a), None)
        where = describe_region(scroller_box(s) if s is not None and s.is_scroller else a.rect_css)
        if s is not None and s.is_scroller and s.autoplay_velocity:
            v = s.autoplay_velocity
            where += f", a scroller moving {_direction(s.axis, v)} at about {_speed_text(v)} px/s"
        parts.append(where)
    many = len(parts) > 1
    return SpecWarning(
        code="ambient_motion_masked",
        severity="info",
        message=f"Part{'s' if many else ''} of the page ({'; '.join(parts)}) moved continuously "
        f"from the start and {'were' if many else 'was'} left out of the analysis.",
    )


def _extra_scrollers_warning(n: int) -> SpecWarning:
    return SpecWarning(
        code="extra_scrollers_ignored",
        severity="info",
        message=f"{n} other continuously moving area(s) were ignored; only one scroller is "
        "analysed.",
    )


def _extra_segments_warning(scan: ScanResult | PipelineError) -> SpecWarning:
    if isinstance(scan, ScanResult):
        what = f"{len(scan.segments)} other motion segment(s) outside the scroller were ignored"
    else:
        what = "Other motion outside the scroller was ignored"
    return SpecWarning(
        code="extra_segments_ignored",
        severity="info",
        message=f"{what}; only the scroller is analysed.",
    )


_UNSUPPORTED_DETAIL = {
    None: "not as a single horizontal or vertical scroller",
    "two_axis": "not as a single horizontal or vertical scroller",
    "untrackable": "its motion could not be tracked as a single horizontal or vertical scroller",
    "opposing_unsplit": "it holds several scrollers moving in different directions that could "
    "not be told apart",
}


def unsupported_error(regime: RegimeReport, scrollers: Sequence[ScrollerAnalysis]) -> PipelineError:
    """``continuous_motion_unsupported`` naming the largest region and the reason (§3.5)."""
    failed = _by_area([s for s in scrollers if not s.is_scroller])
    if failed:
        rect, reason = failed[0].region.rect_css, failed[0].reason
    else:
        rect, reason = regime.ambient[0].rect_css, None
    return PipelineError(
        ErrorCode.CONTINUOUS_MOTION_UNSUPPORTED,
        f"Part of the page ({describe_region(rect)}) moves continuously from the start, but "
        f"{_UNSUPPORTED_DETAIL.get(reason, _UNSUPPORTED_DETAIL[None])}, so it can't be measured. "
        "Record a component that rests before you interact, or pause that animation.",
    )


def decide_mode(
    regime: RegimeReport,
    scrollers: Sequence[ScrollerAnalysis],
    scan: ScanResult | PipelineError,
    frame_css: tuple[float, float],
    params: MeasureParams,
) -> ModeDecision:
    """§3.3 decision table. Raises the routed :class:`PipelineError` for the error rows.

    ``scan`` is :func:`scan_for_regime`'s result (masked iff ``regime`` has ambient regions);
    ``scrollers`` are the analysed ambient regions (``continuous.measure.analyse_ambient``).
    "Masked has runs" = the scan succeeded or failed for a reason other than
    ``no_motion_detected``.

    Rows beyond the plan's table: a masked scan that has runs but cannot be used as a
    transition (``no_stable_state`` / ``unsupported_motion``) is raised when there is no
    scroller to fall back to; with an autoplay-only scroller the marquee is measured instead
    (+ ``extra_segments_ignored``).
    """
    if not regime.has_ambient:
        if isinstance(scan, PipelineError):
            raise scan
        return ModeDecision(mode="transition", scan=scan)

    real = [s for s in scrollers if s.is_scroller]
    if any(_is_page_scroll(s, frame_css, params) for s in real):
        raise PipelineError(ErrorCode.UNSUPPORTED_MOTION)
    has_runs = isinstance(scan, ScanResult) or scan.code != ErrorCode.NO_MOTION_DETECTED
    masked_scan = scan if isinstance(scan, ScanResult) else None
    masked = list(regime.ambient)

    def continuous(chosen: ScrollerAnalysis) -> ModeDecision:
        extra = len(real) - 1
        warnings: list[SpecWarning] = []
        if extra > 0:
            warnings.append(_extra_scrollers_warning(extra))
        if has_runs:
            warnings.append(_extra_segments_warning(scan))
        return ModeDecision(mode="continuous", scan=masked_scan, scroller=chosen,
                            extra_scrollers=extra, masked=masked, warnings=warnings)  # fmt: skip

    def transition() -> ModeDecision:
        if masked_scan is None:
            raise scan  # type: ignore[misc]  # has runs but unusable as a transition
        return ModeDecision(mode="transition", scan=masked_scan, masked=masked,
                            warnings=[_masked_warning(regime, scrollers)])  # fmt: skip

    interacted = _by_area([s for s in real if s.has_interaction])
    if interacted:
        return continuous(interacted[0])
    autoplay_only = _by_area([s for s in real if not s.has_interaction])
    if autoplay_only:
        if has_runs and masked_scan is not None:
            return transition()
        return continuous(autoplay_only[0])
    if has_runs:
        return transition()
    raise unsupported_error(regime, scrollers)
