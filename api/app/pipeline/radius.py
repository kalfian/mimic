"""Border-radius heuristic (PLAN §6.6 "Border radius"; confidence cap 0.5 downstream).

For each bbox corner, walk the 45° diagonal inward until the pixel's Lab distance to the
element interior falls below ``radius_lab_dist``. For a circular corner of radius ``r`` that
distance is ``d = r(1 − 1/√2)`` → ``r ≈ 3.414·d``. Corners that agree within 2 px are averaged.
The interior reference is taken per corner (a patch further along the diagonal) so a corner
covered by an image is skipped instead of producing nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.models.ir import PxValue
from app.models.measure import ElementCandidate, PropertySeries, Rect, WindowFrames
from app.pipeline.params import MeasureParams
from app.pipeline.photometric import bgr_to_lab
from app.pipeline.track import IDENTITY, FrameTrack, SegmentSpan, warp_css_to_px

DIAG_FACTOR = 1.0 / (1.0 - 1.0 / np.sqrt(2.0))  # ≈ 3.414
#: TODO(calibrate): module-local; one lone corner is usually texture, not a rounded corner.
MIN_AGREEING_CORNERS = 2


@dataclass(frozen=True, slots=True)
class RadiusEstimate:
    radius_css: float
    corners: int  # corners that agreed (1..4)


def _corner_depth(lab: np.ndarray, x: float, y: float, dx: int, dy: int, max_d: int,
                  thr: float) -> float | None:  # fmt: skip
    h, w = lab.shape[:2]
    # interior reference: small patch well inside along the diagonal
    rx, ry = int(round(x + dx * max_d)), int(round(y + dy * max_d))
    x0, x1 = max(0, rx - 2), min(w, rx + 3)
    y0, y1 = max(0, ry - 2), min(h, ry + 3)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = lab[y0:y1, x0:x1].reshape(-1, 3)
    if float(patch.std(axis=0).max()) > thr / 2:
        return None  # textured interior (image corner) → no reliable reference
    ref = np.median(patch, axis=0)
    # the corner must be visible: the outside (just beyond the bbox corner) has to differ from
    # the interior, otherwise a low-contrast card on a similar background reads as radius 0
    ox, oy = int(round(x - dx * 2)), int(round(y - dy * 2))
    if 0 <= ox < w and 0 <= oy < h and float(np.linalg.norm(lab[oy, ox] - ref)) < 1.5 * thr:
        return None
    prev_dist = None
    for d in range(0, max_d + 1):
        px, py = int(round(x + dx * d)), int(round(y + dy * d))
        if not (0 <= px < w and 0 <= py < h):
            return None
        dist = float(np.linalg.norm(lab[py, px] - ref))
        if dist < thr:
            if d == 0 or prev_dist is None:
                return 0.0
            # sub-pixel: linear interpolation of the threshold crossing
            frac = (prev_dist - thr) / max(prev_dist - dist, 1e-6)
            return d - 1 + float(np.clip(frac, 0.0, 1.0))
        prev_dist = dist
    return None


def corner_radius(
    img: np.ndarray, box: Rect, k: float, params: MeasureParams
) -> RadiusEstimate | None:
    """Static radius of the element at ``box`` (crop px) → CSS px."""
    p = params.photometric
    lab = bgr_to_lab(img)
    max_d = int(max(2, min(box.w, box.h) * 0.25))
    corners = [
        (box.x, box.y, 1, 1),
        (box.x2 - 1, box.y, -1, 1),
        (box.x, box.y2 - 1, 1, -1),
        (box.x2 - 1, box.y2 - 1, -1, -1),
    ]
    rs = []
    for x, y, dx, dy in corners:
        d = _corner_depth(lab, x, y, dx, dy, max_d, p.radius_lab_dist)
        if d is not None:
            rs.append(DIAG_FACTOR * d / k)
    if not rs:
        return None
    med = float(np.median(rs))
    agree = [r for r in rs if abs(r - med) <= p.radius_corner_agree_css]
    if len(agree) < MIN_AGREEING_CORNERS:
        return None
    return RadiusEstimate(float(np.mean(agree)), len(agree))


@dataclass(slots=True)
class RadiusResult:
    series: list[PropertySeries]
    static: dict[str, RadiusEstimate]


def measure_radii(
    elements: list[ElementCandidate],
    boxes_px: dict[str, tuple[Rect, Rect]],
    warps_px: dict[str, np.ndarray],
    state_a: np.ndarray,
    state_b: np.ndarray,
    windows: list[tuple[WindowFrames, list[SegmentSpan]]],
    params: MeasureParams,
    tracks: dict[tuple[str, str], FrameTrack] | None = None,
) -> RadiusResult:
    """Static radius per element (A) and a ``border-radius`` series if A/B differ ≥ 2 px."""
    tracks = tracks or {}
    k = windows[0][0].scale.k if windows else 1.0
    static: dict[str, RadiusEstimate] = {}
    series: list[PropertySeries] = []
    for e in elements:
        if e.kind not in ("transform", "photometric", "appear"):
            continue
        box_a, box_b = boxes_px[e.id]
        ra = corner_radius(state_a, box_a, k, params) if e.kind != "appear" else None
        rb = corner_radius(state_b, box_b, k, params)
        if ra is not None:
            static[e.id] = ra
        elif rb is not None:
            static[e.id] = rb
        if ra is None or rb is None or e.kind != "transform" and e.kind != "photometric":
            continue
        if abs(rb.radius_css - ra.radius_css) < params.photometric.radius_min_delta_css:
            continue
        for wf, spans in windows:
            ft = tracks.get((e.id, wf.segment_id))
            n = len(wf.crops)
            vals = np.full(n, np.nan)
            q = np.zeros(n)
            ox, oy = wf.crop_origin_css
            for i in range(n):
                w_i = IDENTITY if ft is None else warp_css_to_px(ft.warps[i], k, (ox, oy))
                h, w = wf.crops[i].shape[:2]
                img = wf.crops[i]
                if not np.allclose(w_i, IDENTITY):
                    img = cv2.warpAffine(img, w_i, (w, h),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
                r = corner_radius(img, box_a, k, params)
                if r is not None:
                    vals[i], q[i] = r.radius_css, r.corners / 4.0
            for span in spans:
                t = wf.times
                ok = np.isfinite(vals) & (t >= span.lo_s - 1e-9) & (t <= span.hi_s + 1e-9)
                if not ok.any():
                    continue
                pre = ok & (t <= span.start_s + 1e-9)
                post = ok & (t >= span.end_s - 1e-9)
                v0 = float(np.median(vals[pre])) if pre.any() else float(vals[ok][0])
                v1 = float(np.median(vals[post])) if post.any() else float(vals[ok][-1])
                sel = ok & ~wf.is_dup
                series.append(
                    PropertySeries(
                        element_id=e.id,
                        property="border-radius",
                        segment=span.segment_id,
                        times=t[sel].copy(),
                        values=vals[sel].copy(),
                        quality=q[sel].copy(),
                        v_start=v0,
                        v_end=v1,
                        unit="px",
                        from_value=PxValue(number=round(v0, 1)),
                        to_value=PxValue(number=round(v1, 1)),
                    )  # fmt: skip
                )
    return RadiusResult(series=series, static=static)
