"""Box-shadow heuristics (PLAN §6.6 "Shadow"; confidence cap 0.45 is applied downstream).

* :func:`shadow_change` — ring (box dilated by 40 px minus box) darkening between A and B with
  the element motion compensated (B is warped back onto A). A change needs mean darkening
  > 2 levels over ≥ 30 % of the ring; the realistic deliverable is the direction.
* :func:`estimate_shadow` — static per-state estimate from outward luminance profiles on the
  bottom and top sides vs the far-ring background → ``0 {oy}px {blur}px rgba(0,0,0,a)``.
* :func:`measure_shadows` — stage entry producing ``box-shadow`` series (α progress 0→1) for
  transform elements whose ring changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.models.ir import Shadow, ShadowValue
from app.models.measure import ElementCandidate, PropertySeries, Rect, WindowFrames
from app.pipeline.params import MeasureParams
from app.pipeline.track import IDENTITY, FrameTrack, SegmentSpan, warp_css_to_px


def _lum(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img


def ring_mask(shape, box: Rect, width: float, inner_gap: float = 0.0) -> np.ndarray:
    """Pixels within ``width`` outside ``box`` (minus an ``inner_gap`` hugging the edge)."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    o = box.padded(width)
    x0, y0 = int(max(0, np.floor(o.x))), int(max(0, np.floor(o.y)))
    x1, y1 = int(min(w, np.ceil(o.x2))), int(min(h, np.ceil(o.y2)))
    m[y0:y1, x0:x1] = True
    i = box.padded(inner_gap)
    ix0, iy0 = int(max(0, np.floor(i.x))), int(max(0, np.floor(i.y)))
    ix1, iy1 = int(min(w, np.ceil(i.x2))), int(min(h, np.ceil(i.y2)))
    m[iy0:iy1, ix0:ix1] = False
    return m


@dataclass(frozen=True, slots=True)
class ShadowChange:
    darkening: float  # mean lum(A) − lum(B) over the ring (levels; > 0 → B darker)
    coverage: float  # fraction of ring pixels with |d| > min_darkening
    increases: bool


def _back_to_a(img: np.ndarray, warp_a_to_img: np.ndarray) -> np.ndarray:
    """Warp ``img`` into A coordinates given the element's A→img warp (crop px)."""
    h, w = img.shape[:2]
    if np.allclose(warp_a_to_img, IDENTITY):
        return img
    return cv2.warpAffine(
        img, warp_a_to_img, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )  # fmt: skip


def shadow_change(
    a: np.ndarray,
    b: np.ndarray,
    box_a: Rect,
    warp_ab: np.ndarray,
    k: float,
    params: MeasureParams,
    *,
    exclude: np.ndarray | None = None,
) -> ShadowChange | None:
    p = params.photometric
    ring = ring_mask(a.shape, box_a, p.shadow_ring_css * k, inner_gap=1.5 * k)
    if exclude is not None:
        ring &= ~exclude
    if ring.sum() < 50:
        return None
    la = _lum(a)
    lb = _lum(_back_to_a(b, warp_ab))
    d = la[ring] - lb[ring]
    mean = float(d.mean())
    cov = float((np.abs(d) > p.shadow_min_darkening).mean())
    return ShadowChange(mean, cov, mean > 0)


def is_shadow_change(sc: ShadowChange | None, params: MeasureParams) -> bool:
    p = params.photometric
    return (
        sc is not None
        and abs(sc.darkening) > p.shadow_min_darkening / 2
        and sc.coverage >= p.shadow_min_coverage
    )


def _profile(lum: np.ndarray, box: Rect, side: str, ext: int, bg: float) -> np.ndarray:
    """Darkening (bg − lum) at distances 1..ext outside one box side (median across it)."""
    h, w = lum.shape
    out = np.zeros(ext)
    if side in ("bottom", "top"):
        xs0 = int(max(0, box.x + 0.2 * box.w))
        xs1 = int(min(w, box.x2 - 0.2 * box.w))
        for d in range(1, ext + 1):
            y = int(round(box.y2 - 1 + d)) if side == "bottom" else int(round(box.y - d))
            if 0 <= y < h and xs1 > xs0:
                out[d - 1] = bg - float(np.median(lum[y, xs0:xs1]))
    else:
        ys0 = int(max(0, box.y + 0.2 * box.h))
        ys1 = int(min(h, box.y2 - 0.2 * box.h))
        for d in range(1, ext + 1):
            x = int(round(box.x2 - 1 + d)) if side == "right" else int(round(box.x - d))
            if 0 <= x < w and ys1 > ys0:
                out[d - 1] = bg - float(np.median(lum[ys0:ys1, x]))
    return out


def estimate_shadow(img: np.ndarray, box: Rect, k: float, params: MeasureParams) -> Shadow | None:
    """Static shadow of the element at ``box`` in ``img`` (crop px) → CSS-px ``Shadow``."""
    p = params.photometric
    lum = _lum(img)
    ext = max(4, int(round(p.shadow_ring_css * k)))
    far = ring_mask(lum.shape, box, ext) & ~ring_mask(lum.shape, box, 0.75 * ext)
    if far.sum() < 20:
        return None
    bg = float(np.median(lum[far]))
    skip = max(1, int(round(1.5 * k)))  # border / antialias row
    bot = _profile(lum, box, "bottom", ext, bg)
    top = _profile(lum, box, "top", ext, bg)
    bot[:skip] = 0
    top[:skip] = 0
    peak = float(max(bot.max(), top.max(), 0.0))
    if peak < p.shadow_min_darkening:
        return None

    def extent(prof: np.ndarray) -> float:
        above = np.nonzero(prof > p.shadow_profile_frac * peak)[0]
        return float(above.max() + 1) if above.size else 0.0

    e_b, e_t = extent(bot), extent(top)
    oy = (e_b - e_t) / 2.0
    blur = max(e_b, e_t)
    alpha = float(np.clip(2.0 * peak / max(bg, 1.0), 0.0, p.shadow_alpha_max))
    return Shadow(x=0.0, y=round(oy / k, 1), blur=round(blur / k, 1), spread=0.0,
                  rgba=(0, 0, 0, round(alpha, 3)))  # fmt: skip


@dataclass(slots=True)
class ShadowResult:
    series: list[PropertySeries]
    static: dict[str, Shadow | None]  # element id → state-A shadow estimate
    changes: dict[str, ShadowChange]


def measure_shadows(
    elements: list[ElementCandidate],
    boxes_px: dict[str, tuple[Rect, Rect]],
    warps_px: dict[str, np.ndarray],
    element_masks: dict[str, np.ndarray],
    state_a: np.ndarray,
    state_b: np.ndarray,
    windows: list[tuple[WindowFrames, list[SegmentSpan]]],
    params: MeasureParams,
    tracks: dict[tuple[str, str], FrameTrack] | None = None,
) -> ShadowResult:
    """Shadow presence/direction for top-level transform elements (+ progress series)."""
    tracks = tracks or {}
    k = windows[0][0].scale.k if windows else 1.0
    series: list[PropertySeries] = []
    static: dict[str, Shadow | None] = {}
    changes: dict[str, ShadowChange] = {}
    for e in elements:
        if e.kind != "transform" or e.parent_id is not None:
            continue
        box_a, box_b = boxes_px[e.id]
        w_ab = warps_px[e.id]
        others = np.zeros(state_a.shape[:2], bool)
        for o in elements:
            if o.id != e.id and o.parent_id != e.id and o.kind != "backdrop":
                others |= element_masks[o.id]
        sh_a = estimate_shadow(state_a, box_a, k, params)
        sh_b = estimate_shadow(state_b, box_b, k, params)
        static[e.id] = sh_a
        sc = shadow_change(state_a, state_b, box_a, w_ab, k, params, exclude=others)
        if sc is None:
            continue
        changes[e.id] = sc
        if not is_shadow_change(sc, params):
            continue
        ring = ring_mask(state_a.shape, box_a, params.photometric.shadow_ring_css * k, 1.5 * k)
        ring &= ~others
        la = _lum(state_a)[ring]
        for wf, spans in windows:
            ft = tracks.get((e.id, wf.segment_id))
            n = len(wf.crops)
            vals = np.full(n, np.nan)
            q = np.zeros(n)
            ox, oy = wf.crop_origin_css
            for i in range(n):
                w_i = IDENTITY if ft is None else warp_css_to_px(ft.warps[i], k, (ox, oy))
                if ft is not None and not ft.visible[i]:
                    continue
                li = _lum(_back_to_a(wf.crops[i], w_i))
                sel = np.ones(int(ring.sum()), bool)
                cm = wf.cursor_masks[i]
                if cm is not None:
                    sel = _back_to_a(cm, w_i)[ring] == 0
                if sel.sum() < 20:
                    continue
                d = float((la - li[ring])[sel].mean())
                vals[i] = d / sc.darkening if abs(sc.darkening) > 1e-6 else 0.0
                q[i] = 1.0 if ft is None else float(ft.rho[i])
            note = "shadow increases" if sc.increases else "shadow decreases"
            for span in spans:
                s = _shadow_series(e.id, span, wf, vals, q, sh_a, sh_b, note)
                if s is not None:
                    series.append(s)
    return ShadowResult(series=series, static=static, changes=changes)


def _shadow_series(eid, span, wf, vals, q, sh_a, sh_b, note) -> PropertySeries | None:
    t = wf.times
    ok = np.isfinite(vals)
    inside = ok & (t >= span.lo_s - 1e-9) & (t <= span.hi_s + 1e-9)
    if not inside.any():
        return None
    pre = inside & (t <= span.start_s - 1 / 30 + 1e-9)
    post = inside & (t >= span.end_s + 1 / 30 - 1e-9)
    v0 = float(np.median(vals[pre])) if pre.any() else float(vals[inside][0])
    v1 = float(np.median(vals[post])) if post.any() else float(vals[inside][-1])
    sel = inside & ~wf.is_dup
    at_b = lambda v: v >= 0.5  # noqa: E731
    std_parts = [vals[pre] - v0] if pre.sum() > 1 else []
    std_parts += [vals[post] - v1] if post.sum() > 1 else []
    return PropertySeries(
        element_id=eid, property="box-shadow", segment=span.segment_id,
        times=t[sel].copy(), values=vals[sel].copy(), quality=q[sel].copy(),
        v_start=v0, v_end=v1, unit="shadow",
        from_value=ShadowValue(shadow=sh_b if at_b(v0) else sh_a),
        to_value=ShadowValue(shadow=sh_b if at_b(v1) else sh_a),
        stable_std=float(np.concatenate(std_parts).std()) if std_parts else 0.0,
        notes=[note],
    )  # fmt: skip
