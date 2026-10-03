"""Photometric measurements (PLAN §6.6 "Photometric").

* :func:`blend_alpha` — the universal progress ``α(t) = Σ(I_t−A)·(B−A) / Σ‖B−A‖²`` over the
  changed pixels (cursor-masked), optionally geometry-compensated (``A_t``/``B_t`` warped to
  frame t) for elements that move and change at once.
* :func:`element_colors` — Lab medians (fill: eroded mask; text: extreme pixels vs local bg).
* :func:`opacity_kappa` — "present in both states but dimmer" test ``A−bg ≈ κ(B−bg)``.
* :func:`measure_photometric` — stage entry: per element/segment ``PropertySeries`` for
  ``color`` / ``background-color`` / ``opacity`` (appear, disappear, dimmed, backdrop) /
  ``content``.

``values`` hold opacity (0..1) for opacity series and α progress (0→1) for color/backdrop/
content; ``from_value``/``to_value`` carry the reportable endpoints (``models.measure``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.models.ir import ColorValue, Property, PxValue, RatioValue, TextValue, Value
from app.models.measure import ElementCandidate, PropertySeries, Rect, WindowFrames
from app.pipeline.params import MeasureParams
from app.pipeline.track import IDENTITY, FrameTrack, SegmentSpan, compose, invert, warp_css_to_px

#: "Dimmed" (opacity) needs the κ-model to reproduce the A colour within this ΔE76.
#: TODO(calibrate): module-local; candidate for ``PhotometricParams``.
OPACITY_HUE_MAX_DE = 3.0
#: Pixels where |B − A| ≤ this (any channel) carry no blend information.
BLEND_MIN_DIFF = 6.0

# --------------------------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------------------------


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR (..., 3) → float Lab (L 0..100, a/b ±128) via OpenCV float conversion."""
    arr = np.asarray(bgr, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_BGR2Lab).reshape(np.shape(bgr))


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    arr = np.asarray(lab, dtype=np.float32).reshape(-1, 1, 3)
    out = cv2.cvtColor(arr, cv2.COLOR_Lab2BGR).reshape(np.shape(lab))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def hex_from_lab(lab: np.ndarray) -> str:
    b, g, r = (int(v) for v in lab_to_bgr(np.asarray(lab, np.float32)))
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_from_bgr(bgr) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float)))


def _ring(mask: np.ndarray, width: int) -> np.ndarray:
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), k) > 0
    return dil & ~mask


@dataclass(frozen=True, slots=True)
class ColorPair:
    lab_a: np.ndarray
    lab_b: np.ndarray
    hex_a: str
    hex_b: str
    delta_e: float


def region_color(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    text: bool,
    k: float,
    params: MeasureParams,
    valid: np.ndarray | None = None,
) -> np.ndarray | None:
    """Representative Lab colour of the masked element in ``img`` (§6.6 colour value)."""
    p = params.photometric
    m = mask.copy()
    if valid is not None:
        m &= valid > 0
    lab = bgr_to_lab(img)
    if text:
        if not m.any():
            return None
        ring = _ring(mask, max(2, int(round(3 * k))))
        if valid is not None:
            ring &= valid > 0
        bg = np.median(lab[ring], axis=0) if ring.any() else np.median(lab, axis=(0, 1))
        px = lab[m]
        dist = np.linalg.norm(px - bg, axis=1)
        n = max(1, int(np.ceil(p.text_top_frac * len(px))))
        top = px[np.argsort(-dist)[:n]]
        return np.median(top, axis=0)
    er = max(1, int(round(p.fill_erode_css * k)))
    eroded = cv2.erode(m.astype(np.uint8), np.ones((2 * er + 1, 2 * er + 1), np.uint8)) > 0
    sel = eroded if eroded.sum() >= 10 else m
    if not sel.any():
        return None
    return np.median(lab[sel], axis=0)


def element_colors(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    text: bool,
    k: float,
    params: MeasureParams,
    valid: np.ndarray | None = None,
) -> ColorPair | None:
    la = region_color(a, mask_a, text=text, k=k, params=params, valid=valid)
    lb = region_color(b, mask_b, text=text, k=k, params=params, valid=valid)
    if la is None or lb is None:
        return None
    return ColorPair(la, lb, hex_from_lab(la), hex_from_lab(lb), delta_e76(la, lb))


def opacity_kappa(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, k: float, valid: np.ndarray | None = None
) -> tuple[float, float] | None:
    """Fit ``A − bg ≈ κ·(B − bg)`` over the mask (bg = ring median). Returns ``(κ, R²)``."""
    m = mask.copy()
    if valid is not None:
        m &= valid > 0
    if m.sum() < 20:
        return None
    ring = _ring(mask, max(3, int(round(4 * k))))
    if not ring.any():
        return None
    fa, fb = a.astype(np.float64), b.astype(np.float64)
    bg = np.median(fb[ring], axis=0)
    y = (fa[m] - bg).ravel()
    x = (fb[m] - bg).ravel()
    den = float(x @ x)
    if den < 1e-6:
        return None
    kappa = float(x @ y) / den
    sse = float(((y - kappa * x) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 1e-9 else 0.0
    # Hue check: a dimmed element keeps its hue (every channel scales by the same κ). Blue →
    # darker blue on a light page fits the pixel regression well but predicts the wrong colour.
    med_a = np.median(fa[m], axis=0)
    med_b = np.median(fb[m], axis=0)
    pred = np.clip(bg + kappa * (med_b - bg), 0, 255)
    de = delta_e76(bgr_to_lab(pred.astype(np.uint8)), bgr_to_lab(med_a.astype(np.uint8)))
    if de > OPACITY_HUE_MAX_DE:
        r2 = min(r2, 0.0)
    return kappa, r2


# --------------------------------------------------------------------------------------------
# Blend coefficient
# --------------------------------------------------------------------------------------------


def _warp(img: np.ndarray, m: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    if np.allclose(m, IDENTITY):
        return img
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def blend_alpha(
    frame: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    cursor: np.ndarray | None = None,
    min_diff: float = BLEND_MIN_DIFF,
) -> tuple[float, float]:
    """``α`` and fit quality (1 − SSE/Σ‖B−A‖², clipped) over ``mask`` pixels."""
    return _blend_alpha_prepared(frame, _BlendRef.build(a, b, mask, min_diff), cursor)


@dataclass(slots=True)
class _BlendRef:
    """Frame-independent part of :func:`blend_alpha` (A, B−A and the A≠B mask)."""

    fa: np.ndarray
    d: np.ndarray
    sel: np.ndarray

    @classmethod
    def build(cls, a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_diff: float) -> _BlendRef:
        fa = a.astype(np.float32)
        d = b.astype(np.float32) - fa
        return cls(fa, d, mask & (np.abs(d).max(axis=2) > min_diff))


def _blend_alpha_prepared(
    frame: np.ndarray, ref: _BlendRef, cursor: np.ndarray | None
) -> tuple[float, float]:
    fa, d, sel = ref.fa, ref.d, ref.sel
    if cursor is not None:
        sel = sel & (cursor == 0)
    if sel.sum() < 5:
        return float("nan"), 0.0
    dv = d[sel]
    iv = frame.astype(np.float32)[sel] - fa[sel]
    den = float((dv * dv).sum())
    if den < 1e-6:
        return float("nan"), 0.0
    alpha = float((iv * dv).sum()) / den
    sse = float(((iv - alpha * dv) ** 2).sum())
    return alpha, float(np.clip(1.0 - sse / den, 0.0, 1.0))


def alpha_series(
    wf: WindowFrames,
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    warps_a: np.ndarray | None = None,
    warps_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """α(t) and quality for every window frame.

    ``warps_a[i]`` / ``warps_b[i]`` (crop px) move A / B (and the mask, which lives in the B
    frame) to frame i; None = static.
    """
    n = len(wf.crops)
    al = np.full(n, np.nan)
    q = np.zeros(n)
    shape = a.shape
    m_u8 = mask.astype(np.uint8)
    # Static elements (and frames whose warps repeat) share the frame-independent part.
    refs: dict[bytes, _BlendRef] = {}
    for i in range(n):
        wa = IDENTITY if warps_a is None else warps_a[i]
        wb = IDENTITY if warps_b is None else warps_b[i]
        key = np.asarray(wa, np.float64).tobytes() + np.asarray(wb, np.float64).tobytes()
        ref = refs.get(key)
        if ref is None:
            at = _warp(a, wa, shape)
            bt = _warp(b, wb, shape)
            mt = _warp(m_u8, wb, shape) > 0 if warps_b is not None else mask
            if warps_a is not None and warps_b is None:
                mt = _warp(m_u8, wa, shape) > 0
            ref = refs[key] = _BlendRef.build(at, bt, mt, BLEND_MIN_DIFF)
            if len(refs) > 4:  # moving element: warps rarely repeat, keep memory flat
                refs = {key: ref}
        al[i], q[i] = _blend_alpha_prepared(wf.crops[i], ref, wf.cursor_masks[i])
    return al, q


def backdrop_series(
    wf: WindowFrames, a: np.ndarray, mask: np.ndarray, overlay_alpha: float, darken: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Progress 0→1 from the per-frame mean ratio over backdrop pixels (§6.6 backdrop)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = len(wf.crops)
    prog = np.full(n, np.nan)
    q = np.zeros(n)
    for i in range(n):
        g = cv2.cvtColor(wf.crops[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        sel = mask.copy()
        if wf.cursor_masks[i] is not None:
            sel &= wf.cursor_masks[i] == 0
        if darken:
            sel &= ga > 20
        if sel.sum() < 20:
            continue
        if darken:
            r = (g[sel] + 1) / (ga[sel] + 1)
            al = 1.0 - float(np.median(r))
            spread = float(np.std(r))
        else:
            r = (g[sel] - ga[sel]) / np.maximum(255.0 - ga[sel], 1.0)
            al = float(np.median(r))
            spread = float(np.std(r))
        prog[i] = al / overlay_alpha if overlay_alpha > 1e-3 else 0.0
        q[i] = float(np.clip(1.0 - spread / 0.1, 0.0, 1.0))
    return prog, q


# --------------------------------------------------------------------------------------------
# Series assembly
# --------------------------------------------------------------------------------------------


def _endpoints(
    times: np.ndarray, vals: np.ndarray, ok: np.ndarray, span: SegmentSpan, guard_s: float
) -> tuple[float, float, float]:
    pre = ok & (times >= span.lo_s - 1e-9) & (times <= span.start_s - guard_s + 1e-9)
    post = ok & (times >= span.end_s + guard_s - 1e-9) & (times <= span.hi_s + 1e-9)
    inside = ok & (times >= span.lo_s - 1e-9) & (times <= span.hi_s + 1e-9)
    if not inside.any():
        return float("nan"), float("nan"), 0.0
    v0 = float(np.median(vals[pre])) if pre.any() else float(vals[inside][0])
    v1 = float(np.median(vals[post])) if post.any() else float(vals[inside][-1])
    parts = []
    if pre.sum() >= 2:
        parts.append(vals[pre] - v0)
    if post.sum() >= 2:
        parts.append(vals[post] - v1)
    std = float(np.concatenate(parts).std()) if parts else 0.0
    return v0, v1, std


def _series(
    element_id: str,
    prop: Property,
    span: SegmentSpan,
    times: np.ndarray,
    vals: np.ndarray,
    q: np.ndarray,
    is_dup: np.ndarray,
    unit,
    value_at,  # callable(level: float) -> Value  (level = α or opacity at an endpoint)
    guard_s: float = 1 / 30,
    notes: list[str] | None = None,
) -> PropertySeries | None:
    ok = np.isfinite(vals)
    v0, v1, std = _endpoints(times, vals, ok, span, guard_s)
    if not np.isfinite(v0) or not np.isfinite(v1):
        return None
    sel = ok & ~is_dup & (times >= span.lo_s - 1e-9) & (times <= span.hi_s + 1e-9)
    return PropertySeries(
        element_id=element_id, property=prop, segment=span.segment_id,
        times=times[sel].copy(), values=vals[sel].copy(), quality=q[sel].copy(),
        v_start=v0, v_end=v1, unit=unit,
        from_value=value_at(v0), to_value=value_at(v1), stable_std=std,
        notes=list(notes or []),
    )  # fmt: skip


def _ratio_at(factor: float) -> Callable[[float], Value]:
    """Endpoint level (α or opacity) → ``RatioValue`` (× ``factor``, e.g. overlay alpha)."""

    def f(level: float) -> Value:
        return RatioValue(number=round(float(np.clip(level, 0, 1)) * factor, 4))

    return f


def _content_at(level: float) -> Value:
    return TextValue(text="changed content" if level >= 0.5 else "initial content")


def _color_at(pair: ColorPair) -> Callable[[float], Value]:
    def f(level: float) -> Value:
        return ColorValue(color=pair.hex_b if level >= 0.5 else pair.hex_a)

    return f


def _dimmed(kappa: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(alpha: np.ndarray) -> np.ndarray:
        return kappa + (1 - kappa) * alpha

    return f


@dataclass(slots=True)
class PhotometricResult:
    series: list[PropertySeries]
    colors: dict[str, ColorPair]  # element id → A/B colours (where measured)


def measure_photometric(
    elements: list[ElementCandidate],
    element_masks: dict[str, np.ndarray],
    boxes_px: dict[str, tuple[Rect, Rect]],
    warps_px: dict[str, np.ndarray],
    overlay: dict[str, tuple[str, float]],
    state_a: np.ndarray,
    state_b: np.ndarray,
    valid: np.ndarray,
    windows: list[tuple[WindowFrames, list[SegmentSpan]]],
    params: MeasureParams,
    tracks: dict[tuple[str, str], FrameTrack] | None = None,
) -> PhotometricResult:
    """Stage entry (§6.6): photometric properties for every element kind that needs them."""
    pp = params.photometric
    tracks = tracks or {}
    by_id = {e.id: e for e in elements}
    out: list[PropertySeries] = []
    colors: dict[str, ColorPair] = {}
    for e in elements:
        mask = element_masks[e.id]
        k = windows[0][0].scale.k if windows else 1.0
        if e.kind not in ("photometric", "content_change", "appear", "disappear", "backdrop"):
            continue
        # ---- decide property + endpoint values -------------------------------------------
        prop: Property
        unit = "ratio"
        notes: list[str] = []
        to_opacity: Callable[[np.ndarray], np.ndarray] | None = None  # α → opacity (dimmed)
        if e.kind == "backdrop":
            hex_c, a_ov = overlay.get(e.id, ("#000000", 0.0))
            prop = "opacity"
            notes.append(f"overlay {hex_c}")
            value_at = _ratio_at(a_ov)
        elif e.kind in ("appear", "disappear"):
            prop = "opacity"
            value_at = _ratio_at(1.0)
        elif e.kind == "content_change":
            prop = "content"
            unit = "text"
            value_at = _content_at
        else:  # photometric: dimmed opacity or colour
            kap = opacity_kappa(state_a, state_b, mask, k, valid)
            if kap is not None and kap[1] > pp.opacity_kappa_r2 and 0.0 < kap[0] < 0.97:
                prop = "opacity"
                notes.append("opacity_from_dimming")
                to_opacity = _dimmed(kap[0])
                value_at = _ratio_at(1.0)
            else:
                mask_a = _mask_to_a(mask, warps_px.get(e.id))
                pair = element_colors(
                    state_a, state_b, mask_a, mask, text=e.text_like, k=k, params=params,
                    valid=valid,
                )  # fmt: skip
                if pair is None:
                    continue
                colors[e.id] = pair
                prop = "color" if e.text_like else "background-color"
                unit = "color"
                if pair.delta_e < pp.min_delta_e:
                    notes.append("below_delta_e")
                value_at = _color_at(pair)
        # ---- per window α / opacity series -----------------------------------------------
        for wf, spans in windows:
            n = len(wf.crops)
            if e.kind == "backdrop":
                hex_c, a_ov = overlay.get(e.id, ("#000000", 0.0))
                excl = np.zeros_like(mask)
                for c in elements:
                    if c.parent_id == e.id:
                        excl |= element_masks[c.id]
                vals, q = backdrop_series(wf, state_a, mask & ~excl, a_ov, hex_c == "#000000")
            else:
                wa = wb = None
                ft = tracks.get((e.id, wf.segment_id))
                par_ft = tracks.get((e.parent_id, wf.segment_id)) if e.parent_id else None
                ox, oy = wf.crop_origin_css
                if e.kind in ("appear", "disappear") and ft is not None:
                    # template state (B for appear) → frame i
                    wb_list = [warp_css_to_px(ft.warps[i], wf.scale.k, (ox, oy)) for i in range(n)]
                    if e.kind == "appear":
                        wb = np.stack(wb_list)
                    else:
                        wa = np.stack(wb_list)
                elif par_ft is not None:
                    # child of a moving parent: A and B follow the parent's absolute motion
                    pa = _abs_warps(par_ft, tracks, by_id, e.parent_id, wf)
                    if pa is not None:
                        wa = np.stack(
                            [warp_css_to_px(pa[i], wf.scale.k, (ox, oy)) for i in range(n)]
                        )
                        w_par_ab = warps_px.get(e.parent_id, IDENTITY)
                        wb = np.stack([compose(wa[i], invert(w_par_ab)) for i in range(n)])
                vals, q = alpha_series(wf, state_a, state_b, mask, warps_a=wa, warps_b=wb)
                if e.kind == "disappear":
                    vals = 1.0 - vals
                elif to_opacity is not None:
                    vals = to_opacity(vals)
            for span in spans:
                s = _series(e.id, prop, span, wf.times, vals, q, wf.is_dup, unit, value_at,
                            notes=notes)  # fmt: skip
                if s is not None:
                    out.append(s)
    return PhotometricResult(series=out, colors=colors)


def _mask_to_a(mask_b: np.ndarray, warp_ab: np.ndarray | None) -> np.ndarray:
    """Element mask (B frame) → A frame through the element's absolute A→B warp."""
    if warp_ab is None or np.allclose(warp_ab, IDENTITY, atol=1e-3):
        return mask_b
    h, w = mask_b.shape
    out = cv2.warpAffine(mask_b.astype(np.uint8), warp_ab, (w, h),
                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)  # fmt: skip
    return out > 0


def _abs_warps(ft: FrameTrack, tracks, by_id, eid: str, wf: WindowFrames) -> np.ndarray | None:
    """Absolute (CSS) warps for ``eid``: tracks store child warps relative to the parent."""
    e = by_id.get(eid)
    if e is None:
        return None
    if e.parent_id is None or (e.parent_id, wf.segment_id) not in tracks:
        return ft.warps
    pa = _abs_warps(tracks[(e.parent_id, wf.segment_id)], tracks, by_id, e.parent_id, wf)
    if pa is None:
        return ft.warps
    return np.stack([compose(pa[i], ft.warps[i]) for i in range(len(ft.warps))])


def refine_appear_endpoints(series: list[PropertySeries], min_r2: float = 0.9) -> None:
    """Geometric start/end values of appearing/disappearing elements (in place).

    While opacity ≈ 0 an element cannot be tracked, so its translate/scale series starts at
    the first *visible* frame (e.g. −4.5 px of a −8 → 0 slide-in). Fade + move usually share
    one timing curve, so value is ~linear in opacity: fit ``v = c0 + c1·opacity`` over the
    visible frames and use ``c0`` as the value at opacity 0. Skipped if the fit is poor.
    """
    by_key: dict[tuple[str, str], list[PropertySeries]] = {}
    for s in series:
        by_key.setdefault((s.element_id, s.segment), []).append(s)
    for group in by_key.values():
        op = next((s for s in group if s.property == "opacity" and s.unit == "ratio"), None)
        if op is None or "opacity_from_dimming" in op.notes or not len(op.times):
            continue
        ends = [i for i, v in enumerate((op.v_start, op.v_end)) if v < 0.1]
        if len(ends) != 1:
            continue
        for s in group:
            if s.property not in ("translateX", "translateY", "scale", "scaleX", "scaleY"):
                continue
            common, i_op, i_s = np.intersect1d(np.round(op.times, 5), np.round(s.times, 5),
                                               return_indices=True)  # fmt: skip
            if common.size < 3:
                continue
            x = op.values[i_op]
            y = s.values[i_s]
            sel = (x >= 0.15) & (x <= 0.95)
            if sel.sum() < 3 or np.ptp(x[sel]) < 0.2:
                continue
            c1, c0 = np.polyfit(x[sel], y[sel], 1)
            pred = c0 + c1 * x[sel]
            sst = float(((y[sel] - y[sel].mean()) ** 2).sum())
            r2 = 1 - float(((y[sel] - pred) ** 2).sum()) / sst if sst > 1e-12 else 0.0
            if r2 < min_r2:
                continue
            val_cls = PxValue if s.unit == "px" else RatioValue
            if ends[0] == 0:
                s.v_start = float(c0)
                s.from_value = val_cls(number=round(float(c0), 4))
            else:
                s.v_end = float(c0)
                s.to_value = val_cls(number=round(float(c0), 4))
            s.notes.append("endpoint extrapolated from opacity (assumes shared timing)")
