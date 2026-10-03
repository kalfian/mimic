"""A4 — state frames and element decomposition (PLAN §6.5).

Pipeline inside one interaction crop (all images are crop-sized, analysis px):

1. :func:`state_frames` — per-pixel (cursor-masked) median of the last stable frames before the
   forward segment (A) and the first stable frames after it (B); round trip → B = peak frame.
2. :func:`detect_elements` — change mask ``max_c|B−A| > 14`` (minus cursor, open 3, close 9) →
   regions → per region: appear / disappear / backdrop tests, otherwise the dominant A→B
   transform (ECC from identity / phase / ORB guesses) and the hierarchical residual: what the
   parent transform does not explain becomes children, aligned again with ``A'`` vs ``B`` so a
   child's warp is **relative to its parent** (CSS inheritance).
3. Sibling regions with the same transform are merged (the two edge bands + content of a
   translated card are one element). Expand/collapse (appear + content below shifted by its
   height) is tagged ``resize``.

Element boxes, warps and ids are converted to CSS px at the end (``ElementCandidate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.ir import ElementKind, SpecWarning
from app.models.measure import ElementCandidate, Rect, WindowFrames
from app.pipeline.params import MeasureParams
from app.pipeline.track import (
    IDENTITY,
    box_from_union,
    compose,
    decompose,
    estimate_transform,
    invert,
    is_identity,
    map_box,
    warp_px_to_css,
)

#: Floor for the soft (grouping) change threshold. TODO(calibrate): module-local.
SOFT_MIN_THRESHOLD = 5

# --------------------------------------------------------------------------------------------
# State frames
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class StateFrames:
    a: np.ndarray  # BGR uint8, crop px
    b: np.ndarray
    valid: np.ndarray  # uint8 255 where neither A nor B had a cursor-covered (unknown) pixel
    a_idx: list[int]
    b_idx: list[int]
    peak_idx: int | None = None  # round trip: frame used as B
    post: np.ndarray | None = None  # round trip: state after the motion
    post_idx: list[int] = field(default_factory=list)


def nanmedian_axis0(stack: np.ndarray) -> np.ndarray:
    """``np.nanmedian(stack, axis=0)`` for small stacks, without NumPy's masked-array path.

    NumPy's ``nanmedian`` on a short first axis goes through ``numpy.ma`` and is ~10× slower
    (Phase 5 perf pass). Sorting puts NaN last, so the median of the ``n`` finite values sits
    at ranks ``(n−1)//2`` and ``n//2``. All-NaN positions return NaN like ``nanmedian``.
    """
    s = np.sort(stack, axis=0)
    n = np.sum(~np.isnan(stack), axis=0)
    lo = np.maximum((n - 1) // 2, 0)[None]
    hi = np.maximum(n // 2 - (n == 0), 0)[None]
    med = (np.take_along_axis(s, lo, 0)[0] + np.take_along_axis(s, hi, 0)[0]) / 2
    return np.where(n > 0, med, np.nan).astype(stack.dtype, copy=False)


def masked_median(
    frames: list[np.ndarray], masks: list[np.ndarray | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median ignoring cursor pixels. Returns ``(image, valid uint8)``."""
    stack = np.stack(frames).astype(np.float32)
    plain = np.median(stack, axis=0)
    if all(m is None for m in masks):
        return plain.round().astype(np.uint8), np.full(stack.shape[1:3], 255, np.uint8)
    cov = np.stack([np.zeros(stack.shape[1:3], bool) if m is None else m > 0 for m in masks])
    stack[cov] = np.nan
    valid = ~cov.all(axis=0)
    med = nanmedian_axis0(stack) if cov.any() else plain  # all-NaN pixels → filled below
    med = np.where(valid[..., None], med, plain)
    return med.round().clip(0, 255).astype(np.uint8), (valid * 255).astype(np.uint8)


def _diff_score(frame: np.ndarray, ref: np.ndarray, mask: np.ndarray | None) -> float:
    d = np.abs(frame.astype(np.int16) - ref.astype(np.int16)).max(axis=2).astype(np.float32)
    if mask is not None:
        d[mask > 0] = 0
    return float(d.mean())


def state_frames(
    wf: WindowFrames,
    start_s: float,
    end_s: float,
    params: MeasureParams,
    *,
    round_trip: bool = False,
    guard_s: float = 1 / 60,
) -> StateFrames:
    """A/B state images for a segment ``[start_s, end_s]`` inside window ``wf``."""
    n_med = params.regions.state_median_frames
    t = wf.times
    pre = [i for i in range(len(t)) if t[i] <= start_s - guard_s + 1e-9]
    post = [i for i in range(len(t)) if t[i] >= end_s + guard_s - 1e-9]
    if not pre:
        pre = [0]
    if not post:
        post = [len(t) - 1]
    a_idx = pre[-n_med:]
    a, va = masked_median([wf.crops[i] for i in a_idx], [wf.cursor_masks[i] for i in a_idx])
    if round_trip:
        inside = [i for i in range(len(t)) if start_s - 1e-9 <= t[i] <= end_s + 1e-9] or list(
            range(len(t))
        )
        scores = [_diff_score(wf.crops[i], a, wf.cursor_masks[i]) for i in inside]
        peak = inside[int(np.argmax(scores))]
        b = wf.crops[peak].copy()
        cm = wf.cursor_masks[peak]
        vb = np.full(b.shape[:2], 255, np.uint8) if cm is None else cv2.bitwise_not(cm)
        c_idx = post[:n_med]
        c, _ = masked_median([wf.crops[i] for i in c_idx], [wf.cursor_masks[i] for i in c_idx])
        return StateFrames(a, b, cv2.bitwise_and(va, vb), a_idx, [peak], peak, c, c_idx)
    b_idx = post[:n_med]
    b, vb = masked_median([wf.crops[i] for i in b_idx], [wf.cursor_masks[i] for i in b_idx])
    return StateFrames(a, b, cv2.bitwise_and(va, vb), a_idx, b_idx)


# --------------------------------------------------------------------------------------------
# Internal decomposition tree (crop px)
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Node:
    kind: ElementKind
    box_a: tuple[float, float, float, float]  # x, y, w, h in A (crop px)
    box_b: tuple[float, float, float, float]
    warp_abs: np.ndarray  # A→B crop px, absolute
    warp_rel: np.ndarray  # relative to the parent (== abs at top level)
    mask: np.ndarray  # bool, changed pixels attributed to this node (B frame)
    raw_mask: np.ndarray  # bool, un-closed changed pixels (text-like test)
    energy: float
    rho: float = 1.0
    depth: int = 0
    children: list[Node] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    overlay: str | None = None  # backdrop: "#000000" / "#FFFFFF"
    overlay_alpha: float | None = None
    parent: Node | None = field(default=None, repr=False)
    resize_of: Node | None = field(default=None, repr=False)  # the band that pushed this node

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def _odd(v: float) -> int:
    k = max(1, int(round(v)))
    return k if k % 2 == 1 else k + 1


def _boxes_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: float) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return dx < gap and dy < gap


def components(mask: np.ndarray, min_area_px: float, merge_gap_px: float) -> list[np.ndarray]:
    """Connected components (area ≥ min) whose boxes are merged when closer than the gap.

    Returns a list of boolean masks (crop size), largest first.
    """
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    items: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for j in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[j])
        if area < min_area_px:
            continue
        items.append(((x, y, x + w, y + h), [j]))
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _boxes_close(items[i][0], items[j][0], merge_gap_px):
                    a, b = items[i][0], items[j][0]
                    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    items[i] = (box, items[i][1] + items[j][1])
                    items.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = [np.isin(labels, ids) for _, ids in items]
    out.sort(key=lambda x: -int(x.sum()))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _maxdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)


# --------------------------------------------------------------------------------------------
# Decomposer
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Ctx:
    params: MeasureParams
    k: float
    b: np.ndarray
    valid: np.ndarray  # uint8
    frame_area_px: float
    shape: tuple[int, int]

    @property
    def thr(self) -> int:
        return self.params.regions.change_threshold


def _clean(ctx: _Ctx, raw: np.ndarray) -> np.ndarray:
    p = ctx.params.regions
    m = raw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((p.open_ksize, p.open_ksize), np.uint8))
    ck = _odd(p.close_ksize)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    return m > 0


def _ring_median(img_g: np.ndarray, box: tuple[int, int, int, int], pad: int) -> float | None:
    x0, y0, x1, y1 = box
    h, w = img_g.shape
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = False
    vals = img_g[Y0:Y1, X0:X1][ring]
    return float(np.median(vals)) if vals.size else None


def _bg_like(img_g: np.ndarray, mask: np.ndarray, box, ctx: _Ctx) -> bool:
    """Flat inside the mask and equal to the surrounding ring → background (§6.5 appear)."""
    vals = img_g[mask]
    if vals.size == 0:
        return False
    if float(vals.std()) >= ctx.params.regions.appear_bg_std_max:
        return False
    ring = _ring_median(img_g, box, max(2, int(round(ctx.params.regions.region_pad_css * ctx.k))))
    return ring is not None and abs(float(np.median(vals)) - ring) < 8.0


def _new_node(
    ctx: _Ctx,
    kind: ElementKind,
    mask: np.ndarray,
    raw: np.ndarray,
    energy_map: np.ndarray,
    depth: int,
    box_a=None,
    box_b=None,
    warp_abs=None,
    warp_rel=None,
    rho: float = 1.0,
) -> Node:
    bb = _bbox(mask)
    assert bb is not None
    box = (float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1]))
    return Node(
        kind=kind,
        box_a=box_a or box,
        box_b=box_b or box,
        warp_abs=IDENTITY.copy() if warp_abs is None else warp_abs,
        warp_rel=IDENTITY.copy() if warp_rel is None else warp_rel,
        mask=mask,
        raw_mask=raw & mask,
        energy=float(energy_map[mask].sum()),
        rho=rho,
        depth=depth,
    )


def _ring_band(shape, box: tuple[float, float, float, float], width: int, pad: int) -> np.ndarray:
    """Band of ``width`` px inside the box edge plus ``pad`` px outside (uint8)."""
    h, w = shape
    x, y, bw, bh = box
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = int(math.ceil(x + bw)), int(math.ceil(y + bh))
    m = np.zeros((h, w), np.uint8)
    m[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)] = 255
    ix0, iy0, ix1, iy1 = x0 + width, y0 + width, x1 - width, y1 - width
    if ix1 > ix0 and iy1 > iy0:
        m[iy0:iy1, ix0:ix1] = 0
    return m


def _decompose(
    ctx: _Ctx,
    a_ref: np.ndarray,
    mask: np.ndarray,
    depth: int,
    exclude: np.ndarray | None = None,
) -> Node | None:
    """Classify one region (A_ref vs B) and recurse into its residual."""
    p = ctx.params.regions
    bb = _bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    h, w = ctx.shape
    pad = max(2, int(round(p.region_pad_css * ctx.k)))
    pbox = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    ga, gb = _gray(a_ref), _gray(ctx.b)
    diff = _maxdiff(a_ref, ctx.b).astype(np.float32)
    raw_all = (diff > ctx.thr) & (ctx.valid > 0)

    # --- appear / disappear -----------------------------------------------------------------
    if _bg_like(ga, mask, bb, ctx) and not _bg_like(gb, mask, bb, ctx):
        return _new_node(ctx, "appear", mask, raw_all, diff, depth)
    if _bg_like(gb, mask, bb, ctx) and not _bg_like(ga, mask, bb, ctx):
        return _new_node(ctx, "disappear", mask, raw_all, diff, depth)

    # --- backdrop ---------------------------------------------------------------------------
    if depth == 0 and mask.sum() > p.backdrop_min_frac * ctx.frame_area_px:
        node = _try_backdrop(ctx, a_ref, mask, raw_all, diff)
        if node is not None:
            return node

    # --- dominant transform -----------------------------------------------------------------
    # Hypotheses: (1) whole padded region; (2) depth 0: the region's boundary band — a parent's
    # motion is what moves its outline; depth ≥ 1: the region interior (inset 10 %, no pad) —
    # a clipped child (zooming image) has a static outline but moving content.
    tmask = np.full((h, w), 255, np.uint8)
    if exclude is not None:
        tmask[exclude] = 0
    box_xywh = (pbox[0], pbox[1], pbox[2] - pbox[0], pbox[3] - pbox[1])
    bw, bh = x1 - x0, y1 - y0
    cands = [estimate_transform(a_ref, ctx.b, box_xywh, ctx.params, mask=tmask, valid=ctx.valid)]
    band_w = max(3, int(round(0.08 * min(bw, bh))))
    inset = (x0 + int(0.1 * bw), y0 + int(0.1 * bh), int(0.8 * bw), int(0.8 * bh))
    if depth == 0 and min(bw, bh) > 4 * band_w:
        band = cv2.bitwise_and(_ring_band(ctx.shape, (x0, y0, bw, bh), band_w, pad), tmask)
        cands.append(estimate_transform(a_ref, ctx.b, box_xywh, ctx.params, mask=band,
                                        valid=ctx.valid))  # fmt: skip
    elif depth > 0 and min(inset[2], inset[3]) >= 16:
        cands.append(estimate_transform(a_ref, ctx.b, inset, ctx.params, mask=tmask,
                                        valid=ctx.valid))  # fmt: skip
    interior = np.zeros((h, w), bool)
    interior[inset[1] : inset[1] + inset[3], inset[0] : inset[0] + inset[2]] = True
    best = None
    for al in cands:
        warped = cv2.warpAffine(a_ref, al.warp, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
        res_raw = (_maxdiff(warped, ctx.b) > ctx.thr) & (ctx.valid > 0)
        u = Rect(x0, y0, bw, bh)
        ident = is_identity(al.warp, p.identity_translate_px * ctx.k, p.identity_scale)
        box_a = u if ident else box_from_union(u, al.warp)
        box_b = u if ident else map_box(al.warp, box_a)
        if depth == 0:
            edge = _ring_band(ctx.shape, (box_b.x, box_b.y, box_b.w, box_b.h), band_w, 0) > 0
            score = int((res_raw & edge).sum())
        else:
            score = int((res_raw & interior & mask).sum())
        if best is None or score < best[0] * 0.8:
            best = (score, al, ident)
    assert best is not None
    _, al, ident = best

    if ident:
        kind: ElementKind = "photometric" if al.rho >= p.content_change_rho else "content_change"
        return _new_node(ctx, kind, mask, raw_all, diff, depth, warp_abs=al.warp,
                         warp_rel=al.warp, rho=al.rho)  # fmt: skip
    if al.rho < p.content_change_rho:
        return _new_node(ctx, "content_change", mask, raw_all, diff, depth, rho=al.rho)
    return _transform_node(ctx, a_ref, mask, depth, al.warp, al.rho, raw_all, diff)


def _transform_node(
    ctx: _Ctx,
    a_ref: np.ndarray,
    mask: np.ndarray,
    depth: int,
    warp: np.ndarray,
    rho: float,
    raw_all: np.ndarray | None = None,
    diff: np.ndarray | None = None,
    union_box: Rect | None = None,
) -> Node:
    """Transform node with a known warp; what the warp leaves unexplained inside the element's
    B box becomes children, decomposed against ``A' = warp(A_ref)`` (relative transforms)."""
    p = ctx.params.regions
    h, w = ctx.shape
    if diff is None:
        diff = _maxdiff(a_ref, ctx.b).astype(np.float32)
    if raw_all is None:
        raw_all = (diff > ctx.thr) & (ctx.valid > 0)
    if union_box is None:
        bb = _bbox(mask)
        assert bb is not None
        union_box = Rect(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
    box_a = box_from_union(union_box, warp)
    t = warp[:, 2]
    if abs(t[0]) >= union_box.w * 0.75 or abs(t[1]) >= union_box.h * 0.75:
        # Displacement larger than the region: the change box holds one end of the move only
        # (A side if A has content there, else B side), not the union of both.
        box_a = (
            union_box if _is_a_side(ctx, a_ref, mask, warp) else map_box(invert(warp), union_box)
        )
    box_b = map_box(warp, box_a)
    if (
        depth > 0
        and decompose(warp).scale > 1.0 + p.identity_scale
        and _clipped(a_ref, union_box, ctx)
    ):
        # Zoom inside an overflow:hidden parent: the visible box is the clip box in both states.
        box_a = box_b = union_box
    warped = cv2.warpAffine(a_ref, warp, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
    res_raw = (_maxdiff(warped, ctx.b) > ctx.thr) & (ctx.valid > 0)
    node = _new_node(
        ctx, "transform", mask, raw_all, diff, depth,
        box_a=(box_a.x, box_a.y, box_a.w, box_a.h), box_b=(box_b.x, box_b.y, box_b.w, box_b.h),
        warp_abs=warp, warp_rel=warp, rho=rho,
    )  # fmt: skip
    inside = np.zeros((h, w), bool)
    bx0, by0 = int(max(0, math.floor(box_b.x))), int(max(0, math.floor(box_b.y)))
    bx1, by1 = int(min(w, math.ceil(box_b.x2))), int(min(h, math.ceil(box_b.y2)))
    inside[by0:by1, bx0:bx1] = True
    residual = _clean(ctx, res_raw & inside)
    ratio = residual.sum() / max(1, mask.sum())
    if ratio < p.residual_rigid_ratio or depth + 1 > p.max_depth:
        return node
    min_area = p.min_area_px2 * ctx.k**2
    for comp in components(residual, min_area, p.merge_gap_css * ctx.k):
        child = _decompose(ctx, warped, comp, depth + 1)
        if child is None:
            continue
        child.parent = node  # box_a / warp_abs are fixed up by _finalize (local → A coords)
        node.children.append(child)
    return node


def _int_box(r: Rect) -> tuple[int, int, int, int]:
    return (int(math.floor(r.x)), int(math.floor(r.y)), int(math.ceil(r.x2)), int(math.ceil(r.y2)))


def _is_a_side(ctx: _Ctx, a_ref: np.ndarray, mask: np.ndarray, warp: np.ndarray) -> bool:
    """For a long move: does ``mask`` sit where the content *was* (A side) rather than where
    it went (B side)? A side ⇔ ``A[p] ≈ B[warp(p)]`` explains the region better than
    ``B[p] ≈ A[warp⁻¹(p)]``."""
    h, w = ctx.shape
    flags = cv2.INTER_LINEAR
    # B sampled at warp(p) (A-side hypothesis) / A sampled at warp⁻¹(p) (B-side hypothesis)
    b_at = cv2.warpAffine(ctx.b, warp, (w, h), flags=flags | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
    a_at = cv2.warpAffine(a_ref, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    sel = mask & (ctx.valid > 0)
    err_a = float(_maxdiff(a_ref, b_at)[sel].mean()) if sel.any() else 0.0
    err_b = float(_maxdiff(ctx.b, a_at)[sel].mean()) if sel.any() else 0.0
    return err_a <= err_b


def _clipped(a_ref: np.ndarray, u: Rect, ctx: _Ctx) -> bool:
    """True if A already shows content right at the rim of the change box ``u`` (a clipped
    child zooming in) instead of background (a free element growing into ``u``)."""
    g = _gray(a_ref)
    h, w = g.shape
    x0, y0 = int(max(0, math.floor(u.x))), int(max(0, math.floor(u.y)))
    x1, y1 = int(min(w, math.ceil(u.x2))), int(min(h, math.ceil(u.y2)))
    r = max(2, int(round(2 * ctx.k)))
    if x1 - x0 <= 4 * r or y1 - y0 <= 4 * r:
        return False
    inner = np.zeros((h, w), bool)
    inner[y0:y1, x0:x1] = True
    inner[y0 + r : y1 - r, x0 + r : x1 - r] = False
    outer = np.zeros((h, w), bool)
    outer[max(0, y0 - r) : min(h, y1 + r), max(0, x0 - r) : min(w, x1 + r)] = True
    outer[y0:y1, x0:x1] = False
    if not outer.any():
        return True
    rim, ring = g[inner], g[outer]
    return abs(float(np.median(rim)) - float(np.median(ring))) > 8 or float(rim.std()) > 6


def _finalize(node: Node, parent_abs: np.ndarray | None = None) -> None:
    """Children were decomposed against their parent's warped A (``A'``): convert their
    ``box_a`` to A coords and compose absolute warps, top-down."""
    if parent_abs is None:
        node.warp_abs = node.warp_rel.copy() if node.kind != "backdrop" else IDENTITY.copy()
    else:
        node.warp_abs = compose(node.warp_rel, parent_abs)
        ca = map_box(invert(parent_abs), Rect(*node.box_a))
        node.box_a = (ca.x, ca.y, ca.w, ca.h)
    for c in node.children:
        _finalize(c, node.warp_abs)


def _try_backdrop(
    ctx: _Ctx, a_ref: np.ndarray, mask: np.ndarray, raw_all: np.ndarray, diff: np.ndarray
) -> Node | None:
    """Uniform brightness ratio over most of a large region → backdrop (+ residual children)."""
    p = ctx.params.regions
    ga, gb = _gray(a_ref), _gray(ctx.b)
    sel = mask & (ctx.valid > 0)
    ratio = (gb[sel] + 1.0) / (ga[sel] + 1.0)
    if ratio.size == 0:
        return None
    m = float(np.median(ratio))
    inl = np.abs(ratio - m) < p.backdrop_ratio_std_max * 2
    if inl.mean() < 0.5 or float(ratio[inl].std()) >= p.backdrop_ratio_std_max:
        return None
    h, w = ctx.shape
    full = np.ones((h, w), bool)
    node = _new_node(ctx, "backdrop", full, raw_all, diff, 0)
    node.overlay = "#000000" if m < 1 else "#FFFFFF"
    if m < 1:
        node.overlay_alpha = float(np.clip(1 - m, 0, 1))
    else:  # lighten: B = A + α(255 − A)
        sa = ga[sel][inl]
        sb = gb[sel][inl]
        node.overlay_alpha = float(np.clip(np.median((sb - sa) / np.maximum(255 - sa, 1)), 0, 1))
    # what the overlay does not explain → children (panel / dialog)
    if m < 1:
        pred = (a_ref.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    else:
        al = node.overlay_alpha
        pred = (a_ref.astype(np.float32) * (1 - al) + 255 * al).clip(0, 255).astype(np.uint8)
    res = _clean(ctx, (_maxdiff(pred, ctx.b) > ctx.thr) & (ctx.valid > 0))
    min_area = p.min_area_px2 * ctx.k**2
    for comp in components(res, min_area, p.merge_gap_css * ctx.k):
        child = _decompose(ctx, pred, comp, 1)
        if child is None:
            continue
        if child.kind in ("content_change", "photometric") and child.rho < p.content_change_rho:
            child.kind = "appear"  # new surface on top of the dimmed page
        child.parent = node
        node.children.append(child)
    node.mask = mask
    return node


def _similar(a: np.ndarray, b: np.ndarray, t_tol: float, s_tol: float) -> bool:
    da, db = decompose(a), decompose(b)
    return (
        abs(a[0, 2] - b[0, 2]) <= t_tol
        and abs(a[1, 2] - b[1, 2]) <= t_tol
        and abs(da.scale - db.scale) <= s_tol
    )


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class RegionResult:
    states: StateFrames
    change_mask: np.ndarray  # bool, cleaned change mask (crop px)
    elements: list[ElementCandidate]
    #: Per element id: boolean changed-pixel mask (crop px, B frame) and the A/B boxes (crop px).
    element_masks: dict[str, np.ndarray]
    boxes_px: dict[str, tuple[Rect, Rect]]
    warps_px: dict[str, np.ndarray]  # absolute A→B crop-px warp per element
    overlay: dict[str, tuple[str, float]]  # backdrop id → (hex, alpha)
    #: resize element id → ids of the siblings it pushed (their translateY = its height change)
    resize_links: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[SpecWarning] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)


def text_like(raw_mask: np.ndarray, box: tuple[int, int, int, int], k: float, params) -> bool:
    """Fill ratio < 0.35 and thin strokes (max distance transform < 4 CSS px) (§6.5.6)."""
    x0, y0, x1, y1 = box
    sub = raw_mask[y0:y1, x0:x1].astype(np.uint8)
    if sub.size == 0 or sub.sum() == 0:
        return False
    fill = float(sub.mean())
    dt = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 3)
    if float(dt.max()) >= params.regions.text_stroke_max_css * k:
        return False
    # Thin strokes + sparse fill (plan), or thin strokes split into several glyph blobs (bold
    # text in a tight box easily exceeds the 35 % fill ratio).
    n_glyphs = cv2.connectedComponents(sub, connectivity=8)[0] - 1
    return fill < params.regions.text_fill_ratio_max or n_glyphs >= 3


def detect_elements(
    wf: WindowFrames,
    states: StateFrames,
    frame_css: tuple[float, float],
    params: MeasureParams,
) -> RegionResult:
    """Change mask → decomposed element tree → ``ElementCandidate`` list (CSS px)."""
    p = params.regions
    k = wf.scale.k
    a, b = states.a, states.b
    h, w = a.shape[:2]
    ctx = _Ctx(
        params=params, k=k, b=b, valid=states.valid,
        frame_area_px=frame_css[0] * frame_css[1] * k * k, shape=(h, w),
    )  # fmt: skip
    raw = (_maxdiff(a, b) > p.change_threshold) & (states.valid > 0)
    mask = _clean(ctx, raw)
    regions = components(mask, p.min_area_px2 * k * k, p.merge_gap_css * k)

    # containment: regions inside a bigger region's box are handled by its residual recursion
    boxes = [_bbox(r) for r in regions]
    top: list[int] = []
    nested: dict[int, list[int]] = {}
    for i, bi in enumerate(boxes):
        ri = Rect(bi[0], bi[1], bi[2] - bi[0], bi[3] - bi[1])
        parent = None
        for j in top:
            bj = boxes[j]
            rj = Rect(bj[0], bj[1], bj[2] - bj[0], bj[3] - bj[1])
            if ri.contained_fraction(rj) >= p.containment_parent:
                parent = j
                break
        if parent is None:
            top.append(i)
        else:
            nested.setdefault(parent, []).append(i)

    nodes: list[Node] = []
    for i in top:
        excl = None
        if nested.get(i):
            excl = np.zeros((h, w), bool)
            for j in nested[i]:
                excl |= cv2.dilate(regions[j].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        reg_mask = regions[i].copy()
        for j in nested.get(i, []):
            reg_mask |= regions[j]
        node = _decompose(ctx, a, reg_mask, 0, exclude=excl)
        if node is not None:
            nodes.append(node)

    nodes = _merge_same_transform(ctx, a, nodes)
    soft = soft_change_mask(a, b, states.valid, params)
    nodes = _group_surfaces(ctx, nodes, soft)
    for root in nodes:
        _finalize(root)
    nodes = _tag_resize(ctx, a, nodes)

    # flatten, rank by energy, cap
    flat = [n for root in nodes for n in root.walk()]
    flat.sort(key=lambda n: -n.energy)
    warnings: list[SpecWarning] = []
    if len(flat) > p.max_elements:
        keep = set(map(id, flat[: p.max_elements]))
        flat = flat[: p.max_elements]
        for n in flat:  # re-parent to the nearest kept ancestor
            par = n.parent
            while par is not None and id(par) not in keep:
                par = par.parent
            n.parent = par
        warnings.append(
            SpecWarning(
                code="elements_truncated",
                severity="info",
                message=f"Only the {p.max_elements} most significant changed elements are "
                "reported.",
            )
        )
    ids = {id(n): f"e{i + 1}" for i, n in enumerate(flat)}
    ox, oy = wf.crop_css.x, wf.crop_css.y
    elements: list[ElementCandidate] = []
    masks: dict[str, np.ndarray] = {}
    boxes_px: dict[str, tuple[Rect, Rect]] = {}
    warps_px: dict[str, np.ndarray] = {}
    overlay: dict[str, tuple[str, float]] = {}
    for n in flat:
        eid = ids[id(n)]
        ra, rb = Rect(*n.box_a), Rect(*n.box_b)
        to_css = lambda r: Rect(ox + r.x / k, oy + r.y / k, r.w / k, r.h / k)  # noqa: E731
        bb = _bbox(n.raw_mask if n.raw_mask.any() else n.mask)
        tl = (
            bool(bb)
            and n.kind in ("photometric", "content_change")
            and text_like(n.raw_mask, bb, k, params)
        )
        parent_id = ids.get(id(n.parent)) if n.parent is not None else None
        if parent_id is None and n.parent is None:
            parent_id = _containing(n, flat, ids, p.containment_parent)
        elements.append(
            ElementCandidate(
                id=eid,
                bbox_a=to_css(ra),
                bbox_b=to_css(rb),
                parent_id=parent_id,
                kind=n.kind,
                change_energy=n.energy / (k * k),
                text_like=tl,
                warp_ab=warp_px_to_css(n.warp_rel, k, (ox, oy)),
                notes=list(n.notes),
            )
        )
        masks[eid] = n.mask
        boxes_px[eid] = (ra, rb)
        warps_px[eid] = n.warp_abs
        if n.kind == "backdrop" and n.overlay is not None:
            overlay[eid] = (n.overlay, float(n.overlay_alpha or 0.0))
    _break_cycles(elements)
    links: dict[str, list[str]] = {}
    for n in flat:
        if n.resize_of is not None and id(n.resize_of) in ids:
            links.setdefault(ids[id(n.resize_of)], []).append(ids[id(n)])
    return RegionResult(
        states=states,
        change_mask=mask,
        elements=elements,
        element_masks=masks,
        boxes_px=boxes_px,
        warps_px=warps_px,
        overlay=overlay,
        resize_links=links,
        warnings=warnings,
        nodes=nodes,
    )


def soft_change_mask(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, params: MeasureParams
) -> np.ndarray:
    """Low-threshold change mask (half the change threshold, opened). Faint surfaces — a white
    menu panel on a light page — only show up here; used for grouping and crop checks."""
    thr = max(SOFT_MIN_THRESHOLD, params.regions.change_threshold // 2)
    m = ((_maxdiff(a, b) > thr) & (valid > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


def _group_surfaces(ctx: _Ctx, nodes: list[Node], soft: np.ndarray) -> list[Node]:
    """Appear (or disappear) regions lying on one faint connected surface are one element:
    the items of a menu appear together with its (low-contrast) panel."""
    n_lab, labels = cv2.connectedComponents(soft.astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        return nodes
    out: list[Node] = []
    buckets: dict[tuple[str, int], list[Node]] = {}
    for n in nodes:
        if n.kind not in ("appear", "disappear"):
            out.append(n)
            continue
        labs = labels[n.mask & soft]
        labs = labs[labs > 0]
        if labs.size == 0:
            out.append(n)
            continue
        lab = int(np.bincount(labs).argmax())
        buckets.setdefault((n.kind, lab), []).append(n)
    for (kind, lab), group in buckets.items():
        surface = labels == lab
        if len(group) == 1 and int(surface.sum()) < 2 * int(group[0].mask.sum()):
            out.append(group[0])
            continue
        mask = surface.copy()
        for g in group:
            mask |= g.mask
        node = _new_node(ctx, kind, mask, group[0].raw_mask | mask,
                         np.zeros(ctx.shape, np.float32), 0)  # fmt: skip
        node.energy = float(sum(g.energy for g in group))
        out.append(node)
    return out


def _break_cycles(elements: list[ElementCandidate]) -> None:
    """Drop parent links that would form a cycle (containment fallback is heuristic)."""
    by_id = {e.id: e for e in elements}
    for e in elements:
        seen = {e.id}
        cur = e.parent_id
        while cur is not None and cur in by_id:
            if cur in seen:
                e.parent_id = None
                break
            seen.add(cur)
            cur = by_id[cur].parent_id
        if e.parent_id is not None and e.parent_id not in by_id:
            e.parent_id = None


def _containing(n: Node, flat: list[Node], ids: dict[int, str], frac: float) -> str | None:
    """Fallback hierarchy: smallest other element whose A box contains ≥ frac of this one."""
    me = Rect(*n.box_a)
    best: tuple[float, str] | None = None
    descendants = {id(x) for x in n.walk()}
    for o in flat:
        if id(o) in descendants or o.kind == "backdrop" and n.kind == "backdrop":
            continue
        ob = Rect(*o.box_a)
        if ob.area <= me.area:
            continue
        if me.contained_fraction(ob) >= frac and (best is None or ob.area < best[0]):
            best = (ob.area, ids[id(o)])
    return best[1] if best else None


def _merge_same_transform(ctx: _Ctx, a: np.ndarray, nodes: list[Node]) -> list[Node]:
    """Merge top-level transform siblings that share one transform (§6.5: the edge bands and
    content pieces of one translated card are one element).

    The merged node keeps the parts' consensus warp (area-weighted) instead of re-estimating —
    a zooming child inside the union would otherwise dominate the estimate. A merge is refused
    if warping A by that transform would disturb unchanged pixels between the parts (a static
    element sitting in between means they are separate elements).
    """
    t_tol = max(1.0, 0.75 * ctx.k)
    roots = [n for n in nodes if n.kind == "transform"]
    others = [n for n in nodes if n.kind != "transform"]
    groups: list[list[Node]] = []
    for n in sorted(roots, key=lambda x: -int(x.mask.sum())):
        for g in groups:
            if _similar(g[0].warp_abs, n.warp_abs, t_tol, 0.01) and _gap_ok(ctx, a, g + [n]):
                g.append(n)
                break
        else:
            groups.append([n])

    # A region whose own estimate disagrees (e.g. a clipped zooming image: its own fit is a
    # compromise between the moving outline and the zooming content) joins a multi-part group
    # if the group's transform + residual children explain it at least as well.
    def consensus(g: list[Node]) -> np.ndarray:
        wts = np.array([float(n.mask.sum()) for n in g])
        return sum(n.warp_abs * wt for n, wt in zip(g, wts, strict=True)) / wts.sum()

    # consensus of the members that agreed on their own (joiners below must not shift it)
    warps = {id(g): consensus(g) for g in groups}

    def gbox(g: list[Node]) -> Rect:
        r: Rect | None = None
        for m in g:
            bb = _bbox(m.mask)
            q = Rect(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
            r = q if r is None else r.union(q)
        assert r is not None
        return r

    def try_join(n: Node, own: int) -> bool:
        nb = _bbox(n.mask)
        nr = Rect(nb[0], nb[1], nb[2] - nb[0], nb[3] - nb[1])
        size = int(n.mask.sum())
        for target in sorted(groups, key=lambda x: -sum(int(m.mask.sum()) for m in x)):
            if n in target:
                continue
            t_size = sum(int(m.mask.sum()) for m in target)
            # multi-part groups accept any region; a single region absorbs fragments lying
            # inside its A→B extent (a slice of a moving text line; the destination of a long
            # move) when it is much larger or its transform explains them almost fully
            pad = ctx.params.regions.region_pad_css * ctx.k
            extent = gbox(target)
            for m in target:
                extent = extent.union(Rect(*m.box_b))
            # where the moved content came from / went to: a flat element's vacated (or newly
            # covered) edge band can sit beyond the changed region of its moving part
            gw = warps[id(target)]
            extent = extent.union(map_box(invert(gw), extent)).union(map_box(gw, extent))
            inside = nr.contained_fraction(extent.padded(pad)) >= 0.9
            if len(target) == 1 and not inside:
                continue
            alt = _transform_node(ctx, a, n.mask, 0, warps[id(target)], n.rho)
            u_alt = _unexplained(ctx, a, alt, n.mask)
            if len(target) == 1 and t_size < 5 * size and u_alt > 0.1 * size:
                continue
            # consensus is a strong prior: allow a modestly worse pixel fit
            slack = max(0.5 * own, 0.02 * float(size), 10.0)
            if u_alt <= own + slack:
                target.append(n)
                return True
        return False

    # Pairs of lone transform regions that one transform explains jointly (the left/right edge
    # bands of a pressed button: each band alone cannot pin down the scale).
    tried: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        singles = [g for g in groups if len(g) == 1 and g[0].kind == "transform"]
        singles.sort(key=lambda g: -int(g[0].mask.sum()))
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                ni, nj = singles[i][0], singles[j][0]
                key = (id(ni), id(nj))
                if key in tried:
                    continue
                tried.add(key)
                union = ni.mask | nj.mask
                cand = _decompose(ctx, a, union, 0)
                if cand is None or cand.kind != "transform" or not _gap_ok(ctx, a, [cand]):
                    continue
                sep = _unexplained(ctx, a, ni, ni.mask) + _unexplained(ctx, a, nj, nj.mask)
                slack = max(10.0, 0.05 * float(union.sum()))
                if _unexplained(ctx, a, cand, union) <= 1.1 * sep + slack:
                    groups.remove(singles[i])
                    groups.remove(singles[j])
                    groups.append([cand])
                    warps[id(groups[-1])] = cand.warp_abs
                    changed = True
                    break
            if changed:
                break

    for g in sorted([g for g in groups if len(g) == 1], key=lambda g: int(g[0].mask.sum())):
        if try_join(g[0], _unexplained(ctx, a, g[0], g[0].mask)):
            groups.remove(g)
    # fragments that were not classified as transforms (a slice of a moved text line can look
    # like "disappear") join a moving group whose transform explains them
    for n in [o for o in others if o.kind != "backdrop"]:
        if try_join(n, int((n.mask & (ctx.valid > 0)).sum())):
            others.remove(n)
    out = list(others)
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        warp = warps[id(g)]
        union = np.zeros_like(g[0].mask)
        ubox: Rect | None = None
        for n in g:
            union |= n.mask
            bb = _bbox(n.mask)
            r = Rect(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
            ubox = r if ubox is None else ubox.union(r)
        rho = float(np.median([n.rho for n in g]))
        out.append(_transform_node(ctx, a, union, 0, warp, rho, union_box=ubox))
    return out


def _unexplained(ctx: _Ctx, a: np.ndarray, node: Node, mask: np.ndarray) -> int:
    """Changed pixels of ``mask`` that the node's transform tree does not reproduce."""
    h, w = ctx.shape

    def warp(m: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)  # fmt: skip

    pred = warp(node.warp_rel)

    def apply_children(n: Node, parent_abs: np.ndarray) -> None:
        for c in n.children:
            c_abs = compose(c.warp_rel, parent_abs)
            if c.kind == "transform":
                region = cv2.dilate(c.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                pred[region] = warp(c_abs)[region]
            apply_children(c, c_abs)

    apply_children(node, node.warp_rel)
    bad = (_maxdiff(pred, ctx.b) > ctx.thr) & mask & (ctx.valid > 0)
    return int(bad.sum())


def _gap_ok(ctx: _Ctx, a: np.ndarray, group: list[Node]) -> bool:
    """True if warping A by the group's transform leaves unchanged pixels (outside every part)
    inside the union box unchanged."""
    union = np.zeros_like(group[0].mask)
    for n in group:
        union |= n.mask
    bb = _bbox(union)
    if bb is None:
        return False
    x0, y0, x1, y1 = bb
    warp = group[0].warp_abs
    h, w = ctx.shape
    sub_a = a[y0:y1, x0:x1]
    sub_b = ctx.b[y0:y1, x0:x1]
    local = warp.copy()
    local[:, 2] += warp[:, :2] @ np.array([x0, y0]) - np.array([x0, y0])
    warped = cv2.warpAffine(sub_a, local, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
    static = ~cv2.dilate(union[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
        bool
    )
    static &= ctx.valid[y0:y1, x0:x1] > 0
    if static.sum() < 50:
        return True
    # disturbed = the warp creates a clear difference where A and B agreed (a static element
    # in between); resampling noise on content that really moved with the warp stays below
    d_w = _maxdiff(warped, sub_b).astype(np.int16)
    d_0 = _maxdiff(sub_a, sub_b).astype(np.int16)
    disturbed = (d_w > ctx.thr) & (d_w > d_0 + ctx.thr // 2) & static
    return float(disturbed.sum()) / float(static.sum()) < 0.02


def _band_has_content(ctx: _Ctx, band: np.ndarray) -> bool:
    """≥ 30 % of the band differs from the page background (border median of B)."""
    b = ctx.b.astype(np.int16)
    border = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    page = np.median(border, axis=0)
    differs = np.abs(b - page).max(axis=2) > SOFT_MIN_THRESHOLD
    sel = band & (ctx.valid > 0)
    return bool(sel.any()) and float(differs[sel].mean()) >= 0.3


def _page_color(ctx: _Ctx) -> np.ndarray:
    """Page background estimate: per-channel median of B's crop border."""
    b = ctx.b.astype(np.int16)
    border = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    return np.median(border, axis=0)


def _split_surfaces(ctx: _Ctx, a: np.ndarray, group: list[Node]) -> list[Node]:
    """Rows pushed by an expand/collapse are often low-contrast cards (white on a light page):
    only their text clears the change threshold, and identically moving rows merge into one
    region. In B, each card is a separate surface that differs (softly) from the page. Split /
    re-box the shifted group by those surfaces (Phase 4).

    A surface belongs to the group if it overlaps the group's changed pixels and the group's
    warp reproduces B on it (``|warp(A) − B|`` small), which rejects the newly opened panel.
    Nodes lying inside one surface collapse into that surface's node. Without ≥ 1 accepted
    surface the group is returned unchanged.
    """
    h, w = ctx.shape
    warp = group[0].warp_abs
    page = _page_color(ctx)
    differs = (np.abs(ctx.b.astype(np.int16) - page).max(axis=2) > SOFT_MIN_THRESHOLD) & (
        ctx.valid > 0
    )
    differs = cv2.morphologyEx(differs.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(differs, connectivity=4)
    if n_lab <= 1:
        return group
    gmask = np.zeros(ctx.shape, bool)
    for n in group:
        gmask |= n.mask
    warped = cv2.warpAffine(a, warp, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
    err = _maxdiff(warped, ctx.b)
    min_area = 400 * ctx.k * ctx.k  # ≥ 20×20 CSS px
    surfaces: list[tuple[int, int, int, int]] = []
    for j in range(1, n_lab):
        x, y, bw, bh, area = (int(v) for v in stats[j])
        if area < min_area:
            continue
        comp = labels == j
        if not (comp & gmask).any():
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue  # clipped by the crop: extent unknown
        sel = comp & (ctx.valid > 0)
        if float(np.mean(err[sel] > ctx.thr)) > 0.05:
            continue  # the shift does not explain it (new content, e.g. the opened panel)
        surfaces.append((x, y, x + bw, y + bh))
    if not surfaces:
        return group
    inv = invert(warp)
    out: list[Node] = []
    used: set[int] = set()
    for x0, y0, x1, y1 in sorted(surfaces, key=lambda r: (r[1], r[0])):
        box_b = Rect(x0, y0, x1 - x0, y1 - y0)
        members = [
            n for n in group
            if Rect(*n.box_b).contained_fraction(box_b.padded(2)) >= 0.5
            or (n.mask[y0:y1, x0:x1].sum() >= 0.5 * max(1, int(n.mask.sum())))
        ]  # fmt: skip
        region = np.zeros(ctx.shape, bool)
        region[y0:y1, x0:x1] = True
        mask = gmask & region
        if not mask.any():
            continue
        box_a = map_box(inv, box_b)
        node = _new_node(ctx, "transform", mask, mask, np.zeros(ctx.shape, np.float32), 0,
                         box_a=(box_a.x, box_a.y, box_a.w, box_a.h),
                         box_b=(box_b.x, box_b.y, box_b.w, box_b.h),
                         warp_abs=warp.copy(), warp_rel=warp.copy(),
                         rho=float(np.median([n.rho for n in group])))  # fmt: skip
        part = float(mask.sum()) / max(1.0, float(gmask.sum()))
        node.energy = float(sum(n.energy for n in group)) * part
        for n in members:
            used.add(id(n))
            for c in n.children:
                c.parent = node
                node.children.append(c)
        out.append(node)
    # group nodes not absorbed by any surface stay as they are
    out += [n for n in group if id(n) not in used]
    return out if len(out) >= 1 else group


def _tag_resize(ctx: _Ctx, a: np.ndarray, nodes: list[Node]) -> list[Node]:
    """Expand/collapse (§6.5): content shifted straight down by +h (or up by −h) next to a band
    of new (or vanished) content of height h → a ``resize`` element for that band.

    The opened panel is often low contrast (white on a light page) and its text lines are hard
    to classify, so the panel box is derived from the shifted siblings: it spans their width,
    starts at their original top and is ``h`` tall. Changed regions inside that band are folded
    into the resize element; the shifted siblings get the note ``caused_by_resize``.
    """
    tol = ctx.params.regions.resize_translate_tol_css * ctx.k
    min_shift = 4 * tol
    shifted = [
        n
        for n in nodes
        if n.kind == "transform"
        and abs(float(n.warp_abs[0, 2])) <= tol
        and abs(float(n.warp_abs[1, 2])) >= min_shift
        and abs(decompose(n.warp_abs).scale - 1) < 0.01
    ]
    if not shifted:
        return nodes
    # one band per distinct shift
    groups: list[list[Node]] = []
    for n in sorted(shifted, key=lambda x: float(x.warp_abs[1, 2])):
        ty = float(n.warp_abs[1, 2])
        if groups and abs(float(groups[-1][0].warp_abs[1, 2]) - ty) <= max(tol, 0.05 * abs(ty)):
            groups[-1].append(n)
        else:
            groups.append([n])
    out = [n for n in nodes]
    h_img, w_img = ctx.shape
    for gi, g in enumerate(groups):
        split = _split_surfaces(ctx, a, g)
        left = np.zeros(ctx.shape, bool)  # changed pixels the split surfaces did not keep
        if split is not g:
            for n in g:
                left |= n.mask
            for n in split:
                left &= ~n.mask
            out = [n for n in out if n not in g] + split
            g = groups[gi] = split
        ty = float(np.median([float(n.warp_abs[1, 2]) for n in g]))
        h = abs(ty)
        x0 = min(n.box_a[0] for n in g)
        x1 = max(n.box_a[0] + n.box_a[2] for n in g)
        if ty > 0:  # expand: band opens at the siblings' old top
            top = min(n.box_a[1] for n in g)
            band = Rect(x0, top, x1 - x0, h)  # B frame
            box_a, box_b = (x0, top, x1 - x0, 0.0), (x0, top, x1 - x0, h)
        else:  # collapse: band vanishes above the siblings' new top
            top = min(n.box_b[1] for n in g)
            band = Rect(x0, top, x1 - x0, h)  # A frame ≈ B frame above the siblings
            box_a, box_b = (x0, top, x1 - x0, h), (x0, top, x1 - x0, 0.0)
        bx0, by0, bx1, by1 = _int_box(band)
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(w_img, bx1), min(h_img, by1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        band_mask = np.zeros(ctx.shape, bool)
        band_mask[by0:by1, bx0:bx1] = True
        inside = [
            n
            for n in out
            if n not in g
            and n.kind != "backdrop"
            and (n.mask & band_mask).sum() >= 0.6 * max(1, int(n.mask.sum()))
        ]
        # The band must hold new content in B (an opened panel, or new changed regions such as
        # its text lines), not just the page background the shifted block left behind (a plain
        # move down is not an expand).
        new_in_band = int((left & band_mask).sum()) >= ctx.params.regions.min_area_px2 * ctx.k**2
        if ty > 0 and not inside and not new_in_band and not _band_has_content(ctx, band_mask):
            continue
        if ty < 0 and not inside:
            continue
        mask = band_mask.copy()
        for n in inside:
            mask |= n.mask
            out.remove(n)
        node = _new_node(ctx, "resize", mask, mask, np.zeros(ctx.shape, np.float32), 0,
                         box_a=box_a, box_b=box_b)  # fmt: skip
        node.energy = float(sum(n.energy for n in inside) + sum(n.energy for n in g))
        node.notes.append(f"height {0 if ty > 0 else h:.1f} -> {h if ty > 0 else 0:.1f}")
        out.append(node)
        for n in g:
            n.notes.append("caused_by_resize")
            n.resize_of = node
    return out
