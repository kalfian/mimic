"""Geometric alignment and per-frame tracking (PLAN §6.5 a–c, §6.6 "Geometric").

Conventions (verified against OpenCV 5.0.0):

* ``cv2.findTransformECC(template, input, warp, ...)``: ``warp`` maps **template** pixel coords to
  **input** pixel coords (``input(W·u) ≈ template(u)``); the two images may differ in size.
  ``cv2.findTransformECCWithMask(template, input, templateMask, inputMask, warp, ...)`` takes a
  mask per image (input mask is warped into the template frame; the intersection is used).
* ``cv2.phaseCorrelate(a, b)`` returns the displacement of content from ``a`` to ``b``.
* ``cv2.estimateAffinePartial2D(src, dst, method=RANSAC, ransacReprojThreshold=…)`` returns a 2×3
  similarity mapping ``src`` → ``dst``.

All warps here are 2×3 float64 "A→B" maps: ``x_B = L·x_A + t`` in some pixel frame.
:func:`warp_px_to_css` converts a crop-pixel warp into full-frame CSS px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.ir import PxValue, RatioValue, SegmentId
from app.models.measure import PropertySeries, Rect, WindowFrames
from app.pipeline.params import MeasureParams

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#: Affine beats translation-only only if ``(1−ρ_t) > GAIN·(1−ρ_a) + MARGIN`` (Occam) and the
#: affine is a plausible CSS transform (no rotation/shear; isotropic unless the gain is large).
#: TODO(calibrate): module-local; candidates for ``TrackParams`` in a coordinated edit.
AFFINE_GAIN = 2.0
AFFINE_GAIN_ANISO = 4.0
AFFINE_MARGIN = 0.0005
AFFINE_MAX_SHEAR = 0.01
AFFINE_MAX_ROT_DEG = 1.0
AFFINE_ISO_TOL = 0.01

# --------------------------------------------------------------------------------------------
# Affine helpers
# --------------------------------------------------------------------------------------------


def as3(m: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2] = m
    return out


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a ∘ b`` (apply ``b`` first)."""
    return (as3(a) @ as3(b))[:2]


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(as3(m))[:2]


def apply(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (
        float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
        float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
    )


def shift_frame(m: np.ndarray, src_origin: tuple[float, float], dst_origin: tuple[float, float]):
    """Re-express ``m`` (global coords) as a map between local frames with the given origins."""
    to_g = np.array([[1.0, 0, src_origin[0]], [0, 1.0, src_origin[1]]])
    from_g = np.array([[1.0, 0, -dst_origin[0]], [0, 1.0, -dst_origin[1]]])
    return compose(from_g, compose(m, to_g))


def warp_px_to_css(m_px: np.ndarray, k: float, origin_css: tuple[float, float]) -> np.ndarray:
    """Crop-px warp → full-frame CSS warp (``x_px = k·(x_css − origin)``)."""
    s = np.array([[k, 0, -k * origin_css[0]], [0, k, -k * origin_css[1]]])
    return compose(invert(s), compose(m_px, s))


def warp_css_to_px(m_css: np.ndarray, k: float, origin_css: tuple[float, float]) -> np.ndarray:
    s = np.array([[k, 0, -k * origin_css[0]], [0, k, -k * origin_css[1]]])
    return compose(s, compose(m_css, invert(s)))


@dataclass(frozen=True, slots=True)
class Decomposed:
    sx: float
    sy: float
    scale: float
    theta_deg: float


def decompose(m: np.ndarray) -> Decomposed:
    lin = m[:, :2]
    sx = float(np.hypot(lin[0, 0], lin[1, 0]))
    sy = float(np.hypot(lin[0, 1], lin[1, 1]))
    return Decomposed(
        sx, sy, math.sqrt(max(sx * sy, 0.0)), math.degrees(math.atan2(lin[1, 0], lin[0, 0]))
    )


def is_identity(m: np.ndarray, t_tol: float, s_tol: float) -> bool:
    d = decompose(m)
    return (
        abs(m[0, 2]) < t_tol
        and abs(m[1, 2]) < t_tol
        and abs(d.sx - 1) < s_tol
        and abs(d.sy - 1) < s_tol
        and abs(m[0, 1]) < s_tol
        and abs(m[1, 0]) < s_tol
    )


def center_displacement(m: np.ndarray, box: Rect) -> tuple[float, float]:
    cx, cy = box.center
    x, y = apply(m, cx, cy)
    return x - cx, y - cy


def map_box(m: np.ndarray, box: Rect) -> Rect:
    pts = [apply(m, x, y) for x in (box.x, box.x2) for y in (box.y, box.y2)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def box_from_union(u: Rect, m: np.ndarray) -> Rect:
    """Recover box ``A`` from ``U = bbox(A ∪ m(A))`` for an axis-aligned scale+translate ``m``.

    Solved per axis: each end of ``U`` is either the end of ``A`` or of ``m(A)``.
    """

    def solve(u1: float, u2: float, s: float, t: float) -> tuple[float, float]:
        s = max(s, 1e-6)
        best: tuple[float, float] | None = None
        best_err = math.inf
        for a in (u1, (u1 - t) / s):
            for b in (u2, (u2 - t) / s):
                if b <= a:
                    continue
                lo = min(a, s * a + t)
                hi = max(b, s * b + t)
                err = abs(lo - u1) + abs(hi - u2)
                if err < best_err - 1e-9:
                    best, best_err = (a, b), err
        return best if best is not None else (u1, u2)

    ax1, ax2 = solve(u.x, u.x2, float(m[0, 0]), float(m[0, 2]))
    ay1, ay2 = solve(u.y, u.y2, float(m[1, 1]), float(m[1, 2]))
    return Rect(ax1, ay1, ax2 - ax1, ay2 - ay1)


# --------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------


def gray32(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def _criteria(params: MeasureParams, iters: int | None = None):
    p = params.track
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters or p.ecc_iters, p.ecc_eps)


def ecc(
    tpl: np.ndarray,
    img: np.ndarray,
    init: np.ndarray,
    params: MeasureParams,
    *,
    tpl_mask: np.ndarray | None = None,
    img_mask: np.ndarray | None = None,
    motion: int = cv2.MOTION_AFFINE,
    pyramid: bool = True,
) -> tuple[float, np.ndarray] | None:
    """Coarse-to-fine ECC. ``init``/result map template px → input px. None if it fails."""
    tpl = gray32(tpl)
    img = gray32(img)
    tm = np.full(tpl.shape, 255, np.uint8) if tpl_mask is None else tpl_mask.astype(np.uint8)
    im = np.full(img.shape, 255, np.uint8) if img_mask is None else img_mask.astype(np.uint8)
    if min(tpl.shape) < 8 or min(img.shape) < 8 or (tm > 0).sum() < 30:
        return None
    levels = 0
    if pyramid:
        while levels < 2 and min(tpl.shape) / 2 ** (levels + 1) >= 32:
            levels += 1
    warp = init.astype(np.float32).copy()
    rho = -1.0
    gf = params.track.ecc_gauss_filt
    for lvl in range(levels, -1, -1):
        f = 2.0**lvl
        if lvl:
            size_t = (max(1, int(round(tpl.shape[1] / f))), max(1, int(round(tpl.shape[0] / f))))
            size_i = (max(1, int(round(img.shape[1] / f))), max(1, int(round(img.shape[0] / f))))
            t_l = cv2.resize(tpl, size_t, interpolation=cv2.INTER_AREA)
            i_l = cv2.resize(img, size_i, interpolation=cv2.INTER_AREA)
            tm_l = cv2.resize(tm, size_t, interpolation=cv2.INTER_NEAREST)
            im_l = cv2.resize(im, size_i, interpolation=cv2.INTER_NEAREST)
        else:
            t_l, i_l, tm_l, im_l = tpl, img, tm, im
        w_l = warp.copy()
        w_l[:, 2] /= f
        try:
            rho, w_l = cv2.findTransformECCWithMask(
                t_l, i_l, tm_l, im_l, w_l, motion, _criteria(params), gf
            )
        except cv2.error:
            if lvl == 0:
                return None
            continue
        w_l[:, 2] *= f
        warp = w_l
    if not np.all(np.isfinite(warp)):
        return None
    return float(rho), warp.astype(np.float64)


def orb_similarity(
    a: np.ndarray,
    b: np.ndarray,
    params: MeasureParams,
    *,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float] | None:
    """ORB + Hamming crossCheck + RANSAC similarity (A→B). None unless it passes the gates."""
    p = params.regions
    ga = gray32(a).astype(np.uint8)
    gb = gray32(b).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=p.orb_nfeatures, fastThreshold=p.orb_fast_threshold)
    ka, da = orb.detectAndCompute(ga, mask_a)
    kb, db = orb.detectAndCompute(gb, mask_b)
    if da is None or db is None or len(ka) < p.min_inliers or len(kb) < p.min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < p.min_inliers:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    m, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=p.ransac_reproj_px
    )
    if m is None or inl is None:
        return None
    n_in = int(inl.sum())
    ratio = n_in / len(matches)
    if n_in < p.min_inliers or ratio < p.min_inlier_ratio:
        return None
    return m.astype(np.float64), n_in, ratio


def phase_translation(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Translation A→B via phase correlation (same-size images)."""
    ga, gb = gray32(a), gray32(b)
    if mask is not None:
        mean_a = float(ga[mask > 0].mean()) if (mask > 0).any() else float(ga.mean())
        mean_b = float(gb[mask > 0].mean()) if (mask > 0).any() else float(gb.mean())
        ga = np.where(mask > 0, ga, mean_a).astype(np.float32)
        gb = np.where(mask > 0, gb, mean_b).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(ga, gb, win)
    return np.array([[1.0, 0, sx], [0, 1.0, sy]]), float(resp)


@dataclass(slots=True)
class Alignment:
    warp: np.ndarray  # A→B in the (shared) crop px frame
    rho: float
    method: str  # "ecc" | "orb" | "phase" | "identity"


def estimate_transform(
    a: np.ndarray,
    b: np.ndarray,
    box: tuple[int, int, int, int],
    params: MeasureParams,
    *,
    mask: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> Alignment:
    """Dominant A→B transform of the content in ``box`` (x, y, w, h; crop px) (§6.5 a–c).

    ``a``/``b`` are same-size crops. ``mask`` (crop-size uint8) restricts the template pixels
    (e.g. the region minus nested regions); ``valid`` excludes cursor pixels in both.
    Initial guesses (identity, phase correlation, ORB) are each refined with pyramid ECC;
    the best ECC rho wins. Falls back to phase correlation if every ECC run fails.
    """
    x, y, w, h = box
    H, W = a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    tpl = a[y0:y1, x0:x1]
    tm = None
    if mask is not None:
        tm = mask[y0:y1, x0:x1].copy()
    if valid is not None:
        tv = valid[y0:y1, x0:x1]
        tm = tv.copy() if tm is None else cv2.bitwise_and(tm, tv)
    off = np.array([[1.0, 0, x0], [0, 1.0, y0]])  # template px → crop px

    inits: list[tuple[str, np.ndarray]] = [("identity", IDENTITY.copy())]
    sub_a, sub_b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    pm = None if valid is None else valid[y0:y1, x0:x1]
    t_phase, resp = phase_translation(sub_a, sub_b, pm)
    if resp > 0.05 and (abs(t_phase[0, 2]) > 0.5 or abs(t_phase[1, 2]) > 0.5):
        inits.append(("phase", t_phase))
    orb = orb_similarity(a, b, params, mask_a=_box_mask(a.shape, (x0, y0, x1, y1), valid),
                         mask_b=valid)  # fmt: skip
    if orb is not None:
        inits.append(("orb", orb[0]))

    # Fit translation-only and full affine from every guess. Affine must earn its extra DOF:
    # a single straight edge (one band of a translated box) is degenerate under affine.
    best_t: Alignment | None = None
    best_a: Alignment | None = None
    for name, m in inits:
        for motion in (cv2.MOTION_TRANSLATION, cv2.MOTION_AFFINE):
            init = compose(m, off)
            if motion == cv2.MOTION_TRANSLATION:
                init = np.array([[1.0, 0, init[0, 2]], [0, 1.0, init[1, 2]]])
            res = ecc(tpl, b, init, params, tpl_mask=tm, img_mask=valid, motion=motion)
            if res is None:
                continue
            rho, w_t = res
            al = Alignment(compose(w_t, invert(off)), rho,
                           "ecc" if name == "identity" else f"ecc+{name}")  # fmt: skip
            if motion == cv2.MOTION_TRANSLATION:
                if best_t is None or rho > best_t.rho + 1e-5:
                    best_t = al
            elif best_a is None or rho > best_a.rho + 1e-5:
                best_a = al
    best = best_t
    if best_a is not None and (best_t is None or affine_preferred(best_t.rho, best_a)):
        best = best_a
    if best is not None and best.rho >= params.track.ecc_min_rho:
        return best
    if orb is not None:
        return Alignment(orb[0], best.rho if best else 0.0, "orb")
    if best is not None:
        return best
    return Alignment(t_phase, 0.0, "phase")


def affine_shape(m: np.ndarray) -> tuple[float, float, float]:
    """(|rotation| deg, shear = |cos angle between columns|, |sx − sy|) of a 2×3 warp."""
    d = decompose(m)
    lin = m[:, :2]
    shear = abs(float(lin[:, 0] @ lin[:, 1])) / max(d.sx * d.sy, 1e-9)
    return abs(d.theta_deg), shear, abs(d.sx - d.sy)


def affine_preferred(rho_t: float, aff: Alignment) -> bool:
    """Model selection translation vs affine (see ``AFFINE_*``)."""
    rot, shear, aniso = affine_shape(aff.warp)
    if rot > AFFINE_MAX_ROT_DEG or shear > AFFINE_MAX_SHEAR:
        return False
    gain = AFFINE_GAIN if aniso <= AFFINE_ISO_TOL else AFFINE_GAIN_ANISO
    return (1 - rho_t) > gain * (1 - aff.rho) + AFFINE_MARGIN


def _box_mask(shape, xyxy, valid) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = xyxy
    m[y0:y1, x0:x1] = 255
    if valid is not None:
        m = cv2.bitwise_and(m, valid)
    return m


# --------------------------------------------------------------------------------------------
# Per-frame tracking
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class FrameTrack:
    """Per-frame warps of one element: template-source frame (A, or B if backwards) → frame t.

    ``warps[i]`` is in crop px; ``rho[i]`` is the ECC correlation (0 when lost);
    ``visible[i]`` is False where tracking failed (appear elements before they show up).
    """

    times: np.ndarray
    warps: np.ndarray  # (N, 2, 3)
    rho: np.ndarray
    visible: np.ndarray
    method: list[str] = field(default_factory=list)


def _template_match(tpl: np.ndarray, img: np.ndarray) -> tuple[float, float, float] | None:
    """Translation of ``tpl`` inside ``img`` (TM_CCOEFF_NORMED + parabolic subpixel)."""
    if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
        return None
    r = cv2.matchTemplate(gray32(img), gray32(tpl), cv2.TM_CCOEFF_NORMED)
    _, mx, _, (px, py) = cv2.minMaxLoc(r)

    def sub(v_m, v_0, v_p):
        den = v_m - 2 * v_0 + v_p
        return 0.0 if abs(den) < 1e-9 else 0.5 * (v_m - v_p) / den

    dx = sub(r[py, px - 1], r[py, px], r[py, px + 1]) if 0 < px < r.shape[1] - 1 else 0.0
    dy = sub(r[py - 1, px], r[py, px], r[py + 1, px]) if 0 < py < r.shape[0] - 1 else 0.0
    return px + dx, py + dy, float(mx)


def track_element(
    wf: WindowFrames,
    src_img: np.ndarray,
    box_px: Rect,
    final_warp_px: np.ndarray,
    params: MeasureParams,
    *,
    backwards: bool = False,
    inset_frac: float = 0.0,
    src_valid: np.ndarray | None = None,
    src_exclude: np.ndarray | None = None,
    frame_order: list[int] | None = None,
) -> FrameTrack:
    """Track one element through every window frame (§6.6).

    ``src_img`` is state A (or B when ``backwards``); ``box_px`` the element box in it (crop
    px); ``final_warp_px`` the expected src→other-state warp, used to size the search area.
    Each frame is aligned with ECC warm-started from its neighbour, then ORB, then template
    matching. Result warps map src crop px → frame crop px.
    """
    p = params.track
    k = wf.scale.k
    pad = p.template_pad_css * k
    bx = box_px
    if inset_frac > 0:
        bx = Rect(bx.x + bx.w * inset_frac, bx.y + bx.h * inset_frac,
                  bx.w * (1 - 2 * inset_frac), bx.h * (1 - 2 * inset_frac))  # fmt: skip
    H, W = src_img.shape[:2]
    tx0 = int(max(0, math.floor(bx.x - pad)))
    ty0 = int(max(0, math.floor(bx.y - pad)))
    tx1 = int(min(W, math.ceil(bx.x2 + pad)))
    ty1 = int(min(H, math.ceil(bx.y2 + pad)))
    tpl = gray32(src_img[ty0:ty1, tx0:tx1])
    tmask = None if src_valid is None else src_valid[ty0:ty1, tx0:tx1].copy()
    if src_exclude is not None:
        ex = src_exclude[ty0:ty1, tx0:tx1]
        keep = np.where(ex, 0, 255).astype(np.uint8)
        if (keep > 0).sum() >= 0.15 * keep.size:  # enough of the parent left to lock on
            tmask = keep if tmask is None else cv2.bitwise_and(tmask, keep)
    t_off = np.array([[1.0, 0, tx0], [0, 1.0, ty0]])

    # search area: union of the box and its final position, padded
    end_box = map_box(final_warp_px, Rect(tx0, ty0, tx1 - tx0, ty1 - ty0))
    sbox = Rect(tx0, ty0, tx1 - tx0, ty1 - ty0).union(end_box).padded(p.search_pad_css * k)
    sx0, sy0 = int(max(0, math.floor(sbox.x))), int(max(0, math.floor(sbox.y)))
    sx1, sy1 = int(min(W, math.ceil(sbox.x2))), int(min(H, math.ceil(sbox.y2)))
    s_off = np.array([[1.0, 0, sx0], [0, 1.0, sy0]])

    n = len(wf.crops)
    order = frame_order if frame_order is not None else (
        list(range(n - 1, -1, -1)) if backwards else list(range(n))
    )  # fmt: skip
    warps = np.repeat(IDENTITY[None], n, axis=0)
    rho = np.zeros(n)
    visible = np.zeros(n, dtype=bool)
    methods = [""] * n
    prev: np.ndarray | None = None  # crop-px warp of the previous good frame
    prev2: np.ndarray | None = None
    # Duplicate frames (``is_dup``: ≤ 1 gray level from the previous frame) show the same
    # content as their run's first frame: reuse that frame's result instead of re-running ECC.
    # They are excluded from every series anyway; this only saves time (Phase 5 perf pass).
    run_of = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        run_of[i] = run_of[i - 1] if wf.is_dup[i] else i
    done_run: dict[int, int] = {}
    for i in order:
        j = done_run.get(int(run_of[i]))
        if j is not None:
            warps[i], rho[i], visible[i], methods[i] = warps[j], rho[j], visible[j], methods[j]
            if visible[j]:
                prev2, prev = prev, warps[j]
            continue
        done_run[int(run_of[i])] = i
        frame = wf.crops[i]
        search = frame[sy0:sy1, sx0:sx1]
        cm = wf.cursor_masks[i]
        smask = None if cm is None else cv2.bitwise_not(cm[sy0:sy1, sx0:sx1])
        if prev is None:
            guess = IDENTITY.copy()
        elif prev2 is None:
            guess = prev
        else:  # constant-velocity extrapolation of the translation
            guess = prev.copy()
            guess[:, 2] += prev[:, 2] - prev2[:, 2]
        # template px → search px
        init = compose(invert(s_off), compose(guess, t_off))
        res = ecc(tpl, search, init, params, tpl_mask=tmask, img_mask=smask, pyramid=prev is None)
        ok = False
        if res is not None and res[0] >= p.ecc_min_rho:
            r_i, w_t = res
            m = compose(s_off, compose(w_t, invert(t_off)))
            ok, method = True, "ecc"
        elif res is not None and prev is not None:
            res2 = ecc(tpl, search, init, params, tpl_mask=tmask, img_mask=smask, pyramid=True)
            if res2 is not None and res2[0] >= p.ecc_min_rho:
                r_i, w_t = res2
                m = compose(s_off, compose(w_t, invert(t_off)))
                ok, method = True, "ecc"
        if not ok:
            orb = orb_similarity(tpl.astype(np.uint8), search, params)
            if orb is not None:
                m = compose(s_off, compose(orb[0], invert(t_off)))
                r_i, ok, method = 0.5, True, "orb"
        if not ok:
            tmres = _template_match(tpl, search)
            if tmres is not None and tmres[2] >= p.ecc_min_rho:
                x, y, score = tmres
                m = np.array([[1.0, 0, sx0 + x - tx0], [0, 1.0, sy0 + y - ty0]])
                r_i, ok, method = score, True, "tm"
        if ok:
            warps[i] = m
            rho[i] = r_i
            visible[i] = True
            methods[i] = method
            prev2, prev = prev, m
        else:
            methods[i] = "lost"
    return FrameTrack(times=wf.times.copy(), warps=warps, rho=rho, visible=visible, method=methods)


# --------------------------------------------------------------------------------------------
# Series
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    """Where a segment sits inside a window (absolute seconds)."""

    segment_id: SegmentId
    start_s: float  # last stable frame before motion (pass-1 estimate)
    end_s: float  # first settled frame (pass-1 estimate)
    lo_s: float  # window part that belongs to this segment
    hi_s: float


def _stable_stats(
    times: np.ndarray, values: np.ndarray, ok: np.ndarray, span: SegmentSpan, guard_s: float
) -> tuple[float, float, float]:
    pre = ok & (times >= span.lo_s - 1e-9) & (times <= span.start_s - guard_s + 1e-9)
    post = ok & (times >= span.end_s + guard_s - 1e-9) & (times <= span.hi_s + 1e-9)
    if not pre.any():
        pre = ok & (times <= span.start_s + 1e-9) & (times >= span.lo_s - 1e-9)
    if not post.any():
        post = ok & (times >= span.end_s - 1e-9) & (times <= span.hi_s + 1e-9)
    inside = ok & (times >= span.lo_s - 1e-9) & (times <= span.hi_s + 1e-9)
    v_start = float(np.median(values[pre])) if pre.any() else float(values[inside][0])
    v_end = float(np.median(values[post])) if post.any() else float(values[inside][-1])
    parts = [values[pre] - v_start if pre.any() else np.zeros(0),
             values[post] - v_end if post.any() else np.zeros(0)]  # fmt: skip
    stab = np.concatenate(parts)
    std = float(stab.std()) if stab.size >= 2 else 0.0
    return v_start, v_end, std


def snap_origin(u: float, v: float, snap: float) -> str:
    """(u, v) in bbox-normalised coords → CSS ``transform-origin`` keywords."""

    def nearest(x: float) -> float | None:
        c = min((0.0, 0.5, 1.0), key=lambda q: abs(q - x))
        return c if abs(c - x) <= snap else None

    nu, nv = nearest(u), nearest(v)
    if nu is None or nv is None:
        return f"{u * 100:.0f}% {v * 100:.0f}%"
    xs = {0.0: "left", 0.5: "center", 1.0: "right"}[nu]
    ys = {0.0: "top", 0.5: "center", 1.0: "bottom"}[nv]
    if xs == "center" and ys == "center":
        return "center"
    if xs == "center":
        return f"{ys} center"
    if ys == "center":
        return f"{xs} center"
    return f"{ys} {xs}"


def transform_origin(m: np.ndarray, box: Rect, params: MeasureParams) -> str:
    """Fixed point ``p* = (I − L)^-1 t`` normalised to ``box`` and snapped (§6.6)."""
    d = decompose(m)
    if abs(d.scale - 1) < params.track.origin_min_scale_delta:
        return "center"
    lin = m[:, :2]
    try:
        px, py = np.linalg.solve(np.eye(2) - lin, m[:, 2])
    except np.linalg.LinAlgError:
        return "center"
    u = (px - box.x) / box.w if box.w > 0 else 0.5
    v = (py - box.y) / box.h if box.h > 0 else 0.5
    return snap_origin(float(u), float(v), params.track.origin_snap)


def geometric_series(
    element_id: str,
    warps_css: np.ndarray,
    times: np.ndarray,
    ok: np.ndarray,
    quality: np.ndarray,
    box_css: Rect,
    span: SegmentSpan,
    params: MeasureParams,
    *,
    is_dup: np.ndarray | None = None,
    guard_s: float = 1 / 30,
) -> tuple[list[PropertySeries], list[str]]:
    """translateX / translateY / scale (+ scaleX / scaleY if anisotropic) for one segment.

    ``warps_css`` map the element's reference state to frame t in CSS px; translation is the
    displacement of ``box_css``'s centre. Returns series and warning hints (e.g. rotation).
    """
    sel = (times >= span.lo_s - 1e-9) & (times <= span.hi_s + 1e-9)
    if is_dup is not None:
        sel &= ~is_dup
    tx = np.full(len(times), np.nan)
    ty = np.full(len(times), np.nan)
    sc = np.full(len(times), np.nan)
    sxv = np.full(len(times), np.nan)
    syv = np.full(len(times), np.nan)
    th = np.full(len(times), np.nan)
    for i in range(len(times)):
        if not ok[i]:
            continue
        dx, dy = center_displacement(warps_css[i], box_css)
        d = decompose(warps_css[i])
        tx[i], ty[i], sc[i], sxv[i], syv[i], th[i] = dx, dy, d.scale, d.sx, d.sy, d.theta_deg
    good = sel & ok
    hints: list[str] = []
    if not good.any():
        return [], hints
    t = times[good]
    q = quality[good]
    out: list[PropertySeries] = []

    def mk(prop, vals, unit):
        v_start, v_end, std = _stable_stats(times, vals, ok & sel, span, guard_s)
        val_cls = PxValue if unit == "px" else RatioValue
        return PropertySeries(
            element_id=element_id, property=prop, segment=span.segment_id,
            times=t.copy(), values=vals[good].copy(), quality=q.copy(),
            v_start=v_start, v_end=v_end, unit=unit,
            from_value=val_cls(number=round(v_start, 4)), to_value=val_cls(number=round(v_end, 4)),
            stable_std=std,
        )  # fmt: skip

    out.append(mk("translateX", tx, "px"))
    out.append(mk("translateY", ty, "px"))
    s_ser = mk("scale", sc, "ratio")
    out.append(s_ser)
    sx_ser, sy_ser = mk("scaleX", sxv, "ratio"), mk("scaleY", syv, "ratio")
    if (
        abs(sx_ser.v_end - sy_ser.v_end) > params.track.scale_xy_split
        or abs(sx_ser.v_start - sy_ser.v_start) > params.track.scale_xy_split
    ):
        out.extend([sx_ser, sy_ser])
    _, th_end, _ = _stable_stats(times, th, ok & sel, span, guard_s)
    if abs(th_end) > params.track.rotation_warn_deg:
        hints.append("rotation_detected")
    return out, hints


# --------------------------------------------------------------------------------------------
# Stage entry point
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class GeometryResult:
    series: list[PropertySeries]
    #: (element id, segment id) → per-frame absolute warps (CSS px, reference state → frame t)
    tracks: dict[tuple[str, str], FrameTrack]
    warnings: list[str]  # hint codes, e.g. "rotation_detected"


def segment_spans(
    wf: WindowFrames, start_s: float, end_s: float, *, peak_s: float | None = None
) -> list[SegmentSpan]:
    """Spans for a window: one per segment; a round-trip window splits at ``peak_s``."""
    lo, hi = float(wf.times[0]), float(wf.times[-1])
    if peak_s is None:
        return [SegmentSpan(wf.segment_id, start_s, end_s, lo, hi)]
    return [
        SegmentSpan("rt_in", start_s, peak_s, lo, peak_s),
        SegmentSpan("rt_out", peak_s, end_s, peak_s, hi),
    ]


def _children_exclusion(
    eid: str,
    elements: list,
    boxes_px: dict[str, tuple[Rect, Rect]],
    shape,
    k: float,
    warp_ab: np.ndarray | None = None,
) -> np.ndarray | None:
    """A-frame footprint (+3 CSS px) of all descendants — their A box and their B box mapped
    back through this element's A→B warp — so a parent is tracked on its own pixels only."""
    kids = {e.id for e in elements if e.parent_id == eid}
    frontier = set(kids)
    while frontier:
        nxt = {e.id for e in elements if e.parent_id in frontier}
        frontier = nxt - kids
        kids |= nxt
    if not kids:
        return None
    m = np.zeros(shape, bool)
    pad = 3 * k
    h, w = shape
    inv = None if warp_ab is None else invert(warp_ab)
    for c in kids:
        rects = [boxes_px[c][0]]
        if inv is not None:
            rects.append(map_box(inv, boxes_px[c][1]))
        for r0 in rects:
            r = r0.padded(pad)
            x0, y0 = int(max(0, math.floor(r.x))), int(max(0, math.floor(r.y)))
            x1, y1 = int(min(w, math.ceil(r.x2))), int(min(h, math.ceil(r.y2)))
            m[y0:y1, x0:x1] = True
    return m


def measure_geometry(
    elements: list,  # list[ElementCandidate]
    boxes_px: dict[str, tuple[Rect, Rect]],
    warps_px: dict[str, np.ndarray],
    state_a: np.ndarray,
    state_b: np.ndarray,
    valid: np.ndarray,
    windows: list[tuple[WindowFrames, list[SegmentSpan]]],
    params: MeasureParams,
) -> GeometryResult:
    """Per-frame transforms for ``transform`` (and ``appear``/``disappear``) elements (§6.6).

    Parents are tracked first so children are reported relative to their parent.
    """
    by_id = {e.id: e for e in elements}
    depth = {}

    def d(eid: str) -> int:
        if eid not in depth:
            par = by_id[eid].parent_id
            depth[eid] = 0 if par is None or par not in by_id else d(par) + 1
        return depth[eid]

    order = sorted((e for e in elements if e.kind in ("transform", "appear", "disappear")),
                   key=lambda e: d(e.id))  # fmt: skip
    series: list[PropertySeries] = []
    tracks: dict[tuple[str, str], FrameTrack] = {}
    hints: list[str] = []
    for wf, spans in windows:
        k = wf.scale.k
        origin = wf.crop_origin_css
        abs_css: dict[str, np.ndarray] = {}
        for e in order:
            box_a, box_b = boxes_px[e.id]
            w_ab = warps_px[e.id]
            appear_like = e.kind in ("appear", "disappear")
            if appear_like:
                # template from the state where the element is visible
                visible_in_b = e.kind == "appear"
                src = state_b if visible_in_b else state_a
                box = box_b if visible_in_b else box_a
                fwd_dir = wf.segment_id in ("fwd", "rt_in")
                backwards = fwd_dir == visible_in_b
                ft = track_element(wf, src, box, IDENTITY, params, backwards=backwards,
                                   src_valid=valid)  # fmt: skip
                ref_box_css = Rect(origin[0] + box.x / k, origin[1] + box.y / k, box.w / k,
                                   box.h / k)  # fmt: skip
            else:
                inset = 0.0
                if e.parent_id is not None and abs(decompose(w_ab).scale - 1) >= 0.01:
                    inset = params.track.inner_child_inset_frac
                excl = _children_exclusion(e.id, elements, boxes_px, state_a.shape[:2], k, w_ab)
                ft = track_element(wf, state_a, box_a, w_ab, params, inset_frac=inset,
                                   src_valid=valid, src_exclude=excl)  # fmt: skip
                ref_box_css = Rect(origin[0] + box_a.x / k, origin[1] + box_a.y / k,
                                   box_a.w / k, box_a.h / k)  # fmt: skip
            w_css = np.stack([warp_px_to_css(w, k, origin) for w in ft.warps])
            abs_css[e.id] = w_css
            rel = w_css
            ok = ft.visible.copy()
            par = e.parent_id
            if par in abs_css and not appear_like:
                pw = abs_css[par]
                rel = np.stack([compose(invert(pw[i]), w_css[i]) for i in range(len(w_css))])
                ok &= tracks.get((par, wf.segment_id), ft).visible
            ft_css = FrameTrack(ft.times, rel, ft.rho, ok, ft.method)
            tracks[(e.id, wf.segment_id)] = ft_css
            for span in spans:
                ser, h = geometric_series(e.id, rel, ft.times, ok, ft.rho, ref_box_css, span,
                                          params, is_dup=wf.is_dup)  # fmt: skip
                for s in ser:
                    if s.property == "scale" and not appear_like:
                        m_end = rel[ok][-1] if ok.any() else IDENTITY
                        s.transform_origin = transform_origin(m_end, ref_box_css, params)
                    if par is not None and not appear_like:
                        s.notes.append(f"relative to parent {par}")
                series.extend(ser)
                hints.extend(x for x in h if x not in hints)
    return GeometryResult(series=series, tracks=tracks, warnings=hints)


def derive_height_series(
    resize_links: dict[str, list[str]], series: list[PropertySeries]
) -> list[PropertySeries]:
    """``height`` series for resize elements (§6.5 expand/collapse).

    Content below an expanding panel moves down by exactly the panel's height change, so the
    pushed sibling's ``translateY`` series *is* the height curve: ``height(t) = h0 + ty(t)``
    with ``h0`` = 0 for an opening panel (ty: 0 → +h) and ``h`` for a closing one (ty: 0 → −h).
    """
    out: list[PropertySeries] = []
    for rid, sibs in resize_links.items():
        for seg in sorted({s.segment for s in series}):
            ty = next(
                (s for sid in sibs for s in series
                 if s.element_id == sid and s.segment == seg and s.property == "translateY"),
                None,
            )  # fmt: skip
            if ty is None:
                continue
            # the band's height at the forward reference state A
            h0 = max(0.0, -min(ty.v_start, ty.v_end))
            vals = h0 + ty.values
            v0, v1 = h0 + ty.v_start, h0 + ty.v_end
            out.append(
                PropertySeries(
                    element_id=rid,
                    property="height",
                    segment=seg,
                    times=ty.times.copy(),
                    values=vals,
                    quality=ty.quality.copy(),
                    v_start=v0,
                    v_end=v1,
                    unit="px",
                    from_value=PxValue(number=round(v0, 3)),
                    to_value=PxValue(number=round(v1, 3)),
                    stable_std=ty.stable_std,
                    notes=[f"from {ty.element_id} translateY"],
                )  # fmt: skip
            )
    return out


# --------------------------------------------------------------------------------------------
# Fading elements: un-blended re-tracking (Phase 4)
# --------------------------------------------------------------------------------------------

#: Frames below this opacity are too noisy to un-blend (noise is amplified by 1/α).
UNBLEND_MIN_ALPHA = 0.2


def _alpha_by_frame(wf: WindowFrames, op: PropertySeries) -> np.ndarray:
    """Opacity per window frame (NaN where the opacity series has no sample)."""
    alpha = np.full(len(wf.times), np.nan)
    lut = {round(float(t), 5): float(v) for t, v in zip(op.times, op.values, strict=True)}
    for i, t in enumerate(wf.times):
        v = lut.get(round(float(t), 5))
        if v is not None:
            alpha[i] = v
    return alpha


def retrack_faded(
    geo: GeometryResult,
    opacity: list[PropertySeries],
    elements: list,  # list[ElementCandidate]
    boxes_px: dict[str, tuple[Rect, Rect]],
    state_a: np.ndarray,
    state_b: np.ndarray,
    valid: np.ndarray,
    windows: list[tuple[WindowFrames, list[SegmentSpan]]],
    params: MeasureParams,
) -> list[str]:
    """Re-track appearing / disappearing root elements on *un-blended* frames (in place).

    A frame of an element fading at opacity α over the page is ``I = α·M + (1 − α)·P``. ECC on
    ``I`` against the fully visible template is biased toward the static page at low α (the
    padding around the element keeps full contrast while the element is scaled by α), which
    left S5's slide-in 0.6–0.8 px short. With the measured α(t) and the page ``P`` (the state
    in which the element is absent), ``U = (I − (1 − α)·P) / α`` recovers ``M`` and is tracked
    instead. Skipped when a backdrop exists (the page behind changes too) or for children.
    Returns the ids of re-tracked elements.
    """
    if any(e.kind == "backdrop" for e in elements):
        return []
    ops = {
        (s.element_id, s.segment): s
        for s in opacity
        if s.property == "opacity" and s.unit == "ratio" and "opacity_from_dimming" not in s.notes
    }
    geo_props = ("translateX", "translateY", "scale", "scaleX", "scaleY")
    done: list[str] = []
    for wf, spans in windows:
        k = wf.scale.k
        origin = wf.crop_origin_css
        H, W = state_a.shape[:2]
        for e in elements:
            if e.kind not in ("appear", "disappear") or e.parent_id is not None:
                continue
            appear = e.kind == "appear"
            box_a, box_b = boxes_px[e.id]
            box = box_b if appear else box_a
            src = state_b if appear else state_a
            page = (state_a if appear else state_b).astype(np.float32)
            old = geo.tracks.get((e.id, wf.segment_id))
            pad = params.track.template_pad_css * k
            tx0, ty0 = int(max(0, math.floor(box.x - pad))), int(max(0, math.floor(box.y - pad)))
            tx1, ty1 = int(min(W, math.ceil(box.x2 + pad))), int(min(H, math.ceil(box.y2 + pad)))
            tpl = src[ty0:ty1, tx0:tx1]
            tmask = valid[ty0:ty1, tx0:tx1]
            t_off = np.array([[1.0, 0, tx0], [0, 1.0, ty0]])
            sbox = Rect(tx0, ty0, tx1 - tx0, ty1 - ty0).padded(params.track.search_pad_css * k)
            sx0, sy0 = int(max(0, math.floor(sbox.x))), int(max(0, math.floor(sbox.y)))
            sx1, sy1 = int(min(W, math.ceil(sbox.x2))), int(min(H, math.ceil(sbox.y2)))
            s_off = np.array([[1.0, 0, sx0], [0, 1.0, sy0]])
            for span in spans:
                op = ops.get((e.id, span.segment_id))
                if op is None:
                    continue
                alpha = _alpha_by_frame(wf, op)
                in_span = (wf.times >= span.lo_s - 1e-9) & (wf.times <= span.hi_s + 1e-9)
                use = in_span & np.isfinite(alpha) & (alpha >= UNBLEND_MIN_ALPHA)
                if use.sum() < 3:
                    continue
                old_sc = [s for s in geo.series if s.element_id == e.id
                          and s.segment == span.segment_id and s.property == "scale"]  # fmt: skip
                aniso = any(s.element_id == e.id and s.segment == span.segment_id
                            and s.property in ("scaleX", "scaleY") for s in geo.series)  # fmt: skip
                scaled = (
                    bool(old_sc)
                    and not aniso
                    and (
                        float(np.ptp(old_sc[0].values)) >= 0.01 if len(old_sc[0].values) else False
                    )
                )
                motion = cv2.MOTION_AFFINE if scaled else cv2.MOTION_TRANSLATION
                warps = np.repeat(IDENTITY[None], len(wf.times), axis=0)
                rho = np.zeros(len(wf.times))
                ok = np.zeros(len(wf.times), dtype=bool)
                prev: np.ndarray | None = None
                for i in sorted(np.nonzero(use)[0], key=lambda j: -alpha[j]):
                    a_i = float(min(alpha[i], 1.0))
                    frame = wf.crops[i][sy0:sy1, sx0:sx1].astype(np.float32)
                    u = (frame - (1.0 - a_i) * page[sy0:sy1, sx0:sx1]) / a_i
                    u = np.clip(u, -64.0, 320.0).astype(np.float32)
                    cm = wf.cursor_masks[i]
                    smask = None if cm is None else cv2.bitwise_not(cm[sy0:sy1, sx0:sx1])
                    if old is not None and old.visible[i]:
                        guess = warp_css_to_px(old.warps[i], k, origin)
                    else:
                        guess = prev if prev is not None else IDENTITY.copy()
                    init = compose(invert(s_off), compose(guess, t_off))
                    if motion == cv2.MOTION_TRANSLATION:
                        init = np.array([[1.0, 0, init[0, 2]], [0, 1.0, init[1, 2]]])
                    res = ecc(tpl, u, init, params, tpl_mask=tmask, img_mask=smask,
                              motion=motion)  # fmt: skip
                    if res is None or res[0] < params.track.ecc_min_rho:
                        continue
                    m = compose(s_off, compose(res[1], invert(t_off)))
                    warps[i], rho[i], ok[i] = m, res[0], True
                    prev = m
                if ok.sum() < 3:
                    continue
                w_css = np.stack([warp_px_to_css(w, k, origin) for w in warps])
                ref_css = Rect(origin[0] + box.x / k, origin[1] + box.y / k, box.w / k, box.h / k)
                new, _ = geometric_series(e.id, w_css, wf.times, ok, rho, ref_css, span, params,
                                          is_dup=wf.is_dup)  # fmt: skip
                if not new:
                    continue
                geo.series[:] = [
                    s for s in geo.series
                    if not (s.element_id == e.id and s.segment == span.segment_id
                            and s.property in geo_props)
                ] + new  # fmt: skip
                for s in new:
                    s.notes.append("tracked on un-blended frames")
                if e.id not in done:
                    done.append(e.id)
    return done
