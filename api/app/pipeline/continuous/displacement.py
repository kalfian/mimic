"""Coherence / axis test and per-frame scroller displacement (PLAN-continuous §4.2, §4.3).

Pure numpy / OpenCV on in-memory gray ``float32`` crops (no I/O). Conventions (verified against
OpenCV 5.0.0, same as ``pipeline/track.py``):

* ``cv2.phaseCorrelate(a, b, window)`` returns the displacement of content **from a to b**.
* ``cv2.findTransformECC(template, input, warp, MOTION_TRANSLATION, criteria, mask, gauss)``:
  ``warp`` maps template px to input px (``input(u + t) ≈ template(u)``); the template may be a
  sub-image of a larger input.
* ``cv2.matchTemplate(image, templ, TM_CCOEFF_NORMED)``: result index ``j`` = top-left column
  of ``templ`` inside ``image``.

Internally every image is handled **axis-major**: for a vertical scroller the crops are
transposed so the motion is always along columns. Shifts are analysis px inside this module's
estimators and CSS px (``/ k``) in everything it returns to callers. Sign: + = content moves
right (x) / down (y), like ``SIGN_CONVENTION``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from app.models.ir import Axis
from app.models.measure import DisplacementSeries, NotScrollerReason, Rect
from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams

#: ECC termination (per refine); translation-only converges in a handful of iterations.
#: A run of duplicate frames (a rest) yields a zero-step sample this often (s).
REST_SAMPLE_S = 0.1
ECC_MAX_ITERS = 40
ECC_EPS = 1e-5
#: ECC may move at most this far (analysis px) from the NCC estimate; further = diverged.
ECC_MAX_JUMP_PX = 1.5
#: Template must keep at least this many px (and this fraction of the crop) along the axis.
MIN_TEMPLATE_PX = 24
MIN_TEMPLATE_FRAC = 0.15
#: Smallest crop side (analysis px) that is tracked at all.
MIN_CROP_PX = 16
#: Region refinement: a pixel "changed" if it differs from the reference frame by more than
#: this many gray levels; rows / columns count when this fraction of them changed (relative to
#: the busiest row / column).
REFINE_DIFF_LEVELS = 12.0
REFINE_PROFILE_FRAC = 0.1
REFINE_PAD_PX = 2
#: Coherence test falls back to lagged pairs when fewer moving frame pairs than this exist.
COHERENCE_MIN_MOVING = 3
COHERENCE_LAGS = (2, 4, 8, 16)
#: Low-quality recovery search (see ``estimate_shift``): window = max(mult · window,
#: frac · |d|, min px); accepted only at q ≥ WIDE_MIN_Q and q gain ≥ WIDE_MIN_GAIN.
WIDE_WIN_MULT = 3.0
WIDE_WIN_FRAC = 0.75
WIDE_WIN_MIN_PX = 24.0
WIDE_MIN_Q = 0.8
WIDE_MIN_GAIN = 0.15
#: Keyframe anchoring (``DisplacementTracker._anchor``): re-anchor once the content moved more
#: than KEY_MAX_FRAC of the crop; accept the keyframe match at q ≥ KEY_MIN_Q (and not worse
#: than the frame-to-frame q by more than KEY_Q_SLACK) within KEY_MAX_DISAGREE_PX of the
#: frame-to-frame estimate; NCC window ±KEY_WINDOW_PX (analysis px per CSS px scaled).
KEY_MAX_FRAC = 0.3
KEY_MIN_Q = 0.9
KEY_Q_SLACK = 0.05
KEY_MAX_DISAGREE_PX = 2.0
KEY_WINDOW_PX = 3.0
#: Recorder stalls (see ``stall_mask``): a step ≤ STALL_PX between steps ≥ STALL_MOVING_PX,
#: at most STALL_MAX_FRAMES in a row (CSS px).
STALL_PX = 0.05
STALL_MOVING_PX = 0.5
STALL_MAX_FRAMES = 3
#: Perpendicular shifts below this (CSS px) are treated as zero in the dominance ratio.
PERP_FLOOR_PX = 0.02


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def axis_major(img: np.ndarray, axis: Axis) -> np.ndarray:
    """``img`` with the motion axis along columns (contiguous float32)."""
    a = img if axis == "x" else img.T
    return np.ascontiguousarray(a, dtype=np.float32)


@lru_cache(maxsize=32)
def _hann(h: int, w: int) -> np.ndarray:
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    win.setflags(write=False)
    return win


def hann(shape: tuple[int, ...]) -> np.ndarray:
    return _hann(int(shape[0]), int(shape[1]))


def refined_shift_2d(
    a: np.ndarray, b: np.ndarray, gauss: int = DEFAULT_PARAMS.continuous.ecc_gauss_filt
) -> tuple[float, float, float]:
    """2-D shift ``a → b``: phase correlation refined by ECC translation on the inner part.

    Phase correlation's 5×5-centroid sub-pixel estimate is biased by up to ~0.5 px on smooth,
    compressed content — too coarse for the perpendicular-drift criterion of the coherence
    test (< 0.25 px). Returns ``(dx, dy, phase-correlation response)``.
    """
    dx, dy, resp = phase_shift(a, b)
    h, w = a.shape
    ix = int(math.ceil(abs(dx))) + 4
    iy = int(math.ceil(abs(dy))) + 4
    if w - 2 * ix < MIN_TEMPLATE_PX or h - 2 * iy < 8:
        return dx, dy, resp
    tpl = np.ascontiguousarray(a[iy : h - iy, ix : w - ix])
    warp = np.array([[1.0, 0.0, ix + dx], [0.0, 1.0, iy + dy]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERS, ECC_EPS)
    try:
        _, warp = cv2.findTransformECC(tpl, b, warp, cv2.MOTION_TRANSLATION, crit, None, gauss)
    except cv2.error:
        return dx, dy, resp
    ex, ey = float(warp[0, 2]) - ix, float(warp[1, 2]) - iy
    if not (math.isfinite(ex) and math.isfinite(ey)) or abs(ex - dx) > 2 or abs(ey - dy) > 2:
        return dx, dy, resp
    return ex, ey, resp


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """``(dx, dy, response)`` of content from ``a`` to ``b`` (Hann-windowed phase correlation).

    OpenCV 5.0.0 footgun (verified): ``cv2.phaseCorrelate(a, b, window)`` multiplies ``a`` and
    ``b`` by the window **in place**. The window is applied here on copies instead (identical
    result, inputs untouched).
    """
    win = hann(a.shape)
    (sx, sy), resp = cv2.phaseCorrelate(
        np.multiply(a, win, dtype=np.float32), np.multiply(b, win, dtype=np.float32)
    )
    return float(sx), float(sy), float(resp)


def _parabolic(r: np.ndarray, j: int) -> float:
    """Sub-sample offset of the peak at ``j`` of a 1-D response (−0.5..0.5)."""
    if j <= 0 or j >= len(r) - 1:
        return 0.0
    a, b, c = float(r[j - 1]), float(r[j]), float(r[j + 1])
    den = a - 2.0 * b + c
    if den >= 0 or not math.isfinite(den):
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def axis_box_blur(img: np.ndarray, length: int) -> np.ndarray:
    """Centered box blur along the (axis-major) columns with a kernel of ``length`` px.

    Even lengths are rounded up to odd: an even ``cv2.blur`` kernel is off-centre by ½ px,
    which would bias every shift measured against the blurred image.
    """
    length = int(length)
    if length < 2:
        return img
    length |= 1
    return cv2.blur(img, (length, 1), borderType=cv2.BORDER_REFLECT)


def refine_box(
    frames: Sequence[np.ndarray], min_side: int = MIN_CROP_PX, pad: int = REFINE_PAD_PX
) -> tuple[int, int, int, int]:
    """Bounding box ``(x0, y0, x1, y1)`` (crop px) of the pixels that change over ``frames``.

    Change = max over frames of ``|f_i − f_0|`` (and ``|f_i − f_mid|``) above
    ``REFINE_DIFF_LEVELS``; the box spans the longest run of rows / columns whose changed
    fraction is ≥ ``REFINE_PROFILE_FRAC`` of the busiest one. Falls back to the whole crop when
    nothing (or too little) changes. ``pad`` px are added on every side (a working margin for
    tracking crops; the reported scroller box uses ``pad=0``, else it comes out 2·pad too big).
    """
    h, w = frames[0].shape[:2]
    full = (0, 0, w, h)
    if len(frames) < 2:
        return full
    ref0 = frames[0]
    refm = frames[len(frames) // 2]
    acc = np.zeros((h, w), np.float32)
    for f in frames[1:]:
        np.maximum(acc, cv2.absdiff(f, ref0), out=acc)
        np.maximum(acc, cv2.absdiff(f, refm), out=acc)
    mask = (acc > REFINE_DIFF_LEVELS).astype(np.float32)
    if mask.sum() < 4:
        return full

    def run(profile: np.ndarray) -> tuple[int, int]:
        keep = profile >= REFINE_PROFILE_FRAC * float(profile.max())
        best, cur, best_span = (0, len(profile)), None, -1
        for i, v in enumerate(np.append(keep, False)):
            if v and cur is None:
                cur = i
            elif not v and cur is not None:
                if i - cur > best_span:
                    best_span, best = i - cur, (cur, i)
                cur = None
        return best

    y0, y1 = run(mask.mean(axis=1))
    x0, x1 = run(mask.mean(axis=0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < min(min_side, w) or y1 - y0 < min(min_side, h):
        return full
    return x0, y0, x1, y1


def changed_span(
    frames: Sequence[np.ndarray], box: tuple[int, int, int, int], axis: str
) -> tuple[int, int, int, int]:
    """``box`` widened along ``axis`` to the first … last changed column (row) inside the box's
    cross extent, gaps between changed stretches included (refine_box keeps only the longest
    run; a row of flat cards changes only where its edges and text pass)."""
    x0, y0, x1, y1 = box
    ref0 = frames[0]
    acc = np.zeros_like(ref0, dtype=np.float32)
    for f in frames[1:]:
        np.maximum(acc, cv2.absdiff(f, ref0), out=acc)
    mask = (acc > REFINE_DIFF_LEVELS).astype(np.float32)
    if axis == "x":
        prof = mask[y0:y1, :].mean(axis=0)
    else:
        prof = mask[:, x0:x1].mean(axis=1)
    if prof.size == 0 or float(prof.max()) <= 0:
        return box
    on = np.nonzero(prof >= REFINE_PROFILE_FRAC * float(prof.max()))[0]
    lo, hi = int(on[0]), int(on[-1]) + 1
    n = mask.shape[1] if axis == "x" else mask.shape[0]
    lo, hi = max(0, lo - REFINE_PAD_PX), min(n, hi + REFINE_PAD_PX)
    if axis == "x":
        return min(lo, x0), y0, max(hi, x1), y1
    return x0, min(lo, y0), x1, max(hi, y1)


# --------------------------------------------------------------------------------------------
# §4.2 coherence / axis test
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class CoherenceResult:
    is_scroller: bool
    axis: Axis | None
    coherence: float  # fraction of pairs with response >= coherence_min_response
    reason: NotScrollerReason | None
    median_perp_px: float  # CSS px, over moving pairs
    dominance: float  # median |d_axis| / median |d_perp|
    velocity: float  # median signed axis velocity over moving pairs, CSS px/s (0 if unknown)
    moving_pairs: int


def coherence_test(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    k: float,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
) -> CoherenceResult:
    """Single-axis coherent translation test over ``frames`` (gray float32, same size).

    Frame-to-frame phase correlation; when fewer than ``COHERENCE_MIN_MOVING`` pairs move by at
    least ``coherence_min_shift_px`` (slow marquees at high fps), lagged pairs are used for the
    axis / dominance part (the response criterion always uses consecutive pairs).
    """
    n = len(frames)
    if n < 3:
        return CoherenceResult(False, None, 0.0, "untrackable", 0.0, 0.0, 0.0, 0)
    t = np.asarray(times, dtype=np.float64)
    shifts: list[tuple[float, float, float, float]] = []  # dx, dy (CSS), resp, dt
    for i in range(1, n):
        dx, dy, r = refined_shift_2d(frames[i - 1], frames[i])
        shifts.append((dx / k, dy / k, r, float(t[i] - t[i - 1])))
    arr = np.asarray(shifts)
    coherence = float(np.mean(arr[:, 2] >= params.coherence_min_response))

    def moving(a: np.ndarray) -> np.ndarray:
        return a[np.hypot(a[:, 0], a[:, 1]) >= params.coherence_min_shift_px]

    mov = moving(arr)
    for lag in COHERENCE_LAGS:
        if len(mov) >= COHERENCE_MIN_MOVING or lag >= n:
            break
        lagged = []
        for i in range(lag, n, max(1, lag // 2)):
            dx, dy, r = refined_shift_2d(frames[i - lag], frames[i])
            if r >= params.coherence_min_response:
                lagged.append((dx / k, dy / k, r, float(t[i] - t[i - lag])))
        if lagged:
            mov = moving(np.asarray(lagged))
    if coherence < params.coherence_min_frac:
        return CoherenceResult(False, None, coherence, "untrackable", 0.0, 0.0, 0.0, len(mov))
    if len(mov) < COHERENCE_MIN_MOVING:
        # coherent but (almost) not moving in the window: cannot tell the axis
        return CoherenceResult(False, None, coherence, "untrackable", 0.0, 0.0, 0.0, len(mov))
    ax, ay = np.abs(mov[:, 0]), np.abs(mov[:, 1])
    axis: Axis = "x" if np.median(ax) >= np.median(ay) else "y"
    along, perp = (mov[:, 0], ay) if axis == "x" else (mov[:, 1], ax)
    med_perp = float(np.median(perp))
    dominance = float(np.median(np.abs(along)) / max(med_perp, PERP_FLOOR_PX))
    velocity = float(np.median(along / np.maximum(mov[:, 3], 1e-6)))
    ok = med_perp < params.coherence_max_perp_px and dominance >= params.coherence_axis_dominance
    return CoherenceResult(
        is_scroller=ok,
        axis=axis if ok else None,
        coherence=coherence,
        reason=None if ok else "two_axis",
        median_perp_px=med_perp,
        dominance=dominance,
        velocity=velocity,
        moving_pairs=len(mov),
    )


def strip_velocities(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    axis: Axis,
    strip_px: int,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-strip mean axis velocity (CSS px/s) and mean response over the window (§4.2 split).

    Strips are ``strip_px`` thick perpendicular to the axis (one cell). Returns
    ``(velocity (S,), response (S,))``.
    """
    am = [axis_major(f, axis) for f in frames]
    h = am[0].shape[0]
    n_strips = max(1, h // max(strip_px, 4))
    edges = np.linspace(0, h, n_strips + 1).astype(int)
    t = np.asarray(times, dtype=np.float64)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    vel = np.zeros(n_strips)
    resp = np.zeros(n_strips)
    for s in range(n_strips):
        a, b = edges[s], edges[s + 1]
        if b - a < 4:
            continue
        total, rs = 0.0, []
        for i in range(1, len(am)):
            dx, _, r = phase_shift(am[i - 1][a:b], am[i][a:b])
            total += dx
            rs.append(r)
        vel[s] = total / k / span if span > 0 else 0.0
        resp[s] = float(np.mean(rs)) if rs else 0.0
    return vel, resp


def cluster_strips(
    velocity: np.ndarray,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
    *,
    static_px_s: float = 2.0,
) -> list[np.ndarray]:
    """Group adjacent moving strips with similar velocity (|Δv| ≤ max(min_dv, frac·|v|)).

    Static strips (|v| < ``static_px_s``) separate groups and belong to none. Groups are
    returned largest first as arrays of strip indices.
    """
    groups: list[list[int]] = []
    for i, v in enumerate(velocity):
        if abs(v) < static_px_s:
            groups.append([])
            continue
        if groups and groups[-1]:
            ref = float(np.median(velocity[groups[-1]]))
            tol = max(params.split_min_dv, params.split_min_dv_frac * max(abs(ref), abs(v)))
            if abs(v - ref) <= tol:
                groups[-1].append(i)
                continue
        groups.append([i])
    out = [np.asarray(g) for g in groups if g]
    out.sort(key=len, reverse=True)
    return out


# --------------------------------------------------------------------------------------------
# §4.3 per-frame shift
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ShiftEstimate:
    """Shift of content from ``prev`` to ``cur`` along the axis (analysis px, axis-major)."""

    d: float
    d_perp: float
    q: float  # NCC peak or ECC rho (0..1)
    method: str  # "ecc" | "ncc" | "pc" | "pred"
    response: float = 0.0  # phase-correlation response (0 if not run)


def _ncc_refine(
    prev: np.ndarray, cur: np.ndarray, d0: float, half_win: float, pad: float
) -> tuple[float, float, int] | None:
    """1-D NCC of ``prev``'s inner part against ``cur`` around ``d0`` (±``half_win``).

    Returns ``(d, peak, inset)`` or None if the template would be too small.
    """
    L = prev.shape[1]
    half_win = max(1.0, half_win)
    inset = int(math.ceil(abs(d0) + half_win + pad))
    min_len = max(MIN_TEMPLATE_PX, int(MIN_TEMPLATE_FRAC * L))
    if L - 2 * inset < min_len:
        inset = int(math.ceil(abs(d0) + half_win + 1))
        if L - 2 * inset < min_len:
            return None
    tpl = prev[:, inset : L - inset]
    lo = max(0, inset + int(math.floor(d0 - half_win)))
    hi = min(L, L - inset + int(math.ceil(d0 + half_win)))
    if hi - lo < tpl.shape[1] + 1:
        return None
    strip = cur[:, lo:hi]
    r = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)[0]
    r = np.nan_to_num(r, nan=-1.0)
    j = int(np.argmax(r))
    d = lo + j + _parabolic(r, j) - inset
    return float(d), float(r[j]), inset


def _ecc_refine(
    prev: np.ndarray, cur: np.ndarray, d_init: float, inset: int, gauss: int
) -> tuple[float, float, float] | None:
    """ECC translation warm-started at ``d_init``; ``(d, d_perp, rho)`` or None if it failed."""
    L = prev.shape[1]
    tpl = np.ascontiguousarray(prev[:, inset : L - inset])
    warp = np.array([[1.0, 0.0, inset + d_init], [0.0, 1.0, 0.0]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERS, ECC_EPS)
    try:
        rho, warp = cv2.findTransformECC(tpl, cur, warp, cv2.MOTION_TRANSLATION, crit, None, gauss)
    except cv2.error:
        return None
    d = float(warp[0, 2]) - inset
    dp = float(warp[1, 2])
    if not (math.isfinite(d) and math.isfinite(dp) and math.isfinite(rho)):
        return None
    if abs(d - d_init) > ECC_MAX_JUMP_PX or abs(dp) > ECC_MAX_JUMP_PX:
        return None
    return d, dp, float(rho)


def _refine(
    prev: np.ndarray,
    cur: np.ndarray,
    d0: float,
    half_win: float,
    pad: float,
    gauss: int,
) -> ShiftEstimate | None:
    """Step 3: NCC around ``d0`` then warm-started ECC."""
    ncc = _ncc_refine(prev, cur, d0, half_win, pad)
    if ncc is None:
        return None
    d, peak, inset = ncc
    ecc = _ecc_refine(prev, cur, d, inset, gauss)
    if ecc is not None and ecc[2] >= peak - 0.05:
        return ShiftEstimate(d=ecc[0], d_perp=ecc[1], q=float(np.clip(ecc[2], 0, 1)), method="ecc")
    return ShiftEstimate(d=d, d_perp=0.0, q=float(np.clip(peak, 0, 1)), method="ncc")


def estimate_shift(
    prev: np.ndarray,
    cur: np.ndarray,
    d_hat: float | None,
    *,
    k: float = 1.0,
    accel_px: float = 0.0,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
    allow_multiscale: bool = True,
) -> ShiftEstimate:
    """Shift of content from ``prev`` to ``cur`` (axis-major, analysis px) per §4.3 steps 1–5.

    ``d_hat`` = constant-velocity prediction (analysis px; None = no prediction, e.g. the first
    pair). ``accel_px`` = how far the true shift may deviate from the prediction under the
    acceleration clamp (``max_accel · dt²`` in analysis px); it widens the NCC window when the
    phase-correlation estimate is rejected. ``k`` = analysis px per CSS px (thresholds that are
    CSS px in ``params`` are scaled by it).
    """
    L = prev.shape[1]
    if min(prev.shape) < 4:
        return ShiftEstimate(0.0, 0.0, 0.0, "pred" if d_hat is not None else "pc")

    # step 5: multi-scale for shifts that are large relative to the crop
    if (
        allow_multiscale
        and d_hat is not None
        and abs(d_hat) > params.multiscale_shift_frac * L
        and min(prev.shape) >= 2 * MIN_CROP_PX
    ):
        size = (max(1, prev.shape[1] // 2), max(1, prev.shape[0] // 2))
        p2 = cv2.resize(prev, size, interpolation=cv2.INTER_AREA)
        c2 = cv2.resize(cur, size, interpolation=cv2.INTER_AREA)
        coarse = estimate_shift(
            p2, c2, d_hat / 2.0, k=k / 2.0, accel_px=accel_px / 2.0, params=params,
            allow_multiscale=False,
        )  # fmt: skip
        d_hat_full = 2.0 * coarse.d
        win = max(params.ncc_window_min_px * k, 3.0)
        best = _refine(
            prev, cur, d_hat_full, win, params.ncc_inset_pad_px * k, params.ecc_gauss_filt
        )
        if best is not None:
            best.response = coarse.response
            return _blur_retry(prev, cur, best, win, k, params)
        coarse.d, coarse.d_perp = 2.0 * coarse.d, 2.0 * coarse.d_perp
        return coarse

    # step 2: initial estimate by phase correlation (or the prediction)
    pc_dx, pc_dy, resp = phase_shift(prev, cur)
    if d_hat is None:
        d0, from_pred = pc_dx, False
    else:
        dev_thr = max(params.pc_max_dev_px * k, params.pc_max_dev_frac * abs(d_hat))
        if resp >= params.pc_min_response and abs(pc_dx - d_hat) <= dev_thr:
            d0, from_pred = pc_dx, False
        else:
            d0, from_pred = d_hat, True
    win = max(params.ncc_window_min_px * k, params.ncc_window_frac * abs(d0))
    if from_pred:
        win = max(win, accel_px)
    if d_hat is None:
        # no prediction: phase correlation alone may alias; search wider around it
        win = max(win, 2.0 * params.ncc_window_min_px * k)

    # step 3: NCC + ECC refine
    est = _refine(prev, cur, d0, win, params.ncc_inset_pad_px * k, params.ecc_gauss_filt)
    if est is None:
        # crop too small for the template: report the initial estimate
        q = float(np.clip(resp, 0.0, 1.0)) if not from_pred else 0.0
        return ShiftEstimate(d0, pc_dy if not from_pred else 0.0, q, "pred" if from_pred else "pc",
                             response=resp)  # fmt: skip
    if from_pred and resp >= params.pc_min_response and abs(pc_dx - d0) > win:
        # the rejected phase-correlation peak lies outside the window: keep it only if it is
        # clearly better (prediction-first otherwise — periodic content aliases)
        alt = _refine(prev, cur, pc_dx, params.ncc_window_min_px * k,
                      params.ncc_inset_pad_px * k, params.ecc_gauss_filt)  # fmt: skip
        if alt is not None and alt.q > est.q + 0.1:
            est = alt
    est.response = resp
    # step 4: blur-matched retry
    est = _blur_retry(prev, cur, est, win, k, params)
    if est.q < params.blur_retry_below_q:
        # recovery: the constant-velocity prediction can be far off when recorder timestamps
        # jitter (VFR) — search wide around the prediction and the phase-correlation peak and
        # accept only a clearly better, confident match (periodic content aliases otherwise)
        wide = max(WIDE_WIN_MULT * win, WIDE_WIN_FRAC * abs(d0), WIDE_WIN_MIN_PX * k)
        for centre in {d0, pc_dx}:
            alt = _refine(prev, cur, centre, wide, params.ncc_inset_pad_px * k,
                          params.ecc_gauss_filt)  # fmt: skip
            if alt is not None and alt.q >= WIDE_MIN_Q and alt.q > est.q + WIDE_MIN_GAIN:
                alt.response = resp
                alt.method += "+wide"
                est = alt
    return est


def _blur_retry(
    prev: np.ndarray,
    cur: np.ndarray,
    est: ShiftEstimate,
    win: float,
    k: float,
    params: ContinuousParams,
) -> ShiftEstimate:
    if est.q >= params.blur_retry_below_q:
        return est
    best = est
    for length in sorted({int(math.ceil(abs(est.d) / 2.0)), int(math.ceil(abs(est.d)))}):
        if length < 2:
            continue
        pb = axis_box_blur(prev, length)
        alt = _refine(pb, cur, est.d, max(win, params.ncc_window_min_px * k),
                      params.ncc_inset_pad_px * k, params.ecc_gauss_filt)  # fmt: skip
        if alt is not None and alt.q > best.q:
            alt.response = est.response
            alt.method = alt.method + "+blur"
            best = alt
    return best


# --------------------------------------------------------------------------------------------
# §4.3 streaming tracker
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Frame:
    t: float
    img: np.ndarray  # axis-major, tracking box only


@dataclass
class DisplacementTracker:
    """Streaming displacement of one scroller (push frames in time order; dups are skipped).

    ``box`` = ``(x0, y0, x1, y1)`` tracking box inside the crop (crop px, not axis-major);
    ``k`` = analysis px per CSS px. ``seed_velocity`` (CSS px/s, e.g. from the coherence test)
    provides the first prediction.
    """

    axis: Axis
    k: float
    box: tuple[int, int, int, int] | None = None
    params: ContinuousParams = field(default_factory=lambda: DEFAULT_PARAMS.continuous)
    seed_velocity: float | None = None
    times: list[float] = field(default_factory=list)
    d_css: list[float] = field(default_factory=list)  # per sample i >= 1: shift from i-1
    quality: list[float] = field(default_factory=list)
    responses: list[float] = field(default_factory=list)
    _prev: _Frame | None = None
    _prev2: _Frame | None = None
    _key: _Frame | None = None  # drift-free anchor (see ``_anchor``)
    _key_pos: float = 0.0  # CSS px
    _pos: float = 0.0  # CSS px of the last accepted frame
    _last_dup_t: float | None = None  # time of the latest duplicate frame

    def _anchor(self, cur: _Frame, est: ShiftEstimate) -> float:
        """Position step (CSS px) of ``cur`` measured against the keyframe when possible.

        Summing frame-to-frame shifts accumulates any per-step sub-pixel bias (interpolation
        "pixel locking" at fractional px/frame speeds → a few % speed error over seconds).
        While the content stays within ``KEY_MAX_FRAC`` of the crop of a keyframe, the position
        is measured directly against it, so the bias stays bounded instead of growing.
        """
        p = self.params
        step = est.d / self.k
        pos_pred = self._pos + step
        key = self._key
        length = cur.img.shape[1]
        if key is not None:
            off = (pos_pred - self._key_pos) * self.k
            if abs(off) <= KEY_MAX_FRAC * length:
                # the prediction is already sub-pixel accurate: ECC alone (no NCC search)
                inset = int(math.ceil(abs(off) + KEY_WINDOW_PX * max(self.k, 0.5)))
                inset += int(math.ceil(p.ncc_inset_pad_px * self.k))
                kr = None
                if length - 2 * inset >= max(MIN_TEMPLATE_PX, int(MIN_TEMPLATE_FRAC * length)):
                    kr = _ecc_refine(key.img, cur.img, off, inset, p.ecc_gauss_filt)
                if (
                    kr is not None
                    and kr[2] >= KEY_MIN_Q
                    and kr[2] >= est.q - KEY_Q_SLACK
                    and abs(kr[0] - off) <= KEY_MAX_DISAGREE_PX * max(self.k, 0.5)
                ):
                    return self._key_pos + kr[0] / self.k - self._pos
        # (re)anchor on this frame
        self._key, self._key_pos = cur, pos_pred
        return step

    def _prep(self, img: np.ndarray) -> np.ndarray:
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            img = img[y0:y1, x0:x1]
        return axis_major(img, self.axis)

    def push(self, t: float, img: np.ndarray) -> None:
        cur = _Frame(float(t), self._prep(img))
        p = self.params
        if self._prev is None:
            self._prev = cur
            self.times.append(cur.t)
            self.quality.append(1.0)
            self.responses.append(1.0)
            return
        prev = self._prev
        if float(cv2.absdiff(cur.img, prev.img).max()) <= p.dup_max_abs_diff:
            # duplicate frame: the next real frame spans the merged dt. A run of duplicates is
            # a rest, though (a clean encode of a resting scroller has no other samples): it
            # yields a zero step every REST_SAMPLE_S, and finish() closes a trailing one.
            if cur.t - self.times[-1] >= REST_SAMPLE_S:
                self._zero_step(cur.t)
            self._last_dup_t = cur.t
            return
        dt = cur.t - prev.t
        if dt <= 0:
            return
        if self.d_css:
            dt_prev = prev.t - self.times[-2]
            v_hat = self.d_css[-1] / dt_prev if dt_prev > 0 else 0.0
        else:
            v_hat = self.seed_velocity
        d_hat = None if v_hat is None else v_hat * dt * self.k
        accel = p.max_accel_px_s2 * dt * dt * self.k
        est = estimate_shift(prev.img, cur.img, d_hat, k=self.k, accel_px=accel, params=p)
        if self._key is None:
            self._key, self._key_pos = prev, self._pos
        step = self._anchor(cur, est)
        self._pos += step
        self.times.append(cur.t)
        self.d_css.append(step)
        self.quality.append(est.q)
        self.responses.append(est.response)
        # step 6: consistency d(i, i-2) vs d_i + d_{i-1}
        n = len(self.times) - 1  # index of cur
        if (
            self._prev2 is not None
            and p.consistency_every > 0
            and n % p.consistency_every == 0
            and len(self.d_css) >= 2
        ):
            want = (self.d_css[-1] + self.d_css[-2]) * self.k
            two = _refine(self._prev2.img, cur.img, want, max(p.ncc_window_min_px * self.k, 2.0),
                          p.ncc_inset_pad_px * self.k, p.ecc_gauss_filt)  # fmt: skip
            if two is None or abs(two.d - want) > p.consistency_tol_px * self.k:
                self.quality[-1] = min(self.quality[-1], p.consistency_low_q)
                self.quality[-2] = min(self.quality[-2], p.consistency_low_q)
        self._prev2 = prev
        self._prev = cur

    def _zero_step(self, t: float) -> None:
        self.times.append(float(t))
        self.d_css.append(0.0)
        self.quality.append(1.0)
        self.responses.append(1.0)

    def finish(self) -> None:
        """Close a trailing run of duplicates (the recording ends at rest) with a zero step."""
        last = self._last_dup_t
        if last is not None and self.times and last > self.times[-1]:
            self._zero_step(last)

    def series(self, region: Rect, region_confidence: float) -> DisplacementSeries:
        times = np.asarray(self.times, dtype=np.float64)
        pos = np.concatenate([[0.0], np.cumsum(self.d_css)]) if self.times else np.zeros(0)
        quality = np.clip(np.asarray(self.quality, dtype=np.float64), 0.0, 1.0)
        keep = stall_mask(pos)
        times, pos, quality = times[keep], pos[keep], quality[keep]
        dt = np.diff(times)
        return DisplacementSeries(
            times=times,
            pos=pos.astype(np.float64),
            quality=quality,
            axis=self.axis,
            region=region,
            region_confidence=float(region_confidence),
            dt_median=float(np.median(dt)) if dt.size else 0.0,
        )


def stall_mask(pos: np.ndarray) -> np.ndarray:
    """Keep-mask dropping recorder stalls: up to ``STALL_MAX_FRAMES`` samples that did not move
    (``|step| ≤ STALL_PX``) between moving steps (``|step| ≥ STALL_MOVING_PX``).

    A lossy-encoded duplicate frame is not bit-identical, so the ``dup_max_abs_diff`` test in
    ``iter_region`` misses it; it then shows up as a zero step inside motion followed by a
    catch-up step — a false velocity dip + spike. Dropping it gives the "merged dt" semantics of
    a skipped duplicate. Genuine rests (no motion on both sides) are kept.
    """
    n = len(pos)
    keep = np.ones(n, bool)
    if n < 4:
        return keep
    step = np.abs(np.diff(pos))  # step[i] = move into sample i + 1
    i = 1
    while i < len(step):
        if step[i] <= STALL_PX and step[i - 1] >= STALL_MOVING_PX:
            j = i
            while j < len(step) and step[j] <= STALL_PX and j - i < STALL_MAX_FRAMES:
                j += 1
            if j < len(step) and step[j] >= STALL_MOVING_PX:
                keep[i + 1 : j + 1] = False  # samples whose incoming step was a stall
                i = j + 1
                continue
        i += 1
    return keep


def track_frames(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    axis: Axis,
    k: float = 1.0,
    *,
    box: tuple[int, int, int, int] | None = None,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
    seed_velocity: float | None = None,
) -> DisplacementTracker:
    """Convenience: run a :class:`DisplacementTracker` over in-memory frames."""
    tr = DisplacementTracker(axis=axis, k=k, box=box, params=params, seed_velocity=seed_velocity)
    for t, f in zip(times, frames, strict=True):
        tr.push(t, f)
    return tr


# --------------------------------------------------------------------------------------------
# Velocities (labelling / UI only; fits use positions)
# --------------------------------------------------------------------------------------------


def smoothed_velocity(times: np.ndarray, pos: np.ndarray, half: int = 2) -> np.ndarray:
    """Local linear regression slope of ``pos`` over ±``half`` samples (non-uniform t)."""
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64)
    n = len(t)
    v = np.zeros(n)
    if n < 2:
        return v
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        if b - a < 2:
            a, b = max(0, b - 2), min(n, a + 2)
        tt = t[a:b] - t[a:b].mean()
        den = float(np.sum(tt * tt))
        v[i] = float(np.sum(tt * (x[a:b] - x[a:b].mean())) / den) if den > 0 else 0.0
    return v


def robust_velocity(
    times: np.ndarray,
    pos: np.ndarray,
    quality: np.ndarray | None = None,
    *,
    window_s: float = 0.1,
    min_half: int = 2,
    q_min: float = 0.5,
) -> np.ndarray:
    """Theil–Sen slope of ``pos`` over a window of ≥ ``window_s`` (≥ ±``min_half`` samples).

    For peak speeds: the median of pairwise slopes ignores single-sample tracking glitches and
    averages capture/render timestamp jitter (a frame stamped up to one render interval after
    the content it shows) over several frames. Samples with ``quality < q_min`` are left out
    while at least 3 remain.
    """
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64)
    n = len(t)
    v = np.zeros(n)
    if n < 2:
        return v
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    half = max(min_half, int(math.ceil(window_s / 2.0 / max(dt, 1e-6))))
    good = np.ones(n, bool) if quality is None else np.asarray(quality) >= q_min
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        idx = np.arange(a, b)
        keep = idx[good[a:b]]
        if keep.size >= 3:
            idx = keep
        tt, xx = t[idx], x[idx]
        j, k = np.triu_indices(len(idx), 1)
        span = tt[k] - tt[j]
        ok = span > 0.5 * dt
        if not ok.any():
            continue
        v[i] = float(np.median((xx[k][ok] - xx[j][ok]) / span[ok]))
    return v


def step_velocity(times: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Per-sample velocity of the step *into* sample i (``v[0] = v[1]``)."""
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64)
    if len(t) < 2:
        return np.zeros(len(t))
    v = np.diff(x) / np.maximum(np.diff(t), 1e-9)
    return np.concatenate([[v[0]], v])


__all__ = [
    "CoherenceResult",
    "DisplacementTracker",
    "robust_velocity",
    "ShiftEstimate",
    "axis_box_blur",
    "axis_major",
    "cluster_strips",
    "coherence_test",
    "estimate_shift",
    "phase_shift",
    "refine_box",
    "smoothed_velocity",
    "stall_mask",
    "step_velocity",
    "strip_velocities",
    "track_frames",
]
