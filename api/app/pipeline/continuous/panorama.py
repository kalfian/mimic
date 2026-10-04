"""Panorama of the scroller content: loop period, card pitch, gap (PLAN-continuous §4.4).

Frames (axis-major, tracking box, analysis px) are pasted into a 1-D-extended canvas at the
tracked content position: crop column ``u`` of a frame whose content is displaced by ``pos``
(CSS px, + = right / down) lands on canvas column ``c = u − round(pos · k) + origin``. At most
``pano_max_contrib`` contributions are kept per column; the canvas is their median.

* **Loop period** ``P``: smallest lag ``L ≥ loop_min_lag_frac · crop`` at which the canvas
  matches itself (NCC ≥ ``loop_min_ncc`` over ≥ ``loop_min_overlap_frac · crop`` columns).
  Never guessed: no such lag inside the observed travel → ``None`` (``loop.observed = false``).
* **Pitch** ``p``: dominant period of the edge-energy profile (``|∂I/∂u|`` averaged across the
  perpendicular axis) via autocorrelation, fundamental preferred; must be ``< P`` when ``P`` is
  observed.
* **Gap**: the profile of per-column variation across the perpendicular axis, folded modulo
  ``p``; the longest low-variation stretch (background between cards).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams

#: Canvas width cap (analysis px) — beyond it frames are no longer pasted.
MAX_CANVAS_PX = 40_000
#: Smallest card pitch considered (CSS px); smaller lags are gap / text texture.
MIN_PITCH_CSS = 40.0
#: Fundamental: the smallest-lag autocorrelation peak at least this strong (relative to the
#: strongest peak) is the pitch (multiples of the pitch are equally strong).
FUNDAMENTAL_REL = 0.7
#: Loop verification (``loop_period``): window width (CSS px) and the window-wise NCC median /
#: 10th percentile a true (pixel-identical) repeat must reach. Synth suite: true loops ≥ 0.995 /
#: 0.97 (crf 28); same-layout cards ≤ 0.94 / 0.91. TODO(calibrate P2): module-local.
LOCAL_WIN_CSS = 64.0
LOOP_LOCAL_MED = 0.975
LOOP_LOCAL_P10 = 0.93
#: Gap core: arc bins whose cross-axis std ≤ min + max(GAP_STD_ABS, GAP_STD_FRAC · (median −
#: min)) (gray levels).
GAP_STD_ABS = 2.0
GAP_STD_FRAC = 0.1
#: A pitch needs at least this many periods inside the observed content.
MIN_PITCH_PERIODS = 2.5
#: Edge profile smoothing (CSS px σ).
PROFILE_SIGMA_CSS = 2.0

#: Folding columns modulo the pitch into bins: floating-point slack so exact multiples do not
#: fall into the previous bin (``111 / 216 * 216 == 110.999…``).
FOLD_EPS = 1e-6


@dataclass(slots=True)
class PanoramaResult:
    loop_period_px: float | None  # CSS px
    loop_ncc: float
    loop_confidence: float
    pitch_px: float | None
    pitch_confidence: float
    gap_px: float | None
    gap_confidence: float
    #: Content positions (CSS px, ``pos`` units) at which a card's leading edge / centre sits at
    #: the tracking box's leading edge / centre (modulo pitch) — grid lines for snap evidence.
    grid_phases: list[float]
    travel_px: float  # CSS px of content seen beyond one crop


class PanoramaBuilder:
    """Streaming panorama (see module docstring)."""

    def __init__(self, k: float, params: ContinuousParams = DEFAULT_PARAMS.continuous) -> None:
        self.k = float(k)
        self.c = max(1, int(params.pano_max_contrib))
        self.params = params
        self.stack: np.ndarray | None = None  # (C, H, W) float32, NaN = empty
        self.count: np.ndarray | None = None  # (W,) int
        self.origin = 0  # canvas column of crop column 0 at pos 0
        self.crop_len = 0
        self.full = False

    def _ensure(self, lo: int, hi: int, h: int) -> None:
        assert self.stack is not None and self.count is not None
        w = self.stack.shape[2]
        pad_l = max(0, -lo)
        pad_r = max(0, hi - w)
        if not (pad_l or pad_r):
            return
        if w + pad_l + pad_r > MAX_CANVAS_PX:
            self.full = True
            return
        # grow geometrically to amortise
        pad_l = max(pad_l, w // 2) if pad_l else 0
        pad_r = max(pad_r, w // 2) if pad_r else 0
        new = np.full((self.c, h, w + pad_l + pad_r), np.nan, np.float32)
        new[:, :, pad_l : pad_l + w] = self.stack
        cnt = np.zeros(w + pad_l + pad_r, np.int32)
        cnt[pad_l : pad_l + w] = self.count
        self.stack, self.count = new, cnt
        self.origin += pad_l

    def add(self, img: np.ndarray, pos_css: float) -> None:
        """Paste one axis-major frame whose content is displaced by ``pos_css``."""
        h, length = img.shape
        if self.stack is None:
            self.crop_len = length
            self.origin = length
            self.stack = np.full((self.c, h, 3 * length), np.nan, np.float32)
            self.count = np.zeros(3 * length, np.int32)
        if self.full or img.shape[0] != self.stack.shape[1]:
            return
        off = int(round(self.origin - pos_css * self.k))
        self._ensure(off, off + length, h)
        if self.full:
            return
        off = int(round(self.origin - pos_css * self.k))
        assert self.count is not None and self.stack is not None
        cols = np.arange(off, off + length)
        cnt = self.count[cols]
        sel = cnt < self.c
        if not sel.any():
            return
        self.stack[cnt[sel], :, cols[sel]] = img[:, sel].T
        self.count[cols[sel]] += 1

    def canvas(self) -> tuple[np.ndarray, np.ndarray]:
        """``(median canvas (H, W) float32 with NaN, valid column mask (W,))``."""
        if self.stack is None or self.count is None:
            return np.zeros((0, 0), np.float32), np.zeros(0, bool)
        valid = self.count > 0
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            return np.zeros((self.stack.shape[1], 0), np.float32), np.zeros(0, bool)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        sub = self.stack[:, :, lo:hi]
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub, axis=0) if self.c > 1 else sub[0]
        self._lo = lo
        return med.astype(np.float32), valid[lo:hi]


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def _fill(canvas: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    if (~valid).any():
        mean = float(np.nanmean(canvas[:, valid])) if valid.any() else 0.0
        out[:, ~valid] = mean
    return np.nan_to_num(out, nan=float(np.nanmean(out)) if np.isfinite(out).any() else 0.0)


def _longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
    best, cur, best_len = (0, 0), None, 0
    for i, v in enumerate(np.append(valid, False)):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best_len:
                best_len, best = i - cur, (cur, i)
            cur = None
    return best


def _local_match(img: np.ndarray, lag: int, win: int) -> tuple[float, float]:
    """``(median, p10)`` of window-wise NCC between the canvas and itself shifted by ``lag``
    (textureless windows skipped). A true loop repeats pixel for pixel, so *every* window
    matches; cards that merely share a layout match globally but not locally."""
    n = img.shape[1] - lag
    vals: list[float] = []
    for i in range(0, max(n - win, 0) + 1, max(win // 2, 1)):
        a = img[:, i : i + win]
        b = img[:, i + lag : i + lag + win]
        if a.shape != b.shape or (float(a.std()) < 2.0 and float(b.std()) < 2.0):
            continue
        av, bv = a - a.mean(), b - b.mean()
        den = math.sqrt(float((av * av).sum()) * float((bv * bv).sum()))
        vals.append(float((av * bv).sum()) / den if den > 0 else 0.0)
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals)
    return float(np.median(arr)), float(np.percentile(arr, 10))


def loop_period(
    canvas: np.ndarray,
    valid: np.ndarray,
    crop_len: int,
    params: ContinuousParams,
    k: float = 1.0,
) -> tuple[float | None, float]:
    """Loop period (analysis px) + its NCC, or ``(None, best_ncc)``.

    Candidates: local maxima of the template NCC ≥ ``loop_min_ncc`` at lags ≥
    ``loop_min_lag_frac · crop`` (template = ``loop_min_overlap_frac · crop`` columns, two
    templates must agree), smallest first; each must also pass the window-wise check
    (:func:`_local_match`) — similar-looking cards otherwise produce false periods at multiples
    of the pitch, and a false period is worse than "not observed" (§8.4).
    """
    from app.pipeline.continuous.displacement import _parabolic

    lo, hi = _longest_valid_run(valid)
    span = hi - lo
    min_lag = max(2, int(math.ceil(params.loop_min_lag_frac * crop_len)))
    tw = max(8, int(math.ceil(params.loop_min_overlap_frac * crop_len)))
    if span < min_lag + tw or canvas.shape[0] < 2:
        return None, 0.0
    img = canvas[:, lo:hi].astype(np.float32)
    best_ncc = -1.0
    per_template: list[list[tuple[float, float]]] = []
    for start in (0, max(0, min(span - tw - min_lag, crop_len // 4))):
        tpl = img[:, start : start + tw]
        if float(tpl.std()) < 1e-3:
            continue
        r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)[0]
        r = np.nan_to_num(r, nan=-1.0)
        lags = np.arange(len(r)) - start
        m = lags >= min_lag
        if m.any():
            best_ncc = max(best_ncc, float(r[m].max()))
        peaks = [
            (float(lags[j] + _parabolic(r, j)), float(r[j]))
            for j in range(1, len(r) - 1)
            if lags[j] >= min_lag and r[j] >= params.loop_min_ncc and r[j] >= r[j - 1]
            and r[j] >= r[j + 1]
        ]  # fmt: skip
        per_template.append(peaks)
    if not per_template or not per_template[0]:
        return None, max(best_ncc, 0.0)
    win = max(16, int(round(LOCAL_WIN_CSS * k)))
    for lag, ncc in sorted(per_template[0]):
        if len(per_template) == 2 and not any(abs(lag - o) <= 2.0 for o, _ in per_template[1]):
            continue
        med, p10 = _local_match(img, int(round(lag)), win)
        if med >= LOOP_LOCAL_MED and p10 >= LOOP_LOCAL_P10:
            return lag, ncc
    return None, max(best_ncc, 0.0)


def _autocorr(profile: np.ndarray) -> np.ndarray:
    p = profile - profile.mean()
    n = len(p)
    f = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return np.zeros(n)
    # unbiased normalisation (fewer overlapping samples at large lags)
    ac = ac / (n - np.arange(n))
    return ac / ac[0]


def pitch(
    canvas: np.ndarray,
    valid: np.ndarray,
    k: float,
    params: ContinuousParams,
    max_lag_px: float | None,
) -> tuple[float | None, float]:
    """Card pitch (analysis px) + confidence (≤ ``cap_pitch``)."""
    lo, hi = _longest_valid_run(valid)
    if hi - lo < 8 or canvas.shape[0] < 2:
        return None, 0.0
    img = canvas[:, lo:hi]
    grad = np.abs(np.diff(img, axis=1)).mean(axis=0)
    sig = max(PROFILE_SIGMA_CSS * k, 0.5)
    prof = cv2.GaussianBlur(grad.reshape(1, -1).astype(np.float64), (0, 0), sig).ravel()
    ac = _autocorr(prof)
    n = len(ac)
    min_lag = int(math.ceil(MIN_PITCH_CSS * k))
    max_lag = int(min(n * 0.6, max_lag_px if max_lag_px else n))
    if max_lag <= min_lag + 2:
        return None, 0.0
    seg = ac[min_lag:max_lag]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    top = max(float(seg[i]) for i in peaks)
    if top <= 0.0:  # no positive periodicity (0.7 · a negative top would exceed every peak)
        return None, 0.0
    # fundamental: the smallest lag whose peak is nearly as strong as the strongest one
    lag = min(i for i in peaks if seg[i] >= FUNDAMENTAL_REL * top) + min_lag
    val = float(ac[lag])
    if val < params.pitch_min_prominence:
        return None, 0.0
    from app.pipeline.continuous.displacement import _parabolic

    lag_f = lag + _parabolic(ac, lag)
    # several periods observed -> more reliable; fewer than MIN_PITCH_PERIODS -> not a pitch
    periods = (hi - lo) / max(lag_f, 1.0)
    if periods < MIN_PITCH_PERIODS:
        return None, 0.0
    conf = min(params.cap_pitch, val * min(1.0, periods / 4.0))
    return float(lag_f), float(conf)


def gap_and_grid(
    canvas: np.ndarray, valid: np.ndarray, p_px: float, origin_lo: int
) -> tuple[float | None, float, list[float]]:
    """Gap width (analysis px), confidence and card-start canvas column phase (analysis px).

    The card borders are the edges every card shares, so after folding the edge profile modulo
    the pitch they are the two dominant peaks; the gap is the arc between them whose columns
    vary least across the perpendicular axis (background), the other arc is the card. Returns
    ``(gap, confidence, [card_start_phase_px])`` (phase relative to canvas column
    ``origin_lo``).
    """
    lo, hi = _longest_valid_run(valid)
    if hi - lo < p_px * 1.5 or p_px < 8:
        return None, 0.0, []
    img = canvas[:, lo:hi]
    bins = int(round(p_px))
    cols = np.arange(lo, hi)

    def fold(profile: np.ndarray, at: np.ndarray) -> np.ndarray:
        # + FOLD_EPS: 111 / 216 * 216 = 110.999… must land in bin 111
        idx = np.minimum(np.floor((at % p_px) / p_px * bins + FOLD_EPS).astype(int), bins - 1)
        return np.bincount(idx, weights=profile, minlength=bins) / np.maximum(
            np.bincount(idx, minlength=bins), 1
        )

    grad = np.abs(np.diff(img, axis=1)).mean(axis=0)  # edge between column j and j + 1
    g = fold(grad, cols[:-1] + 0.5)
    var = fold(img.std(axis=0), cols.astype(np.float64))
    order = np.argsort(g)[::-1]
    p1 = int(order[0])
    p2 = next((int(j) for j in order[1:] if min(abs(j - p1), bins - abs(j - p1)) >= 3), None)
    if p2 is None or g[p2] < 0.25 * g[p1] or float(np.median(g)) <= 0:
        return None, 0.0, []

    def arc(a: int, b: int) -> np.ndarray:  # bins strictly between a and b going forward
        return (np.arange(a + 1, a + 1 + (b - a - 1) % bins)) % bins

    arc_a, arc_b = arc(p1, p2), arc(p2, p1)
    if arc_a.size == 0 or arc_b.size == 0:
        return None, 0.0, []
    if float(var[arc_a].mean()) <= float(var[arc_b].mean()):
        gap_bins, start_peak = arc_a, p2
    else:
        gap_bins, start_peak = arc_b, p1
    # the strongest shared edges may be an inset image / content block rather than the card
    # border: keep only the background-flat core of the arc (cards' own margins still vary
    # across the perpendicular axis: card body vs the strip above / below it)
    v_arc = var[gap_bins]
    thr = float(v_arc.min()) + max(
        GAP_STD_ABS, GAP_STD_FRAC * (float(np.median(var)) - float(v_arc.min()))
    )
    flat = v_arc <= thr
    j_min = int(np.argmin(v_arc))
    a_, b_ = j_min, j_min
    while a_ > 0 and flat[a_ - 1]:
        a_ -= 1
    while b_ + 1 < len(flat) and flat[b_ + 1]:
        b_ += 1
    core = gap_bins[a_ : b_ + 1]
    if core.size < gap_bins.size:
        gap_bins = core
        start_peak = int(core[-1])
    gap = gap_bins.size * p_px / bins  # background columns between two cards
    if gap >= 0.5 * p_px:
        return None, 0.0, []
    contrast = min(float(g[p2]) / float(g[p1]), 1.0) * min(
        1.0, float(g[p1]) / (4.0 * float(np.median(g)))
    )
    conf = float(np.clip(contrast, 0.0, 1.0)) * 0.6
    start_col = ((start_peak + 1) * p_px / bins + origin_lo) % p_px
    return float(gap), conf, [float(start_col)]


def analyse(
    builder: PanoramaBuilder, params: ContinuousParams = DEFAULT_PARAMS.continuous
) -> PanoramaResult:
    """Loop period, pitch, gap and grid phases (CSS px) from a filled builder."""
    canvas, valid = builder.canvas()
    k = builder.k
    empty = PanoramaResult(None, 0.0, 0.0, None, 0.0, None, 0.0, [], 0.0)
    if canvas.size == 0 or not valid.any():
        return empty
    travel = max(0.0, (int(valid.sum()) - builder.crop_len) / k)
    filled = _fill(canvas, valid)
    period_px, ncc = loop_period(filled, valid, builder.crop_len, params, k)
    loop_conf = 0.0
    if period_px is not None:
        loop_conf = min(params.cap_loop, float(np.clip((ncc - 0.8) / 0.15, 0.0, 1.0)))
    # search the whole lag range: restricting it below P would turn a card-width edge pair into
    # a fake pitch when the cards are identical (true pitch == P, rejected just below)
    p_px, p_conf = pitch(filled, valid, k, params, None)
    gap = None
    gap_conf = 0.0
    phases: list[float] = []
    if p_px is not None:
        if period_px is not None and p_px >= period_px - 2.0 * k:
            p_px, p_conf = None, 0.0
    if p_px is not None:
        lo0 = getattr(builder, "_lo", 0)
        g, g_conf, starts = gap_and_grid(filled, valid, p_px, lo0)
        if g is not None:
            gap, gap_conf = g / k, min(g_conf, p_conf)
        # canvas column c ↔ crop column u = c − origin + pos·k; card start at crop column 0
        # (leading edge) → pos = (origin − c_s) / k; card centre at the box centre likewise.
        for cs in starts:
            p_css = p_px / k
            lead = ((builder.origin - cs) / k) % p_css
            card = p_px - (g if g is not None else 0.0)
            centre = ((builder.origin + builder.crop_len / 2.0 - cs - card / 2.0) / k) % p_css
            phases.extend([float(lead), float(centre)])
    return PanoramaResult(
        loop_period_px=None if period_px is None else period_px / k,
        loop_ncc=float(ncc),
        loop_confidence=loop_conf,
        pitch_px=None if p_px is None else p_px / k,
        pitch_confidence=p_conf,
        gap_px=gap,
        gap_confidence=gap_conf,
        grid_phases=phases,
        travel_px=travel,
    )


__all__ = ["PanoramaBuilder", "PanoramaResult", "analyse", "gap_and_grid", "loop_period", "pitch"]
