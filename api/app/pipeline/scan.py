"""A2 scan, pass 1 (PLAN §6.3): diff energy, blobs, segmentation, global motion, pairing.

Structure (each piece unit-testable without a video):

* :class:`Pass1Accumulator` consumes ``(t, gray)`` frames one at a time and keeps only small
  per-frame data (blobs, changed-area fraction, phase-correlation shift, 240w thumbnail, and
  the coarse cell-activity grid used by regime detection, PLAN-continuous §3.1).
* :func:`segment_energy` — hysteresis segmentation of the UI energy series (pure numpy).
* :func:`build_scan_result` — cursor separation, energy, segments, stable gaps, global flag,
  forward/reverse pairing and the primary interaction; raises the pipeline errors
  ``no_motion_detected`` / ``unsupported_motion`` / ``no_stable_state``. With ``ambient``
  regions (PLAN-continuous §3.4) the same steps run with those regions masked out; without
  them the computation is exactly the transition-only path.
* :func:`scan_video` — glue: stream pass 1 from ffmpeg and build the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ErrorCode, PipelineError
from app.models.ir import SpecWarning
from app.models.measure import (
    ActiveSegment,
    AmbientRegion,
    CursorTrack,
    PrimaryInteraction,
    ProbeInfo,
    Rect,
    Scale,
    ScanResult,
    StableSegment,
)
from app.pipeline.cursor import Blob, build_cursor_track
from app.pipeline.decode import iter_pass1, pass1_geometry
from app.pipeline.params import MeasureParams

#: Second, lagged diff ``|g_i − g_{i−3}|`` (100 ms at 30 fps) so slow fades/colour changes whose
#: per-frame step stays under ``diff_threshold`` still register as UI energy.
#: TODO(calibrate): module-local; candidate for ``ScanParams``.
SLOW_LAG = 3

# --------------------------------------------------------------------------------------------
# Per-frame accumulation
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Pass1Stats:
    """Everything pass 1 retains (small)."""

    times: np.ndarray  # (N,)
    blobs: list[list[Blob]]  # per frame; frame 0 has none
    slow_blobs: list[list[Blob]]  # per frame: blobs of |g_i − g_{i−SLOW_LAG}| (slow fades)
    changed_frac: np.ndarray  # (N,) changed pixel fraction of the frame
    shift_css: np.ndarray  # (N,) |phaseCorrelate shift| in CSS px (0 if not evaluated)
    response: np.ndarray  # (N,) phaseCorrelate response (0 if not evaluated)
    thumbs: np.ndarray  # uint8 (N, th, tw)
    image_scale: float  # pass-1 px per source px
    thumb_scale: float  # thumbnail px per source px
    frame_css: tuple[float, float]  # full frame size in CSS px
    #: bool (N, gh, gw): cell changed in the fast or the lagged diff (PLAN-continuous §3.1).
    #: Cells tile the full frame: ``gw = ceil(frame_w_css / cell_css)`` (same for rows), so one
    #: cell is ``frame_css / (gw, gh)`` CSS px (≈ ``cell_css``). None = not recorded.
    activity: np.ndarray | None = None


def activity_grid_shape(frame_css: tuple[float, float], cell_css: float) -> tuple[int, int]:
    """``(rows, cols)`` of the pass-1 activity grid for a frame of ``frame_css`` CSS px."""
    w, h = frame_css
    return (max(1, math.ceil(h / cell_css - 1e-9)), max(1, math.ceil(w / cell_css - 1e-9)))


@dataclass(slots=True)
class Pass1Accumulator:
    """Streaming consumer for pass-1 gray frames."""

    params: MeasureParams
    scale: Scale
    image_scale: float  # pass-1 px per source px
    frame_css: tuple[float, float]
    _prev_blur: np.ndarray | None = None
    _prev_f32: np.ndarray | None = None
    _window: np.ndarray | None = None
    _history: list[np.ndarray] = field(default_factory=list)  # last SLOW_LAG blurred frames
    times: list[float] = field(default_factory=list)
    blobs: list[list[Blob]] = field(default_factory=list)
    slow_blobs: list[list[Blob]] = field(default_factory=list)
    changed_frac: list[float] = field(default_factory=list)
    shift_css: list[float] = field(default_factory=list)
    response: list[float] = field(default_factory=list)
    thumbs: list[np.ndarray] = field(default_factory=list)
    activity: list[np.ndarray] = field(default_factory=list)
    _thumb_size: tuple[int, int] | None = None

    def _to_css(self, px: float) -> float:
        return self.scale.to_css(px, self.image_scale)

    def _blobs(
        self, cur: np.ndarray, ref: np.ndarray, top_rows: bool = False
    ) -> tuple[list[Blob], float, np.ndarray]:
        """Thresholded diff → open → dilate → components (CSS px) + changed fraction.

        Also returns the opened diff before dilation (uint8 0/1) for the activity grid.
        """
        p = self.params.scan
        d = (cv2.absdiff(cur, ref) > p.diff_threshold).astype(np.uint8)
        if p.open_ksize > 1:
            d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((p.open_ksize, p.open_ksize), np.uint8))
        opened = d
        dil = max(1, int(round(self.scale.from_css(p.dilate_css, self.image_scale))))
        d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dil + 1, 2 * dil + 1)))
        frac = float(d.mean())  # changed area (as counted in the energy) / frame area
        n, labels, stats, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
        f = self._to_css(1.0)
        out: list[Blob] = []
        for j in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[j])
            top_x = float(x + bw / 2.0)
            if top_rows and bw * f <= p.cursor_max_css and bh * f <= p.cursor_max_css:
                row = np.nonzero(labels[y, x : x + bw] == j)[0]
                if row.size:
                    top_x = float(x + row.mean())
            out.append(
                Blob(x=x * f, y=y * f, w=bw * f, h=bh * f, area=area * f * f, top_x=top_x * f)
            )
        return out, frac, opened

    def _cells(self, *diffs: np.ndarray | None) -> np.ndarray:
        """Cell activity (PLAN-continuous §3.1): ≥ ``cell_active_frac`` of a cell's pixels
        changed in any of the given opened diffs."""
        c = self.params.continuous
        gh, gw = activity_grid_shape(self.frame_css, c.cell_css)
        act = np.zeros((gh, gw), dtype=bool)
        for d in diffs:
            if d is not None:
                frac = cv2.resize(d.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
                act |= frac >= c.cell_active_frac
        return act

    def add(self, t: float, gray: np.ndarray) -> None:
        p = self.params.scan
        h, w = gray.shape
        if self._thumb_size is None:
            tw = min(p.thumb_width, w)
            self._thumb_size = (tw, max(1, int(round(h * tw / w))))
        self.thumbs.append(cv2.resize(gray, self._thumb_size, interpolation=cv2.INTER_AREA))
        self.times.append(float(t))
        blur = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
        prev = self._prev_blur
        self._prev_blur = blur
        lagged = self._history[0] if len(self._history) >= SLOW_LAG else None
        self._history.append(blur)
        if len(self._history) > SLOW_LAG:
            self._history.pop(0)
        slow_d: np.ndarray | None = None
        if lagged is None:
            self.slow_blobs.append([])
        else:
            slow, _, slow_d = self._blobs(blur, lagged)
            self.slow_blobs.append(slow)
        if prev is None:
            self.activity.append(self._cells(slow_d))
            self.blobs.append([])
            self.changed_frac.append(0.0)
            self.shift_css.append(0.0)
            self.response.append(0.0)
            self._prev_f32 = gray.astype(np.float32)
            return
        frame_blobs, frac, fast_d = self._blobs(blur, prev, top_rows=True)
        self.activity.append(self._cells(fast_d, slow_d))
        self.blobs.append(frame_blobs)
        self.changed_frac.append(frac)
        cur_f32 = gray.astype(np.float32)
        shift, resp = 0.0, 0.0
        if frac > p.global_area_frac and self._prev_f32 is not None:
            if self._window is None or self._window.shape != gray.shape:
                self._window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            # OpenCV 5 multiplies both inputs by the window **in place**: pass copies, or
            # ``cur_f32`` (kept as the next frame's reference) would be windowed twice.
            (sx, sy), resp = cv2.phaseCorrelate(self._prev_f32.copy(), cur_f32.copy(), self._window)
            shift = self._to_css(math.hypot(sx, sy))
        self._prev_f32 = cur_f32
        self.shift_css.append(shift)
        self.response.append(float(resp))

    def finish(self) -> Pass1Stats:
        if not self.times:
            raise PipelineError(ErrorCode.DECODE_FAILED, "No frames could be decoded.")
        tw = self._thumb_size[0] if self._thumb_size else self.params.scan.thumb_width
        return Pass1Stats(
            times=np.asarray(self.times, dtype=np.float64),
            blobs=self.blobs,
            slow_blobs=self.slow_blobs,
            changed_frac=np.asarray(self.changed_frac),
            shift_css=np.asarray(self.shift_css),
            response=np.asarray(self.response),
            thumbs=np.stack(self.thumbs),
            image_scale=self.image_scale,
            thumb_scale=self.image_scale * tw / self._prev_blur.shape[1]
            if self._prev_blur is not None
            else self.image_scale,
            frame_css=self.frame_css,
            activity=np.stack(self.activity),
        )


# --------------------------------------------------------------------------------------------
# Segmentation (pure)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnergyThresholds:
    floor: float
    mad: float
    hi: float
    lo: float


def energy_thresholds(energy: np.ndarray, params: MeasureParams) -> EnergyThresholds:
    p = params.scan
    e = np.asarray(energy, dtype=np.float64)
    floor = float(np.median(e)) if e.size else 0.0
    mad = float(np.median(np.abs(e - floor))) if e.size else 0.0
    return EnergyThresholds(
        floor=floor,
        mad=mad,
        hi=max(floor + p.hi_mad_k * mad, p.energy_min_hi_px2),
        lo=max(floor + p.lo_mad_k * mad, p.energy_min_lo_px2),
    )


@dataclass(frozen=True, slots=True)
class Run:
    """Active frames ``[i0, i1]`` (inclusive; energy index = change from frame i-1 to i)."""

    i0: int
    i1: int
    settled: bool = True


def segment_energy(energy: np.ndarray, times: np.ndarray, params: MeasureParams) -> list[Run]:
    """Hysteresis segmentation + gap merging + compression-blip removal (§6.3)."""
    p = params.scan
    e = np.asarray(energy, dtype=np.float64)
    n = len(e)
    if n == 0:
        return []
    th = energy_thresholds(e, params)
    dt = float(np.median(np.diff(times))) if n > 1 else 1.0 / p.fps
    runs: list[Run] = []
    i = 0
    while i < n:
        if e[i] > th.hi:
            i0 = i
            while i0 - 1 >= 1 and e[i0 - 1] >= th.lo:  # include the soft onset
                i0 -= 1
            quiet = 0
            last = i
            j = i + 1
            while j < n:
                if e[j] < th.lo:
                    quiet += 1
                    if quiet >= p.settle_frames:
                        break
                else:
                    quiet = 0
                    last = j
                j += 1
            settled = j < n
            runs.append(Run(i0, last, settled))
            i = j + 1 if settled else n
        else:
            i += 1
    # merge runs separated by < merge_gap (stagger tails)
    merged: list[Run] = []
    for r in runs:
        if merged and (r.i0 - merged[-1].i1 - 1) * dt < p.merge_gap_s - 1e-9:
            prev = merged.pop()
            merged.append(Run(prev.i0, r.i1, r.settled))
        else:
            merged.append(r)
    return [r for r in merged if float(e[r.i0 : r.i1 + 1].sum()) >= p.energy_min_total_px2]


# --------------------------------------------------------------------------------------------
# Pairing helpers
# --------------------------------------------------------------------------------------------


def _roi_px(roi: Rect, k: float, shape: tuple[int, int]) -> tuple[slice, slice]:
    h, w = shape
    x0 = int(max(0, math.floor(roi.x * k)))
    y0 = int(max(0, math.floor(roi.y * k)))
    x1 = int(min(w, math.ceil(roi.x2 * k)))
    y1 = int(min(h, math.ceil(roi.y2 * k)))
    return slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1))


def _masked_diff(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    v = valid.astype(bool)
    return float(d[v].mean()) if v.any() else float(d.mean())


@dataclass(slots=True)
class _ThumbCtx:
    thumbs: np.ndarray
    times: np.ndarray
    k: float  # thumb px per CSS px
    cursor: CursorTrack
    ambient: list[Rect] = field(default_factory=list)  # masked regions (PLAN-continuous §3.4)

    def cursor_valid(self, idx: list[int]) -> np.ndarray:
        """True where no cursor box (of the given frames) nor ambient mask covers the pixel."""
        h, w = self.thumbs.shape[1:]
        valid = np.ones((h, w), dtype=bool)
        for r in self.ambient:
            ys, xs = _roi_px(r, self.k, (h, w))
            valid[ys, xs] = False
        if not self.cursor.boxes_css:
            return valid
        for i in idx:
            b = self.cursor.boxes_css[i] if i < len(self.cursor.boxes_css) else None
            if b is not None:
                ys, xs = _roi_px(b, self.k, (h, w))
                valid[ys, xs] = False
        return valid

    def frames_in(self, t0: float, t1: float) -> list[int]:
        return [i for i, t in enumerate(self.times) if t0 - 1e-6 <= t <= t1 + 1e-6]

    def representative(self, st: StableSegment, n_mid: int) -> tuple[np.ndarray, list[int]]:
        idx = self.frames_in(st.start_s, st.end_s)
        if not idx:
            i = int(np.argmin(np.abs(self.times - (st.start_s + st.end_s) / 2)))
            idx = [i]
        mid = len(idx) // 2
        lo = max(0, mid - n_mid // 2)
        sel = idx[lo : lo + n_mid]
        return self.thumbs[sel].astype(np.float32).mean(axis=0), sel


# --------------------------------------------------------------------------------------------
# Ambient masking (PLAN-continuous §3.4)
# --------------------------------------------------------------------------------------------


def ambient_mask_rects(
    ambient: Sequence[AmbientRegion | Rect], frame_css: tuple[float, float], params: MeasureParams
) -> list[Rect]:
    """Ambient regions as CSS rects grown by ``ambient_mask_dilate_css``, clamped to the frame."""
    pad = params.continuous.ambient_mask_dilate_css
    out: list[Rect] = []
    for a in ambient:
        r = a.rect_css if isinstance(a, AmbientRegion) else a
        c = r.padded(pad).clamped(*frame_css)
        if c.area > 0:
            out.append(c)
    return out


def subtract_masks(r: Rect, masks: Sequence[Rect]) -> tuple[float, Rect | None]:
    """``(fraction of r's area outside every mask, bbox of that remainder or None)``.

    Exact for axis-aligned rects (coordinate compression over the mask edges inside ``r``).
    """
    hits = [m for m in masks if r.intersection(m) is not None]
    if not hits:
        return 1.0, r
    if r.area <= 0:
        return 0.0, None
    xs = sorted({r.x, r.x2, *(v for m in hits for v in (m.x, m.x2) if r.x < v < r.x2)})
    ys = sorted({r.y, r.y2, *(v for m in hits for v in (m.y, m.y2) if r.y < v < r.y2)})
    kept = 0.0
    bx0 = by0 = math.inf
    bx1 = by1 = -math.inf
    for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
        cx = (x0 + x1) / 2.0
        for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
            cy = (y0 + y1) / 2.0
            if any(m.x <= cx <= m.x2 and m.y <= cy <= m.y2 for m in hits):
                continue
            kept += (x1 - x0) * (y1 - y0)
            bx0, by0, bx1, by1 = min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)
    if kept <= 0:
        return 0.0, None
    return kept / r.area, Rect(bx0, by0, bx1 - bx0, by1 - by0)


@dataclass(frozen=True, slots=True)
class _MaskedBlob:
    area: float  # CSS px² outside the masks
    rect: Rect  # bbox of the unmasked part


def _mask_blobs(blobs: list[list[Blob]], masks: Sequence[Rect]) -> list[list[_MaskedBlob | None]]:
    """Per blob: its unmasked area/box, or None when the blob lies entirely inside the masks."""
    out: list[list[_MaskedBlob | None]] = []
    for fb in blobs:
        row: list[_MaskedBlob | None] = []
        for b in fb:
            keep, box = subtract_masks(b.rect, masks)
            row.append(None if box is None else _MaskedBlob(b.area * keep, box))
        out.append(row)
    return out


def masked_cursor_track(
    stats: Pass1Stats, ambient: Sequence[AmbientRegion | Rect], params: MeasureParams
) -> CursorTrack:
    """Cursor track with the blobs lying entirely inside the (grown) ambient regions removed.

    Content moving inside a scroller forms pointer-like tracks and makes every frame "busy"
    (``cursor.MAX_BIG_FRAC``), so the pointer is only tracked outside the regions; while it is
    over a region it is unseen (``continuous.phases`` infers that occupancy from where the
    track is lost / found again).
    """
    masks = ambient_mask_rects(ambient, stats.frame_css, params) if ambient else []
    blobs = stats.blobs
    if masks:
        blobs = [
            [b for b, mb in zip(fb, row, strict=True) if mb is not None]
            for fb, row in zip(stats.blobs, _mask_blobs(stats.blobs, masks), strict=True)
        ]
    return build_cursor_track(blobs, stats.times, params)[0]


def _masked_changed_frac(
    stats: Pass1Stats, masked: list[list[_MaskedBlob | None]], masks: Sequence[Rect]
) -> np.ndarray:
    """Changed fraction of the frame area outside the masks (global-motion check, §3.4).

    The dilated-diff components partition the changed pixels, so Σ blob areas == changed area.
    """
    fw, fh = stats.frame_css
    keep, _ = subtract_masks(Rect(0.0, 0.0, fw, fh), masks)
    free = keep * fw * fh
    out = np.zeros(len(masked), dtype=np.float64)
    if free <= 0:
        return out
    for i, row in enumerate(masked):
        out[i] = sum(mb.area for mb in row if mb is not None) / free
    return out


# --------------------------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------------------------


def build_scan_result(
    stats: Pass1Stats,
    scale: Scale,
    params: MeasureParams,
    ambient: Sequence[AmbientRegion | Rect] | None = None,
) -> ScanResult:
    """Cursor separation → energy → segments → global check → pairing → primary.

    ``ambient`` (PLAN-continuous §3.4): regions moving from the start, masked out (grown by
    ``ambient_mask_dilate_css``): blob areas shrink by their masked fraction, blobs entirely
    inside are dropped (also from cursor tracking — content moving inside a scroller would
    otherwise form pointer-like tracks), ROI boxes cover only the unmasked part, the global
    check uses the unmasked changed fraction, and pairing ignores masked thumbnail pixels.
    ``None`` / empty → the transition-only computation, unchanged.
    """
    p = params.scan
    times = stats.times
    n = len(times)
    masks = ambient_mask_rects(ambient, stats.frame_css, params) if ambient else []
    blobs = stats.blobs
    changed_frac = stats.changed_frac
    kept: list[list[_MaskedBlob]] = []  # masked mode: unmasked part of each blob in ``blobs``
    slow_masked: list[list[_MaskedBlob | None]] = []
    if masks:
        fast_masked = _mask_blobs(stats.blobs, masks)
        slow_masked = _mask_blobs(stats.slow_blobs, masks)
        changed_frac = _masked_changed_frac(stats, fast_masked, masks)
        blobs = [
            [b for b, mb in zip(fb, row, strict=True) if mb is not None]
            for fb, row in zip(stats.blobs, fast_masked, strict=True)
        ]
        kept = [[mb for mb in row if mb is not None] for row in fast_masked]
    cursor, cursor_sets = build_cursor_track(blobs, times, params)
    e_fast = np.zeros(n, dtype=np.float64)
    e_slow = np.zeros(n, dtype=np.float64)
    ui_boxes: list[list[Rect]] = [[] for _ in range(n)]
    for i, fb in enumerate(blobs):
        for j, b in enumerate(fb):
            if j in cursor_sets[i]:
                continue
            if masks:
                e_fast[i] += kept[i][j].area
                ui_boxes[i].append(kept[i][j].rect)
                continue
            e_fast[i] += b.area
            ui_boxes[i].append(b.rect)
    # lagged diff: drop pointer-sized blobs touching the pointer in any frame the lag spans
    # (large blobs are UI even when the pointer rests on them — hover)
    cmax = params.scan.cursor_max_css
    for i, sb in enumerate(stats.slow_blobs):
        span = range(max(0, i - SLOW_LAG), i + 1)
        crects = [blobs[j][k].rect for j in span for k in cursor_sets[j]]
        crects += [
            cursor.boxes_css[j]
            for j in span
            if j < len(cursor.boxes_css) and cursor.boxes_css[j] is not None
        ]
        for j, b in enumerate(sb):
            r = b.rect
            small = b.w <= cmax and b.h <= cmax
            if small and any(r.intersection(c) is not None for c in crects):
                continue
            if masks:
                mb = slow_masked[i][j]
                if mb is None:
                    continue
                e_slow[i] += mb.area
                ui_boxes[i].append(mb.rect)
                continue
            e_slow[i] += b.area
            ui_boxes[i].append(r)
    energy = np.maximum(e_fast, e_slow)

    runs = segment_energy(energy, times, params)
    if not runs:
        raise PipelineError(ErrorCode.NO_MOTION_DETECTED)
    dt = float(np.median(np.diff(times))) if n > 1 else 1.0 / p.fps
    warnings: list[SpecWarning] = []
    segments: list[ActiveSegment] = []
    for r in runs:
        roi: Rect | None = None
        for i in range(r.i0, r.i1 + 1):
            for b in ui_boxes[i]:
                roi = b if roi is None else roi.union(b)
        assert roi is not None
        big = [i for i in range(r.i0, r.i1 + 1) if changed_frac[i] > p.global_area_frac]
        glob = [
            i
            for i in big
            if stats.shift_css[i] > p.global_shift_css and stats.response[i] > p.global_response
        ]
        is_global = bool(glob) and len(glob) >= 0.5 * len(big)
        # onset found only by the lagged diff → the change may have begun up to SLOW_LAG
        # frames earlier (keep the pre-state clean)
        back = SLOW_LAG if e_slow[r.i0] > e_fast[r.i0] else 1
        start = float(times[max(0, r.i0 - back)]) if r.i0 >= 1 else float(times[0]) - dt
        segments.append(
            ActiveSegment(
                start_s=start,
                end_s=float(times[r.i1]),
                roi_css=roi,
                peak_energy=float(energy[r.i0 : r.i1 + 1].max()),
                is_global=is_global,
            )
        )
    if not runs[-1].settled:
        warnings.append(
            SpecWarning(
                code="not_settled",
                severity="warn",
                message="The motion had not settled when the recording ended; "
                "end values may be incomplete.",
            )
        )

    # stable gaps
    t_first, t_last = float(times[0]), float(times[-1])
    stable: list[StableSegment] = []
    bounds = [t_first] + [x for s in segments for x in (s.start_s, s.end_s)] + [t_last]
    for a, b in zip(bounds[0::2], bounds[1::2], strict=True):
        if b - a >= p.min_stable_s - 1e-6:
            stable.append(StableSegment(a, b))

    non_global = [s for s in segments if not s.is_global]
    if not non_global:
        raise PipelineError(ErrorCode.UNSUPPORTED_MOTION)
    if segments[0].start_s - t_first < p.min_stable_s - 1e-6:
        raise PipelineError(ErrorCode.NO_STABLE_STATE)

    ctx = _ThumbCtx(
        thumbs=stats.thumbs,
        times=times,
        k=stats.thumb_scale * scale.pixel_ratio,
        cursor=cursor,
        ambient=masks,
    )
    primary = choose_primary(segments, stable, ctx, params)
    used = 1 + (1 if primary.reverse is not None else 0)
    extra = len(segments) - used
    if extra > 0:
        warnings.append(
            SpecWarning(
                code="extra_segments_ignored",
                severity="info",
                message=f"{extra} other motion segment(s) were ignored; "
                "only the first interaction is analysed.",
            )
        )
    return ScanResult(
        energy=energy,
        frame_times=times,
        segments=segments,
        stable=stable,
        thumbs=stats.thumbs,
        primary=primary,
        cursor=cursor,
        warnings=warnings,
    )


def _stable_before(seg: ActiveSegment, stable: list[StableSegment]) -> StableSegment | None:
    for st in stable:
        if abs(st.end_s - seg.start_s) < 1e-6:
            return st
    return None


def _stable_after(seg: ActiveSegment, stable: list[StableSegment]) -> StableSegment | None:
    for st in stable:
        if abs(st.start_s - seg.end_s) < 1e-6:
            return st
    return None


def choose_primary(
    segments: list[ActiveSegment],
    stable: list[StableSegment],
    ctx: _ThumbCtx,
    params: MeasureParams,
) -> PrimaryInteraction:
    """First non-global segment (+ its reverse if the pair test passes, or round trip)."""
    p = params.scan
    order = sorted(segments, key=lambda s: s.start_s)
    idx = next(i for i, s in enumerate(order) if not s.is_global)
    fwd = order[idx]
    pre = _stable_before(fwd, stable)
    post = _stable_after(fwd, stable)
    if pre is None:
        return PrimaryInteraction(forward=fwd)
    rep_pre, sel_pre = ctx.representative(pre, p.representative_thumbs)
    shape = ctx.thumbs.shape[1:]

    def roi_diff(a: np.ndarray, b: np.ndarray, roi: Rect, sel: list[int]) -> float:
        ys, xs = _roi_px(roi, ctx.k, shape)
        valid = ctx.cursor_valid(sel)[ys, xs]
        return _masked_diff(a[ys, xs], b[ys, xs], valid)

    # forward/reverse pair
    if post is not None and idx + 1 < len(order):
        rev = order[idx + 1]
        post_rev = _stable_after(rev, stable)
        if (
            not rev.is_global
            and post_rev is not None
            and fwd.roi_css.iou(rev.roi_css) >= (p.pair_roi_iou)
        ):
            rep_post, sel_post = ctx.representative(post, p.representative_thumbs)
            rep_back, sel_back = ctx.representative(post_rev, p.representative_thumbs)
            roi = fwd.roi_css.union(rev.roi_css)
            d_fwd = roi_diff(rep_post, rep_pre, roi, sel_pre + sel_post)
            d_back = roi_diff(rep_back, rep_pre, roi, sel_pre + sel_back)
            if d_fwd > 0 and d_back <= p.pair_ratio * d_fwd:
                return PrimaryInteraction(forward=fwd, reverse=rev)
    # round trip (press / pulse)
    if post is not None:
        rep_post, sel_post = ctx.representative(post, p.representative_thumbs)
        d_end = roi_diff(rep_post, rep_pre, fwd.roi_css, sel_pre + sel_post)
        inside = ctx.frames_in(fwd.start_s, fwd.end_s)
        peak = 0.0
        for i in inside:
            peak = max(peak, roi_diff(ctx.thumbs[i].astype(np.float32), rep_pre, fwd.roi_css,
                                      sel_pre + [i]))  # fmt: skip
        if peak > 0 and d_end <= p.pair_ratio * peak:
            return PrimaryInteraction(forward=fwd, round_trip=True)
    return PrimaryInteraction(forward=fwd)


# --------------------------------------------------------------------------------------------
# Glue
# --------------------------------------------------------------------------------------------


def pass1_stats(
    path: Path,
    probe: ProbeInfo,
    scale: Scale,
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
) -> Pass1Stats:
    """Stream pass 1 into :class:`Pass1Stats` (no segmentation; never raises for "no motion")."""
    g = pass1_geometry(probe, params)
    acc = Pass1Accumulator(
        params=params,
        scale=scale,
        image_scale=g.image_scale,
        frame_css=(probe.width / scale.pixel_ratio, probe.height / scale.pixel_ratio),
    )
    for t, gray in iter_pass1(path, probe, params, ffmpeg_bin):
        acc.add(t, gray)
    return acc.finish()


def scan_video(
    path: Path,
    probe: ProbeInfo,
    scale: Scale,
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[ScanResult, Pass1Stats]:
    """Stream pass 1 and build the :class:`ScanResult` (stats returned for debug dumps)."""
    stats = pass1_stats(path, probe, scale, params, ffmpeg_bin)
    return build_scan_result(stats, scale, params), stats
