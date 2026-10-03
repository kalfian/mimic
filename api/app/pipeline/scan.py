"""A2 scan, pass 1 (PLAN §6.3): diff energy, blobs, segmentation, global motion, pairing.

Structure (each piece unit-testable without a video):

* :class:`Pass1Accumulator` consumes ``(t, gray)`` frames one at a time and keeps only small
  per-frame data (blobs, changed-area fraction, phase-correlation shift, 240w thumbnail).
* :func:`segment_energy` — hysteresis segmentation of the UI energy series (pure numpy).
* :func:`build_scan_result` — cursor separation, energy, segments, stable gaps, global flag,
  forward/reverse pairing and the primary interaction; raises the pipeline errors
  ``no_motion_detected`` / ``unsupported_motion`` / ``no_stable_state``.
* :func:`scan_video` — glue: stream pass 1 from ffmpeg and build the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ErrorCode, PipelineError
from app.models.ir import SpecWarning
from app.models.measure import (
    ActiveSegment,
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
    _thumb_size: tuple[int, int] | None = None

    def _to_css(self, px: float) -> float:
        return self.scale.to_css(px, self.image_scale)

    def _blobs(
        self, cur: np.ndarray, ref: np.ndarray, top_rows: bool = False
    ) -> tuple[list[Blob], float]:
        """Thresholded diff → open → dilate → components (CSS px) + changed fraction."""
        p = self.params.scan
        d = (cv2.absdiff(cur, ref) > p.diff_threshold).astype(np.uint8)
        if p.open_ksize > 1:
            d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((p.open_ksize, p.open_ksize), np.uint8))
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
        return out, frac

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
        self.slow_blobs.append([] if lagged is None else self._blobs(blur, lagged)[0])
        if prev is None:
            self.blobs.append([])
            self.changed_frac.append(0.0)
            self.shift_css.append(0.0)
            self.response.append(0.0)
            self._prev_f32 = gray.astype(np.float32)
            return
        frame_blobs, frac = self._blobs(blur, prev, top_rows=True)
        self.blobs.append(frame_blobs)
        self.changed_frac.append(frac)
        cur_f32 = gray.astype(np.float32)
        shift, resp = 0.0, 0.0
        if frac > p.global_area_frac and self._prev_f32 is not None:
            if self._window is None or self._window.shape != gray.shape:
                self._window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (sx, sy), resp = cv2.phaseCorrelate(self._prev_f32, cur_f32, self._window)
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

    def cursor_valid(self, idx: list[int]) -> np.ndarray:
        """True where no cursor box (of the given frames) covers the thumbnail pixel."""
        h, w = self.thumbs.shape[1:]
        valid = np.ones((h, w), dtype=bool)
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
# Result assembly
# --------------------------------------------------------------------------------------------


def build_scan_result(stats: Pass1Stats, scale: Scale, params: MeasureParams) -> ScanResult:
    """Cursor separation → energy → segments → global check → pairing → primary."""
    p = params.scan
    times = stats.times
    n = len(times)
    cursor, cursor_sets = build_cursor_track(stats.blobs, times, params)
    e_fast = np.zeros(n, dtype=np.float64)
    e_slow = np.zeros(n, dtype=np.float64)
    ui_boxes: list[list[Rect]] = [[] for _ in range(n)]
    for i, fb in enumerate(stats.blobs):
        for j, b in enumerate(fb):
            if j in cursor_sets[i]:
                continue
            e_fast[i] += b.area
            ui_boxes[i].append(b.rect)
    # lagged diff: drop pointer-sized blobs touching the pointer in any frame the lag spans
    # (large blobs are UI even when the pointer rests on them — hover)
    cmax = params.scan.cursor_max_css
    for i, sb in enumerate(stats.slow_blobs):
        span = range(max(0, i - SLOW_LAG), i + 1)
        crects = [stats.blobs[j][k].rect for j in span for k in cursor_sets[j]]
        crects += [
            cursor.boxes_css[j]
            for j in span
            if j < len(cursor.boxes_css) and cursor.boxes_css[j] is not None
        ]
        for b in sb:
            r = b.rect
            small = b.w <= cmax and b.h <= cmax
            if small and any(r.intersection(c) is not None for c in crects):
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
        big = [i for i in range(r.i0, r.i1 + 1) if stats.changed_frac[i] > p.global_area_frac]
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


def scan_video(
    path: Path,
    probe: ProbeInfo,
    scale: Scale,
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[ScanResult, Pass1Stats]:
    """Stream pass 1 and build the :class:`ScanResult` (stats returned for debug dumps)."""
    g = pass1_geometry(probe, params)
    acc = Pass1Accumulator(
        params=params,
        scale=scale,
        image_scale=g.image_scale,
        frame_css=(probe.width / scale.pixel_ratio, probe.height / scale.pixel_ratio),
    )
    for t, gray in iter_pass1(path, probe, params, ffmpeg_bin):
        acc.add(t, gray)
    stats = acc.finish()
    return build_scan_result(stats, scale, params), stats
