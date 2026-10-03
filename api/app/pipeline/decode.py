"""FFmpeg rawvideo readers (PLAN §6.3 decode, §6.4 A3).

* **Pass 1** (:func:`iter_pass1`): the whole video, resampled to ``params.scan.fps``, scaled to
  ``params.scan.width`` and converted to gray, streamed frame by frame. Nothing is retained here;
  ``scan.py`` keeps only per-frame statistics and thumbnails.
* **Pass 2** (:func:`decode_windows`): native timestamps (``-fps_mode passthrough``) at analysis
  resolution. The crop is applied **inside ffmpeg** so only the ROI crosses the pipe; frames are
  kept only when they fall inside a primary-interaction window.

Frame index ``i`` of the passthrough stream maps to ``probe.frame_ts[i]``. When the decoded
count disagrees with the probe, timestamps fall back to a uniform grid and the warning
``timestamps_estimated`` is returned (§6.4 consistency check).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ErrorCode, PipelineError
from app.models.ir import SegmentId, SpecWarning
from app.models.measure import (
    ActiveSegment,
    CursorTrack,
    ProbeInfo,
    Rect,
    Scale,
    ScanResult,
    WindowFrames,
)
from app.pipeline.params import MeasureParams

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# Low-level ffmpeg streaming
# --------------------------------------------------------------------------------------------


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def _stream_raw(
    argv: list[str], frame_bytes: int
) -> Iterator[bytes]:  # pragma: no cover - exercised through callers
    """Run ffmpeg and yield exactly ``frame_bytes`` per frame from stdout.

    stderr goes to a temp file (never a pipe, so a chatty ffmpeg cannot deadlock us). Raises
    :class:`PipelineError` ``decode_failed`` if ffmpeg fails before producing any frame.
    """
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err, bufsize=0)
        assert proc.stdout is not None
        produced = 0
        try:
            while True:
                buf = bytearray(frame_bytes)
                view = memoryview(buf)
                got = 0
                while got < frame_bytes:
                    n = proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < frame_bytes:
                    break  # EOF (a trailing partial frame is dropped)
                produced += 1
                yield bytes(buf)
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            rc = proc.wait()
        if rc not in (0, None) and produced == 0:
            err.seek(0)
            tail = err.read()[-800:].decode("utf-8", "replace").strip()
            log.warning("ffmpeg decode failed (rc=%s): %s", rc, tail or "<no stderr>")
            raise PipelineError(
                ErrorCode.DECODE_FAILED,
                "The video could not be decoded. Re-export it as an H.264 MP4 and upload it again.",
            )


# --------------------------------------------------------------------------------------------
# Pass 1
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pass1Geometry:
    width: int
    height: int
    image_scale: float  # pass-1 px per source px
    fps: float


def pass1_geometry(probe: ProbeInfo, params: MeasureParams) -> Pass1Geometry:
    """Output size of the pass-1 stream (``scale=W:H``, even dims, no upscaling)."""
    w = _even(min(params.scan.width, probe.width))
    h = _even(probe.height * w / probe.width)
    return Pass1Geometry(width=w, height=h, image_scale=w / probe.width, fps=float(params.scan.fps))


def iter_pass1(
    path: Path,
    probe: ProbeInfo,
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(t_s, gray uint8 (H, W))`` at ``params.scan.fps`` over the whole video.

    ``t_s = frame_ts[0] + n / fps`` (the fps filter starts at the first frame's pts).
    """
    g = pass1_geometry(probe, params)
    argv = [
        ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(path),
        "-vf", f"fps={params.scan.fps},scale={g.width}:{g.height}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    t0 = float(probe.frame_ts[0]) if len(probe.frame_ts) else 0.0
    for n, raw in enumerate(_stream_raw(argv, g.width * g.height)):
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(g.height, g.width)
        yield t0 + n / g.fps, frame


# --------------------------------------------------------------------------------------------
# Pass 2 planning
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One time window to decode at native rate. ``segment`` is the primary segment it covers."""

    segment_id: SegmentId
    start_s: float
    end_s: float
    segment: ActiveSegment


def analysis_size(probe: ProbeInfo, scale: Scale) -> tuple[int, int]:
    """Pass-2 decode size (analysis px) for the full frame."""
    return _even(probe.width * scale.analysis_scale), _even(probe.height * scale.analysis_scale)


def frame_size_css(probe: ProbeInfo, scale: Scale) -> tuple[float, float]:
    return probe.width / scale.pixel_ratio, probe.height / scale.pixel_ratio


def plan_windows(
    scan: ScanResult, probe: ProbeInfo, params: MeasureParams
) -> tuple[list[WindowSpec], list[SpecWarning]]:
    """Windows ``[start - pad, end + pad]`` per primary segment, clamped to neighbours (§6.4).

    Round trip → a single window with ``segment_id="rt_in"`` that covers both halves; A5 splits
    it at the peak into ``rt_in``/``rt_out``.
    """
    p = params.decode
    prim = scan.primary
    t_first = float(probe.frame_ts[0]) if len(probe.frame_ts) else 0.0
    t_last = float(probe.frame_ts[-1]) if len(probe.frame_ts) else probe.duration_s
    segs = sorted(scan.segments, key=lambda s: s.start_s)

    def neighbours(seg: ActiveSegment) -> tuple[float, float]:
        lo, hi = t_first, t_last
        for s in segs:
            if s is seg:
                continue
            if s.end_s <= seg.start_s:
                lo = max(lo, s.end_s)
            elif s.start_s >= seg.end_s:
                hi = min(hi, s.start_s)
        return lo, hi

    todo: list[tuple[SegmentId, ActiveSegment]] = []
    if prim.round_trip:
        todo.append(("rt_in", prim.forward))
    else:
        todo.append(("fwd", prim.forward))
        if prim.reverse is not None:
            todo.append(("rev", prim.reverse))

    out: list[WindowSpec] = []
    warnings: list[SpecWarning] = []
    short = False
    for sid, seg in todo:
        lo, hi = neighbours(seg)
        start = max(seg.start_s - p.window_pad_s, lo)
        end = min(seg.end_s + p.window_pad_s, hi)
        if seg.start_s - start < p.min_stable_side_s - 1e-6:
            short = True
        if end - seg.end_s < p.min_stable_side_s - 1e-6:
            short = True
        out.append(WindowSpec(sid, start, end, seg))
    if short:
        warnings.append(
            SpecWarning(
                code="short_stable_state",
                severity="warn",
                message="Less than 150 ms of stable frames around the interaction; "
                "start/end values may be less accurate.",
            )
        )
    return out, warnings


def crop_rect_css(
    rois_css: list[Rect], probe: ProbeInfo, scale: Scale, params: MeasureParams
) -> tuple[Rect, tuple[int, int, int, int]]:
    """Shared crop for one interaction: union of ROIs padded by ``max(48 px, 15 %)``.

    Returns ``(crop_css, (x, y, w, h) in analysis px)``; the CSS rect is recomputed from the
    integer analysis rect so both agree exactly.
    """
    p = params.decode
    roi = rois_css[0]
    for r in rois_css[1:]:
        roi = roi.union(r)
    pad = max(p.crop_pad_min_css, p.crop_pad_frac * max(roi.w, roi.h))
    fw, fh = frame_size_css(probe, scale)
    aw, ah = analysis_size(probe, scale)
    r = roi.padded(pad).clamped(fw, fh)
    k = aw / fw  # analysis px per CSS px (== scale.k up to even-rounding)
    x0 = int(math.floor(r.x * k))
    y0 = int(math.floor(r.y * k))
    x1 = min(aw, int(math.ceil(r.x2 * k)))
    y1 = min(ah, int(math.ceil(r.y2 * k)))
    w = max(2, (x1 - x0) // 2 * 2)
    h = max(2, (y1 - y0) // 2 * 2)
    w = min(w, aw - x0)
    h = min(h, ah - y0)
    return Rect(x0 / k, y0 / k, w / k, h / k), (x0, y0, w, h)


def clipped_sides(
    mask: np.ndarray, crop_css: Rect, probe: ProbeInfo, scale: Scale, margin_px: int = 2
) -> set[str]:
    """Crop sides that the change ``mask`` (crop px) touches while the frame continues beyond
    them — the pass-1 ROI missed part of a faint element and the crop must grow."""
    fw, fh = frame_size_css(probe, scale)
    h, w = mask.shape
    sides: set[str] = set()
    if crop_css.x > 0.5 and mask[:, :margin_px].any():
        sides.add("left")
    if crop_css.x2 < fw - 0.5 and mask[:, w - margin_px :].any():
        sides.add("right")
    if crop_css.y > 0.5 and mask[:margin_px, :].any():
        sides.add("top")
    if crop_css.y2 < fh - 0.5 and mask[h - margin_px :, :].any():
        sides.add("bottom")
    return sides


def expand_crop(
    crop_css: Rect, sides: set[str], probe: ProbeInfo, scale: Scale, params: MeasureParams
) -> tuple[Rect, tuple[int, int, int, int]]:
    """Grow ``crop_css`` on ``sides`` by ``max(2·crop_pad_min, 50 %)`` and re-snap to px."""
    grow_x = max(2 * params.decode.crop_pad_min_css, 0.5 * crop_css.w)
    grow_y = max(2 * params.decode.crop_pad_min_css, 0.5 * crop_css.h)
    x0 = crop_css.x - (grow_x if "left" in sides else 0)
    x1 = crop_css.x2 + (grow_x if "right" in sides else 0)
    y0 = crop_css.y - (grow_y if "top" in sides else 0)
    y1 = crop_css.y2 + (grow_y if "bottom" in sides else 0)
    # crop_rect_css pads again; pass a rect pre-shrunk by that pad so the result is [x0, x1]
    no_pad = dataclasses.replace(
        params,
        decode=dataclasses.replace(params.decode, crop_pad_min_css=0.0, crop_pad_frac=0.0),
    )
    return crop_rect_css([Rect(x0, y0, x1 - x0, y1 - y0)], probe, scale, no_pad)


# --------------------------------------------------------------------------------------------
# Pass 2 decode
# --------------------------------------------------------------------------------------------


def _cursor_mask(
    t: float,
    cursor: CursorTrack | None,
    pass1_times: np.ndarray | None,
    crop_css: Rect,
    k: float,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Union of pass-1 cursor boxes within ±1 pass-1 frame of ``t``, in crop analysis px."""
    if cursor is None or pass1_times is None or not cursor.boxes_css or len(pass1_times) == 0:
        return None
    dt = float(np.median(np.diff(pass1_times))) if len(pass1_times) > 1 else 1 / 30
    lo = np.searchsorted(pass1_times, t - dt * 1.01)
    hi = np.searchsorted(pass1_times, t + dt * 1.01, side="right")
    mask: np.ndarray | None = None
    h, w = shape
    for j in range(max(lo, 0), min(hi, len(cursor.boxes_css))):
        b = cursor.boxes_css[j]
        if b is None:
            continue
        x0 = int(math.floor((b.x - crop_css.x) * k))
        y0 = int(math.floor((b.y - crop_css.y) * k))
        x1 = int(math.ceil((b.x2 - crop_css.x) * k))
        y1 = int(math.ceil((b.y2 - crop_css.y) * k))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
    return mask


def decode_windows(
    path: Path,
    probe: ProbeInfo,
    scale: Scale,
    windows: list[WindowSpec],
    crop_css: Rect,
    crop_px: tuple[int, int, int, int],
    params: MeasureParams,
    *,
    cursor: CursorTrack | None = None,
    pass1_times: np.ndarray | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[list[WindowFrames], list[SpecWarning]]:
    """Decode the windows at native timestamps, cropped to ``crop_px`` (analysis px).

    One ffmpeg run streams the whole video; only frames inside a window are retained.
    """
    p = params.decode
    aw, ah = analysis_size(probe, scale)
    cx, cy, cw, ch = crop_px
    k = aw / (probe.width / scale.pixel_ratio)
    filters = []
    if (aw, ah) != (probe.width, probe.height):
        filters.append(f"scale={aw}:{ah}:flags=area")
    filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
    argv = [
        ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(path),
        "-fps_mode", "passthrough", "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip

    ts = np.asarray(probe.frame_ts, dtype=np.float64)
    n_probe = len(ts)
    t0 = float(ts[0]) if n_probe else 0.0
    dt_guess = float(np.median(np.diff(ts))) if n_probe > 1 else 1.0 / max(probe.fps_nominal, 1)
    lo = min(w.start_s for w in windows) - 0.15
    hi = max(w.end_s for w in windows) + 0.15

    def wanted(t: float) -> bool:
        return lo <= t <= hi

    kept: list[tuple[int, np.ndarray]] = []
    count = 0
    for i, raw in enumerate(_stream_raw(argv, cw * ch * 3)):
        count = i + 1
        t_ts = float(ts[i]) if i < n_probe else math.inf
        t_uni = t0 + i * dt_guess
        if wanted(t_ts) or wanted(t_uni):
            kept.append((i, np.frombuffer(raw, dtype=np.uint8).reshape(ch, cw, 3).copy()))
    if count == 0:
        raise PipelineError(ErrorCode.DECODE_FAILED, "No frames could be decoded.")

    warnings: list[SpecWarning] = []
    if count == n_probe and not probe.timestamps_estimated:
        times_all = ts
    else:
        span = max(probe.duration_s, dt_guess * count)
        times_all = t0 + np.arange(count) * (span / count)
        if count != n_probe:
            warnings.append(
                SpecWarning(
                    code="timestamps_estimated",
                    severity="warn",
                    message="Frame timestamps could not be matched to decoded frames; "
                    "a uniform frame rate was assumed.",
                )
            )

    # ≤ max_fps: keep every k-th frame
    step = 1
    if probe.fps_effective > p.max_fps * 1.05:
        step = int(math.ceil(probe.fps_effective / p.max_fps))

    out: list[WindowFrames] = []
    subsampled: list[tuple[str, int, int, float]] = []
    for w in windows:
        sel = [(i, f) for (i, f) in kept if w.start_s - 1e-6 <= times_all[i] <= w.end_s + 1e-6]
        if step > 1:
            sel = [(i, f) for (i, f) in sel if i % step == 0]
        if len(sel) > p.max_window_frames:
            n_before = len(sel)
            idx = np.round(np.linspace(0, len(sel) - 1, p.max_window_frames)).astype(int)
            sel = [sel[j] for j in np.unique(idx)]
            log.info("window %s: %d frames > cap %d, subsampled", w.segment_id, n_before,
                     p.max_window_frames)  # fmt: skip
            subsampled.append((w.segment_id, n_before, len(sel), w.end_s - w.start_s))
        times = np.array([times_all[i] for i, _ in sel], dtype=np.float64)
        crops = [f for _, f in sel]
        is_dup = np.zeros(len(crops), dtype=bool)
        for j in range(1, len(crops)):
            d = int(cv2.absdiff(crops[j], crops[j - 1]).max())  # uint8 |a − b|, no int16 copies
            is_dup[j] = d <= p.dup_max_abs_diff
        masks = [_cursor_mask(t, cursor, pass1_times, crop_css, k, (ch, cw)) for t in times]
        out.append(
            WindowFrames(
                segment_id=w.segment_id,
                times=times,
                crops=crops,
                crop_css=crop_css,
                scale=scale,
                is_dup=is_dup,
                cursor_masks=masks,
            )
        )
    if subsampled:
        warnings.append(subsample_warning(subsampled))
    return out, warnings


def subsample_warning(windows: list[tuple[str, int, int, float]]) -> SpecWarning:
    """``frames_subsampled``: a window had more than ``max_window_frames`` frames (§6.4).

    ``windows`` = ``(segment_id, frames_in_window, frames_kept, window_s)``. The message names
    the effective analysis rate of the most reduced window; the IR's ``timing_resolution_ms``
    (median frame step of the forward window) already reflects the coarser sampling.
    """
    _, n_in, n_kept, span_s = min(windows, key=lambda w: w[2] / max(w[1], 1))
    fps_kept = (n_kept - 1) / span_s if span_s > 0 else 0.0
    return SpecWarning(
        code="frames_subsampled",
        severity="info",
        message=(
            f"The motion lasts long ({n_in} frames around it), so it was analysed at about "
            f"{fps_kept:.0f} fps instead of every frame; start times and durations are less "
            "precise."
        ),
    )
