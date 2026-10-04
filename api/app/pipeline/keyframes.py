"""A9 — keyframe PNGs (PLAN §6.11).

Full frames are grabbed with an accurate input seek (``ffmpeg -ss t -i input -frames:v 1``),
scaled so the long side is ≤ ``params.keyframes.max_long_side``:

* ``state_a`` / ``state_b`` (stable midpoints), ``mid_25`` / ``mid_50`` / ``mid_75``
* ``annotated_a`` / ``annotated_b``: element boxes + ids drawn with OpenCV
* ``el_<id>.png``: A|B side-by-side crop per element (≤ ``max_element_crops``)

Continuous mode (PLAN-continuous §9 P2, :func:`write_continuous_keyframes`): ``state_a`` (a rest
frame when there is one), ``annotated_a`` (scroller / card / title boxes) and ``phase_<k>`` at
phase midpoints (≤ ``MAX_PHASE_KEYFRAMES``).

Names match ``api.schemas.KEYFRAME_NAME_RE``; kinds match ``KeyframeKind``.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ErrorCode, PipelineError
from app.models.measure import ElementCandidate, ProbeInfo, Rect, Scale
from app.pipeline.params import MeasureParams

log = logging.getLogger(__name__)

#: Distinct BGR colours for element annotations (cycled).
PALETTE: list[tuple[int, int, int]] = [
    (40, 40, 230), (40, 170, 40), (230, 120, 30), (180, 40, 180),
    (30, 170, 220), (120, 120, 120), (200, 60, 120), (60, 120, 200),
]  # fmt: skip


@dataclass(frozen=True, slots=True)
class KeyframeFile:
    name: str  # e.g. "state_a.png", "el_e1.png"
    kind: str  # KeyframeKind
    t_s: float | None
    element_id: str | None
    path: Path


#: Continuous mode: at most this many ``phase_<k>.png`` frames.
MAX_PHASE_KEYFRAMES = 6


def keyframe_size(probe: ProbeInfo, params: MeasureParams) -> tuple[int, int]:
    m = params.keyframes.max_long_side
    long = max(probe.width, probe.height)
    f = min(1.0, m / long)
    return max(2, int(round(probe.width * f / 2)) * 2), max(2, int(round(probe.height * f / 2)) * 2)


def css_to_img(probe: ProbeInfo, scale: Scale, params: MeasureParams) -> float:
    """Keyframe image px per CSS px."""
    return keyframe_size(probe, params)[0] / (probe.width / scale.pixel_ratio)


def grab_frame(
    path: Path, t_s: float, size: tuple[int, int], ffmpeg_bin: str = "ffmpeg"
) -> np.ndarray:
    """One BGR frame at ``t_s`` (accurate seek), scaled to ``size``."""
    w, h = size
    argv = [
        ffmpeg_bin, "-v", "error", "-nostdin", "-ss", f"{max(t_s, 0.0):.4f}", "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={w}:{h}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    proc = subprocess.run(argv, capture_output=True, timeout=60)
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        if t_s > 0.05:  # seeking past the last frame → step back once
            return grab_frame(path, t_s - 0.05, size, ffmpeg_bin)
        tail = proc.stderr.decode("utf-8", "replace")[-300:].strip()
        log.warning("keyframe grab failed at %.3fs: %s", t_s, tail or proc.returncode)
        raise PipelineError(
            ErrorCode.DECODE_FAILED,
            f"A frame at {t_s:.2f} s could not be read from the video. Re-export it as an H.264 "
            "MP4 and upload it again.",
        )
    return np.frombuffer(proc.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def _scaled(r: Rect, f: float) -> tuple[int, int, int, int]:
    return (int(round(r.x * f)), int(round(r.y * f)), int(round(r.x2 * f)), int(round(r.y2 * f)))


def annotate(img: np.ndarray, boxes: list[tuple[str, Rect]], css_to_img: float) -> np.ndarray:
    out = img.copy()
    th = max(1, int(round(img.shape[1] / 800)))
    for n, (eid, r) in enumerate(boxes):
        c = PALETTE[n % len(PALETTE)]
        x0, y0, x1, y1 = _scaled(r, css_to_img)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, th + 1, cv2.LINE_AA)
        fs = 0.45 * max(1.0, img.shape[1] / 1000)
        (tw, tht), base = cv2.getTextSize(eid, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ly = max(tht + 4, y0)
        cv2.rectangle(out, (x0, ly - tht - 4), (x0 + tw + 6, ly + base - 2), c, -1)
        cv2.putText(out, eid, (x0 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    th, cv2.LINE_AA)  # fmt: skip
    return out


def element_crop(
    img_a: np.ndarray, img_b: np.ndarray, box: Rect, css_to_img: float, max_long: int
) -> np.ndarray:
    """A|B side-by-side crop of ``box`` (CSS px) padded 12 px, 4 px white separator."""
    h, w = img_a.shape[:2]
    x0, y0, x1, y1 = _scaled(box.padded(12.0), css_to_img)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        x0, y0, x1, y1 = 0, 0, w, h
    a, b = img_a[y0:y1, x0:x1], img_b[y0:y1, x0:x1]
    sep = np.full((a.shape[0], 4, 3), 255, np.uint8)
    out = np.concatenate([a, sep, b], axis=1)
    f = min(1.0, max_long / max(out.shape[:2]))
    if f < 1.0:
        out = cv2.resize(out, (max(1, int(out.shape[1] * f)), max(1, int(out.shape[0] * f))),
                         interpolation=cv2.INTER_AREA)  # fmt: skip
    return out


def _saver(out_dir: Path, files: list[KeyframeFile]):
    def save(name: str, img: np.ndarray, kind: str, t: float | None, eid: str | None = None):
        p = out_dir / name
        if not cv2.imwrite(str(p), img):
            raise PipelineError(
                ErrorCode.INTERNAL_ERROR,
                "The keyframe images could not be saved (is the disk full?).",
            )
        files.append(KeyframeFile(name, kind, t, eid, p))

    return save


def _grab_all(input_path: Path, times: dict, size: tuple[int, int], ffmpeg_bin: str) -> dict:
    if not times:
        return {}
    with ThreadPoolExecutor(max_workers=len(times)) as pool:
        futures = {k: pool.submit(grab_frame, input_path, t, size, ffmpeg_bin)
                   for k, t in times.items()}  # fmt: skip
        return {k: f.result() for k, f in futures.items()}


def pick_phase_times(midpoints: list[float], limit: int = MAX_PHASE_KEYFRAMES) -> list[float]:
    """At most ``limit`` phase midpoints, evenly spread over the list (time order kept)."""
    n = len(midpoints)
    if n <= limit:
        return list(midpoints)
    picks = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [midpoints[i] for i in picks]


def write_continuous_keyframes(
    input_path: Path,
    out_dir: Path,
    probe: ProbeInfo,
    scale: Scale,
    *,
    state_a_s: float,
    phase_s: list[float],
    boxes: list[tuple[str, Rect]],
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
    state_a_img: np.ndarray | None = None,
) -> list[KeyframeFile]:
    """Continuous-mode keyframes: ``state_a``, ``annotated_a`` (``boxes``), ``phase_<k>``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = keyframe_size(probe, params)
    k = size[0] / (probe.width / scale.pixel_ratio)
    files: list[KeyframeFile] = []
    save = _saver(out_dir, files)
    times: dict = {i: t for i, t in enumerate(pick_phase_times(phase_s), start=1)}
    if state_a_img is None:
        times["a"] = state_a_s
    grabbed = _grab_all(input_path, times, size, ffmpeg_bin)
    img_a = state_a_img if state_a_img is not None else grabbed["a"]
    save("state_a.png", img_a, "state_a", state_a_s)
    save("annotated_a.png", annotate(img_a, boxes, k), "annotated_a", state_a_s)
    for i in sorted(key for key in grabbed if isinstance(key, int)):
        save(f"phase_{i}.png", grabbed[i], "phase", times[i])
    return files


def write_keyframes(
    input_path: Path,
    out_dir: Path,
    probe: ProbeInfo,
    scale: Scale,
    *,
    state_a_s: float,
    state_b_s: float,
    mid_s: dict[int, float],
    elements: list[ElementCandidate],
    params: MeasureParams,
    ffmpeg_bin: str = "ffmpeg",
    state_a_img: np.ndarray | None = None,
) -> list[KeyframeFile]:
    """Write all A9 keyframes into ``out_dir`` and describe them (order = display order).

    ``state_a_img``: the frame at ``state_a_s`` already grabbed at :func:`keyframe_size` (the
    appearance stage needs it too), reused instead of seeking again.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    size = keyframe_size(probe, params)
    css_to_img = size[0] / (probe.width / scale.pixel_ratio)
    files: list[KeyframeFile] = []
    save = _saver(out_dir, files)

    # Independent ffmpeg seeks: run them concurrently (each is mostly process start + seek).
    times = {"a": state_a_s, "b": state_b_s} | {pct: mid_s[pct] for pct in (25, 50, 75)
                                                 if pct in mid_s}  # fmt: skip
    if state_a_img is not None:
        del times["a"]
    grabbed = _grab_all(input_path, times, size, ffmpeg_bin)
    if state_a_img is not None:
        grabbed["a"] = state_a_img
    img_a, img_b = grabbed["a"], grabbed["b"]
    save("state_a.png", img_a, "state_a", state_a_s)
    save("state_b.png", img_b, "state_b", state_b_s)
    for pct in (25, 50, 75):
        if pct in mid_s:
            save(f"mid_{pct}.png", grabbed[pct], f"mid_{pct}", mid_s[pct])
    save("annotated_a.png", annotate(img_a, [(e.id, e.bbox_a) for e in elements], css_to_img),
         "annotated_a", state_a_s)  # fmt: skip
    save("annotated_b.png", annotate(img_b, [(e.id, e.bbox_b) for e in elements], css_to_img),
         "annotated_b", state_b_s)  # fmt: skip
    for e in elements[: params.keyframes.max_element_crops]:
        crop = element_crop(img_a, img_b, e.bbox_a.union(e.bbox_b), css_to_img,
                            params.keyframes.max_long_side)  # fmt: skip
        save(f"el_{e.id}.png", crop, "element", None, e.id)
    return files
