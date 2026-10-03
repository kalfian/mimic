"""ffmpeg encode variants + ffprobe facts for synthetic videos (PLAN §11.2).

Raw BGR frames are piped to ffmpeg. All outputs are yuv420p tagged BT.709 limited range
(what macOS screen recordings carry), so color round-trips through any decoder that honors
the tags (ffmpeg does).

Variants: H.264 in MP4 (default, CRF 18) or MOV, VP9 in WebM, any CRF, 30/60 fps (render
rate), and VFR (``mpdecimate`` + ``-fps_mode vfr``: duplicate frames are dropped and the
remaining frames keep their original timestamps, like a recorder that only emits on change).
VFR caps consecutive drops at :data:`VFR_MAX_DROP` so holds still get a frame every ~83 ms
(as screen recorders do) and the file keeps its duration instead of ending at the last change.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np

from .truth import TruthVideo, VideoProbe

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

VFR_MAX_DROP = 4

COLOR_ARGS = [
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-color_range", "tv",
]  # fmt: skip


@dataclass(frozen=True)
class EncodeSpec:
    container: Literal["mp4", "mov", "webm"] = "mp4"
    codec: Literal["h264", "vp9"] = "h264"
    crf: int = 18
    vfr: bool = False

    @property
    def ext(self) -> str:
        return self.container


def tools_available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


@functools.cache
def encoder_available(name: str) -> bool:
    """True if ``ffmpeg -encoders`` lists ``name`` (e.g. ``libvpx-vp9``)."""
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def ffmpeg_args(width: int, height: int, fps: float, spec: EncodeSpec, out: Path) -> list[str]:
    vf = []
    if spec.vfr:
        vf.append(f"mpdecimate=max={VFR_MAX_DROP}")
    vf.append("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    args = [
        FFMPEG, "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-",
        "-vf", ",".join(vf),
    ]  # fmt: skip
    if spec.vfr:
        args += ["-fps_mode", "vfr"]
    if spec.codec == "h264":
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(spec.crf)]
    elif spec.codec == "vp9":
        args += [
            "-c:v", "libvpx-vp9", "-crf", str(spec.crf), "-b:v", "0",
            "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
        ]  # fmt: skip
    else:  # pragma: no cover - Literal guards this
        raise ValueError(spec.codec)
    args += ["-pix_fmt", "yuv420p", *COLOR_ARGS]
    if spec.container == "mp4":
        args += ["-movflags", "+faststart", "-f", "mp4"]
    elif spec.container == "mov":
        args += ["-f", "mov"]
    else:
        args += ["-f", "webm"]
    args.append(str(out))
    return args


def encode_frames(
    frames: Iterable[np.ndarray], width: int, height: int, fps: float, spec: EncodeSpec, out: Path
) -> int:
    """Pipe BGR ``uint8`` frames of size ``width``×``height`` into ffmpeg. Returns frame count."""
    if spec.codec == "vp9" and not encoder_available("libvpx-vp9"):
        raise RuntimeError("ffmpeg has no libvpx-vp9 encoder")
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ffmpeg_args(width, height, fps, spec, out),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    n = 0
    try:
        for f in frames:
            if f.shape != (height, width, 3) or f.dtype != np.uint8:
                raise ValueError(f"frame {n}: expected {(height, width, 3)} uint8, got {f.shape}")
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
            n += 1
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {out.name}: {err.strip()[-2000:]}")
    return n


def _rate(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    return float(Fraction(s))


def frame_times(path: Path) -> np.ndarray:
    """Display timestamps (s) of every video frame, via ffprobe."""
    out = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    vals = []
    for line in out.split():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            vals.append(float(line))
    return np.array(sorted(vals))


def probe(path: Path) -> VideoProbe:
    """ffprobe summary of an encoded synthetic video."""
    raw = subprocess.run(
        [FFPROBE, "-v", "error", "-of", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(raw)
    vs = next(s for s in info["streams"] if s.get("codec_type") == "video")
    ts = frame_times(path)
    dt = np.diff(ts)
    if dt.size >= 2 and float(np.median(dt)) > 0:
        cv = float(np.std(dt) / np.median(dt))
    else:
        cv = 0.0
    dur = float(info["format"].get("duration") or vs.get("duration") or 0.0)
    return VideoProbe(
        format_name=info["format"]["format_name"],
        codec_name=vs["codec_name"],
        width=int(vs["width"]),
        height=int(vs["height"]),
        pix_fmt=vs.get("pix_fmt", ""),
        color_space=vs.get("color_space"),
        avg_fps=round(_rate(vs.get("avg_frame_rate")), 4),
        r_fps=round(_rate(vs.get("r_frame_rate")), 4),
        duration_ms=int(round(dur * 1000)),
        frame_count=int(ts.size),
        is_vfr=cv > 0.15,
        dt_cv=round(cv, 4),
    )


def verify(video: TruthVideo) -> list[str]:
    """Problems found comparing ``video.probe`` against what was requested (empty = ok).

    CFR: exact frame count, fps within 0.01, duration within one frame (+ container slack).
    VFR: frames were actually dropped and timestamps are irregular (``dt_cv > 0.15``).
    """
    p = video.probe
    if p is None:
        return ["no probe info"]
    problems: list[str] = []
    want_codec = {"h264": "h264", "vp9": "vp9"}[video.codec]
    if p.codec_name != want_codec:
        problems.append(f"codec {p.codec_name} != {want_codec}")
    want_fmt = {"mp4": "mp4", "mov": "mov", "webm": "webm"}[video.container]
    if want_fmt not in p.format_name.split(","):
        problems.append(f"format {p.format_name} lacks {want_fmt}")
    if (p.width, p.height) != (video.width, video.height):
        problems.append(f"size {p.width}x{p.height} != {video.width}x{video.height}")
    if p.pix_fmt != "yuv420p":
        problems.append(f"pix_fmt {p.pix_fmt}")
    frame_ms = 1000.0 / video.fps
    if video.vfr:
        if not p.is_vfr:
            problems.append(f"expected VFR, dt_cv={p.dt_cv}")
        if p.frame_count >= video.frames_rendered:
            problems.append("VFR encode dropped no frames")
        if abs(p.duration_ms - video.duration_ms) > (VFR_MAX_DROP + 1) * frame_ms + 2:
            problems.append(f"duration {p.duration_ms}ms != {video.duration_ms}ms")
    else:
        if p.is_vfr:
            problems.append(f"unexpected VFR (dt_cv={p.dt_cv})")
        if p.frame_count != video.frames_rendered:
            problems.append(f"frames {p.frame_count} != {video.frames_rendered}")
        if abs(p.r_fps - video.fps) > 0.01:
            problems.append(f"r_fps {p.r_fps} != {video.fps}")
        if abs(p.duration_ms - video.duration_ms) > frame_ms + 2:
            problems.append(f"duration {p.duration_ms}ms != {video.duration_ms}ms")
    return problems
