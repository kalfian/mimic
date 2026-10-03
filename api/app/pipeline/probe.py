"""A0 — ffprobe metadata, upload validation, frame timestamps, VFR, rotation, pixel ratio (§6.1).

Two entry points:

* :func:`validate_upload` — fast, synchronous checks run by ``POST /api/jobs`` before the 202:
  container sniff, video stream present, duration bounds, first frame decodable.
* :func:`probe` — full A0 for the pipeline: everything above plus per-frame timestamps,
  ``fps_effective`` and ``is_vfr`` (decodes the whole stream once via ffprobe).

Time base: ``ProbeInfo.frame_ts`` is relative to the container ``start_time``, i.e. the same
origin as ``ffmpeg -ss <t> -i input`` seeks and as the H.264 preview's playback clock.
Rotation: ``width``/``height`` are post-rotation, matching what ffmpeg's auto-rotating decoder
emits.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.config import Settings
from app.core.errors import ErrorCode, PipelineError
from app.models.ir import PixelRatio, SpecWarning
from app.models.measure import ProbeInfo, Scale
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams, ProbeParams

log = logging.getLogger(__name__)

ContainerFamily = Literal["isobmff", "webm"]

#: Bytes needed by :func:`sniff_container`.
SNIFF_BYTES = 64

FFPROBE_TIMEOUT_S = 30.0
FRAME_TS_TIMEOUT_S = 180.0
FIRST_FRAME_TIMEOUT_S = 30.0

# Top-level ISO BMFF / QuickTime atoms that may start a file.
_ISOBMFF_BOXES = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"})
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"


# --------------------------------------------------------------------------------------------
# Container sniffing
# --------------------------------------------------------------------------------------------


def sniff_container(head: bytes) -> ContainerFamily | None:
    """Container family from the first bytes of a file, or None if not a supported container.

    MP4/MOV/M4V are ISO BMFF (box type at offset 4); WebM is Matroska/EBML.
    """
    if head[:4] == _EBML_MAGIC:
        return "webm"
    if len(head) >= 8 and head[4:8] in _ISOBMFF_BOXES:
        return "isobmff"
    return None


# --------------------------------------------------------------------------------------------
# ffprobe: format + streams
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormatProbe:
    """``ffprobe -show_format -show_streams`` summary of the first video stream."""

    container: str  # mp4 | mov | m4v | webm | <ffprobe format name>
    codec: str
    width: int  # post-rotation
    height: int
    rotation: int  # degrees, as reported
    duration_s: float | None  # None if the container does not say (e.g. MediaRecorder WebM)
    fps_nominal: float  # 0.0 if unknown
    has_audio: bool
    start_time_s: float


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise PipelineError(
            ErrorCode.INTERNAL_ERROR, f"{argv[0]} is not installed or not on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(
            ErrorCode.DECODE_FAILED, "Reading the video took too long; it may be corrupt."
        ) from exc


def _tail(stderr: bytes, limit: int = 500) -> str:
    return stderr.decode("utf-8", "replace").strip()[-limit:]


def _parse_rate(rate: str | None) -> float:
    """ffprobe ``"30000/1001"`` -> 29.97; ``"0/0"``/missing -> 0.0."""
    if not rate:
        return 0.0
    try:
        num, _, den = rate.partition("/")
        n, d = float(num), float(den or 1)
        return n / d if d and n > 0 else 0.0
    except ValueError:
        return 0.0


def _float_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _rotation(stream: dict[str, Any]) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    rot = (stream.get("tags") or {}).get("rotate")
    try:
        return int(rot) if rot is not None else 0
    except ValueError:
        return 0


def _container_name(fmt: dict[str, Any], ext_hint: str | None) -> str:
    names = set(str(fmt.get("format_name", "")).split(","))
    if "webm" in names or "matroska" in names:
        return "webm" if ext_hint in (None, "webm") else "matroska"
    if "mov" in names or "mp4" in names:
        brand = str((fmt.get("tags") or {}).get("major_brand", "")).strip().lower()
        if brand == "qt":
            return "mov"
        if brand == "m4v":
            return "m4v"
        if ext_hint in ("mov", "m4v", "mp4"):
            return ext_hint
        return "mp4"
    return str(fmt.get("format_name") or "unknown")


def ffprobe_format(path: Path, settings: Settings) -> FormatProbe:
    """Container/stream facts. Raises ``decode_failed`` if unreadable or without video."""
    proc = _run(
        [
            settings.ffprobe_bin, "-v", "error", "-of", "json",
            "-show_format", "-show_streams", str(path),
        ],
        FFPROBE_TIMEOUT_S,
    )  # fmt: skip
    if proc.returncode != 0:
        log.info("ffprobe failed for %s: %s", path.name, _tail(proc.stderr))
        raise PipelineError(
            ErrorCode.DECODE_FAILED,
            "The file could not be read as a video. Re-export it as an H.264 MP4 and upload it "
            "again.",
        )
    try:
        data = json.loads(proc.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise PipelineError(ErrorCode.DECODE_FAILED) from exc

    streams: list[dict[str, Any]] = data.get("streams") or []
    fmt: dict[str, Any] = data.get("format") or {}
    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )  # fmt: skip
    if video is None:
        raise PipelineError(ErrorCode.DECODE_FAILED, "The file has no video stream.")

    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise PipelineError(ErrorCode.DECODE_FAILED, "The video stream has no frame size.")
    rotation = _rotation(video)
    if abs(rotation) % 180 == 90:
        width, height = height, width

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    duration = _float_or_none(fmt.get("duration")) or _float_or_none(video.get("duration"))
    ext_hint = path.suffix.lower().lstrip(".") or None
    return FormatProbe(
        container=_container_name(fmt, ext_hint),
        codec=str(video.get("codec_name") or "unknown"),
        width=width,
        height=height,
        rotation=rotation,
        duration_s=duration if duration and duration > 0 else None,
        fps_nominal=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        start_time_s=_float_or_none(fmt.get("start_time")) or 0.0,
    )


def _packet_duration(path: Path, settings: Settings, fps: float) -> float | None:
    """Duration from the last video packet pts (no decode). For containers without a duration."""
    proc = _run(
        [
            settings.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
        ],
        FFPROBE_TIMEOUT_S,
    )  # fmt: skip
    ts = _parse_times(proc.stdout)
    if proc.returncode != 0 or ts.size == 0:
        return None
    dt = float(np.median(np.diff(np.sort(ts)))) if ts.size > 1 else (1.0 / fps if fps else 0.0)
    return float(ts.max() - ts.min() + dt)


def _check_first_frame(path: Path, settings: Settings) -> None:
    """Decode exactly one (tiny, gray) frame; ``decode_failed`` if none comes out."""
    proc = _run(
        [
            settings.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=32:-2",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        FIRST_FRAME_TIMEOUT_S,
    )  # fmt: skip
    if proc.returncode != 0 or not proc.stdout:
        log.info("first-frame decode failed for %s: %s", path.name, _tail(proc.stderr))
        raise PipelineError(ErrorCode.DECODE_FAILED, "The first video frame could not be decoded.")


def _check_duration(duration_s: float, settings: Settings) -> None:
    if duration_s < settings.min_duration_s:
        raise PipelineError(
            ErrorCode.TOO_SHORT,
            f"The recording is {duration_s:.2f} s long; it must be at least "
            f"{settings.min_duration_s:g} s. Wait about 1 second before interacting and keep "
            "recording until the animation has finished.",
        )
    if duration_s > settings.max_duration_with_tolerance_s:
        raise PipelineError(
            ErrorCode.TOO_LONG,
            f"The recording is {duration_s:.1f} s long. Trim it to "
            f"{settings.max_duration_s:g} seconds or less around the interaction.",
        )


def validate_upload(path: Path, settings: Settings) -> FormatProbe:
    """Synchronous upload validation (PLAN §5 ``POST /api/jobs``).

    Raises :class:`PipelineError` with ``unsupported_format`` / ``decode_failed`` /
    ``too_short`` / ``too_long``. Returns the format probe with a guaranteed ``duration_s``.
    """
    with path.open("rb") as f:
        if sniff_container(f.read(SNIFF_BYTES)) is None:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT)
    fp = ffprobe_format(path, settings)
    duration = fp.duration_s
    if duration is None:
        duration = _packet_duration(path, settings, fp.fps_nominal)
        if duration is None:
            raise PipelineError(ErrorCode.DECODE_FAILED, "The video duration could not be read.")
        fp = _replace_duration(fp, duration)
    _check_duration(duration, settings)
    _check_first_frame(path, settings)
    return fp


def _replace_duration(fp: FormatProbe, duration: float) -> FormatProbe:
    return FormatProbe(
        container=fp.container,
        codec=fp.codec,
        width=fp.width,
        height=fp.height,
        rotation=fp.rotation,
        duration_s=duration,
        fps_nominal=fp.fps_nominal,
        has_audio=fp.has_audio,
        start_time_s=fp.start_time_s,
    )


# --------------------------------------------------------------------------------------------
# Frame timestamps
# --------------------------------------------------------------------------------------------


def _parse_times(raw: bytes) -> np.ndarray:
    """Floats from ffprobe csv lines (first field); drops ``N/A`` and junk."""
    out: list[float] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        field = line.split(",", 1)[0].strip()
        v = _float_or_none(field)
        if v is not None:
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def frame_timestamps(path: Path, settings: Settings) -> np.ndarray:
    """Display-order ``best_effort_timestamp_time`` per decoded frame (absolute, unnormalized).

    Returns an empty array if ffprobe fails; callers fall back to uniform timestamps.
    """
    try:
        proc = _run(
            [
                settings.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
            ],
            FRAME_TS_TIMEOUT_S,
        )  # fmt: skip
    except PipelineError:
        return np.empty(0, dtype=np.float64)
    if proc.returncode != 0:
        log.info("frame timestamp probe failed for %s: %s", path.name, _tail(proc.stderr))
        return np.empty(0, dtype=np.float64)
    return _parse_times(proc.stdout)


def _uniform_timestamps(duration_s: float, fps: float) -> np.ndarray:
    n = max(int(round(duration_s * fps)), 1)
    return np.arange(n, dtype=np.float64) / fps


def timing_stats(frame_ts: np.ndarray, params: ProbeParams) -> tuple[float, bool]:
    """``(fps_effective, is_vfr)`` from frame timestamps (PLAN §6.1 step 3)."""
    if frame_ts.size < 2:
        return 0.0, False
    span = float(frame_ts[-1] - frame_ts[0])
    fps_eff = (frame_ts.size - 1) / span if span > 0 else 0.0
    dt = np.diff(frame_ts)
    med = float(np.median(dt))
    is_vfr = bool(med > 0 and float(np.std(dt)) / med > params.vfr_cv_threshold)
    return fps_eff, is_vfr


# --------------------------------------------------------------------------------------------
# Full probe
# --------------------------------------------------------------------------------------------


def probe(path: Path, settings: Settings, params: MeasureParams = DEFAULT_PARAMS) -> ProbeInfo:
    """Full A0 probe. Raises :class:`PipelineError` like :func:`validate_upload`."""
    fp = validate_upload(path, settings)
    assert fp.duration_s is not None

    raw = frame_timestamps(path, settings)
    estimated = False
    ts = raw - (fp.start_time_s if raw.size else 0.0)
    if raw.size < 2 or not bool(np.all(np.diff(ts) > 0)):
        fps = fp.fps_nominal or 30.0
        log.info(
            "%s: frame timestamps unusable (%d); using uniform %.3f fps", path.name, raw.size, fps
        )
        ts = _uniform_timestamps(fp.duration_s, fps)
        estimated = True
    ts = np.maximum(ts, 0.0) if ts.size and ts[0] > -1e-3 else ts

    fps_eff, is_vfr = timing_stats(ts, params.probe)
    fps_nominal = fp.fps_nominal or fps_eff or 30.0
    return ProbeInfo(
        container=fp.container,
        codec=fp.codec,
        width=fp.width,
        height=fp.height,
        rotation=fp.rotation,
        duration_s=float(fp.duration_s),
        frame_ts=ts,
        fps_nominal=float(fps_nominal),
        fps_effective=float(fps_eff or fps_nominal),
        is_vfr=is_vfr,
        has_audio=fp.has_audio,
        timestamps_estimated=estimated,
    )


def probe_to_json(info: ProbeInfo) -> dict[str, Any]:
    """``probe.json`` payload (debug/inspection; not read back by the pipeline)."""
    dt = np.diff(info.frame_ts) if info.frame_ts.size > 1 else np.empty(0)
    return {
        "container": info.container,
        "codec": info.codec,
        "width": info.width,
        "height": info.height,
        "rotation": info.rotation,
        "duration_s": round(info.duration_s, 6),
        "fps_nominal": round(info.fps_nominal, 6),
        "fps_effective": round(info.fps_effective, 6),
        "is_vfr": info.is_vfr,
        "has_audio": info.has_audio,
        "timestamps_estimated": info.timestamps_estimated,
        "frame_count": int(info.frame_ts.size),
        "median_dt_s": round(float(np.median(dt)), 6) if dt.size else None,
        "frame_ts": [round(float(t), 6) for t in info.frame_ts],
    }


# --------------------------------------------------------------------------------------------
# Pixel ratio / analysis scale / probe warnings
# --------------------------------------------------------------------------------------------


def resolve_pixel_ratio(
    width: int, height: int, option: str, params: ProbeParams = DEFAULT_PARAMS.probe
) -> tuple[PixelRatio, Literal["user", "auto"]]:
    """PLAN P4: user value if given, else 2 for width >= 2560 or height >= 1600, else 1."""
    if option in ("1", "2", "3"):
        return int(option), "user"  # type: ignore[return-value]
    if option != "auto":
        raise ValueError(f"invalid pixel_ratio option {option!r}")
    retina = width >= params.auto_retina_min_width or height >= params.auto_retina_min_height
    return (2 if retina else 1), "auto"


def resolve_scale(
    info: ProbeInfo, option: str, params: MeasureParams = DEFAULT_PARAMS
) -> tuple[Scale, list[SpecWarning]]:
    """Pixel ratio + analysis scale (§6.1 steps 4–5) and the ``pixel_ratio_assumed`` warning.

    ``analysis_scale`` = analysis px per source px: ``1 / pixel_ratio`` (analysis at CSS
    resolution), reduced further if the CSS long side still exceeds ``max_analysis_long_side``.
    """
    ratio, source = resolve_pixel_ratio(info.width, info.height, option, params.probe)
    analysis_scale = 1.0 / ratio
    css_long = max(info.width, info.height) / ratio
    if css_long > params.probe.max_analysis_long_side:
        analysis_scale *= params.probe.max_analysis_long_side / css_long
    warnings: list[SpecWarning] = []
    if source == "auto":
        other = "1x" if ratio != 1 else "2x"
        hint = "was not a Retina recording" if ratio != 1 else "was a Retina recording"
        warnings.append(
            SpecWarning(
                code="pixel_ratio_assumed",
                severity="info",
                message=(
                    f"Display scale assumed {ratio}x (recording is {info.width}x{info.height}). "
                    f"Choose {other} on upload if this {hint}."
                ),
            )
        )
    return Scale(
        pixel_ratio=ratio, pixel_ratio_source=source, analysis_scale=analysis_scale
    ), warnings


def probe_warnings(info: ProbeInfo, params: MeasureParams = DEFAULT_PARAMS) -> list[SpecWarning]:
    """Warnings that follow from the probe alone (closed IR warning codes)."""
    out: list[SpecWarning] = []
    if info.timestamps_estimated:
        out.append(
            SpecWarning(
                code="timestamps_estimated",
                severity="warn",
                message="Frame timestamps could not be read; timing assumes a constant frame rate.",
            )
        )
    if info.is_vfr:
        out.append(
            SpecWarning(
                code="vfr_source",
                severity="info",
                message="The recording has a variable frame rate; timing precision is lower.",
            )
        )
    if info.fps_effective < params.probe.low_fps_threshold:
        out.append(
            SpecWarning(
                code="low_fps_source",
                severity="warn",
                message=(
                    f"The recording has about {info.fps_effective:.0f} fps; short animations and "
                    "easing curves cannot be measured precisely. Record at 60 fps if possible."
                ),
            )
        )
    return out
