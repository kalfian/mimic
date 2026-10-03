"""A0 probe + upload validation on tiny ffmpeg-generated clips (Track A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.core.errors import ErrorCode, PipelineError
from app.models.measure import ProbeInfo
from app.pipeline import probe as probe_mod
from app.pipeline.params import DEFAULT_PARAMS
from app.pipeline.probe import (
    probe,
    probe_to_json,
    probe_warnings,
    resolve_pixel_ratio,
    resolve_scale,
    sniff_container,
    timing_stats,
    validate_upload,
)
from tests.unit.media_clips import build_clip_set, requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_clip_set(tmp_path_factory.mktemp("clips"))


# ---- sniffing -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x00\x00\x00\x20ftypisom", "isobmff"),
        (b"\x00\x00\x00\x14ftypqt  ", "isobmff"),
        (b"\x00\x00\x00\x08wide\x00\x00", "isobmff"),
        (b"\x1a\x45\xdf\xa3\x01\x00", "webm"),
        (b"\x89PNG\r\n\x1a\n", None),
        (b"RIFF\x00\x00\x00\x00AVI ", None),
        (b"", None),
        (b"\x00\x00", None),
    ],
)
def test_sniff_container(head: bytes, expected: str | None) -> None:
    assert sniff_container(head) == expected


def test_sniff_real_files(clips: dict[str, Path]) -> None:
    for key, family in (("ok_mp4", "isobmff"), ("ok_mov", "isobmff"), ("ok_webm", "webm")):
        assert sniff_container(clips[key].read_bytes()[:64]) == family


# ---- full probe -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "container", "codec", "size", "fps"),
    [
        ("ok_mp4", "mp4", "h264", (320, 240), 30.0),
        ("ok_mov", "mov", "h264", (160, 120), 30.0),
        ("ok_webm", "webm", "vp9", (160, 120), 30.0),
    ],
)
def test_probe_formats(
    clips: dict[str, Path],
    settings: Settings,
    key: str,
    container: str,
    codec: str,
    size: tuple[int, int],
    fps: float,
) -> None:
    info = probe(clips[key], settings)
    assert (info.container, info.codec) == (container, codec)
    assert (info.width, info.height) == size
    assert info.rotation == 0
    assert info.fps_nominal == pytest.approx(fps, abs=0.1)
    assert info.fps_effective == pytest.approx(fps, rel=0.02)
    assert not info.is_vfr
    assert not info.has_audio
    assert not info.timestamps_estimated
    # display-order, strictly increasing, starting at the container start
    assert info.frame_ts[0] == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(info.frame_ts) > 0)
    assert info.frame_ts.size == pytest.approx(info.duration_s * fps, abs=1)


def test_probe_rotation_swaps_dimensions(clips: dict[str, Path], settings: Settings) -> None:
    info = probe(clips["rotated"], settings)
    assert abs(info.rotation) == 90
    assert (info.width, info.height) == (240, 320)


def test_probe_vfr(clips: dict[str, Path], settings: Settings) -> None:
    info = probe(clips["vfr"], settings)
    assert info.is_vfr
    dt = np.diff(info.frame_ts)
    assert dt.min() == pytest.approx(1 / 60, abs=2e-3)
    assert dt.max() == pytest.approx(4 / 60, abs=2e-3)
    assert "vfr_source" in {w.code for w in probe_warnings(info)}


def test_probe_low_fps_warning(clips: dict[str, Path], settings: Settings) -> None:
    info = probe(clips["low_fps"], settings)
    assert info.fps_effective == pytest.approx(10, rel=0.05)
    codes = {w.code for w in probe_warnings(info)}
    assert codes == {"low_fps_source"}


def test_probe_duration_from_packets_when_container_has_none(
    clips: dict[str, Path], settings: Settings
) -> None:
    fp = probe_mod.ffprobe_format(clips["no_duration_webm"], settings)
    assert fp.duration_s is None  # premise: the pipe-written WebM really lacks a duration
    validated = validate_upload(clips["no_duration_webm"], settings)
    assert validated.duration_s == pytest.approx(1.0, abs=0.05)


def test_probe_falls_back_to_uniform_timestamps(
    clips: dict[str, Path], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe_mod, "frame_timestamps", lambda *_: np.empty(0))
    info = probe(clips["ok_mp4"], settings)
    assert info.timestamps_estimated
    assert info.frame_ts.size == 60
    assert info.frame_ts[1] == pytest.approx(1 / 30)
    assert "timestamps_estimated" in {w.code for w in probe_warnings(info)}


def test_probe_to_json(clips: dict[str, Path], settings: Settings) -> None:
    data = probe_to_json(probe(clips["ok_mp4"], settings))
    assert data["frame_count"] == len(data["frame_ts"]) == 60
    assert data["median_dt_s"] == pytest.approx(1 / 30, abs=1e-4)


# ---- upload validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "code"),
    [
        ("too_short", ErrorCode.TOO_SHORT),
        ("too_long", ErrorCode.TOO_LONG),
        ("audio_only", ErrorCode.DECODE_FAILED),
        ("corrupt", ErrorCode.DECODE_FAILED),
        ("not_video", ErrorCode.UNSUPPORTED_FORMAT),
    ],
)
def test_validate_upload_errors(
    clips: dict[str, Path], settings: Settings, key: str, code: ErrorCode
) -> None:
    with pytest.raises(PipelineError) as exc:
        validate_upload(clips[key], settings)
    assert exc.value.code is code


def test_validate_upload_duration_tolerance(
    clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """max 1.5 s + 0.5 s tolerance accepts the 2 s clip; 1.4 s + 0.5 s rejects it."""
    from app.config import get_settings

    monkeypatch.setenv("MIMIC_MAX_DURATION_S", "1.5")
    get_settings.cache_clear()
    assert validate_upload(clips["ok_mp4"], get_settings()).duration_s == pytest.approx(2.0)
    monkeypatch.setenv("MIMIC_MAX_DURATION_S", "1.4")
    get_settings.cache_clear()
    with pytest.raises(PipelineError) as exc:
        validate_upload(clips["ok_mp4"], get_settings())
    assert exc.value.code is ErrorCode.TOO_LONG


def test_missing_ffprobe_is_internal_error(
    clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    from app.config import get_settings

    monkeypatch.setenv("MIMIC_FFPROBE_BIN", "/nonexistent/ffprobe")
    get_settings.cache_clear()
    with pytest.raises(PipelineError) as exc:
        validate_upload(clips["ok_mp4"], get_settings())
    assert exc.value.code is ErrorCode.INTERNAL_ERROR


# ---- pixel ratio / scale --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("w", "h", "option", "expected"),
    [
        (1280, 720, "auto", (1, "auto")),
        (2559, 1599, "auto", (1, "auto")),
        (2560, 1440, "auto", (2, "auto")),
        (1200, 1600, "auto", (2, "auto")),
        (2880, 1800, "1", (1, "user")),
        (1280, 720, "3", (3, "user")),
    ],
)
def test_resolve_pixel_ratio(w: int, h: int, option: str, expected: tuple[int, str]) -> None:
    assert resolve_pixel_ratio(w, h, option) == expected


def test_resolve_pixel_ratio_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_pixel_ratio(100, 100, "4")


def _info(width: int, height: int, fps: float = 60.0) -> ProbeInfo:
    return ProbeInfo(
        container="mp4",
        codec="h264",
        width=width,
        height=height,
        rotation=0,
        duration_s=2.0,
        frame_ts=np.arange(120) / fps,
        fps_nominal=fps,
        fps_effective=fps,
        is_vfr=False,
        has_audio=False,
    )


def test_resolve_scale_retina_auto() -> None:
    scale, warnings = resolve_scale(_info(2880, 1800), "auto")
    assert (scale.pixel_ratio, scale.pixel_ratio_source) == (2, "auto")
    assert scale.analysis_scale == pytest.approx(0.5)
    assert scale.to_css(16.0, image_scale=1.0) == pytest.approx(8.0)  # source px -> CSS px
    assert [w.code for w in warnings] == ["pixel_ratio_assumed"]
    assert warnings[0].severity == "info"
    assert "2x" in warnings[0].message and "2880x1800" in warnings[0].message


def test_resolve_scale_downscales_huge_css_canvas() -> None:
    # 5120x2880 @2x is 2560 CSS px wide > 1920 -> extra 0.75 factor
    scale, _ = resolve_scale(_info(5120, 2880), "2")
    assert scale.pixel_ratio_source == "user"
    assert scale.analysis_scale == pytest.approx(0.5 * 1920 / 2560)
    assert scale.k == pytest.approx(1920 / 2560)


def test_resolve_scale_user_has_no_warning() -> None:
    scale, warnings = resolve_scale(_info(1280, 720), "1")
    assert scale.analysis_scale == 1.0
    assert warnings == []


def test_timing_stats_cfr_and_vfr() -> None:
    cfr = np.arange(61) / 60
    assert timing_stats(cfr, DEFAULT_PARAMS.probe) == (pytest.approx(60.0), False)
    vfr = np.concatenate([np.arange(30) / 60, 0.5 + np.arange(1, 16) / 15])
    assert timing_stats(vfr, DEFAULT_PARAMS.probe)[1] is True
    assert timing_stats(np.array([0.0]), DEFAULT_PARAMS.probe) == (0.0, False)
