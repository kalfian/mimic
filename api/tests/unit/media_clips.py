"""Tiny ffmpeg-generated test clips for Track A tests (probe, upload validation, lifecycle).

Clips are generated at test time from ``lavfi`` sources (no binary fixtures in git) and kept
small (≤ 320x240, short, low fps) so a whole set renders in about a second.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not on PATH"
)


def ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", *args], check=True)


def testsrc(
    dst: Path,
    *,
    duration: float = 1.0,
    size: str = "160x120",
    rate: int = 30,
    codec_args: tuple[str, ...] = ("-c:v", "libx264", "-pix_fmt", "yuv420p"),
    pre_output: tuple[str, ...] = (),
) -> Path:
    ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=d={duration}:s={size}:r={rate}",
        *pre_output, *codec_args, str(dst),
    )  # fmt: skip
    return dst


def ui_clip(dst: Path) -> Path:
    """A minimal UI interaction the real pipeline can analyse: a textured 96x64 "card" on a
    flat page moves up 16 px (linear, 300 ms) at t = 1.0 s; 320x240, 60 fps, 2.5 s."""
    card = (
        "color=c=0x3B82F6:s=96x64:r=60:d=2.5,"
        "drawbox=x=12:y=24:w=50:h=10:color=white:t=fill,"
        "drawbox=x=70:y=20:w=14:h=14:color=0x1D4ED8:t=fill"
    )
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=0xE8EAED:s=320x240:r=60:d=2.5",
        "-f", "lavfi", "-i", card,
        "-filter_complex",
        r"[0][1]overlay=x=100:y='70-16*min(max((t-1)/0.3\,0)\,1)':shortest=1,"
        "drawbox=x=20:y=20:w=60:h=12:color=0x9AA0A6:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst),
    )  # fmt: skip
    return dst


def build_clip_set(root: Path) -> dict[str, Path]:
    """Every clip the Track A tests need, keyed by purpose."""
    root.mkdir(parents=True, exist_ok=True)
    clips: dict[str, Path] = {}
    clips["ok_mp4"] = testsrc(root / "ok.mp4", duration=2.0, size="320x240", rate=30)
    clips["ui_mp4"] = ui_clip(root / "ui.mp4")
    clips["ok_mov"] = testsrc(
        root / "ok.mov", codec_args=("-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mov")
    )
    clips["ok_webm"] = testsrc(root / "ok.webm", codec_args=("-c:v", "libvpx-vp9", "-b:v", "200k"))
    clips["too_short"] = testsrc(root / "short.mp4", duration=0.2)
    clips["too_long"] = testsrc(root / "long.mp4", duration=20.0, size="64x48", rate=5)
    clips["low_fps"] = testsrc(root / "lowfps.mp4", duration=1.0, rate=10)
    clips["vfr"] = testsrc(
        root / "vfr.mp4",
        duration=2.0,
        rate=60,
        pre_output=("-vf", r"select='lt(n\,40)+not(mod(n\,4))'", "-fps_mode", "vfr"),
    )
    rotated = root / "rotated.mp4"
    ffmpeg("-display_rotation", "90", "-i", str(clips["ok_mp4"]), "-c", "copy", str(rotated))
    clips["rotated"] = rotated

    # WebM written to a non-seekable pipe carries no duration in its header.
    no_dur = root / "noduration.webm"
    assert FFMPEG is not None
    with no_dur.open("wb") as out:
        subprocess.run(
            [
                FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "testsrc=d=1:s=160x120:r=30", "-c:v", "libvpx-vp9", "-b:v", "200k",
                "-f", "webm", "pipe:1",
            ],
            stdout=out,
            check=True,
        )  # fmt: skip
    clips["no_duration_webm"] = no_dur

    audio = root / "audio_only.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=d=1", "-c:a", "aac", str(audio))
    clips["audio_only"] = audio

    # Sniffs as ISO BMFF (ftyp box) but is otherwise garbage.
    corrupt = root / "corrupt.mp4"
    corrupt.write_bytes(
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + bytes(range(256)) * 40
    )
    clips["corrupt"] = corrupt

    not_video = root / "image.mp4"
    not_video.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048)
    clips["not_video"] = not_video
    return clips
