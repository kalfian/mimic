"""Encode variants on a tiny scene: ffprobe must report the requested codec/fps/duration/VFR."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from .animate import CursorKey, CursorPath, Timeline, Tween
from .encode import EncodeSpec, encode_frames, encoder_available, probe, tools_available, verify
from .scene import Node, Renderer, Scene
from .truth import TruthVideo

pytestmark = pytest.mark.skipif(not tools_available(), reason="ffmpeg/ffprobe not on PATH")

W, H = 160, 96
BLUE = (59, 130, 246)


def _scene() -> Scene:
    box = Node(id="b", x=40, y=24, w=60, h=40, fill=BLUE, radius=6)
    cur = CursorPath([CursorKey(0.0, 10, 80), CursorKey(0.2, 10, 80),
                      CursorKey(0.4, 70, 40, "ease-in-out")])  # fmt: skip
    tl = Timeline([Tween("b", "translateX", "fwd", 0.45, 0.2, 0.0, 12.0, "ease-out")])
    return Scene(W, H, (243, 244, 246), [], [box], tl, cur, duration_s=1.2)


VARIANTS = {
    "mp4": (60.0, EncodeSpec()),
    "mov": (60.0, EncodeSpec(container="mov")),
    "crf28": (60.0, EncodeSpec(crf=28)),
    "30fps": (30.0, EncodeSpec()),
    "vfr": (60.0, EncodeSpec(vfr=True)),
    "webm": (60.0, EncodeSpec(container="webm", codec="vp9", crf=32)),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_probe(variant: str, tmp_path) -> None:
    fps, spec = VARIANTS[variant]
    if spec.codec == "vp9" and not encoder_available("libvpx-vp9"):
        pytest.skip("no libvpx-vp9")
    r = Renderer(_scene())
    out = tmp_path / f"v.{spec.ext}"
    n = encode_frames((f for _, f in r.frames(fps)), r.W, r.H, fps, spec, out)
    assert n == int(1.2 * fps)
    video = TruthVideo(
        file=out.name, container=spec.container, codec=spec.codec, crf=spec.crf, fps=fps,
        vfr=spec.vfr, pixel_ratio=1, css_width=W, css_height=H, width=W, height=H,
        duration_ms=int(round(n / fps * 1000)), frames_rendered=n, probe=probe(out),
    )  # fmt: skip
    assert verify(video) == []
    assert video.probe is not None and video.probe.color_space == "bt709"


def test_color_round_trip(tmp_path) -> None:
    r = Renderer(_scene())
    out = tmp_path / "c.mp4"
    encode_frames((f for _, f in r.frames(60.0)), r.W, r.H, 60.0, EncodeSpec(), out)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"],  # fmt: skip
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
    got = frame[40:48, 60:80].reshape(-1, 3).mean(axis=0)
    assert np.abs(got - np.array(BLUE[::-1])).max() <= 4.0
