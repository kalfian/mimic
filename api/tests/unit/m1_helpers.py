"""Tiny scene renderer + encoder for Track M1 unit tests (independent of tests/synth).

Frames are rendered in numpy with sub-pixel ``warpAffine`` so geometry is known exactly.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.models.measure import ProbeInfo, Scale

BG = (238, 238, 238)


def ease_out(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_in_out(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 3 * u * u - 2 * u**3


def progress(t: float, t0: float, dur: float, fn: Callable[[float], float] = ease_out) -> float:
    if t <= t0:
        return 0.0
    if t >= t0 + dur:
        return 1.0
    return fn((t - t0) / dur)


def textured_patch(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Feature-rich BGR patch (gradient + shapes) so ORB/ECC have something to lock on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (80 + 120 * xx / max(w - 1, 1)).astype(np.uint8)
    img[..., 1] = (60 + 140 * yy / max(h - 1, 1)).astype(np.uint8)
    img[..., 2] = 150
    for _ in range(max(6, w * h // 900)):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, max(4, min(w, h) // 6)))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r // 2 + 2), c, -1)
    return img


def card_sprite(w: int, h: int, *, seed: int = 0, title_color=(40, 40, 40)) -> np.ndarray:
    """BGRA card: white body, 1px border, textured image area, a title text line."""
    spr = np.zeros((h, w, 4), np.uint8)
    spr[..., :3] = 255
    spr[..., 3] = 255
    cv2.rectangle(spr, (0, 0), (w - 1, h - 1), (200, 200, 200, 255), 1)
    ih = int(h * 0.55)
    spr[8 : 8 + ih, 8 : w - 8, :3] = textured_patch(w - 16, ih, seed)
    cv2.putText(
        spr, "Card title", (12, 8 + ih + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (*title_color, 255), 1, cv2.LINE_AA,
    )  # fmt: skip
    return spr


def arrow_sprite() -> np.ndarray:
    """12×19 BGRA arrow pointer, tip at (1, 1)."""
    spr = np.zeros((21, 14, 4), np.uint8)
    pts = np.array([[1, 1], [1, 17], [5, 13], [8, 19], [10, 18], [7, 12], [12, 12]], np.int32)
    cv2.fillPoly(spr, [pts], (255, 255, 255, 255))
    cv2.polylines(spr, [pts], True, (0, 0, 0, 255), 1)
    return spr


def composite(canvas: np.ndarray, sprite: np.ndarray, m: np.ndarray, opacity: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite into ``canvas`` (in place) with a 2×3 affine ``m``."""
    h, w = canvas.shape[:2]
    warped = cv2.warpAffine(
        sprite, m.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )  # fmt: skip
    a = warped[..., 3:4].astype(np.float32) / 255.0 * opacity
    canvas[:] = (
        (canvas.astype(np.float32) * (1 - a) + warped[..., :3].astype(np.float32) * a)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )


def translate(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y]], np.float64)


def scale_about(s: float, cx: float, cy: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    return np.array([[s, 0, cx - s * cx + tx], [0, s, cy - s * cy + ty]], np.float64)


def base_canvas(w: int, h: int) -> np.ndarray:
    c = np.zeros((h, w, 3), np.uint8)
    c[:] = BG
    cv2.rectangle(c, (10, 10), (w - 10, 30), (220, 220, 220), -1)  # static "chrome"
    cv2.putText(c, "page", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return c


@dataclass
class HoverScene:
    """Card hover: cursor enters, card translates up, holds, cursor leaves, card returns."""

    w: int = 480
    h: int = 320
    card_xy: tuple[int, int] = (140, 70)
    card_wh: tuple[int, int] = (180, 200)
    dy: float = -8.0
    fwd_t0: float = 1.20
    fwd_dur: float = 0.30
    rev_t0: float = 2.30
    rev_dur: float = 0.22
    cursor_in: tuple[float, float] = (0.40, 1.15)  # time window the pointer moves in
    cursor_out: tuple[float, float] = (2.00, 2.25)
    with_cursor: bool = True
    _card: np.ndarray = field(init=False, repr=False)
    _arrow: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._card = card_sprite(*self.card_wh)
        self._arrow = arrow_sprite()

    def card_dy(self, t: float) -> float:
        p_f = progress(t, self.fwd_t0, self.fwd_dur, ease_out)
        p_r = progress(t, self.rev_t0, self.rev_dur, ease_in_out)
        return self.dy * p_f * (1 - p_r)

    def cursor_xy(self, t: float) -> tuple[float, float]:
        a = (30.0, 280.0)
        b = (self.card_xy[0] + 90.0, self.card_xy[1] + 100.0)
        c = (430.0, 290.0)
        if t <= self.cursor_in[1]:
            u = progress(t, self.cursor_in[0], self.cursor_in[1] - self.cursor_in[0], ease_in_out)
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        u = progress(t, self.cursor_out[0], self.cursor_out[1] - self.cursor_out[0], ease_in_out)
        return (b[0] + (c[0] - b[0]) * u, b[1] + (c[1] - b[1]) * u)

    def render(self, t: float) -> np.ndarray:
        img = base_canvas(self.w, self.h)
        composite(img, self._card, translate(self.card_xy[0], self.card_xy[1] + self.card_dy(t)))
        if self.with_cursor:
            x, y = self.cursor_xy(t)
            composite(img, self._arrow, translate(round(x) - 1, round(y) - 1))
        return img


def encode(frames: list[np.ndarray], path: Path, fps: float = 30.0, crf: int = 18) -> Path:
    h, w = frames[0].shape[:2]
    argv = [
        shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path),
    ]  # fmt: skip
    proc = subprocess.run(argv, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return path


def make_probe(n: int, fps: float, w: int, h: int) -> ProbeInfo:
    ts = np.arange(n, dtype=np.float64) / fps
    return ProbeInfo(
        container="mp4", codec="h264", width=w, height=h, rotation=0,
        duration_s=n / fps, frame_ts=ts, fps_nominal=fps, fps_effective=fps,
        is_vfr=False, has_audio=False,
    )  # fmt: skip


def scale1() -> Scale:
    return Scale(pixel_ratio=1, pixel_ratio_source="user", analysis_scale=1.0)


def render_video(scene: HoverScene, path: Path, duration: float, fps: float = 30.0):
    n = int(round(duration * fps))
    frames = [scene.render(i / fps) for i in range(n)]
    encode(frames, path, fps)
    return make_probe(n, fps, scene.w, scene.h), frames


requires_ffmpeg = __import__("pytest").mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def window_frames(frames: list[np.ndarray], fps: float, segment_id: str = "fwd", t0: float = 0.0):
    """In-memory WindowFrames (no codec) over full frames (crop = whole frame, CSS == px)."""
    from app.models.measure import Rect, WindowFrames

    h, w = frames[0].shape[:2]
    return WindowFrames(
        segment_id=segment_id,
        times=t0 + np.arange(len(frames)) / fps,
        crops=[f.copy() for f in frames],
        crop_css=Rect(0, 0, w, h),
        scale=scale1(),
        is_dup=np.zeros(len(frames), dtype=bool),
        cursor_masks=[None] * len(frames),
    )


def solid_rect_sprite(w: int, h: int, color, radius: int = 0) -> np.ndarray:
    """BGRA rounded rectangle (anti-aliased corners via 4× supersampling)."""
    ss = 4
    big = np.zeros((h * ss, w * ss, 4), np.uint8)
    r = radius * ss
    col = (*color, 255)
    if r > 0:
        cv2.rectangle(big, (r, 0), (w * ss - 1 - r, h * ss - 1), col, -1)
        cv2.rectangle(big, (0, r), (w * ss - 1, h * ss - 1 - r), col, -1)
        for cx, cy in (
            (r, r),
            (w * ss - 1 - r, r),
            (r, h * ss - 1 - r),
            (w * ss - 1 - r, h * ss - 1 - r),
        ):
            cv2.circle(big, (cx, cy), r, col, -1)
    else:
        big[:] = col
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
