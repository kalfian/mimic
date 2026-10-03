"""A1 — H.264 MP4 preview transcode (PLAN P6, §6.2).

The UI plays ``preview.mp4`` instead of the original (MOV/HEVC/VP9 do not play everywhere).
A failed transcode is non-fatal: the pipeline adds warning ``preview_unavailable`` and sets
``artifacts.video_url`` to null.

The file is written to a temporary name and renamed into place, so ``GET /video`` never
serves a half-written MP4.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)

PREVIEW_TIMEOUT_S = 300.0


def preview_argv(settings: Settings, src: Path, dst: Path) -> list[str]:
    """ffmpeg command line (PLAN §6.2) with machine-readable progress on stdout."""
    return [
        settings.ffmpeg_bin, "-nostdin", "-y", "-v", "error",
        "-i", str(src),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        "-f", "mp4", str(dst),
    ]  # fmt: skip


def make_preview(
    src: Path,
    dst: Path,
    settings: Settings,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    timeout_s: float = PREVIEW_TIMEOUT_S,
) -> bool:
    """Transcode ``src`` to an H.264 ``dst``. Returns False (and logs why) on any failure.

    ``on_progress(fraction)`` is called from this thread with 0..1 when ``duration_s`` is known.
    Exceptions raised by ``on_progress`` (e.g. shutdown) kill ffmpeg and propagate.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".preview.", suffix=".mp4", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        ok = _transcode(src, tmp, settings, duration_s, on_progress, timeout_s)
        if ok and tmp.stat().st_size > 0:
            os.replace(tmp, dst)
            return True
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _transcode(
    src: Path,
    tmp: Path,
    settings: Settings,
    duration_s: float | None,
    on_progress: Callable[[float], None] | None,
    timeout_s: float,
) -> bool:
    try:
        proc = subprocess.Popen(
            preview_argv(settings, src, tmp),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        log.warning("preview: %s not found", settings.ffmpeg_bin)
        return False

    # Drain stderr concurrently so a chatty ffmpeg can never block on a full pipe.
    stderr_chunks: list[bytes] = []
    drain = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read() if proc.stderr else b""),
        daemon=True,
    )
    drain.start()
    timer = threading.Timer(timeout_s, proc.kill)
    timer.start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if on_progress is None or not duration_s:
                continue
            key, _, value = raw.decode("ascii", "replace").strip().partition("=")
            if key == "out_time_us" and value.lstrip("-").isdigit():
                on_progress(min(max(int(value) / 1e6 / duration_s, 0.0), 1.0))
        rc = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        timer.cancel()
        drain.join(timeout=5)

    if rc != 0:
        err = b"".join(stderr_chunks).decode("utf-8", "replace").strip()[-500:]
        log.warning("preview transcode failed (rc=%s): %s", rc, err or "<no stderr>")
        return False
    return True
