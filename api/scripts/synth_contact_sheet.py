"""Visual check for synthetic videos: one contact-sheet PNG per scenario (Track S DoD).

Frames are decoded **from the encoded file** (not re-rendered), so the sheet shows what the
pipeline will see. Layout per scenario:

* Row 0: full frames at state A and state B (or evenly spaced frames for negatives).
* One row per segment (fwd / rev / rt_in / rt_out): ROI crops at onset−150 ms, onset, 25 %,
  50 %, 75 %, 100 % of the segment and settle+150 ms. Each tile shows the truth progress of the
  segment's first transition. Overlays: **green** = truth ``bbox_initial``, **magenta** = truth
  ``bbox_active`` (thin lines), so an element should sit on green before and on magenta after.

Usage (from ``api/``)::

    uv run python scripts/synth_contact_sheet.py               # every *.truth.json in data/synth
    uv run python scripts/synth_contact_sheet.py card_hover_compound --dir /tmp/synth

Output: ``<dir>/contact/<name>.png``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.config import REPO_ROOT  # noqa: E402
from tests.synth.animate import CubicBezier  # noqa: E402
from tests.synth.truth import Truth  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "data" / "synth"
TILE_W = 300
FULL_W = 640
GREEN = (60, 200, 60)
MAGENTA = (200, 60, 200)
INK = (30, 30, 30)


def grab(video: Path, t_s: float, w: int, h: int) -> np.ndarray:
    """Decode the frame displayed at ``t_s`` (accurate seek) as BGR at device resolution."""
    t_s = max(t_s, 0.0)
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t_s:.4f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],  # fmt: skip
        capture_output=True,
        check=True,
    ).stdout
    if len(out) < w * h * 3:  # seeking past the last frame: take the last one
        out = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-sseof",
                "-0.2",
                "-i",
                str(video),
                "-update",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],  # fmt: skip
            capture_output=True,
            check=True,
        ).stdout[-w * h * 3 :]
    return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def label(img: np.ndarray, txt: str) -> np.ndarray:
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def roi_of(truth: Truth) -> tuple[float, float, float, float]:
    boxes = [b for e in truth.elements for b in (e.bbox_initial, e.bbox_active) if b.w > 0]
    if truth.interaction and truth.interaction.trigger_box:
        boxes.append(truth.interaction.trigger_box)
    if not boxes:
        return (0.0, 0.0, float(truth.video.css_width), float(truth.video.css_height))
    x0 = min(b.x for b in boxes) - 48
    y0 = min(b.y for b in boxes) - 48
    x1 = max(b.x + b.w for b in boxes) + 48
    y1 = max(b.y + b.h for b in boxes) + 48
    x0, y0 = max(x0, 0.0), max(y0, 0.0)
    x1 = min(x1, float(truth.video.css_width))
    y1 = min(y1, float(truth.video.css_height))
    return (x0, y0, x1, y1)


def crop_tile(frame: np.ndarray, truth: Truth, roi) -> np.ndarray:
    r = truth.video.pixel_ratio
    x0, y0, x1, y1 = (int(round(v * r)) for v in roi)
    crop = frame[y0:y1, x0:x1].copy()
    k = TILE_W / crop.shape[1]
    tile = cv2.resize(crop, (TILE_W, max(1, int(round(crop.shape[0] * k)))), cv2.INTER_AREA)
    for e in truth.elements:
        for box, col in ((e.bbox_initial, GREEN), (e.bbox_active, MAGENTA)):
            if box.w <= 0 or box.h <= 0:
                continue
            p0 = (int(round((box.x * r - x0) * k)), int(round((box.y * r - y0) * k)))
            p1 = (
                int(round(((box.x + box.w) * r - x0) * k)),
                int(round(((box.y + box.h) * r - y0) * k)),
            )
            cv2.rectangle(tile, p0, p1, col, 1, cv2.LINE_AA)
    return tile


def hstack_pad(tiles: list[np.ndarray], gap: int = 6) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        pad = np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)
        out += [np.vstack([t, pad]), np.full((h, gap, 3), 255, np.uint8)]
    return np.hstack(out[:-1])


def vstack_pad(rows: list[np.ndarray], gap: int = 10) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        pad = np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)
        out += [np.hstack([r, pad]), np.full((gap, w, 3), 255, np.uint8)]
    return np.vstack(out[:-1])


def sheet(truth: Truth, video: Path) -> np.ndarray:
    w, h = truth.video.width, truth.video.height
    rows: list[np.ndarray] = []
    title = f"{truth.name} ({truth.scenario}{', ' + truth.variant if truth.variant else ''})"
    head = np.full((30, 900, 3), 255, np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)
    rows.append(head)

    def full(t_s: float, txt: str) -> np.ndarray:
        f = grab(video, t_s, w, h)
        k = FULL_W / f.shape[1]
        return label(cv2.resize(f, (FULL_W, int(f.shape[0] * k)), cv2.INTER_AREA), txt)

    if truth.expected_error is not None or not truth.segments:
        dur = truth.video.duration_ms / 1000.0
        ts = [dur * k / 5 for k in range(6)]
        tiles = []
        for t in ts:
            f = grab(video, t, w, h)
            k = TILE_W / f.shape[1]
            tiles.append(
                label(
                    cv2.resize(f, (TILE_W, int(f.shape[0] * k)), cv2.INTER_AREA),
                    f"t={t * 1000:.0f}ms",
                )
            )
        rows.append(hstack_pad(tiles))
        return vstack_pad(rows)

    first = truth.segments[0]
    t_a = max(first.onset_ms - 200, 0) / 1000.0
    t_b = (first.settle_ms + 100) / 1000.0
    rows.append(hstack_pad([full(t_a, f"state A  t={t_a * 1000:.0f}ms"),
                            full(t_b, f"state B  t={t_b * 1000:.0f}ms")]))  # fmt: skip
    roi = roi_of(truth)
    for seg in truth.segments:
        tr = next(t for t in truth.transitions if t.segment_id == seg.id)
        bez = CubicBezier(*tr.easing.cubic_bezier)
        span = seg.settle_ms - seg.onset_ms
        marks = [("-150", seg.onset_ms - 150), ("0%", seg.onset_ms)]
        marks += [(f"{q}%", seg.onset_ms + span * q / 100.0) for q in (25, 50, 75, 100)]
        marks.append(("+150", seg.settle_ms + 150))
        tiles = []
        for name, t_ms in marks:
            x = (t_ms - tr.start_ms) / tr.duration_ms
            p = float(bez(np.clip(x, 0.0, 1.0)))
            f = grab(video, t_ms / 1000.0, w, h)
            tiles.append(label(crop_tile(f, truth, roi), f"{seg.id} {name} {t_ms:.0f}ms p={p:.2f}"))
        rows.append(hstack_pad(tiles))
    return vstack_pad(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (default: all truth files)")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help=f"synth dir ({DEFAULT_DIR})")
    args = ap.parse_args(argv)
    truths = sorted(args.dir.glob("*.truth.json"))
    if args.names:
        truths = [args.dir / f"{n}.truth.json" for n in args.names]
    if not truths:
        print(f"no truth files in {args.dir}; run `make synth` first", file=sys.stderr)
        return 2
    out_dir = args.dir / "contact"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tp in truths:
        truth = Truth.load(tp)
        img = sheet(truth, tp.parent / truth.video.file)
        dst = out_dir / f"{truth.name}.png"
        cv2.imwrite(str(dst), img)
        print(f"contact sheet: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
