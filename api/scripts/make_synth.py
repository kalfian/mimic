"""Render the synthetic scenario suite (PLAN §11.2–11.3, Track S).

For every scenario S1–S12 (incl. S10/S11 variants) this writes into ``data/synth/``:

* ``<name>.<mp4|mov|webm>`` — the encoded recording,
* ``<name>.truth.json``     — ground truth (``tests/synth/truth.py``) incl. ffprobe facts,
* ``manifest.json``         — index of everything rendered.

Usage (from ``api/``)::

    uv run python scripts/make_synth.py                 # all scenarios
    uv run python scripts/make_synth.py card_hover_translate modal_open
    uv run python scripts/make_synth.py --list
    uv run python scripts/make_synth.py --jobs 4 --out /tmp/synth

Exit status is non-zero if any encode fails or an ffprobe check (codec / size / fps / frame
count / duration / VFR) does not match the requested variant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.config import REPO_ROOT  # noqa: E402
from tests.synth import encode, scenarios  # noqa: E402
from tests.synth.scene import Renderer  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "synth"


def render_one(name: str, out_dir: str) -> dict:
    """Render + encode + probe one scenario; returns a manifest row (runs in a worker)."""
    t0 = time.perf_counter()
    entry = scenarios.get(name)
    defn = entry.build()
    truth = scenarios.build_truth(entry, defn)
    v = entry.variant
    renderer = Renderer(defn.scene, v.pixel_ratio)
    out = Path(out_dir)
    video_path = out / entry.filename
    frames = (f for _, f in renderer.frames(v.fps))
    n = encode.encode_frames(frames, renderer.W, renderer.H, v.fps, v.encode, video_path)
    if n != truth.video.frames_rendered:
        raise RuntimeError(f"{name}: rendered {n} frames, truth says {truth.video.frames_rendered}")
    truth.video.probe = encode.probe(video_path)
    truth_path = out / f"{name}.truth.json"
    truth.write(truth_path)
    problems = encode.verify(truth.video)
    p = truth.video.probe
    return {
        "name": name,
        "scenario": entry.scenario,
        "variant": v.label,
        "video": video_path.name,
        "truth": truth_path.name,
        "expected_error": truth.expected_error,
        "interaction": truth.interaction.type if truth.interaction else None,
        "transitions": len(truth.transitions),
        "codec": p.codec_name,
        "size": f"{p.width}x{p.height}",
        "fps": p.avg_fps,
        "frames": p.frame_count,
        "duration_ms": p.duration_ms,
        "vfr": p.is_vfr,
        "bytes": video_path.stat().st_size,
        "seconds": round(time.perf_counter() - t0, 1),
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dir ({DEFAULT_OUT})")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args(argv)

    if args.list:
        for e in scenarios.SCENARIOS:
            print(f"{e.scenario:4} {e.name:24} {e.filename}")
        return 0
    if not encode.tools_available():
        print("make_synth: ffmpeg/ffprobe not found on PATH", file=sys.stderr)
        return 2

    names = args.names or [e.name for e in scenarios.SCENARIOS]
    unknown = [n for n in names if n not in scenarios.BY_NAME]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    skipped = []
    if not encode.encoder_available("libvpx-vp9"):
        skipped = [n for n in names if scenarios.get(n).variant.encode.codec == "vp9"]
        names = [n for n in names if n not in skipped]
        for n in skipped:
            print(f"skip {n}: ffmpeg has no libvpx-vp9 encoder")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict] = []
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futs = {n: pool.submit(render_one, n, str(args.out)) for n in names}
        for n, f in futs.items():
            try:
                rows.append(f.result())
            except Exception as exc:  # noqa: BLE001 - report every failing scenario
                failed.append(n)
                print(f"FAILED {n}: {exc}", file=sys.stderr)

    hdr = f"{'scenario':4} {'name':24} {'codec':5} {'size':9} {'fps':>6} {'frames':>6} " \
          f"{'dur_ms':>6} {'vfr':5} {'tr':>3} {'sec':>5}  check"  # fmt: skip
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        status = "ok" if not r["problems"] else "; ".join(r["problems"])
        print(
            f"{r['scenario']:4} {r['name']:24} {r['codec']:5} {r['size']:9} {r['fps']:>6g} "
            f"{r['frames']:>6} {r['duration_ms']:>6} {str(r['vfr']):5} {r['transitions']:>3} "
            f"{r['seconds']:>5}  {status}"
        )
        if r["problems"]:
            failed.append(r["name"])

    if not args.names:
        manifest = {
            "generated_by": "api/scripts/make_synth.py",
            "scenarios": [{k: v for k, v in r.items() if k != "seconds"} for r in rows],
            "skipped": skipped,
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(rows)} rendered, {len(failed)} failed, {len(skipped)} skipped "
          f"in {time.perf_counter() - t0:.1f}s -> {args.out}")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
