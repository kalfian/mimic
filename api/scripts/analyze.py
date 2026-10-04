"""Analyze one local video without the server: video → IR (+ the four outputs).

    uv run python scripts/analyze.py VIDEO [--pixel-ratio auto|1|2|3] [--out result.json]
        [--print technical|llm_prompt|css|js|json|ir] [--keyframes DIR] [--debug DIR]
        [--use-interpreter] [--no-keyframes]

Writes a ResultEnvelope-shaped JSON (``job_id``, ``spec``, ``outputs``; no artifact URLs) or,
for an unanalysable recording, the API error body ``{"error": {"code", "message"}}`` (exit 3).
AI labeling only runs with ``--use-interpreter`` *and* a configured ``MIMIC_INTERPRETER``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.config import get_settings  # noqa: E402
from app.core.errors import PipelineError  # noqa: E402
from app.generate import render_all  # noqa: E402
from app.pipeline.run import NullProgress, analyze  # noqa: E402


def job_id_for(path: Path) -> str:
    """Deterministic 32-hex id per absolute path (IR ``job_id`` format)."""
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()  # noqa: S324


def analyze_file(
    video: Path,
    *,
    pixel_ratio: str = "auto",
    use_interpreter: bool = False,
    keyframes_dir: Path | None = None,
    debug_dir: Path | None = None,
    with_outputs: bool = True,
    write_keyframes: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    """``(result_json, stage_timings)``; result is an envelope or an error body.

    ``write_keyframes=False`` writes no frame images at all (not even to the temporary job
    directory): for recordings whose frames must not be stored.
    """
    settings = get_settings()
    progress = NullProgress()
    jid = job_id_for(video)
    with tempfile.TemporaryDirectory(prefix="mimic-analyze-") as tmp:
        job_dir = Path(tmp)
        kf_dir = (keyframes_dir or (job_dir / "keyframes")) if write_keyframes else None
        try:
            res = analyze(
                video,
                settings=settings,
                job_id=jid,
                filename=video.name,
                job_dir=job_dir,
                pixel_ratio=pixel_ratio,
                use_interpreter=use_interpreter,
                progress=progress,
                keyframes_dir=kf_dir,
                debug_dir=debug_dir,
            )
        except PipelineError as exc:
            return exc.to_body(), progress.finish()
        spec = res.spec
        data: dict[str, Any] = {"job_id": jid, "spec": spec.model_dump(mode="json")}  # type: ignore[attr-defined]
        if with_outputs:
            data["outputs"] = render_all(spec).model_dump(mode="json")  # type: ignore[arg-type]
    return data, progress.finish()


def continuous_summary(name: str, spec: dict[str, Any], elapsed: float, timings: dict) -> str:
    """Phase table of a continuous result (numbers only)."""
    c = spec["continuous"]
    ap = c.get("autoplay")
    lines = [
        f"{name}: continuous ({spec['interaction']['type']}), axis {c['axis']}, "
        f"{len(c['phases'])} phases, {elapsed:.1f}s {timings}",
        "  autoplay: "
        + (f"{ap['velocity_px_s']:+g} px/s ({ap['speed_px_s']['confidence']['band']})"
           if ap else "none"),
        f"  {'id':>4} {'kind':10} {'start':>6} {'end':>6} {'dur':>5} {'v_start':>8} "
        f"{'v_end':>8} {'v_peak':>8}  fit",
    ]  # fmt: skip
    for p in c["phases"]:
        fit = p.get("fit") or {}
        model = fit.get("model", "")
        if model == "exponential":
            stop = f", stop {fit['stop_px_s']:g} px/s" if fit.get("stop_px_s") else ""
            extra = f"τ {fit['tau_ms']['value']:g} ms, v0 {fit['v0_px_s']['value']:+g}{stop}"
        elif model in ("ramp", "tween"):
            extra = f"{fit['duration_ms']['value']:g} ms {fit['easing']['nearest_named']}"
        elif model == "constant":
            extra = f"{fit['velocity_px_s']['value']:+g} px/s"
        else:
            extra = ""
        lines.append(
            f"  {p['id']:>4} {p['kind']:10} {p['start_ms']:6} {p['end_ms']:6} "
            f"{p['end_ms'] - p['start_ms']:5} {p['v_start_px_s']:8.1f} {p['v_end_px_s']:8.1f} "
            f"{p['v_peak_px_s']:8.1f}  {model} {extra}".rstrip()
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--pixel-ratio", default="auto", choices=["auto", "1", "2", "3"])
    ap.add_argument("--out", type=Path, help="write the JSON here (default: stdout summary only)")
    ap.add_argument(
        "--print",
        dest="show",
        choices=["technical", "llm_prompt", "css", "js", "json", "ir"],
        help="print one output (or the full IR); js exists in continuous mode only",
    )
    ap.add_argument("--keyframes", type=Path, help="keep keyframe PNGs in this directory")
    ap.add_argument("--debug", type=Path, help="write MIMIC_DEBUG-style dumps here")
    ap.add_argument("--use-interpreter", action="store_true", help="opt in to AI labeling")
    ap.add_argument(
        "--no-keyframes",
        action="store_true",
        help="write no frame images at all (numbers-only analysis of a private recording)",
    )
    ns = ap.parse_args(argv)
    if not ns.video.is_file():
        ap.error(f"no such file: {ns.video}")

    t = time.perf_counter()
    data, timings = analyze_file(
        ns.video,
        pixel_ratio=ns.pixel_ratio,
        use_interpreter=ns.use_interpreter,
        keyframes_dir=ns.keyframes,
        debug_dir=ns.debug,
        write_keyframes=not ns.no_keyframes,
    )
    elapsed = time.perf_counter() - t
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if "error" in data:
        print(f"error: {data['error']['code']}: {data['error']['message']}", file=sys.stderr)
        return 3
    spec = data["spec"]
    if ns.show == "ir":
        print(json.dumps(spec, indent=2, ensure_ascii=False))
    elif ns.show:
        out = data["outputs"].get(ns.show)
        print(out if out is not None else f"(no {ns.show} output in {spec['mode']} mode)")
    elif spec.get("mode") == "continuous":
        print(continuous_summary(ns.video.name, spec, elapsed, timings))
    else:
        it = spec["interaction"]
        print(f"{ns.video.name}: {it['type']} ({it['trigger']['kind']}, {it['direction']}), "
              f"{len(spec['elements'])} elements, {len(spec['transitions'])} transitions, "
              f"{elapsed:.1f}s {timings}")  # fmt: skip
        for tr in spec["transitions"]:
            c = tr["confidence"]
            name = tr["easing"]["nearest_named"]
            print(f"  {tr['id']:>4} {tr['segment_id']:6} {tr['element_id']:4} "
                  f"{tr['property']:16} start {tr['start_ms']:5} dur {tr['duration_ms']:4} "
                  f"{name:18} conf {c['overall']:.2f} {c['band']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
