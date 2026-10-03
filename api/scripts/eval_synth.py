"""Run the pipeline on the synthetic suite and check PLAN §11.4 thresholds (``make eval``).

    # default: analyze every data/synth/<name>.{mp4,mov,webm} (no server), write
    # data/synth/<name>.ir.json, compare each against <name>.truth.json, print the tables
    uv run python scripts/eval_synth.py [names...] [--jobs 4]

    # compare existing IR files only (no pipeline run)
    uv run python scripts/eval_synth.py --no-run

    # one pair
    uv run python scripts/eval_synth.py --truth ../data/synth/card_hover_translate.truth.json \\
        --ir /path/to/result.json

    # self-test with a perfect pipeline (truth converted to a MotionSpec)
    uv run python scripts/eval_synth.py --oracle

IR files may be a ResultEnvelope (``GET /api/jobs/{id}/result``), a bare MotionSpec, or an error
body ``{"error": {"code": ...}}`` (expected for the S12 negatives).

Suite verdict (§11.4): every scenario passes its per-transition / interaction / relationship
checks, and the easing family is correct on ≥ 80 % of eligible transitions (aggregate).

Exit: 0 all pass, 1 any threshold failed, 2 nothing to evaluate / bad input.
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
from tests.synth.evaluate import (  # noqa: E402
    compare,
    format_report,
    format_summary,
    load_ir,
    oracle_spec,
    summarize,
)
from tests.synth.truth import Truth  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "data" / "synth"


def _ir_for(truth_path: Path) -> Path | None:
    name = truth_path.name.removesuffix(".truth.json")
    for cand in (truth_path.parent / f"{name}.ir.json", truth_path.parent / "ir" / f"{name}.json"):
        if cand.exists():
            return cand
    return None


def _video_for(truth_path: Path) -> Path:
    truth = Truth.load(truth_path)
    return truth_path.parent / truth.video.file


def _pixel_ratio_option(truth_path: Path) -> str:
    """Synthetic videos are analysed like a user upload: ``auto`` (S10 tests the heuristic)."""
    return "auto"


def run_one(truth_path: str) -> tuple[str, float, str]:
    """Analyze one scenario video → ``<name>.ir.json``. Returns (name, seconds, status)."""
    from scripts.analyze import analyze_file

    tp = Path(truth_path)
    name = tp.name.removesuffix(".truth.json")
    video = _video_for(tp)
    t = time.perf_counter()
    data, _ = analyze_file(video, pixel_ratio=_pixel_ratio_option(tp), with_outputs=True)
    dt = time.perf_counter() - t
    out = tp.parent / f"{name}.ir.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    status = data["error"]["code"] if "error" in data else "ok"
    return name, dt, status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names in --dir (default: all)")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--truth", type=Path, help="single truth file (use with --ir)")
    ap.add_argument("--ir", type=Path, help="IR / ResultEnvelope / error JSON for --truth")
    ap.add_argument("--oracle", action="store_true", help="evaluate the truth against itself")
    ap.add_argument("--no-run", action="store_true", help="compare existing <name>.ir.json only")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    if args.truth:
        pairs = [(args.truth, args.ir)]
        if args.ir is None and not args.oracle:
            ap.error("--truth needs --ir (or --oracle)")
    else:
        truths = sorted(args.dir.glob("*.truth.json"))
        if args.names:
            truths = [args.dir / f"{n}.truth.json" for n in args.names]
        for tp in truths:
            if not tp.exists():
                print(f"missing truth file {tp}", file=sys.stderr)
                return 2
        if not args.oracle and not args.no_run and truths:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
                for name, dt, status in ex.map(run_one, [str(t) for t in truths]):
                    print(f"analyzed {name:24} {dt:5.1f}s  {status}")
            print(f"pipeline: {len(truths)} videos in {time.perf_counter() - t0:.1f}s\n")
        pairs = [(t, None if args.oracle else _ir_for(t)) for t in truths]

    reports, missing = [], []
    for tp, ip in pairs:
        if not tp.exists():
            print(f"missing truth file {tp}", file=sys.stderr)
            return 2
        truth = Truth.load(tp)
        if args.oracle:
            if truth.expected_error is not None:
                spec, code = None, truth.expected_error
            else:
                spec, code = oracle_spec(truth), None
        elif ip is None:
            missing.append(truth.name)
            continue
        else:
            spec, code = load_ir(json.loads(ip.read_text(encoding="utf-8")))
        rep = compare(truth, spec, code)
        reports.append(rep)
        print(format_report(rep) + "\n")

    if missing:
        print(f"no IR for: {', '.join(missing)}")
    if not reports:
        print(
            "eval: nothing to evaluate. Run `make synth` first (or pass --truth/--ir). "
            "`--oracle` self-tests the evaluator."
        )
        return 2
    summary = summarize(reports)
    print(format_summary(summary))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
