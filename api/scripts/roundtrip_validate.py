"""Validate the round-trip harness on known truth (PLAN-continuous §14) before pointing it at a
real recording.

    uv run python scripts/roundtrip_validate.py [--work DIR] [--only c4_wrong.html ...] [--reuse]

For every entry of ``tools/roundtrip/reference/manifest.json``: render the synthetic scenario
(``scripts/make_synth.py --out <work>/synth``; ``data/synth`` is never touched), analyze it →
source IR, run ``scripts/roundtrip.py`` with the hand-written reference page → replica IR, and
compare. Correct references must PASS; deliberately wrong ones must FAIL exactly the rows
listed in ``must_fail`` while the ``must_pass`` rows still pass. Exit 0 when every expectation
holds. Default work dir: ``/tmp/mimic-roundtrip/validate`` (``--reuse`` keeps rendered videos
and source IRs from a previous run).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
API_DIR = SCRIPTS_DIR.parent
REPO_ROOT = API_DIR.parent
REFERENCE_DIR = REPO_ROOT / "tools" / "roundtrip" / "reference"
DEFAULT_WORK = Path("/tmp/mimic-roundtrip/validate")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roundtrip as rt  # noqa: E402
from analyze import analyze_file  # noqa: E402


def load_manifest() -> list[dict[str, Any]]:
    return json.loads((REFERENCE_DIR / "manifest.json").read_text("utf-8"))["references"]


def check_expectation(entry: dict[str, Any], rows: list[dict[str, Any]], verdict: str) -> list[str]:
    """Problems with a compare result w.r.t. the manifest entry (empty = as expected)."""
    status = {r["criterion"]: r["status"] for r in rows}
    known = entry.get("known_fail", {})
    problems: list[str] = []
    if entry["expect"] == "pass":
        bad = [c for c, s in status.items() if s == "FAIL" and c not in known]
        if bad or not any(s == "PASS" for s in status.values()):
            problems.append(f"expected PASS, failed rows: {', '.join(bad) or 'none judged'}")
        return problems
    if verdict != "FAIL":
        problems.append("expected FAIL, got PASS")
    for c in entry.get("must_fail", []):
        if status.get(c) != "FAIL":
            problems.append(f"row '{c}' should FAIL, got {status.get(c, 'missing')}")
    for c in entry.get("must_pass", []):
        if status.get(c) != "PASS":
            problems.append(f"row '{c}' should PASS, got {status.get(c, 'missing')}")
    return problems


def prepare_sources(scenarios: list[str], work: Path, *, reuse: bool) -> dict[str, Path]:
    synth = work / "synth"
    synth.mkdir(parents=True, exist_ok=True)
    todo = [s for s in scenarios if not (reuse and list(synth.glob(f"{s}.mp4")))]
    if todo:
        cmd = [sys.executable, str(SCRIPTS_DIR / "make_synth.py"), "--out", str(synth), *todo]
        proc = subprocess.run(cmd, cwd=API_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"make_synth failed: {proc.stdout[-600:]} {proc.stderr[-600:]}")
    out: dict[str, Path] = {}
    for s in scenarios:
        ir = synth / f"{s}.source.json"
        if not (reuse and ir.is_file()) or s in todo:
            data, _ = analyze_file(synth / f"{s}.mp4", write_keyframes=False)
            if "error" in data:
                raise RuntimeError(f"{s}: source analysis failed: {data['error']}")
            ir.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
        out[s] = ir
    return out


def run(work: Path, *, only: list[str] | None = None, reuse: bool = False, log=print) -> list[dict]:
    entries = [e for e in load_manifest() if not only or e["html"] in only]
    sources = prepare_sources(sorted({e["scenario"] for e in entries}), work, reuse=reuse)
    results = []
    for e in entries:
        html = REFERENCE_DIR / e["html"]
        out = work / Path(e["html"]).stem
        t = time.perf_counter()
        log(f"== {e['html']} (source {e['scenario']}, expect {e['expect']})")
        res = rt.roundtrip(html, sources[e["scenario"]], out, keep_frames=False, log=log)
        cmp_ = json.loads(Path(res["compare"]).read_text("utf-8"))
        problems = check_expectation(e, cmp_["rows"], cmp_["verdict"])
        log(res["table"])
        log(
            f"   {time.perf_counter() - t:.1f}s {res['timings']}  -> "
            f"{'as expected' if not problems else 'UNEXPECTED: ' + '; '.join(problems)}"
        )
        known = sorted(set(res["failed"]) & set(e.get("known_fail", {})))
        for c in known:
            log(f"   known measurement limitation, {c}: {e['known_fail'][c]}")
        results.append({**e, "verdict": cmp_["verdict"], "failed": res["failed"], "known": known,
                        "problems": problems, "seconds": round(time.perf_counter() - t, 1),
                        "timings": res["timings"], "out_dir": res["out_dir"]})  # fmt: skip
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--only", nargs="*", help="reference html file names to run")
    ap.add_argument("--reuse", action="store_true", help="reuse rendered videos / source IRs")
    ns = ap.parse_args(argv)
    t = time.perf_counter()
    try:
        results = run(ns.work, only=ns.only, reuse=ns.reuse)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print("\nsummary")
    for r in results:
        ok = "ok  " if not r["problems"] else "FAIL"
        failed = ", ".join(c + (" (known)" if c in r["known"] else "") for c in r["failed"]) or "-"
        print(f"  [{ok}] {r['html']:32} expect {r['expect']:4} got {r['verdict']:4} "
              f"{r['seconds']:6.1f}s  failed rows: {failed}")  # fmt: skip
    good = all(not r["problems"] for r in results)
    n_known = sum(len(r["known"]) for r in results)
    extra = f", {n_known} known measurement limitation(s) listed above" if n_known else ""
    took = time.perf_counter() - t
    print(f"harness self-check: {'PASS' if good else 'FAIL'}{extra} ({took:.0f}s)")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
