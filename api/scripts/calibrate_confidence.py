"""Confidence-band calibration on simulated progress series (PLAN §6.8, §12 Phase 4).

The synthetic video suite is small and almost every transition lands inside its §11.4 targets,
so it cannot show whether a *high* band is earned. This Monte Carlo feeds noisy, ideal
translate series (every Appendix B curve, several durations, 60 / 30 fps, σ = 0.01 / 0.02
progress noise) through the production fit + confidence code (``run.fit_transitions``) and
reports, per band, how often the result is within every §11.4 timing/curve target.

    uv run python scripts/calibrate_confidence.py [--trials 6] [--seed 1]

Target (PLAN §12 Phase 4): rows in the *high* band are within targets ≥ 90 % of the time.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.models.ir import PxValue  # noqa: E402
from app.models.measure import (  # noqa: E402
    CursorTrack,
    ElementCandidate,
    ProbeInfo,
    PropertySeries,
    Rect,
)
from app.pipeline.assemble import SegmentWindow  # noqa: E402
from app.pipeline.easing import CANDIDATES, eval_curve, families_match  # noqa: E402
from app.pipeline.params import DEFAULT_PARAMS  # noqa: E402
from app.pipeline.run import fit_transitions  # noqa: E402

DURATIONS_MS = (150, 200, 300, 450)
FPS = (60.0, 30.0)
SIGMAS = (0.01, 0.02)
DELTA_PX = -8.0


def _within(start_err: float, dur_err: float, dur: float, rmse: float, fps: float) -> bool:
    fast = fps >= 50
    onset_tol = 17.0 if fast else 34.0
    abs_ms, rel = (20.0, 0.10) if fast else (35.0, 0.12)
    return abs(start_err) <= onset_tol and abs(dur_err) <= max(abs_ms, rel * dur) and rmse <= 0.05


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ns = ap.parse_args(argv)
    rng = np.random.default_rng(ns.seed)
    el = ElementCandidate("e1", Rect(0, 0, 200, 100), Rect(0, -8, 200, 100), None,
                          "transform", 1.0)  # fmt: skip
    cursor = CursorTrack(visible=False)
    curves = [c for c in CANDIDATES if not c.overshoot_only]
    stats: dict[str, list[bool]] = defaultdict(list)
    fam: dict[str, list[bool]] = defaultdict(list)
    worst: list[tuple[float, str]] = []
    for fps in FPS:
        info = ProbeInfo("mp4", "h264", 1280, 800, 0, 3.0, np.arange(0, 3.0, 1 / fps), fps,
                         fps, False, False)  # fmt: skip
        for dur_ms in DURATIONS_MS:
            for sigma in SIGMAS:
                for curve in curves:
                    for _ in range(ns.trials):
                        t0 = 1.0 + rng.uniform(0, 1 / fps)
                        t = np.arange(0.6, 2.2, 1 / fps) + rng.uniform(0, 1 / fps)
                        x = np.clip((t - t0) / (dur_ms / 1000), 0, 1)
                        p = eval_curve(curve.bezier, x) + rng.normal(0, sigma, t.size)
                        vals = DELTA_PX * p
                        s = PropertySeries(
                            "e1", "translateY", "fwd", t, vals, np.full(t.size, 0.99), 0.0,
                            DELTA_PX, "px", PxValue(number=0.0), PxValue(number=DELTA_PX),
                            stable_std=abs(DELTA_PX) * sigma,
                        )  # fmt: skip
                        fts = fit_transitions([s], [el], info, cursor,
                                              {"fwd": SegmentWindow("fwd", 0.9, 1.6)},
                                              DEFAULT_PARAMS)  # fmt: skip
                        if not fts:
                            stats["(dropped)"].append(False)
                            continue
                        ft = fts[0]
                        start_err = (ft.t0_s - t0) * 1000
                        dur_err = ft.duration_s * 1000 - dur_ms
                        xs = np.linspace(-0.2, 1.4, 400)
                        tt = t0 + xs * dur_ms / 1000
                        pt = eval_curve(curve.bezier, np.clip(xs, 0, 1))
                        pf = eval_curve(ft.easing.cubic_bezier,
                                        np.clip((tt - ft.t0_s) / ft.duration_s, 0, 1))  # fmt: skip
                        rmse = float(np.sqrt(np.mean((pt - pf) ** 2)))
                        ok = _within(start_err, dur_err, dur_ms, rmse, fps)
                        band = ft.confidence.band
                        stats[band].append(ok)
                        if fps >= 50 and dur_ms >= 150:
                            fam[band].append(families_match(curve.family, ft.easing.family))
                        if band == "high" and not ok:
                            worst.append((abs(dur_err), f"{curve.name} {dur_ms}ms {fps:.0f}fps "
                                          f"σ{sigma}: start {start_err:+.0f} dur {dur_err:+.0f} "
                                          f"timing {ft.confidence.timing:.2f}"))  # fmt: skip
    print("band    within-targets   family-correct (60 fps)")
    for band in ("high", "medium", "low", "(dropped)"):
        v = stats.get(band, [])
        f = fam.get(band, [])
        rate = f"{np.mean(v):.0%}" if v else "-"
        frate = f"{np.mean(f):.0%}" if f else "-"
        print(f"{band:9} {sum(v):4}/{len(v):<4} {rate:>5}     {sum(f):4}/{len(f):<4} {frate:>5}")
    if worst:
        print("\nworst high-band misses:")
        for _, w in sorted(worst, reverse=True)[:8]:
            print(f"  {w}")
    hv = stats.get("high", [])
    ok = not hv or float(np.mean(hv)) >= 0.9
    print(f"\nhigh band within targets >= 90%: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
