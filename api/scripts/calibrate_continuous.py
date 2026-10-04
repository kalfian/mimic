"""Continuous confidence-band calibration on simulated scroller profiles (PLAN-continuous §9 P2).

The synthetic video suite has a dozen scrollers; whether a *high* band is earned needs many
more. This Monte Carlo builds C4-shaped velocity profiles (numbers only, no frames) over the
§9 P2 ranges — autoplay 20–120 px/s, drag peaks 300–2500 px/s, τ 80–600 ms (decaying to the
10 px/s cut the synthetic truth uses), resume ramps 300–1000 ms with five easings, 60 / 30 fps,
position noise σ 0.05 / 0.2 px, and a quarter of the trials on the motivating recording's
capture grid (75 Hz stamps, 60 Hz page) — runs the production segmentation
(``continuous.phases.segment``) and reports, per confidence band, how often each value is
within its §8.4 target (60 fps targets; 30 fps / σ 0.2 / capture grid use the relaxed ones).

    uv run python scripts/calibrate_continuous.py [--trials 160] [--seed 1]

Target: values in the *high* band are within target ≥ 90 % of the time. Exit 1 otherwise.
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

from app.models.ir import ExponentialFit, RampFit  # noqa: E402
from app.pipeline.continuous.phases import segment  # noqa: E402
from tests.unit.test_continuous_kinematics import (  # noqa: E402
    ease,
    make_grid_series,
    make_series,
)

EASINGS = ("ease-in-out", "ease-in", "ease", "linear", "ease-out")
#: §8.4 targets: (fast, slow)
TARGETS = {
    "autoplay.speed": (0.02, 0.03),  # relative
    "inertia.tau": (0.15, 0.25),  # relative, τ ≥ 120 ms
    "inertia.v0": (0.15, 0.25),  # relative
    "drag.peak": (0.10, 0.15),  # relative
    "resume.delay": (50.0, 80.0),  # ms
    "resume.ramp": ((60.0, 0.20), (90.0, 0.25)),  # max(ms, rel)
}
CUT_PX_S = 10.0


class Profile:
    """autoplay → press → drag (ease to the peak, hold) → release → inertia τ (cut at 10 px/s)
    → rest → resume ramp → autoplay."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.va = float(rng.uniform(20, 120)) * (1 if rng.random() < 0.5 else -1)
        self.peak = float(rng.uniform(300, 2500)) * (1 if rng.random() < 0.5 else -1)
        self.t_press = 1.5
        self.drag_s = float(rng.uniform(0.3, 0.6))
        self.tau = float(rng.uniform(0.08, 0.6))
        self.t_rel = self.t_press + self.drag_s
        self.t_rest = self.t_rel + self.tau * float(np.log(abs(self.peak) / CUT_PX_S))
        self.rest_s = float(rng.uniform(0.6, 1.2))
        self.t_res = self.t_rest + self.rest_s
        self.ramp_s = float(rng.uniform(0.3, 1.0))
        self.easing = EASINGS[int(rng.integers(len(EASINGS)))]
        self.dur = self.t_res + self.ramp_s + 1.0

    def __call__(self, t: np.ndarray) -> np.ndarray:
        v = np.zeros_like(t)
        v[t < self.t_press] = self.va
        m = (t >= self.t_press) & (t < self.t_rel)
        v[m] = self.peak * np.sin(np.pi / 2 * np.clip((t[m] - self.t_press) / 0.12, 0, 1))
        m = (t >= self.t_rel) & (t < self.t_rest)
        v[m] = self.peak * np.exp(-(t[m] - self.t_rel) / self.tau)
        m = (t >= self.t_res) & (t < self.t_res + self.ramp_s)
        v[m] = self.va * ease(self.easing, (t[m] - self.t_res) / self.ramp_s)
        v[t >= self.t_res + self.ramp_s] = self.va
        return v


def _rel(got: float, want: float) -> float:
    return abs(abs(got) - abs(want)) / abs(want)


def trial(rng: np.random.Generator) -> list[tuple[str, str, bool]]:
    prof = Profile(rng)
    fps = 60.0 if rng.random() < 0.5 else 30.0
    sigma = 0.05 if rng.random() < 0.5 else 0.2
    grid = rng.random() < 0.25
    seed = int(rng.integers(1 << 30))
    if grid:
        s = make_grid_series(prof, dur=prof.dur, sigma=sigma, seed=seed)
        slow = True
    else:
        s = make_series(prof, fps=fps, dur=prof.dur, sigma=sigma, seed=seed)
        slow = fps < 50 or sigma > 0.1
    k = 1 if slow else 0
    seg = segment(s, is_vfr=grid)
    out: list[tuple[str, str, bool]] = []
    ap = [p for p in seg.phases if p.kind == "autoplay" and p.fit is not None]
    if ap and seg.autoplay_velocity is not None:
        band = ap[0].fit.velocity_px_s.confidence.band  # type: ignore[union-attr]
        ok = _rel(seg.autoplay_velocity, prof.va) <= TARGETS["autoplay.speed"][k]
        out.append(("autoplay.speed", band, ok))
    inert = [p for p in seg.phases if p.kind == "inertia" and isinstance(p.fit, ExponentialFit)]
    if inert:
        f = inert[0].fit
        assert isinstance(f, ExponentialFit)
        if prof.tau >= 0.12:
            ok = _rel(f.tau_ms.value / 1000, prof.tau) <= TARGETS["inertia.tau"][k]
            out.append(("inertia.tau", f.tau_ms.confidence.band, ok))
        ok = _rel(f.v0_px_s.value, prof.peak) <= TARGETS["inertia.v0"][k]
        out.append(("inertia.v0", f.v0_px_s.confidence.band, ok))
    b = seg.behavior
    if b.drag is not None:
        ok = _rel(b.drag.peak_speed_px_s.value, prof.peak) <= TARGETS["drag.peak"][k]
        out.append(("drag.peak", b.drag.peak_speed_px_s.confidence.band, ok))
    res = [p for p in seg.phases if p.kind == "resume" and isinstance(p.fit, RampFit)]
    if b.resume is not None and res:
        d = b.resume.delay_after_rest_ms.value - prof.rest_s * 1000
        out.append(("resume.delay", b.resume.delay_after_rest_ms.confidence.band,
                    abs(d) <= TARGETS["resume.delay"][k]))  # fmt: skip
        ab, rl = TARGETS["resume.ramp"][k]
        tol = max(ab, rl * prof.ramp_s * 1000)
        out.append(("resume.ramp", b.resume.ramp_ms.confidence.band,
                    abs(b.resume.ramp_ms.value - prof.ramp_s * 1000) <= tol))  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1)
    ns = ap.parse_args(argv)
    rng = np.random.default_rng(ns.seed)
    per: dict[tuple[str, str], list[bool]] = defaultdict(list)
    failed = 0
    for _ in range(ns.trials):
        try:
            rows = trial(rng)
        except Exception as exc:  # report, keep going: a crash is a finding too
            failed += 1
            print(f"trial failed: {exc!r}")
            continue
        for name, band, ok in rows:
            per[(name, band)].append(ok)
    print(f"{'value':16} {'band':7} {'within/n':>9}  rate")
    for (name, band), oks in sorted(per.items()):
        print(f"{name:16} {band:7} {sum(oks):4}/{len(oks):<4}  {sum(oks) / len(oks):.0%}")
    total = defaultdict(lambda: [0, 0])
    for (_, band), oks in per.items():
        total[band][0] += sum(oks)
        total[band][1] += len(oks)
    print("bands:")
    for band in ("high", "medium", "low"):
        ok, n = total[band]
        print(f"  {band:6} {ok}/{n} {ok / n:.0%}" if n else f"  {band:6} 0/0 -")
    ok, n = total["high"]
    passed = n == 0 or ok / n >= 0.9
    crashed = f" ({failed} crashed)" if failed else ""
    print(f"high band ≥ 90 %: {'PASS' if passed else 'FAIL'}{crashed}")
    return 0 if passed and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
