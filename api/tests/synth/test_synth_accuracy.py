"""End-to-end accuracy on synthetic videos vs ground truth (PLAN §11.4). Owned by Track INT.

Runs the real pipeline (no server, interpreter off) on every ``data/synth/<name>.truth.json``
scenario and asserts the §11.4 thresholds: per scenario (values, start, duration, curve RMSE,
interaction type/direction, relationships, expected errors for the negatives) and the suite-
level easing-family rate (≥ 80 % of eligible transitions).

Only runs with ``pytest -m synth`` (``make eval`` runs the same checks via
``scripts/eval_synth.py`` and prints the full tables). Needs ``make synth`` first.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from app.config import REPO_ROOT
from tests.synth.evaluate import Report, compare, format_report, format_summary, load_ir, summarize
from tests.synth.truth import Truth

pytestmark = pytest.mark.synth

SYNTH_DIR = REPO_ROOT / "data" / "synth"
TRUTHS = sorted(SYNTH_DIR.glob("*.truth.json"))


def _analyze(truth_path: str) -> tuple[str, dict]:
    from scripts.analyze import analyze_file

    tp = Path(truth_path)
    truth = Truth.load(tp)
    data, _ = analyze_file(tp.parent / truth.video.file, with_outputs=True)
    return truth_path, data


@pytest.fixture(scope="module")
def reports() -> dict[str, Report]:
    if not TRUTHS:
        pytest.skip("no synthetic videos; run `make synth` first")
    out: dict[str, Report] = {}
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        for tp, data in ex.map(_analyze, [str(t) for t in TRUTHS]):
            truth = Truth.load(Path(tp))
            spec, code = load_ir(data)
            out[truth.name] = compare(truth, spec, code)
    return out


@pytest.mark.parametrize("name", [t.name.removesuffix(".truth.json") for t in TRUTHS])
def test_scenario(reports: dict[str, Report], name: str) -> None:
    rep = reports[name]
    assert rep.ok, format_report(rep)


def test_suite_thresholds(reports: dict[str, Report]) -> None:
    summary = summarize(list(reports.values()))
    assert summary.ok, format_summary(summary)
    # calibration: transitions reported with high confidence are within every target ≥ 90 %
    ok, n = summary.calibration["high"]
    assert n == 0 or ok / n >= 0.9, format_summary(summary)
