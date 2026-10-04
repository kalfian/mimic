"""Round-trip harness end to end on synthetic truth (PLAN-continuous §14). Opt-in:

    MIMIC_ROUNDTRIP=1 uv run pytest tests/unit/test_roundtrip_integration.py

Renders C4 / C3 / C8 / S2 into a temp dir, analyzes them, replays the hand-written reference
pages in headless Chrome, re-analyzes the replicas and compares (~2.5 min). Correct references
must PASS; the deliberately wrong C4 must FAIL exactly the manifest's rows. Needs Node >= 22,
ffmpeg and a headless Chrome (Playwright's chrome-headless-shell or ``$MIMIC_CHROME``).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _browser_available() -> bool:
    import roundtrip as rt  # noqa: PLC0415

    try:
        rt.find_chrome()
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(os.environ.get("MIMIC_ROUNDTRIP") != "1", reason="set MIMIC_ROUNDTRIP=1"),
]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if shutil.which("node") is None or shutil.which("ffmpeg") is None:
        pytest.skip("node and ffmpeg are required")
    if not _browser_available():
        pytest.skip("no headless Chrome")
    import roundtrip_validate as rv  # noqa: PLC0415

    work = tmp_path_factory.mktemp("roundtrip")
    return {r["html"]: r for r in rv.run(work, log=lambda *_: None)}


@pytest.mark.parametrize(
    "html",
    [
        "c4_carousel_drag_inertia.html",
        "c3_marquee_hover_pause.html",
        "c8_carousel_drag_snap.html",
        "s2_card_hover_compound.html",
    ],
)
def test_reference_round_trip_passes(results, html):
    """Every judged row passes, except rows the manifest documents as a known measurement
    limitation of the pipeline (``known_fail``; reported, never counted as PASS)."""
    r = results[html]
    assert not r["problems"], r["problems"]
    assert set(r["failed"]) <= set(r.get("known_fail", {})), r["failed"]


def test_wrong_reference_fails_the_right_rows(results):
    r = results["c4_wrong.html"]
    assert r["verdict"] == "FAIL"
    assert not r["problems"], r["problems"]
    for row in ("autoplay.speed", "inertia.tau", "phase sequence", "resume.delay", "resume.ramp"):
        assert row in r["failed"]
    assert "inertia.mode" not in r["failed"]
