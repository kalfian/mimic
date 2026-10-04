"""PLAN-continuous P2 integration: the real pipeline on a small synthetic carousel clip.

``media_clips.scroller_clip`` (480×240, 30 fps, 4 s): autoplay right at 40 px/s, press at
1.5 s, drag left at −500 px/s, release at 1.8 s (inertia τ 200 ms), rest. Checks the
``run.analyze`` wiring (regime → scroller analysis → continuous ``Measurement`` → assemble →
outputs incl. ``js``), the keyframe set, and that ``measurement.json`` round-trips byte for
byte and re-assembles to the same spec (re-run interpretation).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app import PIPELINE_VERSION
from app.config import get_settings
from app.generate import render_all
from app.interpret.fallback import FallbackInterpreter
from app.models.ir import MotionSpec
from app.pipeline import assemble as asm
from app.pipeline.persist import dump_measurement, load_measurement
from app.pipeline.run import analyze, run_interpreter
from tests.unit.media_clips import requires_ffmpeg, scroller_clip

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def analysis(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("scroller")
    video = scroller_clip(root / "scroller.mp4")
    kf = root / "keyframes"
    res = analyze(
        video, settings=get_settings(), job_id="0" * 32, filename="scroller.mp4", job_dir=root,
        keyframes_dir=kf,
    )  # fmt: skip
    return res, kf


def test_scroller_clip_is_analysed_as_continuous(analysis) -> None:
    res, _ = analysis
    spec: MotionSpec = res.spec
    assert spec.mode == "continuous" and spec.schema_version == "0.2"
    assert spec.meta.pipeline_version == PIPELINE_VERSION
    c = spec.continuous
    assert c is not None and c.axis == "x"
    kinds = [p.kind for p in c.phases]
    assert kinds[:3] == ["autoplay", "drag", "inertia"] and kinds[-1] == "paused"
    assert c.autoplay is not None and c.autoplay.direction == "right"
    assert c.autoplay.speed_px_s.value == pytest.approx(40, rel=0.03)
    b = c.behavior
    assert b.inertia is not None and b.inertia.tau_ms is not None
    assert b.inertia.tau_ms.value == pytest.approx(200, rel=0.25)
    # press stops autoplay at once (no cursor: trigger unknown)
    assert b.pause is not None and b.pause.decel_ms is None and b.pause.on == "unknown"
    assert spec.interaction.type == "drag" and spec.interaction.target_element_id == "e1"
    assert spec.scene is not None and spec.scene.viewport_css.w == 480
    out = render_all(spec)
    assert out.js is not None and "INERTIA_TAU_MS" in out.js
    assert "touch-action" in out.llm_prompt


def test_scroller_keyframes(analysis) -> None:
    res, kf = analysis
    names = {k.name for k in res.keyframes}
    assert {"state_a.png", "annotated_a.png", "phase_1.png"} <= names
    assert not names & {"state_b.png", "mid_50.png", "annotated_b.png"}
    phases = [k for k in res.keyframes if k.kind == "phase"]
    assert 1 <= len(phases) <= 6 and all(k.t_s is not None for k in phases)
    assert all((kf / k.name).is_file() for k in res.keyframes)


def test_continuous_measurement_round_trips_byte_identical(analysis, tmp_path: Path) -> None:
    res, kf = analysis
    m = res.measurement
    assert m.continuous is not None and m.heuristic.type == "drag"
    text = json.dumps(dump_measurement(m, res.extra_warnings))
    m2, extra2 = load_measurement(json.loads(text))
    assert json.dumps(dump_measurement(m2, extra2)) == text
    # re-run interpretation on the stored measurement: same spec as the first run
    interp = run_interpreter(m2, tmp_path, {}, FallbackInterpreter(), None, 1.0)  # type: ignore[arg-type]
    kw = dict(
        job_id="0" * 32,
        filename="scroller.mp4",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        pipeline_version=PIPELINE_VERSION,
        use_interpreter=False,
    )
    first = asm.assemble(m, res.interpretation, extra_warnings=res.extra_warnings, **kw)
    again = asm.assemble(m2, interp, extra_warnings=extra2, **kw)
    assert again.model_dump_json() == first.model_dump_json()
