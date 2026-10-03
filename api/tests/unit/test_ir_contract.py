"""Phase 0 contract tests.

* ``docs/contract/sample-result.json`` validates against the Pydantic models AND the generated
  JSON Schema, and is canonical (model round-trip reproduces it byte-for-byte as JSON).
* Generated files (schema + ``web/lib/types.ts``) are up to date with the models.
* The IR integrity rules reject the classes of mistakes later tracks are likely to make.
* Frozen P0 modules keep their basic invariants (stages, errors, params, render_all).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from app import contract
from app.api.schemas import KEYFRAME_NAME_RE, ResultEnvelope
from app.core.errors import ERROR_HTTP_STATUS, ErrorCode, PipelineError
from app.core.stages import STAGE_LABELS, STAGE_ORDER, STAGE_WINDOWS, Stage, overall_progress
from app.generate import Outputs, render_all
from app.models.ir import MotionSpec, band_for
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams

# --------------------------------------------------------------------------------------------
# fixture <-> models <-> schema
# --------------------------------------------------------------------------------------------


def test_sample_result_validates_against_models(sample_result: dict[str, Any]) -> None:
    env = ResultEnvelope.model_validate(sample_result)
    assert env.spec.job_id == env.job_id
    assert env.spec.interaction.direction == "forward_reverse"
    assert len(env.spec.elements) == 4
    bands = {t.confidence.band for t in env.spec.transitions}
    assert bands == {"high", "medium", "low"}, "fixture must mix confidence bands"
    assert len(env.spec.warnings) == 1


def test_sample_result_is_canonical(sample_result: dict[str, Any]) -> None:
    """Dumping the validated model reproduces the fixture exactly (incl. derived states)."""
    env = ResultEnvelope.model_validate(sample_result)
    assert env.model_dump(mode="json") == sample_result


def test_sample_result_validates_against_json_schema(sample_result: dict[str, Any]) -> None:
    schema = json.loads(contract.SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(sample_result), key=str)
    assert not errors, "\n".join(str(e) for e in errors[:5])


def test_schema_file_is_up_to_date() -> None:
    expected = contract.schema_text()
    actual = contract.SCHEMA_PATH.read_text(encoding="utf-8")
    assert actual == expected, "motion-spec.schema.json is stale: run `make contract`"


def test_types_ts_is_up_to_date() -> None:
    expected = contract.render_ts()
    actual = contract.TYPES_PATH.read_text(encoding="utf-8")
    assert actual == expected, "web/lib/types.ts is stale: run `make contract`"


def test_contract_generation_is_deterministic() -> None:
    assert contract.schema_text() == contract.schema_text()
    assert contract.render_ts() == contract.render_ts()


def test_derived_states_are_recomputed_not_trusted(sample_result: dict[str, Any]) -> None:
    sample_result["spec"]["initial_state"] = {"e1": {"translateY": {"kind": "px", "number": 99}}}
    env = ResultEnvelope.model_validate(sample_result)
    assert env.spec.initial_state["e1"]["translateY"].number == 0.0  # type: ignore[union-attr]
    assert env.spec.active_state["e2"]["scale"].number == pytest.approx(1.06)  # type: ignore[union-attr]


def test_export_without_samples_still_valid(sample_result: dict[str, Any]) -> None:
    """The JSON tab export shape (PLAN §9.5) is producible from the model."""
    spec = MotionSpec.model_validate(sample_result["spec"])
    dumped = spec.model_dump(mode="json", exclude={"transitions": {"__all__": {"samples"}}})
    assert all("samples" not in t for t in dumped["transitions"])
    assert dumped["transitions"][0]["from"] == {"kind": "px", "number": 0.0}


# --------------------------------------------------------------------------------------------
# IR integrity rules
# --------------------------------------------------------------------------------------------


def _spec(sample: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(sample["spec"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s["transitions"][0].update(element_id="e9"), "unknown element"),
        (lambda s: s["transitions"][0].update(segment_id="rt_in"), "unknown segment"),
        (lambda s: s["transitions"][0].update(to={"kind": "color", "color": "#FFFFFF"}), "needs"),
        (lambda s: s["transitions"][0].update(delta=-4.0), "delta"),
        (lambda s: s["transitions"][1].update(delta=1.0), "delta must be null"),
        (lambda s: s["transitions"][0]["confidence"].update(band="low"), "band"),
        (lambda s: s["transitions"][0]["easing"].update(cubic_bezier=[0.1, 0, 0.58, 1]), "keyword"),
        (lambda s: s["transitions"][0]["easing"].update(cubic_bezier=[1.2, 0, 0.5, 1]), "x1"),
        (lambda s: s["transitions"][1].update(id="t1"), "duplicate transition"),
        (lambda s: s["elements"][1].update(parent_id="e7"), "parent_id"),
        (lambda s: s["interaction"].update(direction="round_trip"), "direction"),
        (lambda s: s["interaction"].update(target_element_id="e8"), "target_element_id"),
        (lambda s: s["relationships"][0].update(transition_ids=["t1", "t99"]), "unknown ids"),
        (lambda s: s["segments"][1].update(kind="forward"), "kind"),
        (lambda s: s["warnings"][0].update(code="made_up_code"), "code"),
        (lambda s: s.update(surprise=True), "surprise"),
        (lambda s: s["transitions"][3]["to"].update(color="#5252ff"), "color"),
        (lambda s: s.update(job_id="not-a-job-id"), "job_id"),
    ],
)
def test_ir_rejects_invalid(sample_result: dict[str, Any], mutate: Any, match: str) -> None:
    spec = _spec(sample_result)
    mutate(spec)
    with pytest.raises(ValidationError, match=match):
        MotionSpec.model_validate(spec)


def test_envelope_job_id_must_match(sample_result: dict[str, Any]) -> None:
    sample_result["job_id"] = "0" * 32
    with pytest.raises(ValidationError, match="spec.job_id"):
        ResultEnvelope.model_validate(sample_result)


def test_band_thresholds() -> None:
    assert band_for(0.8) == "high"
    assert band_for(0.79) == "medium"
    assert band_for(0.5) == "medium"
    assert band_for(0.49) == "low"


def test_keyframe_whitelist(sample_result: dict[str, Any]) -> None:
    for kf in sample_result["artifacts"]["keyframes"]:
        assert KEYFRAME_NAME_RE.match(kf["name"])
    for bad in ("../result.json", "state_a.PNG", "el_x.png", "debug/mask.png", "state_c.png"):
        assert not KEYFRAME_NAME_RE.match(bad)


# --------------------------------------------------------------------------------------------
# other frozen P0 modules
# --------------------------------------------------------------------------------------------


def test_stage_windows_are_contiguous() -> None:
    assert STAGE_ORDER[0] == Stage.QUEUED and STAGE_ORDER[-1] == Stage.DONE
    assert set(STAGE_LABELS) == set(Stage) == set(STAGE_WINDOWS)
    prev_end = 0.0
    for stage in STAGE_ORDER:
        start, end = STAGE_WINDOWS[stage]
        assert start == pytest.approx(prev_end) and end >= start
        prev_end = end
    assert prev_end == 1.0
    assert overall_progress(Stage.SCANNING, 0.5) == pytest.approx(0.21)
    assert overall_progress(Stage.GENERATING, 2.0) == 1.0


def test_error_codes() -> None:
    assert set(ERROR_HTTP_STATUS) == set(ErrorCode)
    err = PipelineError(ErrorCode.FILE_TOO_LARGE)
    assert err.http_status == 413
    assert err.to_body()["error"]["code"] == "file_too_large"
    assert PipelineError("too_long", "custom").message == "custom"


def test_params_defaults_match_plan() -> None:
    p = MeasureParams()
    assert p == DEFAULT_PARAMS
    assert (p.scan.fps, p.scan.width, p.scan.thumb_width) == (30, 960, 240)
    assert p.scan.cursor_max_css == 56.0
    assert (p.regions.change_threshold, p.regions.max_elements) == (14, 8)
    assert (p.decode.max_fps, p.decode.max_window_frames) == (60.0, 150)
    assert p.confidence.cap_shadow == 0.45
    with pytest.raises(AttributeError):
        p.scan.fps = 60  # type: ignore[misc]  # frozen


def test_render_all_contract(sample_result: dict[str, Any]) -> None:
    """``render_all`` either renders all 4 tabs or reports missing generators cleanly."""
    spec = MotionSpec.model_validate(sample_result["spec"])
    try:
        out = render_all(spec)
    except NotImplementedError:
        return  # Track G not landed yet
    assert isinstance(out, Outputs)
    assert set(out.model_dump()) == {"technical", "llm_prompt", "css", "json"}
