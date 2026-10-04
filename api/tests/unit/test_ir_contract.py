"""Phase 0 contract tests (PLAN P0 + PLAN-continuous P0).

* ``docs/contract/sample-result.json`` (transition) and ``sample-continuous-result.json``
  (continuous) validate against the Pydantic models AND the generated JSON Schema, and are
  canonical (model round-trip reproduces them byte-for-byte as JSON).
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
from app.core.errors import DEFAULT_MESSAGES, ERROR_HTTP_STATUS, ErrorCode, PipelineError
from app.core.stages import STAGE_LABELS, STAGE_ORDER, STAGE_WINDOWS, Stage, overall_progress
from app.generate import (
    CONTINUOUS_GENERATOR_MODULES,
    GENERATOR_MODULES,
    Outputs,
    export_json,
    render_all,
)
from app.interpret.schema import INTERACTION_TYPES, ROLES
from app.models.ir import (
    CONTINUOUS_SAMPLE_MAX,
    PHASE_KINDS,
    SCHEMA_VERSION,
    ElementStatic,
    MotionSpec,
    band_for,
)
from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams, MeasureParams

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
    """Transition mode renders exactly the 4 tabs; ``js`` stays null and is not serialized."""
    spec = MotionSpec.model_validate(sample_result["spec"])
    out = render_all(spec)
    assert isinstance(out, Outputs)
    assert out.js is None
    assert set(out.model_dump()) == {"technical", "llm_prompt", "css", "json"}
    assert "js" not in json.loads(out.model_dump_json())


def test_sample_result_outputs_match_generators(sample_result: dict[str, Any]) -> None:
    """The fixture's outputs are what the generators produce for its spec (kept in sync)."""
    spec = MotionSpec.model_validate(sample_result["spec"])
    assert render_all(spec).model_dump() == sample_result["outputs"]


# --------------------------------------------------------------------------------------------
# IR 0.2: mode, backward compatibility, appearance (PLAN-continuous §5, §13)
# --------------------------------------------------------------------------------------------


def test_transition_fixture_is_current_version(sample_result: dict[str, Any]) -> None:
    spec = MotionSpec.model_validate(sample_result["spec"])
    assert spec.schema_version == SCHEMA_VERSION == "0.2"
    assert spec.mode == "transition" and spec.continuous is None and spec.scene is None
    assert list(sample_result["spec"])[:4] == ["schema_version", "job_id", "meta", "mode"]


def test_stored_v01_spec_still_loads(sample_result: dict[str, Any]) -> None:
    """Results written before IR 0.2 (no ``mode``) load as transition specs for re-runs."""
    spec = _spec(sample_result)
    spec["schema_version"] = "0.1"
    del spec["mode"]
    loaded = MotionSpec.model_validate(spec)
    assert loaded.mode == "transition" and loaded.schema_version == "0.1"
    dumped = loaded.model_dump(mode="json")
    assert dumped["mode"] == "transition"
    assert {k: v for k, v in dumped.items() if k not in ("mode", "schema_version")} == {
        k: v for k, v in sample_result["spec"].items() if k not in ("mode", "schema_version")
    }


def test_new_optional_fields_are_omitted_when_null(sample_result: dict[str, Any]) -> None:
    """A transition spec differs from 0.1 only in ``schema_version`` and ``mode`` (§5)."""
    dumped = MotionSpec.model_validate(sample_result["spec"]).model_dump(mode="json")
    assert "continuous" not in dumped and "scene" not in dumped
    for el in dumped["elements"]:
        assert set(el["static"]) == {"border_radius_px", "shadow"}


def test_appearance_fields_round_trip(sample_result: dict[str, Any]) -> None:
    spec = _spec(sample_result)
    conf = {"value": 0.7, "band": "medium"}
    spec["scene"] = {
        "viewport_css": {"w": 1440.0, "h": 900.0},
        "page_background": {"value": "#F7F7F8", "confidence": conf},
    }
    spec["elements"][0]["static"]["background_color"] = {"value": "#FFFFFF", "confidence": conf}
    spec["elements"][2]["static"]["text_color"] = {"value": "#111111", "confidence": conf}
    spec["elements"][2]["static"]["font_size_px"] = {"value": 18.0, "confidence": conf}
    loaded = MotionSpec.model_validate(spec)
    assert loaded.scene is not None and loaded.scene.viewport_css.w == 1440.0
    assert MotionSpec.model_validate(loaded.model_dump(mode="json")) == loaded
    assert loaded.model_dump(mode="json")["elements"][1]["static"] == {
        "border_radius_px": None,
        "shadow": None,
    }


@pytest.mark.parametrize(
    ("static", "match"),
    [
        ({"font_size_px": {"value": 16, "confidence": {"value": 0.9, "band": "high"}}}, "capped"),
        (
            {"text_color": {"value": "#abcdef", "confidence": {"value": 0.5, "band": "medium"}}},
            "should match pattern",
        ),
        ({"background_color": {"value": "#FFFFFF"}}, "confidence"),
    ],
)
def test_appearance_rejects_invalid(static: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ElementStatic.model_validate(static)


def test_scene_requires_positive_viewport(sample_result: dict[str, Any]) -> None:
    spec = _spec(sample_result)
    spec["scene"] = {"viewport_css": {"w": 0, "h": 800}, "page_background": None}
    with pytest.raises(ValidationError, match="greater than 0"):
        MotionSpec.model_validate(spec)


# --------------------------------------------------------------------------------------------
# Continuous mode fixture + validators (PLAN-continuous §5.2)
# --------------------------------------------------------------------------------------------


def test_continuous_fixture_validates(sample_continuous_result: dict[str, Any]) -> None:
    env = ResultEnvelope.model_validate(sample_continuous_result)
    spec = env.spec
    assert spec.mode == "continuous" and spec.continuous is not None
    c = spec.continuous
    assert spec.interaction.direction == "continuous" and spec.interaction.type == "drag"
    assert spec.segments == [] and spec.transitions == [] and spec.relationships == []
    assert [p.kind for p in c.phases] == [
        "autoplay", "decelerate", "drag", "inertia", "drag", "inertia", "drag", "inertia",
        "drag", "stop", "paused", "resume", "autoplay",
    ]  # fmt: skip
    assert c.autoplay is not None and c.autoplay.direction == "right"
    assert c.behavior.inertia is not None and c.behavior.inertia.tau_ms is not None
    assert c.behavior.inertia.tau_ms.value == 200
    assert c.behavior.resume is not None and c.behavior.resume.ramp_ms.value == 750
    assert c.behavior.snap is not None and c.behavior.snap.kind == "abrupt_ambiguous"
    assert 0 < len(c.samples) <= CONTINUOUS_SAMPLE_MAX
    assert spec.scene is not None and spec.scene.page_background is not None
    statics = {e.id: e.static for e in spec.elements}
    assert statics["e2"].background_color is not None
    assert statics["e3"].text_color is not None and statics["e3"].font_size_px is not None
    assert env.outputs.js is not None
    assert {kf.kind for kf in env.artifacts.keyframes} == {"phase"}
    codes = {w.code for w in spec.warnings}
    assert {"tracking_degraded", "loop_period_not_observed"} <= codes


def test_continuous_fixture_is_canonical(sample_continuous_result: dict[str, Any]) -> None:
    env = ResultEnvelope.model_validate(sample_continuous_result)
    assert env.model_dump(mode="json") == sample_continuous_result


def test_continuous_fixture_validates_against_json_schema(
    sample_continuous_result: dict[str, Any],
) -> None:
    schema = json.loads(contract.SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(sample_continuous_result), key=str)
    assert not errors, "\n".join(str(e) for e in errors[:5])


def test_continuous_fixture_json_output_is_export(sample_continuous_result: dict[str, Any]) -> None:
    """``outputs.json`` is the mode-agnostic export with the samples dropped (the text / js
    tabs are ``render_all`` output too: ``test_generators_continuous``)."""
    spec = MotionSpec.model_validate(sample_continuous_result["spec"])
    assert sample_continuous_result["outputs"]["json"] == export_json.render(spec)
    exported = json.loads(export_json.render(spec))
    assert "samples" not in exported["continuous"]
    assert exported["continuous"]["phases"][0]["id"] == "p1"


def test_render_all_dispatches_continuous(sample_continuous_result: dict[str, Any]) -> None:
    spec = MotionSpec.model_validate(sample_continuous_result["spec"])
    assert set(CONTINUOUS_GENERATOR_MODULES) == set(GENERATOR_MODULES) | {"js"}
    out = render_all(spec)
    assert out.js is not None
    assert set(out.model_dump()) == {"technical", "llm_prompt", "css", "json", "js"}


def _cspec(sample: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(sample["spec"])


def _set_phase_fit(s: dict[str, Any], i: int, fit: Any) -> None:
    s["continuous"]["phases"][i]["fit"] = fit


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        # mode <-> continuous <-> direction <-> type
        (lambda s: s.update(mode="transition"), "forbids continuous section"),
        (lambda s: s.update(continuous=None), "requires continuous section"),
        (lambda s: s["interaction"].update(direction="forward"), "direction"),
        (lambda s: s["interaction"].update(type="hover"), "interaction.type"),
        (lambda s: s.update(schema_version="0.1"), "schema_version"),
        (
            lambda s: s.update(
                segments=[
                    {
                        "id": "fwd",
                        "kind": "forward",
                        "start_ms": 0,
                        "end_ms": 1,
                        "onset_ms": 0,
                        "settle_ms": 1,
                    }
                ]
            ),
            "empty segments",
        ),  # fmt: skip
        # scroller element / target / duration
        (lambda s: s["elements"][0].update(kind="transform"), "kind 'scroller'"),
        (lambda s: s["continuous"].update(element_id="e9"), "kind 'scroller'"),
        (lambda s: s["interaction"].update(target_element_id="e2"), "target_element_id"),
        (lambda s: s["interaction"]["total_duration_ms"].update(forward=1), "analysed span"),
        (lambda s: s["interaction"]["total_duration_ms"].update(reverse=10), "analysed span"),
        (lambda s: s["interaction"].update(type="continuous"), "drag"),
        # phases
        (lambda s: s["continuous"]["phases"][1].update(id="p7"), "p1..pN"),
        (lambda s: s["continuous"]["phases"][1].update(start_ms=1400), "overlaps"),
        (lambda s: s["continuous"]["phases"][0].update(end_ms=0), "after start_ms"),
        (lambda s: s["continuous"]["phases"][-1].update(end_ms=9400), "within span_ms"),
        (lambda s: _set_phase_fit(s, 2, s["continuous"]["phases"][0]["fit"]), "not allowed"),
        (lambda s: _set_phase_fit(s, 0, None), "not allowed"),
        (lambda s: _set_phase_fit(s, 3, s["continuous"]["phases"][1]["fit"]), "not allowed"),
        (lambda s: s["continuous"]["phases"][0].update(kind="spin"), "kind"),
        (lambda s: s["continuous"].update(phases=[]), "at least 1"),
        # behaviour <-> phases
        (lambda s: s["continuous"]["behavior"]["drag"].update(count=3), "drag.count"),
        (lambda s: s["continuous"]["behavior"].update(inertia=None), "behavior.inertia"),
        (lambda s: s["continuous"]["behavior"]["inertia"].update(instances=2), "instances"),
        (lambda s: s["continuous"]["behavior"]["inertia"].update(tau_ms=None), "tau_ms"),
        (
            lambda s: s["continuous"]["behavior"]["inertia"].update(release_speed_min_px_s=2000),
            "release_speed_min",
        ),
        (lambda s: s["continuous"]["behavior"]["snap"].update(kind="grid"), "step_px"),
        # autoplay
        (lambda s: s["continuous"]["autoplay"].update(velocity_px_s=-39.0), "sign"),
        (lambda s: s["continuous"]["autoplay"].update(velocity_px_s=40.0), "magnitude|equal"),
        (lambda s: s["continuous"]["autoplay"].update(direction="down"), "not on x"),
        (lambda s: s["continuous"]["autoplay"]["loop"].update(observed=True), "observed"),
        (lambda s: s["continuous"]["autoplay"].update(easing="ease"), "linear"),
        # samples
        (lambda s: s["continuous"]["samples"].__setitem__(1, [0, 39.0, 0.0, 0.9]), "increasing"),
        (lambda s: s["continuous"]["samples"].__setitem__(0, [0, 39.0, 0.0, 1.5]), "less than"),
        (
            lambda s: s["continuous"].update(samples=[[i, 0.0, 0.0, 1.0] for i in range(901)]),
            "at most 900",
        ),
        (lambda s: s["continuous"].update(sign_convention="positive = left"), "sign_convention"),
    ],
)
def test_continuous_rejects_invalid(
    sample_continuous_result: dict[str, Any], mutate: Any, match: str
) -> None:
    spec = _cspec(sample_continuous_result)
    mutate(spec)
    with pytest.raises(ValidationError, match=match):
        MotionSpec.model_validate(spec)


def test_transition_rejects_continuous_only_values(sample_result: dict[str, Any]) -> None:
    for itype in ("continuous", "drag"):
        spec = _spec(sample_result)
        spec["interaction"]["type"] = itype
        with pytest.raises(ValidationError, match="forbids interaction.type"):
            MotionSpec.model_validate(spec)
    spec = _spec(sample_result)
    spec["segments"] = []
    spec["transitions"] = []
    spec["relationships"] = []
    with pytest.raises(ValidationError, match="at least one segment"):
        MotionSpec.model_validate(spec)


def test_loop_observed_requires_consistent_duration(
    sample_continuous_result: dict[str, Any],
) -> None:
    spec = _cspec(sample_continuous_result)
    conf = {"value": 0.85, "band": "high"}
    loop = {
        "observed": True,
        "period_px": {"value": 1728.0, "confidence": conf},
        "duration_ms": {"value": 44308.0, "confidence": conf},  # 1728 / 39 * 1000 = 44307.7
    }
    spec["continuous"]["autoplay"]["loop"] = loop
    assert MotionSpec.model_validate(spec).continuous.autoplay.loop.observed  # type: ignore[union-attr]
    loop["duration_ms"]["value"] = 40000.0
    with pytest.raises(ValidationError, match="period_px / speed"):
        MotionSpec.model_validate(spec)


def _card_scale(ref: float = 270.0, at_ref: float = 1.15, mean: float = 1.0990) -> dict[str, Any]:
    # fixture region 760 px wide: half 380 -> mean = 1 + 0.15 (380 / 270)² / 3 = 1.0990
    return {
        "model": "quadratic",
        "origin": "scroller_center",
        "reference_distance_px": ref,
        "scale_at_reference": at_ref,
        "mean_scale": mean,
        "confidence": {"value": 0.6, "band": "medium"},
    }


def test_card_scale_optional_and_validated(sample_continuous_result: dict[str, Any]) -> None:
    """P2c: ``continuous.card_scale`` is omitted when null; when set, the reference distance lies
    inside the region and ``mean_scale`` matches the model; the scale grows (> 1)."""
    spec = _cspec(sample_continuous_result)
    assert "card_scale" not in MotionSpec.model_validate(spec).model_dump(mode="json")["continuous"]
    spec["continuous"]["card_scale"] = _card_scale()
    loaded = MotionSpec.model_validate(spec)
    assert loaded.continuous is not None and loaded.continuous.card_scale is not None
    dumped = loaded.model_dump(mode="json")["continuous"]["card_scale"]
    assert dumped == _card_scale()
    for bad, match in (
        (_card_scale(mean=1.2), "mean_scale must equal"),
        (_card_scale(ref=400.0, mean=1.0846), "inside the region"),
        (_card_scale(at_ref=1.0, mean=1.0), "greater than 1"),
        (_card_scale(ref=0.0), "greater than 0"),
        ({**_card_scale(), "model": "linear"}, "quadratic"),
        ({**_card_scale(), "origin": "scroller_edge"}, "scroller_center"),
    ):
        spec["continuous"]["card_scale"] = bad
        with pytest.raises(ValidationError, match=match):
            MotionSpec.model_validate(spec)


def test_marquee_without_drag_is_valid(sample_continuous_result: dict[str, Any]) -> None:
    """Autoplay-only marquee: type ``continuous``, one autoplay phase, empty behaviour."""
    spec = _cspec(sample_continuous_result)
    c = spec["continuous"]
    c["phases"] = [{**c["phases"][0], "end_ms": 9300}]
    c["behavior"] = {"pause": None, "drag": None, "inertia": None, "snap": None, "resume": None}
    spec["interaction"].update(type="continuous", pattern="marquee")
    spec["interaction"]["trigger"].update(kind="autoplay")
    loaded = MotionSpec.model_validate(spec)
    assert loaded.continuous is not None and loaded.continuous.behavior.drag is None


def test_keyframe_whitelist_phase_names() -> None:
    for ok in ("phase_1.png", "phase_6.png", "phase_12.png"):
        assert KEYFRAME_NAME_RE.match(ok)
    for bad in ("phase_0.png", "phase_100.png", "phase_.png", "phase_1.jpg"):
        assert not KEYFRAME_NAME_RE.match(bad)


def test_continuous_vocabulary_exports() -> None:
    assert PHASE_KINDS[0] == "autoplay" and "unknown" in PHASE_KINDS and len(PHASE_KINDS) == 9
    ts = contract.render_ts()
    assert "export const PHASE_KINDS" in ts
    assert f"export const CONTINUOUS_SAMPLE_MAX = {CONTINUOUS_SAMPLE_MAX};" in ts
    assert 'export const SCHEMA_VERSION = "0.2" as const;' in ts


def test_interpreter_vocabulary_excludes_continuous_only_members() -> None:
    """Layer B must not put continuous-only values into a transition spec (Track I owns the
    continuous wording)."""
    assert "scroller" not in ROLES
    assert "continuous" not in INTERACTION_TYPES and "drag" not in INTERACTION_TYPES
    assert "card" in ROLES and "hover" in INTERACTION_TYPES


def test_continuous_error_code() -> None:
    err = PipelineError(ErrorCode.CONTINUOUS_MOTION_UNSUPPORTED)
    assert err.http_status == 422
    assert err.to_body()["error"]["code"] == "continuous_motion_unsupported"
    assert "scroller" in DEFAULT_MESSAGES[ErrorCode.CONTINUOUS_MOTION_UNSUPPORTED]


def test_continuous_params_defaults_match_plan() -> None:
    c = MeasureParams().continuous
    assert c == ContinuousParams()
    assert (c.cell_css, c.lead_window_s, c.ambient_lead_frac) == (16.0, 0.6, 0.5)
    assert (c.max_ambient_regions, c.page_scroll_area_frac) == (3, 0.6)
    assert (c.coherence_min_frac, c.coherence_axis_dominance) == (0.7, 5.0)
    assert (c.tau_min_s, c.tau_max_s, c.max_accel_px_s2) == (0.04, 2.0, 60_000.0)
    assert c.samples_max == CONTINUOUS_SAMPLE_MAX
    with pytest.raises(AttributeError):
        c.cell_css = 8.0  # type: ignore[misc]  # frozen


def test_envelope_js_only_in_continuous_mode(
    sample_result: dict[str, Any], sample_continuous_result: dict[str, Any]
) -> None:
    sample_result["outputs"]["js"] = "// driver"
    with pytest.raises(ValidationError, match="outputs.js"):
        ResultEnvelope.model_validate(sample_result)
    del sample_continuous_result["outputs"]["js"]
    with pytest.raises(ValidationError, match="outputs.js"):
        ResultEnvelope.model_validate(sample_continuous_result)
