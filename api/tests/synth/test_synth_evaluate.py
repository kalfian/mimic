"""Evaluator self-test: the oracle IR (truth as MotionSpec) must pass; perturbations must fail."""

from __future__ import annotations

import pytest

from app.models.ir import MotionSpec, PxValue

from . import scenarios as S
from .evaluate import compare, format_report, load_ir, oracle_spec

POSITIVE = [e.name for e in S.SCENARIOS if not e.name.startswith("negative_")]


@pytest.mark.parametrize("name", POSITIVE)
def test_oracle_is_valid_ir_and_passes(name: str) -> None:
    truth = S.build_truth(S.get(name))
    spec = oracle_spec(truth)
    # survives a JSON round trip through the frozen contract
    spec = MotionSpec.model_validate_json(spec.model_dump_json())
    rep = compare(truth, spec)
    assert rep.ok, format_report(rep)
    assert len(rep.rows) == len(truth.transitions)


def test_perturbed_result_fails() -> None:
    truth = S.build_truth(S.get("card_hover_translate"))
    data = oracle_spec(truth).model_dump(mode="json")
    t = data["transitions"][0]
    t["to"] = PxValue(number=-6.0).model_dump()
    t["delta"] = -6.0
    t["duration_ms"] += 80
    rep = compare(truth, MotionSpec.model_validate(data))
    row = rep.rows[0]
    assert not rep.ok and row.value_ok is False and row.dur_ok is False and row.rmse_ok is False


def test_negative_expects_error_code() -> None:
    truth = S.build_truth(S.get("negative_scroll"))
    spec, code = load_ir({"error": {"code": "unsupported_motion", "message": "x"}})
    assert compare(truth, spec, code).ok
    assert not compare(truth, None, "no_motion_detected").ok


# ---- PLAN-continuous --------------------------------------------------------------------------


def _failed(rep) -> set[str]:
    return {what for what, ok, _ in rep.checks if ok is False}


def test_oracle_covers_every_continuous_scenario() -> None:
    names = {e.name for e in S.CONTINUOUS_SCENARIOS if not e.name.startswith("negative_")}
    assert names <= set(POSITIVE)
    spec = oracle_spec(S.build_truth(S.get("carousel_drag_inertia")))
    assert spec.mode == "continuous" and spec.continuous is not None
    assert [p.kind for p in spec.continuous.phases][:3] == ["autoplay", "drag", "inertia"]
    assert spec.scene is not None and spec.elements[0].static.background_color is not None


def test_perturbed_continuous_result_fails() -> None:
    truth = S.build_truth(S.get("carousel_drag_inertia"))
    data = oracle_spec(truth).model_dump(mode="json")
    c = data["continuous"]
    c["autoplay"]["speed_px_s"]["value"] = 44.0  # +10 %
    c["autoplay"]["velocity_px_s"] = 44.0
    inertia = next(p for p in c["phases"] if p["kind"] == "inertia")
    inertia["fit"]["tau_ms"]["value"] = 300.0  # +50 %
    c["behavior"]["inertia"]["tau_ms"]["value"] = 300.0
    drag = next(p for p in c["phases"] if p["kind"] == "drag")
    drag["start_ms"] += 60  # sharp boundary off by 60 ms
    c["behavior"]["resume"]["delay_after_rest_ms"]["value"] = 1200.0
    rep = compare(truth, MotionSpec.model_validate(data))
    assert not rep.ok
    assert {"autoplay.speed", "phase 2 drag start", "phase 3 inertia tau",
            "resume.delay_after_rest"} <= _failed(rep)  # fmt: skip


def test_wrong_sequence_false_snap_and_false_period_fail() -> None:
    truth = S.build_truth(S.get("marquee_autoplay"))
    data = oracle_spec(truth).model_dump(mode="json")
    loop = data["continuous"]["autoplay"]["loop"]
    loop.update(observed=True, period_px={"value": 1500.0, "confidence": {"value": 0.9,
                "band": "high"}}, duration_ms={"value": 25000.0, "confidence": {"value": 0.9,
                "band": "high"}})  # fmt: skip
    rep = compare(truth, MotionSpec.model_validate(data))
    assert "loop.not_observed" in _failed(rep)

    truth = S.build_truth(S.get("carousel_drag_inertia"))
    data = oracle_spec(truth).model_dump(mode="json")
    c = data["continuous"]
    c["behavior"]["snap"] = {"kind": "abrupt_ambiguous", "step_px": None, "duration_ms": None,
                             "easing": None, "overshoot": False}  # fmt: skip
    c["phases"][2].update(kind="stop", fit=None)  # inertia relabelled as an abrupt stop
    c["behavior"]["inertia"]["instances"] = 1
    rep = compare(truth, MotionSpec.model_validate(data))
    assert {"phase sequence", "snap"} <= _failed(rep)


def test_card_scale_required_on_c13_and_false_on_rigid_cards() -> None:
    """P2c: C13's cards scale (×1 → ×1.165): a missing or wrong scale fails; a rigid
    scenario reporting one fails too."""
    truth = S.build_truth(S.get("carousel_dark_edge_zoom"))
    data = oracle_spec(truth).model_dump(mode="json")
    cs = data["continuous"]["card_scale"]
    assert cs is not None and compare(truth, MotionSpec.model_validate(data)).ok
    data["continuous"]["card_scale"] = None
    assert "card_scale" in _failed(compare(truth, MotionSpec.model_validate(data)))
    half = data["continuous"]["region"]["w"] / 2.0
    wrong = dict(cs, scale_at_reference=round(cs["scale_at_reference"] + 0.05, 4))
    wrong["mean_scale"] = round(
        1.0 + (wrong["scale_at_reference"] - 1.0) * (half / wrong["reference_distance_px"]) ** 2
        / 3.0, 4)  # fmt: skip
    data["continuous"]["card_scale"] = wrong
    assert {"card_scale.at_ref", "card_scale.mean"} <= _failed(
        compare(truth, MotionSpec.model_validate(data))
    )

    rigid = S.build_truth(S.get("carousel_drag_inertia"))
    data = oracle_spec(rigid).model_dump(mode="json")
    half = data["continuous"]["region"]["w"] / 2.0
    data["continuous"]["card_scale"] = dict(cs, reference_distance_px=round(0.7 * half, 1),
                                            mean_scale=round(1.0 + (cs["scale_at_reference"] - 1.0)
                                                             / 0.49 / 3.0, 4))  # fmt: skip
    assert "card_scale" in _failed(compare(rigid, MotionSpec.model_validate(data)))


def test_transition_spec_fails_a_continuous_truth() -> None:
    truth = S.build_truth(S.get("marquee_hover_pause"))
    s1 = oracle_spec(S.build_truth(S.get("card_hover_translate")))
    rep = compare(truth, s1)
    assert not rep.ok and "mode" in _failed(rep)


def test_c11_needs_transition_mode_and_the_ambient_warning() -> None:
    truth = S.build_truth(S.get("hover_with_ambient_marquee"))
    data = oracle_spec(truth).model_dump(mode="json")
    assert compare(truth, MotionSpec.model_validate(data)).ok
    data["warnings"] = [w for w in data["warnings"] if w["code"] != "ambient_motion_masked"]
    rep = compare(truth, MotionSpec.model_validate(data))
    assert "warning ambient_motion_masked" in _failed(rep)


def test_n1_expects_continuous_motion_unsupported() -> None:
    truth = S.build_truth(S.get("negative_ambient_pulse"))
    assert compare(truth, None, "continuous_motion_unsupported").ok
    assert not compare(truth, None, "no_motion_detected").ok


def test_continuous_suite_enabled_since_p2() -> None:
    from . import evaluate as E

    assert E.CONTINUOUS_SUITE_ENABLED is True
    for e in S.SCENARIOS:
        assert E.is_pending(S.build_truth(e)) is False
    # the pending machinery (and its summary line) is kept for future suites
    summary = E.summarize(
        [compare(S.build_truth(S.get("card_hover_translate")),
                 oracle_spec(S.build_truth(S.get("card_hover_translate"))))],
        pending=["marquee_autoplay"],
    )  # fmt: skip
    text = E.format_summary(summary)
    assert summary.ok and "1 pending until P2" in text and "SUITE: PASS" in text
