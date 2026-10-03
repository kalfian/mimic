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
