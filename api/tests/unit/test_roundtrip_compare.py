"""Round-trip compare logic (PLAN-continuous §14): pure, no browser, no video.

The contract fixtures are the source IRs; replicas are perturbed copies.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import load_sample_continuous_result, load_sample_result

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_roundtrip as cr  # noqa: E402


def status(res: cr.Result) -> dict[str, str]:
    return {r.criterion: r.status for r in res.rows}


def spec(env: dict[str, Any]) -> dict[str, Any]:
    return env["spec"]


@pytest.fixture
def cont() -> dict[str, Any]:
    return load_sample_continuous_result()


@pytest.fixture
def trans() -> dict[str, Any]:
    return load_sample_result()


# ---- continuous ------------------------------------------------------------------------------


def test_identical_continuous_passes_every_row(cont):
    res = cr.compare(cont, copy.deepcopy(cont))
    st = status(res)
    assert res.ok, cr.format_table(res)
    for crit in ("mode", "axis", "phase sequence", "autoplay.direction", "autoplay.speed",
                 "inertia.tau", "inertia.mode", "resume.delay", "resume.ramp", "pause.decel",
                 "paused.durations", "snap.presence", "scroller.box", "card.size", "pitch", "gap",
                 "colour.page_background", "colour.scroller_background",
                 "colour.card_background", "colour.card_text"):  # fmt: skip
        assert st[crit] == cr.PASS, crit


def test_autoplay_speed_tolerance_and_direction(cont):
    rep = copy.deepcopy(cont)
    ap = spec(rep)["continuous"]["autoplay"]
    ap["speed_px_s"]["value"] *= 1.04
    assert status(cr.compare(cont, rep))["autoplay.speed"] == cr.PASS
    ap["speed_px_s"]["value"] = spec(cont)["continuous"]["autoplay"]["speed_px_s"]["value"] * 1.06
    assert status(cr.compare(cont, rep))["autoplay.speed"] == cr.FAIL
    ap["direction"] = "left"
    assert status(cr.compare(cont, rep))["autoplay.direction"] == cr.FAIL


def test_missing_autoplay_on_replica_fails(cont):
    rep = copy.deepcopy(cont)
    spec(rep)["continuous"]["autoplay"] = None
    st = status(cr.compare(cont, rep))
    assert st["autoplay.speed"] == cr.FAIL
    assert st["autoplay.direction"] == cr.FAIL


def test_no_autoplay_on_source_is_na_not_pass(cont):
    src = copy.deepcopy(cont)
    spec(src)["continuous"]["autoplay"] = None
    st = status(cr.compare(src, cont))
    assert st["autoplay.speed"] == cr.NA
    assert st["autoplay.direction"] == cr.NA


@pytest.mark.parametrize(("factor", "want"), [(1.19, cr.PASS), (0.81, cr.PASS), (1.21, cr.FAIL),
                                              (2.0, cr.FAIL), (0.5, cr.FAIL)])  # fmt: skip
def test_inertia_tau_tolerance(cont, factor, want):
    rep = copy.deepcopy(cont)
    spec(rep)["continuous"]["behavior"]["inertia"]["tau_ms"]["value"] *= factor
    assert status(cr.compare(cont, rep))["inertia.tau"] == want


def test_inertia_mode(cont):
    rep = copy.deepcopy(cont)
    ph = next(p for p in spec(rep)["continuous"]["phases"] if p["kind"] == "inertia")
    ph["fit"]["v_inf_px_s"] = 39.0
    assert cr.inertia_mode(ph) == "decay-towards-autoplay"
    assert status(cr.compare(cont, rep))["inertia.mode"] == cr.FAIL
    ph["fit"] = {"model": "tween", "distance_px": {"value": 100.0, "confidence": ph["fit"][
        "tau_ms"]["confidence"]}}  # fmt: skip
    assert cr.inertia_mode(ph) == "glide"


def test_no_inertia_on_source_is_na(cont):
    src = copy.deepcopy(cont)
    spec(src)["continuous"]["behavior"]["inertia"] = None
    for ph in spec(src)["continuous"]["phases"]:
        if ph["kind"] == "inertia":
            ph["kind"] = "drag"
            ph["fit"] = None
    st = status(cr.compare(src, cont))
    assert st["inertia.tau"] == cr.NA
    assert st["inertia.mode"] == cr.NA


def test_missing_resume_fails_and_sequence_differs(cont):
    rep = copy.deepcopy(cont)
    c = spec(rep)["continuous"]
    c["behavior"]["resume"] = None
    c["phases"] = [p for p in c["phases"] if p["kind"] not in ("resume", "autoplay")
                   or p["start_ms"] < 1000]  # fmt: skip
    st = status(cr.compare(cont, rep))
    assert st["resume.delay"] == cr.FAIL
    assert st["resume.ramp"] == cr.FAIL
    assert st["phase sequence"] == cr.FAIL


def test_resume_tolerance_100ms(cont):
    """Delay ±100 ms; the ramp by the time it takes to reach 90 % of its speed, ±100 ms."""
    rep = copy.deepcopy(cont)
    r = spec(rep)["continuous"]["behavior"]["resume"]
    src = spec(cont)["continuous"]["behavior"]["resume"]
    r["delay_after_rest_ms"]["value"] += 99
    reach = cr.ramp_reach_ms(src["ramp_ms"]["value"], src["easing"])
    # a duration change that moves the 90 % point by just over 100 ms fails
    scale = (reach - 101.0) / reach
    r["ramp_ms"]["value"] = src["ramp_ms"]["value"] * scale
    st = status(cr.compare(cont, rep))
    assert st["resume.delay"] == cr.PASS
    assert st["resume.ramp"] == cr.FAIL
    r["ramp_ms"]["value"] = src["ramp_ms"]["value"] * (reach - 99.0) / reach
    assert status(cr.compare(cont, rep))["resume.ramp"] == cr.PASS


def test_ramp_compared_by_reach_not_by_its_end():
    """An easeOutQuart over 1077 ms and a material-decelerate over 815 ms reach 90 % of their
    speed within ≈25 ms of each other: the same motion, two fits (acceptance run)."""
    material = {"cubic_bezier": [0.0, 0.0, 0.2, 1.0]}
    quart = {"cubic_bezier": [0.165, 0.84, 0.44, 1.0]}
    a, b = cr.ramp_reach_ms(815.0, material), cr.ramp_reach_ms(1077.0, quart)
    assert abs(a - b) < 30.0
    assert cr.ramp_reach_ms(800.0, None) == pytest.approx(720.0)  # linear when no easing


def test_pause_decel_and_instant_press(cont):
    rep = copy.deepcopy(cont)
    spec(rep)["continuous"]["behavior"]["pause"]["decel_ms"]["value"] += 150
    assert status(cr.compare(cont, rep))["pause.decel"] == cr.FAIL
    src = copy.deepcopy(cont)
    spec(src)["continuous"]["behavior"]["pause"]["decel_ms"] = None
    row = next(r for r in cr.compare(src, cont).rows if r.criterion == "pause.decel")
    assert row.status == cr.NA and "at once" in row.detail


def test_short_phases_merge_into_the_sequence(cont):
    """A sub-2-frame phase on one side only does not break the sequence (§14)."""
    rep = copy.deepcopy(cont)
    phases = spec(rep)["continuous"]["phases"]
    a = phases[0]  # autoplay 0..1500
    split = dict(copy.deepcopy(a), id="px", start_ms=700, end_ms=725, kind="unknown", fit=None)
    a2 = dict(copy.deepcopy(a), start_ms=725)
    a["end_ms"] = 700
    phases[0:1] = [a, split, a2]
    for i, p in enumerate(phases, 1):
        p["id"] = f"p{i}"
    assert [m["kind"] for m in cr.merged_phases(spec(rep))][:2] == ["autoplay", "decelerate"]
    assert status(cr.compare(cont, rep))["phase sequence"] == cr.PASS
    phases[1]["end_ms"] = 760  # 60 ms ≈ 3 frames: a real phase now
    phases[2]["start_ms"] = 760
    assert status(cr.compare(cont, rep))["phase sequence"] == cr.FAIL


def test_snap_presence(cont):
    rep = copy.deepcopy(cont)
    spec(rep)["continuous"]["behavior"]["snap"] = None
    assert status(cr.compare(cont, rep))["snap.presence"] == cr.FAIL


def test_geometry_rows(cont):
    rep = copy.deepcopy(cont)
    c = spec(rep)["continuous"]
    c["region"]["w"] += 2
    c["pitch_px"]["value"] += 2.5
    st = status(cr.compare(cont, rep))
    assert st["scroller.box"] == cr.PASS
    assert st["pitch"] == cr.FAIL
    c["gap_px"] = None
    assert status(cr.compare(cont, rep))["gap"] == cr.NA  # measured on both only
    card = next(e for e in spec(rep)["elements"] if e["role"] == "card")
    card["bbox_initial"]["w"] += 3
    assert status(cr.compare(cont, rep))["card.size"] == cr.FAIL


def test_colour_rows(cont):
    rep = copy.deepcopy(cont)
    els = {e["id"]: e for e in spec(rep)["elements"]}
    els["e1"]["static"]["background_color"]["value"] = "#9CA3AF"
    del els["e3"]["static"]["text_color"]
    st = status(cr.compare(cont, rep))
    assert st["colour.scroller_background"] == cr.FAIL
    assert st["colour.card_text"] == cr.NA


def test_delta_e_matches_known_values():
    assert cr.delta_e76("#FFFFFF", "#FFFFFF") == 0
    assert cr.delta_e76("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.01)
    assert cr.delta_e76("#F3F4F6", "#F1F4F3") < 3


def test_mode_mismatch_fails(cont, trans):
    res = cr.compare(cont, trans)
    st = status(res)
    assert st["mode"] == cr.FAIL
    assert not res.ok


def test_replica_error_body_fails(cont):
    res = cr.compare(cont, {"error": {"code": "no_motion_detected", "message": "x"}})
    assert not res.ok and res.rows[0].status == cr.FAIL


def test_source_error_body_is_not_a_pass(cont):
    res = cr.compare({"error": {"code": "no_motion_detected"}}, cont)
    assert not res.ok


# ---- transition ------------------------------------------------------------------------------


def test_identical_transition_passes(trans):
    res = cr.compare(trans, copy.deepcopy(trans))
    assert res.ok, cr.format_table(res)
    st = status(res)
    assert st["fwd e1(card).box-shadow duration"] == cr.NA  # shadow: direction only
    assert st["fwd e1(card).box-shadow value"] == cr.PASS


def _tr(env, tid):
    return next(t for t in spec(env)["transitions"] if t["id"] == tid)


def test_transition_duration_tolerance(trans):
    rep = copy.deepcopy(trans)
    _tr(rep, "t1")["duration_ms"] = 280 + 28  # 10 % of 280 = 28 > 20
    _tr(rep, "t7")["duration_ms"] = 220 + 23  # tolerance 22
    st = status(cr.compare(trans, rep))
    assert st["fwd e1(card).translateY duration"] == cr.PASS
    assert st["rev e1(card).translateY duration"] == cr.FAIL


def test_transition_values(trans):
    rep = copy.deepcopy(trans)
    _tr(rep, "t1")["to"]["number"] += 0.6
    _tr(rep, "t3")["to"]["number"] += 0.005
    st = status(cr.compare(trans, rep))
    assert st["fwd e1(card).translateY value"] == cr.FAIL
    assert st["fwd e2(image).scale value"] == cr.PASS
    rep2 = copy.deepcopy(trans)
    t2 = _tr(rep2, "t2")
    t2["from"], t2["to"] = t2["to"], t2["from"]
    assert status(cr.compare(trans, rep2))["fwd e1(card).box-shadow value"] == cr.FAIL


def test_easing_family_and_equivalence(trans):
    rep = copy.deepcopy(trans)
    e = _tr(rep, "t1")["easing"]
    e.update(family="ease-in", nearest_named="ease-in", keyword="ease-in",
             cubic_bezier=[0.42, 0.0, 1.0, 1.0])  # fmt: skip
    assert status(cr.compare(trans, rep))["fwd e1(card).translateY easing"] == cr.FAIL
    rep = copy.deepcopy(trans)
    _tr(rep, "t7")["easing"].update(family="ease", nearest_named="ease", keyword="ease",
                                    cubic_bezier=[0.25, 0.1, 0.25, 1.0])  # fmt: skip
    assert status(cr.compare(trans, rep))["rev e1(card).translateY easing"] == cr.PASS


def test_uncertain_source_easing_is_na(trans):
    src = copy.deepcopy(trans)
    _tr(src, "t1")["confidence"]["easing"] = 0.2
    rep = copy.deepcopy(trans)
    _tr(rep, "t1")["easing"]["family"] = "linear"
    row = next(r for r in cr.compare(src, rep).rows
               if r.criterion == "fwd e1(card).translateY easing")  # fmt: skip
    assert row.status == cr.NA and "uncertain" in row.detail


def test_missing_transition_and_segment_fail(trans):
    rep = copy.deepcopy(trans)
    s = spec(rep)
    s["transitions"] = [t for t in s["transitions"] if t["segment_id"] == "fwd"]
    s["segments"] = [sg for sg in s["segments"] if sg["id"] == "fwd"]
    res = cr.compare(trans, rep)
    st = status(res)
    assert st["segments"] == cr.FAIL
    assert st["rev e1(card).translateY present"] == cr.FAIL
    assert not res.ok


def test_extra_replica_transition_is_info(trans):
    rep = copy.deepcopy(trans)
    extra = copy.deepcopy(_tr(rep, "t1"))
    extra["id"], extra["property"] = "t99", "translateX"
    spec(rep)["transitions"].append(extra)
    res = cr.compare(trans, rep)
    assert res.ok
    assert any(r.status == cr.INFO and r.criterion.startswith("extra") for r in res.rows)


def test_cli_exit_codes(tmp_path, cont):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(cont))
    rep = copy.deepcopy(cont)
    spec(rep)["continuous"]["behavior"]["inertia"]["tau_ms"]["value"] *= 2
    b.write_text(json.dumps(rep))
    out = tmp_path / "cmp.json"
    assert cr.main([str(a), str(a), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    assert cr.main([str(a), str(b), "--no-validate"]) == 1
    assert cr.main([str(a), str(tmp_path / "missing.json")]) == 2


# --------------------------------------------------------------------------------------------
# Card scaling (acceptance run): like-for-like geometry
# --------------------------------------------------------------------------------------------

SCALE = {"model": "quadratic", "origin": "scroller_center", "reference_distance_px": 261.3,
         "scale_at_reference": 1.1694, "mean_scale": 1.1144,
         "confidence": {"value": 0.6, "band": "medium"}}  # fmt: skip


def _row(res: cr.Result, name: str) -> cr.Row:
    return next(r for r in res.rows if r.criterion == name)


def test_card_scale_compared_at_the_source_reference(cont):
    src = copy.deepcopy(cont)
    spec(src)["continuous"]["card_scale"] = dict(SCALE)
    rep = copy.deepcopy(src)
    # another reference distance, same curve: x1.1694 at 261.3 px == x1.2 at 286.3 px (quadratic)
    k = (1.1694 - 1.0) / 261.3**2
    spec(rep)["continuous"]["card_scale"] = dict(
        SCALE, reference_distance_px=286.3, scale_at_reference=round(1.0 + k * 286.3**2, 4)
    )
    assert _row(cr.compare(src, rep), "card.scale").status == cr.PASS
    spec(rep)["continuous"]["card_scale"]["scale_at_reference"] = 1.30
    assert _row(cr.compare(src, rep), "card.scale").status == cr.FAIL


def test_scaling_on_one_side_only_fails_and_skips_unlike_geometry(cont):
    src = copy.deepcopy(cont)
    spec(src)["continuous"]["card_scale"] = dict(SCALE)
    rep = copy.deepcopy(cont)  # rigid replica: its card / pitch are uniform, not the centre's
    res = cr.compare(src, rep)
    assert _row(res, "card.scale").status == cr.FAIL
    for name in ("card.size", "pitch"):
        row = _row(res, name)
        assert row.status == cr.NA and "only on the source" in row.detail
    assert _row(res, "gap").status == cr.PASS  # the gap is the same quantity either way
    # rigid on both: nothing to judge
    assert _row(cr.compare(cont, copy.deepcopy(cont)), "card.scale").status == cr.NA
