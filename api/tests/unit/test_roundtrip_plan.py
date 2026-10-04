"""Round-trip input plan (PLAN-continuous §14): IR → timed mouse script. Pure, no browser."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from tests.conftest import load_sample_continuous_result, load_sample_result

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import roundtrip as rt  # noqa: E402


def _drag_moves(events, t0, t1):
    return [e for e in events if e["type"] == "move" and e["buttons"] == 1 and t0 < e["t_ms"] <= t1]


@pytest.mark.parametrize(
    ("distance", "v", "dur"),
    [(-400.0, -1100.0, 0.45), (300.0, 1000.0, 0.4), (400.0, 300.0, 0.5), (-200.0, 2500.0, 0.3)],
)
def test_drag_offsets_cover_distance_and_end_at_release_velocity(distance, v, dur):
    n = int(round(dur * 1000))
    ts = [i / 1000 for i in range(n + 1)]
    off = rt.drag_offsets(distance, v, dur, ts)
    assert off[0] == pytest.approx(0.0, abs=1e-9)
    assert off[-1] == pytest.approx(distance, abs=1e-6)
    v_end = (off[-1] - off[-11]) / 0.010  # last 10 ms
    assert v_end == pytest.approx(v, rel=0.01)


def test_feasible_drag_uses_the_synthetic_model():
    # synth C4 drag: 400 px in 450 ms released at 1100 px/s → smoothstep for a·T then constant
    ts = [i / 1000 for i in range(451)]
    off = rt.drag_offsets(-400.0, -1100.0, 0.45, ts)
    a = 2 * (1 - 400 / (1100 * 0.45))
    k = int(a * 450) + 5
    for i in range(k, 450):  # constant-velocity tail
        assert off[i + 1] - off[i] == pytest.approx(-1.1, abs=1e-6)


def test_continuous_plan_replays_each_drag():
    env = load_sample_continuous_result()
    plan = rt.build_plan(env)
    c = env["spec"]["continuous"]
    assert plan["viewport"] == {"w": 1280, "h": 800}
    assert plan["dpr"] == env["spec"]["source"]["pixel_ratio"]
    assert plan["frames"] == int(env["spec"]["source"]["duration_ms"] * 60 / 1000)
    ev = plan["events"]
    downs = [e for e in ev if e["type"] == "down"]
    ups = [e for e in ev if e["type"] == "up"]
    drags = [p for p in c["phases"] if p["kind"] == "drag"]
    assert len(downs) == len(ups) == len(drags)
    reg = c["region"]
    for d, dn, up in zip(drags, downs, ups, strict=True):
        assert abs(dn["t_ms"] - d["start_ms"]) <= 1000 / 60 / 2 + 1e-6
        assert abs(up["t_ms"] - d["end_ms"]) <= 1000 / 60 / 2 + 1e-6
        # press inside the scroller, travel = content displacement
        assert reg["x"] <= dn["x"] <= reg["x"] + reg["w"]
        assert reg["y"] <= dn["y"] <= reg["y"] + reg["h"]
        assert up["x"] - dn["x"] == pytest.approx(d["displacement_px"], abs=0.02)
        moves = _drag_moves(ev, dn["t_ms"], up["t_ms"])
        dt_s = (moves[-1]["t_ms"] - moves[-2]["t_ms"]) / 1000
        v_last = (moves[-1]["x"] - moves[-2]["x"]) / dt_s
        assert v_last == pytest.approx(d["v_end_px_s"], rel=0.05)
    # events are time-ordered and all on the frame grid (or the warm-up)
    ts = [e["t_ms"] for e in ev]
    assert ts == sorted(ts)
    for t in ts[1:]:
        k = round(t * 60 / 1000)
        assert t == pytest.approx(rt.frame_t(k), abs=1e-3)


def test_parked_pointer_is_outside_the_scroller():
    env = load_sample_continuous_result()
    plan = rt.build_plan(env)
    reg = env["spec"]["continuous"]["region"]
    first = plan["events"][0]
    assert first["t_ms"] < 0
    assert not (reg["x"] <= first["x"] <= reg["x"] + reg["w"]
                and reg["y"] <= first["y"] <= reg["y"] + reg["h"])  # fmt: skip


def test_hover_pause_is_replayed_as_enter_and_leave():
    env = load_sample_continuous_result()
    c = env["spec"]["continuous"]
    c["behavior"]["pause"]["on"] = "hover"
    c["behavior"]["resume"]["delay_after_leave_ms"] = {
        "value": 300.0, "confidence": {"value": 0.8, "band": "high"}}  # fmt: skip
    plan = rt.build_plan(env)
    dec = next(p for p in c["phases"] if p["kind"] == "decelerate")
    res = next(p for p in c["phases"] if p["kind"] == "resume")
    reg = c["region"]

    def inside(e):
        return (reg["x"] <= e["x"] <= reg["x"] + reg["w"]
                and reg["y"] <= e["y"] <= reg["y"] + reg["h"])  # fmt: skip

    enter = next(e for e in plan["events"] if e["type"] == "move" and inside(e))
    assert enter["t_ms"] == pytest.approx(rt.frame_t(rt.frame_of(dec["start_ms"])))
    # the pointer stays inside through the drags and leaves 300 ms before the resume
    outside_moves = [e for e in plan["events"] if e["type"] == "move" and not inside(e)
                     and e["buttons"] == 0 and e["t_ms"] > enter["t_ms"]]  # fmt: skip
    # (never before the last release inside the hover window)
    last_up = max(p["end_ms"] for p in c["phases"] if p["kind"] == "drag")
    k_leave = max(rt.frame_of(res["start_ms"] - 300), rt.frame_of(last_up) + 1)
    assert outside_moves[0]["t_ms"] == pytest.approx(rt.frame_t(k_leave))


def test_transition_hover_plan():
    env = load_sample_result()
    plan = rt.build_plan(env)
    s = env["spec"]
    segs = {x["id"]: x for x in s["segments"]}
    ev = plan["events"]
    assert plan["mode"] == "transition"
    assert [e["type"] for e in ev] == ["move", "move", "move"]
    target = next(e for e in s["elements"] if e["id"] == s["interaction"]["target_element_id"])
    b = target["bbox_initial"]
    enter, leave = ev[1], ev[2]
    t_f = segs["fwd"].get("trigger_event_ms") or segs["fwd"]["onset_ms"]
    assert enter["t_ms"] == pytest.approx(rt.frame_t(rt.frame_of(t_f)))
    assert b["x"] <= enter["x"] <= b["x"] + b["w"] and b["y"] <= enter["y"] <= b["y"] + b["h"]
    assert not (b["x"] <= leave["x"] <= b["x"] + b["w"] and b["y"] <= leave["y"] <= b["y"] + b["h"])


def test_transition_click_plan():
    env = load_sample_result()
    s = env["spec"]
    s["interaction"]["type"] = "click"
    s["interaction"]["trigger"]["kind"] = "click"
    s["interaction"]["trigger"]["reverse_kind"] = "click"
    ev = rt.build_plan(env)["events"]
    assert [e["type"] for e in ev].count("down") == 2
    assert [e["type"] for e in ev].count("up") == 2


def test_error_body_is_rejected():
    with pytest.raises(ValueError):
        rt.build_plan({"error": {"code": "no_motion_detected"}})


def test_viewport_falls_back_to_source_size():
    env = copy.deepcopy(load_sample_result())
    env["spec"].pop("scene", None)
    env["spec"]["source"].update(width=2560, height=1600, pixel_ratio=2)
    assert rt.viewport_of(env["spec"]) == (1280, 800, 2)


def test_find_chrome_honours_env(monkeypatch, tmp_path):
    fake = tmp_path / "chrome-headless-shell"
    fake.write_text("")
    monkeypatch.setenv("MIMIC_CHROME", str(fake))
    assert rt.find_chrome() == (str(fake), None)
    monkeypatch.setenv("MIMIC_CHROME", "/opt/chrome")
    assert rt.find_chrome() == ("/opt/chrome", "--headless=new")


# --------------------------------------------------------------------------------------------
# Acceptance run (iteration 1): presses, drags that end at rest, sample-driven replay
# --------------------------------------------------------------------------------------------


def _held_stop_spec() -> dict:
    """The continuous fixture rewritten as the motivating recording's tail: a fling, then a
    drag that ends at rest → held still (paused) → a drag that continues the same press → a
    drag that ends at rest → paused → resume."""
    env = load_sample_continuous_result()
    c = env["spec"]["continuous"]
    reg = c["region"]
    del reg  # geometry from the fixture
    ph = [
        {"kind": "autoplay", "start_ms": 0, "end_ms": 2000, "v_end_px_s": 40.0},
        {"kind": "drag", "start_ms": 2000, "end_ms": 2400, "displacement_px": -400.0,
         "v_end_px_s": -1100.0},
        {"kind": "inertia", "start_ms": 2400, "end_ms": 3400, "v_end_px_s": 0.0},
        {"kind": "drag", "start_ms": 3400, "end_ms": 4000, "displacement_px": -300.0,
         "v_end_px_s": -4.0},
        {"kind": "paused", "start_ms": 4000, "end_ms": 4120, "v_end_px_s": 0.0},
        {"kind": "drag", "start_ms": 4120, "end_ms": 4600, "displacement_px": 250.0,
         "v_end_px_s": 0.0},
        {"kind": "paused", "start_ms": 4600, "end_ms": 4800, "v_end_px_s": 0.0},
        {"kind": "resume", "start_ms": 4800, "end_ms": 5600, "v_end_px_s": 40.0},
    ]  # fmt: skip
    for i, p in enumerate(ph):
        p.setdefault("displacement_px", 0.0)
        p.update(id=f"p{i + 1}", notes=[], v_start_px_s=0.0)
    c["phases"] = ph
    c["samples"] = []
    c["behavior"]["pause"]["on"] = "unknown"
    return env


def test_press_lands_on_the_enter_frame():
    """No hover frame before a press: a page pausing on hover would stop autoplay a frame
    before the press the recording shows (the acceptance run's spurious 'decelerate')."""
    plan = rt.build_plan(_held_stop_spec())
    ev = plan["events"]
    for i, e in enumerate(ev):
        if e["type"] == "down":
            prev = ev[i - 1]
            assert prev["type"] == "move" and prev["t_ms"] == e["t_ms"]
            assert (prev["x"], prev["y"]) == (e["x"], e["y"])


def test_drag_ending_at_rest_holds_through_a_rest_into_the_next_drag():
    plan = rt.build_plan(_held_stop_spec())
    ev = plan["events"]
    downs = [e["t_ms"] for e in ev if e["type"] == "down"]
    ups = [e["t_ms"] for e in ev if e["type"] == "up"]
    # fling: press 2000 / release 2400; the held stop at 4000 continues into the 4120 drag (one
    # press), which ends at rest at 4600 and is let go after RELEASE_HOLD_MS of standing still
    assert downs == [pytest.approx(2000.0), pytest.approx(rt.frame_t(rt.frame_of(3400.0)))]
    k_hold = rt.frame_of(rt.RELEASE_HOLD_MS)
    assert ups == [pytest.approx(rt.frame_t(rt.frame_of(2400.0))),
                   pytest.approx(rt.frame_t(rt.frame_of(4600.0) + k_hold))]  # fmt: skip
    # no movement while held still, and the press is released inside the rest (before 4800)
    still = [e for e in ev if e["type"] == "move" and 4000 < e["t_ms"] < 4120]
    assert not still
    assert ups[-1] < 4800


def test_stop_drag_slows_to_rest_without_a_tail():
    ts = [i / 1000 for i in range(501)]
    off = rt.drag_offsets(300.0, 0.0, 0.5, ts, stop=True)
    assert off[-1] == pytest.approx(300.0)
    v = [(b - a) * 1000 for a, b in zip(off, off[1:], strict=False)]
    assert max(v) == pytest.approx(1200.0, rel=0.02)  # 2·D/T at the middle
    assert abs(v[-1]) < 1.0 and abs(v[0]) < 1.0


def test_sample_offsets_replay_one_unimodal_drag():
    """The content's own path (IR samples) replays a stopping drag; dips inside it are not
    replayed (they would read as release + re-grab), the speed tapers to rest at the end."""
    samples = []
    x = 0.0
    for i in range(0, 61):
        t = 1000.0 + i * 10.0
        v = 1500.0 if 1100 <= t <= 1400 else 300.0  # a plateau
        if 1200 <= t <= 1240:
            v = 600.0  # a dip
        if t > 1500:
            v = 0.0
        samples.append([t, v, x, 1.0])
        x += v * 0.010
    times = [1000.0 + k * 1000.0 / 60 for k in range(1, 31)]
    off = rt.sample_offsets(samples, 1000.0, 1500.0, times, x)
    assert off is not None and off[-1] == pytest.approx(x)
    steps = [b - a for a, b in zip([0.0, *off], off, strict=False)]
    peak = max(range(len(steps)), key=lambda i: steps[i])
    assert all(a <= b + 1e-9 for a, b in zip(steps[:peak], steps[1 : peak + 1], strict=False))
    assert all(a >= b - 1e-9 for a, b in zip(steps[peak:], steps[peak + 1 :], strict=False))
    assert steps[-1] < 0.5 * steps[-4]  # tapered


def test_unimodal_fit():
    assert rt.unimodal([0, 1, 3, 2, 5, 4, 4, 6, 3, 1, 2, 0]) == pytest.approx(
        [0, 1, 2.5, 2.5, 13 / 3, 13 / 3, 13 / 3, 6, 3, 1.5, 1.5, 0]
    )
