"""Track G: continuous-mode generators (PLAN-continuous §6.5).

* Snapshots of the 5 outputs (technical, LLM prompt, CSS, JSON, JS) for the continuous fixture
  (``docs/contract/sample-continuous-result.json``) and three variants: an autoplay-only marquee,
  a vertical drag + inertia scroller without cursor (with uncertain values, no appearance) and a
  drag + grid snap carousel with cursor (``UPDATE_SNAPSHOTS=1`` rewrites them).
* Determinism, number provenance (technical / LLM prompt / CSS / JS; the JS template may also
  print ``0``, ``1``, ``2`` and ``1000``), prompt contract (single HTML file, touch-action,
  reduced motion), no ``@keyframes`` when the driver is needed, uncertain values only under the
  uncertain sections.
* The generated JS is syntax-checked and run against a stub DOM with Node (skipped without
  ``node``): autoplay speed, drag, momentum decay, snap, resume.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.api.schemas import ResultEnvelope
from app.generate import export_json, render_all
from app.generate.continuous import css, js, llm_prompt, technical
from app.generate.continuous import phrasing as cp
from app.models.ir import MotionSpec, band_for
from tests.conftest import load_sample_continuous_result
from tests.unit.test_generators import (
    CSS_IDENTITY,
    allowed_numbers,
    unexplained_numbers,
)

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"
OUTPUT_FILES = {
    "technical": "technical.txt",
    "llm_prompt": "llm_prompt.txt",
    "css": "css.css",
    "json": "json.json",
    "js": "js.js",
}
#: JS-only template literals (structural ``0`` / ``1`` / ``2`` and ms <-> s ``1000``), see
#: ``js.TEMPLATE_LITERALS``; documented next to ``CSS_IDENTITY`` (test_generators.py).
JS_TEMPLATE_LITERALS = set(js.TEMPLATE_LITERALS)
EXTRA_LITERALS: dict[str, set[float]] = {
    "css": set(CSS_IDENTITY),
    "llm_prompt": {float(cp.MIN_VIEWPORT_PX)},
    "js": JS_TEMPLATE_LITERALS,
}

# --------------------------------------------------------------------------------------------
# Spec builders
# --------------------------------------------------------------------------------------------


def _conf(v: float) -> dict[str, Any]:
    return {"value": v, "band": band_for(v)}


def _mn(v: float, c: float) -> dict[str, Any]:
    return {"value": v, "confidence": _conf(c)}


def _easing(keyword: str | None, bez: list[float], family: str, named: str) -> dict[str, Any]:
    return {
        "keyword": keyword,
        "cubic_bezier": bez,
        "nearest_named": named,
        "family": family,
        "rmse": 0.03,
        "flags": [],
    }


EASE_OUT = _easing("ease-out", [0.0, 0.0, 0.58, 1.0], "ease-out", "ease-out")
EASE_IN = _easing("ease-in", [0.42, 0.0, 1.0, 1.0], "ease-in", "ease-in")
EASE_IN_OUT_CUBIC = _easing(None, [0.65, 0.0, 0.35, 1.0], "ease-in-out", "easeInOutCubic")


def _phase(
    i: int,
    kind: str,
    start: int,
    end: int,
    v: tuple[float, float, float],
    disp: float,
    conf: float,
    fit: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"p{i}",
        "kind": kind,
        "start_ms": start,
        "end_ms": end,
        "v_start_px_s": v[0],
        "v_end_px_s": v[1],
        "v_peak_px_s": v[2],
        "displacement_px": disp,
        "fit": fit,
        "interrupted": extra.get("interrupted", False),
        "evidence": extra.get("evidence", "velocity"),
        "confidence": _conf(conf),
        "notes": extra.get("notes", []),
    }


def fixture_dict() -> dict[str, Any]:
    return copy.deepcopy(load_sample_continuous_result()["spec"])


def fixture_spec() -> MotionSpec:
    return MotionSpec.model_validate(fixture_dict())


def _set_span(s: dict[str, Any], end: int) -> None:
    s["continuous"]["span_ms"] = {"start_ms": 0, "end_ms": end}
    s["interaction"]["total_duration_ms"] = {"forward": end, "reverse": None}


def marquee_dict() -> dict[str, Any]:
    """Autoplay-only marquee (C2-like): 240 px/s left, loop observed (864 px / 3600 ms)."""
    s = fixture_dict()
    s["interaction"].update(
        type="continuous",
        pattern="marquee",
        type_confidence=_conf(0.85),
        trigger={
            "kind": "autoplay",
            "reverse_kind": "none",
            "description": "Logo strip that scrolls on its own; no interaction recorded.",
            "confidence": _conf(0.9),
        },
    )
    c = s["continuous"]
    c["autoplay"] = {
        "direction": "left",
        "speed_px_s": _mn(240.0, 0.92),
        "velocity_px_s": -240.0,
        "easing": "linear",
        "loop": {
            "observed": True,
            "period_px": _mn(864.0, 0.9),
            "duration_ms": _mn(3600.0, 0.88),
        },
    }
    c["pitch_px"] = _mn(216.0, 0.75)
    c["gap_px"] = _mn(16.0, 0.6)
    c["phases"] = [
        _phase(
            1,
            "autoplay",
            0,
            8000,
            (-240.0, -240.0, -240.0),
            -1920.0,
            0.92,
            {"model": "constant", "velocity_px_s": _mn(-240.0, 0.92)},
        )
    ]
    c["behavior"] = {"pause": None, "drag": None, "inertia": None, "snap": None, "resume": None}
    c["samples"] = [
        [0, -240.0, 0.0, 0.95],
        [4000, -240.0, -960.0, 0.95],
        [8000, -240.0, -1920.0, 0.95],
    ]
    _set_span(s, 8000)
    s["warnings"] = []
    return s


def vertical_inertia_dict() -> dict[str, Any]:
    """Vertical ticker dragged once, exponential momentum, no cursor, no appearance, display
    scale 2x set by the user; the resume delay (25%) and one paused phase (25%) are uncertain."""
    s = fixture_dict()
    region = {"x": 480.0, "y": 100.0, "w": 320.0, "h": 600.0}
    s["source"].update(pixel_ratio=2, pixel_ratio_source="user", is_vfr=False, fps_effective=60.0)
    s.pop("scene")
    s["elements"] = [
        {
            "id": "e1",
            "label": "News ticker (e1)",
            "label_source": "interpreter",
            "role": "scroller",
            "parent_id": None,
            "kind": "scroller",
            "bbox_initial": region,
            "bbox_active": region,
            "text_like": False,
            "static": {"border_radius_px": None, "shadow": None},
        }
    ]
    s["interaction"].update(target_label="News ticker (e1)", type_confidence=_conf(0.7))
    c = s["continuous"]
    c.update(axis="y", region=region, region_confidence=_conf(0.65), pitch_px=None, gap_px=None)
    c["autoplay"] = {
        "direction": "up",
        "speed_px_s": _mn(30.0, 0.88),
        "velocity_px_s": -30.0,
        "easing": "linear",
        "loop": {"observed": False, "period_px": None, "duration_ms": None},
    }
    c["phases"] = [
        _phase(1, "autoplay", 0, 2000, (-30.0, -30.0, -30.0), -60.0, 0.88,
               {"model": "constant", "velocity_px_s": _mn(-30.0, 0.88)}),
        _phase(2, "drag", 2000, 2600, (-30.0, -800.0, -900.0), -300.0, 0.8),
        _phase(3, "inertia", 2600, 3800, (-800.0, -6.0, -800.0), -200.0, 0.75,
               {"model": "exponential", "tau_ms": _mn(250.0, 0.75),
                "v0_px_s": _mn(-800.0, 0.7), "v_inf_px_s": 0.0}),
        _phase(4, "paused", 3800, 4200, (0.0, 0.0, 0.0), 0.0, 0.25),
        _phase(5, "resume", 4200, 4800, (0.0, -30.0, -30.0), -9.0, 0.6,
               {"model": "ramp", "from_px_s": 0.0, "to_px_s": -30.0,
                "duration_ms": _mn(600.0, 0.6), "easing": EASE_IN}),
        _phase(6, "autoplay", 4800, 7000, (-30.0, -30.0, -30.0), -66.0, 0.85,
               {"model": "constant", "velocity_px_s": _mn(-30.0, 0.85)}),
    ]  # fmt: skip
    c["behavior"] = {
        "pause": None,
        "drag": {
            "count": 1,
            "follows_pointer": None,
            "pointer_ratio": None,
            "peak_speed_px_s": _mn(900.0, 0.7),
        },
        "inertia": {
            "model": "exponential",
            "tau_ms": _mn(250.0, 0.75),
            "duration_ms": None,
            "easing": None,
            "release_speed_min_px_s": 800.0,
            "release_speed_max_px_s": 800.0,
            "instances": 1,
        },
        "snap": None,
        "resume": {
            "delay_after_rest_ms": _mn(400.0, 0.25),
            "delay_after_release_ms": None,
            "ramp_ms": _mn(600.0, 0.6),
            "easing": EASE_IN,
            "to_speed_px_s": 30.0,
            "direction_preserved": True,
        },
    }
    c["samples"] = [[0, -30.0, 0.0, 0.95], [2300, -500.0, -150.0, 0.8], [7000, -30.0, -635.0, 0.95]]
    _set_span(s, 7000)
    s["warnings"] = [
        {
            "code": "cursor_not_visible",
            "severity": "warn",
            "message": "No cursor was visible; what starts and stops the motion is inferred "
            "from the motion only.",
        }
    ]
    return s


def snap_cursor_dict() -> dict[str, Any]:
    """Carousel with a visible cursor: hover pause (ease-out), one drag, grid snap 216 px /
    300 ms, resume with a fitted bezier. Gap not measured (CSS uses pitch - card width); the card
    title colour is uncertain (20%)."""
    s = fixture_dict()
    region = {"x": 120.0, "y": 200.0, "w": 1200.0, "h": 240.0}
    s["source"].update(width=2880, height=1800, pixel_ratio=2, is_vfr=False, fps_effective=60.0)
    s["scene"] = {"viewport_css": {"w": 1440.0, "h": 900.0}, "page_background": _mn("#FFFFFF", 0.9)}
    s["interaction"].update(
        type_source="interpreter",
        type_confidence=_conf(0.85),
        target_label="Product carousel (e1)",
        trigger={
            "kind": "drag",
            "reverse_kind": "none",
            "description": "Hovering pauses the carousel; dragging moves it, then it snaps.",
            "confidence": _conf(0.85),
        },
    )
    s["elements"] = [
        {
            "id": "e1", "label": "Product carousel (e1)", "label_source": "interpreter",
            "role": "scroller", "parent_id": None, "kind": "scroller",
            "bbox_initial": region, "bbox_active": region, "text_like": False,
            "static": {"border_radius_px": None, "shadow": None},
        },
        {
            "id": "e2", "label": "Product card (e2)", "label_source": "interpreter",
            "role": "card", "parent_id": "e1", "kind": "transform",
            "bbox_initial": {"x": 136.0, "y": 240.0, "w": 200.0, "h": 160.0},
            "bbox_active": {"x": 136.0, "y": 240.0, "w": 200.0, "h": 160.0}, "text_like": False,
            "static": {
                "border_radius_px": _mn(8.0, 0.6),
                "shadow": {
                    "value": {"x": 0.0, "y": 4.0, "blur": 12.0, "spread": 0.0,
                              "rgba": [0, 0, 0, 0.08]},
                    "confidence": _conf(0.5),
                },
                "background_color": _mn("#F4F4F6", 0.8),
            },
        },
        {
            "id": "e3", "label": "Card title (e3)", "label_source": "interpreter",
            "role": "title", "parent_id": "e2", "kind": "transform",
            "bbox_initial": {"x": 152.0, "y": 252.0, "w": 120.0, "h": 20.0},
            "bbox_active": {"x": 152.0, "y": 252.0, "w": 120.0, "h": 20.0}, "text_like": True,
            "static": {"border_radius_px": None, "shadow": None,
                       "text_color": _mn("#222222", 0.2), "font_size_px": _mn(14.0, 0.5)},
        },
    ]  # fmt: skip
    c = s["continuous"]
    c.update(region=region, region_confidence=_conf(0.75), pitch_px=_mn(216.0, 0.8), gap_px=None)
    c["autoplay"] = {
        "direction": "left",
        "speed_px_s": _mn(40.0, 0.9),
        "velocity_px_s": -40.0,
        "easing": "linear",
        "loop": {"observed": False, "period_px": None, "duration_ms": None},
    }
    c["phases"] = [
        _phase(1, "autoplay", 0, 2000, (-40.0, -40.0, -40.0), -80.0, 0.9,
               {"model": "constant", "velocity_px_s": _mn(-40.0, 0.9)}),
        _phase(2, "decelerate", 2000, 2400, (-40.0, 0.0, -40.0), -10.0, 0.85,
               {"model": "ramp", "from_px_s": -40.0, "to_px_s": 0.0,
                "duration_ms": _mn(400.0, 0.8), "easing": EASE_OUT},
               evidence="velocity+cursor"),
        _phase(3, "paused", 2400, 2600, (0.0, 0.0, 0.0), 0.0, 0.8),
        _phase(4, "drag", 2600, 3000, (0.0, -700.0, -800.0), -200.0, 0.85, evidence="cursor"),
        _phase(5, "snap", 3000, 3300, (-700.0, 0.0, -700.0), -16.0, 0.75,
               {"model": "tween", "distance_px": _mn(-16.0, 0.7),
                "duration_ms": _mn(300.0, 0.75), "easing": EASE_OUT}),
        _phase(6, "paused", 3300, 3700, (0.0, 0.0, 0.0), 0.0, 0.8),
        _phase(7, "resume", 3700, 4300, (0.0, -40.0, -40.0), -12.0, 0.7,
               {"model": "ramp", "from_px_s": 0.0, "to_px_s": -40.0,
                "duration_ms": _mn(600.0, 0.7), "easing": EASE_IN_OUT_CUBIC}),
        _phase(8, "autoplay", 4300, 8000, (-40.0, -40.0, -40.0), -148.0, 0.9,
               {"model": "constant", "velocity_px_s": _mn(-40.0, 0.9)}),
    ]  # fmt: skip
    c["behavior"] = {
        "pause": {
            "on": "hover",
            "on_confidence": _conf(0.85),
            "decel_ms": _mn(400.0, 0.8),
            "easing": EASE_OUT,
            "stops_completely": True,
        },
        "drag": {
            "count": 1,
            "follows_pointer": True,
            "pointer_ratio": _mn(1.0, 0.85),
            "peak_speed_px_s": _mn(800.0, 0.8),
        },
        "inertia": None,
        "snap": {
            "kind": "grid",
            "step_px": _mn(216.0, 0.8),
            "duration_ms": _mn(300.0, 0.75),
            "easing": EASE_OUT,
            "overshoot": False,
        },
        "resume": {
            "delay_after_rest_ms": _mn(400.0, 0.7),
            "delay_after_release_ms": _mn(700.0, 0.6),
            "ramp_ms": _mn(600.0, 0.7),
            "easing": EASE_IN_OUT_CUBIC,
            "to_speed_px_s": 40.0,
            "direction_preserved": True,
        },
    }
    c["samples"] = [[0, -40.0, 0.0, 0.95], [2800, -600.0, -150.0, 0.9], [8000, -40.0, -466.0, 0.95]]
    _set_span(s, 8000)
    s["cursor"] = {
        "visible": True,
        "confidence": 0.9,
        "events": [
            {"t_ms": 1990, "kind": "enter", "element_id": "e1"},
            {"t_ms": 3400, "kind": "leave", "element_id": "e1"},
        ],
    }
    s["interpretation"] = {
        "provider": "claude_cli",
        "model": "sonnet",
        "status": "ok",
        "duration_ms": 4200,
    }
    s["warnings"] = []
    return s


def _hover_marquee_dict() -> dict[str, Any]:
    """Marquee whose autoplay pauses on hover (no resume seen): CSS keyframes path + hover."""
    s = marquee_dict()
    c = s["continuous"]
    c["phases"] = [
        _phase(1, "autoplay", 0, 3000, (-240.0, -240.0, -240.0), -720.0, 0.92,
               {"model": "constant", "velocity_px_s": _mn(-240.0, 0.92)}),
        _phase(2, "decelerate", 3000, 3500, (-240.0, 0.0, -240.0), -60.0, 0.8,
               {"model": "ramp", "from_px_s": -240.0, "to_px_s": 0.0,
                "duration_ms": _mn(500.0, 0.8), "easing": EASE_OUT}),
        _phase(3, "paused", 3500, 8000, (0.0, 0.0, 0.0), 0.0, 0.8),
    ]  # fmt: skip
    c["behavior"]["pause"] = {
        "on": "hover",
        "on_confidence": _conf(0.85),
        "decel_ms": _mn(500.0, 0.8),
        "easing": EASE_OUT,
        "stops_completely": True,
    }
    return s


def _p2_fields_dict() -> dict[str, Any]:
    """The fixture with the P2 IR additions: the press stops autoplay at once (no decelerate
    phase), momentum stop speed on the inertia fits + behaviour, and a resume delay counted
    from the pointer leaving (hover)."""
    s = fixture_dict()
    c = s["continuous"]
    phases = c["phases"]
    assert phases[1]["kind"] == "decelerate"
    phases[0]["end_ms"] = phases[1]["end_ms"]
    del phases[1]
    for k, ph in enumerate(phases, start=1):
        ph["id"] = f"p{k}"
        if ph["kind"] == "inertia" and ph["fit"] is not None:
            ph["fit"]["stop_px_s"] = 10.0
    b = c["behavior"]
    b["pause"] = {
        "on": "hover",
        "on_confidence": _conf(0.85),
        "decel_ms": None,
        "easing": None,
        "stops_completely": True,
    }
    b["inertia"]["stop_px_s"] = _mn(10.0, 0.8)
    b["resume"]["delay_after_leave_ms"] = _mn(300.0, 0.7)
    return s


def _with_card_scale(s: dict[str, Any], ref: float, at_ref: float, conf: float) -> dict[str, Any]:
    c = s["continuous"]
    r = c["region"]
    half = (r["w"] if c["axis"] == "x" else r["h"]) / 2.0
    c["card_scale"] = {
        "model": "quadratic",
        "origin": "scroller_center",
        "reference_distance_px": ref,
        "scale_at_reference": at_ref,
        "mean_scale": round(1.0 + (at_ref - 1.0) * (half / ref) ** 2 / 3.0, 4),
        "confidence": _conf(conf),
    }
    return s


def card_scale_dict() -> dict[str, Any]:
    """The fixture with cards that grow towards the scroller edges (P2c, the motivating
    recording's effect): ×1 at the centre, ×1.15 at 270 px (mean magnification 1.099)."""
    return _with_card_scale(fixture_dict(), 270.0, 1.15, 0.6)


def _blend_dict(*, resume: bool = True, pause: bool = True) -> dict[str, Any]:
    """The fixture whose momentum decays towards the autoplay velocity (every exponential fit's
    ``v_inf`` = +39 px/s, like the motivating recording): releases blend into autoplay."""
    s = fixture_dict()
    c = s["continuous"]
    for ph in c["phases"]:
        if ph["kind"] == "inertia" and ph["fit"] is not None:
            ph["fit"]["v_inf_px_s"] = 39.0
    if not resume:
        c["behavior"]["resume"] = None
    if not pause:
        c["behavior"]["pause"] = None
    return s


def _held_stop_dict() -> dict[str, Any]:
    """The motivating recording's behaviour (acceptance run): momentum blends into autoplay;
    every pause is a press (no slowdown phase); the last drag ends with the pointer stopped →
    paused → resume (no momentum, no stop phase); the scroller is not centred."""
    s = _blend_dict()
    c = s["continuous"]
    ph = c["phases"]
    ph[0]["end_ms"] = 2000  # autoplay straight into the press (no decelerate)
    del ph[1]
    drag = next(p for p in ph if p["kind"] == "drag" and p["start_ms"] == 7000)
    drag.update(end_ms=7750, v_end_px_s=0.0)
    ph[:] = [p for p in ph if p["kind"] != "stop"]
    paused = next(p for p in ph if p["kind"] == "paused")
    paused["start_ms"] = 7750
    for i, p in enumerate(ph):
        p["id"] = f"p{i + 1}"
    b = c["behavior"]
    b["pause"].update(decel_ms=None, easing=None, stops_completely=True)
    b["snap"] = None
    b["resume"]["delay_after_release_ms"] = None
    return s


def _off_centre_dict() -> dict[str, Any]:
    """The fixture's scroller moved off-centre (the recording: x 93, y 307 in 904×786)."""
    s = _held_stop_dict()
    c = s["continuous"]
    dx, dy = 93.0 - c["region"]["x"], 307.0 - c["region"]["y"]
    c["region"]["x"], c["region"]["y"] = 93.0, 307.0
    for el in s["elements"]:
        for key in ("bbox_initial", "bbox_active"):
            if el.get(key):
                el[key]["x"] += dx
                el[key]["y"] += dy
    return s


VARIANT_DICTS: dict[str, Callable[[], dict[str, Any]]] = {
    "sample": fixture_dict,
    "marquee": marquee_dict,
    "vertical_inertia": vertical_inertia_dict,
    "snap_cursor": snap_cursor_dict,
    "card_scale": card_scale_dict,
}
VARIANTS: dict[str, Callable[[], MotionSpec]] = {
    name: (lambda f=f: MotionSpec.model_validate(f())) for name, f in VARIANT_DICTS.items()
}
ALL_VARIANTS = {
    **VARIANTS,
    "hover_marquee": lambda: MotionSpec.model_validate(_hover_marquee_dict()),
    "p2_fields": lambda: MotionSpec.model_validate(_p2_fields_dict()),
    "card_scale_uncertain": lambda: MotionSpec.model_validate(
        _with_card_scale(fixture_dict(), 270.0, 1.15, 0.2)
    ),
    "card_scale_marquee": lambda: MotionSpec.model_validate(
        _with_card_scale(marquee_dict(), 300.0, 1.2, 0.6)
    ),
    "card_scale_vertical": lambda: MotionSpec.model_validate(
        _with_card_scale(vertical_inertia_dict(), 250.0, 1.1, 0.4)
    ),
    "blend": lambda: MotionSpec.model_validate(_blend_dict()),
    "held_stop": lambda: MotionSpec.model_validate(_held_stop_dict()),
    "off_centre": lambda: MotionSpec.model_validate(_off_centre_dict()),
    "blend_no_resume": lambda: MotionSpec.model_validate(_blend_dict(resume=False)),
    "blend_card_scale": lambda: MotionSpec.model_validate(
        _with_card_scale(_blend_dict(), 270.0, 1.15, 0.6)
    ),
}


def _section(text: str, header: str, next_headers: list[str]) -> str:
    """Text of one section (from its header line to the next known header)."""
    start = text.index(f"\n{header}\n")
    ends = [text.find(f"\n{h}\n", start + 1) for h in next_headers]
    ends = [e for e in ends if e != -1]
    return text[start : min(ends) if ends else len(text)]


# --------------------------------------------------------------------------------------------
# Snapshots + determinism
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("key", list(OUTPUT_FILES))
def test_snapshot(variant: str, key: str) -> None:
    out = render_all(VARIANTS[variant]()).model_dump()[key]
    path = SNAPSHOT_DIR / f"continuous_{variant}.{OUTPUT_FILES[key]}"
    if UPDATE:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")
    assert path.exists(), f"missing snapshot {path.name}: run with UPDATE_SNAPSHOTS=1"
    assert out == path.read_text(encoding="utf-8"), f"{path.name} changed (UPDATE_SNAPSHOTS=1)"


@pytest.mark.parametrize("variant", list(ALL_VARIANTS))
def test_deterministic(variant: str) -> None:
    spec = ALL_VARIANTS[variant]()
    first = render_all(spec).model_dump()
    assert render_all(spec).model_dump() == first
    reloaded = MotionSpec.model_validate_json(spec.model_dump_json())
    assert render_all(reloaded).model_dump() == first
    assert render_all(ALL_VARIANTS[variant]()).model_dump() == first


def test_render_all_uses_continuous_modules() -> None:
    spec = fixture_spec()
    out = render_all(spec)
    assert out.technical == technical.render(spec)
    assert out.llm_prompt == llm_prompt.render(spec)
    assert out.css == css.render(spec)
    assert out.js == js.render(spec)
    assert out.json_ == export_json.render(spec)
    assert set(out.model_dump()) == {"technical", "llm_prompt", "css", "json", "js"}


def test_fixture_outputs_are_rendered() -> None:
    """The continuous fixture's ``outputs`` equal ``render_all(spec)`` (regenerated by Track G)."""
    doc = load_sample_continuous_result()
    env = ResultEnvelope.model_validate(doc)
    assert env.outputs.model_dump() == render_all(env.spec).model_dump()


# --------------------------------------------------------------------------------------------
# Number provenance
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(ALL_VARIANTS))
@pytest.mark.parametrize("key", ["technical", "llm_prompt", "css", "js"])
def test_number_provenance(variant: str, key: str) -> None:
    spec = ALL_VARIANTS[variant]()
    text = render_all(spec).model_dump()[key]
    extra = EXTRA_LITERALS.get(key, set())
    assert unexplained_numbers(text, allowed_numbers(spec), extra) == []


def test_js_provenance_catches_invented_numbers() -> None:
    spec = fixture_spec()
    text = js.render(spec) + "\nconst FRICTION = 0.92;\n"
    assert unexplained_numbers(text, allowed_numbers(spec), JS_TEMPLATE_LITERALS) == [0.92]


# --------------------------------------------------------------------------------------------
# Content: prompt contract, behaviours, appearance
# --------------------------------------------------------------------------------------------


def test_prompt_structure_and_contract() -> None:
    text = llm_prompt.render(fixture_spec())
    assert text.startswith(llm_prompt.opening_line(fixture_spec()) + "\n")
    assert text.rstrip("\n").splitlines()[-1] == llm_prompt.FINAL_LINE
    headers = [
        "STRUCTURE",
        "APPEARANCE (measured, approximate)",
        "BEHAVIOUR",
        "OUTPUT",
        "CONSTRAINTS",
    ]
    pos = [text.index(f"\n{h}\n") for h in headers]
    assert pos == sorted(pos)
    for needle in (
        "one self-contained, responsive HTML file named index.html",
        "one inline <script> (the motion needs it)",
        "neutral placeholders instead of real images or copy",
        "320px wide phones",
        "setPointerCapture",
        "touch-action: pan-y",
        "prefers-reduced-motion: reduce, do not autoplay at all",
        "dragging and the release momentum still work",
        "visibilitychange",
        "requestAnimationFrame",
        "e^(−dt/τ)",
        "τ ≈200 ms",
        "to the right at ≈39 px/s",
        "Repeat the cards once (two identical copies back to back)",
        "do not add snapping unless intended",
        "ramping up from standstill to ≈39 px/s over ≈750 ms using an ease-in-out curve",
        "Measured release speeds were ≈750–1400 px/s",
        "Pressing again while it is still coasting stops it at once",
    ):
        assert needle in text, needle
    steps = [ln.split(".")[0] for ln in _section(text, "BEHAVIOUR", ["OUTPUT"]).splitlines()
             if ln[:1].isdigit()]  # fmt: skip
    assert steps == ["1", "2", "3", "4", "5", "6"]


def test_prompt_appearance_block() -> None:
    text = llm_prompt.render(fixture_spec())
    block = _section(text, "APPEARANCE (measured, approximate)", ["BEHAVIOUR"])
    for needle in (
        "Recorded viewport: 1280×800 px; page background #FAFAFA.",
        "Scroller: ≈760×200 px (medium confidence), top-left corner at x 260, y 300 of the "
        "recorded viewport (horizontally and vertically centred)",
        "background #F1F1F4 (medium confidence)",
        "Cards: ≈200×160 px each, vertically centred in the scroller",
        "gap ≈16 px (medium confidence)",
        "background #FFFFFF (medium confidence)",
        "corner radius possibly ≈12 px (low confidence)",
        "colour #1A1A1A (medium confidence)",
        "font size possibly ≈16 px (low confidence)",
        "all content stays placeholder",
    ):
        assert needle in block, needle
    # no scene / element appearance measured: only the band box is passed on
    vtext = llm_prompt.render(VARIANTS["vertical_inertia"]())
    vblock = _section(vtext, "APPEARANCE (measured, approximate)", ["BEHAVIOUR"])
    assert "- Scroller: ≈320×600 px (medium confidence), top-left corner at x 480, y 100." in vblock
    assert "Recorded viewport" not in vblock and "Cards" not in vblock


def test_prompt_vertical_axis_wording() -> None:
    text = llm_prompt.render(VARIANTS["vertical_inertia"]())
    assert "vertical carousel" in text
    assert "moves continuously upward at ≈30 px/s" in text
    assert "touch-action: pan-x" in text and "pan-y" not in text
    assert "horizontal page scrolling still works" in text
    assert "(display scale 2x as set on upload)" in text


def test_prompt_marquee_without_drag() -> None:
    text = llm_prompt.render(VARIANTS["marquee"]())
    assert "horizontal marquee" in text
    assert "one copy of the cards is ≈864 px long and one loop takes ≈3600 ms" in text
    assert "only if the motion needs it, one inline <script>" in text
    assert "@keyframes" in text
    for absent in ("touch-action", "setPointerCapture", "Dragging", "Resume", "momentum"):
        assert absent not in text, absent


def test_prompt_snap_and_hover_with_cursor() -> None:
    text = llm_prompt.render(VARIANTS["snap_cursor"]())
    for needle in (
        "Pause: when the pointer hovers the product carousel, autoplay slows to a stop over "
        "≈400 ms using an ease-out curve.",
        "follows the pointer exactly along the horizontal axis;",
        "Snap: on release, settle on the nearest card position (step ≈216 px) over ≈300 ms "
        "using an ease-out curve (medium confidence).",
        "- Card title: font size ≈14 px (medium confidence); placed in the upper part of the "
        "product card.",
        "(autoplay, dragging, and snapping all change that one value)",
        "dragging still works because the user drives them",
        "once the motion has come to rest and the pointer has left, wait ≈400 ms",
        "cubic-bezier(0.65, 0, 0.35, 1) (close to easeInOutCubic)",
        'pointerType "mouse"',
        "shadow 0 4px 12px rgba(0, 0, 0, 0.08) (medium confidence)",
    ):
        assert needle in text, needle
    assert "assumed: the pointer was not visible" not in text
    assert "The pointer was not visible in the recording" not in text


def test_uncertain_values_only_in_uncertain_sections() -> None:
    spec = VARIANTS["vertical_inertia"]()
    prompt = llm_prompt.render(spec)
    unc = _section(prompt, "UNCERTAIN (OPTIONAL)", ["OUTPUT"])
    assert "resume delay after rest: ≈400 ms (25% confidence)" in unc
    assert "≈400 ms" not in prompt.replace(unc, "")
    assert "wait a short delay (the delay is uncertain; see UNCERTAIN)" in prompt

    tech = technical.render(spec)
    tunc = _section(tech, "UNCERTAIN OBSERVATIONS", ["NOTES"])
    assert "resume delay after rest: ≈400 ms (25% confidence). Not included above." in tunc
    assert "p4, 3800–4200 ms: possibly paused; confidence low (25%)" in tunc
    assert "≈400 ms" not in tech.replace(tunc, "")
    assert "\np4 " not in _section(tech, "PHASES", ["BEHAVIOUR"])

    snap = VARIANTS["snap_cursor"]()
    sprompt = llm_prompt.render(snap)
    sunc = _section(sprompt, "UNCERTAIN (OPTIONAL)", ["OUTPUT"])
    assert "card title text colour: #222222 (20% confidence)" in sunc
    assert "#222222" not in sprompt.replace(sunc, "")
    scss = css.render(snap)
    assert "/* uncertain (20% confidence), omitted: color #222222 */" in scss
    assert "color: #222222" not in scss


def test_fixture_has_no_uncertain_items() -> None:
    assert "UNCERTAIN (OPTIONAL)" not in llm_prompt.render(fixture_spec())
    tech = technical.render(fixture_spec())
    assert _section(tech, "UNCERTAIN OBSERVATIONS", ["NOTES"]).strip().endswith("None.")


def test_technical_structure() -> None:
    text = technical.render(fixture_spec())
    headers = [
        "CONTINUOUS MOTION", "AUTOPLAY", "PHASES", "BEHAVIOUR", "APPEARANCE",
        "UNCERTAIN OBSERVATIONS", "NOTES",
    ]  # fmt: skip
    pos = [0 if text.startswith(f"{h}\n") else text.index(f"\n{h}\n") for h in headers]
    assert pos == sorted(pos)
    for needle in (
        "Type: Draggable scroller (carousel) — medium confidence (75%), from heuristic",
        "Region: 760×200 px at x 260, y 300 — medium (70%)",
        "Speed: ≈39 px/s, linear — high (90%)",
        "Loop: not observed",
        "τ ≈200 ms, v0 -1400 px/s",
        "decelerate (interrupted)",
        "Slowdown: possibly ≈700 ms, linear — low (40%)",
        "Comes to a full stop: no (cut short by the next interaction in the recording)",
        "Follows the pointer: unknown (no cursor visible to compare)",
        "Release speeds: ≈750–1400 px/s",
        "Snap: abrupt stops without grid evidence",
        "Ramp: ≈750 ms, ease-in-out — medium (70%)",
        "Signed velocities: positive = content moves right (x) / down (y).",
        "- p10: pointer likely stopped before release, or hard snap",
    ):
        assert needle in text, needle
    phase_rows = [ln for ln in _section(text, "PHASES", ["BEHAVIOUR"]).splitlines()
                  if ln.startswith("p")]  # fmt: skip
    assert [r.split()[0] for r in phase_rows] == [f"p{i}" for i in range(1, 14)]


def test_technical_snap_variant_values() -> None:
    text = technical.render(VARIANTS["snap_cursor"]())
    for needle in (
        "Snap: to the card grid",
        "  Step: ≈216 px — high (80%)",
        "  Duration: ≈300 ms, ease-out — medium (75%)",
        "Follows the pointer: yes (content / pointer speed 1.00 — high (85%))",
        "decelerate [velocity+cursor]",
        "tween ≈16 px (medium) over ≈300 ms (medium), ease-out",
        "Card pitch (card + gap): ≈216 px — high (80%)",
    ):
        assert needle in text, needle
    assert "Gap between cards" not in text
    assert "Element labels are heuristic" not in text


# --------------------------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------------------------


def test_css_no_keyframes_when_driver_needed() -> None:
    for name in ("sample", "vertical_inertia", "snap_cursor"):
        text = css.render(VARIANTS[name]())
        assert "@keyframes" not in text, name
        assert "need the JS driver (JS tab)" in text, name
        assert "touch-action:" in text, name


def test_css_marquee_keyframes() -> None:
    text = css.render(VARIANTS["marquee"]())
    assert "@keyframes scroller-autoplay {\n  to { translate: calc(0px - 864px) 0; }\n}" in text
    assert "animation: scroller-autoplay 3600ms linear infinite;" in text
    assert "animation-direction: reverse" not in text  # moves left = towards negative
    assert "@media (prefers-reduced-motion: reduce)" in text
    assert "JS driver" not in text


def test_css_hover_pause_and_unobserved_loop() -> None:
    text = css.render(ALL_VARIANTS["hover_marquee"]())
    assert ".scroller:hover .scroller__track {\n  animation-play-state: paused;" in text
    assert "slows over ≈500 ms rather than stopping instantly — see the JS tab" in text

    s = marquee_dict()
    a = s["continuous"]["autoplay"]
    a.update(direction="right", velocity_px_s=240.0)
    a["loop"] = {"observed": False, "period_px": None, "duration_ms": None}
    s["continuous"]["phases"][0]["fit"]["velocity_px_s"]["value"] = 240.0
    text = css.render(MotionSpec.model_validate(s))
    assert "translate: calc(0px - var(--copy-size) * 1px) 0;" in text
    assert "animation-duration: calc(var(--copy-size) / 240 * 1s);" in text
    assert "animation-direction: reverse;" in text


def test_css_appearance_and_gap_from_pitch() -> None:
    text = css.render(fixture_spec())
    assert ".scroller {\n  display: flex;\n  align-items: center;\n  max-width: 760px;" in text
    assert "gap: 16px; /* estimated, medium confidence */" in text
    assert ".scroller__card {\n  flex: none;\n  width: 200px;\n  height: 160px;" in text
    assert "body {\n  background-color: #FAFAFA; /* estimated, high confidence */\n}" in text

    snap = css.render(VARIANTS["snap_cursor"]())
    assert "gap: calc(216px - 200px); /* estimated, high confidence */" in snap
    assert ".product-carousel__product-card" not in snap  # BEM drops shared words
    assert ".product-carousel:active {\n  cursor: grabbing;" in snap


# --------------------------------------------------------------------------------------------
# JS
# --------------------------------------------------------------------------------------------


def test_js_constants_follow_behaviours() -> None:
    sample = js.render(fixture_spec())
    for needle in (
        "const AUTOPLAY_PX_S = 39;",
        "const PAUSE_DECEL_MS = 700;",
        "const INERTIA_TAU_MS = 200;",
        "const RESUME_DELAY_MS = 100;",
        "const RESUME_RAMP_MS = 750;",
        "const RESUME_EASE = [0.42, 0, 0.58, 1];",
        "v *= Math.exp((-dt * 1000) / INERTIA_TAU_MS)",
        "setPointerCapture",
        "matchMedia('(prefers-reduced-motion: reduce)')",
        "visibilitychange",
        "b.getBoundingClientRect().left - kids[0].getBoundingClientRect().left",
    ):
        assert needle in sample, needle
    assert "SNAP_STEP_PX" not in sample  # abrupt_ambiguous: no snapping
    assert "no snapping is implemented" in sample

    marquee = js.render(VARIANTS["marquee"]())
    for absent in ("pointerdown", "INERTIA", "RESUME", "PAUSE", "SNAP", "ease("):
        assert absent not in marquee, absent
    assert "const AUTOPLAY_PX_S = -240;" in marquee

    snap = js.render(VARIANTS["snap_cursor"]())
    for needle in ("const SNAP_STEP_PX = 216;", "const SNAP_MS = 300;", "snapTo(x);",
                   "const RESUME_PX_S = -40;",
                   "const RESUME_EASE = [0.65, 0, 0.35, 1];"):  # fmt: skip
        assert needle in snap, needle
    assert "INERTIA" not in snap

    vertical = js.render(VARIANTS["vertical_inertia"]())
    assert "e.clientY" in vertical and "getBoundingClientRect().top" in vertical
    assert "`0 ${((x % size) - size) % size}px`" in vertical
    assert "// resume" not in vertical and "uncertain (25% confidence)" in vertical


NODE = shutil.which("node")

#: Stub DOM + scripted interaction. Prints a JSON trace of checkpoints.
HARNESS = r"""
const listeners = {};
const style = {};
let clock = 0;
let cb = null;
// two "copies" whose first cards are 2000 px apart (the wrap period)
const kid = (at) => ({ getBoundingClientRect: () => ({ left: at, top: at }) });
const track = { children: [kid(0), kid(2000)], style };
const scroller = {
  addEventListener: (t, f) => { (listeners[t] ||= []).push(f); },
  setPointerCapture() {},
  matches: () => false,
  querySelector: () => track,
};
globalThis.document = { querySelector: () => scroller, addEventListener() {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.performance = { now: () => clock };
globalThis.requestAnimationFrame = (f) => { cb = f; };
const fire = (type, e = {}) => (listeners[type] || []).forEach((f) => f({
  button: 0, pointerId: 1, pointerType: 'mouse', timeStamp: clock, clientX: 500, clientY: 500,
  preventDefault() {}, stopPropagation() {}, ...e,
}));
const step = (ms) => {
  const end = clock + ms;
  while (clock < end) { clock += 16; const f = cb; cb = null; f(clock); }
};
const expose = '\n;return { get x() { return x; }, get v() { return v; }, '
  + 'get state() { return state; } };';
const api = new Function(DRIVER + expose)();
const trace = {};
const snap = (k) => { trace[k] = { x: api.x, v: api.v, state: api.state }; };
step(1008); snap('autoplay');
fire('pointerenter'); step(1600); snap('hovered');
fire('pointerdown');
let p = 500;
for (let i = 0; i < 10; i++) { step(16); p -= 20; fire('pointermove', { clientX: p, clientY: p }); }
snap('dragged');
fire('pointerup', { clientX: p, clientY: p }); snap('released');
step(3000); snap('after_release');
fire('pointerleave'); step(3000); snap('resumed');
console.log(JSON.stringify(trace));
"""


def _run_driver(spec: MotionSpec) -> dict[str, Any]:
    assert NODE is not None
    code = js.render(spec)
    script = f"const DRIVER = {json.dumps(code)};\n{HARNESS}"
    res = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("variant", list(ALL_VARIANTS))
def test_js_syntax(variant: str, tmp_path: Path) -> None:
    path = tmp_path / "driver.js"
    path.write_text(js.render(ALL_VARIANTS[variant]()), encoding="utf-8")
    res = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_runs_fixture_behaviour() -> None:
    t = _run_driver(fixture_spec())
    assert t["autoplay"]["state"] == "autoplay"
    assert t["autoplay"]["x"] == pytest.approx(39.0, abs=1.0)  # ≈39 px/s for ≈1 s
    assert t["hovered"]["state"] == "paused" and t["hovered"]["v"] == 0
    assert t["dragged"]["state"] == "drag"
    assert t["dragged"]["x"] - t["hovered"]["x"] == pytest.approx(-200.0)  # follows the pointer
    assert t["released"]["state"] == "inertia"
    assert t["released"]["v"] == pytest.approx(-1250.0, rel=0.05)  # 20 px per 16 ms
    # momentum: exponential decay travels ≈ v0 · τ further, then rests (still hovered)
    travel = t["after_release"]["x"] - t["released"]["x"]
    assert travel == pytest.approx(-1250.0 * 0.2, rel=0.1)
    assert t["after_release"]["state"] == "paused"
    # pointer left: delay + ramp, then autoplay again at the measured speed
    assert t["resumed"]["state"] == "autoplay" and t["resumed"]["v"] == pytest.approx(39.0)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_runs_snap_behaviour() -> None:
    t = _run_driver(VARIANTS["snap_cursor"]())
    assert t["autoplay"]["x"] == pytest.approx(-40.0, abs=1.0)
    assert t["hovered"]["state"] == "paused"
    assert t["released"]["state"] == "snap"
    steps = t["after_release"]["x"] / 216
    assert steps == pytest.approx(round(steps), abs=1e-9)  # landed exactly on the card grid
    assert t["resumed"]["state"] == "autoplay" and t["resumed"]["v"] == pytest.approx(-40.0)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_runs_marquee_and_vertical() -> None:
    t = _run_driver(VARIANTS["marquee"]())
    assert t["autoplay"]["x"] == pytest.approx(-240.0, abs=5.0)
    assert t["resumed"]["state"] == "autoplay"  # nothing reacts to the pointer

    t = _run_driver(VARIANTS["vertical_inertia"]())
    assert t["autoplay"]["x"] == pytest.approx(-30.0, abs=1.0)
    assert t["hovered"]["state"] == "autoplay"  # no hover pause observed
    assert t["released"]["state"] == "inertia"
    assert t["resumed"]["state"] == "autoplay" and t["resumed"]["v"] == pytest.approx(-30.0)


# --------------------------------------------------------------------------------------------
# P2 IR additions: momentum stop speed, resume after leave, press stops autoplay at once
# --------------------------------------------------------------------------------------------


def test_p2_fields_are_used_by_every_output() -> None:
    spec = ALL_VARIANTS["p2_fields"]()
    out = render_all(spec)
    assert "autoplay stops at once (no slowdown)" in out.llm_prompt
    assert "stop it once it is slower than ≈10 px/s" in out.llm_prompt
    assert "when the pointer leaves the" in out.llm_prompt and "≈300 ms" in out.llm_prompt
    assert "Slowdown: none (autoplay stops at once)" in out.technical
    assert "Stops below: ≈10 px/s" in out.technical
    assert "Delay after the pointer leaves: ≈300 ms" in out.technical
    assert out.js is not None
    assert "const INERTIA_STOP_PX_S = 10;" in out.js
    assert "if (Math.abs(v) < INERTIA_STOP_PX_S) rest();" in out.js
    assert "const RESUME_AFTER_LEAVE_MS = 300;" in out.js
    assert "letGo(RESUME_AFTER_LEAVE_MS);" in out.js
    assert "PAUSE_DECEL_MS" not in out.js  # no slowdown measured


def test_p2_fields_absent_render_as_before() -> None:
    """Specs without the P2 additions (the fixtures) keep their outputs: see the snapshots."""
    out = render_all(fixture_spec())
    assert "INERTIA_STOP_PX_S" not in (out.js or "")
    assert "RESUME_AFTER_LEAVE_MS" not in (out.js or "")


# --------------------------------------------------------------------------------------------
# P2c: cards that scale with their distance from the scroller centre
# --------------------------------------------------------------------------------------------


def test_card_scale_in_every_output() -> None:
    out = render_all(VARIANTS["card_scale"]())
    tech = out.technical
    assert (
        "Card scaling: by distance from the scroller centre, ×1 at the centre → ≈×1.15 at "
        "≈270 px from it (quadratic, cards stay packed; visible cards only, off-screen cards "
        "unscaled); on screen the content moves ≈×1.10 as "
        "fast as the unscaled track — medium (60%)"
    ) in tech
    assert "Card pitch (card + gap) at the scroller centre: ≈216 px" in tech
    assert "200×160 px (at the scroller centre; scales with its position)" in tech

    prompt = out.llm_prompt
    behaviour = _section(prompt, "BEHAVIOUR", ["OUTPUT"])
    assert "2. Card scaling: every card is scaled by its distance d from the scroller centre" in (
        behaviour
    )
    for needle in (
        "×1 at the centre → ≈×1.15 at ≈270 px from it (the farthest whole card measured)",
        "scale = 1 + (1.15 − 1) · (d / 270)²",
        "(medium confidence). Update every card's scale each frame",
        # acceptance iteration 2: a builder that scaled the off-screen cards too got an
        # unsolvable packing far out and a card ×2000 covering the scroller
        "Apply the curve only to cards that are at least partly inside the scroller and leave "
        "every card that is entirely outside it unscaled",
        "never extrapolate it off screen",
        "The gaps keep their measured size, so the cards stay packed",
        "divided by ≈1.10",
        "while dragging, by the pointer's movement divided by ≈1.10: the content then keeps up "
        "with the pointer on average (the unmagnified card at the centre moves slightly less "
        "than the pointer)",
        # dragging divides by the MEAN magnification: 1:1 on average, not "exactly"
        "Dragging: the content follows the pointer 1:1 on screen on average along the "
        "horizontal axis (assumed: the pointer was not visible in the recording to compare); "
        "with the card scaling, the cards near the scroller centre move slightly less than the "
        "pointer and those near the edges slightly more (see Card scaling);",
    ):
        assert needle in behaviour, needle
    assert "follows the pointer exactly" not in prompt
    assert (
        "packed with a constant gap and scaled by their position; at the scroller centre the "
        "pitch is ≈216 px (medium confidence)"
    ) in _section(prompt, "STRUCTURE", ["APPEARANCE (measured, approximate)"])
    assert "≈200×160 px each at the scroller centre (they grow towards the edges" in prompt
    assert "Update the card scales and positions in the same loop" in prompt
    steps = [ln.split(".")[0] for ln in behaviour.splitlines() if ln[:1].isdigit()]
    assert steps == ["1", "2", "3", "4", "5", "6", "7"]

    assert "Autoplay, dragging, and momentum and position-dependent card scaling" not in out.css
    assert "position-dependent card scaling need the JS driver" in out.css
    assert "×1 at the centre → ≈×1.15 at ≈270 px from it (estimated, medium confidence) */" in (
        out.css
    )
    # the track gets the layer (moved every frame); cards are scaled without one of their own
    # (fixed-scale layer textures re-rasterised mid-motion jittered by ±1 px in round trips)
    track_rule = out.css.split(".scroller__track {", 1)[1].split("}", 1)[0]
    card_rule = out.css.split(".scroller__card {", 1)[1].split("}", 1)[0]
    assert "will-change: translate;" in track_rule
    assert "will-change" not in card_rule
    assert out.css.count("position: relative;") == 2

    assert out.js is not None
    for needle in (
        "const SCALE_REF_PX = 270;",
        "const SCALE_AT_REF = 1.15;",
        "const SPEED_SCALE = 1.10;",
        "x += (v * dt) / SPEED_SCALE;",
        "x += (p - lastP) / SPEED_SCALE;",
        "scaleCards(offset);",
        "b.offsetLeft - kids[0].offsetLeft",
    ):
        assert needle in out.js, needle
    # layout offsets: the cards' own translate / scale must not change the wrap period
    assert "scrollWidth" not in out.js and "getBoundingClientRect" not in out.js


def test_card_scale_absent_or_uncertain() -> None:
    """Rigid cards render exactly as before (snapshots); an uncertain scale is listed only under
    the uncertain sections and not implemented."""
    spec = ALL_VARIANTS["card_scale_uncertain"]()
    out = render_all(spec)
    item = "card scaling by distance from the scroller centre: ×1 at the centre → ≈×1.15 at"
    unc = _section(out.llm_prompt, "UNCERTAIN (OPTIONAL)", ["OUTPUT"])
    assert item in unc
    assert "Card scaling" not in out.llm_prompt and "≈×1.15" not in out.llm_prompt.replace(unc, "")
    tunc = _section(out.technical, "UNCERTAIN OBSERVATIONS", ["NOTES"])
    assert item in tunc
    assert "Card scaling: uncertain (see UNCERTAIN OBSERVATIONS)" in out.technical
    assert "SCALE_" not in (out.js or "") and "scaleCards" not in (out.js or "")
    assert "will-change: translate, scale" not in out.css and "card scaling" not in out.css


def test_card_scale_marquee_needs_the_driver() -> None:
    out = render_all(ALL_VARIANTS["card_scale_marquee"]())
    assert "@keyframes" not in out.css
    assert "Autoplay and position-dependent card scaling need the JS driver" in out.css
    assert "(autoplay changes that one value)" in out.llm_prompt
    assert out.js is not None and "scaleCards(offset);" in out.js


def test_card_scale_vertical_axis() -> None:
    out = render_all(ALL_VARIANTS["card_scale_vertical"]())
    assert out.js is not None
    needles = ("c.offsetTop", "c.offsetHeight", "scroller.clientHeight", "`0 ${at[i] - u[i]}px`",
               "b.offsetTop - kids[0].offsetTop", "`0 ${offset}px`")  # fmt: skip
    for needle in needles:
        assert needle in out.js, needle
    assert "possibly ×1 at the centre" in out.llm_prompt  # low confidence (40%)
    assert "along the vertical axis" in out.llm_prompt


#: Stub DOM with 20 cards (200 px + 16 px gaps, pitch 216) in a 760 px scroller.
CARD_HARNESS = r"""
const listeners = {};
let clock = 0;
let cb = null;
const N = 20;
const cards = [];
for (let i = 0; i < N; i++) {
  cards.push({ offsetLeft: i * 216, offsetTop: 0, offsetWidth: 200, offsetHeight: 160,
               style: {} });
}
const track = {
  offsetLeft: 0, offsetTop: 20, offsetWidth: N * 216 - 16, children: cards, style: {},
};
const scroller = {
  clientWidth: 760,
  style: {},
  addEventListener: (t, f) => { (listeners[t] ||= []).push(f); },
  setPointerCapture() {},
  matches: () => false,
  querySelector: () => track,
};
globalThis.document = { querySelector: () => scroller, addEventListener() {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.getComputedStyle = () => ({ position: 'static' });
globalThis.performance = { now: () => clock };
globalThis.requestAnimationFrame = (f) => { cb = f; };
const fire = (type, e = {}) => (listeners[type] || []).forEach((f) => f({
  button: 0, pointerId: 1, pointerType: 'mouse', timeStamp: clock, clientX: 500, clientY: 500,
  preventDefault() {}, stopPropagation() {}, ...e,
}));
const step = (ms) => {
  const end = clock + ms;
  while (clock < end) { clock += 16; const f = cb; cb = null; f(clock); }
};
const expose = '\n;return { get x() { return x; }, scaleCards };';
const api = new Function(DRIVER + expose)();
const snapshot = () => cards.map((c, i) => ({
  u: i * 216 + 100,
  t: c.style.translate === '' || c.style.translate === undefined
    ? null : parseFloat(c.style.translate),
  s: c.style.scale === '' || c.style.scale === undefined ? null : parseFloat(c.style.scale),
}));
const out = { positioned: [scroller.style.position, track.style.position] };
step(1008);
out.x_autoplay = api.x;
out.after_frame = snapshot();
const sweep = [];
for (let off = -800; off > -800 - 216; off -= 9) {
  api.scaleCards(off);
  const a = snapshot();
  api.scaleCards(off - 0.01);
  sweep.push({ off, a, b: snapshot() });
}
out.sweep = sweep;
fire('pointerenter');
step(1600);
const before = api.x;
fire('pointerdown');
let p = 500;
for (let i = 0; i < 10; i++) { step(16); p -= 20; fire('pointermove', { clientX: p }); }
out.drag = api.x - before;
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_card_scaling_runs() -> None:
    spec = VARIANTS["card_scale"]()
    cs = spec.continuous.card_scale  # type: ignore[union-attr]
    assert cs is not None
    k = (1.15 - 1.0) / 270.0**2  # SCALE_AT_REF, SCALE_REF_PX as printed
    speed_scale = 1.10
    code = js.render(spec)
    res = subprocess.run(
        [NODE, "-e", f"const DRIVER = {json.dumps(code)};\n{CARD_HARNESS}"],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    t = json.loads(res.stdout)
    assert t["positioned"] == ["relative", "relative"]
    # autoplay: the unscaled track moves at the on-screen speed / SPEED_SCALE
    assert t["x_autoplay"] == pytest.approx(39.0 / speed_scale, abs=1.0)
    assert any(c["s"] is not None and c["s"] > 1.0 for c in t["after_frame"])
    # dragging 200 px on screen moves the unscaled track 200 / SPEED_SCALE
    assert t["drag"] == pytest.approx(-200.0 / speed_scale)

    half, offset0 = 380.0, 0.0
    seen_edge = False
    for row in t["sweep"]:
        off = row["off"]
        vis = []
        for c in row["a"]:
            u = c["u"] + offset0 + off - half  # unscaled centre from the scroller centre
            if abs(u) - 100.0 > half:
                assert c["t"] is None and c["s"] is None  # off screen: untouched
                continue
            d = u + c["t"]
            # scale = 1 + k d² by the card's own on-screen distance
            assert c["s"] == pytest.approx(1.0 + k * d * d, abs=1e-9)
            vis.append((d, c["s"]))
        # packed: every gap keeps its 16 px layout size
        for (d0, s0), (d1, s1) in zip(vis, vis[1:], strict=False):
            assert (d1 - 100.0 * s1) - (d0 + 100.0 * s0) == pytest.approx(16.0, abs=1e-6)
        # the card nearest the centre is (almost) unscaled; edge cards reach the measured scale
        near = min(vis, key=lambda v: abs(v[0]))
        assert abs(near[0]) <= 120.0 and near[1] == pytest.approx(1.0 + k * near[0] ** 2)
        # a card at the measured reference distance (270 px) has the measured scale ×1.15
        seen_edge = seen_edge or any(abs(dd) >= 270.0 and sc >= 1.15 for dd, sc in vis)
        # continuous: a 0.01 px offset change moves every visible card by < 0.1 px
        for ca, cb in zip(row["a"], row["b"], strict=True):
            if ca["t"] is not None and cb["t"] is not None:
                assert abs((ca["u"] + ca["t"]) - (cb["u"] + cb["t"]) + 0.0) < 0.1
    assert seen_edge


# --------------------------------------------------------------------------------------------
# Wrap period: one copy = first card of copy A -> first card of copy B (gap included)
# --------------------------------------------------------------------------------------------

#: Stub DOM: two copies of 6 cards (200 px + 16 px flex gap, pitch 216, one copy 1296 px; half
#: the track is 1288 px, half a gap short) in a 760 px scroller. Runs the driver frame by frame
#: (autoplay, or a fast drag) across several loops and reports the largest frame-to-frame jump
#: of the cards near the scroller centre (on screen, matched to the nearest card one frame
#: earlier: cards are identical), separately for the frames where the track wrapped and the
#: others, and the largest wrap error (track offset change minus the unwrapped position change,
#: modulo the true copy).
WRAP_HARNESS = r"""
const N = 6, W = 200, GAP = 16, PITCH = W + GAP, COPY = N * PITCH, HALF = 380;
const listeners = {};
let clock = 0;
let cb = null;
const px = (v) => (v === undefined || v === '' ? 0 : parseFloat(v));
const cards = [];
const track = { offsetLeft: 0, offsetTop: 20, style: {}, children: cards };
track.offsetWidth = track.scrollWidth = 2 * N * PITCH - GAP;
track.offsetHeight = track.scrollHeight = 160;
for (let i = 0; i < 2 * N; i++) {
  const c = { offsetLeft: i * PITCH, offsetTop: 0, offsetWidth: W, offsetHeight: 160, style: {} };
  c.getBoundingClientRect = () => {
    const s = c.style.scale ? parseFloat(c.style.scale) : 1;
    const left = track.offsetLeft + px(track.style.translate) + c.offsetLeft
      + px(c.style.translate) + (W - W * s) / 2;
    return { left, top: track.offsetTop, width: W * s, height: 160 * s };
  };
  cards.push(c);
}
const scroller = {
  clientWidth: 2 * HALF, clientHeight: 200, style: {},
  addEventListener: (t, f) => { (listeners[t] ||= []).push(f); },
  setPointerCapture() {},
  matches: () => false,
  querySelector: () => track,
};
globalThis.document = { querySelector: () => scroller, addEventListener() {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.getComputedStyle = () => ({ position: 'static' });
globalThis.performance = { now: () => clock };
globalThis.requestAnimationFrame = (f) => { cb = f; };
const fire = (type, e = {}) => (listeners[type] || []).forEach((f) => f({
  button: 0, pointerId: 1, pointerType: 'mouse', timeStamp: clock, clientX: 500, clientY: 500,
  preventDefault() {}, stopPropagation() {}, ...e,
}));
const api = new Function(DRIVER + '\n;return { get x() { return x; } };')();
// on-screen card centres relative to the scroller centre
const centres = () => cards.map((c) => track.offsetLeft + px(track.style.translate)
  + c.offsetLeft + W / 2 + px(c.style.translate) - HALF);
const frame = () => { clock += 16; const f = cb; cb = null; f(clock); };
frame();
let prev = centres(), prevT = px(track.style.translate), prevX = api.x;
let p = 500;
if (MODE === 'drag') fire('pointerdown', { clientX: p });
let maxJump = 0, maxWrapJump = 0, maxStep = 0, maxWrapError = 0, wraps = 0;
for (let k = 0; k < FRAMES; k++) {
  if (MODE === 'drag') { p -= STEP; fire('pointermove', { clientX: p }); }
  frame();
  const now = centres();
  const t = px(track.style.translate);
  const wrapped = Math.abs(t - prevT) > PITCH;
  for (const c of now) {
    if (Math.abs(c) > HALF - 120) continue;
    const jump = Math.min(...prev.map((q) => Math.abs(c - q)));
    if (wrapped) maxWrapJump = Math.max(maxWrapJump, jump);
    else maxJump = Math.max(maxJump, jump);
  }
  const r = (t - prevT) - (api.x - prevX);
  maxWrapError = Math.max(maxWrapError, Math.abs(r - Math.round(r / COPY) * COPY));
  maxStep = Math.max(maxStep, Math.abs(api.x - prevX));
  if (wrapped) wraps += 1;
  prev = now; prevT = t; prevX = api.x;
}
console.log(JSON.stringify({ maxJump, maxWrapJump, maxStep, maxWrapError, wraps }));
"""


def _run_wrap(code: str, mode: str, frames: int, step: float = 0.0) -> dict[str, Any]:
    assert NODE is not None
    script = (
        f"const DRIVER = {json.dumps(code)};\nconst MODE = {json.dumps(mode)};\n"
        f"const FRAMES = {frames};\nconst STEP = {step};\n{WRAP_HARNESS}"
    )
    res = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


#: (variant, mode, frames, pointer px per frame): autoplay 240 px/s for 20 s, or a 24 px/frame
#: drag for 200 frames — 4800 px on screen either way, more than three copies (1296 px).
WRAP_CASES = [
    ("marquee", "autoplay", 1250, 0.0),
    ("card_scale_marquee", "autoplay", 1250, 0.0),
    ("sample", "drag", 200, 24.0),
    ("card_scale", "drag", 200, 24.0),
]


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize(("variant", "mode", "frames", "step"), WRAP_CASES)
def test_js_wrap_is_seamless(variant: str, mode: str, frames: int, step: float) -> None:
    spec = ALL_VARIANTS[variant]()
    t = _run_wrap(js.render(spec), mode, frames, step)
    assert t["wraps"] >= 3  # several loops
    # the track offset only ever jumps by exactly one copy (1296 px), never by 1296 - gap / 2
    assert t["maxWrapError"] < 1e-6
    # on screen, a wrap frame moves the cards near the centre no further than any other frame
    assert t["maxWrapJump"] <= t["maxJump"] + 1e-6
    if spec.continuous.card_scale is None:  # type: ignore[union-attr]
        # rigid: every card moves exactly by the track step
        assert t["maxJump"] == pytest.approx(t["maxStep"], abs=1e-6)
    else:
        # scaled: magnified, but packed and smooth (well under a gap / 2 + the step)
        assert t["maxStep"] < t["maxJump"] < t["maxStep"] * 1.25


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("variant", ["marquee", "card_scale_marquee"])
def test_js_wrap_harness_catches_half_track(variant: str) -> None:
    """Control: the old period (half the track) jumps by half a gap (8 px) on every loop."""
    code = js.render(ALL_VARIANTS[variant]())
    start = code.index("const copySize = () => {")
    end = code.index("\n};\n", start) + len("\n};\n")
    old = code[:start] + "const copySize = () => track.offsetWidth / 2;\n" + code[end:]
    t = _run_wrap(old, "autoplay", 1250)
    assert t["wraps"] >= 3
    assert t["maxWrapError"] == pytest.approx(8.0, abs=1e-6)
    assert t["maxWrapJump"] > t["maxJump"] + 4.0


# --------------------------------------------------------------------------------------------
# Momentum that decays towards the autoplay velocity: it blends into autoplay, never "rests"
# --------------------------------------------------------------------------------------------


def test_momentum_towards_zero_comes_to_rest() -> None:
    out = render_all(fixture_spec())
    assert "let it decay exponentially towards zero with τ ≈200 ms: v = v · e^(−dt/τ)" in (
        out.llm_prompt
    )
    assert "until it comes to rest." in out.llm_prompt
    assert "blends" not in out.llm_prompt and "Decays towards" not in out.technical
    assert out.js is not None and "v *= Math.exp(" in out.js and "vEnd" not in out.js


def test_momentum_blending_into_autoplay_wording() -> None:
    out = render_all(ALL_VARIANTS["blend"]())
    prompt = out.llm_prompt
    momentum = next(ln for ln in prompt.splitlines() if "Release (momentum)" in ln)
    for needle in (
        "let it decay exponentially towards the autoplay velocity with τ ≈200 ms: "
        "v = v_auto + (v − v_auto) · e^(−dt/τ) each frame",
        "It does not come to rest: it blends back into autoplay, which simply keeps running, so "
        "no rest, delay or resume ramp follows such a release (Resume below applies only after "
        "the motion has come to rest, e.g. after a pause)",
        "the hover pause starts again only when the pointer re-enters the scroller",
    ):
        assert needle in momentum, needle
    for absent in ("comes to rest", "until", "v = v · e^"):
        assert absent not in momentum, absent
    assert "(with no autoplay to blend into, the momentum decays to zero and stops)" in prompt
    assert (
        "  Decays towards: the autoplay velocity (blends back into autoplay, which keeps running; "
        "no rest or separate resume after such a release)"
    ) in out.technical
    assert out.js is not None
    for needle in (
        "const INERTIA_TAU_MS = 200; // momentum decay time constant (towards the autoplay "
        "velocity)",
        "const vEnd = canAutoplay() ? autoV : 0;",
        "v = vEnd + (v - vEnd) * Math.exp((-dt * 1000) / INERTIA_TAU_MS);",
        "const left = ((v - vEnd) * INERTIA_TAU_MS) / 1000;",
        "      else state = 'autoplay';",
    ):
        assert needle in out.js, needle
    assert "v *= Math.exp(" not in out.js


def test_momentum_blending_without_resume() -> None:
    prompt = llm_prompt.render(ALL_VARIANTS["blend_no_resume"]())
    assert "so no rest, delay or resume ramp follows such a release;" in prompt
    assert "Resume below" not in prompt
    # releases go back to autoplay by themselves: only the pause can leave it stopped
    assert "did not show autoplay resuming after a pause; resume it only if intended." in prompt
    assert "after the interaction" not in prompt
    bare = MotionSpec.model_validate(_blend_dict(resume=False, pause=False))
    assert "did not show autoplay resuming" not in llm_prompt.render(bare)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_runs_momentum_blend() -> None:
    t = _run_driver(ALL_VARIANTS["blend"]())
    assert t["released"]["state"] == "inertia"
    assert t["released"]["v"] == pytest.approx(-1250.0, rel=0.05)
    # decays towards +39 px/s and hands over to autoplay (still hovered: no pause, no resume)
    assert t["after_release"]["state"] == "autoplay"
    assert t["after_release"]["v"] == pytest.approx(39.0)
    # travel: v_inf · t + (v0 − v_inf) · τ over the 3 s
    travel = t["after_release"]["x"] - t["released"]["x"]
    assert travel == pytest.approx(39.0 * 3.0 + (-1250.0 - 39.0) * 0.2, rel=0.1)
    assert t["resumed"]["state"] == "autoplay" and t["resumed"]["v"] == pytest.approx(39.0)


# --------------------------------------------------------------------------------------------
# Acceptance run (iteration 1): press-only pause, drags that end at rest, scroller position
# --------------------------------------------------------------------------------------------


def test_press_only_pause_is_described_and_implemented_as_a_press() -> None:
    """Pause trigger not visible, but every pause was a drag starting (no slowdown phase): the
    outputs describe a press pause and add no hover pause (after releases the motion carried on
    while the pointer was presumably still over the scroller)."""
    spec = ALL_VARIANTS["held_stop"]()
    assert cp.pause_by_press_inferred(spec) and cp.pause_trigger(spec) == "press"
    out = render_all(spec)
    beh = _section(out.llm_prompt, "BEHAVIOUR", ["OUTPUT"])
    assert "Pause: when the pointer presses the scroller (every pause in the recording was the "
    assert "a pause on hover alone was not observed: do not add one" in beh
    assert "It stays stopped while it is held." in beh
    assert "the hover pause starts again" not in beh and "and the pointer has left" not in beh
    assert "- Hover:" not in out.llm_prompt
    assert out.js is not None
    assert "pointerenter" not in out.js and "hovered" not in out.js
    assert "every pause was a drag starting, so it is implemented as a press" in out.js
    assert "Pause: while pressed (trigger not visible in the recording; every pause was a drag "
    assert "starting)" in out.technical
    # a recording with a measured slowdown keeps the hover reading
    assert not cp.pause_by_press_inferred(fixture_spec())


def test_drag_ending_at_rest_is_a_release_without_momentum() -> None:
    spec = ALL_VARIANTS["held_stop"]()
    assert [p.id for p in cp.drag_stops(spec)] == ["p8"]
    assert cp.resumes_after_drag_stop(spec)
    out = render_all(spec)
    beh = _section(out.llm_prompt, "BEHAVIOUR", ["OUTPUT"])
    for needle in (
        "when the pointer is let go while it is still moving (a fling)",
        "no rest, delay or resume ramp follows a fling",
        "if the pointer did not move during that time (it had stopped before being let go) the "
        "release velocity is zero",
        "the pointer slows down and stops before it is let go, and the content stops together "
        "with it",
        "not a separate glide",
        "Such a release has zero velocity, so there is no momentum: it stays where it stopped "
        "and Resume below applies.",
        "Resume: once the content has come to rest after a release without momentum, autoplay "
        "restarts ≈100 ms after the content stopped moving (count from the last movement, not "
        "from the release, which was not visible; if the pointer is still pressed then, "
        "restart when it is let go)",
    ):
        assert needle in beh, needle
    assert "after release" not in beh.split("Resume:", 1)[1]
    assert "Released at rest: drags that end with the pointer stopped have no momentum (the "
    assert "content rests, then Resume)" in out.technical
    js = out.js or ""
    for needle in ("let movedAt = 0;", "if (p !== lastP) movedAt = e.timeStamp;",
                   "if (vRelease === 0) {", "restAt = movedAt;"):  # fmt: skip
        assert needle in js, needle
    # no drag ends at rest in the fixture: none of this
    plain = render_all(ALL_VARIANTS["blend"]())
    assert "movedAt" not in (plain.js or "") and "Such a release" not in plain.llm_prompt


def test_js_rest_release_rests_from_the_last_movement() -> None:
    """Node: drag, hold still 150 ms, let go → no momentum, resume RESUME_DELAY_MS after the
    last movement (not after the release); a fling still blends into autoplay."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = render_all(ALL_VARIANTS["held_stop"]()).js
    assert js is not None
    harness = r"""
const listeners = {};
let now = 0;
let cb = null;
const kids = Array.from({ length: 16 }, (_, i) => ({
  getBoundingClientRect: () => ({ left: i * 216, top: 0 }), style: {},
}));
const track = { style: {}, children: kids, offsetLeft: 0 };
const scroller = {
  querySelector: () => track, matches: () => false, setPointerCapture() {},
  clientWidth: 760, addEventListener: (t, f) => { (listeners[t] ||= []).push(f); },
};
globalThis.document = { querySelector: () => scroller, addEventListener() {} };
globalThis.performance = { now: () => now };
globalThis.matchMedia = () => ({ matches: false });
globalThis.requestAnimationFrame = (f) => { cb = f; };
globalThis.getComputedStyle = () => ({ position: 'relative' });
const fire = (type, x) => (listeners[type] || []).forEach((f) => f({
  button: 0, pointerId: 1, pointerType: 'mouse', timeStamp: now, clientX: x, clientY: 0,
  preventDefault() {}, stopPropagation() {},
}));
const api = new Function(DRIVER + '\n;return { get state() { return state; } };')();
const step = (ms) => { for (let i = 0; i < ms / 16; i++) { now += 16; const f = cb; f(now); } };
step(160);
fire('pointerdown', 400);
for (let i = 1; i <= 20; i++) { step(16); fire('pointermove', 400 - i * 10); }
const stoppedAt = now;
step(160);  // held still
fire('pointerup', 200);
const out = { stoppedAt, releasedAt: now, states: [] };
for (let i = 0; i < 40; i++) { step(16); out.states.push([now, api.state]); }
console.log(JSON.stringify(out));
"""
    script = f"const DRIVER = {json.dumps(js)};\n{harness}"
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout)
    states = res["states"]
    assert all(st != "inertia" for _, st in states)  # no momentum
    first_resume = next(t for t, st in states if st == "resume")
    # resume RESUME_DELAY_MS (100 ms in the fixture) after the last movement, i.e. right after
    # the release here (held 160 ms > 100 ms)
    assert first_resume - res["releasedAt"] <= 32
    assert first_resume - res["stoppedAt"] >= 100


def test_off_centre_scroller_position_in_prompt_and_css() -> None:
    out = render_all(ALL_VARIANTS["off_centre"]())
    app = _section(out.llm_prompt, "APPEARANCE (measured, approximate)", ["BEHAVIOUR"])
    assert "top-left corner at x 93, y 307 of the recorded viewport (not centred)" in app
    assert "Match these sizes, colours and positions at the recorded viewport size" in app
    assert "place its top-left corner exactly there" in out.llm_prompt
    assert "do not centre it unless APPEARANCE says it is centred" in out.llm_prompt
    body = out.css.split("body {", 1)[1].split("}", 1)[0]
    scroller = out.css.split(".scroller {", 1)[1].split("}", 1)[0]
    assert "margin: 0;" in body
    assert "margin: 307px 0 0 93px;" in scroller and "margin-inline: auto" not in scroller
    # a centred scroller keeps auto margins
    centred = render_all(fixture_spec()).css
    assert "margin-inline: auto;" in centred.split(".scroller {", 1)[1].split("}", 1)[0]
