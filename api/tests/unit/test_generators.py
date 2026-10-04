"""Track G: generators (PLAN §9, §11.1(5)).

* Snapshots of the 4 outputs for ``docs/contract/sample-result.json``, a low-confidence
  variant and a measured-appearance variant (PLAN-continuous §13)
  (``UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_generators.py`` to rewrite).
* Determinism: same spec -> byte-identical text, also after a JSON round trip.
* Number provenance: every number printed in Technical / LLM prompt / CSS maps to an IR value
  (after abs / display rounding / x100 for confidences).
* Low-confidence wording and omissions, PRD §18/§19/§21 structure, selector variants.
* Measured appearance: absent → byte-identical outputs (the sample / low-confidence snapshots
  predate it); present → APPEARANCE block / section, CSS size + colour declarations.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest

from app.generate import css, export_json, llm_prompt, phrasing, render_all, technical
from app.models.ir import Easing, MotionSpec, Shadow
from tests.conftest import load_sample_result

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"
OUTPUT_FILES = {
    "technical": "technical.txt",
    "llm_prompt": "llm_prompt.txt",
    "css": "css.css",
    "json": "json.json",
}

# --------------------------------------------------------------------------------------------
# Spec builders
# --------------------------------------------------------------------------------------------


def _conf(overall: float, value: float, timing: float, easing: float) -> dict[str, Any]:
    return {
        "overall": overall,
        "value": value,
        "timing": timing,
        "easing": easing,
        "band": phrasing.band_for(overall),
    }


def sample_spec_dict() -> dict[str, Any]:
    return copy.deepcopy(load_sample_result()["spec"])


def sample_spec() -> MotionSpec:
    return MotionSpec.model_validate(sample_spec_dict())


def low_confidence_spec_dict() -> dict[str, Any]:
    """Fixture degraded: no cursor, heuristic labels, unknown trigger, forward only,
    mixed low / uncertain (< 0.3) transitions, user-set 1x display scale."""
    s = sample_spec_dict()
    s["source"].update(
        width=1440,
        height=900,
        fps_nominal=30.0,
        fps_effective=30.0,
        pixel_ratio=1,
        pixel_ratio_source="user",
        timing_resolution_ms=33,
    )
    s["interaction"].update(
        type="hover",
        type_source="heuristic",
        type_confidence={"value": 0.4, "band": "low"},
        pattern="card",
        target_label="Card (e1)",
        target_description=None,
        direction="forward",
        total_duration_ms={"forward": 350, "reverse": None},
        trigger={
            "kind": "unknown",
            "reverse_kind": "unknown",
            "description": "Trigger could not be determined (no cursor visible).",
            "confidence": {"value": 0.3, "band": "low"},
        },
    )
    for el, (label, role) in zip(
        s["elements"],
        [
            ("Card (e1)", "card"),
            ("Image (e2)", "image"),
            ("Text (e3)", "text"),
            ("Icon (e4)", "icon"),
        ],  # fmt: skip
        strict=True,
    ):
        el.update(label=label, label_source="heuristic", role=role)
    s["segments"] = [seg for seg in s["segments"] if seg["id"] == "fwd"]
    s["segments"][0]["trigger_event_ms"] = None
    s["transitions"] = [t for t in s["transitions"] if t["segment_id"] == "fwd"]
    by_id = {t["id"]: t for t in s["transitions"]}
    by_id["t1"]["confidence"] = _conf(0.55, 0.62, 0.6, 0.45)
    by_id["t1"]["easing"] = {
        "keyword": None,
        "cubic_bezier": [0.215, 0.61, 0.355, 1.0],
        "nearest_named": "easeOutCubic",
        "family": "ease-out",
        "rmse": 0.05,
        "flags": [],
    }
    by_id["t2"]["confidence"] = _conf(0.2, 0.25, 0.3, 0.15)  # uncertain
    by_id["t3"]["confidence"] = _conf(0.25, 0.3, 0.35, 0.2)  # uncertain
    by_id["t4"]["confidence"] = _conf(0.42, 0.5, 0.45, 0.33)  # low
    by_id["t5"]["confidence"] = _conf(0.35, 0.4, 0.4, 0.3)  # low
    by_id["t6"]["confidence"] = _conf(0.28, 0.3, 0.3, 0.25)  # uncertain
    s["relationships"] = [r for r in s["relationships"] if r["segment_id"] == "fwd"]
    s["cursor"] = {"visible": False, "confidence": 0.0, "events": []}
    s["structure"] = ["image", "text", "icon"]
    s["interpretation"] = {
        "provider": "none",
        "model": None,
        "status": "disabled",
        "duration_ms": None,
    }
    s["warnings"] = [
        {
            "code": "cursor_not_visible",
            "severity": "warn",
            "message": "No cursor was visible; the trigger is inferred from the motion only.",
        },
        {
            "code": "reverse_not_recorded",
            "severity": "info",
            "message": "Only the forward motion was recorded; reverse timings are not measured.",
        },
    ]
    return s


def low_confidence_spec() -> MotionSpec:
    return MotionSpec.model_validate(low_confidence_spec_dict())


def _variant(mutate: Callable[[dict[str, Any]], None]) -> MotionSpec:
    s = sample_spec_dict()
    mutate(s)
    return MotionSpec.model_validate(s)


def _click(s: dict[str, Any]) -> None:
    s["interaction"].update(type="click")
    s["interaction"]["trigger"].update(kind="click", reverse_kind="click")


def _press(s: dict[str, Any]) -> None:
    rename = {"fwd": "rt_in", "rev": "rt_out"}
    s["interaction"].update(type="press", direction="round_trip", pattern="button")
    s["interaction"]["trigger"].update(kind="click", reverse_kind="none")
    for seg in s["segments"]:
        seg["id"] = rename[seg["id"]]
    for t in s["transitions"]:
        t["segment_id"] = rename[t["segment_id"]]
    for r in s["relationships"]:
        r["segment_id"] = rename[r["segment_id"]]


def _dropdown(s: dict[str, Any]) -> None:
    _click(s)
    s["interaction"].update(type="dropdown", pattern="menu")


def _modal(s: dict[str, Any]) -> None:
    """The target is the opened surface (kind ``appear``), so the prompt adds a trigger button."""
    _click(s)
    s["interaction"].update(type="modal", pattern="modal")
    s["elements"][0]["kind"] = "appear"


def _stagger(s: dict[str, Any]) -> None:
    s["relationships"] = [
        {
            "kind": "stagger",
            "segment_id": "fwd",
            "transition_ids": ["t1", "t3", "t5"],
            "offset_ms": None,
            "interval_ms": 40,
        }
    ]


def _height(s: dict[str, Any]) -> None:
    by_id = {t["id"]: t for t in s["transitions"]}
    by_id["t5"].update(property="height", to={"kind": "px", "number": 120.0}, delta=120.0)
    by_id["t11"].update(property="height", **{"from": {"kind": "px", "number": 120.0}})
    by_id["t11"].update(delta=-120.0)
    s["interaction"].update(type="expand_collapse", pattern="accordion")


def _overshoot(s: dict[str, Any]) -> None:
    s["transitions"][0]["easing"] = {
        "keyword": None,
        "cubic_bezier": [0.34, 1.56, 0.64, 1.0],
        "nearest_named": "backOut",
        "family": "overshoot",
        "rmse": 0.02,
        "flags": ["overshoot"],
    }


def _subsampled(s: dict[str, Any]) -> None:
    s["source"]["timing_resolution_ms"] = 25
    s["warnings"].append({
        "code": "frames_subsampled",
        "severity": "info",
        "message": "The motion lasts long (220 frames around it), so it was analysed at about "
        "40 fps instead of every frame; start times and durations are less precise.",
    })  # fmt: skip


def _mc(value: str, conf: float) -> dict[str, Any]:
    return {"value": value, "confidence": {"value": conf, "band": phrasing.band_for(conf)}}


def _mn(value: float, conf: float) -> dict[str, Any]:
    return {"value": value, "confidence": {"value": conf, "band": phrasing.band_for(conf)}}


def appearance_spec_dict() -> dict[str, Any]:
    """Sample + measured appearance (PLAN-continuous §13): scene, fills, text colour, font sizes,
    and a static text element (no transitions) that only the appearance outputs mention."""
    s = sample_spec_dict()
    s["scene"] = {"viewport_css": {"w": 1440.0, "h": 900.0},
                  "page_background": _mc("#F3F4F6", 0.85)}  # fmt: skip
    st = {e["id"]: e["static"] for e in s["elements"]}
    st["e1"]["background_color"] = _mc("#FFFFFF", 0.85)
    st["e2"]["background_color"] = _mc("#C7D2FE", 0.38)  # low band -> "possibly"
    st["e3"]["text_color"] = _mc("#111111", 0.8)  # animated (t3 color): left to the steps
    st["e3"]["font_size_px"] = _mn(17.8, 0.45)
    s["elements"].append({
        "id": "e5", "label": "Card description", "label_source": "interpreter", "role": "text",
        "parent_id": "e1", "kind": "transform",
        "bbox_initial": {"x": 584.0, "y": 508.0, "w": 272.0, "h": 40.0},
        "bbox_active": {"x": 584.0, "y": 500.0, "w": 272.0, "h": 40.0},
        "text_like": True,
        "static": {"border_radius_px": None, "shadow": None,
                   "text_color": _mc("#6B7280", 0.72), "font_size_px": _mn(14.2, 0.5)},
    })  # fmt: skip
    return s


def appearance_spec() -> MotionSpec:
    return MotionSpec.model_validate(appearance_spec_dict())


VARIANTS: dict[str, Callable[[], MotionSpec]] = {
    "sample": sample_spec,
    "low_confidence": low_confidence_spec,
    "click": lambda: _variant(_click),
    "press": lambda: _variant(_press),
    "dropdown": lambda: _variant(_dropdown),
    "modal": lambda: _variant(_modal),
    "stagger": lambda: _variant(_stagger),
    "height": lambda: _variant(_height),
    "overshoot": lambda: _variant(_overshoot),
    "subsampled": lambda: _variant(_subsampled),
    "appearance": appearance_spec,
}
SNAPSHOT_VARIANTS = ["sample", "low_confidence", "appearance"]

# --------------------------------------------------------------------------------------------
# Snapshots + determinism
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", SNAPSHOT_VARIANTS)
@pytest.mark.parametrize("key", list(OUTPUT_FILES))
def test_snapshot(variant: str, key: str) -> None:
    out = render_all(VARIANTS[variant]()).model_dump()[key]
    path = SNAPSHOT_DIR / f"{variant}.{OUTPUT_FILES[key]}"
    if UPDATE:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")
    assert path.exists(), f"missing snapshot {path.name}: run with UPDATE_SNAPSHOTS=1"
    assert out == path.read_text(encoding="utf-8"), f"{path.name} changed (UPDATE_SNAPSHOTS=1)"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_deterministic(variant: str) -> None:
    spec = VARIANTS[variant]()
    first = render_all(spec).model_dump()
    assert render_all(spec).model_dump() == first
    reloaded = MotionSpec.model_validate_json(spec.model_dump_json())
    assert render_all(reloaded).model_dump() == first
    assert render_all(VARIANTS[variant]()).model_dump() == first


def test_render_all_uses_each_module() -> None:
    spec = sample_spec()
    out = render_all(spec)
    assert out.technical == technical.render(spec)
    assert out.llm_prompt == llm_prompt.render(spec)
    assert out.css == css.render(spec)
    assert out.json_ == export_json.render(spec)
    assert set(out.model_dump()) == {"technical", "llm_prompt", "css", "json"}


# --------------------------------------------------------------------------------------------
# Number provenance invariant
# --------------------------------------------------------------------------------------------

NUM_RE = re.compile(r"(?<![\w#.\-])-?\d+(?:\.\d+)?")
HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")
ORDINAL_RE = re.compile(r"^\d+\.\s", re.MULTILINE)
ROUND_STEPS = ("1", "0.5", "0.1", "0.05", "0.01", "0.001", "10")
#: CSS-only structural literals (``translate: 0 -8px``, ``0fr``/``1fr``, ``var(--i, 0)``).
CSS_IDENTITY = {0.0, 1.0}
#: LLM-prompt-only layout guidance that is not a measurement: the narrowest viewport the
#: requested HTML file must support (OUTPUT section). Keep this list minimal and explicit.
PROMPT_LAYOUT_LITERALS = {float(llm_prompt.MIN_VIEWPORT_PX)}
EXTRA_LITERALS = {"css": CSS_IDENTITY, "llm_prompt": PROMPT_LAYOUT_LITERALS}


def _numbers_in(text: str) -> list[float]:
    text = HEX_RE.sub(" ", text)
    text = ORDINAL_RE.sub("", text)  # numbered step markers are not measurements
    return [float(m) for m in NUM_RE.findall(text)]


def _collect(obj: Any, path: tuple[str, ...], out: list[tuple[float, tuple[str, ...]]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "samples":
                _collect(v, (*path, k), out)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, path, out)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, int | float):
        out.append((float(obj), path))
    elif isinstance(obj, str):
        out.extend((n, path) for n in _numbers_in(obj))


def _forms(v: float, is_confidence: bool) -> set[float]:
    bases = {v, abs(v)}
    if is_confidence:
        bases.add(v * 100)
    out: set[float] = set()
    for b in bases:
        out.add(round(b, 6))
        d = Decimal(repr(b))
        for step in ROUND_STEPS:
            s = Decimal(step)
            for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                out.add(round(float((d / s).quantize(Decimal(1), rounding=mode) * s), 6))
    return out


def allowed_numbers(spec: MotionSpec) -> set[float]:
    found: list[tuple[float, tuple[str, ...]]] = []
    _collect(spec.model_dump(mode="json"), (), found)
    allowed: set[float] = set()
    for v, path in found:
        allowed |= _forms(v, any("confidence" in p for p in path))
    return allowed


def offset_numbers(spec: MotionSpec) -> set[float]:
    """APPEARANCE positions are relative to the parent (PLAN-continuous §13): child box − parent
    box, state A (active box when the state-A box is empty). The only derived magnitude the
    generators may print; computed here independently of ``technical.parent_offset``."""

    def box(e: Any) -> Any:
        b = e.bbox_initial
        return b if b.w > 0 and b.h > 0 else e.bbox_active

    by_id = {e.id: e for e in spec.elements}
    out: set[float] = set()
    for e in spec.elements:
        if e.parent_id in by_id:
            b, pb = box(e), box(by_id[e.parent_id])
            out |= _forms(b.x - pb.x, False) | _forms(b.y - pb.y, False)
    return out


def unexplained_numbers(text: str, allowed: set[float], extra: set[float] = frozenset()) -> list:
    ok = allowed | set(extra)
    return [n for n in _numbers_in(text) if round(n, 6) not in ok]


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("key", ["technical", "llm_prompt", "css"])
def test_number_provenance(variant: str, key: str) -> None:
    spec = VARIANTS[variant]()
    text = render_all(spec).model_dump()[key]
    extra = EXTRA_LITERALS.get(key, set())
    allowed = allowed_numbers(spec) | offset_numbers(spec)
    assert unexplained_numbers(text, allowed, extra) == []


def test_provenance_check_catches_invented_numbers() -> None:
    spec = sample_spec()
    allowed = allowed_numbers(spec)
    text = llm_prompt.render(spec) + "\nAlso rotate the card 7deg over 333ms.\n"
    assert unexplained_numbers(text, allowed) == [7.0, 333.0]


def test_json_export_is_ir_without_samples() -> None:
    spec = sample_spec()
    data = json.loads(export_json.render(spec))
    assert data == spec.model_dump(mode="json", exclude={"transitions": {"__all__": {"samples"}}})
    assert all("samples" not in t for t in data["transitions"])
    assert list(data)[:3] == ["schema_version", "job_id", "meta"]
    assert "initial_state" in data and "active_state" in data
    MotionSpec.model_validate(data)  # the export is itself a valid IR


# --------------------------------------------------------------------------------------------
# PRD structure (fixture)
# --------------------------------------------------------------------------------------------


def _section_order(text: str, headers: list[str]) -> list[int]:
    return [text.index(f"\n{h}\n") if not text.startswith(f"{h}\n") else 0 for h in headers]


def test_technical_structure_prd18() -> None:
    text = technical.render(sample_spec())
    headers = [
        "INTERACTION", "PRODUCT CARD (e1)", "PRODUCT IMAGE (e2)", "CARD TITLE (e3)",
        "ARROW ICON (e4)", "REVERSE", "TIMING RELATIONSHIPS", "UNCERTAIN OBSERVATIONS", "NOTES",
    ]  # fmt: skip
    pos = _section_order(text, headers)
    assert pos == sorted(pos), "sections out of PRD §18 / PLAN §9.2 order"
    for needle in (
        "Type: Hover interaction",
        "Target: Product card (e1)",
        "Trigger: Animation begins when the pointer enters",
        "Initial: translateY(0px)",
        "Hover: approximately translateY(-8px)",
        "Duration: approximately 280ms",
        "Easing: ease-out",
        "Delay: approximately 30ms",
        "Transform origin: center",
        "shadow becomes larger and darker",
        "moves approximately 4px to the right",
        "opacity increases from approximately 0.7 to 1",
        "Product image scaling begins approximately 30ms after the product card movement.",
        "display scale assumed 2x",
    ):
        assert needle in text, needle


def test_llm_prompt_structure_prd19() -> None:
    text = llm_prompt.render(sample_spec())
    assert text.startswith(llm_prompt.OPENING_LINE + "\n")
    assert text.rstrip("\n").splitlines()[-1] == llm_prompt.FINAL_LINE
    pos = _section_order(text, ["STRUCTURE", "INTERACTION", "OUTPUT", "CONSTRAINTS"])
    assert pos == sorted(pos)
    assert "Create a product card containing:\n- image\n" in text
    assert "The entire product card acts as the hover target." in text
    assert "When the pointer enters the product card:" in text
    steps = re.findall(r"^(\d+)\. ", text, re.MULTILINE)
    assert steps == [str(i) for i in range(1, 7)]
    assert "1. Move the entire product card upward approximately 8px over approximately 280ms " \
        "using an ease-out curve." in text  # fmt: skip
    assert "Scale the product image from 1.00 to approximately 1.06" in text
    assert "starting approximately 30ms after the product card movement begins" in text
    assert "Move the arrow icon approximately 4px to the right" in text
    assert "smooth and responsive, not springy" in text
    assert "When the pointer leaves the product card, reverse all properties" in text
    assert "(display scale assumed 2x); keep them as targets" in text
    assert llm_prompt.UNKNOWN_TRIGGER_LINE not in text


def test_subsampled_warning_in_notes_and_prompt() -> None:
    """Phase 5: ``frames_subsampled`` shows up in Technical NOTES and the prompt CONSTRAINTS."""
    spec = VARIANTS["subsampled"]()
    tech = technical.render(spec)
    notes = tech[tech.index("NOTES") :]
    assert "analysed at about 40 fps instead of every frame" in notes
    prompt = llm_prompt.render(spec)
    constraints = prompt[prompt.index("CONSTRAINTS") :]
    assert "timing was measured at a reduced frame rate (about 25ms" in constraints
    assert "reduced frame rate" not in llm_prompt.render(sample_spec())


def test_css_structure_prd21() -> None:
    text = css.render(sample_spec())
    assert text.startswith(css.HEADER + "\n")
    assert ".product-card:hover {\n  translate: 0 -8px;" in text
    assert ".product-card:hover .product-card__image {\n  scale: 1.06;" in text
    assert "transition: scale 320ms cubic-bezier(0.16, 1, 0.3, 1) 30ms;" in text
    # forward timing on the active rule, reverse timing on the base rule
    base = text.split(".product-card {\n", 1)[1].split("}", 1)[0]
    active = text.split(".product-card:hover {\n", 1)[1].split("}", 1)[0]
    assert "translate 220ms ease-in-out" in base and "translate 280ms ease-out" in active
    assert "box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16); /* estimated, low confidence */" in text
    assert "transform:" not in text, "use individual transform properties (P8)"
    assert text.count("{") == text.count("}")


# --------------------------------------------------------------------------------------------
# Low-confidence wording and omissions
# --------------------------------------------------------------------------------------------


def test_low_confidence_technical() -> None:
    text = technical.render(low_confidence_spec())
    main, rest = text.split("\nREVERSE\n", 1)
    # uncertain (< 0.3): t2 shadow, t3 scale, t6 opacity -> not in the main body
    assert "PRODUCT IMAGE" not in main and "IMAGE (e2)" not in main
    assert "box-shadow" not in main and "opacity" not in main.split("ICON (e4)")[1]
    uncertain = rest.split("UNCERTAIN OBSERVATIONS\n", 1)[1].split("\nNOTES\n")[0]
    for needle in ("Card (e1), forward: possibly a subtle box-shadow change",
                   "Image (e2), forward: possibly a subtle scale change",
                   "Icon (e4), forward: possibly a subtle opacity change"):  # fmt: skip
        assert needle in uncertain
    assert "Type: possibly hover interaction — low confidence (40%)" in text
    assert "Initial: possibly color: #111111" in text  # low band -> "possibly"
    assert "Change: possibly moves approximately 4px to the right" in text
    assert "Not recorded — assume the forward timings, reversed." in rest
    # fitted bezier with easing confidence < 0.6 -> family keyword only
    assert "cubic-bezier(0.215" not in text and "Easing: ease-out, low confidence" in text
    assert "Element labels are heuristic" in text
    assert "1x as set on upload" in text
    # a relationship whose other members are all uncertain disappears
    rel = rest.split("TIMING RELATIONSHIPS\n", 1)[1].split("\nUNCERTAIN")[0]
    assert "scaling" not in rel
    assert "Card movement and text color animate simultaneously." in rel


def test_low_confidence_prompt() -> None:
    text = llm_prompt.render(low_confidence_spec())
    lines = text.rstrip("\n").splitlines()
    assert lines[-1] == llm_prompt.FINAL_LINE
    assert lines[-2] == llm_prompt.UNKNOWN_TRIGGER_LINE
    steps = [ln for ln in lines if re.match(r"^\d+\. ", ln)]
    assert len(steps) == 3  # t1, t4, t5 (t2, t3, t6 uncertain)
    assert not any("scale" in s.lower() or "shadow" in s or "opacity" in s for s in steps)
    assert "(low confidence)" in steps[1] and "subtly change" in steps[1]
    assert "(medium confidence)" in steps[0] and "cubic-bezier" not in steps[0]
    assert "When the interaction starts (most likely on hover):" in text
    assert "The interaction appears to be: hover interaction (low confidence)." in text
    assert "The reverse transition was not recorded." in text
    optional = text.split("UNCERTAIN (OPTIONAL)\n", 1)[1].split("\nOUTPUT\n")[0]
    assert "a subtle image scaling change (1.00 to 1.06)" in optional
    assert "(display scale 1x as set on upload)" in text


def test_low_confidence_css() -> None:
    text = css.render(low_confidence_spec())
    assert "/* Trigger could not be determined; implemented as hover. */" in text
    assert "/* Reverse not recorded: base rules reuse the forward timings. */" in text
    assert "  scale: " not in text, "uncertain scale must not become a rule"
    assert "/* uncertain (25% confidence), omitted: scale 1.06, 320ms */" in text
    assert "/* uncertain (20% confidence), omitted: box-shadow" in text
    assert "box-shadow 280ms" not in text
    # base rule carries the forward timing when no reverse was recorded
    base = text.split(".card {\n", 1)[1].split("}", 1)[0]
    assert "transition: translate 280ms ease-out;" in base


# --------------------------------------------------------------------------------------------
# LLM prompt OUTPUT section: one self-contained, responsive HTML file
# --------------------------------------------------------------------------------------------

HOVER_MEDIA = "@media (hover: hover) and (pointer: fine)"


def _output_section(text: str) -> str:
    return text.split("\nOUTPUT\n", 1)[1].split("\nCONSTRAINTS\n", 1)[0]


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_prompt_output_section_common(variant: str) -> None:
    text = llm_prompt.render(VARIANTS[variant]())
    assert text.count("\nOUTPUT\n") == 1
    pos = _section_order(text, ["STRUCTURE", "INTERACTION", "OUTPUT", "CONSTRAINTS"])
    assert pos == sorted(pos)
    assert text.rstrip("\n").splitlines()[-1] == llm_prompt.FINAL_LINE
    out = _output_section(text)
    for needle in (
        "one self-contained, responsive HTML file named index.html",
        "one <style> block and, only if the interaction needs it, one inline <script>",
        "No frameworks, CDNs, external fonts, external images or build step",
        "opened directly in a browser",
        # placeholders, not recorded content
        "neutral placeholders instead of real images or copy",
        "solid or gradient block with a fixed aspect-ratio",
        "Do not add UI beyond what is needed to demonstrate the interaction.",
        # responsive layout, fixed motion values
        "mobile-first and fluid (max-width with percentages or clamp())",
        f"working from {llm_prompt.MIN_VIEWPORT_PX}px wide phones to wide desktops",
        '<meta name="viewport">',
        "collapses to one column",
        "keep every motion value above as specified in CSS px and ms",
        "do not scale them with the viewport",
        "durations, delays and easing curves exactly as listed above",
        # accessibility
        "- Reduced motion: under @media (prefers-reduced-motion: reduce), ",
        'aria-hidden="true"',
    ):
        assert needle in out, needle


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_prompt_output_section_restates_no_measurements(variant: str) -> None:
    """OUTPUT repeats no IR number; its only literal is the allow-listed viewport width."""
    out = _output_section(llm_prompt.render(VARIANTS[variant]()))
    assert _numbers_in(out) == [float(llm_prompt.MIN_VIEWPORT_PX)]


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        ("sample", "hover"),
        ("low_confidence", "hover"),
        ("click", "toggle"),
        ("dropdown", "toggle"),
        ("modal", "toggle"),
        ("press", "press"),
    ],  # fmt: skip
)
def test_prompt_input_mode_matches_css_selectors(variant: str, mode: str) -> None:
    spec = VARIANTS[variant]()
    assert llm_prompt.input_mode(spec) == mode
    out = _output_section(llm_prompt.render(spec))
    sheet = css.render(spec)
    assert (HOVER_MEDIA in out) == (mode == "hover") == (":hover" in sheet)
    assert ("aria-expanded" in out) == (mode == "toggle") == ("data-state" in sheet)
    assert (":active" in out) == (mode == "press") == (":active" in sheet)


def test_prompt_output_hover_fallbacks() -> None:
    out = _output_section(llm_prompt.render(sample_spec()))
    assert f"- Hover: wrap the hover styles in {HOVER_MEDIA}." in out
    assert "Make the product card focusable" in out
    assert "apply the same styles on :focus-visible, outside that media query" in out
    assert "let a tap toggle the same state" in out
    assert "Put the forward timings on the hover state and the reverse timings on the base " \
        "state" in out  # fmt: skip
    assert "Use the individual translate and scale properties rather than transform." in out


def test_prompt_output_low_confidence_assumes_hover() -> None:
    out = _output_section(llm_prompt.render(low_confidence_spec()))
    assert f"- Hover (the assumed trigger): wrap the hover styles in {HOVER_MEDIA}." in out
    assert "Make the card focusable" in out
    assert "the base state reuses them because the reverse was not recorded" in out


def test_prompt_output_click_toggle() -> None:
    out = _output_section(llm_prompt.render(VARIANTS["click"]()))
    assert 'make the product card (or a control inside it) a <button type="button"> with ' \
        "aria-expanded." in out  # fmt: skip
    assert 'set data-state="open"' in out and "no hover fallback is needed" in out
    assert "forward timings on the open state and the reverse timings" in out
    assert "Escape" not in out and "role=" not in out


def test_prompt_output_dropdown_and_modal() -> None:
    dropdown = _output_section(llm_prompt.render(VARIANTS["dropdown"]()))
    assert "Close it on Escape and on a click outside." in dropdown
    modal = llm_prompt.render(VARIANTS["modal"]())
    assert "The entire product card is what opens." in modal
    out = _output_section(modal)
    assert "add one plain placeholder <button type=\"button\"> as the trigger, with " \
        "aria-expanded and aria-controls pointing at the product card." in out  # fmt: skip
    assert 'role="dialog" and aria-modal="true"' in out
    assert "return focus to the button" in out


def test_prompt_output_press() -> None:
    out = _output_section(llm_prompt.render(VARIANTS["press"]()))
    assert '- Press: render the product card as a real <button type="button">' in out
    assert "pressed styles on :active" in out
    assert "do not wrap them in a hover media query" in out
    assert "forward timings on the :active state and the release timings" in out


def _uncertain_props(props: set[str]) -> MotionSpec:
    def mutate(s: dict[str, Any]) -> None:
        for t in s["transitions"]:
            if t["property"] in props:
                t["confidence"] = _conf(0.2, 0.2, 0.2, 0.2)

    return _variant(mutate)


def test_prompt_output_reduced_motion_follows_measured_properties() -> None:
    def bullet(spec: MotionSpec) -> str:
        out = _output_section(llm_prompt.render(spec))
        return next(ln for ln in out.splitlines() if ln.startswith("- Reduced motion:"))

    expected = (
        "drop the movement and scaling (leave them out or apply them without a transition) "
        "but keep the opacity, text color, and shadow changes"
    )
    assert expected in bullet(sample_spec())
    # uncertain transitions (< 0.3) are not part of the main text, so not named here either
    assert "drop the movement (" in bullet(low_confidence_spec())
    assert "keep the text color changes" in bullet(low_confidence_spec())
    only_moves = bullet(_uncertain_props({"box-shadow", "color", "opacity"}))
    assert "apply the movement and scaling without a transition" in only_moves
    no_moves = bullet(_uncertain_props({"translateX", "translateY", "scale"}))
    assert "nothing moves; keep the opacity, text color, and shadow changes" in no_moves


# --------------------------------------------------------------------------------------------
# Variants: selectors, stagger, height idiom, overshoot
# --------------------------------------------------------------------------------------------


def test_click_uses_data_state() -> None:
    spec = VARIANTS["click"]()
    text = css.render(spec)
    assert '.product-card[data-state="open"] {' in text
    assert '.product-card[data-state="open"] .product-card__image {' in text
    assert "toggle" in text.lower() and ":hover" not in text
    prompt = llm_prompt.render(spec)
    assert "When the product card is clicked:" in prompt
    assert "When the product card is clicked again (closing)" in prompt


def test_press_uses_active_and_return() -> None:
    spec = VARIANTS["press"]()
    text = css.render(spec)
    assert ".product-card:active {" in text and ":hover" not in text
    prompt = llm_prompt.render(spec)
    assert "While the product card is pressed:" in prompt
    assert "On release, return all properties" in prompt
    assert "round trip" in technical.render(spec)


def test_stagger_css_and_sentence() -> None:
    spec = VARIANTS["stagger"]()
    text = css.render(spec)
    assert "calc(var(--i, 0) * 40ms)" in text
    assert "set --i to each item's index" in text
    assert "staggered approximately 40ms apart" in technical.render(spec)


def test_height_uses_grid_rows_idiom() -> None:
    text = css.render(VARIANTS["height"]())
    assert "display: grid;" in text and "max-height" in text
    assert "grid-template-rows: 1fr;" in text
    assert "grid-template-rows 200ms ease-out 80ms" in text


def test_overshoot_wording() -> None:
    spec = VARIANTS["overshoot"]()
    prompt = llm_prompt.render(spec)
    assert "slight overshoot (spring-like; exact spring not reconstructed)" in prompt
    assert "cubic-bezier(0.34, 1.56, 0.64, 1)" in prompt
    assert "slightly springy" in prompt
    assert "translate 280ms cubic-bezier(0.34, 1.56, 0.64, 1)" in css.render(spec)


# --------------------------------------------------------------------------------------------
# phrasing units
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-8.4, "-8px"),
        (8.5, "9px"),
        (3.0, "3px"),
        (2.74, "2.5px"),
        (2.75, "3px"),
        (1.2, "1px"),
        (0.2, "0px"),
        (-0.0, "0px"),
        (-1.26, "-1.5px"),
    ],  # fmt: skip
)
def test_px(value: float, expected: str) -> None:
    assert phrasing.px(value) == expected


def test_number_formats() -> None:
    assert phrasing.scale_num(1.0) == "1.00"
    assert phrasing.scale_num(1.0649) == "1.06"
    assert phrasing.scale_num(0.955) == "0.96"
    assert phrasing.opacity_num(0.68) == "0.7"
    assert phrasing.opacity_num(0.475) == "0.5"
    assert phrasing.opacity_num(1.0) == "1"
    assert phrasing.ms(284) == "280ms"
    assert phrasing.ms(285) == "290ms"
    assert phrasing.ms(4) == "4ms"
    assert phrasing.pct(0.29) == "29%"
    assert phrasing.bezier((0.0, 0.0, 0.58, 1.0)) == "cubic-bezier(0, 0, 0.58, 1)"
    assert phrasing.css_px(0.1) == "0"
    shadow = Shadow(x=0, y=4, blur=12, spread=2, rgba=(0, 0, 0, 0.081))
    assert phrasing.shadow_css(shadow) == "0 4px 12px 2px rgba(0, 0, 0, 0.08)"
    assert phrasing.shadow_css(None) == "none"


def test_easing_text_rules() -> None:
    kw = Easing(
        keyword="ease-out", cubic_bezier=(0, 0, 0.58, 1), nearest_named="ease-out",
        family="ease-out", rmse=0.01,
    )  # fmt: skip
    fitted = Easing(
        cubic_bezier=(0.16, 1, 0.3, 1), nearest_named="expoOut", family="ease-out", rmse=0.02
    )
    assert phrasing.easing_text(kw, 0.2) == "ease-out"
    assert phrasing.easing_text(fitted, 0.6) == "cubic-bezier(0.16, 1, 0.3, 1) (close to expoOut)"
    assert phrasing.easing_text(fitted, 0.59) == "ease-out"
    assert phrasing.css_easing(fitted, 0.59) == "ease-out"
    assert phrasing.easing_phrase(kw, 0.9) == "using an ease-out curve"
    lin = kw.model_copy(update={"keyword": "linear", "cubic_bezier": (0, 0, 1, 1)})
    assert phrasing.easing_phrase(lin, 0.9) == "using a linear curve"


def test_class_names_bem_and_collisions() -> None:
    names = phrasing.class_names(sample_spec())
    assert names == {
        "e1": "product-card",
        "e2": "product-card__image",
        "e3": "product-card__title",
        "e4": "product-card__arrow-icon",
    }

    def dupes(s: dict[str, Any]) -> None:
        for el in s["elements"][1:]:
            el["label"] = "Product image"

    names = phrasing.class_names(_variant(dupes))
    assert list(names.values()) == [
        "product-card",
        "product-card__image",
        "product-card__image-2",
        "product-card__image-3",
    ]
    fallback = phrasing.class_names(low_confidence_spec())
    assert fallback["e1"] == "card" and fallback["e2"] == "card__image"


# --------------------------------------------------------------------------------------------
# Measured appearance (PLAN-continuous §13)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "appearance"])
def test_no_appearance_no_new_text(variant: str) -> None:
    spec = VARIANTS[variant]()
    assert not technical.has_appearance(spec)
    out = render_all(spec)
    assert "APPEARANCE" not in out.llm_prompt and "APPEARANCE" not in out.technical
    assert "- Appearance:" not in out.llm_prompt
    assert "width:" not in out.css and "body {" not in out.css


def test_has_appearance_triggers() -> None:
    def scene_only(s: dict[str, Any]) -> None:
        s["scene"] = {"viewport_css": {"w": 1440.0, "h": 900.0}, "page_background": None}

    def font_only(s: dict[str, Any]) -> None:
        s["elements"][2]["static"]["font_size_px"] = _mn(16.0, 0.4)

    for mutate in (scene_only, font_only):
        spec = _variant(mutate)
        assert technical.has_appearance(spec)
        assert "\nAPPEARANCE (measured, approximate)\n" in llm_prompt.render(spec)
    prompt = llm_prompt.render(_variant(font_only))
    assert "Recorded viewport" not in prompt and "Page background" not in prompt


def _appearance_block(text: str) -> str:
    return text.split("\nAPPEARANCE (measured, approximate)\n", 1)[1].split("\nINTERACTION\n")[0]


def test_prompt_appearance_block() -> None:
    text = llm_prompt.render(appearance_spec())
    pos = _section_order(
        text, ["STRUCTURE", "APPEARANCE (measured, approximate)", "INTERACTION", "OUTPUT"]
    )
    assert pos == sorted(pos)
    block = _appearance_block(text)
    for needle in (
        "- Recorded viewport: 1440 x 900px.",
        "- Page background: approximately #F3F4F6.",
        "- Product card: 320 x 400px, placed 560px from the left and 240px from the top of the "
        "viewport; background approximately #FFFFFF; corner radius possibly 12px (low "
        "confidence).",
        "- Product image: 320 x 200px, at the top-left corner of the product card; background "
        "possibly #C7D2FE (low confidence).",
        "- Card title: 240 x 28px, placed 24px from the left and 228px from the top of the "
        "product card; font size possibly 18px (low confidence).",
        "- Arrow icon: 20 x 20px, placed 276px from the left and 356px from the top of the "
        "product card.",
        "- Card description: 272 x 40px, placed 24px from the left and 268px from the top of the "
        "product card; text color approximately #6B7280 (medium confidence); font size "
        "approximately 14px (medium confidence).",
    ):
        assert needle in block, needle
    # animated properties are described by the steps, not repeated as static values
    assert "#111111" not in block and "shadow" not in block


def test_prompt_appearance_output_bullet() -> None:
    out = _output_section(llm_prompt.render(appearance_spec()))
    bullet = next(ln for ln in out.splitlines() if ln.startswith("- Appearance:"))
    for needle in (
        "match the sizes, positions, colors, corner radii, shadows and font sizes",
        "Keep that measured layout at the recorded viewport width and wider",
        "reflow responsively below it",
        "system sans-serif stack",
        "Content stays placeholder",
    ):
        assert needle in bullet, needle
    lines = out.splitlines()
    assert lines.index(bullet) == next(i for i, ln in enumerate(lines) if ln.startswith(
        "- Layout:")) + 1  # fmt: skip


def test_technical_appearance_section() -> None:
    text = technical.render(appearance_spec())
    pos = _section_order(
        text, ["INTERACTION", "APPEARANCE (initial state, measured)", "PRODUCT CARD (e1)", "NOTES"]
    )
    assert pos == sorted(pos)
    sec = text.split("\nAPPEARANCE (initial state, measured)\n", 1)[1].split(
        "\nPRODUCT CARD (e1)\n")[0]  # fmt: skip
    for needle in (
        "Viewport: 1440 x 900px (recorded frame, CSS px)",
        "Page background: approximately #F3F4F6 — high confidence (85%)",
        "Product card (e1): 320 x 400px at x 560px, y 240px in the viewport",
        "  Background: approximately #FFFFFF — high confidence (85%)",
        "  Border radius: possibly 12px — low confidence (45%)",
        "Card title (e3): 240 x 28px at x 24px, y 228px inside Product card (e1)",
        "  Font size: possibly 18px — low confidence (45%)",
        "Card description (e5): 272 x 40px at x 24px, y 268px inside Product card (e1)",
    ):
        assert needle in sec, needle
    assert "Shadow" not in sec  # e1 animates box-shadow
    assert "(static)" not in text  # radius / shadow moved into APPEARANCE


def test_css_appearance_declarations() -> None:
    text = css.render(appearance_spec())
    assert "/* Sizes and colors measured at the recorded 1440 x 900px viewport. */" in text
    assert "body {\n  background-color: #F3F4F6; /* estimated, high confidence */\n}" in text
    card = text.split(".product-card {\n", 1)[1].split("}", 1)[0]
    assert card.startswith(
        "  width: 320px;\n  height: 400px;\n"
        "  background-color: #FFFFFF; /* estimated, high confidence */\n"
        "  border-radius: 12px; /* estimated, low confidence */\n"
    )
    title = text.split(".product-card__title {\n", 1)[1].split("}", 1)[0]
    assert "width" not in title and "height" not in title  # text boxes follow their text
    assert "  font-size: 18px; /* estimated, low confidence */" in title
    assert title.count("color:") == 1  # the animated colour only (from the transition)
    # a static element (no transitions) gets a base rule from its appearance alone
    assert (
        ".product-card__description {\n"
        "  color: #6B7280; /* estimated, medium confidence */\n"
        "  font-size: 14px; /* estimated, medium confidence */\n}"
    ) in text
    assert ".product-card:hover .product-card__description" not in text
    assert text.count("{") == text.count("}")


def test_appearance_keeps_motion_text_identical() -> None:
    """Appearance adds text; it never changes the motion description."""
    base = llm_prompt.render(sample_spec())
    with_app = llm_prompt.render(appearance_spec())
    steps = base.split("\nINTERACTION\n", 1)[1].split("\nOUTPUT\n")[0]
    assert steps in with_app
