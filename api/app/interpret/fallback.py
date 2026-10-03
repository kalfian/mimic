"""Deterministic labels / roles / hierarchy / structure without an LLM (PLAN §7.6).

Used when AI labeling is off (``status="disabled"``, the default — PLAN P9), when the CLI is
unavailable (``"fallback"``), and as the complete baseline every ``ClaudeCliInterpreter``
failure returns (``"timeout"`` / ``"error"``). Same input -> same output, no clock except the
caller-supplied ``duration_ms``.

Role rules, first match wins (CSS px; frame = video size / pixel ratio):

1. kind ``backdrop`` -> ``backdrop``
2. ``appear`` + large (>= 10 % of the frame) + centered (center within 20 % of frame center)
   -> ``modal_panel``
3a. ≥ 3 similar-size ``appear`` elements stacked in one column -> ``list_item`` (Phase 4)
3. ``appear`` below a trigger (top within 24 px under a non-appear element it overlaps
   horizontally, or the heuristic type is ``dropdown``) -> ``dropdown_menu``
4. largest ``transform`` element that has children and area > 20 000 px² -> ``card`` (Phase 4:
   without such an element and with nothing opening, the largest root ``transform`` element
   over 20 000 px²)
5. ``resize`` -> ``accordion_panel`` (addition to §7.6: the resize kind only comes from the
   expand/collapse pattern)
6. high-texture, colorful child (Laplacian variance + saturation of ``el_<id>.png``) -> ``image``
7. text-like -> ``text``
8. small (< 240 x 72) with a fill change (``photometric`` or a color change), or a small moving
   root element (a pressed button) -> ``button``
9. else ``other``

The interpreter input has no cursor coordinates, only ``cursor_summary`` text; the target is
the first known element id mentioned there (e.g. "enters e1 at 1120 ms"), else the largest
``transform`` element, else the largest element.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from app.interpret.base import (
    DESCRIPTION_MAX,
    ElementSummary,
    InterpretationInput,
    InterpretationResult,
    InterpretedElement,
    InterpretedInteraction,
)
from app.models.ir import InterpretationStatus, InterpreterProvider, Role, TriggerKind

CARD_MIN_AREA = 20_000.0
MODAL_MIN_AREA_FRAC = 0.10
MODAL_CENTER_TOL = 0.20
DROPDOWN_GAP_PX = 24.0
BUTTON_MAX_W, BUTTON_MAX_H = 240.0, 72.0
IMAGE_MIN_LAPLACIAN_VAR = 150.0
IMAGE_MIN_SATURATION = 35.0

#: Per-role confidence of the geometric rule that produced it.
ROLE_CONFIDENCE: dict[str, float] = {
    "backdrop": 0.7,
    "modal_panel": 0.5,
    "dropdown_menu": 0.5,
    "card": 0.4,
    "accordion_panel": 0.4,
    "image": 0.4,
    "text": 0.4,
    "button": 0.35,
    "other": 0.2,
}

TRIGGER_TEXT: dict[str, str] = {
    "pointer_enter": "Pointer enters the target",
    "click": "Click on the target",
    "unknown": "Trigger could not be determined",
}

STATUS_NOTE: dict[str, str] = {
    "disabled": "AI labeling is off; labels are heuristic.",
    "fallback": "AI labeling unavailable; labels are heuristic.",
    "timeout": "AI labeling timed out; labels are heuristic.",
    "error": "AI labeling failed; labels are heuristic.",
}

_ID_RE = re.compile(r"\be[1-9][0-9]*\b")


def _area(el: ElementSummary) -> float:
    return el.bbox.w * el.bbox.h


def _frame_css(inp: InterpretationInput) -> tuple[float, float]:
    m = inp.video_meta
    return m.width / m.pixel_ratio, m.height / m.pixel_ratio


def _overlap_x(a: ElementSummary, b: ElementSummary) -> bool:
    return min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w) > max(a.bbox.x, b.bbox.x)


def _is_modal_like(el: ElementSummary, fw: float, fh: float) -> bool:
    if el.kind != "appear" or fw <= 0 or fh <= 0:
        return False
    if _area(el) < MODAL_MIN_AREA_FRAC * fw * fh:
        return False
    cx, cy = el.bbox.x + el.bbox.w / 2, el.bbox.y + el.bbox.h / 2
    return abs(cx - fw / 2) <= MODAL_CENTER_TOL * fw and abs(cy - fh / 2) <= MODAL_CENTER_TOL * fh


def _is_below_trigger(el: ElementSummary, others: list[ElementSummary]) -> bool:
    for o in others:
        if o.id == el.id or o.kind == "appear":
            continue
        gap = el.bbox.y - (o.bbox.y + o.bbox.h)
        if -2.0 <= gap <= DROPDOWN_GAP_PX and _overlap_x(el, o):
            return True
    return False


def _has_fill_change(el: ElementSummary) -> bool:
    return el.kind == "photometric" or "color" in el.change_summary.lower()


def _texture(path: Path | None) -> tuple[float, float] | None:
    """(Laplacian variance, mean HSV saturation) of an element crop, or None."""
    if path is None or not path.is_file():
        return None
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        return lap, sat
    except Exception:  # noqa: BLE001 - texture is a hint only; never fail the fallback
        return None


def _element_image(inp: InterpretationInput, el_id: str) -> Path | None:
    path = inp.keyframe_paths.get(f"el_{el_id}.png")
    if path is None:
        return None
    root = (inp.job_dir / "keyframes").resolve()
    resolved = path.resolve()
    return resolved if resolved.parent == root else None


def _list_items(appears: list[ElementSummary]) -> set[str]:
    """≥ 3 appearing elements of similar size (±30 %) stacked in one column -> list items."""
    if len(appears) < 3:
        return set()
    areas = sorted(_area(e) for e in appears)
    med = areas[len(areas) // 2]
    similar = [e for e in appears if med > 0 and abs(_area(e) - med) <= 0.3 * med]
    if len(similar) < 3:
        return set()
    xs = [e.bbox.x for e in similar]
    if max(xs) - min(xs) > 0.2 * max(e.bbox.w for e in similar):
        return set()
    return {e.id for e in similar}


def assign_roles(inp: InterpretationInput) -> dict[str, Role]:
    """Role per element id (rules in the module docstring)."""
    els = inp.elements
    fw, fh = _frame_css(inp)
    child_parents = {e.parent_id for e in els if e.parent_id is not None}
    card_candidates = [
        e
        for e in els
        if e.kind == "transform" and e.id in child_parents and _area(e) > CARD_MIN_AREA
    ]
    opener = any(e.kind in ("resize", "appear", "backdrop") for e in els)
    if not card_candidates and not opener:
        # a lone large moving root (a card lifting on hover) is a card too; not when something
        # opens (rows pushed by an accordion are not cards)
        card_candidates = [
            e for e in els if e.kind == "transform" and e.parent_id is None
            and _area(e) > CARD_MIN_AREA
        ]  # fmt: skip
    card_id = max(card_candidates, key=_area).id if card_candidates else None
    appears = [e for e in els if e.kind == "appear"]
    list_ids = _list_items(appears)

    roles: dict[str, Role] = {}
    for el in els:
        role: Role
        if el.kind == "backdrop":
            role = "backdrop"
        elif _is_modal_like(el, fw, fh):
            role = "modal_panel"
        elif el.id in list_ids:
            role = "list_item"
        elif el.kind == "appear" and (
            inp.heuristic_type == "dropdown" or _is_below_trigger(el, els)
        ):
            role = "dropdown_menu"
        elif el.id == card_id:
            role = "card"
        elif el.kind == "resize":
            role = "accordion_panel"
        elif (
            el.parent_id is not None
            and not el.text_like
            and (tex := _texture(_element_image(inp, el.id))) is not None
            and tex[0] >= IMAGE_MIN_LAPLACIAN_VAR
            and tex[1] >= IMAGE_MIN_SATURATION
        ):
            role = "image"
        elif el.text_like:
            role = "text"
        elif (
            el.bbox.w < BUTTON_MAX_W
            and el.bbox.h < BUTTON_MAX_H
            and (_has_fill_change(el) or (el.kind == "transform" and el.parent_id is None))
        ):
            role = "button"
        else:
            role = "other"
        roles[el.id] = role
    return roles


def role_title(role: str) -> str:
    """``dropdown_menu`` -> ``Dropdown menu``; ``other`` -> ``Element``."""
    if role == "other":
        return "Element"
    return role.replace("_", " ").capitalize()


def pick_target(inp: InterpretationInput) -> str | None:
    """First known id in the cursor summary, else largest transform element, else largest."""
    known = {e.id for e in inp.elements}
    for m in _ID_RE.finditer(inp.cursor_summary):
        if m.group(0) in known:
            return m.group(0)
    transforms = [e for e in inp.elements if e.kind == "transform"]
    pool = transforms or list(inp.elements)
    return max(pool, key=_area).id if pool else None


def _structure(inp: InterpretationInput, roles: dict[str, Role], target: str | None) -> list[str]:
    if target is None:
        return []
    out: list[str] = []
    for el in inp.elements:
        if el.parent_id == target:
            name = roles[el.id].replace("_", " ")
            if name not in out:
                out.append(name)
    return out[:12]


def build_fallback(
    inp: InterpretationInput,
    *,
    status: InterpretationStatus,
    provider: InterpreterProvider = "none",
    model: str | None = None,
    duration_ms: int | None = None,
    notes: Iterable[str] = (),
) -> InterpretationResult:
    """Complete deterministic result: every input element labeled, interaction set."""
    roles = assign_roles(inp)
    target = pick_target(inp)
    # Repeated roles get reading-order numbers ("List item 1 (e3)") so every name in the
    # generated text points at one element.
    numbers: dict[str, int] = {}
    for role in set(roles.values()):
        same = [e for e in inp.elements if roles[e.id] == role]
        if len(same) > 1:
            for n, e in enumerate(sorted(same, key=lambda e: (e.bbox.y, e.bbox.x)), start=1):
                numbers[e.id] = n
    elements: dict[str, InterpretedElement] = {}
    for el in inp.elements:
        role = roles[el.id]
        summary = " ".join(el.change_summary.split())
        title = role_title(role) + (f" {numbers[el.id]}" if el.id in numbers else "")
        elements[el.id] = InterpretedElement(
            label=f"{title} ({el.id})",
            role=role,
            parent_id=el.parent_id,
            description=summary[:DESCRIPTION_MAX] or None,
            confidence=ROLE_CONFIDENCE.get(role, 0.2),
            label_source="heuristic",
        )

    trigger: TriggerKind = inp.heuristic_trigger
    interaction = InterpretedInteraction(
        type_confirmation=inp.heuristic_type,
        type_confidence=0.0,  # no independent confirmation; assemble keeps the heuristic
        target_element_id=target,
        target_description=elements[target].label if target is not None else None,
        trigger_description=TRIGGER_TEXT[trigger],
    )

    all_notes = [STATUS_NOTE.get(status, "Labels are heuristic.")] if status != "ok" else []
    all_notes += [" ".join(n.split())[:DESCRIPTION_MAX] for n in notes if n and n.strip()]
    return InterpretationResult(
        status=status,
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        elements=elements,
        interaction=interaction,
        structure=_structure(inp, roles, target),
        notes=list(dict.fromkeys(all_notes))[:5],
    )


class FallbackInterpreter:
    """Interpreter that never calls out. ``status`` is ``disabled`` (opt-out) or ``fallback``."""

    provider: InterpreterProvider = "none"

    def __init__(self, status: InterpretationStatus = "disabled", reason: str | None = None):
        self.status: InterpretationStatus = status
        self.reason = reason

    def interpret(self, inp: InterpretationInput) -> InterpretationResult:
        notes = [self.reason] if self.reason else []
        return build_fallback(inp, status=self.status, provider="none", notes=notes)
