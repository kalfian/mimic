"""Shared phrasing for the generators (PLAN §9.1). Pure and deterministic.

Number provenance rule: every number a generator prints comes from an IR field, at most
transformed by ``abs``, rounding (the helpers below) or x100 for confidence percentages.
Never compute new magnitudes (sums, ratios, percentages of change) in prose.

Rounding is decimal half-up so output never depends on binary float artefacts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal

from app.models.ir import (
    UNCERTAIN_BELOW,
    ConfidenceBand,
    Easing,
    MotionElement,
    MotionSpec,
    Relationship,
    Segment,
    Shadow,
    Transition,
    Value,
    band_for,
)

# --------------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------------


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def round_step(x: float, step: str) -> Decimal:
    """Round ``x`` to a multiple of ``step`` (decimal string), half away from zero."""
    s = Decimal(step)
    return (_dec(x) / s).quantize(Decimal(1), rounding=ROUND_HALF_UP) * s


def fmt_decimal(d: Decimal) -> str:
    """Plain decimal text without trailing zeros and without ``-0``."""
    if d == 0:
        return "0"
    text = format(d.normalize(), "f")
    return text


def num_px(v: float) -> str:
    """px magnitude: |v| >= 3 -> whole px, otherwise nearest 0.5 (``2.5``, ``1``)."""
    step = "1" if abs(v) >= 3 else "0.5"
    return fmt_decimal(round_step(v, step))


def px(v: float) -> str:
    """``-8px``, ``0px``, ``2.5px``."""
    return f"{num_px(v)}px"


def css_px(v: float) -> str:
    """Like :func:`px` but a zero length is the bare ``0`` (idiomatic CSS)."""
    n = num_px(v)
    return "0" if n == "0" else f"{n}px"


def scale_num(v: float) -> str:
    """Scale factors always show 2 decimals: ``1.00``, ``1.06``, ``0.96``."""
    d = _dec(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return "0.00" if d == 0 else format(d, "f")


def opacity_num(v: float) -> str:
    """Opacity to the nearest 0.05: ``0.7``, ``1``, ``0.45``."""
    return fmt_decimal(round_step(v, "0.05"))


def ms_int(v: float) -> int:
    """Milliseconds rounded to 10 ms; tiny non-zero values keep their raw value."""
    r = int(round_step(v, "10"))
    if r == 0 and v > 0:
        return int(fmt_decimal(round_step(v, "1")))
    return r


def ms(v: float) -> str:
    """``280ms`` (rounded to 10 ms)."""
    return f"{ms_int(v)}ms"


def pct(score: float) -> str:
    """Confidence score as a whole percentage: ``85%``."""
    return f"{fmt_decimal(round_step(score * 100, '1'))}%"


def alpha_num(v: float) -> str:
    """Shadow alpha to 2 decimals without trailing zeros: ``0.08``, ``0.5``."""
    return fmt_decimal(round_step(v, "0.01"))


def bezier_num(v: float) -> str:
    return fmt_decimal(round_step(v, "0.001"))


def bezier(b: Sequence[float]) -> str:
    """``cubic-bezier(0.16, 1, 0.3, 1)``."""
    return "cubic-bezier(" + ", ".join(bezier_num(x) for x in b) + ")"


def shadow_css(sh: Shadow | None) -> str:
    """``0 12px 32px rgba(0, 0, 0, 0.16)`` (spread omitted when zero) or ``none``."""
    if sh is None:
        return "none"
    parts = [css_px(sh.x), css_px(sh.y), css_px(sh.blur)]
    if num_px(sh.spread) != "0":
        parts.append(css_px(sh.spread))
    r, g, b, a = sh.rgba
    parts.append(f"rgba({r}, {g}, {b}, {alpha_num(a)})")
    return " ".join(parts)


# --------------------------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------------------------

TRANSFORM_PROPS = ("translateX", "translateY", "scale", "scaleX", "scaleY")


def value_short(prop: str, v: Value) -> str:
    """Bare value: ``-8px``, ``1.06``, ``0.7``, ``#5252FF``, shadow CSS, quoted text."""
    if v.kind == "px":
        return px(v.number)
    if v.kind == "ratio":
        return opacity_num(v.number) if prop == "opacity" else scale_num(v.number)
    if v.kind == "color":
        return v.color.upper()
    if v.kind == "shadow":
        return shadow_css(v.shadow)
    return f'"{v.text}"'


def value_css(prop: str, v: Value) -> str:
    """Value as a CSS literal (zero lengths as bare ``0``)."""
    if v.kind == "px":
        return css_px(v.number)
    return value_short(prop, v)


def is_identity(prop: str, v: Value) -> bool:
    """Value equals the CSS default (translate 0, scale/opacity 1) after display rounding."""
    if prop in ("translateX", "translateY"):
        return num_px(v.number) == "0"  # type: ignore[union-attr]
    if prop in ("scale", "scaleX", "scaleY", "opacity"):
        return scale_num(v.number) == "1.00"  # type: ignore[union-attr]
    return False


def value_fn(prop: str, v: Value) -> str:
    """Technical notation: ``translateY(-8px)``, ``scale(1.06)``, ``opacity: 0.7``."""
    if prop in TRANSFORM_PROPS:
        return f"{prop}({value_short(prop, v)})"
    return f"{prop}: {value_short(prop, v)}"


# --------------------------------------------------------------------------------------------
# Confidence and easing wording
# --------------------------------------------------------------------------------------------


def is_uncertain(t: Transition) -> bool:
    """``overall < 0.3``: excluded from main text, listed as uncertain, commented out in CSS."""
    return t.confidence.overall < UNCERTAIN_BELOW


def band_note(band: ConfidenceBand) -> str:
    """Suffix for a statement: ``""`` / ``" (medium confidence)"`` / ``" (low confidence)"``."""
    return "" if band == "high" else f" ({band} confidence)"


def value_prefix(band: ConfidenceBand) -> str:
    """Hedge before a measured value: high/medium -> approximately, low -> possibly."""
    return "possibly " if band == "low" else "approximately "


def confidence_text(score: float) -> str:
    """``high (85%)``."""
    return f"{band_for(score)} ({pct(score)})"


def confidence_clause(score: float) -> str:
    """``high confidence (85%)``."""
    return f"{band_for(score)} confidence ({pct(score)})"


#: Timing confidence below which a duration carries an explicit hedge (Phase 4: strong
#: ease-out tails hide in the noise, so their fitted duration can be far off).
DURATION_HEDGE_BELOW = 0.5


def duration_uncertain(t: Transition) -> bool:
    return t.confidence.timing < DURATION_HEDGE_BELOW


def duration_text(t: Transition) -> str:
    """``approximately 280ms`` / ``approximately 200ms (duration uncertain)``."""
    text = f"approximately {ms(t.duration_ms)}"
    return text + " (duration uncertain)" if duration_uncertain(t) else text


def has_overshoot(e: Easing) -> bool:
    return "overshoot" in e.flags or e.family == "overshoot"


#: Easing confidence needed before quoting a fitted (non-keyword) bezier.
EXACT_BEZIER_MIN_CONF = 0.6


def css_easing(e: Easing, easing_conf: float) -> str:
    """CSS timing function: keyword, fitted bezier (if trustworthy / overshoot) or family."""
    if e.keyword is not None:
        return e.keyword
    if has_overshoot(e) or easing_conf >= EXACT_BEZIER_MIN_CONF:
        return bezier(e.cubic_bezier)
    return e.family


def easing_text(e: Easing, easing_conf: float) -> str:
    """Prose easing: ``ease-out`` / ``cubic-bezier(…) (close to expoOut)`` / overshoot hint."""
    if has_overshoot(e):
        return (
            "a slight overshoot (spring-like; exact spring not reconstructed), "
            f"approximate with {bezier(e.cubic_bezier)}"
        )
    if e.keyword is not None:
        return e.keyword
    if easing_conf >= EXACT_BEZIER_MIN_CONF:
        return f"{bezier(e.cubic_bezier)} (close to {e.nearest_named})"
    return e.family


def easing_phrase(e: Easing, easing_conf: float) -> str:
    """Prompt form: ``using an ease-out curve`` / ``using cubic-bezier(…) (close to …)``."""
    if has_overshoot(e):
        return "with " + easing_text(e, easing_conf)
    text = easing_text(e, easing_conf)
    if text.startswith("cubic-bezier"):
        return f"using {text}"
    article = "an" if text[0] in "aeiou" else "a"
    return f"using {article} {text} curve"


# --------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------


def lower_first(s: str) -> str:
    """Lowercase the first letter unless the word looks like an acronym (``API button``)."""
    if not s or (len(s) > 1 and s[1].isupper()):
        return s
    return s[0].lower() + s[1:]


def upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def join_and(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b, and c``."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def display_label(el: MotionElement) -> str:
    """Element label without a trailing ``(eN)`` id tag (fallback labels carry one)."""
    label = re.sub(rf"\s*\({re.escape(el.id)}\)\s*$", "", el.label).strip()
    return label or el.role.replace("_", " ")


def ref_name(el: MotionElement) -> str:
    """Mid-sentence element name: ``product card``."""
    return lower_first(display_label(el))


def heading(el: MotionElement) -> str:
    """Technical block heading: ``PRODUCT CARD (e1)``."""
    return f"{display_label(el).upper()} ({el.id})"


# --------------------------------------------------------------------------------------------
# Spec navigation
# --------------------------------------------------------------------------------------------


def element_map(spec: MotionSpec) -> dict[str, MotionElement]:
    return {e.id: e for e in spec.elements}


def element_order(spec: MotionSpec) -> list[MotionElement]:
    """Depth-first hierarchy order (roots in list order, children in list order)."""
    children: dict[str | None, list[MotionElement]] = {}
    ids = {e.id for e in spec.elements}
    for e in spec.elements:
        parent = e.parent_id if e.parent_id in ids else None
        children.setdefault(parent, []).append(e)
    out: list[MotionElement] = []
    seen: set[str] = set()

    def visit(el: MotionElement) -> None:
        if el.id in seen:
            return
        seen.add(el.id)
        out.append(el)
        for child in children.get(el.id, []):
            visit(child)

    for root in children.get(None, []):
        visit(root)
    for e in spec.elements:  # parent cycles: keep everything, in list order
        visit(e)
    return out


def descendant_ids(spec: MotionSpec, root_id: str) -> set[str]:
    """All element ids below ``root_id`` (excluding it)."""
    out: set[str] = set()
    frontier = [root_id]
    while frontier:
        pid = frontier.pop()
        for e in spec.elements:
            if e.parent_id == pid and e.id not in out and e.id != root_id:
                out.add(e.id)
                frontier.append(e.id)
    return out


def forward_segment(spec: MotionSpec) -> Segment:
    return next(s for s in spec.segments if s.kind == "forward")


def reverse_segment(spec: MotionSpec) -> Segment | None:
    return next((s for s in spec.segments if s.kind == "reverse"), None)


PROPERTY_ORDER = (
    "translateX", "translateY", "scale", "scaleX", "scaleY", "opacity", "color",
    "background-color", "box-shadow", "border-radius", "height", "content",
)  # fmt: skip


def sort_transitions(spec: MotionSpec, ts: Iterable[Transition]) -> list[Transition]:
    """Order by start time, then hierarchy position, then property order."""
    pos = {e.id: i for i, e in enumerate(element_order(spec))}
    return sorted(
        ts,
        key=lambda t: (t.start_ms, pos.get(t.element_id, 999), PROPERTY_ORDER.index(t.property)),
    )


def segment_transitions(spec: MotionSpec, segment_id: str) -> list[Transition]:
    return [t for t in spec.transitions if t.segment_id == segment_id]


def primary_transition(spec: MotionSpec, segment_id: str) -> Transition | None:
    """Reference transition: first id of the segment's relationships, else earliest certain."""
    by_id = {t.id: t for t in spec.transitions}
    for r in spec.relationships:
        if r.segment_id == segment_id:
            ref = by_id[r.transition_ids[0]]
            if not is_uncertain(ref):
                return ref
            break
    main = [t for t in segment_transitions(spec, segment_id) if not is_uncertain(t)]
    return sort_transitions(spec, main)[0] if main else None


def effective_target(spec: MotionSpec) -> MotionElement | None:
    """Interaction target, else the first root element (fallback for selectors/naming)."""
    els = element_map(spec)
    if spec.interaction.target_element_id is not None:
        return els[spec.interaction.target_element_id]
    order = element_order(spec)
    return order[0] if order else None


# --------------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------------


def kebab(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if s and s[0].isdigit():
        s = f"el-{s}"
    return s


def class_names(spec: MotionSpec) -> dict[str, str]:
    """Element id -> CSS class (BEM: ``product-card``, ``product-card__image``).

    A child drops words it shares with its parent's label (``Product image`` under
    ``Product card`` -> ``image``). Collisions get ``-2``, ``-3``… suffixes.
    """
    els = element_map(spec)
    out: dict[str, str] = {}
    used: set[str] = set()

    def root_of(el: MotionElement) -> MotionElement:
        seen: set[str] = set()
        while el.parent_id in els and el.id not in seen:
            seen.add(el.id)
            el = els[el.parent_id]  # type: ignore[index]
        return el

    for el in element_order(spec):
        base = kebab(display_label(el)) or kebab(el.role) or el.id
        root = root_of(el)
        if root.id != el.id:
            parent = els[el.parent_id]  # type: ignore[index]
            parent_words = set(kebab(display_label(parent)).split("-"))
            own = [w for w in base.split("-") if w and w not in parent_words]
            part = "-".join(own) or base
            base = f"{out[root.id]}__{part}"
        name = base
        n = 2
        while name in used:
            name = f"{base}-{n}"
            n += 1
        used.add(name)
        out[el.id] = name
    return out


# --------------------------------------------------------------------------------------------
# Change descriptions
# --------------------------------------------------------------------------------------------

PROP_NOUN = {
    "translateX": "movement",
    "translateY": "movement",
    "scale": "scaling",
    "scaleX": "horizontal scaling",
    "scaleY": "vertical scaling",
    "opacity": "opacity",
    "color": "text color",
    "background-color": "background color",
    "box-shadow": "shadow",
    "border-radius": "corner radius",
    "height": "height",
    "content": "content",
}


def direction_word(prop: str, delta: float) -> str:
    if prop == "translateY":
        return "upward" if delta < 0 else "downward"
    return "to the right" if delta > 0 else "to the left"


def shadow_trend(a: Shadow | None, b: Shadow | None) -> str:
    """``becomes larger and darker`` etc. Only compares IR values, prints none."""
    if a is None and b is None:
        return "stays absent"
    if a is None:
        return "appears"
    if b is None:
        return "disappears"
    size_a, size_b = a.blur + abs(a.y) + a.spread, b.blur + abs(b.y) + b.spread
    words = []
    if size_b > size_a:
        words.append("larger")
    elif size_b < size_a:
        words.append("smaller")
    if b.rgba[3] > a.rgba[3]:
        words.append("darker")
    elif b.rgba[3] < a.rgba[3]:
        words.append("lighter")
    return "becomes " + " and ".join(words) if words else "changes"


def movement_phrase(prop: str, delta: float) -> str:
    """``upward approximately 8px`` / ``approximately 4px to the right`` (PRD wording)."""
    amount = f"approximately {px(abs(delta))}"
    word = direction_word(prop, delta)
    return f"{word} {amount}" if prop == "translateY" else f"{amount} {word}"


def change_summary(t: Transition) -> str:
    """Descriptive one-liner: ``moves upward approximately 8px``, ``shadow becomes larger``."""
    p = t.property
    a, b = t.from_, t.to
    if p in ("translateX", "translateY") and t.delta is not None:
        return f"moves {movement_phrase(p, t.delta)}"
    if p in ("scale", "scaleX", "scaleY"):
        verb = "scales up" if (t.delta or 0) > 0 else "scales down"
        return f"{verb} from {value_short(p, a)} to approximately {value_short(p, b)}"
    if p == "opacity":
        verb = "increases" if (t.delta or 0) > 0 else "decreases"
        return f"opacity {verb} from approximately {value_short(p, a)} to {value_short(p, b)}"
    if p in ("color", "background-color"):
        return (
            f"{PROP_NOUN[p]} changes from approximately {value_short(p, a)} to {value_short(p, b)}"
        )
    if p == "box-shadow":
        return f"shadow {shadow_trend(a.shadow, b.shadow)}"  # type: ignore[union-attr]
    if p in ("height", "border-radius"):
        verb = "grows" if (t.delta or 0) > 0 else "shrinks"
        return (
            f"{PROP_NOUN[p]} {verb} from approximately {value_short(p, a)} to {value_short(p, b)}"
        )
    return f"content changes from {value_short(p, a)} to {value_short(p, b)}"


# --------------------------------------------------------------------------------------------
# Relationship sentences (PRD §14)
# --------------------------------------------------------------------------------------------


def noun_after(name: str, prop: str) -> str:
    """Property noun after an element name, without stutter (``text`` + ``text color``)."""
    noun = PROP_NOUN[prop]
    head, _, tail = noun.partition(" ")
    if tail and name.lower().split(" ")[-1] == head:
        return tail
    return noun


def describe_group(spec: MotionSpec, ts: Sequence[Transition]) -> str:
    """``product card movement and shadow, and arrow icon opacity`` (grouped per element)."""
    els = element_map(spec)
    groups: dict[str, list[str]] = {}
    for t in ts:
        nouns = groups.setdefault(t.element_id, [])
        noun = noun_after(ref_name(els[t.element_id]), t.property)
        if noun not in nouns:
            nouns.append(noun)
    parts = [f"{ref_name(els[eid])} {join_and(nouns)}" for eid, nouns in groups.items()]
    if len(parts) == 2 and any(len(n) > 1 for n in groups.values()):
        return f"{parts[0]}, and {parts[1]}"  # avoid "a and b and c" ambiguity
    return join_and(parts)


def _count_nouns(spec: MotionSpec, ts: Sequence[Transition]) -> int:
    return len({(t.element_id, PROP_NOUN[t.property]) for t in ts})


def relationship_sentence(spec: MotionSpec, r: Relationship) -> str | None:
    """One sentence per relationship; uncertain transitions are left out (None if too few)."""
    by_id = {t.id: t for t in spec.transitions}
    ts = [by_id[i] for i in r.transition_ids]
    ref, others = ts[0], [t for t in ts[1:] if not is_uncertain(t)]
    if is_uncertain(ref) or not others:
        return None
    if r.kind == "simultaneous":
        return upper_first(describe_group(spec, [ref, *others])) + " animate simultaneously."
    verb = "begins" if _count_nouns(spec, others) == 1 else "begin"
    who = upper_first(describe_group(spec, others))
    ref_text = describe_group(spec, [ref])
    if r.kind == "delayed" and r.offset_ms is not None:
        return f"{who} {verb} approximately {ms(r.offset_ms)} after the {ref_text}."
    if r.kind == "sequential" and r.offset_ms is not None:
        return (
            f"{who} {verb} after the {ref_text} finishes "
            f"(approximately {ms(r.offset_ms)} after it starts)."
        )
    if r.kind == "stagger" and r.interval_ms is not None:
        if len(others) + 1 < 3:
            return None
        return (
            f"{upper_first(describe_group(spec, [ref, *others]))} are staggered "
            f"approximately {ms(r.interval_ms)} apart, starting with the {ref_text}."
        )
    return None


def relationship_sentences(spec: MotionSpec, segment_id: str) -> list[str]:
    out = []
    for r in spec.relationships:
        if r.segment_id == segment_id:
            s = relationship_sentence(spec, r)
            if s is not None:
                out.append(s)
    return out


# --------------------------------------------------------------------------------------------
# Interaction wording
# --------------------------------------------------------------------------------------------

TYPE_TITLE = {
    "hover": "Hover interaction",
    "click": "Click interaction",
    "press": "Press (button micro-interaction)",
    "expand_collapse": "Expand / collapse",
    "dropdown": "Dropdown open",
    "modal": "Modal open",
    "unknown": "Unknown interaction type",
}

#: Label of the active state per interaction type ("Initial" / <this>).
ACTIVE_LABEL = {
    "hover": "Hover",
    "click": "Active",
    "press": "Pressed",
    "expand_collapse": "Expanded",
    "dropdown": "Open",
    "modal": "Open",
    "unknown": "Active",
}

#: Interaction types implemented as a JS-toggled open state.
TOGGLE_TYPES = frozenset({"click", "dropdown", "modal", "expand_collapse"})


def pixel_ratio_sentence(spec: MotionSpec) -> str:
    src = spec.source
    if src.pixel_ratio_source == "auto":
        return f"display scale assumed {src.pixel_ratio}x"
    return f"display scale {src.pixel_ratio}x as set on upload"
