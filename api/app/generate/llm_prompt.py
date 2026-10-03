"""LLM-ready prompt (PRD §19, PLAN §9.3) — the primary MVP output. Pure template over the IR.

The prompt asks the coding LLM for one self-contained, responsive ``index.html`` (OUTPUT
section) built with neutral placeholders; the motion numbers stay where they were measured.
"""

from __future__ import annotations

from app.generate import phrasing as ph
from app.models.ir import MotionElement, MotionSpec, Transition

OPENING_LINE = (
    "Recreate the reference UI interaction as a single self-contained, responsive HTML file,"
    " as follows."
)
FINAL_LINE = "Do not invent additional animations that are not described above."
UNKNOWN_TRIGGER_LINE = (
    "The trigger could not be determined; implement as hover unless the component is a menu/modal."
)


def _name(spec: MotionSpec, el: MotionElement, whole: bool = False) -> str:
    """``the product card``; ``the entire product card`` when ``whole`` and it has children."""
    target = spec.interaction.target_element_id
    if whole and el.id == target and ph.descendant_ids(spec, el.id):
        return f"the entire {ph.ref_name(el)}"
    return f"the {ph.ref_name(el)}"


def _action(spec: MotionSpec, t: Transition, el: MotionElement) -> tuple[str, str]:
    """(verb, rest) for one transition, e.g. (``Move``, ``the card upward approximately 8px``)."""
    p, a, b = t.property, t.from_, t.to
    name = _name(spec, el, whole=p in ph.TRANSFORM_PROPS)
    if p in ("translateX", "translateY") and t.delta is not None:
        rest = f"{name} {ph.movement_phrase(p, t.delta)}"
        if a.number != 0:  # type: ignore[union-attr]
            rest += f" ({p} from {ph.value_short(p, a)} to {ph.value_short(p, b)})"
        return "Move", rest
    if p in ("scale", "scaleX", "scaleY"):
        axis = {"scale": "", "scaleX": " horizontally", "scaleY": " vertically"}[p]
        rest = f"{name}{axis} from {ph.value_short(p, a)} to approximately {ph.value_short(p, b)}"
        if t.transform_origin == "center":
            rest += " around its center"
        elif t.transform_origin is not None:
            rest += f" with transform-origin {t.transform_origin}"
        return "Scale", rest
    if p == "opacity":
        if a.number == 0:  # type: ignore[union-attr]
            return "Fade in", f"{name} (opacity {ph.value_short(p, a)} to {ph.value_short(p, b)})"
        if b.number == 0:  # type: ignore[union-attr]
            return "Fade out", f"{name} (opacity {ph.value_short(p, a)} to {ph.value_short(p, b)})"
        verb = "Increase" if (t.delta or 0) > 0 else "Decrease"
        return verb, (
            f"{name} opacity from approximately {ph.value_short(p, a)} to {ph.value_short(p, b)}"
        )
    if p in ("color", "background-color"):
        return "Change", (
            f"{name} {ph.noun_after(name, p)} from approximately {ph.value_short(p, a)} to "
            f"{ph.value_short(p, b)}"
        )
    if p == "box-shadow":
        trend = ph.shadow_trend(a.shadow, b.shadow)  # type: ignore[union-attr]
        verb = {"becomes larger and darker": "Increase", "becomes smaller and lighter": "Reduce"}
        return verb.get(trend, "Change"), (
            f"{name} shadow from approximately {ph.value_short(p, a)} to {ph.value_short(p, b)}"
        )
    if p == "height":
        verb = "Expand" if (t.delta or 0) > 0 else "Collapse"
        return verb, (
            f"{name} height from approximately {ph.value_short(p, a)} to {ph.value_short(p, b)}"
        )
    if p == "border-radius":
        return "Change", (
            f"{name} corner radius from approximately {ph.value_short(p, a)} to "
            f"{ph.value_short(p, b)}"
        )
    return "Swap", f"{name} content from {ph.value_short(p, a)} to {ph.value_short(p, b)}"


def _structure(spec: MotionSpec) -> list[str]:
    els = ph.element_map(spec)
    target = (
        els[spec.interaction.target_element_id]
        if spec.interaction.target_element_id is not None
        else None
    )
    lines = ["STRUCTURE", ""]
    animated = [
        ph.ref_name(e)
        for e in ph.element_order(spec)
        if any(t.element_id == e.id for t in spec.transitions)
    ]
    if target is None:
        lines.append("Create a component with these animated parts:")
        lines += [f"- {n}" for n in animated]
        return lines
    inside = ph.descendant_ids(spec, target.id) | {target.id}
    outside = [
        ph.ref_name(e)
        for e in ph.element_order(spec)
        if e.id not in inside and any(t.element_id == e.id for t in spec.transitions)
    ]
    if outside and not spec.structure:
        # several independent animated elements (a staggered list, a backdrop + its dialog)
        lines.append("Create these elements:")
        roots = [target] + [e for e in ph.element_order(spec) if ph.ref_name(e) in outside]
        roots.sort(key=lambda e: (e.bbox_initial.y, e.bbox_initial.x, e.id))
        lines += [f"- {ph.ref_name(e)}" for e in roots]
        return lines
    parts = list(spec.structure) or [
        ph.ref_name(els[i]) for i in sorted(ph.descendant_ids(spec, target.id))
    ]
    article = "an" if ph.ref_name(target)[:1] in "aeiou" else "a"
    if parts:
        lines.append(f"Create {article} {ph.ref_name(target)} containing:")
        lines += [f"- {p}" for p in parts]
    else:
        lines.append(f"Create {article} {ph.ref_name(target)}.")
    if (spec.structure or outside) and animated:
        lines += ["", f"Animated elements referenced below: {', '.join(animated)}."]
    return lines


def _target_sentence(spec: MotionSpec) -> str:
    it = spec.interaction
    els = ph.element_map(spec)
    if it.target_element_id is None:
        return "The interaction target could not be identified from the recording."
    target = els[it.target_element_id]
    name = ph.upper_first(_name(spec, target, whole=True))
    if _opens(spec):
        return (
            f"{name} is what opens. The control that triggers it did not change visibly, so it "
            "is not described here; use the existing trigger of your component."
        )
    role = {
        "hover": "acts as the hover target",
        "press": "is the press target",
        "unknown": "appears to be the interaction target",
    }.get(it.type, "is the click target (it toggles the open state)")
    return f"{name} {role}."


def _opens(spec: MotionSpec) -> bool:
    """The target is the opened surface (menu / modal / panel), not the control clicked."""
    it = spec.interaction
    if it.target_element_id is None or it.type not in ("dropdown", "modal", "expand_collapse"):
        return False
    el = ph.element_map(spec)[it.target_element_id]
    return el.kind in ("appear", "resize", "backdrop")


def _forward_intro(spec: MotionSpec) -> str:
    it = spec.interaction
    els = ph.element_map(spec)
    target = (
        _name(spec, els[it.target_element_id])
        if it.target_element_id is not None
        else "the component"
    )
    if _opens(spec):
        target = "the trigger"
    kind = it.trigger.kind
    if it.type == "press":
        text = f"While {target} is pressed"
    elif kind == "pointer_enter":
        text = f"When the pointer enters {target}"
    elif kind == "click":
        text = f"When {target} is clicked"
    elif it.type == "hover":
        text = "When the interaction starts (most likely on hover)"
    else:
        text = "When the interaction starts"
    if it.trigger.confidence.band == "low" and kind != "unknown":
        text += " (trigger inferred with low confidence)"
    return text + ":"


def _step(spec: MotionSpec, t: Transition, prefix: str, start: str) -> str:
    el = ph.element_map(spec)[t.element_id]
    verb, rest = _action(spec, t, el)
    band = t.confidence.band
    if band == "low":
        verb = f"subtly {verb.lower()}"
    if prefix:
        verb = f"{prefix}{verb[0].lower()}{verb[1:]}"
    text = (
        f"{ph.upper_first(verb)} {rest} over {ph.duration_text(t)} "
        f"{ph.easing_phrase(t.easing, t.confidence.easing)}"
    )
    if start:
        text += f", {start}"
    return text + ph.band_note(band) + "."


def _start_phrase(spec: MotionSpec, t: Transition) -> str:
    """When a delayed step starts; offsets come from relationships, else ``delay_ms``."""
    for r in spec.relationships:
        if r.segment_id == t.segment_id and t.id in r.transition_ids[1:]:
            ref = next(x for x in spec.transitions if x.id == r.transition_ids[0])
            ref_text = ph.describe_group(spec, [ref])
            if r.kind == "delayed" and r.offset_ms:
                return f"starting approximately {ph.ms(r.offset_ms)} after the {ref_text} begins"
            if r.kind == "sequential" and r.offset_ms is not None:
                return f"starting after the {ref_text} finishes"
            if r.kind == "stagger" and r.interval_ms is not None:
                return (
                    f"staggered approximately {ph.ms(r.interval_ms)} after the previous item "
                    f"(first: the {ref_text})"
                )
    if ph.ms_int(t.delay_ms) > 0:
        return f"starting approximately {ph.ms(t.delay_ms)} after the animation begins"
    return ""


def _steps(spec: MotionSpec) -> list[str]:
    fwd = ph.forward_segment(spec)
    main = [t for t in ph.segment_transitions(spec, fwd.id) if not ph.is_uncertain(t)]
    if not main:
        return ["No reliable motion was measured; describe the change only from the reference."]
    lines = []
    prev_delay: int | None = None
    for i, t in enumerate(ph.sort_transitions(spec, main), start=1):
        delay = ph.ms_int(t.delay_ms)
        start = _start_phrase(spec, t)
        prefix = "Simultaneously, " if prev_delay is not None and delay == prev_delay else ""
        if prefix and start.startswith("starting"):
            start = ""  # same start as the previous step, already said by "Simultaneously"
        lines.append(f"{i}. {_step(spec, t, prefix, start)}")
        prev_delay = delay
    return lines


def _feel(spec: MotionSpec) -> str:
    fwd = ph.forward_segment(spec)
    main = [t for t in ph.segment_transitions(spec, fwd.id) if not ph.is_uncertain(t)]
    if any(ph.has_overshoot(t.easing) for t in main):
        return "The animation should feel slightly springy (a small overshoot), not bouncy."
    primary = ph.primary_transition(spec, fwd.id)
    if primary is None:
        return "Keep the motion subtle."
    fam = primary.easing.family
    if fam == "linear":
        return "The animation should feel mechanical and linear."
    quick = primary.duration_ms <= 400
    if fam in ("ease-out", "ease"):
        return (
            "The animation should feel smooth and responsive, not springy."
            if quick
            else "The animation should feel smooth and unhurried, not springy."
        )
    if fam == "ease-in-out":
        return "The animation should feel smooth and even, not springy."
    return "The animation should accelerate gently into its end state, not springy."


def _reverse(spec: MotionSpec) -> list[str]:
    it = spec.interaction
    els = ph.element_map(spec)
    target = (
        _name(spec, els[it.target_element_id])
        if it.target_element_id is not None
        else "the component"
    )
    if _opens(spec):
        target = "the trigger"
    rev = ph.reverse_segment(spec)
    rk = it.trigger.reverse_kind
    if it.direction == "round_trip":
        when = "On release"
    elif rk == "pointer_leave":
        when = f"When the pointer leaves {target}"
    elif rk == "click":
        when = f"When {target} is clicked again (closing)"
    else:
        when = "When the interaction ends"
    if rev is None:
        return [
            f"The reverse transition was not recorded. {when}, reverse all properties to "
            "their original states, mirroring the timings above."
        ]
    main = [t for t in ph.segment_transitions(spec, rev.id) if not ph.is_uncertain(t)]
    verb = "return" if it.direction == "round_trip" else "reverse"
    if not main:
        return [
            f"{when}, {verb} all properties to their original states, mirroring the timings above."
        ]
    groups: dict[tuple[int, str, int], list[Transition]] = {}
    for t in ph.sort_transitions(spec, main):
        key = (
            ph.ms_int(t.duration_ms),
            ph.easing_phrase(t.easing, t.confidence.easing),
            ph.ms_int(t.delay_ms),
        )
        groups.setdefault(key, []).append(t)

    def timing(key: tuple[int, str, int], ts: list[Transition]) -> str:
        dur, easing, delay = key
        text = f"over approximately {dur}ms {easing}"
        if any(ph.duration_uncertain(t) for t in ts):
            text += " (duration uncertain)"
        if delay > 0:
            text += f", starting approximately {delay}ms later"
        if all(t.confidence.band == "low" for t in ts):
            text += " (low confidence)"
        return text

    if len(groups) == 1:
        ((key, ts),) = groups.items()
        return [f"{when}, {verb} all properties to their original states {timing(key, ts)}."]
    lines = [f"{when}, {verb} all properties to their original states:"]
    for key, ts in groups.items():
        lines.append(f"- {ph.upper_first(ph.describe_group(spec, ts))}: {timing(key, ts)}.")
    return lines


def _uncertain(spec: MotionSpec) -> list[str]:
    fwd = ph.forward_segment(spec)
    items = [t for t in ph.segment_transitions(spec, fwd.id) if ph.is_uncertain(t)]
    if not items:
        return []
    lines = [
        "UNCERTAIN (OPTIONAL)",
        "",
        "Possibly also (very low confidence; include only if it matches the reference):",
    ]
    for t in ph.sort_transitions(spec, items):
        lines.append(
            f"- a subtle {ph.describe_group(spec, [t])} change "
            f"({ph.value_short(t.property, t.from_)} to {ph.value_short(t.property, t.to)})"
        )
    return lines


# --------------------------------------------------------------------------------------------
# OUTPUT: the deliverable asked of the coding LLM (one self-contained, responsive HTML file)
# --------------------------------------------------------------------------------------------

#: Narrowest viewport the HTML must support. The only number in OUTPUT that does not come from
#: the IR (generic layout guidance; allow-listed explicitly by the provenance test).
MIN_VIEWPORT_PX = 320

#: Properties that move or resize an element: dropped / made instant under reduced motion.
_MOTION_NOUN = {
    "translateX": "movement",
    "translateY": "movement",
    "scale": "scaling",
    "scaleX": "scaling",
    "scaleY": "scaling",
}


def input_mode(spec: MotionSpec) -> str:
    """``hover`` / ``press`` / ``toggle``; same mapping as the CSS tab's selectors."""
    it = spec.interaction
    if it.type in ph.TOGGLE_TYPES:
        return "toggle"
    if it.type == "press":
        return "press"
    return "hover"  # hover / unknown -> hover (UNKNOWN_TRIGGER_LINE, CSS header)


def _main_forward(spec: MotionSpec) -> list[Transition]:
    fwd = ph.forward_segment(spec)
    return [t for t in ph.segment_transitions(spec, fwd.id) if not ph.is_uncertain(t)]


def _target_ref(spec: MotionSpec) -> str:
    target = ph.effective_target(spec)
    return _name(spec, target) if target is not None else "the component"


def _motion_bullet(spec: MotionSpec) -> str:
    mode = input_mode(spec)
    state = {"hover": "hover state", "press": ":active state", "toggle": "open state"}[mode]
    text = (
        "- Motion: implement the steps as CSS transitions with the durations, delays and easing "
        "curves exactly as listed above. "
    )
    if ph.reverse_segment(spec) is None:
        text += (
            f"Put the forward timings on the {state}; the base state reuses them because the "
            "reverse was not recorded."
        )
    else:
        back = "release" if spec.interaction.direction == "round_trip" else "reverse"
        text += (
            f"Put the forward timings on the {state} and the {back} timings on the base state "
            "(CSS uses the transition of the state being entered)."
        )
    if any(t.property in ph.TRANSFORM_PROPS for t in _main_forward(spec)):
        text += " Use the individual translate and scale properties rather than transform."
    return text


def _input_bullet(spec: MotionSpec) -> str:
    it = spec.interaction
    mode = input_mode(spec)
    target = _target_ref(spec)
    if mode == "hover":
        assumed = it.type == "unknown" or it.trigger.kind == "unknown"
        label = "Hover (the assumed trigger)" if assumed else "Hover"
        return (
            f"- {label}: wrap the hover styles in @media (hover: hover) and (pointer: fine). Make "
            f"{target} focusable (for example render it as a link) and apply the same styles on "
            ":focus-visible, outside that media query, so keyboard users get the effect. On touch "
            "screens there is no hover: let a tap toggle the same state (a class set by a few "
            "lines of inline script) so the effect is still reachable."
        )
    if mode == "press":
        return (
            f'- Press: render {target} as a real <button type="button"> and put the pressed '
            "styles on :active, which works with mouse and touch; do not wrap them in a "
            "hover media query."
        )
    if _opens(spec):
        text = (
            "- Click: the recording does not show the control, so add one plain placeholder "
            f'<button type="button"> as the trigger, with aria-expanded and aria-controls '
            f"pointing at {target}."
        )
    else:
        text = (
            f'- Click: make {target} (or a control inside it) a <button type="button"> with '
            "aria-expanded."
        )
    text += (
        " Toggle the open state with a few lines of inline script that flip aria-expanded and "
        'set data-state="open"; a button works with mouse, touch and keyboard alike, so no '
        "hover fallback is needed."
    )
    if it.type == "dropdown":
        text += " Close it on Escape and on a click outside."
    elif it.type == "modal":
        text += (
            ' Give the panel role="dialog" and aria-modal="true", close it on Escape and on '
            "a backdrop click, and return focus to the button."
        )
    return text


def _reduced_motion_bullet(spec: MotionSpec) -> str:
    props = sorted({t.property for t in _main_forward(spec)}, key=ph.PROPERTY_ORDER.index)
    moving = list(dict.fromkeys(_MOTION_NOUN[p] for p in props if p in _MOTION_NOUN))
    kept = list(dict.fromkeys(ph.PROP_NOUN[p] for p in props if p not in _MOTION_NOUN))
    text = "- Reduced motion: under @media (prefers-reduced-motion: reduce), "
    if moving and kept:
        return text + (
            f"drop the {ph.join_and(moving)} (leave them out or apply them without a "
            f"transition) but keep the {ph.join_and(kept)} changes so the state change stays "
            "visible."
        )
    if moving:
        return text + (
            f"apply the {ph.join_and(moving)} without a transition (straight to the end "
            "values), so the state still changes without motion."
        )
    if kept:
        return text + f"nothing moves; keep the {ph.join_and(kept)} changes, optionally shorter."
    return text + "remove movement and scaling but keep the state change visible."


def _output(spec: MotionSpec) -> list[str]:
    return [
        "OUTPUT",
        "",
        "Build this as one self-contained, responsive HTML file named index.html: the markup, "
        "one <style> block and, only if the interaction needs it, one inline <script>. No "
        "frameworks, CDNs, external fonts, external images or build step; the file must work "
        "when opened directly in a browser.",
        "",
        "- Content: build only the structure above, with neutral placeholders instead of real "
        "images or copy: images as a solid or gradient block with a fixed aspect-ratio, text as "
        'short role-named or lorem-style text (for example "Title", "Description"), icons as a '
        "simple inline SVG. Do not add UI beyond what is needed to demonstrate the interaction.",
        "- Layout: mobile-first and fluid (max-width with percentages or clamp()), working from "
        f"{MIN_VIEWPORT_PX}px wide phones to wide desktops, with the standard responsive "
        '<meta name="viewport"> tag. The layout may reflow (for example a row of cards '
        "collapses to one column), but keep every motion value above as specified in CSS px "
        "and ms: they are measured design values, so do not scale them with the viewport.",
        _motion_bullet(spec),
        _input_bullet(spec),
        _reduced_motion_bullet(spec),
        "- Semantics: use semantic elements (article, a, button, headings, p) and mark "
        'decorative placeholders with aria-hidden="true".',
    ]


def render(spec: MotionSpec) -> str:
    it = spec.interaction
    out = [
        OPENING_LINE,
        "",
        *_structure(spec),
        "",
    ]
    out += ["INTERACTION", "", _target_sentence(spec)]
    if it.type_confidence.band != "high":
        type_word = ph.TYPE_TITLE[it.type].lower()
        out.append(
            f"The interaction appears to be: {type_word} ({it.type_confidence.band} confidence)."
        )
    out += ["", _forward_intro(spec), "", *_steps(spec), "", _feel(spec), "", *_reverse(spec)]
    uncertain = _uncertain(spec)
    if uncertain:
        out += ["", *uncertain]
    out += ["", *_output(spec)]
    out += [
        "",
        "CONSTRAINTS",
        "",
        f"All numbers are estimates measured from a screen recording "
        f"({ph.pixel_ratio_sentence(spec)}); keep them as targets, not exact source values.",
    ]
    if any(w.code == "frames_subsampled" for w in spec.warnings):
        out.append(
            f"The motion was long, so timing was measured at a reduced frame rate (about "
            f"{spec.source.timing_resolution_ms}ms between analysed frames); treat start times "
            f"and durations as approximate."
        )
    if it.trigger.kind == "unknown":
        out.append(UNKNOWN_TRIGGER_LINE)
    out.append(FINAL_LINE)
    return "\n".join(out) + "\n"
