"""Technical description (PRD §18, PLAN §9.2). Pure template over the IR.

Measured appearance (PLAN-continuous §13): when the spec carries ``scene`` or any element
``background_color`` / ``text_color`` / ``font_size_px``, an APPEARANCE section follows
INTERACTION (and the per-element static radius / shadow lines move into it). Without those
fields the output is unchanged. The ``has_appearance`` / ``appearance_box`` / ``parent_offset`` /
``animated_props`` / ``size_text`` helpers live in ``phrasing`` (shared with ``llm_prompt`` and
``css``); the names stay importable from here.
"""

from __future__ import annotations

from app.generate import phrasing as ph
from app.models.ir import (
    MeasuredColor,
    MeasuredNumber,
    MotionSpec,
    Transition,
    band_for,
)

# Measured appearance helpers: live in ``phrasing`` (shared with llm_prompt / css)
has_appearance = ph.has_appearance
appearance_box = ph.appearance_box
parent_offset = ph.parent_offset
animated_props = ph.animated_props
size_text = ph.size_text


def _measured_line(label: str, value: str, conf_value: float) -> str:
    band = band_for(conf_value)
    return f"  {label}: {ph.value_prefix(band)}{value} — {ph.confidence_clause(conf_value)}"


def _color_line(label: str, c: MeasuredColor) -> str:
    return _measured_line(label, c.value, c.confidence.value)


def _number_line(label: str, n: MeasuredNumber) -> str:
    return _measured_line(label, ph.px(n.value), n.confidence.value)


def _appearance(spec: MotionSpec) -> list[str]:
    lines = ["APPEARANCE (initial state, measured)", ""]
    if spec.scene is not None:
        sc = spec.scene
        lines.append(
            f"Viewport: {ph.num_px(sc.viewport_css.w)} x {ph.num_px(sc.viewport_css.h)}px "
            "(recorded frame, CSS px)"
        )
        if sc.page_background is not None:
            pb = sc.page_background
            lines.append(
                f"Page background: {ph.value_prefix(pb.confidence.band)}{pb.value} — "
                f"{ph.confidence_clause(pb.confidence.value)}"
            )
        lines.append("")
    for el in ph.element_order(spec):
        x, y, parent = parent_offset(spec, el)
        where = (
            "in the viewport"
            if parent is None
            else f"inside {ph.display_label(parent)} ({parent.id})"
        )
        lines.append(
            f"{ph.display_label(el)} ({el.id}): {size_text(appearance_box(el))} at "
            f"x {ph.px(x)}, y {ph.px(y)} {where}"
        )
        animated = animated_props(spec, el.id)
        st = el.static
        if st.background_color is not None and "background-color" not in animated:
            lines.append(_color_line("Background", st.background_color))
        if st.text_color is not None and "color" not in animated:
            lines.append(_color_line("Text color", st.text_color))
        if st.font_size_px is not None:
            lines.append(_number_line("Font size", st.font_size_px))
        if st.border_radius_px is not None and "border-radius" not in animated:
            lines.append(_number_line("Border radius", st.border_radius_px))
        if st.shadow is not None and "box-shadow" not in animated:
            sh = st.shadow
            lines.append(_measured_line("Shadow", ph.shadow_css(sh.value), sh.confidence.value))
    return lines


def _value_line(t: Transition, label: str, side: str) -> str:
    v = t.from_ if side == "from" else t.to
    band = t.confidence.band
    text = ph.value_fn(t.property, v)
    # Identity values (translate 0, scale 1) are CSS defaults, not measurements worth hedging.
    hedge = "" if band != "low" and ph.is_identity(t.property, v) else ph.value_prefix(band)
    return f"  {label}: {hedge}{text}"


def _property_block(t: Transition, active_label: str) -> list[str]:
    c = t.confidence
    head = t.property
    if t.property == "box-shadow":
        head += " (estimated)"
    lines = [
        head,
        _value_line(t, "Initial", "from"),
        _value_line(t, active_label, "to"),
        f"  Change: {'possibly ' if c.band == 'low' else ''}{ph.change_summary(t)}",
        f"  Duration: {ph.duration_text(t)}",
    ]
    if ph.ms_int(t.delay_ms) > 0:
        lines.append(f"  Delay: approximately {ph.ms(t.delay_ms)}")
    easing_band = band_for(c.easing)
    easing_note = "" if easing_band == "high" else f", {easing_band} confidence"
    lines.append(f"  Easing: {ph.easing_text(t.easing, c.easing)}{easing_note}")
    if t.transform_origin is not None:
        lines.append(f"  Transform origin: {t.transform_origin}")
    lines.append(f"  Confidence: {ph.confidence_text(c.overall)}")
    if t.notes:
        lines.append(f"  Notes: {'; '.join(t.notes)}")
    return lines


def _static_lines(spec: MotionSpec, el_id: str, animated: set[str]) -> list[str]:
    el = ph.element_map(spec)[el_id]
    out = []
    radius = el.static.border_radius_px
    if radius is not None and "border-radius" not in animated:
        out.append(
            f"Border radius (static): {ph.value_prefix(radius.confidence.band)}"
            f"{ph.px(radius.value)}{ph.band_note(radius.confidence.band)}"
        )
    shadow = el.static.shadow
    if shadow is not None and "box-shadow" not in animated:
        out.append(
            f"Shadow (static): {ph.value_prefix(shadow.confidence.band)}"
            f"{ph.shadow_css(shadow.value)}{ph.band_note(shadow.confidence.band)}"
        )
    return out


def _interaction(spec: MotionSpec) -> list[str]:
    it = spec.interaction
    els = ph.element_map(spec)
    type_line = ph.TYPE_TITLE[it.type]
    if it.type_confidence.band == "low":
        type_line = f"possibly {ph.lower_first(type_line)}"
    lines = [
        "INTERACTION",
        "",
        f"Type: {type_line} — {ph.confidence_clause(it.type_confidence.value)}, "
        f"from {it.type_source}",
        f"Pattern: {it.pattern}",
    ]
    if it.target_element_id is not None:
        target = f"{ph.display_label(els[it.target_element_id])} ({it.target_element_id})"
    else:
        target = f"{it.target_label} (not identified as a measured element)"
    if it.target_description:
        target += f" — {it.target_description}"
    lines.append(f"Target: {target}")
    trig = it.trigger
    lines.append(f"Trigger: {trig.description}")
    lines.append(
        f"Trigger kind: {trig.kind} / reverse: {trig.reverse_kind} — "
        f"{ph.confidence_clause(trig.confidence.value)}"
    )
    direction = {
        "forward": "forward only (reverse not recorded)",
        "forward_reverse": "forward + reverse",
        "round_trip": "round trip (press and release)",
    }[it.direction]
    lines.append(f"Direction: {direction}")
    total = f"approximately {ph.ms(it.total_duration_ms.forward)} forward"
    if it.total_duration_ms.reverse is not None:
        total += f", approximately {ph.ms(it.total_duration_ms.reverse)} reverse"
    lines.append(f"Total duration: {total}")
    return lines


def _reverse(spec: MotionSpec) -> list[str]:
    lines = ["REVERSE", ""]
    rev = ph.reverse_segment(spec)
    if rev is None:
        lines.append("Not recorded — assume the forward timings, reversed.")
        return lines
    if spec.interaction.direction == "round_trip":
        lines.append("Return phase of the round trip (release).")
    els = ph.element_map(spec)
    main = [t for t in ph.segment_transitions(spec, rev.id) if not ph.is_uncertain(t)]
    if not main:
        lines.append("No reliable reverse measurements — assume the forward timings, reversed.")
        return lines
    for t in ph.sort_transitions(spec, main):
        c = t.confidence
        el = els[t.element_id]
        text = (
            f"- {ph.display_label(el)} ({el.id}) {t.property}: "
            f"{ph.value_short(t.property, t.from_)} → {ph.value_short(t.property, t.to)}, "
            f"{ph.duration_text(t)}"
        )
        if ph.ms_int(t.delay_ms) > 0:
            text += f", delay approximately {ph.ms(t.delay_ms)}"
        text += (
            f", {ph.easing_text(t.easing, c.easing)}; confidence {ph.confidence_text(c.overall)}"
        )
        lines.append(text)
    return lines


def _relationships(spec: MotionSpec) -> list[str]:
    lines = ["TIMING RELATIONSHIPS", ""]
    body: list[str] = []
    for seg in spec.segments:
        sentences = ph.relationship_sentences(spec, seg.id)
        if sentences:
            title = "Forward" if seg.kind == "forward" else "Reverse"
            body.append(f"{title}:")
            body.extend(f"- {s}" for s in sentences)
    lines.extend(body or ["No timing relationships detected."])
    return lines


def _uncertain(spec: MotionSpec) -> list[str]:
    lines = ["UNCERTAIN OBSERVATIONS", ""]
    els = ph.element_map(spec)
    seg_title = {s.id: ("forward" if s.kind == "forward" else "reverse") for s in spec.segments}
    items = [t for t in spec.transitions if ph.is_uncertain(t)]
    for t in ph.sort_transitions(spec, items):
        el = els[t.element_id]
        lines.append(
            f"- {ph.display_label(el)} ({el.id}), {seg_title[t.segment_id]}: possibly a subtle "
            f"{t.property} change ({ph.value_short(t.property, t.from_)} → "
            f"{ph.value_short(t.property, t.to)}); "
            f"confidence {ph.confidence_text(t.confidence.overall)}. Not included above."
        )
    if not items:
        lines.append("None.")
    return lines


def _notes(spec: MotionSpec) -> list[str]:
    src = spec.source
    lines = ["NOTES", "", f"- {spec.meta.disclaimer}"]
    scale = f"- Geometry is in CSS px; {ph.pixel_ratio_sentence(spec)}"
    if src.pixel_ratio_source == "auto":
        scale += f" from the {src.width}x{src.height} recording"
    lines.append(scale + ".")
    lines.append(
        f"- Timing resolution: about {src.timing_resolution_ms}ms per frame "
        f"({ph.fmt_decimal(ph.round_step(src.fps_effective, '0.1'))} fps effective)."
    )
    if spec.interpretation.status != "ok":
        lines.append(
            f"- Element labels are heuristic (AI labeling status: {spec.interpretation.status})."
        )
    for w in spec.warnings:
        if w.code == "pixel_ratio_assumed":
            continue  # covered by the display-scale line
        lines.append(f"- {w.message}")
    return lines


def render(spec: MotionSpec) -> str:
    active_label = ph.ACTIVE_LABEL[spec.interaction.type]
    fwd = ph.forward_segment(spec)
    out: list[str] = _interaction(spec)
    appearance = has_appearance(spec)
    if appearance:
        out += ["", *_appearance(spec)]

    for el in ph.element_order(spec):
        ts = [
            t
            for t in ph.segment_transitions(spec, fwd.id)
            if t.element_id == el.id and not ph.is_uncertain(t)
        ]
        if not ts:
            continue
        ordered = sorted(ts, key=lambda t: ph.PROPERTY_ORDER.index(t.property))
        out += ["", ph.heading(el), ""]
        meta = f"Role: {el.role.replace('_', ' ')}"
        if el.parent_id is not None:
            meta += (
                f"; inside {ph.display_label(ph.element_map(spec)[el.parent_id])} ({el.parent_id})"
            )
        out.append(meta)
        if not appearance:  # otherwise listed under APPEARANCE
            out += _static_lines(spec, el.id, {t.property for t in ordered})
        for t in ordered:
            out.append("")
            out += _property_block(t, active_label)

    for section in (_reverse, _relationships, _uncertain, _notes):
        out += ["", *section(spec)]
    return "\n".join(out) + "\n"
