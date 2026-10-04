"""Technical tab, continuous mode (PLAN-continuous §6.1). Pure template over the IR.

Sections: CONTINUOUS MOTION -> AUTOPLAY -> PHASES -> BEHAVIOUR -> APPEARANCE -> UNCERTAIN
OBSERVATIONS -> NOTES. Values below the uncertainty threshold (< 0.3) only appear under
UNCERTAIN OBSERVATIONS.
"""

from __future__ import annotations

from app.generate import phrasing as ph
from app.generate.continuous import phrasing as cp
from app.models.ir import (
    ConstantFit,
    Easing,
    ExponentialFit,
    MeasuredNumber,
    MotionElement,
    MotionSpec,
    Phase,
    RampFit,
    TweenFit,
    band_for,
)

TYPE_TITLE = {
    "continuous": "Continuous autoplay (marquee)",
    "drag": "Draggable scroller (carousel)",
}


def _value(m: MeasuredNumber | None, fmt: str, easing: Easing | None = None) -> str | None:
    """``possibly ≈700 ms, linear — low (40%)``, or None when absent / uncertain.

    ``easing`` (ramps, tweens) is worded with the value's confidence as the easing confidence.
    """
    if not cp.usable(m):
        return None
    assert m is not None
    score = m.confidence.value
    text = cp.hedged(
        {"ms": cp.ms_approx, "px": cp.px_approx, "speed": cp.speed}[fmt](m.value), score
    )
    word = cp.easing_word(easing, score)
    if word:
        text += f", {word}"
    return text + cp.conf_suffix(score)


def _overview(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    it = spec.interaction
    el = cp.scroller(spec)
    lines = ["CONTINUOUS MOTION", ""]
    lines.append(
        f"Scroller: {ph.display_label(el)} ({el.id}) — {it.pattern}, "
        f"{cp.AXIS_WORD[c.axis]} ({c.axis} axis)"
    )
    type_line = TYPE_TITLE.get(it.type, it.type)
    if it.type_confidence.band == "low":
        type_line = f"possibly {ph.lower_first(type_line)}"
    lines.append(
        f"Type: {type_line} — {ph.confidence_clause(it.type_confidence.value)}, "
        f"from {it.type_source}"
    )
    if it.target_description:
        lines.append(f"Description: {it.target_description}")
    lines.append(f"Trigger: {it.trigger.description}")
    lines.append(
        f"Trigger kind: {it.trigger.kind} — {ph.confidence_clause(it.trigger.confidence.value)}"
    )
    r = c.region
    if cp.is_uncertain_score(c.region_confidence.value):
        lines.append("Region: uncertain (see UNCERTAIN OBSERVATIONS)")
    else:
        lines.append(
            f"Region: {cp.size_text(r.w, r.h)} at x {ph.num_px(r.x)}, y {ph.num_px(r.y)}"
            f"{cp.conf_suffix(c.region_confidence.value)}"
        )
    pitch = _value(c.pitch_px, "px")
    at_centre = " at the scroller centre" if pitch and cp.card_scale(spec) else ""
    lines.append(f"Card pitch (card + gap){at_centre}: {pitch or 'not observed'}")
    gap = _value(c.gap_px, "px")
    if gap is not None:
        lines.append(f"Gap between cards: {gap}")
    cs = c.card_scale
    if cs is not None:
        if cp.card_scale(spec) is None:
            lines.append("Card scaling: uncertain (see UNCERTAIN OBSERVATIONS)")
        else:
            score = cs.confidence.value
            lines.append(
                "Card scaling: by distance from the scroller centre, "
                f"{cp.hedged(cp.card_scale_range(cs), score)} (quadratic, cards stay packed; "
                "visible cards only, off-screen cards unscaled); "
                f"on screen the content moves ≈×{cp.scale_factor(cs.mean_scale)} as fast as the "
                f"unscaled track{cp.conf_suffix(score)}"
            )
    span = c.span_ms
    lines.append(
        f"Analysed span: {cp.ms_num(span.start_ms)}–{cp.ms_num(span.end_ms)} ms "
        f"({cp.ms_approx(it.total_duration_ms.forward)})"
    )
    return lines


def _autoplay(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    lines = ["AUTOPLAY", ""]
    a = c.autoplay
    if a is None:
        lines.append("None observed (the scroller did not move on its own before the first drag).")
        return lines
    lines.append(f"Direction: {a.direction} (content moves {cp.DIRECTION_PHRASE[a.direction]})")
    sp = a.speed_px_s
    if cp.usable(sp):
        score = sp.confidence.value
        speed = f"{cp.hedged(cp.speed(sp.value), score)}, {a.easing}{cp.conf_suffix(score)}"
    else:
        speed = "uncertain (see UNCERTAIN OBSERVATIONS)"
    lines.append(f"Speed: {speed}")
    loop = a.loop
    if not loop.observed:
        lines.append("Loop: not observed (the content did not repeat within the recording)")
    else:
        period = _value(loop.period_px, "px") or "uncertain"
        duration = _value(loop.duration_ms, "ms") or "uncertain"
        lines.append(f"Loop: one copy {period}; one loop {duration}")
    return lines


def _fit_text(p: Phase, uncertain: list[cp.Uncertain]) -> str:
    """Key value of a phase's fit (or peak speed for drags)."""

    def num(m: MeasuredNumber, kind: str, what: str) -> str | None:
        text = {"ms": cp.ms_approx, "px": cp.px_approx, "speed": cp.speed}[kind](m.value)
        if cp.is_uncertain_score(m.confidence.value):
            uncertain.append(cp.Uncertain(f"{p.id} {what}", text, m.confidence.value))
            return None
        band = band_for(m.confidence.value)
        return text if band == "high" else f"{text} ({band})"

    f = p.fit
    if isinstance(f, ConstantFit):
        v = num(f.velocity_px_s, "speed", "speed")
        return f"constant {v}" if v else "constant (speed uncertain)"
    if isinstance(f, ExponentialFit):
        tau = num(f.tau_ms, "ms", "time constant τ")
        v0 = num(f.v0_px_s, "speed", "release speed")
        parts = [f"τ {tau}" if tau else "τ uncertain"]
        if v0:
            parts.append(f"v0 {cp.signed_speed(f.v0_px_s.value)} px/s")
        if cp.speed_num(f.v_inf_px_s) != "0":
            parts.append(f"towards {cp.signed_speed(f.v_inf_px_s)} px/s")
        return ", ".join(parts)
    if isinstance(f, RampFit):
        dur = num(f.duration_ms, "ms", "ramp duration")
        text = f"ramp {cp.signed_speed(f.from_px_s)} → {cp.signed_speed(f.to_px_s)} px/s"
        text += f" over {dur}" if dur else " (duration uncertain)"
        return text + f", {ph.easing_text(f.easing, f.duration_ms.confidence.value)}"
    if isinstance(f, TweenFit):
        dist = num(f.distance_px, "px", "tween distance")
        dur = num(f.duration_ms, "ms", "tween duration")
        text = f"tween {dist or '(distance uncertain)'} over {dur or '(duration uncertain)'}"
        return text + f", {ph.easing_text(f.easing, f.duration_ms.confidence.value)}"
    if p.kind == "drag":
        return f"peak {cp.signed_speed(p.v_peak_px_s)} px/s"
    return "—"


def _phases(spec: MotionSpec, uncertain: list[cp.Uncertain]) -> list[str]:
    c = cp.cont(spec)
    lines = ["PHASES", ""]
    header = ["#", "Kind", "Time (ms)", "Velocity (px/s)", "Key value", "Confidence"]
    rows: list[list[str]] = []
    notes: list[str] = []
    for p in c.phases:
        if cp.is_uncertain_score(p.confidence.value):
            continue
        kind = cp.PHASE_WORD[p.kind] + (" (interrupted)" if p.interrupted else "")
        if p.evidence != "velocity":
            kind += f" [{p.evidence}]"
        rows.append(
            [
                p.id,
                kind,
                f"{cp.ms_num(p.start_ms)}–{cp.ms_num(p.end_ms)}",
                f"{cp.signed_speed(p.v_start_px_s)} → {cp.signed_speed(p.v_end_px_s)}",
                _fit_text(p, uncertain),
                ph.confidence_text(p.confidence.value),
            ]
        )
        notes += [f"- {p.id}: {n}" for n in p.notes]
    if not rows:
        lines.append("No phase could be labelled with enough confidence.")
        return lines
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def fmt(r: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip()

    lines.append(fmt(header))
    lines += [fmt(r) for r in rows]
    if notes:
        lines += ["", "Phase notes:", *notes]
    return lines


def _behaviour(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    b = c.behavior
    lines = ["BEHAVIOUR", ""]

    pz = b.pause
    if pz is None:
        lines.append("Pause: not observed")
    else:
        on_score = pz.on_confidence.value
        trigger = {
            "hover": "on hover",
            "press": "while pressed",
            "unknown": "on hover or press (trigger not visible in the recording)",
        }[pz.on]
        if cp.pause_by_press_inferred(spec):
            trigger = (
                "while pressed (trigger not visible in the recording; every pause was a drag "
                "starting)"
            )
        if pz.on != "unknown" and cp.is_uncertain_score(on_score):
            trigger = "trigger uncertain (see UNCERTAIN OBSERVATIONS)"
            lines.append(f"Pause: {trigger}")
        else:
            lines.append(f"Pause: {trigger}{cp.conf_suffix(on_score)}")
        decel = _value(pz.decel_ms, "ms", pz.easing)
        if decel is not None:
            lines.append(f"  Slowdown: {decel}")
        elif pz.decel_ms is None and not cp.phases_of(spec, "decelerate"):
            lines.append("  Slowdown: none (autoplay stops at once)")
        elif pz.decel_ms is None:
            lines.append("  Slowdown: not measured")
        interrupted = any(p.interrupted for p in cp.phases_of(spec, "decelerate"))
        full = "yes" if pz.stops_completely else "no"
        if not pz.stops_completely and interrupted:
            full += " (cut short by the next interaction in the recording)"
        lines.append(f"  Comes to a full stop: {full}")

    d = b.drag
    if d is None:
        lines.append("Drag: not observed")
    else:
        lines.append(f"Drag: {d.count} drag{'s' if d.count != 1 else ''}")
        if d.follows_pointer is None:
            lines.append("  Follows the pointer: unknown (no cursor visible to compare)")
        else:
            follows = "yes" if d.follows_pointer else "no"
            if cp.usable(d.pointer_ratio):
                r = d.pointer_ratio
                assert r is not None
                follows += (
                    f" (content / pointer speed {cp.ratio_num(r.value)}"
                    f"{cp.conf_suffix(r.confidence.value)})"
                )
            lines.append(f"  Follows the pointer: {follows}")
        peak = _value(d.peak_speed_px_s, "speed")
        if peak is not None:
            lines.append(f"  Peak speed: {peak}")

    i = b.inertia
    if i is None:
        lines.append("Momentum (inertia): not observed")
    else:
        model = "exponential decay" if i.model == "exponential" else "eased glide (tween)"
        n = i.instances
        lines.append(f"Momentum (inertia): {model}, {n} release{'s' if n != 1 else ''}")
        tau = _value(i.tau_ms, "ms")
        if tau is not None:
            lines.append(f"  Time constant τ: {tau}")
        dur = _value(i.duration_ms, "ms", i.easing)
        if dur is not None:
            lines.append(f"  Duration: {dur}")
        lo, hi = cp.speed_num(i.release_speed_min_px_s), cp.speed_num(i.release_speed_max_px_s)
        lines.append(
            f"  Release speeds: ≈{lo} px/s" if lo == hi else f"  Release speeds: ≈{lo}–{hi} px/s"
        )
        if cp.inertia_blends_into_autoplay(spec):
            lines.append(
                "  Decays towards: the autoplay velocity (blends back into autoplay, which keeps "
                "running; no rest or separate resume after such a release)"
            )
            if cp.drag_stops(spec):
                after = ", then Resume" if cp.resumes_after_drag_stop(spec) else ""
                lines.append(
                    "  Released at rest: drags that end with the pointer stopped have no "
                    f"momentum (the content rests{after})"
                )
        else:
            stop = _value(i.stop_px_s, "speed")
            if stop is not None:
                lines.append(f"  Stops below: {stop}")

    s = b.snap
    if s is None:
        lines.append("Snap: not observed")
    elif s.kind == "abrupt_ambiguous":
        lines.append(
            "Snap: abrupt stops without grid evidence (a snap, or the pointer stopping before "
            "release; ambiguous without a visible cursor)"
        )
    else:
        lines.append("Snap: to the card grid")
        step = _value(s.step_px, "px")
        if step is not None:
            lines.append(f"  Step: {step}")
        dur = _value(s.duration_ms, "ms", s.easing)
        if dur is not None:
            lines.append(f"  Duration: {dur}")
    if s is not None:
        lines.append(
            f"  Overshoot: {'yes (spring-like; spring not reconstructed)' if s.overshoot else 'no'}"
        )

    r = b.resume
    if r is None:
        lines.append("Resume: not observed")
    else:
        lines.append("Resume:")
        rest = _value(r.delay_after_rest_ms, "ms")
        if rest is not None:
            lines.append(f"  Delay after rest: {rest}")
        rel = _value(r.delay_after_release_ms, "ms")
        if rel is not None:
            lines.append(f"  Delay after release: {rel}")
        leave = _value(r.delay_after_leave_ms, "ms")
        if leave is not None:
            lines.append(f"  Delay after the pointer leaves: {leave}")
        ramp = _value(r.ramp_ms, "ms", r.easing)
        if ramp is not None:
            lines.append(f"  Ramp: {ramp}")
        direction = "same direction" if r.direction_preserved else "opposite direction"
        lines.append(f"  Speed after resuming: {cp.speed(r.to_speed_px_s)}, {direction}")
    return lines


def _element_line(spec: MotionSpec, el: MotionElement) -> str:
    c = cp.cont(spec)
    st = el.static
    is_scroller = el.id == c.element_id
    box = c.region if is_scroller else el.bbox_initial
    parts = [cp.size_text(box.w, box.h)]
    card = cp.card_element(spec)
    if cp.card_scale(spec) is not None and card is not None and el.id == card.id:
        parts[0] += " (at the scroller centre; scales with its position)"
    if is_scroller:
        parts[0] += f" at x {ph.num_px(box.x)}, y {ph.num_px(box.y)}"
        if cp.is_uncertain_score(c.region_confidence.value):
            parts = ["size uncertain"]
    elif el.parent_id is not None:
        parent = ph.element_map(spec)[el.parent_id]
        pbox = c.region if parent.id == c.element_id else parent.bbox_initial
        where = (
            "vertically centred in"
            if cp.centred_in(box, pbox, "y")
            else f"{cp.vertical_place(box, pbox)} of"
        )
        parts[0] += f", {where} the {ph.ref_name(parent)}"
    for what, col in (("background", st.background_color), ("text colour", st.text_color)):
        if cp.usable(col):
            assert col is not None
            parts.append(
                f"{what} {cp.hedged(col.value, col.confidence.value)}"
                f"{cp.conf_suffix(col.confidence.value)}"
            )
    for what, m in (("corner radius", st.border_radius_px), ("font size", st.font_size_px)):
        val = _value(m, "px")
        if val is not None:
            parts.append(f"{what} {val}")
    if st.shadow is not None and cp.usable(st.shadow):  # type: ignore[arg-type]
        parts.append(
            f"shadow {cp.hedged(ph.shadow_css(st.shadow.value), st.shadow.confidence.value)}"
            f"{cp.conf_suffix(st.shadow.confidence.value)}"
        )
    return f"{ph.display_label(el)} ({el.id}): " + "; ".join(parts)


def _appearance(spec: MotionSpec) -> list[str]:
    lines = ["APPEARANCE", ""]
    if spec.scene is not None:
        vp = spec.scene.viewport_css
        lines.append(f"Viewport: {cp.size_text(vp.w, vp.h)} (recorded frame)")
        bg = spec.scene.page_background
        if cp.usable(bg):
            assert bg is not None
            lines.append(
                f"Page background: {cp.hedged(bg.value, bg.confidence.value)}"
                f"{cp.conf_suffix(bg.confidence.value)}"
            )
    for el in [cp.scroller(spec), *cp.inner_elements(spec)]:
        lines.append(_element_line(spec, el))
    return lines


def _uncertain(spec: MotionSpec, extra: list[cp.Uncertain]) -> list[str]:
    lines = ["UNCERTAIN OBSERVATIONS", ""]
    body = []
    for p in cp.uncertain_phases(spec):
        body.append(
            f"- {p.id}, {cp.ms_num(p.start_ms)}–{cp.ms_num(p.end_ms)} ms: possibly "
            f"{cp.PHASE_WORD[p.kind]}; confidence {ph.confidence_text(p.confidence.value)}. "
            "Not included above."
        )
    items = cp.uncertain_motion(spec) + extra + cp.uncertain_appearance(spec)
    body += [f"- {u.text()}. Not included above." for u in items]
    return lines + (body or ["None."])


def _notes(spec: MotionSpec) -> list[str]:
    src = spec.source
    c = cp.cont(spec)
    lines = ["NOTES", "", f"- {spec.meta.disclaimer}"]
    scale = f"- Geometry is in CSS px; {ph.pixel_ratio_sentence(spec)}"
    if src.pixel_ratio_source == "auto":
        scale += f" from the {src.width}x{src.height} recording"
    lines.append(scale + ".")
    lines.append(
        f"- Timing resolution: about {src.timing_resolution_ms}ms per frame "
        f"({ph.fmt_decimal(ph.round_step(src.fps_effective, '0.1'))} fps effective)."
    )
    lines.append(f"- Signed velocities: {c.sign_convention}.")
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
    phase_uncertain: list[cp.Uncertain] = []
    out = _overview(spec)
    out += ["", *_autoplay(spec)]
    out += ["", *_phases(spec, phase_uncertain)]
    out += ["", *_behaviour(spec)]
    out += ["", *_appearance(spec)]
    out += ["", *_uncertain(spec, phase_uncertain)]
    out += ["", *_notes(spec)]
    return "\n".join(out) + "\n"
