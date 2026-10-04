"""LLM prompt tab, continuous mode (PLAN-continuous §6.2, §13). Pure template over the IR.

Same deliverable as the transition prompt: ONE self-contained, responsive ``index.html`` with
neutral placeholder content (nothing from the recording). The measured appearance (band box, card
size, gap / pitch, colours, viewport) is passed on hedged by confidence; behaviours are numbered
and only present when observed. Values below the uncertainty threshold only appear under
UNCERTAIN (OPTIONAL).
"""

from __future__ import annotations

from app.generate import phrasing as ph
from app.generate.continuous import phrasing as cp
from app.models.ir import Box, MeasuredNumber, MotionSpec, RampFit, band_for

FINAL_LINE = "Do not invent additional motion or behaviours that are not described above."

PATTERN_NOUN = {"marquee": "marquee", "carousel": "carousel"}


def _noun(spec: MotionSpec) -> str:
    """``horizontal carousel`` / ``vertical marquee`` / ``horizontal scroller``."""
    c = cp.cont(spec)
    return f"{cp.AXIS_WORD[c.axis]} {PATTERN_NOUN.get(spec.interaction.pattern, 'scroller')}"


def opening_line(spec: MotionSpec) -> str:
    return (
        f"Recreate the reference UI motion, a continuously moving {_noun(spec)}, as a single "
        "self-contained, responsive HTML file, as follows."
    )


def _val(m: MeasuredNumber, text: str) -> str:
    """Hedged value with its band note: ``possibly ≈700 ms (low confidence)``."""
    return cp.hedged_note(text, m.confidence.value)


# --------------------------------------------------------------------------------------------
# STRUCTURE / APPEARANCE
# --------------------------------------------------------------------------------------------


def _structure(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    axis = cp.AXIS_WORD[c.axis]
    row = "row" if c.axis == "x" else "column"
    lines = [
        "STRUCTURE",
        "",
        f"Create a {_noun(spec)}: a viewport (the scroller) that clips a track holding one {row} "
        "of placeholder cards.",
    ]
    placed = "side by side" if c.axis == "x" else "one below the other"
    if cp.card_scale(spec) is not None:
        # the spacing grows with the cards towards the edges: only the centre pitch is constant
        text = f"- The cards sit {placed}, packed with a constant gap and scaled by their position"
        if cp.usable(c.pitch_px):
            assert c.pitch_px is not None
            text += (
                f"; at the scroller centre the pitch is "
                f"{_val(c.pitch_px, cp.px_approx(c.pitch_px.value))}, one card plus the gap to "
                "the next"
            )
        lines.append(text + " (see Card scaling under BEHAVIOUR).")
    elif cp.usable(c.pitch_px):
        assert c.pitch_px is not None
        lines.append(
            f"- The cards sit {placed} at a constant spacing: pitch "
            f"{_val(c.pitch_px, cp.px_approx(c.pitch_px.value))}, "
            "one card plus the gap to the next."
        )
    else:
        lines.append(f"- The cards sit {placed} at a constant spacing.")
    lines.append(
        "- Repeat the cards once (two identical copies back to back) so the track can wrap "
        f"seamlessly; one copy must be at least as {'wide' if c.axis == 'x' else 'tall'} as the "
        "scroller."
    )
    if spec.structure:
        lines.append("- The scroller contains: " + ", ".join(spec.structure) + ".")
    else:
        card = cp.card_element(spec)
        if card is not None:
            parts = [ph.ref_name(e) for e in cp.inner_elements(spec) if e.parent_id == card.id]
            if parts:
                lines.append(f"- Each {ph.ref_name(card)} contains: {', '.join(parts)}.")
    lines.append(f"- The motion runs along the {axis} axis only.")
    return lines


def _colour(label: str, col) -> str | None:  # noqa: ANN001 - MeasuredColor | None
    if not cp.usable(col):
        return None
    return f"{label} {cp.hedged_note(col.value, col.confidence.value)}"


def _appearance(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    lines: list[str] = []
    scene = spec.scene
    if scene is not None:
        vp = scene.viewport_css
        text = f"- Recorded viewport: {cp.size_text(vp.w, vp.h)}"
        bg = _colour("page background", scene.page_background)
        lines.append(text + (f"; {bg}." if bg else "."))
    sc = cp.scroller(spec)
    r = c.region
    rscore = c.region_confidence.value
    if not cp.is_uncertain_score(rscore):
        text = f"- Scroller: {cp.hedged_note('≈' + cp.size_text(r.w, r.h), rscore)}"
        where = f"top-left corner at x {ph.num_px(r.x)}, y {ph.num_px(r.y)}"
        if scene is not None:
            where += " of the recorded viewport"
            vbox = Box(x=0.0, y=0.0, w=scene.viewport_css.w, h=scene.viewport_css.h)
            centred = [a for a, ax in (("horizontally", "x"), ("vertically", "y"))
                       if cp.centred_in(r, vbox, ax)]  # fmt: skip
            where += f" ({' and '.join(centred)} centred)" if centred else " (not centred)"
        text += f", {where}"
        bg = _colour("background", sc.static.background_color)
        lines.append(text + (f"; {bg}." if bg else "."))
    card = cp.card_element(spec)
    if card is not None:
        lines.append(_card_line(spec, card))
        for el in cp.inner_elements(spec):
            if el.parent_id == card.id:
                line = _part_line(spec, el, card)
                if line:
                    lines.append(line)
    if not lines:
        return []
    return [
        "APPEARANCE (measured, approximate)",
        "",
        *lines,
        "Match these sizes, colours and positions at the recorded viewport size; all content "
        "stays placeholder.",
    ]


def _card_line(spec: MotionSpec, card) -> str:  # noqa: ANN001 - MotionElement
    c = cp.cont(spec)
    st = card.static
    b = card.bbox_initial
    name = ph.ref_name(card)
    parts = [f"≈{cp.size_text(b.w, b.h)} each"]
    if cp.card_scale(spec) is not None:
        parts[0] += " at the scroller centre (they grow towards the edges, see Card scaling)"
    other = "x" if c.axis == "y" else "y"
    if cp.centred_in(b, c.region, other):
        parts[0] += f", {cp.AXIS_ADVERB[other]} centred in the scroller"
    if cp.usable(c.gap_px):
        assert c.gap_px is not None
        parts.append(f"gap {_val(c.gap_px, cp.px_approx(c.gap_px.value))}")
    bg = _colour("background", st.background_color)
    if bg:
        parts.append(bg)
    if cp.usable(st.border_radius_px):
        m = st.border_radius_px
        parts.append(f"corner radius {_val(m, cp.px_approx(m.value))}")
    if st.shadow is not None and cp.usable(st.shadow):  # type: ignore[arg-type]
        parts.append(
            f"shadow {cp.hedged_note(ph.shadow_css(st.shadow.value), st.shadow.confidence.value)}"
        )
    return f"- {ph.upper_first(name)}s: " + "; ".join(parts) + "."


def _part_line(spec: MotionSpec, el, card) -> str | None:  # noqa: ANN001 - MotionElement
    st = el.static
    parts = []
    col = _colour("colour", st.text_color)
    if col:
        parts.append(col)
    if cp.usable(st.font_size_px):
        m = st.font_size_px
        parts.append(f"font size {_val(m, cp.px_approx(m.value))}")
    bg = _colour("background", st.background_color)
    if bg:
        parts.append(bg)
    if cp.usable(st.border_radius_px):
        m = st.border_radius_px
        parts.append(f"corner radius {_val(m, cp.px_approx(m.value))}")
    if not parts:
        return None
    where = cp.vertical_place(el.bbox_initial, card.bbox_initial)
    parts.append(f"placed in the {where} of the {ph.ref_name(card)}")
    card_words = set(ph.ref_name(card).split())
    part = ph.ref_name(el)
    # "card title" inside a "product card" stays "Card title" (no "Product card card title")
    label = part if card_words & set(part.split()) else f"{ph.ref_name(card)} {part}"
    return f"- {ph.upper_first(label)}: " + "; ".join(parts) + "."


# --------------------------------------------------------------------------------------------
# BEHAVIOUR
# --------------------------------------------------------------------------------------------


def _autoplay(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    a = c.autoplay
    if a is None:
        return "At rest: the track does not move on its own; it only moves when the user drags it."
    way = cp.DIRECTION_PHRASE[a.direction]
    sp = a.speed_px_s
    if cp.usable(sp):
        text = (
            f"Autoplay: the track moves continuously {way} at "
            f"{cp.hedged(cp.speed(sp.value), sp.confidence.value)} with constant (linear) velocity"
            f"{ph.band_note(sp.confidence.band)}"
        )
    else:
        text = (
            f"Autoplay: the track moves continuously {way} with constant (linear) velocity (the "
            "measured speed is uncertain; see UNCERTAIN)"
        )
    loop = a.loop
    if loop.observed and cp.usable(loop.period_px) and cp.usable(loop.duration_ms):
        assert loop.period_px is not None and loop.duration_ms is not None
        return text + (
            f" and loops seamlessly: one copy of the cards is "
            f"{_val(loop.period_px, cp.px_approx(loop.period_px.value))} long and one loop takes "
            f"{_val(loop.duration_ms, cp.ms_approx(loop.duration_ms.value))}."
        )
    return text + (
        " and loops seamlessly by wrapping after one copy of the cards (the loop length was not "
        "visible in the recording)."
    )


def _card_scale(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    cs = cp.card_scale(spec)
    assert cs is not None
    score = cs.confidence.value
    axis = cp.AXIS_WORD[c.axis]
    at = cp.scale_factor(cs.scale_at_reference)
    ref = ph.num_px(cs.reference_distance_px)
    mean = cp.scale_factor(cs.mean_scale)
    near, far = ("left", "right") if c.axis == "x" else ("top", "bottom")
    return (
        f"Card scaling: every card is scaled by its distance d from the scroller centre (card "
        f"centre to scroller centre, along the {axis} axis): "
        f"{cp.hedged(cp.card_scale_range(cs), score)} (the farthest whole card measured), growing "
        f"with d²: scale = 1 + ({at} − 1) · (d / {ref})²; continue the same curve for the partly "
        f"visible cards at the edges{ph.band_note(band_for(score))}. Update every card's scale "
        "each frame from its current on-screen position, scaling it about its centre (scale both "
        "width and height). The gaps keep their measured size, so the cards stay packed and move "
        "apart as they grow: lay them out from the scroller centre outwards, each card's "
        f"{near} / {far} edge one gap away from its neighbour's scaled edge. Apply the curve only "
        "to cards that are at least partly inside the scroller and leave every card that is "
        "entirely outside it unscaled: far from the centre the curve grows without bound and the "
        "packed layout has no solution there, so never extrapolate it off screen (an off-screen "
        "card scaled by it can grow larger than the scroller and cover it). The speeds and "
        "distances in this prompt are measured on screen; with this magnification the content "
        f"moves on average ≈{mean} times as fast as the unscaled track, so move the unscaled "
        f"track at the measured speeds divided by ≈{mean} and, while dragging, by the pointer's "
        f"movement divided by ≈{mean}: the content then keeps up with the pointer on average "
        "(the unmagnified card at the centre moves slightly less than the pointer)."
    )


def _pause_trigger(spec: MotionSpec) -> str:
    pz = cp.cont(spec).behavior.pause
    assert pz is not None
    name = cp.scroller_name(spec)
    score = pz.on_confidence.value
    if cp.pause_by_press_inferred(spec):
        return (
            f"When the pointer presses the {name} (every pause in the recording was the start of "
            "a drag; the pointer was not visible, so a pause on hover alone was not observed: do "
            "not add one)"
        )
    if pz.on == "unknown" or cp.is_uncertain_score(score):
        return (
            f"When the pointer hovers or presses the {name} (the trigger was not visible in the "
            "recording; implement it as hover with a mouse, and pressing starts a drag anyway)"
        )
    low = " (trigger inferred with low confidence)" if band_for(score) == "low" else ""
    if pz.on == "hover":
        return f"When the pointer hovers the {name}{low}"
    return f"While the pointer is pressed on the {name}{low}"


def _pause(spec: MotionSpec) -> str:
    pz = cp.cont(spec).behavior.pause
    assert pz is not None
    decels = cp.phases_of(spec, "decelerate")
    ramps = [p.fit for p in decels if isinstance(p.fit, RampFit)]
    to_zero = pz.stops_completely or all(cp.speed_num(f.to_px_s) == "0" for f in ramps)
    target = "to a stop" if to_zero else "down"
    m = pz.decel_ms
    if not decels and m is None:
        # autoplay -> drag: the press that starts the drag stops autoplay at once
        stay = {"press": " while it is held", "hover": " while the pointer stays over it"}.get(
            cp.pause_trigger(spec) or "", ""
        )
        return (
            f"Pause: {ph.lower_first(_pause_trigger(spec))}, autoplay stops at once (no "
            f"slowdown). It stays stopped{stay}."
        )
    text = f"Pause: {ph.lower_first(_pause_trigger(spec))}, autoplay slows {target}"
    if cp.usable(m):
        assert m is not None
        text += f" over {cp.hedged(cp.ms_approx(m.value), m.confidence.value)}"
        text += cp.easing_using(pz.easing, m.confidence.value)
        text += ph.band_note(m.confidence.band)
    elif m is not None:
        text += " (the slowdown duration is uncertain; see UNCERTAIN)"
    if not pz.stops_completely and any(p.interrupted for p in decels) and to_zero:
        text += (
            "; in the recording the slowdown was cut short by the next interaction, so its full "
            "length is extrapolated"
        )
    stay = (
        " while it is held"
        if cp.pause_trigger(spec) == "press"
        else " while the pointer stays over it"
    )
    return text + f". It stays paused{stay}."


def _drag(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    d = c.behavior.drag
    assert d is not None
    axis = cp.AXIS_WORD[c.axis]
    scaled = cp.card_scale(spec) is not None
    # with card scaling the pointer movement is divided by the MEAN magnification: the content
    # keeps up with the pointer on average, not card by card (the centre card lags slightly)
    on_avg = ", on screen and on average" if scaled else ""
    if d.follows_pointer is False and cp.usable(d.pointer_ratio):
        assert d.pointer_ratio is not None
        follows = (
            "the content moves "
            f"{cp.hedged(cp.ratio_num(d.pointer_ratio.value), d.pointer_ratio.confidence.value)} "
            f"times as far as the pointer along the {axis} axis{on_avg}"
        )
    else:
        exactly = "1:1 on screen on average" if scaled else "exactly"
        follows = f"the content follows the pointer {exactly} along the {axis} axis"
        if d.follows_pointer is None:
            follows += " (assumed: the pointer was not visible in the recording to compare)"
    if scaled:
        follows += (
            "; with the card scaling, the cards near the scroller centre move slightly less than "
            "the pointer and those near the edges slightly more (see Card scaling)"
        )
    text = f"Dragging: {follows}"
    if c.autoplay is not None:
        text += "; grabbing the track takes over from autoplay immediately"
    text += (
        ". Use Pointer Events with setPointerCapture on the scroller; set touch-action: "
        f"{cp.TOUCH_ACTION[c.axis]} on the scroller so "
        f"{'vertical' if c.axis == 'x' else 'horizontal'} page scrolling still works on touch; "
        "prevent text and image selection while dragging, and suppress the click on a card when "
        "the pointer moved."
    )
    n = d.count
    count = f"{n} drag{'s' if n != 1 else ''} recorded"
    if cp.usable(d.peak_speed_px_s):
        m = d.peak_speed_px_s
        band = m.confidence.band
        note = "" if band == "high" else f"{band} confidence; "
        text += (
            f" Drags in the recording reached {cp.hedged(cp.speed(m.value), m.confidence.value)} "
            f"({note}{count})."
        )
    else:
        text += f" ({ph.upper_first(count)}.)"
    return text


def _regrab(spec: MotionSpec) -> str:
    """Re-grab sentence when an inertia phase was cut by a new drag."""
    phases = cp.cont(spec).phases
    for a, b in zip(phases, phases[1:], strict=False):
        if a.kind == "inertia" and a.interrupted and b.kind == "drag":
            return (
                " Pressing again while it is still coasting stops it at once and starts a new drag."
            )
    return ""


def _inertia_blend(spec: MotionSpec, tau: str) -> str:
    """Momentum that decays towards the autoplay velocity: it never comes to rest, it blends
    back into autoplay (which keeps running), so no resume follows such a release."""
    b = cp.cont(spec).behavior
    stops = bool(cp.drag_stops(spec))
    fling = " while it is still moving (a fling)" if stops else ""
    such = "a fling" if stops else "such a release"
    text = (
        f"Release (momentum): when the pointer is let go{fling}, keep moving with the pointer's "
        "release velocity and let it decay exponentially towards the autoplay velocity "
        f"{tau}: v = v_auto + (v − v_auto) · e^(−dt/τ) each frame (v_auto is the autoplay "
        "velocity, in the autoplay direction), using the real frame time so it is frame-rate "
        "independent. It does not come to rest: it blends back into autoplay, which simply keeps "
        f"running, so no rest, delay or resume ramp follows {such}"
    )
    if b.resume is not None and not stops:
        text += " (Resume below applies only after the motion has come to rest, e.g. after a pause)"
    if cp.pauses_on_hover(spec):
        text += "; the hover pause starts again only when the pointer re-enters the scroller"
    return text + "."


def _stop_release(spec: MotionSpec) -> str:
    """Drags that ended with the pointer stopped: a release without velocity, no momentum."""
    then = (
        "it stays where it stopped and Resume below applies"
        if cp.resumes_after_drag_stop(spec)
        else "it stays where it stopped"
    )
    return (
        " Several drags in the recording end differently: the pointer slows down and stops "
        "before it is let go, and the content stops together with it (it follows the pointer "
        "1:1 while pressed, so this is part of the drag, not a separate glide). Such a release "
        f"has zero velocity, so there is no momentum: {then}."
    )


def _inertia(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    i = c.behavior.inertia
    assert i is not None
    lo, hi = cp.speed_num(i.release_speed_min_px_s), cp.speed_num(i.release_speed_max_px_s)
    measured = f"≈{lo} px/s" if lo == hi else f"≈{lo}–{hi} px/s"
    window = " (about the last half of τ)" if i.model == "exponential" else ""
    tail = (
        f" Measured release speeds were {measured}; compute the release velocity from the most "
        f"recent pointer movement{window}, do not hardcode it"
    )
    if cp.drag_stops(spec):
        tail += (
            "; if the pointer did not move during that time (it had stopped before being let go) "
            "the release velocity is zero"
        )
    tail += "."
    if cp.drag_stops(spec):
        tail += _stop_release(spec)
    if i.model == "exponential":
        if cp.usable(i.tau_ms):
            assert i.tau_ms is not None
            tau = f"with τ {_val(i.tau_ms, cp.ms_approx(i.tau_ms.value))}"
        else:
            tau = "(the time constant τ is uncertain; see UNCERTAIN)"
        if cp.inertia_blends_into_autoplay(spec):
            text = _inertia_blend(spec, tau)
        else:
            until = "until it comes to rest"
            if cp.usable(i.stop_px_s):
                assert i.stop_px_s is not None
                stop = i.stop_px_s
                until = (
                    f"and stop it once it is slower than "
                    f"{cp.hedged(f'≈{cp.speed_num(stop.value)} px/s', stop.confidence.value)} "
                    "(that is when it counts as at rest)"
                )
            text = (
                "Release (momentum): on release, keep moving with the pointer's release velocity "
                f"and let it decay exponentially towards zero {tau}: v = v · e^(−dt/τ) each "
                f"frame, using the real frame time so it is frame-rate independent, {until}."
            )
    else:
        dur = i.duration_ms
        over = (
            f"over {cp.hedged(cp.ms_approx(dur.value), dur.confidence.value)}"
            if cp.usable(dur)
            else "(duration uncertain; see UNCERTAIN)"
        )
        score = dur.confidence.value if dur is not None else 0.0
        text = (
            f"Release (momentum): on release, glide to rest {over}"
            f"{cp.easing_using(i.easing, score)}, starting at the pointer's release velocity (the "
            f"glide distance grows with the release speed){ph.band_note(band_for(score))}."
        )
    return text + tail + _regrab(spec)


def _snap(spec: MotionSpec) -> str:
    b = cp.cont(spec).behavior
    s = b.snap
    assert s is not None
    if s.kind == "abrupt_ambiguous":
        return (
            "Abrupt stops: some drags ended abruptly without coasting, and no alignment to the "
            "cards was found. This may be a snap or the pointer stopping before release (they look "
            "the same without a visible cursor); do not add snapping unless intended."
        )
    when = "when the momentum is nearly spent" if b.inertia is not None else "on release"
    text = f"Snap: {when}, settle on the nearest card position"
    if cp.usable(s.step_px):
        assert s.step_px is not None
        text += f" (step {_val(s.step_px, cp.px_approx(s.step_px.value))})"
    if cp.usable(s.duration_ms):
        assert s.duration_ms is not None
        m = s.duration_ms
        text += f" over {cp.hedged(cp.ms_approx(m.value), m.confidence.value)}"
        if s.overshoot and s.easing is not None:
            text += " " + ph.easing_phrase(s.easing, m.confidence.value)
        else:
            text += cp.easing_using(s.easing, m.confidence.value)
        text += ph.band_note(m.confidence.band)
    if s.overshoot and (s.easing is None or not ph.has_overshoot(s.easing)):
        text += ", with a slight overshoot (spring-like; exact spring not reconstructed)"
    return text + "."


def _resume(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    r = c.behavior.resume
    assert r is not None
    when = "the motion has come to rest"
    if cp.pauses_on_hover(spec):
        when += " and the pointer has left"
    m = r.delay_after_rest_ms
    notes = []
    if cp.resumes_after_drag_stop(spec):
        # the rest follows a drag that stopped with the pointer: the stop is visible, the moment
        # the pointer was let go is not -> count from the stop, but never resume while pressed
        when = "the content has come to rest after a release without momentum"
        if cp.usable(m):
            lead = (
                f"Resume: once {when}, autoplay restarts "
                f"{cp.hedged(cp.ms_approx(m.value), m.confidence.value)} after the content "
                "stopped moving (count from the last movement, not from the release, which was "
                "not visible; if the pointer is still pressed then, restart when it is let go)"
            )
            if m.confidence.band != "high":
                notes.append(f"{m.confidence.band} confidence")
        else:
            lead = f"Resume: once {when}, autoplay restarts after a short delay"
            notes.append("the delay is uncertain; see UNCERTAIN")
    elif cp.usable(m):
        lead = f"Resume: once {when}, wait {cp.hedged(cp.ms_approx(m.value), m.confidence.value)}"
        if m.confidence.band != "high":
            notes.append(f"{m.confidence.band} confidence")
    else:
        lead = f"Resume: once {when}, wait a short delay"
        notes.append("the delay is uncertain; see UNCERTAIN")
    rel = r.delay_after_release_ms
    if cp.usable(rel):
        assert rel is not None
        text = f"{cp.hedged(cp.ms_approx(rel.value), rel.confidence.value)} after release"
        if rel.confidence.band != "high":
            text += f", {rel.confidence.band} confidence"
        notes.append(text)
    if notes:
        lead += f" ({'; '.join(notes)})"
    leave = r.delay_after_leave_ms
    if cp.usable(leave):
        assert leave is not None
        # hover pause: the clock that matters starts when the pointer leaves
        lead = (
            f"Resume: when the pointer leaves the {cp.scroller_name(spec)}, wait "
            f"{cp.hedged(cp.ms_approx(leave.value), leave.confidence.value)}"
            f"{ph.band_note(leave.confidence.band)}"
        )
    direction = "its original direction" if r.direction_preserved else "the opposite direction"
    resumes = "in" if cp.resumes_after_drag_stop(spec) else "then autoplay resumes in"
    text = (
        f"{lead}, {resumes} {direction}, ramping up from standstill to {cp.speed(r.to_speed_px_s)}"
    )
    ramp = r.ramp_ms
    if cp.usable(ramp):
        text += f" over {cp.hedged(cp.ms_approx(ramp.value), ramp.confidence.value)}"
        text += cp.easing_using(r.easing, ramp.confidence.value)
        text += ph.band_note(ramp.confidence.band)
    else:
        text += " (the ramp duration is uncertain; see UNCERTAIN)"
    return text + "."


def _no_resume(spec: MotionSpec) -> str | None:
    c = cp.cont(spec)
    b = c.behavior
    if c.autoplay is None or b.resume is not None or not (b.pause or b.drag):
        return None
    if cp.inertia_blends_into_autoplay(spec):
        # releases blend back into autoplay: only a pause could leave it stopped
        if b.pause is None:
            return None
        return (
            "The recording did not show autoplay resuming after a pause; resume it only if "
            "intended."
        )
    return (
        "The recording did not show autoplay resuming after the interaction; resume it only if "
        "intended."
    )


def _behaviour(spec: MotionSpec) -> list[str]:
    b = cp.cont(spec).behavior
    items = [_autoplay(spec)]
    if cp.card_scale(spec) is not None:
        items.append(_card_scale(spec))
    if b.pause is not None:
        items.append(_pause(spec))
    if b.drag is not None:
        items.append(_drag(spec))
    if b.inertia is not None:
        items.append(_inertia(spec))
    if b.snap is not None:
        items.append(_snap(spec))
    if b.resume is not None:
        items.append(_resume(spec))
    lines = ["BEHAVIOUR", ""] + [f"{i}. {t}" for i, t in enumerate(items, start=1)]
    extra = _no_resume(spec)
    if extra:
        lines += ["", extra]
    return lines


def _uncertain(spec: MotionSpec) -> list[str]:
    items = cp.uncertain_motion(spec) + cp.uncertain_appearance(spec)
    if not items:
        return []
    return [
        "UNCERTAIN (OPTIONAL)",
        "",
        "Possibly also (very low confidence; use only if it matches the reference):",
        *[f"- {u.text()}" for u in items],
    ]


# --------------------------------------------------------------------------------------------
# OUTPUT / CONSTRAINTS
# --------------------------------------------------------------------------------------------


def _motion_bullet(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    if not cp.needs_js(spec):
        text = (
            "- Motion: a CSS @keyframes animation on the track is enough: translate it by exactly "
            "one copy of the cards per loop, linear, infinite, so the wrap is invisible."
        )
        if c.behavior.pause is not None:
            text += (
                " For the pause use animation-play-state: paused; "
                "if the measured slowdown matters, "
                "drive the position from a small requestAnimationFrame loop instead."
            )
        return text
    b = c.behavior
    drivers = cp.motion_drivers(spec)
    text = (
        "- Motion: drive the track from one requestAnimationFrame loop that owns a single "
        f"position value ({ph.join_and(drivers)} "
        f"{'all change' if len(drivers) > 1 else 'changes'} that one value). Advance it "
        "with the real frame time (dt) so speeds do not depend on the frame rate, apply it with "
        "the translate property, and wrap it modulo the size of one copy of the cards so the "
        "track never runs out of content."
    )
    if b.pause is not None or b.resume is not None:
        text += (
            " Velocity ramps (slowdown and resume) interpolate the velocity with the easing "
            "curves given above."
        )
    if cp.card_scale(spec) is not None:
        text += (
            " Update the card scales and positions in the same loop, right after the track "
            "position (use translate and scale on the cards, not width / height, so the layout "
            "does not reflow every frame)."
        )
    return text


def _input_bullets(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    b = c.behavior
    out = []
    if b.drag is not None:
        out.append(
            "- Pointer: pointerdown / pointermove / pointerup / pointercancel with "
            "setPointerCapture; touch-action: "
            f"{cp.TOUCH_ACTION[c.axis]} on the scroller; user-select: none, and "
            'draggable="false" on placeholder images; cursor: grab, and grabbing while dragging.'
        )
    if cp.pauses_on_hover(spec):
        out.append(
            "- Hover: react to pointerenter / pointerleave from mouse pointers only (pointerType "
            '"mouse"), so a touch does not leave the scroller paused.'
        )
    return out


def _reduced_motion_bullet(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    text = (
        "- Reduced motion: under prefers-reduced-motion: reduce, do not autoplay at all (the "
        "track starts and stays still)"
    )
    b = c.behavior
    if b.drag is not None:
        moves = (
            "dragging and the release momentum still work" if b.inertia else "dragging still works"
        )
        text += f"; {moves} because the user drives them"
        if cp.inertia_blends_into_autoplay(spec):
            text += " (with no autoplay to blend into, the momentum decays to zero and stops)"
    return text + "."


def _output(spec: MotionSpec) -> list[str]:
    sentence = cp.html_file_sentence(script_needed=cp.needs_js(spec))
    has_appearance = bool(_appearance(spec))
    keep = (
        "At the recorded viewport size keep the measured sizes, colours and the scroller's "
        "position from APPEARANCE (place its top-left corner exactly there, e.g. with the page's "
        "padding or margins; do not centre it unless APPEARANCE says it is centred); below that "
        "width the scroller may span the full width while the cards keep their measured size. "
        if has_appearance
        else ""
    )
    return [
        "OUTPUT",
        "",
        sentence,
        "",
        cp.CONTENT_BULLET,
        "- Layout: mobile-first and fluid, working from "
        f"{cp.MIN_VIEWPORT_PX}px wide phones to wide desktops, with the standard responsive "
        f'<meta name="viewport"> tag. {keep}Keep every motion value above as specified in CSS '
        "px, px/s and ms: they are measured design values, so do not scale them with the viewport.",
        _motion_bullet(spec),
        *_input_bullets(spec),
        "- Focus and visibility: make the scroller keyboard-focusable with an aria-label and "
        "pause autoplay only while it has keyboard focus (:focus-visible), not after a pointer "
        "press or drag; pause while the tab is hidden (visibilitychange) and "
        "reset the frame clock when it becomes visible so the track does not jump.",
        _reduced_motion_bullet(spec),
        cp.SEMANTICS_BULLET,
    ]


def _constraints(spec: MotionSpec) -> list[str]:
    lines = [
        "CONSTRAINTS",
        "",
        f"All numbers are estimates measured from a screen recording "
        f"({ph.pixel_ratio_sentence(spec)}); keep them as targets, not exact source values.",
    ]
    if not spec.cursor.visible:
        lines.append(
            "The pointer was not visible in the recording; what starts and stops the motion is "
            "inferred from the motion alone."
        )
    if any(w.code == "tracking_degraded" for w in spec.warnings):
        lines.append(
            "Fast drags could not be tracked precisely; treat peak and release speeds as rough."
        )
    lines.append(FINAL_LINE)
    return lines


def render(spec: MotionSpec) -> str:
    out = [opening_line(spec), "", *_structure(spec)]
    appearance = _appearance(spec)
    if appearance:
        out += ["", *appearance]
    out += ["", *_behaviour(spec)]
    uncertain = _uncertain(spec)
    if uncertain:
        out += ["", *uncertain]
    out += ["", *_output(spec)]
    out += ["", *_constraints(spec)]
    return "\n".join(out) + "\n"
