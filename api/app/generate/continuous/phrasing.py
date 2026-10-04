"""Shared phrasing for the continuous-mode generators (PLAN-continuous §6). Pure, deterministic.

Builds on :mod:`app.generate.phrasing` (decimal half-up rounding, confidence wording, naming)
and keeps the same **number provenance** rule: every printed number is an IR value, at most
transformed by ``abs``, display rounding or x100 for confidence percentages. Derived magnitudes
(pitch - card width, loop duration, offsets between boxes) are never computed here; geometric
relations are only *compared* and described in words ("vertically centred").

Display rounding: speeds >= 100 px/s to 10, else to 1 (``≈39 px/s``); durations to 10 ms
(``≈700 ms``); lengths to whole px (``≈216 px``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.generate import phrasing as ph
from app.models.ir import (
    UNCERTAIN_BELOW,
    Box,
    CardScale,
    ContinuousMotion,
    Easing,
    ExponentialFit,
    MeasuredColor,
    MeasuredNumber,
    MotionElement,
    MotionSpec,
    Phase,
    RampFit,
    band_for,
)

# --------------------------------------------------------------------------------------------
# Output contract shared with the transition prompt (single self-contained HTML file): the
# sentences live in ``app.generate.phrasing``; these names keep the continuous call sites short
# --------------------------------------------------------------------------------------------

MIN_VIEWPORT_PX = ph.MIN_VIEWPORT_PX


def html_file_sentence(script_needed: bool) -> str:
    """The deliverable sentence; the script is mandatory when the motion needs the JS driver."""
    return ph.html_file_sentence("motion", script_required=script_needed)


CONTENT_BULLET = ph.content_bullet("motion")
SEMANTICS_BULLET = (
    "- Semantics: use semantic elements (a section with an aria-label for the scroller, a list "
    "or articles for the cards, headings, p) and mark decorative placeholders with "
    'aria-hidden="true".'
)

# --------------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------------


def speed_num(v: float) -> str:
    """|v| in px/s: >= 100 rounded to 10, else to 1 (``39``, ``1400``)."""
    step = "10" if abs(v) >= 100 else "1"
    return ph.fmt_decimal(ph.round_step(abs(v), step))


def signed_speed(v: float) -> str:
    """Signed velocity for tables: ``+39``, ``-1400``, ``0``."""
    n = speed_num(v)
    if n == "0":
        return "0"
    return f"+{n}" if v > 0 else f"-{n}"


def speed(v: float) -> str:
    """``≈39 px/s`` (magnitude; the direction is said in words)."""
    return f"≈{speed_num(v)} px/s"


def ms_num(v: float) -> str:
    return str(ph.ms_int(v))


def ms_approx(v: float) -> str:
    """``≈700 ms``."""
    return f"≈{ms_num(v)} ms"


def px_approx(v: float) -> str:
    """``≈216 px``."""
    return f"≈{ph.num_px(abs(v))} px"


def size_text(w: float, h: float) -> str:
    """``760×200 px``."""
    return f"{ph.num_px(w)}×{ph.num_px(h)} px"


def ratio_num(v: float) -> str:
    """Pointer ratio, 2 decimals: ``1.00``, ``0.98``."""
    return ph.scale_num(v)


def scale_factor(v: float) -> str:
    """Card scale / magnification, 2 decimals: ``1.17``."""
    return ph.scale_num(v)


# --------------------------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------------------------


def is_uncertain_score(score: float) -> bool:
    """``< 0.3``: kept out of the main text, listed under uncertain observations."""
    return score < UNCERTAIN_BELOW


def usable(m: MeasuredNumber | MeasuredColor | None) -> bool:
    """Measured and not uncertain."""
    return m is not None and not is_uncertain_score(m.confidence.value)


def hedged(text: str, score: float) -> str:
    """``≈700 ms`` / ``possibly ≈700 ms`` (low band)."""
    return f"possibly {text}" if band_for(score) == "low" else text


def hedged_note(text: str, score: float) -> str:
    """Hedged value + band note: ``possibly ≈700 ms (low confidence)``."""
    return hedged(text, score) + ph.band_note(band_for(score))


def conf_suffix(score: float) -> str:
    """`` — medium (70%)`` (technical tab)."""
    return f" — {ph.confidence_text(score)}"


def css_note(score: float) -> str:
    """``/* estimated, medium confidence */``."""
    return f"/* estimated, {band_for(score)} confidence */"


# --------------------------------------------------------------------------------------------
# Easing
# --------------------------------------------------------------------------------------------


def easing_word(e: Easing | None, conf: float) -> str | None:
    """Prose easing (``ease-in-out`` / fitted bezier / family) or None when not measured."""
    return None if e is None else ph.easing_text(e, conf)


def easing_using(e: Easing | None, conf: float) -> str:
    """`` using an ease-out curve`` (leading space) or ``""``."""
    return "" if e is None else " " + ph.easing_phrase(e, conf)


# --------------------------------------------------------------------------------------------
# Axis / direction wording
# --------------------------------------------------------------------------------------------

AXIS_WORD = {"x": "horizontal", "y": "vertical"}
AXIS_ADVERB = {"x": "horizontally", "y": "vertically"}
#: Direction word for a signed velocity on an axis (sign per ``SIGN_CONVENTION``).
DIRECTION = {("x", True): "right", ("x", False): "left", ("y", True): "down", ("y", False): "up"}
DIRECTION_PHRASE = {
    "right": "to the right",
    "left": "to the left",
    "up": "upward",
    "down": "downward",
}
#: Touch gesture the browser keeps when the scroller owns the drag axis.
TOUCH_ACTION = {"x": "pan-y", "y": "pan-x"}
POSITIVE_WORD = {"x": "right", "y": "down"}

PHASE_WORD = {
    "autoplay": "autoplay",
    "decelerate": "decelerate",
    "paused": "paused",
    "drag": "drag",
    "inertia": "inertia",
    "snap": "snap",
    "stop": "stop",
    "resume": "resume",
    "unknown": "unknown",
}


def direction_of(axis: str, v: float) -> str:
    return DIRECTION[(axis, v > 0)]


# --------------------------------------------------------------------------------------------
# Spec navigation
# --------------------------------------------------------------------------------------------


def cont(spec: MotionSpec) -> ContinuousMotion:
    assert spec.continuous is not None, "continuous generators need mode == 'continuous'"
    return spec.continuous


def scroller(spec: MotionSpec) -> MotionElement:
    return ph.element_map(spec)[cont(spec).element_id]


def scroller_name(spec: MotionSpec) -> str:
    """Mid-sentence name: ``scroller`` / ``product carousel``."""
    return ph.ref_name(scroller(spec))


def inner_elements(spec: MotionSpec) -> list[MotionElement]:
    """Elements below the scroller in hierarchy order (cards, card text)."""
    inside = ph.descendant_ids(spec, cont(spec).element_id)
    return [e for e in ph.element_order(spec) if e.id in inside]


def card_element(spec: MotionSpec) -> MotionElement | None:
    """The representative card: first non-text child of the scroller (``role: card`` first)."""
    sid = cont(spec).element_id
    kids = [e for e in ph.element_order(spec) if e.parent_id == sid and not e.text_like]
    cards = [e for e in kids if e.role == "card"]
    return (cards or kids or [None])[0]


def phases_of(spec: MotionSpec, kind: str) -> list[Phase]:
    return [p for p in cont(spec).phases if p.kind == kind]


def needs_js(spec: MotionSpec) -> bool:
    """Anything beyond a constant-speed loop (+ instant hover pause) needs the JS driver; so do
    cards that scale with their position (updated every frame)."""
    c = cont(spec)
    b = c.behavior
    if c.autoplay is None or card_scale(spec) is not None:
        return True
    if b.drag or b.inertia or b.snap or b.resume:
        return True
    return False


def motion_drivers(spec: MotionSpec) -> list[str]:
    """What moves the track, in words: ``autoplay``, ``dragging``, ``momentum``, ``snapping``."""
    c = cont(spec)
    b = c.behavior
    seen = (
        ("autoplay", c.autoplay is not None or b.resume is not None),
        ("dragging", b.drag is not None),
        ("momentum", b.inertia is not None),
        ("snapping", b.snap is not None and b.snap.kind == "grid"),
    )
    return [name for name, on in seen if on]


def inertia_blends_into_autoplay(spec: MotionSpec) -> bool:
    """Exponential momentum whose every fit decays towards a non-zero velocity (``v_inf``, the
    autoplay velocity): the release blends back into autoplay, which keeps running, instead of
    coming to rest (no rest, delay or resume ramp follows such a release)."""
    i = cont(spec).behavior.inertia
    if i is None or i.model != "exponential":
        return False
    fits = [p.fit for p in phases_of(spec, "inertia") if isinstance(p.fit, ExponentialFit)]
    return bool(fits) and all(speed_num(f.v_inf_px_s) != "0" for f in fits)


def pause_by_press_inferred(spec: MotionSpec) -> bool:
    """The pause trigger was not visible, but every time autoplay stopped a drag started (no
    slowdown phase, drags recorded): the only pause evidenced is the press that starts a drag.
    Generators then describe / implement a press pause, not a hover pause (after a release the
    motion carried on while the pointer was presumably still over the scroller)."""
    c = cont(spec)
    pz = c.behavior.pause
    return (
        pz is not None
        and pz.on == "unknown"
        and c.behavior.drag is not None
        and not phases_of(spec, "decelerate")
    )


def pause_trigger(spec: MotionSpec) -> str | None:
    """Effective pause trigger for the outputs: ``hover`` / ``press`` / ``unknown`` (implemented
    as hover) or None without a pause; ``press`` when :func:`pause_by_press_inferred`."""
    pz = cont(spec).behavior.pause
    if pz is None:
        return None
    return "press" if pause_by_press_inferred(spec) else pz.on


def pauses_on_hover(spec: MotionSpec) -> bool:
    return pause_trigger(spec) in ("hover", "unknown")


def drag_stops(spec: MotionSpec) -> list[Phase]:
    """Drags that end at rest with the pointer (the content rests right after them: a ``paused``
    phase follows instead of momentum): the pointer stopped before letting go."""
    ps = cont(spec).phases
    return [a for a, b in zip(ps, ps[1:], strict=False) if a.kind == "drag" and b.kind == "paused"]


def resumes_after_drag_stop(spec: MotionSpec) -> bool:
    """A resume follows a drag that ended at rest (drag → paused → resume)."""
    ps = cont(spec).phases
    return any(
        a.kind == "drag" and b.kind == "paused" and r.kind == "resume"
        for a, b, r in zip(ps, ps[1:], ps[2:], strict=False)
    )


def card_scale(spec: MotionSpec) -> CardScale | None:
    """The position-dependent card scale when measured and not uncertain (else None: rigid
    cards are described; an uncertain scale is only listed under the uncertain sections)."""
    cs = cont(spec).card_scale
    if cs is None or is_uncertain_score(cs.confidence.value):
        return None
    return cs


def card_scale_range(cs: CardScale) -> str:
    """``×1 at the centre → ≈×1.17 at ≈261 px from it`` (×1 is the model's definition, the
    other numbers are IR values)."""
    return (
        f"×1 at the centre → ≈×{scale_factor(cs.scale_at_reference)} at "
        f"{px_approx(cs.reference_distance_px)} from it"
    )


def resume_velocity(spec: MotionSpec) -> float | None:
    """Signed velocity autoplay resumes to: ``to_speed_px_s`` with the sign of the resumed
    motion (autoplay direction, flipped when not preserved; else the resume phase's end)."""
    c = cont(spec)
    r = c.behavior.resume
    if r is None:
        return None
    if c.autoplay is not None:
        positive = c.autoplay.velocity_px_s > 0
        if not r.direction_preserved:
            positive = not positive
    else:
        ramps = [p.fit for p in phases_of(spec, "resume") if isinstance(p.fit, RampFit)]
        end = ramps[0].to_px_s if ramps else phases_of(spec, "resume")[0].v_end_px_s
        positive = end >= 0
    return r.to_speed_px_s if positive else -r.to_speed_px_s


# --------------------------------------------------------------------------------------------
# Geometry relations (compared, never printed as derived numbers)
# --------------------------------------------------------------------------------------------

#: Two insets within this many CSS px count as equal ("centred").
CENTRED_TOL_PX = 2.0


def centred_in(inner: Box, outer: Box, axis: str) -> bool:
    """``inner`` is centred inside ``outer`` along ``axis`` (insets differ by <= 2 px)."""
    if axis == "x":
        a, b = inner.x - outer.x, (outer.x + outer.w) - (inner.x + inner.w)
    else:
        a, b = inner.y - outer.y, (outer.y + outer.h) - (inner.y + inner.h)
    return a >= 0 and b >= 0 and abs(a - b) <= CENTRED_TOL_PX


def vertical_place(inner: Box, outer: Box) -> str:
    """``top`` / ``middle`` / ``lower part`` of ``outer`` by the centre of ``inner``."""
    rel = (inner.y + inner.h / 2 - outer.y) / outer.h if outer.h > 0 else 0.5
    if rel < 1 / 3:
        return "upper part"
    if rel > 2 / 3:
        return "lower part"
    return "middle"


# --------------------------------------------------------------------------------------------
# Uncertain observations
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertain:
    """One value below the uncertainty threshold, phrased once for every tab."""

    what: str  # "pause slowdown duration"
    value: str  # "≈700 ms"
    score: float

    def text(self) -> str:
        return f"{self.what}: {self.value} ({ph.pct(self.score)} confidence)"


def _num_items(pairs: Sequence[tuple[str, MeasuredNumber | None, str]]) -> list[Uncertain]:
    out = []
    for what, m, kind in pairs:
        if m is None or not is_uncertain_score(m.confidence.value):
            continue
        value = {
            "speed": speed(m.value),
            "ms": ms_approx(m.value),
            "px": px_approx(m.value),
            "ratio": ratio_num(m.value),
        }[kind]
        out.append(Uncertain(what, value, m.confidence.value))
    return out


def uncertain_motion(spec: MotionSpec) -> list[Uncertain]:
    """Motion values (autoplay, behaviour, geometry of the band) below the threshold."""
    c = cont(spec)
    b = c.behavior
    pairs: list[tuple[str, MeasuredNumber | None, str]] = []
    if c.autoplay is not None:
        pairs += [
            ("autoplay speed", c.autoplay.speed_px_s, "speed"),
            ("loop length (one copy)", c.autoplay.loop.period_px, "px"),
            ("loop duration", c.autoplay.loop.duration_ms, "ms"),
        ]
    pairs += [("card pitch", c.pitch_px, "px"), ("gap between cards", c.gap_px, "px")]
    out = _num_items(pairs)
    cs = c.card_scale
    if cs is not None and is_uncertain_score(cs.confidence.value):
        out.append(
            Uncertain(
                "card scaling by distance from the scroller centre",
                card_scale_range(cs),
                cs.confidence.value,
            )
        )
    if b.pause is not None and is_uncertain_score(b.pause.on_confidence.value):
        out.append(Uncertain("pause trigger", b.pause.on, b.pause.on_confidence.value))
    pairs = []
    if b.pause is not None:
        pairs.append(("pause slowdown duration", b.pause.decel_ms, "ms"))
    if b.drag is not None:
        pairs += [
            ("pointer ratio", b.drag.pointer_ratio, "ratio"),
            ("peak drag speed", b.drag.peak_speed_px_s, "speed"),
        ]
    if b.inertia is not None:
        pairs += [
            ("momentum decay time constant", b.inertia.tau_ms, "ms"),
            ("momentum duration", b.inertia.duration_ms, "ms"),
        ]
    if b.snap is not None:
        pairs += [("snap step", b.snap.step_px, "px"), ("snap duration", b.snap.duration_ms, "ms")]
    if b.resume is not None:
        pairs += [
            ("resume delay after rest", b.resume.delay_after_rest_ms, "ms"),
            ("resume delay after release", b.resume.delay_after_release_ms, "ms"),
            ("resume ramp", b.resume.ramp_ms, "ms"),
        ]
    return out + _num_items(pairs)


def uncertain_appearance(spec: MotionSpec) -> list[Uncertain]:
    """Appearance values below the threshold (page background, element fills, text, radius)."""
    out: list[Uncertain] = []
    if spec.scene is not None and spec.scene.page_background is not None:
        bg = spec.scene.page_background
        if is_uncertain_score(bg.confidence.value):
            out.append(Uncertain("page background", bg.value, bg.confidence.value))
    if is_uncertain_score(cont(spec).region_confidence.value):
        r = cont(spec).region
        out.append(
            Uncertain("scroller band size", size_text(r.w, r.h), cont(spec).region_confidence.value)
        )
    for el in [scroller(spec), *inner_elements(spec)]:
        name = ph.ref_name(el)
        st = el.static
        for what, col in (("background", st.background_color), ("text colour", st.text_color)):
            if col is not None and is_uncertain_score(col.confidence.value):
                out.append(Uncertain(f"{name} {what}", col.value, col.confidence.value))
        out += _num_items(
            [
                (f"{name} corner radius", st.border_radius_px, "px"),
                (f"{name} font size", st.font_size_px, "px"),
            ]
        )
        if st.shadow is not None and is_uncertain_score(st.shadow.confidence.value):
            out.append(
                Uncertain(
                    f"{name} shadow", ph.shadow_css(st.shadow.value), st.shadow.confidence.value
                )
            )
    return out


def uncertain_phases(spec: MotionSpec) -> list[Phase]:
    return [p for p in cont(spec).phases if is_uncertain_score(p.confidence.value)]
