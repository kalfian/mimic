"""CSS tab, continuous mode (PLAN-continuous §6.3, §13). Pure template over the IR.

* Layout rules carry the measured appearance (band size, card size, gap, colours, radius).
* Autoplay-only marquees (no drag / inertia / snap / resume) get a ``@keyframes`` loop; anything
  interactive is driven by the JS tab and the keyframes are omitted.
* Values below the uncertainty threshold are emitted as comments, never as declarations.
"""

from __future__ import annotations

from app.generate import phrasing as ph
from app.generate.continuous import phrasing as cp
from app.models.ir import Box, MeasuredColor, MeasuredNumber, MotionElement, MotionSpec

HEADER = "/* Suggested implementation — estimated from a screen recording, not the original CSS. */"


def _rule(selector: str, lines: list[str]) -> str:
    return "\n".join([f"{selector} {{", *lines, "}"])


def _decl(prop: str, value: str, score: float) -> str:
    """``  gap: 16px; /* estimated, medium confidence */`` or an ``uncertain`` comment."""
    if cp.is_uncertain_score(score):
        return f"  /* uncertain ({ph.pct(score)} confidence), omitted: {prop} {value} */"
    return f"  {prop}: {value}; {cp.css_note(score)}"


def _px_decl(prop: str, m: MeasuredNumber | None) -> list[str]:
    return [] if m is None else [_decl(prop, ph.css_px(m.value), m.confidence.value)]


def _color_decl(prop: str, m: MeasuredColor | None) -> list[str]:
    return [] if m is None else [_decl(prop, m.value, m.confidence.value)]


def _static_decls(el: MotionElement) -> list[str]:
    st = el.static
    out = _color_decl("background-color", st.background_color)
    out += _px_decl("border-radius", st.border_radius_px)
    if st.shadow is not None and st.shadow.value is not None:
        out.append(_decl("box-shadow", ph.shadow_css(st.shadow.value), st.shadow.confidence.value))
    out += _color_decl("color", st.text_color)
    out += _px_decl("font-size", st.font_size_px)
    return out


def _autoplay_css(spec: MotionSpec, scroller: str, track: str) -> tuple[list[str], list[str]]:
    """Autoplay loop for marquees (PLAN-continuous §6.3): (track declarations, extra blocks).

    Empty when the JS driver is needed (drag / momentum / snap / resume) or there is no autoplay.
    """
    c = cp.cont(spec)
    a = c.autoplay
    if cp.needs_js(spec) or a is None:
        return [], []
    name = f"{scroller}-autoplay"
    axis_pos = "{} 0" if c.axis == "x" else "0 {}"
    loop = a.loop
    decls: list[str] = []
    if loop.observed and cp.usable(loop.period_px) and cp.usable(loop.duration_ms):
        assert loop.period_px is not None and loop.duration_ms is not None
        # the period is a magnitude: negate it in CSS so the printed number stays the IR value
        shift = f"calc(0px - {ph.num_px(loop.period_px.value)}px)"
        decls.append(f"  animation: {name} {ph.ms(loop.duration_ms.value)} linear infinite;")
    else:
        shift = "calc(0px - var(--copy-size) * 1px)"
        decls += [
            "  /* loop length not observed: set --copy-size to one copy's width / height in px "
            "(unitless) */",
            f"  animation: {name} linear infinite;",
            "  animation-duration: calc(var(--copy-size) / "
            f"{cp.speed_num(a.speed_px_s.value)} * 1s);"
            f" {cp.css_note(a.speed_px_s.confidence.value)}",
        ]
    if a.direction in ("right", "down"):
        decls.append("  animation-direction: reverse;")
    blocks = [_rule(f"@keyframes {name}", [f"  to {{ translate: {axis_pos.format(shift)}; }}"])]
    pz = c.behavior.pause
    if pz is not None:
        trigger = ":active" if cp.pause_trigger(spec) == "press" else ":hover"
        lines = ["  animation-play-state: paused;"]
        if pz.decel_ms is not None and cp.usable(pz.decel_ms):
            lines.append(
                f"  /* measured: slows over {cp.ms_approx(pz.decel_ms.value)} rather than "
                "stopping instantly — see the JS tab */"
            )
        if cp.pause_by_press_inferred(spec):
            lines.append("  /* trigger not visible in the recording; every pause was a press */")
        elif pz.on == "unknown":
            lines.append("  /* trigger not visible in the recording; implemented as hover */")
        blocks.append(_rule(f".{scroller}{trigger} .{track}", lines))
    blocks.append(
        _rule(
            "@media (prefers-reduced-motion: reduce)",
            [f"  .{track} {{", "    animation: none;", "  }"],
        )
    )
    return decls, blocks


def render(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    names = ph.class_names(spec)
    sc = cp.scroller(spec)
    scroller = names[sc.id]
    track = f"{scroller}__track"
    b = c.behavior
    js = cp.needs_js(spec)

    header = [HEADER, f"/* All values are approximate; {ph.pixel_ratio_sentence(spec)}. */"]
    copies = "the cards repeated twice (two identical copies) for a seamless loop"
    header.append(f"/* Markup: .{scroller} > .{track} > {copies}. */")
    cs = cp.card_scale(spec)
    if js:
        names_ = cp.motion_drivers(spec) + (["position-dependent card scaling"] if cs else [])
        drivers = ph.upper_first(ph.join_and(names_))
        header.append(f"/* {drivers} need the JS driver (JS tab); these rules are layout only. */")
    blocks = ["\n".join(header)]

    r = c.region
    rscore = c.region_confidence.value
    # the scroller's place at the recorded viewport: centred -> auto margins (as recorded), else
    # its top-left corner (round trip §14 compares the box at ±2 px)
    vp = spec.scene.viewport_css if spec.scene is not None else None
    placed = vp is not None and not cp.centred_in(r, Box(x=0.0, y=0.0, w=vp.w, h=vp.h), "x")
    body: list[str] = ["  margin: 0;"] if placed else []
    if spec.scene is not None and spec.scene.page_background is not None:
        body += _color_decl("background-color", spec.scene.page_background)
    if body:
        blocks.append(_rule("body", body))

    lines = ["  display: flex;"]
    card = cp.card_element(spec)
    other = "y" if c.axis == "x" else "x"
    if card is not None and cp.centred_in(card.bbox_initial, r, other):
        lines.append("  align-items: center;")
    lines += [_decl("max-width", ph.css_px(r.w), rscore), _decl("height", ph.css_px(r.h), rscore)]
    if placed:
        lines.append(
            f"  margin: {ph.css_px(r.y)} 0 0 {ph.css_px(r.x)}; "
            "/* top-left corner at the recorded viewport size, not centred */"
        )
    else:
        lines.append("  margin-inline: auto;")
    lines.append("  overflow: hidden;")
    if cs is not None:
        lines.append("  position: relative; /* the JS driver reads card offsets from here */")
    if b.drag is not None:
        lines += [
            f"  touch-action: {cp.TOUCH_ACTION[c.axis]}; /* page scroll stays on the other axis */",
            "  user-select: none;",
            "  cursor: grab;",
        ]
    lines += _static_decls(sc)
    blocks.append(_rule(f".{scroller}", lines))
    if b.drag is not None:
        blocks.append(_rule(f".{scroller}:active", ["  cursor: grabbing;"]))

    lines = ["  display: flex;", "  flex: none;"]
    if c.axis == "y":
        lines.append("  flex-direction: column;")
    if c.gap_px is not None:
        lines += _px_decl("gap", c.gap_px)
    elif c.pitch_px is not None and card is not None:
        size = card.bbox_initial.w if c.axis == "x" else card.bbox_initial.h
        lines.append(
            _decl(
                "gap",
                f"calc({ph.css_px(c.pitch_px.value)} - {ph.css_px(size)})",
                c.pitch_px.confidence.value,
            )
        )
        lines.append("  /* gap not measured: card pitch minus card size */")
    lines.append(f"  {'width' if c.axis == 'x' else 'height'}: max-content;")
    if cs is not None:
        lines.append("  position: relative;")
    if js:
        # the driver moves the track every frame: its own layer keeps sub-pixel motion smooth
        # (the cards get no layer of their own: their scale changes every frame, and fixed-scale
        # layer textures re-rasterised mid-motion showed up as ±1 px jitter in round trips)
        lines.append("  will-change: translate;")
    autoplay_decls, autoplay_blocks = _autoplay_css(spec, scroller, track)
    blocks.append(_rule(f".{track}", lines + autoplay_decls))

    for el in cp.inner_elements(spec):
        lines = []
        if card is not None and el.id == card.id:
            lines += [
                "  flex: none;",
                f"  width: {ph.css_px(el.bbox_initial.w)};",
                f"  height: {ph.css_px(el.bbox_initial.h)};",
            ]
            if cs is not None:
                lines += [
                    "  /* size at the scroller centre; translate + scale are set every frame by "
                    "the JS driver",
                    "     from the card's distance to the centre: "
                    f"{cp.card_scale_range(cs)} (estimated, {cs.confidence.band} confidence) */",
                    "  transform-origin: center;",
                ]
        lines += _static_decls(el)
        if lines:
            blocks.append(_rule(f".{names[el.id]}", lines))

    blocks += autoplay_blocks
    return "\n\n".join(blocks) + "\n"
