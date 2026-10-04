"""Suggested CSS implementation (PRD §21, PLAN §9.4). Pure template over the IR.

* Individual transform properties (``translate``, ``scale``) so translate and scale on one
  element keep independent timings (P8).
* Forward timing lives in the active-state rule, reverse timing in the base rule (CSS uses the
  transition of the destination state).
* Uncertain transitions (``overall < 0.3``) are emitted as comments, never as rules.
* Measured appearance (PLAN-continuous §13), only when the spec carries it: a ``body`` rule with
  the page background, and base rules gain ``width`` / ``height`` (non-text elements, from the
  state-A box), ``background-color``, ``color``, ``font-size`` next to the static radius /
  shadow. Elements without transitions then get a base rule too. Without appearance values the
  output is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.generate import phrasing as ph
from app.generate.phrasing import appearance_box, has_appearance
from app.models.ir import MeasuredColor, MotionElement, MotionSpec, Transition

HEADER = "/* Suggested implementation — estimated from a screen recording, not the original CSS. */"

#: IR property -> CSS property it is written to.
CSS_PROP = {
    "translateX": "translate",
    "translateY": "translate",
    "scale": "scale",
    "scaleX": "scale",
    "scaleY": "scale",
    "opacity": "opacity",
    "color": "color",
    "background-color": "background-color",
    "box-shadow": "box-shadow",
    "border-radius": "border-radius",
    "height": "height",
}


@dataclass
class _Rule:
    selector: str
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([f"{self.selector} {{", *self.lines, "}"])


def _uses_grid_idiom(t: Transition) -> bool:
    """``height`` from/to zero → ``grid-template-rows: 0fr ↔ 1fr`` idiom."""
    return t.property == "height" and (t.from_.number == 0 or t.to.number == 0)  # type: ignore[union-attr]


def _css_prop(t: Transition) -> str:
    return "grid-template-rows" if _uses_grid_idiom(t) else CSS_PROP[t.property]


def _comment(t: Transition) -> str:
    band = t.confidence.band
    if t.property == "box-shadow":
        return f" /* estimated, {band} confidence */"
    return " /* low confidence */" if band == "low" else ""


def _stagger_ids(spec: MotionSpec) -> dict[str, int]:
    """Transition id -> stagger interval_ms for members of a stagger relationship."""
    out: dict[str, int] = {}
    for r in spec.relationships:
        if r.kind == "stagger" and r.interval_ms is not None:
            for tid in r.transition_ids:
                out[tid] = r.interval_ms
    return out


def _pick(ts: list[Transition]) -> dict[str, Transition]:
    """One representative transition per CSS property (largest change wins ties for translate)."""
    out: dict[str, Transition] = {}
    for t in ts:
        prop = _css_prop(t)
        cur = out.get(prop)
        if cur is None or abs(t.delta or 0) > abs(cur.delta or 0):
            out[prop] = t
    return out


def _transition_decl(spec: MotionSpec, ts: list[Transition]) -> list[str]:
    stagger = _stagger_ids(spec)
    entries = []
    for prop, t in _pick(ts).items():
        entry = f"{prop} {ph.ms(t.duration_ms)} {ph.css_easing(t.easing, t.confidence.easing)}"
        if t.id in stagger:
            entry += f" calc(var(--i, 0) * {ph.ms(stagger[t.id])})"
        elif ph.ms_int(t.delay_ms) > 0:
            entry += f" {ph.ms(t.delay_ms)}"
        entries.append(entry)
    if not entries:
        return []
    lines = (
        [f"  transition: {entries[0]};"]
        if len(entries) == 1
        else ["  transition:", *[f"    {e}," for e in entries[:-1]], f"    {entries[-1]};"]
    )
    if any(t.id in stagger for t in ts):
        lines.append("  /* stagger: set --i to each item's index, starting at zero */")
    return lines


def _state_values(ts: list[Transition], side: str) -> list[str]:
    """Declarations for one state (``from`` or ``to``) of an element's transitions."""
    out: list[str] = []
    tx = next((t for t in ts if t.property == "translateX"), None)
    ty = next((t for t in ts if t.property == "translateY"), None)
    done: set[str] = set()
    for t in ts:
        v = t.from_ if side == "from" else t.to
        prop = _css_prop(t)
        if prop in done:
            continue
        done.add(prop)
        if prop == "translate":
            x = (tx.from_ if side == "from" else tx.to).number if tx else 0.0  # type: ignore[union-attr]
            y = (ty.from_ if side == "from" else ty.to).number if ty else 0.0  # type: ignore[union-attr]
            src = [t for t in (tx, ty) if t is not None]
            note = " /* low confidence */" if all(s.confidence.band == "low" for s in src) else ""
            out.append(f"  translate: {ph.css_px(x)} {ph.css_px(y)};{note}")
        elif prop == "scale":
            sx = next((s for s in ts if s.property == "scaleX"), None)
            sy = next((s for s in ts if s.property == "scaleY"), None)
            if sx is not None or sy is not None:
                vx = (sx.from_ if side == "from" else sx.to).number if sx else 1.0  # type: ignore[union-attr]
                vy = (sy.from_ if side == "from" else sy.to).number if sy else 1.0  # type: ignore[union-attr]
                out.append(f"  scale: {ph.scale_num(vx)} {ph.scale_num(vy)};{_comment(t)}")
            else:
                out.append(f"  scale: {ph.value_css(t.property, v)};{_comment(t)}")
        elif prop == "grid-template-rows":
            fr = "0fr" if v.number == 0 else "1fr"  # type: ignore[union-attr]
            out.append(f"  grid-template-rows: {fr};{_comment(t)}")
        else:
            out.append(f"  {prop}: {ph.value_css(t.property, v)};{_comment(t)}")
    return out


def _uncertain_comments(ts: list[Transition], side: str) -> list[str]:
    out = []
    for t in ts:
        v = t.from_ if side == "from" else t.to
        out.append(
            f"  /* uncertain ({ph.pct(t.confidence.overall)} confidence), omitted: "
            f"{t.property} {ph.value_short(t.property, v)}, {ph.ms(t.duration_ms)} */"
        )
    return out


def _selectors(spec: MotionSpec, names: dict[str, str]) -> dict[str, str]:
    """Element id -> active-state selector."""
    it = spec.interaction
    target = ph.effective_target(spec)
    if target is None:
        return {}
    tname = f".{names[target.id]}"
    if it.type in ph.TOGGLE_TYPES:
        state = '[data-state="open"]'
    elif it.type == "press":
        state = ":active"
    else:  # hover / unknown -> hover
        state = ":hover"
    inside = ph.descendant_ids(spec, target.id)
    out = {}
    for el in spec.elements:
        if el.id == target.id:
            out[el.id] = f"{tname}{state}"
        elif el.id in inside:
            out[el.id] = f"{tname}{state} .{names[el.id]}"
        else:  # outside the target: drive with its own state attribute
            out[el.id] = f'.{names[el.id]}[data-state="open"]'
    return out


def _estimated(band: str) -> str:
    return f"/* estimated, {band} confidence */"


def _color_decl(prop: str, c: MeasuredColor) -> str:
    return f"  {prop}: {c.value}; {_estimated(c.confidence.band)}"


def _appearance_decls(el: MotionElement, animated: set[str]) -> list[str]:
    """Measured size / colours / font size (only called when the spec has appearance)."""
    out = []
    st = el.static
    if not el.text_like:  # a text box's size follows its (placeholder) text
        b = appearance_box(el)
        out.append(f"  width: {ph.css_px(b.w)};")
        if "height" not in animated:
            out.append(f"  height: {ph.css_px(b.h)};")
    if st.background_color is not None and "background-color" not in animated:
        out.append(_color_decl("background-color", st.background_color))
    if st.text_color is not None and "color" not in animated:
        out.append(_color_decl("color", st.text_color))
    if st.font_size_px is not None:
        fs = st.font_size_px
        out.append(f"  font-size: {ph.css_px(fs.value)}; {_estimated(fs.confidence.band)}")
    return out


def _static_decls(el: MotionElement, animated: set[str], appearance: bool = False) -> list[str]:
    out = _appearance_decls(el, animated) if appearance else []
    r = el.static.border_radius_px
    if r is not None and "border-radius" not in animated:
        out.append(
            f"  border-radius: {ph.css_px(r.value)}; "
            f"/* estimated, {r.confidence.band} confidence */"
        )
    s = el.static.shadow
    if s is not None and s.value is not None and "box-shadow" not in animated:
        out.append(
            f"  box-shadow: {ph.shadow_css(s.value)}; "
            f"/* estimated, {s.confidence.band} confidence */"
        )
    return out


def _header(spec: MotionSpec, names: dict[str, str]) -> list[str]:
    it = spec.interaction
    target = ph.effective_target(spec)
    lines = [HEADER]
    tsel = f".{names[target.id]}" if target is not None else "the target"
    lines.append(f"/* All values are approximate; {ph.pixel_ratio_sentence(spec)}. */")
    if it.type in ph.TOGGLE_TYPES:
        lines.append(f'/* Toggle data-state="open" on {tsel} via JS (e.g. on click). */')
    elif it.type == "unknown" or it.trigger.kind == "unknown":
        lines.append("/* Trigger could not be determined; implemented as hover. */")
    if ph.reverse_segment(spec) is None:
        lines.append("/* Reverse not recorded: base rules reuse the forward timings. */")
    if spec.scene is not None:
        vw = spec.scene.viewport_css
        lines.append(
            f"/* Sizes and colors measured at the recorded {ph.num_px(vw.w)} x "
            f"{ph.num_px(vw.h)}px viewport. */"
        )
    return lines


def _page_rule(spec: MotionSpec) -> list[str]:
    if spec.scene is None or spec.scene.page_background is None:
        return []
    return [_Rule("body", [_color_decl("background-color", spec.scene.page_background)]).text()]


def render(spec: MotionSpec) -> str:
    names = ph.class_names(spec)
    selectors = _selectors(spec, names)
    fwd = ph.forward_segment(spec)
    rev = ph.reverse_segment(spec)
    blocks: list[str] = ["\n".join(_header(spec, names))]
    appearance = has_appearance(spec)
    blocks += _page_rule(spec)

    for el in ph.element_order(spec):
        f_all = [t for t in ph.segment_transitions(spec, fwd.id) if t.element_id == el.id]
        r_all = (
            [t for t in ph.segment_transitions(spec, rev.id) if t.element_id == el.id]
            if rev is not None
            else []
        )
        if not f_all and not r_all:
            if appearance:  # static element: measured appearance only
                static = _static_decls(el, set(), appearance=True)
                if static:
                    blocks.append(_Rule(f".{names[el.id]}", static).text())
            continue
        order = lambda t: ph.PROPERTY_ORDER.index(t.property)  # noqa: E731
        f_main = sorted((t for t in f_all if not ph.is_uncertain(t)), key=order)
        f_unc = sorted((t for t in f_all if ph.is_uncertain(t)), key=order)
        r_main = sorted((t for t in r_all if not ph.is_uncertain(t)), key=order)
        r_unc = sorted((t for t in r_all if ph.is_uncertain(t)), key=order)
        content = [t for t in f_main if t.property == "content"]
        f_main = [t for t in f_main if t.property != "content"]
        r_main = [t for t in r_main if t.property != "content"]

        base = _Rule(f".{names[el.id]}")
        if any(t.property == "height" and _uses_grid_idiom(t) for t in f_main):
            base.lines += [
                "  /* height animates via the grid-rows idiom: this element is a grid wrapper",
                "     whose single child has min-height: 0 and overflow: hidden.",
                "     Alternative: transition max-height to the measured open height. */",
                "  display: grid;",
            ]
        origin = next((t.transform_origin for t in f_main if t.transform_origin), None)
        if origin is not None:
            base.lines.append(f"  transform-origin: {origin};")
        base.lines += _static_decls(el, {t.property for t in f_main}, appearance)
        base.lines += _state_values(
            [t for t in f_main if not ph.is_identity(t.property, t.from_)], "from"
        )
        if rev is not None:
            base.lines += _transition_decl(spec, r_main)
            base.lines += _uncertain_comments(r_unc, "to")
        else:
            base.lines += _transition_decl(spec, f_main)
        for t in content:
            base.lines.append(
                f"  /* content swaps ({ph.value_short('content', t.from_)} to "
                f"{ph.value_short('content', t.to)}): not a CSS transition; "
                "cross-fade with opacity if needed */"
            )

        active = _Rule(selectors[el.id])
        active.lines += _state_values(f_main, "to")
        active.lines += _transition_decl(spec, f_main)
        active.lines += _uncertain_comments(f_unc, "to")

        if base.lines:
            blocks.append(base.text())
        if active.lines:
            blocks.append(active.text())

    return "\n\n".join(blocks) + "\n"
