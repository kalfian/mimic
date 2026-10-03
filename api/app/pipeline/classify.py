"""Interaction type + trigger rules on abstract features (PLAN §6.10).

:func:`classify` is a pure rule table over :class:`~app.models.measure.ClassifyFeatures`; the
first matching rule wins:

==  ==================================================  ===================  ==================
#   condition                                           type                 trigger
==  ==================================================  ===================  ==================
1   backdrop + appear                                   modal                click if stationary
2   appear, area < 40 % frame, appear_below_trigger     dropdown             enter / click / ?
3   resize_pattern                                      expand_collapse      as #2
4   round_trip, duration < 500 ms, scale or color       press                click
5   existing elements only, enter_before_onset          hover                pointer_enter
6   existing elements only, stationary_at_onset         click                click
7   no cursor, forward_reverse, no appear               hover (conf 0.4)     unknown
8   else                                                unknown              unknown
==  ==================================================  ===================  ==================

:func:`cursor_features` is an optional helper that derives the cursor booleans of
``ClassifyFeatures`` from a ``CursorTrack`` and the target box (§6.10 definitions).
"""

from __future__ import annotations

import numpy as np

from app.models.ir import Direction, InteractionType, Pattern, ReverseTriggerKind, TriggerKind
from app.models.measure import ClassifyFeatures, CursorTrack, HeuristicInteraction, Rect
from app.pipeline.params import DEFAULT_PARAMS, ClassifyParams, ConfidenceParams

_COLOR_PROPS = frozenset({"color", "background-color"})
_SCALE_PROPS = frozenset({"scale", "scaleX", "scaleY"})


def direction_for(f: ClassifyFeatures) -> Direction:
    if f.round_trip:
        return "round_trip"
    return "forward_reverse" if f.forward_reverse else "forward"


def classify(
    f: ClassifyFeatures,
    params: ClassifyParams = DEFAULT_PARAMS.classify,
    conf: ConfidenceParams = DEFAULT_PARAMS.confidence,
) -> HeuristicInteraction:
    """Apply the PLAN §6.10 rule table."""
    direction = direction_for(f)
    with_cursor = conf.classify_with_cursor
    shape_only = conf.classify_shape_only

    def result(
        rule: int, itype: InteractionType, trigger: TriggerKind, confidence: float
    ) -> HeuristicInteraction:
        return HeuristicInteraction(
            type=itype,
            trigger=trigger,
            reverse_trigger=_reverse_trigger(f, itype, trigger),
            direction=direction,
            confidence=confidence,
            rule=rule,
        )

    def opener_trigger() -> tuple[TriggerKind, float]:
        """Trigger for rules 2/3 (and 1 without the enter branch)."""
        if f.cursor_visible and f.enter_before_onset:
            return "pointer_enter", with_cursor
        if f.cursor_visible and f.stationary_at_onset:
            return "click", with_cursor
        return "unknown", shape_only

    # 1 — modal
    if f.has_backdrop and f.has_appear:
        if f.cursor_visible and f.stationary_at_onset:
            return result(1, "modal", "click", with_cursor)
        return result(1, "modal", "unknown", shape_only)

    # 2 — dropdown
    if (
        f.has_appear
        and f.appear_area_frac < params.dropdown_max_area_frac
        and f.appear_below_trigger
    ):
        trig, c = opener_trigger()
        return result(2, "dropdown", trig, c)

    # 3 — expand / collapse
    if f.resize_pattern:
        trig, c = opener_trigger()
        return result(3, "expand_collapse", trig, c)

    # 4 — press
    props = f.properties
    # "scale < 1": ``scale_direction`` when known, else any scale property (a round trip that
    # scales *up* is rare for a press).
    shrinks = f.scale_direction < 0 if f.scale_direction != 0 else bool(props & _SCALE_PROPS)
    if (
        f.round_trip
        and f.forward_duration_s * 1000.0 < params.press_max_duration_ms
        and (shrinks or props & _COLOR_PROPS)
    ):
        cursor_ok = f.cursor_visible and f.stationary_at_onset
        return result(4, "press", "click", with_cursor if cursor_ok else shape_only)

    existing_only = not f.has_appear and not f.has_backdrop and not f.resize_pattern
    # 5 — hover
    if existing_only and f.cursor_visible and f.enter_before_onset:
        return result(5, "hover", "pointer_enter", with_cursor)
    # 6 — click
    if existing_only and f.cursor_visible and f.stationary_at_onset:
        return result(6, "click", "click", with_cursor)
    # 7 — hover guessed from shape (no cursor)
    if not f.cursor_visible and f.forward_reverse and not f.has_appear:
        return result(7, "hover", "unknown", conf.classify_no_cursor_hover)
    # 8
    return result(8, "unknown", "unknown", conf.classify_unknown)


def _reverse_trigger(
    f: ClassifyFeatures, itype: InteractionType, trigger: TriggerKind
) -> ReverseTriggerKind:
    """Reverse trigger: ``none`` without a recorded reverse (or for a round trip)."""
    if f.round_trip or not f.forward_reverse:
        return "none"
    if trigger == "pointer_enter":
        return "pointer_leave" if f.leave_before_reverse else "unknown"
    if trigger == "click":
        return "click"
    return "unknown"


# --------------------------------------------------------------------------------------------
# Pattern
# --------------------------------------------------------------------------------------------

_ROLE_PATTERN: dict[str, Pattern] = {
    "card": "card",
    "button": "button",
    "link": "button",
    "badge": "button",
    "icon": "button",
    "dropdown_menu": "menu",
    "menu_item": "menu",
    "accordion_panel": "accordion",
    "modal_panel": "modal",
    "backdrop": "modal",
}
_TYPE_PATTERN: dict[str, Pattern] = {
    "modal": "modal",
    "dropdown": "menu",
    "expand_collapse": "accordion",
    "press": "button",
}


def pattern_for(role: str | None, interaction_type: InteractionType) -> Pattern:
    """``interaction.pattern`` from the target role, falling back to the interaction type."""
    if role is not None and role in _ROLE_PATTERN:
        return _ROLE_PATTERN[role]
    return _TYPE_PATTERN.get(interaction_type, "generic")


# --------------------------------------------------------------------------------------------
# Cursor features (optional helper for M1 / INT)
# --------------------------------------------------------------------------------------------


def cursor_features(
    cursor: CursorTrack,
    target: Rect,
    onset_s: float,
    reverse_onset_s: float | None = None,
    params: ClassifyParams = DEFAULT_PARAMS.classify,
) -> tuple[bool, bool, bool]:
    """``(enter_before_onset, stationary_at_onset, leave_before_reverse)`` (PLAN §6.10).

    * enter: the hotspot crosses into ``target`` within ``enter_window_ms`` of ``onset``.
    * stationary: the last ``stationary_min_frames`` steps before onset each move less than
      ``stationary_speed_css`` (px per sample), or the track marks them ``stationary``.
    * leave: the hotspot crosses out of ``target`` within the same window of the reverse onset.
    """
    if not cursor.visible or not cursor.samples:
        return False, False, False
    samples = sorted(cursor.samples, key=lambda s: s.t_s)
    known = [s for s in samples if s.state != "unknown"]
    if not known:
        return False, False, False
    lo, hi = params.enter_window_ms[0] / 1000.0, params.enter_window_ms[1] / 1000.0

    t = np.array([s.t_s for s in known])
    inside = np.array([_inside(target, s.x_css, s.y_css) for s in known])
    enters = t[1:][~inside[:-1] & inside[1:]]
    leaves = t[1:][inside[:-1] & ~inside[1:]]

    enter = bool(np.any((enters >= onset_s + lo) & (enters <= onset_s + hi)))
    leave = False
    if reverse_onset_s is not None:
        leave = bool(np.any((leaves >= reverse_onset_s + lo) & (leaves <= reverse_onset_s + hi)))

    before = [s for s in known if s.t_s <= onset_s + 1e-9]
    n = params.stationary_min_frames
    stationary = False
    if len(before) >= n + 1:
        last = before[-(n + 1) :]
        xy = np.array([[s.x_css, s.y_css] for s in last])
        steps = np.hypot(*np.diff(xy, axis=0).T)
        stationary = bool(
            np.all(steps < params.stationary_speed_css)
            or all(s.state == "stationary" for s in last[1:])
        )
    elif before and before[-1].state == "stationary":
        stationary = True
    return enter, stationary, leave


def _inside(r: Rect, x: float, y: float) -> bool:
    return r.x <= x <= r.x2 and r.y <= y <= r.y2


__all__ = ["classify", "cursor_features", "direction_for", "pattern_for"]
