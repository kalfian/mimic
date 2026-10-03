"""Track M2 — classify.py: feature dicts → rules (PLAN §6.10)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.models.measure import ClassifyFeatures, CursorSample, CursorTrack, Rect
from app.pipeline import classify as cl

BASE = ClassifyFeatures(
    cursor_visible=True,
    enter_before_onset=False,
    stationary_at_onset=False,
    leave_before_reverse=False,
    has_appear=False,
    appear_area_frac=0.0,
    has_backdrop=False,
    appear_below_trigger=False,
    resize_pattern=False,
    round_trip=False,
    forward_reverse=True,
    forward_duration_s=0.28,
    properties=frozenset({"translateY"}),
)


def f(**kw: object) -> ClassifyFeatures:
    return replace(BASE, **kw)  # type: ignore[arg-type]


def summary(h) -> tuple:  # type: ignore[no-untyped-def]
    return (h.rule, h.type, h.trigger, h.reverse_trigger, h.direction, h.confidence)


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        # 1 modal
        (
            f(has_backdrop=True, has_appear=True, appear_area_frac=0.3, stationary_at_onset=True),
            (1, "modal", "click", "click", "forward_reverse", 0.85),
        ),
        (
            f(has_backdrop=True, has_appear=True, cursor_visible=False, forward_reverse=False),
            (1, "modal", "unknown", "none", "forward", 0.55),
        ),
        # 2 dropdown
        (
            f(
                has_appear=True,
                appear_area_frac=0.1,
                appear_below_trigger=True,
                stationary_at_onset=True,
            ),
            (2, "dropdown", "click", "click", "forward_reverse", 0.85),
        ),
        (
            f(
                has_appear=True,
                appear_area_frac=0.1,
                appear_below_trigger=True,
                enter_before_onset=True,
                leave_before_reverse=True,
            ),
            (2, "dropdown", "pointer_enter", "pointer_leave", "forward_reverse", 0.85),
        ),
        (
            f(has_appear=True, appear_area_frac=0.1, appear_below_trigger=True),
            (2, "dropdown", "unknown", "unknown", "forward_reverse", 0.55),
        ),
        # 3 expand/collapse (also: large appear that is not below a trigger falls through)
        (
            f(
                resize_pattern=True,
                has_appear=True,
                appear_area_frac=0.5,
                appear_below_trigger=True,
                stationary_at_onset=True,
                forward_reverse=False,
            ),
            (3, "expand_collapse", "click", "none", "forward", 0.85),
        ),
        # 4 press
        (
            f(
                round_trip=True,
                forward_reverse=False,
                forward_duration_s=0.21,
                properties=frozenset({"scale"}),
                stationary_at_onset=True,
            ),
            (4, "press", "click", "none", "round_trip", 0.85),
        ),
        (
            f(
                round_trip=True,
                forward_reverse=False,
                forward_duration_s=0.15,
                properties=frozenset({"background-color"}),
                cursor_visible=False,
            ),
            (4, "press", "click", "none", "round_trip", 0.55),
        ),
        # 5 hover
        (
            f(enter_before_onset=True, leave_before_reverse=True),
            (5, "hover", "pointer_enter", "pointer_leave", "forward_reverse", 0.85),
        ),
        (
            f(enter_before_onset=True, forward_reverse=False),
            (5, "hover", "pointer_enter", "none", "forward", 0.85),
        ),
        (
            f(enter_before_onset=True, stationary_at_onset=True),  # first matching rule wins
            (5, "hover", "pointer_enter", "unknown", "forward_reverse", 0.85),
        ),
        # 6 click
        (
            f(stationary_at_onset=True, forward_reverse=False),
            (6, "click", "click", "none", "forward", 0.85),
        ),
        # 7 no cursor, forward_reverse
        (
            f(cursor_visible=False),
            (7, "hover", "unknown", "unknown", "forward_reverse", 0.4),
        ),
        # 8 fallthrough
        (f(), (8, "unknown", "unknown", "unknown", "forward_reverse", 0.3)),
        (
            f(cursor_visible=False, forward_reverse=False),
            (8, "unknown", "unknown", "none", "forward", 0.3),
        ),
        # press guards: too long / no scale or color -> not a press
        (
            f(
                round_trip=True,
                forward_reverse=False,
                forward_duration_s=0.6,
                properties=frozenset({"scale"}),
                stationary_at_onset=True,
            ),
            (6, "click", "click", "none", "round_trip", 0.85),
        ),
        (
            f(
                round_trip=True,
                forward_reverse=False,
                properties=frozenset({"translateY"}),
                cursor_visible=False,
            ),
            (8, "unknown", "unknown", "none", "round_trip", 0.3),
        ),
    ],
)
def test_rules(features: ClassifyFeatures, expected: tuple) -> None:
    assert summary(cl.classify(features)) == expected


def test_dropdown_requires_small_appear_below_trigger() -> None:
    big = f(
        has_appear=True, appear_area_frac=0.45, appear_below_trigger=True, stationary_at_onset=True
    )
    h = cl.classify(big)
    assert h.type != "dropdown"
    # appear without backdrop is not "existing elements only" -> no click rule either
    assert h.rule == 8


def test_pattern_for() -> None:
    assert cl.pattern_for("card", "hover") == "card"
    assert cl.pattern_for("menu_item", "hover") == "menu"
    assert cl.pattern_for("modal_panel", "unknown") == "modal"
    assert cl.pattern_for("other", "dropdown") == "menu"
    assert cl.pattern_for(None, "expand_collapse") == "accordion"
    assert cl.pattern_for(None, "press") == "button"
    assert cl.pattern_for("text", "hover") == "generic"


# --------------------------------------------------------------------------------------------
# cursor_features helper
# --------------------------------------------------------------------------------------------

CARD = Rect(100, 100, 300, 400)


def track(points: list[tuple[float, float, float, str]]) -> CursorTrack:
    return CursorTrack(
        visible=True, samples=[CursorSample(t, x, y, s) for t, x, y, s in points], confidence=0.9
    )


def test_cursor_hover_enter_and_leave() -> None:
    pts = [(0.5 + i / 30, 20 + i * 10, 300, "moving") for i in range(20)]  # crosses x=100
    enter_t = next(t for t, x, _, _ in pts if x >= 100)
    leave_pts = [(2.5 + i / 30, 300 + i * 10, 300, "moving") for i in range(20)]  # exits x=400
    leave_t = next(t for t, x, _, _ in leave_pts if x > 400)
    enter, stationary, leave = cl.cursor_features(
        track(pts + leave_pts), CARD, onset_s=enter_t + 0.02, reverse_onset_s=leave_t + 0.03
    )
    assert enter and not stationary and leave
    # onset far after the enter -> not "enter before onset"
    enter, _, _ = cl.cursor_features(track(pts), CARD, onset_s=enter_t + 0.6)
    assert not enter


def test_cursor_stationary_click() -> None:
    pts = [(0.5 + i / 30, 50 + i * 15, 300, "moving") for i in range(10)]
    last_t = pts[-1][0]
    pts += [(last_t + (i + 1) / 30, pts[-1][1] + 0.5 * (i % 2), 300, "moving") for i in range(15)]
    onset = pts[-1][0] + 0.01
    enter, stationary, _ = cl.cursor_features(track(pts), CARD, onset_s=onset)
    assert stationary and not enter
    # state-only evidence (cursor blob vanished: samples marked stationary)
    pts2 = pts[:10] + [(last_t + (i + 1) / 30, pts[9][1], 300, "stationary") for i in range(4)]
    assert cl.cursor_features(track(pts2), CARD, onset_s=pts2[-1][0])[1]


def test_cursor_invisible_or_unknown() -> None:
    assert cl.cursor_features(CursorTrack(visible=False), CARD, onset_s=1.0) == (
        False,
        False,
        False,
    )
    unknown = track([(0.5, 200, 200, "unknown"), (0.6, 200, 200, "unknown")])
    assert cl.cursor_features(unknown, CARD, onset_s=1.0) == (False, False, False)
