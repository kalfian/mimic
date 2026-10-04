"""Scenario definitions S1–S12, C1–C14, N1 + ground-truth derivation (PLAN §11.3,
PLAN-continuous §8.2).

Each builder returns a :class:`ScenarioDef`: the scene (nodes, tweens, cursor path) **and** the
semantic metadata the renderer can't know (roles, expected element kinds, interaction type,
explicit timing relationships). The truth file is derived from the very same tweens that drive
the renderer, so the two cannot drift apart.

Canvas: 1280×800 CSS px (2560×1600 device px for the Retina variant). A static "page chrome"
(header, sidebar, footer text) keeps frames from being empty; scenario content sits right of
the sidebar.

The continuous scenarios (C*, N1) add scroller strips whose track ``translate`` is driven by a
velocity-integrated :class:`kinematics.VelocityProfile` (``Scene.drivers``); their truth is
derived from the same profile. Every truth also carries the rendered appearance (§13).
"""

from __future__ import annotations

import functools
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from app.models.ir import Box, ColorValue, PxValue, RatioValue, Shadow, ShadowValue, Size

from .animate import (
    CSS_KEYWORDS,
    EASINGS,
    CursorKey,
    CursorPath,
    Timeline,
    Tween,
    first_time,
    solve_start_for_crossing,
)
from .encode import EncodeSpec
from .kinematics import (
    Autoplay,
    Drag,
    DragCursor,
    EdgeZoom,
    HeldDrag,
    Hold,
    Inertia,
    Ramp,
    SnapTween,
    VelocityProfile,
    card_zoom_driver,
    strip_driver,
)
from .scene import Driver, Node, Scene, build_sprite, hex_to_rgb, rgb_to_hex, text_size
from .truth import (
    Truth,
    TruthAppearance,
    TruthCard,
    TruthContinuous,
    TruthCursor,
    TruthCursorEvent,
    TruthEasing,
    TruthElement,
    TruthInteraction,
    TruthLoop,
    TruthPhase,
    TruthRelationship,
    TruthScene,
    TruthSegment,
    TruthTransition,
    TruthVideo,
    TruthZoom,
)

CSS_W, CSS_H = 1280, 800
PAGE_BG = "#F3F4F6"
SEGMENT_ORDER = ("fwd", "rev", "rt_in", "rt_out")


def c(h: str) -> tuple[float, float, float]:
    return hex_to_rgb(h)


# --------------------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementMeta:
    """A scene node the pipeline is expected to report as an element."""

    node: str
    label: str
    role: str
    kind: str
    text_like: bool = False


@dataclass(frozen=True)
class RelSpec:
    """Explicit timing relationship; ``keys`` are ``(node, prop)`` within ``segment``."""

    kind: str
    segment: str
    keys: tuple[tuple[str, str], ...]
    offset_ms: int | None = None
    interval_ms: int | None = None


@dataclass
class ScenarioDef:
    scenario: str
    description: str
    checks: list[str]
    scene: Scene
    elements: list[ElementMeta] = field(default_factory=list)
    interaction: dict | None = None  # TruthInteraction fields except boxes
    trigger_node: str | None = None
    segment_triggers: dict[str, float] = field(default_factory=dict)
    relationships: list[RelSpec] = field(default_factory=list)
    expected_error: str | None = None
    # PLAN-continuous additions
    suite: str = "transition"
    mode: str = "transition"
    expected_warnings: list[str] = field(default_factory=list)
    continuous: ContinuousDef | None = None
    ambient: list[ContinuousDef] = field(default_factory=list)


@dataclass(frozen=True)
class Variant:
    """Render/encode variant (S10/S11). ``fps`` is the render rate."""

    fps: float = 60.0
    pixel_ratio: int = 1
    encode: EncodeSpec = EncodeSpec()
    label: str | None = None


@dataclass(frozen=True)
class ScenarioEntry:
    name: str
    scenario: str
    build: Callable[[], ScenarioDef]
    variant: Variant = Variant()

    @property
    def filename(self) -> str:
        return f"{self.name}.{self.variant.encode.ext}"


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------


def text(
    node_id: str, s: str, x: float, y: float, px: float, color: str, weight: float = 1.0
) -> Node:
    return Node(
        id=node_id, shape="text", x=x, y=y, text=s, font_px=px, fill=c(color), weight=weight
    )


def centered_text(
    node_id: str, s: str, w: float, h: float, px: float, color: str, weight: float = 1.0
) -> Node:
    tw, asc, desc = text_size(s, px, weight)
    return text(
        node_id, s, round((w - tw) / 2.0, 1), round((h - asc - desc) / 2.0, 1), px, color, weight
    )


def chevron(node_id: str, x: float, y: float, color: str = "#6B7280") -> Node:
    return Node(
        id=node_id,
        shape="icon",
        x=x,
        y=y,
        w=10,
        h=6,
        fill=c(color),
        points=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
    )


def circle_points(n: int = 24) -> tuple[tuple[float, float], ...]:
    import math

    return tuple(
        (0.5 + 0.5 * math.cos(2 * math.pi * k / n), 0.5 + 0.5 * math.sin(2 * math.pi * k / n))
        for k in range(n)
    )


def page_chrome(header: bool = True) -> list[Node]:
    """Static page furniture: header bar, sidebar card with menu lines, footer."""
    nodes: list[Node] = []
    if header:
        nodes.append(header_node())
    side = Node(id="sidebar", x=32, y=88, w=200, h=400, fill=c("#FFFFFF"), radius=10)
    for i, label in enumerate(
        ["Overview", "Products", "Orders", "Customers", "Reports", "Settings"]
    ):
        side.children.append(
            Node(id=f"side_dot{i}", x=18, y=26 + i * 56, w=14, h=14, radius=4, fill=c("#D1D5DB"))
        )
        side.children.append(text(f"side_t{i}", label, 44, 26 + i * 56, 11, "#374151"))
    nodes.append(side)
    nodes.append(text("footer", "Synthetic scene - Mimic test suite", 32, 764, 10, "#9CA3AF"))
    return nodes


def header_node() -> Node:
    hdr = Node(id="header", x=0, y=0, w=CSS_W, h=56, fill=c("#FFFFFF"))
    hdr.children.append(text("logo", "Mimic Store", 24, 19, 16, "#111827", 1.5))
    for i, label in enumerate(["Products", "Pricing", "About"]):
        hdr.children.append(text(f"nav{i}", label, 960 + i * 100, 22, 11, "#4B5563"))
    return hdr


def product_card(prefix: str = "", *, shadow=(0.0, 4.0, 12.0, 0.0, 0.08)) -> Node:
    """320×400 product card at (480, 180) with image (clipped), title, body text, price."""
    card = Node(
        id=f"{prefix}card", x=480, y=180, w=320, h=400, fill=c("#FFFFFF"), radius=12,
        clip_children=True, shadow=shadow,
    )  # fmt: skip
    card.children += [
        Node(id=f"{prefix}image", shape="image", x=0, y=0, w=320, h=200, seed=7),
        text(f"{prefix}title", "Product title", 20, 222, 15, "#111111", 1.4),
        text(f"{prefix}body1", "Soft cotton crew neck tee", 20, 256, 11, "#6B7280"),
        text(f"{prefix}body2", "Available in 6 colors", 20, 278, 11, "#6B7280"),
        text(f"{prefix}price", "$49.00", 20, 340, 15, "#111111", 1.4),
    ]
    return card


def button(
    node_id: str, label: str, x: float, y: float, w: float, h: float, bg: str, fg: str, r: float = 8
) -> Node:
    b = Node(id=node_id, x=x, y=y, w=w, h=h, fill=c(bg), radius=r, pointer=True)
    b.children.append(centered_text(f"{node_id}_label", label, w, h, 13, fg, 1.3))
    return b


def scene_of(
    nodes: list[Node],
    cursor: CursorPath | None,
    duration_s: float,
    static=None,
    drivers: dict[str, Driver] | None = None,
    background: str = PAGE_BG,
) -> Scene:
    return Scene(
        width=CSS_W,
        height=CSS_H,
        background=c(background),
        static_nodes=page_chrome() if static is None else static,
        nodes=nodes,
        timeline=Timeline(),
        cursor=cursor,
        duration_s=duration_s,
        drivers=dict(drivers or {}),
    )


def add(scene: Scene, *tweens: Tween) -> None:
    scene.timeline.add(*tweens)
    scene.timeline.validate()


def ms(t: float) -> float:
    """Snap a time to whole milliseconds (truth times are integer ms)."""
    return round(t * 1000.0) / 1000.0


def hover_cursor(edge_x: float, y: float, end_x: float, t_enter: float = 1.0) -> CursorPath:
    """Cursor resting left of the target, moving right so its hotspot crosses ``edge_x`` at
    ``t_enter``, then resting inside at ``end_x``. Exit keys are appended by the scenario."""
    t0, t1 = t_enter - 0.5, t_enter + 0.3
    x0 = solve_start_for_crossing(end_x, edge_x, t0, t1, t_enter - 0.0004, "ease-in-out")
    return CursorPath(
        keys=[CursorKey(0.0, x0, y), CursorKey(t0, x0, y), CursorKey(t1, end_x, y, "ease-in-out")]
    )


def time_inside(scene: Scene, node_id: str, *, entering: bool, t_from: float, t_to: float) -> float:
    """First whole ms where the cursor hotspot is inside (``entering``) / outside the node box."""
    assert scene.cursor is not None

    def inside(t: float) -> bool:
        x, y = scene.cursor.position(t)  # type: ignore[union-attr]
        bx, by, bw, bh = scene.box(node_id, t)
        return bx <= x <= bx + bw and by <= y <= by + bh

    hit = first_time(lambda t: inside(t) == entering, t_from=t_from, t_to=t_to)
    if hit is None:
        raise RuntimeError(f"cursor never {'enters' if entering else 'leaves'} {node_id}")
    return ms(hit)


def click_path(
    start: tuple[float, float], target: tuple[float, float], arrive_s: float = 0.9
) -> CursorPath:
    return CursorPath(
        keys=[
            CursorKey(0.0, *start),
            CursorKey(0.3, *start),
            CursorKey(arrive_s, *target, "ease-in-out"),
        ],
        style="auto",
    )


# --------------------------------------------------------------------------------------------
# S1 / S2 / S9 — card hover
# --------------------------------------------------------------------------------------------


def card_hover(*, compound: bool = False, cursor_visible: bool = True) -> ScenarioDef:
    card = product_card()
    cur = hover_cursor(edge_x=480, y=420, end_x=600)
    cur.visible = cursor_visible
    scene = scene_of([card], cur, duration_s=10.0)
    t_enter = time_inside(scene, "card", entering=True, t_from=0.0, t_to=2.0)

    fwd: list[Tween] = [
        Tween("card", "translateY", "fwd", t_enter, 0.280, 0.0, -8.0, "ease-out"),
    ]
    if compound:
        fwd += [
            Tween("image", "scale", "fwd", ms(t_enter + 0.030), 0.320, 1.0, 1.06, "ease-out"),
            Tween(
                "card",
                "box-shadow",
                "fwd",
                t_enter,
                0.280,
                (0.0, 4.0, 12.0, 0.0, 0.08),
                (0.0, 12.0, 32.0, 0.0, 0.16),
                "ease-out",
            ),  # fmt: skip
            Tween("title", "color", "fwd", t_enter, 0.200, c("#111111"), c("#5252FF"), "ease"),
        ]
    add(scene, *fwd)

    # leave: rest inside until 2.55 s, then exit through the right edge
    cur.keys += [CursorKey(2.55, 600, 420), CursorKey(3.15, 980, 470, "ease-in-out")]
    t_leave = time_inside(scene, "card", entering=False, t_from=2.55, t_to=3.15)
    rev = [Tween("card", "translateY", "rev", t_leave, 0.220, -8.0, 0.0, "ease-in-out")]
    if compound:
        rev += [
            Tween("image", "scale", "rev", t_leave, 0.220, 1.06, 1.0, "ease-in-out"),
            Tween(
                "card",
                "box-shadow",
                "rev",
                t_leave,
                0.220,
                (0.0, 12.0, 32.0, 0.0, 0.16),
                (0.0, 4.0, 12.0, 0.0, 0.08),
                "ease-in-out",
            ),  # fmt: skip
            Tween(
                "title", "color", "rev", t_leave, 0.220, c("#5252FF"), c("#111111"), "ease-in-out"
            ),
        ]
    add(scene, *rev)
    scene.duration_s = round(t_leave + 0.220 + 1.1, 3)

    elements = [ElementMeta("card", "Product card", "card", "transform")]
    rels: list[RelSpec] = []
    if compound:
        elements += [
            ElementMeta("image", "Product image", "image", "transform"),
            ElementMeta("title", "Product title", "title", "photometric", text_like=True),
        ]
        rels = [
            RelSpec(
                "simultaneous",
                "fwd",
                (("card", "translateY"), ("card", "box-shadow"), ("title", "color")),
            ),
            RelSpec("delayed", "fwd", (("card", "translateY"), ("image", "scale")), offset_ms=30),
            RelSpec(
                "simultaneous",
                "rev",
                (
                    ("card", "translateY"),
                    ("image", "scale"),
                    ("card", "box-shadow"),
                    ("title", "color"),
                ),
            ),  # fmt: skip
        ]
    if cursor_visible:
        inter = dict(type="hover", accept_types=["hover"], trigger="pointer_enter",
                     reverse_trigger="pointer_leave")  # fmt: skip
    else:
        inter = dict(type="hover", accept_types=["hover", "unknown"], trigger="unknown",
                     reverse_trigger="unknown")  # fmt: skip
    inter |= dict(direction="forward_reverse", pattern="card", target_element_id="card")
    if compound:
        desc = (
            "Card hover: card translateY 0→-8 (280ms ease-out) + shadow 0 4 12 .08→0 12 32 .16, "
            "inner image scale 1→1.06 (320ms ease-out, +30ms, clipped), title color "
            "#111111→#5252FF (200ms ease); reverse 220ms ease-in-out."
        )
        checks = ["hierarchy", "relative scale", "delay", "color", "shadow presence"]
    else:
        desc = (
            "Card hover: translateY 0→-8, 280ms ease-out on pointer enter; hold; pointer leave → "
            "reverse 220ms ease-in-out."
        )
        checks = ["translate", "duration", "easing family", "fwd+rev", "hover"]
    if not cursor_visible:
        desc += " No cursor rendered."
        checks = ["trigger unknown/hover (low)", "measurements still correct"]
    return ScenarioDef(
        scenario="S2" if compound else ("S1" if cursor_visible else "S9"),
        description=desc,
        checks=checks,
        scene=scene,
        elements=elements,
        interaction=inter,
        trigger_node="card",
        segment_triggers={"fwd": t_enter, "rev": t_leave},
        relationships=rels,
    )


# --------------------------------------------------------------------------------------------
# S3 — button background color on hover
# --------------------------------------------------------------------------------------------


def button_bg_color() -> ScenarioDef:
    btn = button("button", "Button", 560, 378, 160, 44, "#3B82F6", "#FFFFFF")
    cur = hover_cursor(edge_x=560, y=400, end_x=640)
    cur.style = "auto"
    scene = scene_of([btn], cur, duration_s=10.0)
    t_enter = time_inside(scene, "button", entering=True, t_from=0.0, t_to=2.0)
    add(
        scene,
        Tween(
            "button",
            "background-color",
            "fwd",
            t_enter,
            0.150,
            c("#3B82F6"),
            c("#1D4ED8"),
            "linear",
        ),
    )
    cur.keys += [CursorKey(2.4, 640, 400), CursorKey(3.0, 780, 560, "ease-in-out")]
    t_leave = time_inside(scene, "button", entering=False, t_from=2.4, t_to=3.0)
    add(
        scene,
        Tween(
            "button",
            "background-color",
            "rev",
            t_leave,
            0.150,
            c("#1D4ED8"),
            c("#3B82F6"),
            "linear",
        ),
    )
    scene.duration_s = round(t_leave + 0.150 + 1.1, 3)
    return ScenarioDef(
        scenario="S3",
        description="Button hover: background-color #3B82F6→#1D4ED8, 150ms linear; "
        "reverse on leave.",
        checks=["color ΔE", "short duration"],
        scene=scene,
        elements=[ElementMeta("button", "Button", "button", "photometric")],
        interaction=dict(
            type="hover",
            accept_types=["hover"],
            trigger="pointer_enter",
            reverse_trigger="pointer_leave",
            direction="forward_reverse",
            pattern="button",
            target_element_id="button",
        ),  # fmt: skip
        trigger_node="button",
        segment_triggers={"fwd": t_enter, "rev": t_leave},
    )


# --------------------------------------------------------------------------------------------
# S4 — button press (round trip)
# --------------------------------------------------------------------------------------------


def button_press() -> ScenarioDef:
    btn = button("button", "Submit", 540, 372, 200, 56, "#111827", "#FFFFFF", r=10)
    cur = click_path((300, 620), (640, 400))
    t_down, t_up = 1.4, 1.5
    cur.clicks = [(t_down, t_up)]
    scene = scene_of([btn], cur, duration_s=10.0)
    add(
        scene,
        Tween("button", "scale", "rt_in", t_down, 0.090, 1.0, 0.96, "ease-out"),
        Tween("button", "scale", "rt_out", t_up, 0.120, 0.96, 1.0, "ease-out"),
    )
    scene.duration_s = round(t_up + 0.120 + 1.2, 3)
    return ScenarioDef(
        scenario="S4",
        description="Button press: scale 1→0.96 (90ms ease-out) on mousedown, →1 (120ms ease-out) "
        "on mouseup; cursor stationary.",
        checks=["round_trip", "press", "scale"],
        scene=scene,
        elements=[ElementMeta("button", "Submit button", "button", "transform")],
        interaction=dict(
            type="press",
            accept_types=["press"],
            trigger="click",
            reverse_trigger="click",
            direction="round_trip",
            pattern="button",
            target_element_id="button",
        ),  # fmt: skip
        trigger_node="button",
        segment_triggers={"rt_in": t_down, "rt_out": t_up},
    )


# --------------------------------------------------------------------------------------------
# S5 — dropdown open / close
# --------------------------------------------------------------------------------------------


def dropdown_open() -> ScenarioDef:
    trig = button("trigger", "Options", 520, 200, 168, 40, "#FFFFFF", "#374151")
    trig.children[0] = replace(trig.children[0], x=trig.children[0].x - 8)
    trig.children.append(chevron("trigger_chevron", 140, 17))
    menu = Node(
        id="menu", x=520, y=248, w=220, h=184, fill=c("#FFFFFF"), radius=8,
        shadow=(0.0, 8.0, 24.0, 0.0, 0.12), opacity=0.0, ty=-8.0,
    )  # fmt: skip
    for i, label in enumerate(["Edit", "Duplicate", "Archive", "Delete"]):
        color = "#DC2626" if label == "Delete" else "#374151"
        menu.children.append(text(f"menu_item{i}", label, 18, 18 + i * 42, 12, color))
    cur = click_path((360, 640), (596, 222))
    t_open, t_close = 1.2, 2.6
    cur.clicks = [(t_open - 0.06, t_open), (t_close - 0.06, t_close)]
    scene = scene_of([trig, menu], cur, duration_s=10.0)
    add(
        scene,
        Tween("menu", "opacity", "fwd", t_open, 0.200, 0.0, 1.0, "expoOut"),
        Tween("menu", "translateY", "fwd", t_open, 0.200, -8.0, 0.0, "expoOut"),
        Tween("menu", "opacity", "rev", t_close, 0.150, 1.0, 0.0, "ease-in"),
        Tween("menu", "translateY", "rev", t_close, 0.150, 0.0, -8.0, "ease-in"),
    )
    scene.duration_s = round(t_close + 0.150 + 1.1, 3)
    return ScenarioDef(
        scenario="S5",
        description="Dropdown: click trigger → menu opacity 0→1 + translateY -8→0, 200ms "
        "cubic-bezier(.16,1,.3,1); click again → close 150ms ease-in.",
        checks=["appear", "opacity", "dropdown", "click"],
        scene=scene,
        elements=[ElementMeta("menu", "Options menu", "dropdown_menu", "appear")],
        interaction=dict(
            type="dropdown",
            accept_types=["dropdown"],
            trigger="click",
            reverse_trigger="click",
            direction="forward_reverse",
            pattern="menu",
            target_element_id="menu",
        ),  # fmt: skip
        trigger_node="trigger",
        segment_triggers={"fwd": t_open, "rev": t_close},
        relationships=[
            RelSpec("simultaneous", "fwd", (("menu", "opacity"), ("menu", "translateY"))),
            RelSpec("simultaneous", "rev", (("menu", "opacity"), ("menu", "translateY"))),
        ],
    )


# --------------------------------------------------------------------------------------------
# S6 — modal open
# --------------------------------------------------------------------------------------------


def modal_open() -> ScenarioDef:
    opener = button("open_button", "Open dialog", 560, 380, 160, 44, "#2563EB", "#FFFFFF")
    backdrop = Node(id="backdrop", x=0, y=0, w=CSS_W, h=CSS_H, fill=c("#000000"), opacity=0.0)
    panel = Node(
        id="panel", x=400, y=250, w=480, h=300, fill=c("#FFFFFF"), radius=12,
        opacity=0.0, scale=0.95, shadow=(0.0, 20.0, 48.0, 0.0, 0.25),
    )  # fmt: skip
    panel.children += [
        text("panel_title", "Delete project?", 28, 28, 17, "#111827", 1.6),
        text("panel_body1", "This permanently removes the project and", 28, 76, 11, "#4B5563"),
        text("panel_body2", "all of its files. This cannot be undone.", 28, 98, 11, "#4B5563"),
        button("panel_cancel", "Cancel", 220, 232, 110, 40, "#E5E7EB", "#111827"),
        button("panel_delete", "Delete", 346, 232, 110, 40, "#DC2626", "#FFFFFF"),
    ]
    cur = click_path((340, 640), (640, 402))
    t_open = 1.2
    cur.clicks = [(t_open - 0.06, t_open)]
    scene = scene_of([opener, backdrop, panel], cur, duration_s=10.0)
    add(
        scene,
        Tween("backdrop", "opacity", "fwd", t_open, 0.250, 0.0, 0.5, "ease-out"),
        Tween("panel", "scale", "fwd", t_open, 0.250, 0.95, 1.0, "ease-out"),
        Tween("panel", "opacity", "fwd", t_open, 0.250, 0.0, 1.0, "ease-out"),
    )
    scene.duration_s = round(t_open + 0.250 + 1.3, 3)
    return ScenarioDef(
        scenario="S6",
        description="Modal: click → backdrop black opacity 0→0.5 (250ms ease-out); panel scale "
        ".95→1 + opacity 0→1 (250ms ease-out). Not closed (forward only).",
        checks=["backdrop", "modal", "scale on appear"],
        scene=scene,
        elements=[
            ElementMeta("backdrop", "Backdrop", "backdrop", "backdrop"),
            ElementMeta("panel", "Dialog panel", "modal_panel", "appear"),
        ],
        interaction=dict(
            type="modal",
            accept_types=["modal"],
            trigger="click",
            reverse_trigger="none",
            direction="forward",
            pattern="modal",
            target_element_id="panel",
        ),  # fmt: skip
        trigger_node="open_button",
        segment_triggers={"fwd": t_open},
        relationships=[
            RelSpec(
                "simultaneous",
                "fwd",
                (("panel", "scale"), ("panel", "opacity"), ("backdrop", "opacity")),
            ),  # fmt: skip
        ],
    )


# --------------------------------------------------------------------------------------------
# S7 — accordion expand
# --------------------------------------------------------------------------------------------


def accordion_expand() -> ScenarioDef:
    def header(node_id: str, label: str, y: float, pointer: bool = True) -> Node:
        h = Node(id=node_id, x=440, y=y, w=400, h=52, fill=c("#FFFFFF"), radius=8, pointer=pointer)
        h.children += [
            text(f"{node_id}_label", label, 20, 20, 12, "#111827", 1.3),
            chevron(f"{node_id}_chevron", 370, 23),
        ]
        return h

    h1 = header("header1", "Shipping", 160)
    panel = Node(
        id="panel", x=440, y=216, w=400, h=120, fill=c("#FFFFFF"), radius=8,
        clip_children=True, clip_h=0.0,
    )  # fmt: skip
    for i, line in enumerate(
        ["Orders ship within 2 business days.", "Free shipping over $50.", "Tracking by email."]
    ):
        panel.children.append(text(f"panel_line{i}", line, 20, 22 + i * 30, 11, "#4B5563"))
    h2 = header("header2", "Returns", 220)
    h3 = header("header3", "Warranty", 276)
    cur = click_path((320, 620), (560, 186))
    t_open = 1.2
    cur.clicks = [(t_open - 0.06, t_open)]
    scene = scene_of([h1, panel, h2, h3], cur, duration_s=10.0)
    add(
        scene,
        Tween("panel", "height", "fwd", t_open, 0.300, 0.0, 120.0, "ease-in-out"),
        Tween(
            "header2",
            "translateY",
            "fwd",
            t_open,
            0.300,
            0.0,
            120.0,
            "ease-in-out",
            ("caused_by_resize",),
        ),  # fmt: skip
        Tween(
            "header3",
            "translateY",
            "fwd",
            t_open,
            0.300,
            0.0,
            120.0,
            "ease-in-out",
            ("caused_by_resize",),
        ),  # fmt: skip
    )
    scene.duration_s = round(t_open + 0.300 + 1.2, 3)
    return ScenarioDef(
        scenario="S7",
        description="Accordion: click header → panel height 0→120 (300ms ease-in-out); the two "
        "rows below shift +120 (same timing). Forward only.",
        checks=["resize pattern", "expand_collapse"],
        scene=scene,
        elements=[
            ElementMeta("panel", "Accordion panel", "accordion_panel", "resize"),
            ElementMeta("header2", "Returns row", "button", "transform"),
            ElementMeta("header3", "Warranty row", "button", "transform"),
        ],
        interaction=dict(
            type="expand_collapse",
            accept_types=["expand_collapse"],
            trigger="click",
            reverse_trigger="none",
            direction="forward",
            pattern="accordion",
            target_element_id="panel",
        ),  # fmt: skip
        trigger_node="header1",
        segment_triggers={"fwd": t_open},
        relationships=[
            RelSpec(
                "simultaneous",
                "fwd",
                (("panel", "height"), ("header2", "translateY"), ("header3", "translateY")),
            ),  # fmt: skip
        ],
    )


# --------------------------------------------------------------------------------------------
# S8 — staggered list
# --------------------------------------------------------------------------------------------


def stagger_list() -> ScenarioDef:
    btn = button("show_button", "Show items", 520, 160, 168, 40, "#111827", "#FFFFFF")
    items: list[Node] = []
    for i in range(4):
        it = Node(
            id=f"item{i + 1}", x=520, y=216 + i * 56, w=280, h=48, fill=c("#FFFFFF"), radius=8,
            opacity=0.0, ty=8.0,
        )  # fmt: skip
        it.children += [
            Node(id=f"item{i + 1}_avatar", shape="icon", x=12, y=12, w=24, h=24,
                 fill=c(["#93C5FD", "#FCA5A5", "#86EFAC", "#FDE68A"][i]), points=circle_points()),
            text(f"item{i + 1}_label", f"Notification {i + 1}", 48, 18, 12, "#111827"),
        ]  # fmt: skip
        items.append(it)
    cur = click_path((340, 620), (600, 180))
    t_click = 1.2
    cur.clicks = [(t_click - 0.06, t_click)]
    scene = scene_of([btn, *items], cur, duration_s=10.0)
    for i in range(4):
        t0 = ms(t_click + 0.060 * i)
        add(
            scene,
            Tween(f"item{i + 1}", "opacity", "fwd", t0, 0.200, 0.0, 1.0, "ease-out"),
            Tween(f"item{i + 1}", "translateY", "fwd", t0, 0.200, 8.0, 0.0, "ease-out"),
        )
    scene.duration_s = round(t_click + 0.180 + 0.200 + 1.2, 3)
    ids = [f"item{i + 1}" for i in range(4)]
    return ScenarioDef(
        scenario="S8",
        description="Stagger: click → 4 list items fade in + translateY 8→0, 200ms ease-out each, "
        "60ms stagger. Forward only.",
        checks=["stagger interval"],
        scene=scene,
        elements=[
            ElementMeta(i, f"Notification {k + 1}", "list_item", "appear")
            for k, i in enumerate(ids)
        ],
        interaction=dict(
            type="dropdown",
            accept_types=["dropdown", "click"],
            trigger="click",
            reverse_trigger="none",
            direction="forward",
            pattern="menu",
            target_element_id="item1",
        ),  # fmt: skip
        trigger_node="show_button",
        segment_triggers={"fwd": t_click},
        relationships=[
            RelSpec("stagger", "fwd", tuple((i, "opacity") for i in ids), interval_ms=60),
            RelSpec("stagger", "fwd", tuple((i, "translateY") for i in ids), interval_ms=60),
        ],
    )


# --------------------------------------------------------------------------------------------
# S12 — negatives
# --------------------------------------------------------------------------------------------


def negative_static() -> ScenarioDef:
    card = product_card()
    cur = CursorPath(
        keys=[
            CursorKey(0.0, 300, 640),
            CursorKey(0.5, 300, 640),
            CursorKey(1.3, 900, 300, "ease-in-out"),
            CursorKey(1.9, 900, 300),
            CursorKey(2.6, 380, 700, "ease-in-out"),
        ]
    )
    scene = scene_of([card], cur, duration_s=3.6)
    return ScenarioDef(
        scenario="S12",
        description="Static page; only the cursor moves (never over the card). No UI motion.",
        checks=["no_motion_detected"],
        scene=scene,
        expected_error="no_motion_detected",
    )


def negative_scroll() -> ScenarioDef:
    page = Node(id="page", shape="group", x=0, y=56, w=CSS_W, h=1600)
    side = page_chrome(header=False)[0]
    side.y -= 56
    page.children.append(side)
    for row in range(5):
        for col in range(3):
            k = row * 3 + col
            cd = Node(
                id=f"pc{k}", x=272 + col * 324, y=24 + row * 300, w=300, h=280,
                fill=c("#FFFFFF"), radius=10,
            )  # fmt: skip
            cd.children += [
                Node(id=f"pc{k}_img", shape="image", x=12, y=12, w=276, h=160, seed=100 + k),
                text(f"pc{k}_t", f"Item {k + 1}", 16, 190, 13, "#111827", 1.3),
                text(f"pc{k}_d", "Short description line", 16, 220, 10, "#6B7280"),
            ]
            page.children.append(cd)
    cur = CursorPath(keys=[CursorKey(0.0, 700, 460), CursorKey(0.4, 700, 460),
                           CursorKey(0.9, 760, 430, "ease-in-out")])  # fmt: skip
    scene = scene_of([page, header_node()], cur, duration_s=10.0, static=[])
    add(scene, Tween("page", "translateY", "fwd", 1.2, 0.600, 0.0, -360.0, "ease-out"))
    scene.duration_s = round(1.2 + 0.6 + 1.2, 3)
    return ScenarioDef(
        scenario="S12",
        description="Whole page content scrolls up 360px (600ms ease-out) under a fixed header.",
        checks=["unsupported_motion"],
        scene=scene,
        expected_error="unsupported_motion",
    )


# --------------------------------------------------------------------------------------------
# PLAN-continuous §8.2 — scroller strips (C1–C12) and the ambient negative (N1)
# --------------------------------------------------------------------------------------------

STRIP_BG = "#E5E7EB"
CARD_BG = "#FFFFFF"
CARD_TITLE = "#111827"
CARD_SUB = "#6B7280"
IDLE_CURSOR = (1120.0, 660.0)


@dataclass(frozen=True)
class Strip:
    """A scroller: viewport box (``overflow: hidden``) holding a track with **two identical
    copies** of ``cards`` distinct cards (seamless loop). ``lead`` = offset of the first card
    along the axis inside the viewport, ``cross`` = offset across it. Geometry in CSS px."""

    prefix: str
    axis: str  # "x" | "y"
    x: float
    y: float
    w: float
    h: float
    cards: int = 8
    card_w: float = 200.0
    card_h: float = 160.0
    gap: float = 16.0
    lead: float = 8.0
    cross: float = 20.0
    radius: float = 12.0
    card_radius: float = 10.0
    layout: str = "card"  # "card" (image + title + subtitle) | "chip" (thumb + title)
    title_px: float = 13.0
    seed: int = 300
    # theme (P2b; defaults = the light §8.2 strip, rendered unchanged)
    track_bg: str = STRIP_BG
    card_bg: str = CARD_BG
    title_color: str = CARD_TITLE
    sub_color: str = CARD_SUB
    #: 1 px card border colour (drawn as the card box with the fill inset by 1 px); None = none
    border: str | None = None
    #: every ``dark_every``-th card (k % n == 1) shows a dark photo (texture × ``dark_tone``)
    dark_every: int = 0
    dark_tone: float = 0.3
    #: magnification at the viewport edges (:class:`kinematics.EdgeZoom`); 1.0 = rigid strip
    zoom_edge: float = 1.0
    #: page background behind a strip shown without the light page chrome; None = §8.2 page
    page_bg: str | None = None
    #: flat placeholder images (one plain fill per card from ``FLAT_FILLS``) instead of the
    #: procedural photo texture: low-texture content (acceptance run, PLAN-continuous P2e)
    flat: bool = False

    @property
    def viewport_id(self) -> str:
        return f"{self.prefix}scroller"

    @property
    def track_id(self) -> str:
        return f"{self.prefix}track"

    @property
    def pitch(self) -> float:
        return (self.card_w if self.axis == "x" else self.card_h) + self.gap

    @property
    def period(self) -> float:
        return self.cards * self.pitch

    @property
    def view_len(self) -> float:
        return self.w if self.axis == "x" else self.h

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def _card(self, copy: int, k: int) -> Node:
        along = self.lead + copy * self.period + k * self.pitch
        cx, cy = (along, self.cross) if self.axis == "x" else (self.cross, along)
        cid = f"{self.prefix}c{copy}_{k}"
        if self.border is None:
            card = Node(id=cid, x=cx, y=cy, w=self.card_w, h=self.card_h, fill=c(self.card_bg),
                        radius=self.card_radius)  # fmt: skip
        else:
            card = Node(id=cid, x=cx, y=cy, w=self.card_w, h=self.card_h, fill=c(self.border),
                        radius=self.card_radius)  # fmt: skip
            card.children.append(
                Node(
                    id=f"{cid}_fill",
                    x=1,
                    y=1,
                    w=self.card_w - 2,
                    h=self.card_h - 2,
                    fill=c(self.card_bg),
                    radius=max(self.card_radius - 1, 0),
                )  # fmt: skip
            )
        tone = self.dark_tone if self.dark_every and k % self.dark_every == 1 else 1.0
        title = f"Item {k + 1}"
        if self.layout == "card":
            img_h = round(self.card_h * 0.6)
            img = (
                Node(id=f"{cid}_img", x=8, y=8, w=self.card_w - 16, h=img_h, radius=6,
                     fill=c(FLAT_FILLS[k % len(FLAT_FILLS)]))
                if self.flat
                else Node(id=f"{cid}_img", shape="image", x=8, y=8, w=self.card_w - 16,
                          h=img_h, radius=6, seed=self.seed + k, tone=tone)
            )  # fmt: skip
            card.children += [
                img,
                text(f"{cid}_t", title, 12, 8 + img_h + 12, self.title_px, self.title_color, 1.3),
                text(f"{cid}_s", "Placeholder", 12, 8 + img_h + 34, 10, self.sub_color),
            ]
        else:
            thumb = self.card_h - 16
            card.children += [
                Node(
                    id=f"{cid}_img",
                    shape="image",
                    x=8,
                    y=8,
                    w=thumb,
                    h=thumb,
                    radius=6,
                    seed=self.seed + k,
                    tone=tone,
                ),  # fmt: skip
                text(
                    f"{cid}_t",
                    title,
                    thumb + 16,
                    round((self.card_h - self.title_px) / 2),
                    self.title_px,
                    self.title_color,
                    1.3,
                ),  # fmt: skip
            ]
        return card

    def node(self) -> Node:
        track_len = 2 * self.period + self.lead
        tw, th = (track_len, self.h) if self.axis == "x" else (self.w, track_len)
        track = Node(id=self.track_id, shape="group", x=0, y=0, w=tw, h=th)
        track.children = [self._card(cp, k) for cp in (0, 1) for k in range(self.cards)]
        return Node(id=self.viewport_id, x=self.x, y=self.y, w=self.w, h=self.h,
                    fill=c(self.track_bg), radius=self.radius, clip_children=True,
                    children=[track])  # fmt: skip

    def driver(self, profile: VelocityProfile) -> Driver:
        if self.view_len > self.period:
            raise ValueError("strip copy must be at least as long as the viewport")
        return strip_driver(profile, self.axis, self.period)  # type: ignore[arg-type]

    @property
    def zoom(self) -> EdgeZoom | None:
        return EdgeZoom(self.zoom_edge, self.view_len / 2.0) if self.zoom_edge != 1.0 else None

    def drivers(self, profile: VelocityProfile) -> dict[str, Driver]:
        """The track driver, plus one ``(tx, ty, scale)`` driver per card when zoomed."""
        track = self.driver(profile)
        out: dict[str, Driver] = {self.track_id: track}
        zoom = self.zoom
        if zoom is None:
            return out
        card_len = self.card_w if self.axis == "x" else self.card_h
        for cp in (0, 1):
            for k in range(self.cards):
                along = self.lead + cp * self.period + k * self.pitch
                centre = along + card_len / 2.0 - self.view_len / 2.0
                out[f"{self.prefix}c{cp}_{k}"] = card_zoom_driver(
                    track,
                    zoom,
                    self.axis,
                    centre,  # type: ignore[arg-type]
                )
        return out

    def centre_geometry(self) -> tuple[float, float]:
        """``(gap, pitch)`` as drawn at the viewport centre: two cards placed symmetrically
        about it (scale ≈ 1). Equals ``(gap, pitch)`` for a rigid strip."""
        zoom = self.zoom
        if zoom is None:
            return self.gap, self.pitch
        card_len = self.card_w if self.axis == "x" else self.card_h
        u = self.pitch / 2.0
        gap = 2.0 * (zoom.screen(u) - zoom.scale(u) * card_len / 2.0)
        return gap, card_len + gap


#: Plain placeholder image fills of ``Strip(flat=True)`` (distinct per card, low contrast).
FLAT_FILLS = ("#2E3440", "#3B4252", "#434C5E", "#4C566A", "#3A3F4B", "#2B303B", "#454B57",
              "#363C48")  # fmt: skip


@dataclass
class ContinuousDef:
    """A scroller and its velocity profile, plus behaviour facts the profile can't express."""

    strip: Strip
    profile: VelocityProfile
    pause_on: str | None = None
    pause_on_accept: tuple[str, ...] = ()
    resume_delay_after_leave_ms: int | None = None


def idle_cursor(x: float = IDLE_CURSOR[0], y: float = IDLE_CURSOR[1]) -> CursorPath:
    """A visible cursor resting outside the scroller for the whole recording."""
    return CursorPath(keys=[CursorKey(0.0, x, y)])


def first_ms_inside(
    cur: CursorPath, box: tuple[float, float, float, float], *, entering: bool, t_from: float,
    t_to: float,
) -> float:  # fmt: skip
    """First whole ms in ``[t_from, t_to]`` where the hotspot is inside (``entering``) /
    outside a static box (edges count as inside, like ``time_inside``)."""
    bx, by, bw, bh = box

    def inside(t: float) -> bool:
        x, y = cur.position(t)
        return bx <= x <= bx + bw and by <= y <= by + bh

    hit = first_time(lambda t: inside(t) == entering, t_from=t_from, t_to=t_to)
    if hit is None:
        raise RuntimeError(f"cursor never {'enters' if entering else 'leaves'} {box}")
    return ms(hit)


def drag_cursor(
    profile: VelocityProfile,
    axis: str,
    start: tuple[float, float],
    grabs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> DragCursor:
    """Cursor for a drag scenario: for each drag phase ``(press point, park point)``: arrive at
    the press point 300 ms before the press (600 ms leg), follow the content 1:1 until release
    (mousedown/mouseup at press/release), rest 300 ms, then move to the park point (600 ms)."""
    windows = profile.drag_windows()
    if len(windows) != len(grabs):
        raise ValueError("one (press, park) pair per drag phase")
    keys = [CursorKey(0.0, *start)]
    clicks: list[tuple[float, float]] = []
    pos = start
    for (press, release), (p, park) in zip(windows, grabs, strict=True):
        d = profile.position(release) - profile.position(press)
        rel = (p[0] + d, p[1]) if axis == "x" else (p[0], p[1] + d)
        if keys[-1].t_s > press - 0.9 + 1e-9:
            raise ValueError("drags too close together for the cursor path")
        keys += [
            CursorKey(round(press - 0.9, 3), *pos),
            CursorKey(round(press - 0.3, 3), *p, "ease-in-out"),
            CursorKey(press, *p),
            CursorKey(release, *rel, "linear"),  # overridden by DragCursor inside the window
            CursorKey(round(release + 0.3, 3), *rel),
            CursorKey(round(release + 0.9, 3), *park, "ease-in-out"),
        ]
        clicks.append((press, release))
        pos = park
    return DragCursor(keys=keys, clicks=clicks, profile=profile, axis=axis)  # type: ignore[arg-type]


def _strip_scene(strip: Strip, profile: VelocityProfile, cursor: CursorPath | None) -> Scene:
    if strip.page_bg is None:
        return scene_of(
            [strip.node()],
            cursor,
            duration_s=profile.end_ms / 1000.0,
            drivers=strip.drivers(profile),
        )
    return scene_of(
        [strip.node()],
        cursor,
        duration_s=profile.end_ms / 1000.0,
        static=[],
        drivers=strip.drivers(profile),
        background=strip.page_bg,
    )


def _scroller_meta() -> list[ElementMeta]:
    return [ElementMeta("scroller", "Scroller", "scroller", "scroller")]


def _continuous_interaction(kind: str) -> dict:
    drag = kind == "drag"
    return dict(
        type="drag" if drag else "continuous",
        accept_types=["drag"] if drag else ["continuous"],
        trigger="drag" if drag else "autoplay",
        reverse_trigger="none",
        direction="continuous",
        pattern="carousel" if drag else "marquee",
        target_element_id="scroller",
    )


def _continuous_def(
    scenario: str, description: str, checks: list[str], strip: Strip, profile: VelocityProfile,
    cursor: CursorPath | None, *, kind: str, **cdef_kw,
) -> ScenarioDef:  # fmt: skip
    scene = _strip_scene(strip, profile, cursor)
    return ScenarioDef(
        scenario=scenario,
        description=description,
        checks=checks,
        scene=scene,
        elements=_scroller_meta(),
        interaction=_continuous_interaction(kind),
        suite="continuous",
        mode="continuous",
        continuous=ContinuousDef(strip=strip, profile=profile, **cdef_kw),
    )


def _main_strip(**kw) -> Strip:
    """§8.2 strip: viewport 760×200 at (260, 300); cards 200×160, gap 16 → pitch 216."""
    return Strip("", "x", 260, 300, 760, 200, **kw)


# ---- C1 / C2 — autoplay-only marquees ------------------------------------------------------


def marquee_autoplay() -> ScenarioDef:
    strip = _main_strip()
    profile = VelocityProfile([Autoplay(-60.0, 8.0)])
    return _continuous_def(
        "C1",
        "Marquee: 8 cards (pitch 216, period 1728) move left at 60 px/s for 8 s; idle cursor "
        "outside. Travel 480 px < one period → loop length not observable.",
        ["speed", "direction", "mode continuous", "loop not observed"],
        strip, profile, idle_cursor(), kind="autoplay",
    )  # fmt: skip


def marquee_fast_loop() -> ScenarioDef:
    strip = _main_strip(cards=4)
    profile = VelocityProfile([Autoplay(-240.0, 8.0)])
    return _continuous_def(
        "C2",
        "Marquee: 4-card loop (period 864) moving left at 240 px/s for 8 s (1920 px travel).",
        ["loop period observed", "loop duration"],
        strip, profile, idle_cursor(), kind="autoplay",
    )  # fmt: skip


# ---- C3 / C10 — hover pause ----------------------------------------------------------------


def marquee_hover_pause(*, vertical: bool = False) -> ScenarioDef:
    if vertical:
        strip = Strip("", "y", 520, 140, 240, 520)  # cards 200×160 stacked, pitch 176
        v = -30.0
        cur = hover_cursor(edge_x=520, y=400, end_x=640, t_enter=2.0)
        cur.keys += [CursorKey(3.7, 640, 400), CursorKey(4.3, 880, 400, "ease-in-out")]
    else:
        strip = _main_strip()
        v = 40.0
        cur = hover_cursor(edge_x=260, y=400, end_x=360, t_enter=2.0)
        cur.keys += [CursorKey(3.7, 360, 400), CursorKey(4.3, 360, 600, "ease-in-out")]
    t_enter = first_ms_inside(cur, strip.box, entering=True, t_from=0.0, t_to=3.0)
    t_leave = first_ms_inside(cur, strip.box, entering=False, t_from=3.7, t_to=4.3)
    decel, delay, ramp, total = 0.4, 0.3, 0.6, 8.0
    profile = VelocityProfile([
        Autoplay(v, t_enter),
        Ramp(0.0, decel, "ease-out", "decelerate"),
        Hold(round(t_leave + delay - (t_enter + decel), 3)),
        Ramp(v, ramp, "ease-in", "resume"),
        Autoplay(v, round(total - (t_leave + delay + ramp), 3)),
    ])  # fmt: skip
    if vertical:
        return _continuous_def(
            "C10",
            "Vertical ticker: 8 cards (pitch 176) move up at 30 px/s; pointer enters at 2.0 s → "
            "decelerate 400 ms ease-out to 0; leaves at 4.0 s → after 300 ms resume over 600 ms "
            "ease-in.",
            ["axis y", "pause on hover", "decel/ramp", "resume delay"],
            strip, profile, cur, kind="autoplay", pause_on="hover", pause_on_accept=("hover",),
            resume_delay_after_leave_ms=300,
        )  # fmt: skip
    return _continuous_def(
        "C3",
        "Marquee 40 px/s right; pointer enters at 2.0 s → decelerate 400 ms ease-out to 0; leaves "
        "at 4.0 s → after 300 ms resume over 600 ms ease-in.",
        ["pause on hover", "decel/ramp", "resume delay"],
        strip, profile, cur, kind="autoplay", pause_on="hover", pause_on_accept=("hover",),
        resume_delay_after_leave_ms=300,
    )  # fmt: skip


# ---- C4–C9 — drag carousels ----------------------------------------------------------------

#: Cursor rest / park points for the drag carousels (outside the 260..1020 × 300..500 viewport).
DRAG_START = (560.0, 690.0)


def carousel_drag_inertia(*, cursor_visible: bool = True) -> ScenarioDef:
    strip = _main_strip()
    v = 40.0
    profile = VelocityProfile([
        Autoplay(v, 2.0),
        Drag(-400.0, 0.45, -1100.0),  # press at 2.0 s stops autoplay; release v0 = -1100
        Inertia(0.200),
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 1.0),
        Drag(300.0, 0.40, 1000.0),
        Inertia(0.200),
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 1.289),
    ])  # fmt: skip
    cur = drag_cursor(
        profile,
        "x",
        DRAG_START,
        [((700.0, 400.0), (420.0, 650.0)), ((500.0, 400.0), (900.0, 650.0))],
    )
    cur.visible = cursor_visible
    desc = (
        "Carousel: autoplay 40 px/s right; press at 2.0 s stops it; drag left 400 px in 450 ms "
        "(release -1100 px/s) → inertia τ 200 ms → rest; resume after 1000 ms (750 ms ease-in-out "
        "ramp); second drag right 300 px / 400 ms (release +1000 px/s), τ 200 ms, same resume."
    )
    if cursor_visible:
        return _continuous_def(
            "C4", desc, ["phase sequence", "tau", "v0", "resume", "pointer ratio"],
            strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("press",),
        )  # fmt: skip
    return _continuous_def(
        "C5", desc + " No cursor rendered (the real-recording analogue).",
        ["same values as C4", "pause trigger unknown"],
        strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("unknown",),
    )  # fmt: skip


def carousel_drag_snap() -> ScenarioDef:
    strip = _main_strip()
    v = 40.0
    snap = SnapTween(0.300, "ease-out", step=strip.pitch, min_px=strip.pitch / 4)
    profile = VelocityProfile([
        Autoplay(v, 2.0),
        Drag(-380.0, 0.45, -1000.0),
        snap,
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 1.5),
        Drag(350.0, 0.40, 1100.0),
        snap,
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 1.55),
    ])  # fmt: skip
    cur = drag_cursor(
        profile,
        "x",
        DRAG_START,
        [((700.0, 400.0), (420.0, 650.0)), ((450.0, 400.0), (900.0, 650.0))],
    )
    return _continuous_def(
        "C8",
        "Carousel with snap: autoplay 40 px/s right; 2 drags (left 380 px, right 350 px); on "
        "release the track tweens to the next card in the drag direction (grid 216 px) over "
        "300 ms ease-out; rest 1000 ms; resume 750 ms ease-in-out.",
        ["snap grid", "snap step", "snap duration"],
        strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("press",),
    )  # fmt: skip


def carousel_fling_fast() -> ScenarioDef:
    strip = _main_strip()
    v = 40.0
    profile = VelocityProfile([
        Autoplay(v, 2.0),
        Drag(-700.0, 0.40, -2500.0),
        Inertia(0.250),
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 1.0),
        Drag(600.0, 0.40, 2200.0),
        Inertia(0.250),
        Hold(1.0),
        Ramp(v, 0.75, "ease-in-out", "resume"),
        Autoplay(v, 0.772),
    ])  # fmt: skip
    cur = drag_cursor(
        profile,
        "x",
        DRAG_START,
        [((980.0, 400.0), (420.0, 650.0)), ((320.0, 400.0), (900.0, 650.0))],
    )
    return _continuous_def(
        "C9",
        "Fast flings: autoplay 40 px/s right; drag left 700 px / 400 ms (release -2500 px/s), "
        "τ 250 ms; drag right 600 px / 400 ms (release +2200 px/s), τ 250 ms; rendered at 30 fps "
        "and encoded at CRF 28 (≈83 px per frame at peak).",
        ["large-shift tracking", "tracking_degraded allowed", "relaxed targets"],
        strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("press",),
    )  # fmt: skip


# ---- C11 — transition with an ambient marquee ----------------------------------------------


#: P2b dark theme (the motivating recording): page = track ≈ #020202, low-contrast cards.
DARK_PAGE = "#020202"
DARK_CARD = "#141418"
DARK_BORDER = "#26262C"
DARK_TITLE = "#E5E7EB"
DARK_SUB = "#9CA3AF"
#: Card scale at the viewport edges measured on the motivating recording (164 → ≈192 px).
REAL_ZOOM_EDGE = 1.165


def _inertia_to_autoplay_s(tau: float, v0: float, v_auto: float) -> float:
    """Inertia → autoplay duration: until the decay is within the pipeline's autoplay band
    (``max(3, 0.12 |v_auto|)`` px/s, PLAN-continuous §4.5 step 2) of ``v_auto``."""
    tol = max(3.0, 0.12 * abs(v_auto))
    return round(tau * float(np.log(abs(v0 - v_auto) / tol)), 3)


def carousel_dark_edge_zoom(*, zoom_edge: float = REAL_ZOOM_EDGE) -> ScenarioDef:
    """P2b regression for the motivating recording's card layout (C13).

    Dark page (= track) #020202, 8 equal low-contrast cards #141418 (1 px #26262C border, inset
    photo, every third one dark) 164×150 with 4 px gaps, scaled by their distance from the
    viewport centre (×1 at the centre, ×``zoom_edge`` at the edges, gaps scale along), no
    cursor, captured on a 75 Hz grid (MOV, like C12). Motion: autoplay +35 px/s; press stops it;
    fling left → momentum blends back into autoplay; fling right → rest 200 ms → resume.
    """
    strip = Strip(
        "", "x", 260, 300, 760, 200, card_w=164.0, card_h=150.0, gap=4.0, cross=25.0,
        card_radius=10.0, track_bg=DARK_PAGE, card_bg=DARK_CARD, title_color=DARK_TITLE,
        sub_color=DARK_SUB, border=DARK_BORDER, dark_every=3, zoom_edge=zoom_edge,
        page_bg=DARK_PAGE, seed=500,
    )  # fmt: skip
    v = 35.0
    tau1 = 0.26
    profile = VelocityProfile([
        Autoplay(v, 2.0),
        Drag(-450.0, 0.40, -1400.0),  # press at 2.0 s stops autoplay; release v0 = -1400
        Inertia(tau1, v_inf=v, dur=_inertia_to_autoplay_s(tau1, -1400.0, v)),
        Autoplay(v, 1.2),
        Drag(320.0, 0.40, 1100.0),
        Inertia(0.25),  # decays to rest (10 px/s cut-off)
        Hold(0.2),
        Ramp(v, 0.815, "material-decelerate", "resume"),
        Autoplay(v, 1.493),  # ends at 9.200 s = 552 frames at 60 fps
    ])  # fmt: skip
    cur = drag_cursor(
        profile,
        "x",
        DRAG_START,
        [((760.0, 400.0), (420.0, 650.0)), ((450.0, 400.0), (900.0, 650.0))],
    )
    cur.visible = False
    zoom_txt = (
        f"cards scale with their distance from the viewport centre (×1 → ×{zoom_edge:g} at the "
        "edges); "
        if zoom_edge != 1.0
        else ""
    )
    return _continuous_def(
        "C13",
        "Dark carousel (page = track #020202, low-contrast #141418 cards with a 1 px border and "
        "inset photos, every third one dark; 164×150, gap 4 → pitch 168 at the centre); "
        + zoom_txt
        + "autoplay 35 px/s right; press at 2.0 s stops it; fling left 450 px (release "
        "-1400 px/s) → momentum τ 260 ms blends into autoplay; fling right 320 px (release "
        "+1100 px/s) → τ 250 ms to rest → 200 ms → resume 815 ms material-decelerate. No "
        "cursor; 75 Hz capture grid, MOV.",
        ["card size / pitch / gap reported", "phase sequence", "tau", "v0", "resume"],
        strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("unknown",),
    )  # fmt: skip


def carousel_flat_held_stop() -> ScenarioDef:
    """Acceptance-run regression (C14): the motivating recording's flow on low-texture
    placeholder cards, as an independent replica of it renders them.

    Dark page, flat placeholder cards (one plain fill per card + a title, no photo texture),
    so the pass-1 activity grid only sees the card edges / titles and splits the row into
    pieces (``regime.merge_row_pieces``). Rigid cards: with flat cards that also scale, the
    tracker under-reads the mean on-screen speed by ≈4 % (it follows the few, mostly central
    features), beyond §8.4's 2 % — a documented limitation, not part of this regression.
    Motion: autoplay +35 px/s; fling left → momentum blends into autoplay; fling right →
    blends into autoplay; a drag that slows to a stop **while pressed** (no glide, no momentum)
    → held 200 ms → resume 815 ms material-decelerate. No cursor, 60 fps.
    """
    strip = Strip(
        "", "x", 260, 300, 760, 200, card_w=164.0, card_h=150.0, gap=4.0, cross=25.0,
        card_radius=10.0, track_bg=DARK_PAGE, card_bg=DARK_CARD, title_color=DARK_TITLE,
        sub_color=DARK_SUB, page_bg=DARK_PAGE, flat=True, seed=700,
    )  # fmt: skip
    v = 35.0
    tau = 0.26
    profile = VelocityProfile([
        Autoplay(v, 2.0),
        Drag(-450.0, 0.40, -1400.0),
        Inertia(tau, v_inf=v, dur=_inertia_to_autoplay_s(tau, -1400.0, v)),
        Autoplay(v, 1.0),
        Drag(320.0, 0.40, 1100.0),
        Inertia(tau, v_inf=v, dur=_inertia_to_autoplay_s(tau, 1100.0, v)),
        Autoplay(v, 1.0),
        HeldDrag(-600.0, 0.70),  # the pointer slows to a stop before letting go
        Hold(0.2),
        Ramp(v, 0.815, "material-decelerate", "resume"),
        Autoplay(v, 1.196),  # ends at 10.667 s = 640 frames at 60 fps
    ])  # fmt: skip
    cur = drag_cursor(
        profile,
        "x",
        DRAG_START,
        [((760.0, 400.0), (420.0, 650.0)), ((450.0, 400.0), (900.0, 650.0)),
         ((800.0, 400.0), (420.0, 650.0))],
    )  # fmt: skip
    cur.visible = False
    return _continuous_def(
        "C14",
        "Dark carousel with flat placeholder cards (plain fill + title, no photo texture; "
        "164×150, gap 4, rigid); autoplay 35 px/s right; fling "
        "left 450 px (release -1400 px/s) and fling right 320 px (release +1100 px/s), each "
        "momentum τ 260 ms blending into autoplay; a 600 px drag left that slows to a stop "
        "while pressed → held 200 ms → resume 815 ms material-decelerate. No cursor, 60 fps.",
        ["region / card on flat cards", "phase sequence", "drag ends at rest", "resume"],
        strip, profile, cur, kind="drag", pause_on="press", pause_on_accept=("unknown",),
    )  # fmt: skip


def hover_with_ambient_marquee() -> ScenarioDef:
    """S1 card hover, plus an autoplay-only chip ticker below it moving from t=0."""
    base = card_hover()
    old = base.scene
    ticker = Strip("amb_", "x", 260, 612, 760, 80, card_w=140, card_h=56, gap=16, lead=8,
                   cross=12, layout="chip", seed=500)  # fmt: skip
    profile = VelocityProfile([Autoplay(-50.0, old.duration_s)])
    scene = Scene(
        width=old.width, height=old.height, background=old.background,
        static_nodes=old.static_nodes, nodes=[*old.nodes, ticker.node()],
        timeline=old.timeline, cursor=old.cursor, duration_s=old.duration_s,
        drivers={ticker.track_id: ticker.driver(profile)},
    )  # fmt: skip
    return replace(
        base,
        scenario="C11",
        description=base.description + " Plus an autoplay-only ticker (chips 140×56, pitch 156) "
        "at y 612 moving left at 50 px/s from t=0 (ambient; must be masked).",
        checks=["transition mode", "S1 §11.4 thresholds", "warning ambient_motion_masked"],
        scene=scene,
        suite="continuous",
        mode="transition",
        expected_warnings=["ambient_motion_masked"],
        ambient=[ContinuousDef(strip=ticker, profile=profile)],
    )


# ---- N1 — ambient motion that is not a scroller --------------------------------------------


def negative_ambient_pulse() -> ScenarioDef:
    box = Node(id="pulse", x=560, y=320, w=160, h=160, fill=c("#6366F1"), radius=16)
    scene = scene_of([box], idle_cursor(), duration_s=6.0)
    lo, hi, step = 0.35, 1.0, 0.6
    for k in range(10):
        a, b = (hi, lo) if k % 2 == 0 else (lo, hi)
        add(scene, Tween("pulse", "opacity", "fwd", round(k * step, 3), step, a, b, "ease-in-out"))
    return ScenarioDef(
        scenario="N1",
        description="A box pulses opacity 1 ↔ 0.35 (600 ms ease-in-out legs) from t=0 for 6 s; "
        "nothing else moves. Ambient motion that is not a scroller.",
        checks=["continuous_motion_unsupported"],
        scene=scene,
        expected_error="continuous_motion_unsupported",
        suite="continuous",
    )


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------

_S1 = card_hover


SCENARIOS: list[ScenarioEntry] = [
    ScenarioEntry("card_hover_translate", "S1", _S1),
    ScenarioEntry("card_hover_compound", "S2", lambda: card_hover(compound=True)),
    ScenarioEntry("button_bg_color", "S3", button_bg_color),
    ScenarioEntry("button_press", "S4", button_press),
    ScenarioEntry("dropdown_open", "S5", dropdown_open),
    ScenarioEntry("modal_open", "S6", modal_open),
    ScenarioEntry("accordion_expand", "S7", accordion_expand),
    ScenarioEntry("stagger_list", "S8", stagger_list),
    ScenarioEntry("card_hover_no_cursor", "S9", lambda: card_hover(cursor_visible=False)),
    ScenarioEntry("card_hover_retina", "S10", _S1, Variant(pixel_ratio=2, label="retina_2x")),
    ScenarioEntry("card_hover_vfr", "S11", _S1, Variant(encode=EncodeSpec(vfr=True), label="vfr")),
    ScenarioEntry("card_hover_30fps", "S11", _S1, Variant(fps=30.0, label="30fps")),
    ScenarioEntry(
        "card_hover_crf28", "S11", _S1, Variant(encode=EncodeSpec(crf=28), label="crf28")
    ),
    ScenarioEntry(
        "card_hover_mov", "S11", _S1, Variant(encode=EncodeSpec(container="mov"), label="mov")
    ),
    ScenarioEntry(
        "card_hover_webm",
        "S11",
        _S1,
        Variant(encode=EncodeSpec(container="webm", codec="vp9", crf=32), label="webm_vp9"),
    ),
    ScenarioEntry("negative_static", "S12", negative_static),
    ScenarioEntry("negative_scroll", "S12", negative_scroll),
]

_C4 = carousel_drag_inertia

#: PLAN-continuous §8.2 (truth ``suite == "continuous"``; evaluated from P2 on).
CONTINUOUS_SCENARIOS: list[ScenarioEntry] = [
    ScenarioEntry("marquee_autoplay", "C1", marquee_autoplay),
    ScenarioEntry("marquee_fast_loop", "C2", marquee_fast_loop),
    ScenarioEntry("marquee_hover_pause", "C3", marquee_hover_pause),
    ScenarioEntry("carousel_drag_inertia", "C4", _C4),
    ScenarioEntry(
        "carousel_drag_inertia_no_cursor", "C5", lambda: carousel_drag_inertia(cursor_visible=False)
    ),
    ScenarioEntry("carousel_drag_inertia_30fps", "C6", _C4, Variant(fps=30.0, label="30fps")),
    ScenarioEntry(
        "carousel_drag_inertia_vfr", "C7", _C4, Variant(encode=EncodeSpec(vfr=True), label="vfr")
    ),
    ScenarioEntry("carousel_drag_snap", "C8", carousel_drag_snap),
    ScenarioEntry(
        "carousel_fling_fast_crf28",
        "C9",
        carousel_fling_fast,
        Variant(fps=30.0, encode=EncodeSpec(crf=28), label="30fps_crf28"),
    ),
    ScenarioEntry("ticker_vertical_autoplay", "C10", lambda: marquee_hover_pause(vertical=True)),
    ScenarioEntry("hover_with_ambient_marquee", "C11", hover_with_ambient_marquee),
    # P2 regression for the motivating recording's clock: C5 (no cursor) captured on a 75 Hz
    # grid (13.3 / 26.7 ms stamps) while the page renders at 60 Hz, MOV container
    ScenarioEntry(
        "carousel_drag_inertia_grid75",
        "C12",
        lambda: carousel_drag_inertia(cursor_visible=False),
        Variant(encode=EncodeSpec(container="mov", capture_grid_hz=75.0), label="grid75_mov"),
    ),
    # P2b regression for the motivating recording's cards: dark theme, low-contrast cards with
    # photos, 4 px gaps, cards scaled by their distance from the viewport centre, momentum that
    # blends into autoplay; 75 Hz capture grid like C12
    ScenarioEntry(
        "carousel_dark_edge_zoom",
        "C13",
        carousel_dark_edge_zoom,
        Variant(encode=EncodeSpec(container="mov", capture_grid_hz=75.0), label="grid75_mov"),
    ),
    # acceptance-run regression: flat placeholder cards (the row splits into activity pieces),
    # a drag that stops while pressed, then rest → resume (no glide, no momentum)
    ScenarioEntry("carousel_flat_held_stop", "C14", carousel_flat_held_stop),
    ScenarioEntry("negative_ambient_pulse", "N1", negative_ambient_pulse),
]

#: The transition suite (S1–S12) as it existed before PLAN-continuous (17 entries).
TRANSITION_SCENARIOS: list[ScenarioEntry] = list(SCENARIOS)
SCENARIOS += CONTINUOUS_SCENARIOS

BY_NAME: dict[str, ScenarioEntry] = {e.name: e for e in SCENARIOS}


def get(name: str) -> ScenarioEntry:
    return BY_NAME[name]


# --------------------------------------------------------------------------------------------
# Truth derivation
# --------------------------------------------------------------------------------------------


def _ms_int(t: float) -> int:
    return int(round(t * 1000.0))


def _r(v: float, nd: int = 4) -> float:
    out = round(float(v), nd)
    return 0.0 if out == 0 else out  # no "-0.0" in the truth


def _value(prop: str, v):
    if prop in ("translateX", "translateY", "height", "border-radius"):
        return PxValue(number=_r(v))
    if prop in ("scale", "scaleX", "scaleY", "opacity"):
        return RatioValue(number=_r(v))
    if prop in ("color", "background-color"):
        return ColorValue(color=rgb_to_hex(v))
    if prop == "box-shadow":
        x, y, blur, spread, a = v
        return ShadowValue(
            shadow=Shadow(x=_r(x), y=_r(y), blur=_r(blur), spread=_r(spread), rgba=(0, 0, 0, _r(a)))
        )
    raise ValueError(prop)


def origin_name(u: float, v: float) -> str:
    hx = {0.0: "left", 0.5: "center", 1.0: "right"}[u]
    vy = {0.0: "top", 0.5: "center", 1.0: "bottom"}[v]
    return "center" if hx == vy == "center" else f"{vy} {hx}"


def _box(b: tuple[float, ...]) -> Box:
    return Box(x=_r(b[0], 2), y=_r(b[1], 2), w=_r(b[2], 2), h=_r(b[3], 2))


def _effective_opacity(scene: Scene, node_id: str, t: float) -> float:
    node, ancestors = scene._index[node_id]
    op = 1.0
    for n in (*ancestors, node):
        op *= float(scene.value(n, "opacity", t))  # type: ignore[arg-type]
    return op


def state_times(defn: ScenarioDef) -> tuple[float, float]:
    """(t_A, t_B): instants inside the stable state before / after the forward motion."""
    tws = defn.scene.timeline.tweens
    first = "rt_in" if any(w.segment == "rt_in" for w in tws) else "fwd"
    fw = [w for w in tws if w.segment == first]
    onset = min(w.start_s for w in fw)
    settle = max(w.end_s for w in fw)
    later = [w.start_s for w in tws if w.segment != first and w.start_s >= settle - 1e-9]
    t_b = (settle + min(later)) / 2.0 if later else settle + 0.05
    return max(onset - 0.05, 0.0), t_b


def cursor_events(defn: ScenarioDef) -> list[TruthCursorEvent]:
    cur = defn.scene.cursor
    if cur is None or not cur.visible:
        return []
    events: list[TruthCursorEvent] = []
    keys = cur.keys
    moving_prev = False
    for a, b in zip(keys, keys[1:], strict=False):
        moving = (a.x, a.y) != (b.x, b.y)
        if moving and not moving_prev:
            events.append(TruthCursorEvent(t_ms=_ms_int(a.t_s), kind="move_start"))
        if not moving and moving_prev:
            events.append(TruthCursorEvent(t_ms=_ms_int(a.t_s), kind="stationary"))
        moving_prev = moving
    if moving_prev:
        events.append(TruthCursorEvent(t_ms=_ms_int(keys[-1].t_s), kind="stationary"))
    el_ids = {e.node for e in defn.elements}
    if defn.trigger_node is not None:
        node_ref = defn.trigger_node if defn.trigger_node in el_ids else None
        scene = defn.scene
        trig = defn.trigger_node

        def inside(i: int) -> bool:
            t = i / 1000.0
            x, y = cur.position(t)
            bx, by, bw, bh = scene.box(trig, t)
            return bx <= x <= bx + bw and by <= y <= by + bh

        # coarse 10 ms scan, refined to the exact ms (no double crossing within 10 ms)
        n = int(round(scene.duration_s * 1000))
        inside_prev, prev = False, 0
        for i in [*range(0, n + 1, 10), n]:
            now = inside(i)
            if now != inside_prev:
                hit = next(j for j in range(prev + 1, i + 1) if inside(j) == now) if i else 0
                kind = "enter" if now else "leave"
                events.append(TruthCursorEvent(t_ms=hit, kind=kind, element_id=node_ref))
            inside_prev, prev = now, i
    for down, up in cur.clicks:
        events.append(TruthCursorEvent(t_ms=_ms_int(down), kind="mousedown"))
        events.append(TruthCursorEvent(t_ms=_ms_int(up), kind="mouseup"))
    events.sort(key=lambda e: (e.t_ms, e.kind))
    return events


@functools.cache
def cap_height(font_px: float, weight: float = 1.0) -> float:
    """Rendered ink height (CSS px) of a capital "H" in the synth font (measured at 8×)."""
    node = Node(id="_cap", shape="text", text="H", font_px=font_px, weight=weight)
    cov = build_sprite(node, 8.0).cov
    rows = np.flatnonzero((cov > 0.5).any(axis=1))
    return round((rows[-1] - rows[0] + 1) / 8.0, 2)


def appearance(scene: Scene, node_id: str, t_a: float, t_b: float) -> TruthAppearance:
    """Rendered static appearance of a node in state A (PLAN-continuous §13)."""
    node = scene.node(node_id)
    fill = scene.value(node, "fill", t_a)
    bg = rgb_to_hex(fill) if node.shape in ("rect", "icon") else None  # type: ignore[arg-type]
    texts = [n for n in node.walk() if n.shape == "text"]
    colors = {rgb_to_hex(scene.value(n, "fill", t_a)) for n in texts}  # type: ignore[arg-type]
    fonts = {(n.font_px, n.weight) for n in texts}
    tc = colors.pop() if len(colors) == 1 else None
    font = fonts.pop() if len(fonts) == 1 else None
    return TruthAppearance(
        background_color=bg,
        text_color=tc,
        font_size_px=font[0] if font else None,
        cap_height_px=cap_height(*font) if font else None,
        border_radius_px=node.radius if node.shape in ("rect", "image") else None,
        opacity_initial=_r(_effective_opacity(scene, node_id, t_a)),
        opacity_active=_r(_effective_opacity(scene, node_id, t_b)),
    )


def scene_truth(scene: Scene) -> TruthScene:
    return TruthScene(
        viewport_css=Size(w=scene.width, h=scene.height),
        page_background=rgb_to_hex(scene.background),
    )


def _flips(pred: Callable[[int], bool], n: int) -> list[tuple[int, bool]]:
    """``(ms, new_state)`` where ``pred`` changes on ``0..n`` (10 ms scan refined to 1 ms;
    assumes no double flip within 10 ms). A true state at 0 is reported as a flip at 0."""
    state = pred(0)
    out = [(0, True)] if state else []
    prev = 0
    for i in [*range(10, n + 1, 10), n]:
        now = pred(i)
        if now != state:
            j = next(j for j in range(prev + 1, i + 1) if pred(j) == now)
            out.append((j, now))
            state = now
        prev = i
    return out


def sampled_cursor_events(
    cur: CursorPath | None, box: tuple[float, ...], duration_s: float, element_id: str | None
) -> list[TruthCursorEvent]:
    """Cursor events sampled from the actual position function (works for ``DragCursor``):
    move_start / stationary, enter / leave of ``box``, mousedown / mouseup."""
    if cur is None or not cur.visible:
        return []
    n = int(round(duration_s * 1000))
    bx, by, bw, bh = box

    def pos(i: int) -> tuple[float, float]:
        return cur.position(i / 1000.0)

    def moving(i: int) -> bool:
        (x0, y0), (x1, y1) = pos(i), pos(i + 1)
        return abs(x1 - x0) + abs(y1 - y0) > 1e-6

    def inside(i: int) -> bool:
        x, y = pos(i)
        return bx <= x <= bx + bw and by <= y <= by + bh

    events = [
        TruthCursorEvent(t_ms=t, kind="move_start" if on else "stationary")
        for t, on in _flips(moving, n)
    ]
    events += [
        TruthCursorEvent(t_ms=t, kind="enter" if on else "leave", element_id=element_id)
        for t, on in _flips(inside, n)
    ]
    for down, up in cur.clicks:
        events.append(TruthCursorEvent(t_ms=_ms_int(down), kind="mousedown"))
        events.append(TruthCursorEvent(t_ms=_ms_int(up), kind="mouseup"))
    events.sort(key=lambda e: (e.t_ms, e.kind))
    return events


def _truth_easing(name: str) -> TruthEasing:
    bez, fam = EASINGS[name]
    return TruthEasing(
        name=name, cubic_bezier=bez, family=fam, keyword=name if name in CSS_KEYWORDS else None
    )


def _median(vals: list[float]) -> float | None:
    return float(statistics.median(vals)) if vals else None


def _median_ms(vals: list[int]) -> int | None:
    m = _median([float(v) for v in vals])
    return None if m is None else int(round(m))


def continuous_truth(cdef: ContinuousDef, *, cursor_visible: bool) -> TruthContinuous:
    """Ground truth of one scroller, derived from the profile that drives the renderer."""
    strip, prof = cdef.strip, cdef.profile
    segs = prof.segments
    zoom = strip.zoom
    # zoomed strip (P2b): on screen every content point moves at s(x) times the layout speed;
    # the truth speeds are those of the rigid strip that covers the same screen distance on
    # average over the viewport (``EdgeZoom.mean_scale``); 1.0 (identity) for a rigid strip
    f = zoom.mean_scale() if zoom is not None else 1.0
    phases: list[TruthPhase] = []
    last_rest_ms: int | None = None
    last_release_ms: int | None = None
    for sg in segs:
        kw: dict = {}
        spec = sg.spec
        if sg.kind == "inertia":
            kw = dict(tau_ms=_r(spec.tau * 1000.0), v0_px_s=_r(sg.v_a * f),  # type: ignore[union-attr]
                      v_inf_px_s=_r(sg.v_b * f))  # fmt: skip
        elif sg.kind in ("decelerate", "resume"):
            kw = dict(ramp_ms=sg.t1_ms - sg.t0_ms, easing=_truth_easing(spec.easing))  # type: ignore[union-attr]
        elif sg.kind == "snap":
            kw = dict(easing=_truth_easing(spec.easing), snap_step_px=_r(spec.step * f),  # type: ignore[union-attr]
                      snap_distance_px=_r((sg.target - sg.x0) * f))  # fmt: skip
        elif sg.kind == "drag":
            kw = dict(pointer_ratio=1.0)
        if sg.kind == "resume":
            if last_rest_ms is None:
                raise ValueError("resume without a preceding rest")
            kw["delay_after_rest_ms"] = sg.t0_ms - last_rest_ms
            kw["delay_after_release_ms"] = (
                sg.t0_ms - last_release_ms if last_release_ms is not None else None
            )
            last_release_ms = None
        if sg.kind == "drag":
            # a held stop is let go at an invisible moment: no delay after release (as measured)
            last_release_ms = sg.t1_ms if abs(sg.v_end) >= 1e-6 else None
        # rest-reaching phases (inertia ends at its v_stop cut-off, then the content rests; an
        # inertia that blends into autoplay (v_inf != 0) never rests)
        if (
            sg.kind in ("snap", "stop")
            or (sg.kind == "inertia" and sg.v_b == 0.0)
            or (sg.kind == "decelerate" and sg.v_end == 0)
            # a drag that slowed to a stop while pressed (HeldDrag): rests with the pointer
            or (sg.kind == "drag" and abs(sg.v_end) < 1e-6)
        ):
            last_rest_ms = sg.t1_ms
        phases.append(
            TruthPhase(
                kind=sg.kind,
                start_ms=sg.t0_ms,
                end_ms=sg.t1_ms,
                v_start_px_s=_r(sg.v_start * f),
                v_end_px_s=_r(sg.v_end * f),
                v_peak_px_s=_r(sg.v_peak() * f),
                displacement_px=_r(sg.displacement * f),
                **kw,
            )
        )

    auto = next((sg for sg in segs if sg.kind == "autoplay"), None)
    v_auto = auto.spec.v * f if auto is not None else None  # type: ignore[union-attr]
    if v_auto is None:
        direction = None
    elif strip.axis == "x":
        direction = "right" if v_auto > 0 else "left"
    else:
        direction = "down" if v_auto > 0 else "up"
    observable = prof.travel_range() >= strip.period + 0.5 * strip.view_len
    drags = [p for p in phases if p.kind == "drag"]
    resumes = [p for p in phases if p.kind == "resume"]
    decels = [p for p in phases if p.kind == "decelerate"]
    snaps = [p for p in phases if p.kind == "snap"]
    inertias = [p for p in phases if p.kind == "inertia"]
    releases = [
        abs(a.v_end_px_s)
        for a, b in zip(phases, phases[1:], strict=False)
        if a.kind == "drag" and b.kind in ("inertia", "snap", "stop")
    ]
    sample = next(n for n in strip.node().walk() if n.id == f"{strip.prefix}c0_0_t")
    t_end = prof.end_ms / 1000.0
    profile_rows = [
        (t, _r(prof.position(t / 1000.0) * f), _r(prof.velocity(t / 1000.0) * f))
        for t in range(prof.start_ms, prof.end_ms + 1, 20)
    ]
    del t_end
    gap, pitch = strip.centre_geometry()
    return TruthContinuous(
        element_id=strip.viewport_id,
        axis=strip.axis,  # type: ignore[arg-type]
        region=_box(strip.box),
        autoplay_velocity_px_s=v_auto,
        autoplay_direction=direction,  # type: ignore[arg-type]
        loop=TruthLoop(
            period_px=_r(strip.period * f),
            observable=observable,
            duration_ms=_r(strip.period * f / abs(v_auto) * 1000.0) if v_auto else None,
        ),
        pitch_px=_r(pitch),
        gap_px=_r(gap),
        card=TruthCard(
            count=strip.cards,
            w=strip.card_w,
            h=strip.card_h,
            background_color=strip.card_bg,
            border_radius_px=strip.card_radius,
            text_color=rgb_to_hex(sample.fill),
            font_size_px=sample.font_px,
            cap_height_px=cap_height(sample.font_px, sample.weight),
        ),
        span_ms=(prof.start_ms, prof.end_ms),
        phases=phases,
        pause_on=cdef.pause_on,  # type: ignore[arg-type]
        pause_on_accept=list(cdef.pause_on_accept),  # type: ignore[arg-type]
        pause_decel_ms=_median_ms([p.ramp_ms for p in decels if p.ramp_ms is not None]),
        resume_delay_after_rest_ms=_median_ms(
            [p.delay_after_rest_ms for p in resumes if p.delay_after_rest_ms is not None]
        ),
        resume_delay_after_release_ms=_median_ms(
            [p.delay_after_release_ms for p in resumes if p.delay_after_release_ms is not None]
        ),
        resume_delay_after_leave_ms=cdef.resume_delay_after_leave_ms,
        resume_ramp_ms=_median_ms([p.ramp_ms for p in resumes if p.ramp_ms is not None]),
        resume_direction_preserved=(
            all((p.v_end_px_s > 0) == (v_auto > 0) for p in resumes)
            if resumes and v_auto is not None
            else None
        ),
        snap_kind="grid" if snaps else None,
        drag_count=len(drags),
        drag_peak_speed_px_s=max((abs(p.v_peak_px_s) for p in drags), default=None),
        drag_follows_pointer=True if drags and cursor_visible else None,
        drag_pointer_ratio=1.0 if drags and cursor_visible else None,
        inertia_tau_ms=_median([p.tau_ms for p in inertias if p.tau_ms is not None]),
        release_speeds_px_s=[_r(v) for v in releases],
        profile=profile_rows,
        zoom=(
            TruthZoom(edge_scale=strip.zoom_edge, speed_scale=_r(f, 6))
            if zoom is not None
            else None
        ),
    )


def build_truth(entry: ScenarioEntry, defn: ScenarioDef | None = None) -> Truth:
    """Ground truth for a scenario entry (pure; no rendering, no probe info)."""
    defn = defn or entry.build()
    scene = defn.scene
    v = entry.variant
    r = v.pixel_ratio
    n_frames = int(scene.duration_s * v.fps + 1e-9)
    video = TruthVideo(
        file=entry.filename,
        container=v.encode.container,
        codec=v.encode.codec,
        crf=v.encode.crf,
        fps=v.fps,
        vfr=v.encode.irregular,
        capture_grid_hz=v.encode.capture_grid_hz,
        pixel_ratio=r,  # type: ignore[arg-type]
        css_width=scene.width,
        css_height=scene.height,
        width=scene.width * r,
        height=scene.height * r,
        duration_ms=int(round(n_frames / v.fps * 1000)),
        frames_rendered=n_frames,
    )
    cursor = TruthCursor(
        visible=scene.cursor is not None and scene.cursor.visible,
        style=scene.cursor.style if scene.cursor is not None else "arrow",  # type: ignore[arg-type]
        events=cursor_events(defn),
    )
    common = dict(
        name=entry.name,
        scenario=entry.scenario,
        variant=v.label,
        description=defn.description,
        checks=defn.checks,
        suite=defn.suite,
        mode=defn.mode,
        expected_warnings=list(defn.expected_warnings),
        video=video,
        scene=scene_truth(scene),
        cursor=cursor,
    )
    if defn.expected_error is not None:
        return Truth(expected_error=defn.expected_error, **common)  # type: ignore[arg-type]
    if defn.continuous is not None:
        return _build_continuous_truth(defn, common)

    tws = scene.timeline.tweens
    t_a, t_b = state_times(defn)
    measured = {e.node for e in defn.elements}

    def measured_parent(node_id: str) -> str | None:
        _, ancestors = scene._index[node_id]
        for a in reversed(ancestors):
            if a.id in measured:
                return a.id
        return None

    elements = []
    for em in defn.elements:
        b_a, b_b = scene.box(em.node, t_a), scene.box(em.node, t_b)
        elements.append(
            TruthElement(
                id=em.node,
                label=em.label,
                role=em.role,  # type: ignore[arg-type]
                parent_id=measured_parent(em.node),
                kind=em.kind,  # type: ignore[arg-type]
                text_like=em.text_like,
                bbox_initial=_box(b_a),
                bbox_active=_box(b_b),
                visible_initial=_effective_opacity(scene, em.node, t_a) > 0.01 and b_a[3] > 0.5,
                visible_active=_effective_opacity(scene, em.node, t_b) > 0.01 and b_b[3] > 0.5,
                appearance=appearance(scene, em.node, t_a, t_b),
            )
        )
    el_order = {e.node: i for i, e in enumerate(defn.elements)}
    for w in tws:
        if w.node not in measured:
            raise ValueError(f"{entry.name}: tween on {w.node!r} which is not a declared element")

    segs: list[TruthSegment] = []
    onset: dict[str, float] = {}
    for sid in SEGMENT_ORDER:
        st = [w for w in tws if w.segment == sid]
        if not st:
            continue
        onset[sid] = min(w.start_s for w in st)
        trig = defn.segment_triggers.get(sid)
        segs.append(
            TruthSegment(
                id=sid,  # type: ignore[arg-type]
                kind="forward" if sid in ("fwd", "rt_in") else "reverse",
                onset_ms=_ms_int(onset[sid]),
                settle_ms=_ms_int(max(w.end_s for w in st)),
                trigger_event_ms=_ms_int(trig) if trig is not None else None,
            )
        )

    ordered = sorted(
        tws, key=lambda w: (SEGMENT_ORDER.index(w.segment), w.start_s, el_order[w.node], w.prop)
    )
    transitions: list[TruthTransition] = []
    key_to_id: dict[tuple[str, str, str], str] = {}
    for i, w in enumerate(ordered, start=1):
        node = scene.node(w.node)
        fv, tv = _value(w.prop, w.from_value), _value(w.prop, w.to_value)
        delta = None
        if isinstance(fv, PxValue | RatioValue):
            delta = _r(tv.number - fv.number)  # type: ignore[union-attr]
        bez, fam = EASINGS[w.easing]
        notes = list(w.notes)
        parent = measured_parent(w.node)
        if parent is not None and w.prop in ("translateX", "translateY", "scale"):
            notes.append(f"relative to parent {parent}")
        tid = f"t{i}"
        key_to_id[(w.segment, w.node, w.prop)] = tid
        transitions.append(
            TruthTransition(
                id=tid,
                segment_id=w.segment,  # type: ignore[arg-type]
                element_id=w.node,
                property=w.prop,  # type: ignore[arg-type]
                from_=fv,
                to=tv,
                delta=delta,
                start_ms=_ms_int(w.start_s),
                delay_ms=_ms_int(w.start_s) - _ms_int(onset[w.segment]),
                duration_ms=_ms_int(w.duration_s),
                easing=TruthEasing(
                    name=w.easing,
                    cubic_bezier=bez,
                    family=fam,
                    keyword=w.easing if w.easing in CSS_KEYWORDS else None,
                ),
                transform_origin=origin_name(*node.origin) if w.prop == "scale" else None,
                notes=notes,
            )
        )

    rels = [
        TruthRelationship(
            kind=rs.kind,  # type: ignore[arg-type]
            segment_id=rs.segment,  # type: ignore[arg-type]
            transition_ids=[key_to_id[(rs.segment, n, p)] for n, p in rs.keys],
            offset_ms=rs.offset_ms,
            interval_ms=rs.interval_ms,
        )
        for rs in defn.relationships
    ]

    assert defn.interaction is not None
    trig_box = _box(scene.box(defn.trigger_node, t_a)) if defn.trigger_node else None
    return Truth(
        interaction=TruthInteraction(trigger_box=trig_box, **defn.interaction),
        elements=elements,
        segments=segs,
        transitions=transitions,
        relationships=rels,
        ambient=[continuous_truth(a, cursor_visible=False) for a in defn.ambient],
        **common,  # type: ignore[arg-type]
    )


def _build_continuous_truth(defn: ScenarioDef, common: dict) -> Truth:
    cdef = defn.continuous
    assert cdef is not None and defn.interaction is not None
    scene = defn.scene
    cur = scene.cursor
    visible = cur is not None and cur.visible
    vp = cdef.strip.viewport_id
    t_end = scene.duration_s
    elements = [
        TruthElement(
            id=vp,
            label=em.label,
            role=em.role,  # type: ignore[arg-type]
            kind=em.kind,  # type: ignore[arg-type]
            bbox_initial=_box(scene.box(vp, 0.0)),
            bbox_active=_box(scene.box(vp, 0.0)),
            visible_initial=True,
            visible_active=True,
            appearance=appearance(scene, vp, 0.0, 0.0),
        )
        for em in defn.elements
    ]
    common = dict(common)
    common["cursor"] = TruthCursor(
        visible=visible,
        style=cur.style if cur is not None else "arrow",  # type: ignore[arg-type]
        events=sampled_cursor_events(cur, cdef.strip.box, t_end, vp),
    )
    return Truth(
        interaction=TruthInteraction(**defn.interaction),
        elements=elements,
        continuous=continuous_truth(cdef, cursor_visible=visible),
        **common,  # type: ignore[arg-type]
    )
