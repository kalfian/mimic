"""Scenario definitions S1–S12 + ground-truth derivation (PLAN §11.3).

Each builder returns a :class:`ScenarioDef`: the scene (nodes, tweens, cursor path) **and** the
semantic metadata the renderer can't know (roles, expected element kinds, interaction type,
explicit timing relationships). The truth file is derived from the very same tweens that drive
the renderer, so the two cannot drift apart.

Canvas: 1280×800 CSS px (2560×1600 device px for the Retina variant). A static "page chrome"
(header, sidebar, footer text) keeps frames from being empty; scenario content sits right of
the sidebar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from app.models.ir import Box, ColorValue, PxValue, RatioValue, Shadow, ShadowValue

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
from .scene import Node, Scene, hex_to_rgb, rgb_to_hex, text_size
from .truth import (
    Truth,
    TruthCursor,
    TruthCursorEvent,
    TruthEasing,
    TruthElement,
    TruthInteraction,
    TruthRelationship,
    TruthSegment,
    TruthTransition,
    TruthVideo,
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


def scene_of(nodes: list[Node], cursor: CursorPath | None, duration_s: float, static=None) -> Scene:
    return Scene(
        width=CSS_W,
        height=CSS_H,
        background=c(PAGE_BG),
        static_nodes=page_chrome() if static is None else static,
        nodes=nodes,
        timeline=Timeline(),
        cursor=cursor,
        duration_s=duration_s,
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
        vfr=v.encode.vfr,
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
        video=video,
        cursor=cursor,
    )
    if defn.expected_error is not None:
        return Truth(expected_error=defn.expected_error, **common)  # type: ignore[arg-type]

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
        **common,  # type: ignore[arg-type]
    )
