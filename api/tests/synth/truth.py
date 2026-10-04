"""Ground-truth file format for synthetic scenarios: ``<name>.truth.json`` (PLAN §11.2).

The truth mirrors the IR vocabulary (``app.models.ir``) so the evaluator can compare field by
field: values reuse the IR ``Value`` union, boxes reuse ``Box``, properties/segments/kinds reuse
the IR literals. Element ids are **semantic names** (``card``, ``image``), not ``e1..eN``; the
evaluator matches them to pipeline elements by bounding-box IoU.

Conventions (identical to the IR): CSS px, full-frame coordinates, integer ms on the absolute
video timeline, child transforms relative to the parent element, ``delay_ms`` relative to the
segment onset (= earliest ``start_ms`` in the segment).

Additions for PLAN-continuous (all optional, so earlier truth files still load):

* ``scene`` + per-element ``appearance`` (§13): viewport, page background, element fill, text
  colour, font size and radius as rendered (state A). Present for every scenario; adding them
  did not change any rendered video.
* ``continuous`` (§8.1): the scroller's ground truth — axis, region, autoplay, loop, pitch/gap,
  card appearance, phases (kind, ms boundaries, velocities, model parameters), behaviour values
  and a 20 ms position/velocity profile. Velocities follow the IR sign convention.
* ``ambient``: autoplay-only scrollers that are on screen from t=0 but are not the recorded
  interaction (C11: expected to be masked, ``mode == "transition"``).
* ``suite``: ``continuous`` for the PLAN-continuous scenarios (C1-C12, N1). The evaluator reports
  those as *pending* until the pipeline analyses them (P2, ``evaluate.CONTINUOUS_SUITE_ENABLED``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.ir import (
    SIGN_CONVENTION,
    AutoplayDirection,
    Axis,
    Box,
    Direction,
    EasingFamily,
    ElementKind,
    HexColor,
    InteractionType,
    Pattern,
    PauseTrigger,
    PhaseKind,
    Property,
    RelationshipKind,
    ReverseTriggerKind,
    Role,
    SegmentId,
    SegmentKind,
    SignConvention,
    Size,
    SnapKind,
    SpecMode,
    TriggerKind,
    Value,
    WarningCode,
)

TRUTH_SCHEMA = "mimic-synth-truth/1"

ExpectedError = Literal["no_motion_detected", "unsupported_motion", "continuous_motion_unsupported"]
Suite = Literal["transition", "continuous"]
TruthCursorEventKind = Literal["move_start", "stationary", "enter", "leave", "mousedown", "mouseup"]


class TruthModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class TruthEasing(TruthModel):
    name: str = Field(description="Named curve (PLAN Appendix B).")
    cubic_bezier: tuple[float, float, float, float]
    family: EasingFamily
    keyword: str | None = Field(description="CSS keyword if the curve is one, else null.")


class TruthAppearance(TruthModel):
    """Static appearance as rendered (PLAN-continuous §13). Values are taken in state A; the
    effective (group) opacities tell whether the element is visible there.

    ``text_color``/``font_size_px``/``cap_height_px``: a text node's own values, or for a
    container the values shared by **all** its text descendants (null when they differ).
    ``font_size_px`` is the renderer's font size (``Node.font_px`` = the font's ascent metric
    from ``cv2.getTextSize``); ``cap_height_px`` is the **rendered** ink height of a capital "H"
    at that size (unambiguous). A UI font with cap height ≈ 0.7 em gives CSS
    ``font-size ≈ cap_height_px / 0.7``."""

    background_color: HexColor | None = Field(
        description="Solid fill (rect / icon); null for text, image (texture) and layout groups."
    )
    text_color: HexColor | None = None
    font_size_px: float | None = None
    cap_height_px: float | None = None
    border_radius_px: float | None = Field(
        default=None, description="rect / image corner radius; null otherwise."
    )
    opacity_initial: float = Field(ge=0, le=1, description="Effective opacity in state A.")
    opacity_active: float = Field(ge=0, le=1, description="Effective opacity in state B.")


class TruthScene(TruthModel):
    """Recorded page context (PLAN-continuous §13)."""

    viewport_css: Size
    page_background: HexColor


class TruthElement(TruthModel):
    id: str = Field(description="Semantic id (scene node id).")
    label: str
    role: Role
    parent_id: str | None = None
    kind: ElementKind
    text_like: bool = False
    bbox_initial: Box = Field(description="Visible box in state A (geometric even if invisible).")
    bbox_active: Box = Field(description="Visible box in state B.")
    visible_initial: bool
    visible_active: bool
    appearance: TruthAppearance | None = None


class TruthSegment(TruthModel):
    id: SegmentId
    kind: SegmentKind
    onset_ms: int
    settle_ms: int
    trigger_event_ms: int | None = None


class TruthTransition(TruthModel):
    id: str = Field(pattern=r"^t[1-9][0-9]*$")
    segment_id: SegmentId
    element_id: str
    property: Property
    from_: Value = Field(alias="from")
    to: Value
    delta: float | None
    start_ms: int
    delay_ms: int
    duration_ms: int = Field(gt=0)
    easing: TruthEasing
    transform_origin: str | None = None
    notes: list[str] = Field(default_factory=list)


class TruthRelationship(TruthModel):
    kind: RelationshipKind
    segment_id: SegmentId
    transition_ids: list[str] = Field(min_length=2, description="[0] is the reference.")
    offset_ms: int | None = None
    interval_ms: int | None = None


class TruthCursorEvent(TruthModel):
    t_ms: int
    kind: TruthCursorEventKind
    element_id: str | None = None


class TruthCursor(TruthModel):
    visible: bool
    style: Literal["arrow", "hand", "auto"]
    events: list[TruthCursorEvent] = Field(default_factory=list)


class TruthInteraction(TruthModel):
    type: InteractionType
    accept_types: list[InteractionType] = Field(
        description="Types the evaluator accepts (always includes ``type``)."
    )
    trigger: TriggerKind
    reverse_trigger: ReverseTriggerKind
    direction: Direction
    pattern: Pattern
    target_element_id: str | None
    trigger_box: Box | None = Field(
        default=None, description="Box of the hovered/clicked control (may be unanimated)."
    )


class VideoProbe(TruthModel):
    """ffprobe facts of the encoded file (filled after encoding)."""

    format_name: str
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    color_space: str | None
    avg_fps: float
    r_fps: float
    duration_ms: int
    frame_count: int
    is_vfr: bool
    dt_cv: float = Field(description="std(dt) / median(dt) of frame timestamps.")


class TruthVideo(TruthModel):
    file: str = Field(description="File name, relative to the truth file.")
    container: Literal["mp4", "mov", "webm"]
    codec: Literal["h264", "vp9"]
    crf: int
    fps: float = Field(description="Render frame rate (nominal).")
    vfr: bool = Field(
        description="Irregular timestamps: mpdecimate + -fps_mode vfr, or a capture grid."
    )
    capture_grid_hz: float | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Recorder clock grid (Hz) different from the render clock (encode.py).",
    )
    pixel_ratio: Literal[1, 2]
    css_width: int
    css_height: int
    width: int = Field(description="Device px.")
    height: int
    duration_ms: int = Field(description="Rendered duration (frames / fps).")
    frames_rendered: int
    probe: VideoProbe | None = None


# --------------------------------------------------------------------------------------------
# Continuous motion (PLAN-continuous §8.1)
# --------------------------------------------------------------------------------------------


class TruthCard(TruthModel):
    """One card of the scroller strip (all cards share geometry and colours)."""

    count: int = Field(gt=0, description="Distinct cards in one copy of the strip.")
    w: float
    h: float
    background_color: HexColor
    border_radius_px: float
    text_color: HexColor = Field(description="Title text colour.")
    font_size_px: float = Field(description="Title font size (see TruthAppearance).")
    cap_height_px: float = Field(description="Rendered cap height of the title font.")


class TruthLoop(TruthModel):
    period_px: float = Field(description="Length of one content copy (always known in synth).")
    observable: bool = Field(
        description="Content travel range >= period + 0.5 x viewport length along the axis "
        "(PLAN-continuous §4.4); otherwise the pipeline must not report a period."
    )
    duration_ms: float | None = Field(description="period / autoplay speed x 1000 (if autoplay).")


class TruthPhase(TruthModel):
    """One phase of the velocity profile. Velocities CSS px/s, signed (IR sign convention)."""

    kind: PhaseKind
    start_ms: int
    end_ms: int
    v_start_px_s: float
    v_end_px_s: float
    v_peak_px_s: float
    displacement_px: float
    interrupted: bool = False
    # model parameters (by kind)
    tau_ms: float | None = Field(default=None, description="inertia")
    v0_px_s: float | None = Field(default=None, description="inertia: release velocity")
    v_inf_px_s: float | None = Field(default=None, description="inertia")
    ramp_ms: int | None = Field(default=None, description="decelerate / resume duration")
    easing: TruthEasing | None = Field(default=None, description="decelerate / resume / snap")
    snap_step_px: float | None = Field(default=None, description="snap: grid step")
    snap_distance_px: float | None = Field(default=None, description="snap: signed distance")
    pointer_ratio: float | None = Field(default=None, description="drag: content / pointer")
    delay_after_rest_ms: int | None = Field(default=None, description="resume")
    delay_after_release_ms: int | None = Field(default=None, description="resume")


class TruthZoom(TruthModel):
    """Cards that grow towards the scroller edges (P2b, the motivating recording's effect).

    Card / gap / pitch truths are the geometry at the viewport centre (scale 1); speeds and
    displacements are scaled by ``speed_scale`` (see ``scenarios.continuous_truth``)."""

    edge_scale: float = Field(description="Card scale at the viewport edges (centre = 1).")
    speed_scale: float = Field(description="Mean screen magnification over the viewport.")


def _omit_if_none(v: object) -> bool:
    return v is None


class TruthContinuous(TruthModel):
    """Ground truth of one scroller (``mode == "continuous"`` or an ambient marquee)."""

    element_id: str = Field(description="Semantic id of the scroller (viewport) element.")
    axis: Axis
    region: Box = Field(description="Scroller viewport, CSS px.")
    sign_convention: SignConvention = SIGN_CONVENTION
    autoplay_velocity_px_s: float | None = Field(description="Signed; null = no autoplay.")
    autoplay_direction: AutoplayDirection | None
    loop: TruthLoop
    pitch_px: float = Field(description="Card + gap.")
    gap_px: float
    card: TruthCard
    span_ms: tuple[int, int]
    phases: list[TruthPhase] = Field(min_length=1)
    # behaviour (what a perfect pipeline aggregates; instances agree by construction)
    pause_on: PauseTrigger | None = Field(description="Actual trigger of the autoplay stop.")
    pause_on_accept: list[PauseTrigger] = Field(
        default_factory=list,
        description="Triggers the evaluator accepts (unknown when no cursor is visible).",
    )
    pause_decel_ms: int | None = Field(description="null = autoplay stops instantly (press).")
    resume_delay_after_rest_ms: int | None
    resume_delay_after_release_ms: int | None
    resume_delay_after_leave_ms: int | None = Field(
        default=None, description="Hover scenarios: pointer leave -> resume start (not in IR)."
    )
    resume_ramp_ms: int | None
    resume_direction_preserved: bool | None
    snap_kind: SnapKind | None
    drag_count: int = 0
    drag_peak_speed_px_s: float | None
    drag_follows_pointer: bool | None = Field(description="null = no drag or no visible cursor.")
    drag_pointer_ratio: float | None
    inertia_tau_ms: float | None
    release_speeds_px_s: list[float] = Field(default_factory=list)
    profile: list[tuple[int, float, float]] = Field(
        default_factory=list, description="[t_ms, pos_px, v_px_s] every 20 ms."
    )
    zoom: TruthZoom | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Cards scale with their distance from the viewport centre (P2b); null = "
        "rigid strip (omitted).",
    )


class Truth(TruthModel):
    schema_: Literal["mimic-synth-truth/1"] = Field(default=TRUTH_SCHEMA, alias="schema")
    name: str
    scenario: str = Field(description="S1..S12, C1..C12, N1")
    variant: str | None = None
    description: str
    checks: list[str] = Field(default_factory=list, description="Primary checks (PLAN §11.3).")
    suite: Suite = "transition"
    mode: SpecMode = Field(default="transition", description="Expected IR mode.")
    expected_error: ExpectedError | None = None
    expected_warnings: list[WarningCode] = Field(default_factory=list)
    video: TruthVideo
    scene: TruthScene | None = None
    interaction: TruthInteraction | None = None
    cursor: TruthCursor
    elements: list[TruthElement] = Field(default_factory=list)
    segments: list[TruthSegment] = Field(default_factory=list)
    transitions: list[TruthTransition] = Field(default_factory=list)
    relationships: list[TruthRelationship] = Field(default_factory=list)
    continuous: TruthContinuous | None = None
    ambient: list[TruthContinuous] = Field(default_factory=list)

    # ---- I/O ----------------------------------------------------------------------------

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

    def write(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Truth:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    # ---- lookups ------------------------------------------------------------------------

    def element(self, element_id: str) -> TruthElement:
        for e in self.elements:
            if e.id == element_id:
                return e
        raise KeyError(element_id)

    def transition(self, transition_id: str) -> TruthTransition:
        for t in self.transitions:
            if t.id == transition_id:
                return t
        raise KeyError(transition_id)
