"""Ground-truth file format for synthetic scenarios: ``<name>.truth.json`` (PLAN §11.2).

The truth mirrors the IR vocabulary (``app.models.ir``) so the evaluator can compare field by
field: values reuse the IR ``Value`` union, boxes reuse ``Box``, properties/segments/kinds reuse
the IR literals. Element ids are **semantic names** (``card``, ``image``), not ``e1..eN``; the
evaluator matches them to pipeline elements by bounding-box IoU.

Conventions (identical to the IR): CSS px, full-frame coordinates, integer ms on the absolute
video timeline, child transforms relative to the parent element, ``delay_ms`` relative to the
segment onset (= earliest ``start_ms`` in the segment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.ir import (
    Box,
    Direction,
    EasingFamily,
    ElementKind,
    InteractionType,
    Pattern,
    Property,
    RelationshipKind,
    ReverseTriggerKind,
    Role,
    SegmentId,
    SegmentKind,
    TriggerKind,
    Value,
)

TRUTH_SCHEMA = "mimic-synth-truth/1"

ExpectedError = Literal["no_motion_detected", "unsupported_motion"]
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
    vfr: bool = Field(description="Encoded with mpdecimate + -fps_mode vfr.")
    pixel_ratio: Literal[1, 2]
    css_width: int
    css_height: int
    width: int = Field(description="Device px.")
    height: int
    duration_ms: int = Field(description="Rendered duration (frames / fps).")
    frames_rendered: int
    probe: VideoProbe | None = None


class Truth(TruthModel):
    schema_: Literal["mimic-synth-truth/1"] = Field(default=TRUTH_SCHEMA, alias="schema")
    name: str
    scenario: str = Field(description="S1..S12")
    variant: str | None = None
    description: str
    checks: list[str] = Field(default_factory=list, description="Primary checks (PLAN §11.3).")
    expected_error: ExpectedError | None = None
    video: TruthVideo
    interaction: TruthInteraction | None = None
    cursor: TruthCursor
    elements: list[TruthElement] = Field(default_factory=list)
    segments: list[TruthSegment] = Field(default_factory=list)
    transitions: list[TruthTransition] = Field(default_factory=list)
    relationships: list[TruthRelationship] = Field(default_factory=list)

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
