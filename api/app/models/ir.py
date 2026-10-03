"""MotionSpec intermediate representation (PLAN §8). FROZEN CONTRACT after Phase 0.

The IR is the single source of truth for every output (D5). Conventions:

* Geometry is CSS px (``source.pixel_ratio`` already divided out), full-frame coordinates.
* Times are integer milliseconds on the absolute video timeline unless the field says otherwise
  (``delay_ms`` is relative to the segment onset).
* Every value is an estimate (``meta.estimated`` is always ``true``).
* Serialized JSON uses the field aliases (``Transition.from``) — ``model_dump()`` does this by
  default via ``serialize_by_alias``.
* ``initial_state`` / ``active_state`` are computed from forward transitions; any copies present
  in input JSON are ignored and recomputed.

Changing this module after P0 needs one coordinated edit + ``make contract``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Literal["0.1"] = "0.1"
DISCLAIMER = "Values are visual estimates from a screen recording, not source CSS."

# --------------------------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------------------------

InteractionType = Literal[
    "hover", "click", "press", "expand_collapse", "dropdown", "modal", "unknown"
]
TypeSource = Literal["heuristic", "interpreter"]
Pattern = Literal["card", "button", "menu", "accordion", "modal", "generic"]
TriggerKind = Literal["pointer_enter", "click", "unknown"]
ReverseTriggerKind = Literal["pointer_leave", "click", "none", "unknown"]
Direction = Literal["forward", "forward_reverse", "round_trip"]

ElementKind = Literal[
    "transform", "photometric", "appear", "disappear", "backdrop", "resize", "content_change"
]
Role = Literal[
    "card", "button", "image", "icon", "text", "title", "label", "container",
    "dropdown_menu", "menu_item", "modal_panel", "backdrop", "list_item",
    "link", "input", "badge", "accordion_panel", "other",
]  # fmt: skip
LabelSource = Literal["interpreter", "heuristic"]

SegmentId = Literal["fwd", "rev", "rt_in", "rt_out"]
SegmentKind = Literal["forward", "reverse"]

Property = Literal[
    "translateX", "translateY", "scale", "scaleX", "scaleY", "opacity", "color",
    "background-color", "box-shadow", "border-radius", "height", "content",
]  # fmt: skip
ValueKind = Literal["px", "ratio", "color", "shadow", "text"]

CssEasingKeyword = Literal["linear", "ease", "ease-in", "ease-out", "ease-in-out"]
EasingFamily = Literal["linear", "ease", "ease-in", "ease-out", "ease-in-out", "overshoot"]
EasingFlag = Literal["overshoot"]

ConfidenceBand = Literal["high", "medium", "low"]
RelationshipKind = Literal["simultaneous", "sequential", "stagger", "delayed"]
CursorEventKind = Literal["enter", "leave", "stationary", "move_start"]
InterpreterProvider = Literal["claude_cli", "openai_compat", "none"]
InterpretationStatus = Literal["ok", "fallback", "timeout", "error", "disabled"]
PixelRatio = Literal[1, 2, 3]
PixelRatioSource = Literal["user", "auto"]
WarningSeverity = Literal["info", "warn"]
WarningCode = Literal[
    "pixel_ratio_assumed", "timestamps_estimated", "vfr_source", "cursor_not_visible",
    "extra_segments_ignored", "elements_truncated", "short_stable_state", "not_settled",
    "rotation_detected", "interpretation_fallback", "interpretation_disagrees",
    "low_fps_source", "preview_unavailable", "reverse_not_recorded", "frames_subsampled",
]  # fmt: skip

JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
ElementId = Annotated[str, Field(pattern=r"^e[1-9][0-9]*$")]
TransitionId = Annotated[str, Field(pattern=r"^t[1-9][0-9]*$")]
HexColor = Annotated[str, Field(pattern=r"^#[0-9A-F]{6}$", description="Uppercase #RRGGBB.")]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
Ms = Annotated[int, Field(ge=0, description="Milliseconds.")]

#: Which ``Value.kind`` each property must use for ``from``/``to``.
PROPERTY_VALUE_KIND: dict[str, str] = {
    "translateX": "px",
    "translateY": "px",
    "height": "px",
    "border-radius": "px",
    "scale": "ratio",
    "scaleX": "ratio",
    "scaleY": "ratio",
    "opacity": "ratio",
    "color": "color",
    "background-color": "color",
    "box-shadow": "shadow",
    "content": "text",
}
NUMERIC_VALUE_KINDS = frozenset({"px", "ratio"})

#: The CSS keyword curves. ``Easing.keyword`` implies exactly this ``cubic_bezier``.
CSS_KEYWORD_BEZIER: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
}

#: Confidence band thresholds (PLAN §6.8): high >= 0.8, medium >= 0.5, low < 0.5.
BAND_HIGH_MIN = 0.8
BAND_MEDIUM_MIN = 0.5
#: Below this a transition is "uncertain": excluded from main text, commented out in CSS (§9).
UNCERTAIN_BELOW = 0.3


def band_for(score: float) -> ConfidenceBand:
    """Confidence band for a 0..1 score."""
    if score >= BAND_HIGH_MIN:
        return "high"
    if score >= BAND_MEDIUM_MIN:
        return "medium"
    return "low"


# --------------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------------


class IRModel(BaseModel):
    """Strict base: unknown keys are rejected, aliases are used on output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        json_schema_serialization_defaults_required=True,
    )


# --------------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------------


class Box(IRModel):
    """Axis-aligned rectangle in CSS px, full-frame coordinates (x, y = top-left)."""

    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)


class Confidence(IRModel):
    """A 0..1 score plus its band. ``band`` must equal ``band_for(value)``."""

    value: Score
    band: ConfidenceBand

    @classmethod
    def of(cls, value: float) -> Confidence:
        """Build from a score, deriving the band."""
        return cls(value=value, band=band_for(value))

    @model_validator(mode="after")
    def _band_matches(self) -> Confidence:
        if self.band != band_for(self.value):
            raise ValueError(f"band {self.band!r} does not match value {self.value}")
        return self


class Shadow(IRModel):
    """One box-shadow layer: ``{x}px {y}px {blur}px {spread}px rgba(r,g,b,a)``."""

    x: float
    y: float
    blur: float = Field(ge=0)
    spread: float
    rgba: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[float, Field(ge=0, le=1)],
    ]


class PxValue(IRModel):
    """Length in CSS px (translate, height, border-radius)."""

    kind: Literal["px"] = "px"
    number: float


class RatioValue(IRModel):
    """Unitless ratio (scale, opacity)."""

    kind: Literal["ratio"] = "ratio"
    number: float


class ColorValue(IRModel):
    """Solid color (color, background-color)."""

    kind: Literal["color"] = "color"
    color: HexColor


class ShadowValue(IRModel):
    """Box shadow. ``shadow: null`` means ``none`` / not detectable."""

    kind: Literal["shadow"] = "shadow"
    shadow: Shadow | None


class TextValue(IRModel):
    """Opaque content description (``content`` property; not measured geometrically)."""

    kind: Literal["text"] = "text"
    text: str


Value = Annotated[
    PxValue | RatioValue | ColorValue | ShadowValue | TextValue,
    Field(discriminator="kind"),
]


class MeasuredNumber(IRModel):
    """A static (non-animated) measurement with its own confidence."""

    value: float
    confidence: Confidence


class MeasuredShadow(IRModel):
    """Static shadow estimate. ``value: null`` = none or very subtle."""

    value: Shadow | None
    confidence: Confidence


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------


class Meta(IRModel):
    generated_at: AwareDatetime = Field(description="UTC timestamp set by the pipeline.")
    pipeline_version: str
    estimated: Literal[True] = True
    disclaimer: str = DISCLAIMER


class Source(IRModel):
    """Input video facts. ``width``/``height`` are source pixels after rotation (not CSS px)."""

    filename: str
    container: str = Field(description="e.g. mp4, mov, webm (from ffprobe format).")
    codec: str = Field(description="e.g. h264, hevc, vp9.")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    fps_nominal: float = Field(gt=0)
    fps_effective: float = Field(gt=0)
    is_vfr: bool
    pixel_ratio: PixelRatio
    pixel_ratio_source: PixelRatioSource
    timing_resolution_ms: int = Field(gt=0, description="Median frame interval, rounded.")


class Trigger(IRModel):
    kind: TriggerKind
    reverse_kind: ReverseTriggerKind
    description: str
    confidence: Confidence


class TotalDuration(IRModel):
    """``settle - onset`` per direction; ``reverse`` is null when no reverse was recorded."""

    forward: int = Field(ge=0)
    reverse: int | None = Field(default=None, ge=0)


class Interaction(IRModel):
    type: InteractionType
    type_source: TypeSource
    type_confidence: Confidence
    pattern: Pattern
    target_element_id: ElementId | None
    target_label: str
    target_description: str | None = None
    trigger: Trigger
    direction: Direction
    total_duration_ms: TotalDuration


class ElementStatic(IRModel):
    """Non-animated element properties (low confidence heuristics)."""

    border_radius_px: MeasuredNumber | None = None
    shadow: MeasuredShadow | None = None


class MotionElement(IRModel):
    """A changed UI element. Children's transforms are relative to ``parent_id``."""

    id: ElementId
    label: str
    label_source: LabelSource
    role: Role
    parent_id: ElementId | None = None
    kind: ElementKind
    bbox_initial: Box = Field(description="Box in state A (initial).")
    bbox_active: Box = Field(description="Box in state B (active / open / pressed).")
    text_like: bool = False
    static: ElementStatic = Field(default_factory=ElementStatic)


class Segment(IRModel):
    """An analysed motion window. ``fwd``/``rt_in`` are forward, ``rev``/``rt_out`` reverse.

    ``start_ms``/``end_ms`` = detected active run; ``onset_ms``/``settle_ms`` = earliest fitted
    start / latest fitted end of its transitions; ``trigger_event_ms`` = cursor enter/click time.
    """

    id: SegmentId
    kind: SegmentKind
    start_ms: Ms
    end_ms: Ms
    onset_ms: Ms
    settle_ms: Ms
    trigger_event_ms: Ms | None = None

    @model_validator(mode="after")
    def _check(self) -> Segment:
        expected = "forward" if self.id in ("fwd", "rt_in") else "reverse"
        if self.kind != expected:
            raise ValueError(f"segment {self.id} must have kind {expected!r}")
        if self.end_ms < self.start_ms or self.settle_ms < self.onset_ms:
            raise ValueError(f"segment {self.id}: end before start")
        return self


class Easing(IRModel):
    """Fitted timing function. ``keyword`` is set only when the fit *is* a CSS keyword curve."""

    keyword: CssEasingKeyword | None = None
    cubic_bezier: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2")
    nearest_named: str = Field(description="Closest named candidate (PLAN Appendix B).")
    family: EasingFamily
    rmse: float = Field(ge=0, description="Progress RMSE of the fit (0..1 progress units).")
    flags: list[EasingFlag] = Field(default_factory=list)

    @field_validator("cubic_bezier")
    @classmethod
    def _x_in_unit(cls, v: tuple[float, float, float, float]) -> tuple[float, ...]:
        if not (0.0 <= v[0] <= 1.0 and 0.0 <= v[2] <= 1.0):
            raise ValueError("cubic_bezier x1 and x2 must be within [0, 1]")
        return v

    @model_validator(mode="after")
    def _keyword_matches(self) -> Easing:
        if self.keyword is not None:
            ref = CSS_KEYWORD_BEZIER[self.keyword]
            if any(abs(a - b) > 1e-9 for a, b in zip(self.cubic_bezier, ref, strict=True)):
                raise ValueError(f"keyword {self.keyword!r} requires cubic_bezier {ref}")
        return self


class TransitionConfidence(IRModel):
    """Per-transition confidences; ``band`` derives from ``overall``."""

    overall: Score
    value: Score
    timing: Score
    easing: Score
    band: ConfidenceBand

    @model_validator(mode="after")
    def _band_matches(self) -> TransitionConfidence:
        if self.band != band_for(self.overall):
            raise ValueError(f"band {self.band!r} does not match overall {self.overall}")
        return self


class Transition(IRModel):
    """One property animating on one element in one segment."""

    id: TransitionId
    segment_id: SegmentId
    element_id: ElementId
    property: Property
    from_: Value = Field(alias="from")
    to: Value
    delta: float | None = Field(
        description="to - from for px/ratio values; null for color/shadow/text."
    )
    start_ms: Ms = Field(description="Absolute video time of the fitted start (t0).")
    delay_ms: Ms = Field(description="start_ms - segment onset_ms.")
    duration_ms: int = Field(gt=0)
    easing: Easing
    transform_origin: str | None = Field(
        default=None, description='CSS transform-origin, e.g. "center", "top center".'
    )
    confidence: TransitionConfidence
    notes: list[str] = Field(default_factory=list)
    samples: list[tuple[int, float]] = Field(
        default_factory=list,
        max_length=60,
        description="[t_ms, progress] pairs for timeline/debug. Excluded from the JSON export.",
    )

    @model_validator(mode="after")
    def _check_values(self) -> Transition:
        kind = PROPERTY_VALUE_KIND[self.property]
        if self.from_.kind != kind or self.to.kind != kind:
            raise ValueError(f"{self.id}: property {self.property} needs {kind!r} values")
        if kind in NUMERIC_VALUE_KINDS:
            expected = self.to.number - self.from_.number  # type: ignore[union-attr]
            if self.delta is None or abs(self.delta - expected) > 1e-3:
                raise ValueError(f"{self.id}: delta must equal to - from ({expected})")
        elif self.delta is not None:
            raise ValueError(f"{self.id}: delta must be null for {kind!r} values")
        return self


class Relationship(IRModel):
    """Timing relation between transitions of one segment (PLAN §6.9).

    ``transition_ids[0]`` is the reference (primary) transition. ``offset_ms`` is the start
    offset of the others relative to it; ``interval_ms`` is the stagger step.
    """

    kind: RelationshipKind
    segment_id: SegmentId
    transition_ids: list[TransitionId] = Field(min_length=2)
    offset_ms: int | None = None
    interval_ms: int | None = None

    @model_validator(mode="after")
    def _check(self) -> Relationship:
        if self.kind == "stagger" and (self.interval_ms is None or len(self.transition_ids) < 3):
            raise ValueError("stagger needs >= 3 transitions and interval_ms")
        if self.kind in ("delayed", "sequential") and self.offset_ms is None:
            raise ValueError(f"{self.kind} needs offset_ms")
        return self


class CursorEvent(IRModel):
    t_ms: Ms
    kind: CursorEventKind
    element_id: ElementId | None = None


class Cursor(IRModel):
    visible: bool
    confidence: Score
    events: list[CursorEvent] = Field(default_factory=list)


class Interpretation(IRModel):
    """Which Layer B provider produced labels/roles/structure and how it went."""

    provider: InterpreterProvider
    model: str | None = None
    status: InterpretationStatus
    duration_ms: int | None = Field(default=None, ge=0)


class SpecWarning(IRModel):
    code: WarningCode
    severity: WarningSeverity
    message: str


# --------------------------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------------------------

StateMap = dict[str, dict[Property, Value]]


class MotionSpec(IRModel):
    """Root of the IR. Cross-references (element/segment/transition ids) are validated."""

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    job_id: JobId
    meta: Meta
    source: Source
    interaction: Interaction
    elements: list[MotionElement]
    segments: list[Segment] = Field(min_length=1)
    transitions: list[Transition]
    relationships: list[Relationship] = Field(default_factory=list)
    cursor: Cursor
    structure: list[str] = Field(default_factory=list)
    interpretation: Interpretation
    warnings: list[SpecWarning] = Field(default_factory=list)

    # ---- derived ------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _drop_derived(cls, data: Any) -> Any:
        """Derived states are never trusted from input; they are recomputed."""
        if isinstance(data, dict) and ("initial_state" in data or "active_state" in data):
            data = {k: v for k, v in data.items() if k not in ("initial_state", "active_state")}
        return data

    def _forward_state(self, side: Literal["from", "to"]) -> StateMap:
        forward = {s.id for s in self.segments if s.kind == "forward"}
        out: StateMap = {}
        for el in self.elements:
            props: dict[Property, Value] = {}
            for t in self.transitions:
                if t.element_id == el.id and t.segment_id in forward:
                    props[t.property] = t.from_ if side == "from" else t.to
            if props:
                out[el.id] = props
        return out

    @computed_field(description="Element -> property -> value before the forward motion.")
    @property
    def initial_state(self) -> StateMap:
        return self._forward_state("from")

    @computed_field(description="Element -> property -> value after the forward motion.")
    @property
    def active_state(self) -> StateMap:
        return self._forward_state("to")

    # ---- integrity ------------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_refs(self) -> MotionSpec:
        el_ids = [e.id for e in self.elements]
        if len(set(el_ids)) != len(el_ids):
            raise ValueError("duplicate element ids")
        known_el = set(el_ids)
        for e in self.elements:
            if e.parent_id is not None and (e.parent_id not in known_el or e.parent_id == e.id):
                raise ValueError(f"element {e.id}: invalid parent_id {e.parent_id!r}")

        seg_ids = [s.id for s in self.segments]
        if len(set(seg_ids)) != len(seg_ids):
            raise ValueError("duplicate segment ids")
        expected_segs = {
            "forward": {"fwd"},
            "forward_reverse": {"fwd", "rev"},
            "round_trip": {"rt_in", "rt_out"},
        }[self.interaction.direction]
        if set(seg_ids) != expected_segs:
            raise ValueError(
                f"direction {self.interaction.direction!r} requires segments "
                f"{sorted(expected_segs)}, got {seg_ids}"
            )

        t_ids = [t.id for t in self.transitions]
        if len(set(t_ids)) != len(t_ids):
            raise ValueError("duplicate transition ids")
        seen: set[tuple[str, str, str]] = set()
        for t in self.transitions:
            if t.element_id not in known_el:
                raise ValueError(f"transition {t.id}: unknown element {t.element_id}")
            if t.segment_id not in seg_ids:
                raise ValueError(f"transition {t.id}: unknown segment {t.segment_id}")
            key = (t.segment_id, t.element_id, t.property)
            if key in seen:
                raise ValueError(f"transition {t.id}: duplicate {key}")
            seen.add(key)

        known_t = set(t_ids)
        for r in self.relationships:
            if r.segment_id not in seg_ids or not set(r.transition_ids) <= known_t:
                raise ValueError(f"relationship refers to unknown ids: {r.transition_ids}")

        if (
            self.interaction.target_element_id is not None
            and self.interaction.target_element_id not in known_el
        ):
            raise ValueError("interaction.target_element_id is not an element")
        for ev in self.cursor.events:
            if ev.element_id is not None and ev.element_id not in known_el:
                raise ValueError(f"cursor event refers to unknown element {ev.element_id}")
        return self


__all__ = [
    "BAND_HIGH_MIN",
    "BAND_MEDIUM_MIN",
    "CSS_KEYWORD_BEZIER",
    "DISCLAIMER",
    "PROPERTY_VALUE_KIND",
    "SCHEMA_VERSION",
    "UNCERTAIN_BELOW",
    "Box",
    "ColorValue",
    "Confidence",
    "Cursor",
    "CursorEvent",
    "Easing",
    "MotionElement",
    "ElementStatic",
    "Interaction",
    "Interpretation",
    "MeasuredNumber",
    "MeasuredShadow",
    "Meta",
    "MotionSpec",
    "PxValue",
    "RatioValue",
    "Relationship",
    "Segment",
    "Shadow",
    "ShadowValue",
    "Source",
    "SpecWarning",
    "TextValue",
    "TotalDuration",
    "Transition",
    "TransitionConfidence",
    "Trigger",
    "Value",
    "band_for",
]
