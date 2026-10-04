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
* IR 0.2 (PLAN-continuous §5) adds ``mode`` (always serialized) and the optional ``continuous``
  section (continuous mode only). Fields added in 0.2 that a transition spec does not use
  (``continuous``, ``scene``, ``ElementStatic`` appearance values) are **omitted from the JSON
  when null**, so a 0.2 transition spec differs from 0.1 only in ``schema_version`` and ``mode``.
  Stored 0.1 documents (no ``mode``) still validate and load as ``mode="transition"``.

Changing this module after P0 needs one coordinated edit + ``make contract``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

#: Version the pipeline emits. ``MotionSpec.schema_version`` also accepts ``"0.1"`` (stored jobs).
SCHEMA_VERSION: Literal["0.2"] = "0.2"
SchemaVersion = Literal["0.1", "0.2"]
DISCLAIMER = "Values are visual estimates from a screen recording, not source CSS."

# --------------------------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------------------------

#: ``continuous`` = autoplaying scroller without pointer drag (marquee); ``drag`` = scroller the
#: user drags (carousel). Both only in ``mode="continuous"`` (see ``CONTINUOUS_INTERACTION_TYPES``).
InteractionType = Literal[
    "hover", "click", "press", "expand_collapse", "dropdown", "modal", "unknown",
    "continuous", "drag",
]  # fmt: skip
TypeSource = Literal["heuristic", "interpreter"]
Pattern = Literal["card", "button", "menu", "accordion", "modal", "generic", "marquee", "carousel"]
TriggerKind = Literal["pointer_enter", "click", "unknown", "autoplay", "drag"]
ReverseTriggerKind = Literal["pointer_leave", "click", "none", "unknown"]
#: ``continuous`` <-> ``segments == []`` and ``mode == "continuous"``.
Direction = Literal["forward", "forward_reverse", "round_trip", "continuous"]

ElementKind = Literal[
    "transform", "photometric", "appear", "disappear", "backdrop", "resize", "content_change",
    "scroller",
]  # fmt: skip
Role = Literal[
    "card", "button", "image", "icon", "text", "title", "label", "container",
    "dropdown_menu", "menu_item", "modal_panel", "backdrop", "list_item",
    "link", "input", "badge", "accordion_panel", "other", "scroller",
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
    "ambient_motion_masked", "extra_scrollers_ignored", "tracking_degraded",
    "loop_period_not_observed",
]  # fmt: skip

# ---- continuous mode (PLAN-continuous §5) ----------------------------------------------------
SpecMode = Literal["transition", "continuous"]
Axis = Literal["x", "y"]
AutoplayDirection = Literal["left", "right", "up", "down"]
PhaseKind = Literal[
    "autoplay", "decelerate", "paused", "drag", "inertia", "snap", "stop", "resume", "unknown"
]
PhaseEvidence = Literal["velocity", "cursor", "velocity+cursor"]
PauseTrigger = Literal["hover", "press", "unknown"]
InertiaModel = Literal["exponential", "tween"]
SnapKind = Literal["grid", "abrupt_ambiguous"]
#: Documents the sign of every signed velocity / displacement in the ``continuous`` section.
SIGN_CONVENTION = "positive = content moves right (x) / down (y)"
SignConvention = Literal["positive = content moves right (x) / down (y)"]

#: Interaction types / roles that only exist in continuous mode. The interpreter schema for
#: transition jobs excludes them (``interpret/schema.py``) so Layer B cannot pick them there.
CONTINUOUS_INTERACTION_TYPES: tuple[str, ...] = ("continuous", "drag")
CONTINUOUS_ROLES: tuple[str, ...] = ("scroller",)
#: Phase kinds in grammar order (exported to ``types.ts`` as ``PHASE_KINDS``).
PHASE_KINDS: tuple[str, ...] = get_args(PhaseKind)
#: Upper bound of ``ContinuousMotion.samples`` (decimated velocity profile for the UI chart).
CONTINUOUS_SAMPLE_MAX = 900
#: Allowed ``Phase.fit.model`` per phase kind (``None`` = no fit).
PHASE_FIT_MODELS: dict[str, frozenset[str | None]] = {
    "autoplay": frozenset({"constant"}),
    "decelerate": frozenset({"ramp"}),
    "resume": frozenset({"ramp"}),
    "inertia": frozenset({"exponential", "tween"}),
    "snap": frozenset({"tween", None}),
    "drag": frozenset({None}),
    "stop": frozenset({None}),
    "paused": frozenset({None}),
    "unknown": frozenset({None}),
}
#: |autoplay.velocity_px_s| must equal autoplay.speed_px_s.value within this.
VELOCITY_SPEED_TOL = 1e-6
#: |loop.duration_ms - period_px / speed * 1000| tolerance (rounding of stored values), ms.
LOOP_DURATION_TOL_MS = 1.0

JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
ElementId = Annotated[str, Field(pattern=r"^e[1-9][0-9]*$")]
TransitionId = Annotated[str, Field(pattern=r"^t[1-9][0-9]*$")]
HexColor = Annotated[str, Field(pattern=r"^#[0-9A-F]{6}$", description="Uppercase #RRGGBB.")]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
Ms = Annotated[int, Field(ge=0, description="Milliseconds.")]
PhaseId = Annotated[str, Field(pattern=r"^p[1-9][0-9]*$")]


def _omit_if_none(v: Any) -> bool:
    """``Field(exclude_if=...)`` predicate: optional 0.2 additions are left out when null."""
    return v is None


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


class MeasuredColor(IRModel):
    """A static (non-animated) solid colour estimate with its own confidence."""

    value: HexColor
    confidence: Confidence


class Size(IRModel):
    """Width x height in CSS px."""

    w: float = Field(gt=0)
    h: float = Field(gt=0)


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
    """Non-animated element properties (low confidence heuristics), measured in state A.

    The appearance values (PLAN-continuous §13) are optional and omitted from the JSON when not
    measured: ``background_color`` (element fill), ``text_color`` (text-like elements only) and
    ``font_size_px`` (rough estimate from the text line height; confidence at most medium).
    """

    border_radius_px: MeasuredNumber | None = None
    shadow: MeasuredShadow | None = None
    background_color: MeasuredColor | None = Field(default=None, exclude_if=_omit_if_none)
    text_color: MeasuredColor | None = Field(default=None, exclude_if=_omit_if_none)
    font_size_px: MeasuredNumber | None = Field(default=None, exclude_if=_omit_if_none)

    @model_validator(mode="after")
    def _font_size_capped(self) -> ElementStatic:
        if self.font_size_px is not None and self.font_size_px.confidence.band == "high":
            raise ValueError("font_size_px confidence is capped at medium (< 0.8)")
        return self


class Scene(IRModel):
    """Recorded page context (PLAN-continuous §13). Optional; omitted when not measured."""

    viewport_css: Size = Field(description="Recorded frame size in CSS px.")
    page_background: MeasuredColor | None = Field(default=None, exclude_if=_omit_if_none)


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
# Continuous mode (PLAN-continuous §5): one single-axis scroller
# --------------------------------------------------------------------------------------------
#
# Velocities are CSS px/s and signed per ``SIGN_CONVENTION`` (+ = content moves right / down);
# ``*_ms`` times are absolute video milliseconds like the rest of the IR.


class LoopInfo(IRModel):
    """Seamless-loop length of the autoplay content. Never guessed: ``observed`` is false when
    the content did not repeat within the recording (``period_px``/``duration_ms`` null)."""

    observed: bool
    period_px: MeasuredNumber | None = Field(description="Length of one content copy, CSS px.")
    duration_ms: MeasuredNumber | None = Field(
        description="period_px / speed (stored so outputs can quote it, never recomputed)."
    )

    @model_validator(mode="after")
    def _check(self) -> LoopInfo:
        if self.observed != (self.period_px is not None):
            raise ValueError("loop.observed must be true exactly when period_px is set")
        if self.observed != (self.duration_ms is not None):
            raise ValueError("loop.duration_ms must be set exactly when the loop is observed")
        return self


class Autoplay(IRModel):
    """Constant-velocity autoplay before any interaction."""

    direction: AutoplayDirection
    speed_px_s: MeasuredNumber = Field(description="|velocity|, CSS px/s (> 0).")
    velocity_px_s: float = Field(description="Signed velocity; same magnitude as speed_px_s.")
    easing: Literal["linear"] = "linear"
    loop: LoopInfo

    @model_validator(mode="after")
    def _check(self) -> Autoplay:
        speed = self.speed_px_s.value
        if speed <= 0:
            raise ValueError("autoplay.speed_px_s must be > 0")
        if abs(abs(self.velocity_px_s) - speed) > VELOCITY_SPEED_TOL:
            raise ValueError("autoplay |velocity_px_s| must equal speed_px_s.value")
        positive = self.direction in ("right", "down")
        if (self.velocity_px_s > 0) != positive:
            raise ValueError(
                f"autoplay velocity sign does not match direction {self.direction!r} "
                f"({SIGN_CONVENTION})"
            )
        if self.loop.observed:
            assert self.loop.period_px is not None and self.loop.duration_ms is not None
            expected = self.loop.period_px.value / speed * 1000.0
            if abs(self.loop.duration_ms.value - expected) > LOOP_DURATION_TOL_MS:
                raise ValueError(f"loop.duration_ms must equal period_px / speed ({expected:.1f})")
        return self


class ConstantFit(IRModel):
    """``x(t) = x0 + v t`` (autoplay)."""

    model: Literal["constant"] = "constant"
    velocity_px_s: MeasuredNumber


class ExponentialFit(IRModel):
    """Inertia: ``v(t) = v_inf + (v0 - v_inf) e^(-t/tau)``, fitted in the position domain."""

    model: Literal["exponential"] = "exponential"
    tau_ms: MeasuredNumber
    v0_px_s: MeasuredNumber = Field(description="Release velocity (signed).")
    v_inf_px_s: float = Field(description="Asymptotic velocity: 0 or the autoplay velocity.")
    stop_px_s: float | None = Field(
        default=None,
        ge=0,
        exclude_if=_omit_if_none,
        description="Speed (|v| of the decay) at which the content was observed to come to "
        "rest; the momentum is dropped below it. Omitted when the decay did not end at rest.",
    )


class RampFit(IRModel):
    """Eased velocity ramp ``v = from + (to - from) E((t - t0) / D)`` (decelerate / resume)."""

    model: Literal["ramp"] = "ramp"
    from_px_s: float
    to_px_s: float
    duration_ms: MeasuredNumber
    easing: Easing


class TweenFit(IRModel):
    """Eased position tween ``x = x0 + D E((t - t0) / T)`` (snap, tween-style inertia)."""

    model: Literal["tween"] = "tween"
    distance_px: MeasuredNumber = Field(description="Signed distance travelled.")
    duration_ms: MeasuredNumber
    easing: Easing


PhaseFit = Annotated[
    ConstantFit | ExponentialFit | RampFit | TweenFit,
    Field(discriminator="model"),
]


class Phase(IRModel):
    """One labelled stretch of the scroller's velocity profile (chronological, no overlap)."""

    id: PhaseId
    kind: PhaseKind
    start_ms: Ms
    end_ms: Ms
    v_start_px_s: float
    v_end_px_s: float
    v_peak_px_s: float = Field(description="Signed velocity with the largest magnitude.")
    displacement_px: float
    fit: PhaseFit | None = None
    interrupted: bool = Field(
        default=False, description="Cut short, e.g. decelerate cut by a drag, inertia re-grabbed."
    )
    evidence: PhaseEvidence = "velocity"
    confidence: Confidence = Field(description="Label confidence.")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Phase:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"phase {self.id}: end_ms must be after start_ms")
        model = self.fit.model if self.fit is not None else None
        if model not in PHASE_FIT_MODELS[self.kind]:
            allowed = sorted(str(m) for m in PHASE_FIT_MODELS[self.kind])
            raise ValueError(
                f"phase {self.id}: fit {model!r} not allowed for {self.kind} {allowed}"
            )
        return self


class PauseBehavior(IRModel):
    """Autoplay slows to a stop when the pointer hovers / presses (``on``)."""

    on: PauseTrigger
    on_confidence: Confidence
    decel_ms: MeasuredNumber | None
    easing: Easing | None
    stops_completely: bool


class DragBehavior(IRModel):
    count: int = Field(ge=1, description="Number of drag phases.")
    follows_pointer: bool | None = Field(description="null = no cursor visible to compare.")
    pointer_ratio: MeasuredNumber | None = Field(description="Content / pointer velocity.")
    peak_speed_px_s: MeasuredNumber


class InertiaBehavior(IRModel):
    """Momentum after release, aggregated over ``instances`` inertia phases."""

    model: InertiaModel
    tau_ms: MeasuredNumber | None = Field(description="Set when model is exponential.")
    duration_ms: MeasuredNumber | None = Field(description="Set when model is tween.")
    easing: Easing | None = Field(description="Set when model is tween.")
    release_speed_min_px_s: float = Field(ge=0)
    release_speed_max_px_s: float = Field(ge=0)
    instances: int = Field(ge=1)
    stop_px_s: MeasuredNumber | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Exponential model: minimum speed at which the momentum stops (aggregated "
        "ExponentialFit.stop_px_s). Omitted when not observed.",
    )

    @model_validator(mode="after")
    def _check(self) -> InertiaBehavior:
        if self.model == "exponential" and self.tau_ms is None:
            raise ValueError("exponential inertia needs tau_ms")
        if self.model == "tween" and (self.duration_ms is None or self.easing is None):
            raise ValueError("tween inertia needs duration_ms and easing")
        if self.release_speed_min_px_s > self.release_speed_max_px_s:
            raise ValueError("release_speed_min_px_s must be <= release_speed_max_px_s")
        return self


class SnapBehavior(IRModel):
    """``grid``: rests land on the card pitch. ``abrupt_ambiguous``: abrupt stops that may be a
    snap or the pointer stopping before release (no grid evidence; from ``stop`` phases)."""

    kind: SnapKind
    step_px: MeasuredNumber | None
    duration_ms: MeasuredNumber | None
    easing: Easing | None
    overshoot: bool = Field(description="Spring-like overshoot seen; spring not reconstructed.")

    @model_validator(mode="after")
    def _check(self) -> SnapBehavior:
        if self.kind == "grid" and self.step_px is None:
            raise ValueError("grid snap needs step_px")
        if self.kind == "abrupt_ambiguous" and self.step_px is not None:
            raise ValueError("abrupt_ambiguous snap has no step_px")
        return self


class ResumeBehavior(IRModel):
    delay_after_rest_ms: MeasuredNumber
    delay_after_release_ms: MeasuredNumber | None
    delay_after_leave_ms: MeasuredNumber | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Hover pause: resume onset after the pointer left the scroller. Omitted "
        "when no pointer leave was seen before the resume.",
    )
    ramp_ms: MeasuredNumber
    easing: Easing | None
    to_speed_px_s: float = Field(ge=0)
    direction_preserved: bool


class Behavior(IRModel):
    """What the generators print; each value aggregated over phase instances. null = not seen."""

    pause: PauseBehavior | None = None
    drag: DragBehavior | None = None
    inertia: InertiaBehavior | None = None
    snap: SnapBehavior | None = None
    resume: ResumeBehavior | None = None


class Span(IRModel):
    start_ms: Ms
    end_ms: Ms

    @model_validator(mode="after")
    def _check(self) -> Span:
        if self.end_ms <= self.start_ms:
            raise ValueError("span end_ms must be after start_ms")
        return self


#: ``[t_ms, v_px_s, pos_px, quality]`` (velocity smoothed; position integrated, signed).
ContinuousSample = tuple[Ms, float, float, Score]

#: |card_scale.mean_scale - 1 - (scale_at_reference - 1) (half / reference)² / 3| tolerance
#: (rounding of the stored values).
CARD_SCALE_MEAN_TOL = 1e-3
#: card_scale.reference_distance_px may exceed half the region length by this much (rounding).
CARD_SCALE_REACH_TOL_PX = 1.0


class CardScale(IRModel):
    """Cards scale with their on-screen distance from the scroller centre (a position-driven
    transform updated every frame; PLAN-continuous P2c).

    ``scale(d) = 1 + (scale_at_reference - 1) · (d / reference_distance_px)²`` where ``d`` is
    the distance (CSS px, along the axis) of a card's centre from the scroller centre: 1 at the
    centre, so the card element box, ``pitch_px`` and ``gap_px`` describe the card at the
    centre. ``reference_distance_px`` is the farthest whole card measured (the curve beyond is
    an extrapolation). The cards stay packed — the gaps keep their size — so neighbours move
    apart as they grow.

    Speeds and displacements in the ``continuous`` section are on-screen values; on average over
    the scroller the content is magnified ``mean_scale`` times (``1 + (scale_at_reference - 1)
    · (half / reference_distance_px)² / 3``, half = half the region length along the axis),
    so an unscaled track moves at ``speed / mean_scale``. Stored so outputs can quote it.
    """

    model: Literal["quadratic"] = "quadratic"
    origin: Literal["scroller_center"] = "scroller_center"
    reference_distance_px: float = Field(
        gt=0, description="Farthest measured card-centre distance from the scroller centre."
    )
    scale_at_reference: float = Field(gt=1, description="Card scale at reference_distance_px.")
    mean_scale: float = Field(
        ge=1, description="Mean on-screen magnification over the scroller (speed conversion)."
    )
    confidence: Confidence


class ContinuousMotion(IRModel):
    """The analysed scroller (``mode == "continuous"`` only)."""

    element_id: ElementId = Field(description='MotionElement with kind "scroller".')
    axis: Axis
    region: Box = Field(description="Scroller viewport, CSS px.")
    region_confidence: Confidence
    sign_convention: SignConvention = SIGN_CONVENTION
    autoplay: Autoplay | None = Field(description="null = no autoplay before interaction.")
    pitch_px: MeasuredNumber | None = Field(description="Card spacing (card + gap) if observed.")
    gap_px: MeasuredNumber | None = Field(
        default=None, description="Gap between cards if observed (PLAN-continuous §13)."
    )
    card_scale: CardScale | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Cards scale with their distance from the scroller centre. Omitted when the "
        "cards were rigid or not measured.",
    )
    phases: list[Phase] = Field(min_length=1)
    behavior: Behavior
    span_ms: Span = Field(description="Analysed span (= interaction.total_duration_ms.forward).")
    samples: list[ContinuousSample] = Field(
        default_factory=list,
        max_length=CONTINUOUS_SAMPLE_MAX,
        description="[t_ms, v_px_s, pos_px, quality] for the velocity chart. Excluded from the "
        "JSON export.",
    )

    @model_validator(mode="after")
    def _check(self) -> ContinuousMotion:  # noqa: C901 - flat list of rules
        if self.autoplay is not None:
            want = ("left", "right") if self.axis == "x" else ("up", "down")
            if self.autoplay.direction not in want:
                raise ValueError(
                    f"autoplay direction {self.autoplay.direction!r} not on {self.axis}"
                )
        prev_end = self.span_ms.start_ms
        for i, ph in enumerate(self.phases):
            if ph.id != f"p{i + 1}":
                raise ValueError(f"phase ids must be p1..pN in order, got {ph.id!r} at {i}")
            if ph.start_ms < prev_end:
                raise ValueError(f"phase {ph.id} overlaps the previous phase or the span start")
            prev_end = ph.end_ms
        if prev_end > self.span_ms.end_ms:
            raise ValueError("phases must end within span_ms")
        kinds = [ph.kind for ph in self.phases]
        b = self.behavior
        if (b.drag is not None) != ("drag" in kinds):
            raise ValueError("behavior.drag must be set exactly when a drag phase exists")
        if b.drag is not None and b.drag.count != kinds.count("drag"):
            raise ValueError("behavior.drag.count must equal the number of drag phases")
        if (b.inertia is not None) != ("inertia" in kinds):
            raise ValueError("behavior.inertia must be set exactly when an inertia phase exists")
        if b.inertia is not None and b.inertia.instances != kinds.count("inertia"):
            raise ValueError("behavior.inertia.instances must equal the number of inertia phases")
        if b.snap is not None:
            need = "snap" if b.snap.kind == "grid" else "stop"
            if need not in kinds:
                raise ValueError(f"{b.snap.kind} snap needs a {need!r} phase")
        if b.resume is not None and "resume" not in kinds:
            raise ValueError("behavior.resume needs a resume phase")
        last_t = -1
        for t_ms, _v, _x, _q in self.samples:
            if t_ms <= last_t:
                raise ValueError("samples must be strictly increasing in t_ms")
            last_t = t_ms
        cs = self.card_scale
        if cs is not None:
            half = (self.region.w if self.axis == "x" else self.region.h) / 2.0
            if cs.reference_distance_px > half + CARD_SCALE_REACH_TOL_PX:
                raise ValueError("card_scale.reference_distance_px must lie inside the region")
            rel = half / cs.reference_distance_px
            expected = 1.0 + (cs.scale_at_reference - 1.0) * rel * rel / 3.0
            if abs(cs.mean_scale - expected) > CARD_SCALE_MEAN_TOL:
                raise ValueError(f"card_scale.mean_scale must equal {expected:.4f}")
        return self


# --------------------------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------------------------

StateMap = dict[str, dict[Property, Value]]


class MotionSpec(IRModel):
    """Root of the IR. Cross-references (element/segment/transition ids) are validated.

    ``mode`` selects the shape: ``transition`` (segments + transitions, PLAN §8) or
    ``continuous`` (one scroller in ``continuous``; no segments/transitions/relationships).
    """

    schema_version: SchemaVersion = SCHEMA_VERSION
    job_id: JobId
    meta: Meta
    mode: SpecMode = "transition"
    source: Source
    scene: Scene | None = Field(default=None, exclude_if=_omit_if_none)
    interaction: Interaction
    elements: list[MotionElement]
    segments: list[Segment] = Field(
        description="Transition mode: >= 1 segment. Continuous mode: empty."
    )
    transitions: list[Transition]
    relationships: list[Relationship] = Field(default_factory=list)
    continuous: ContinuousMotion | None = Field(default=None, exclude_if=_omit_if_none)
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
    def _check_mode(self) -> MotionSpec:
        """``mode == continuous`` <=> ``continuous`` set <=> direction ``continuous`` <=>
        interaction type in ``CONTINUOUS_INTERACTION_TYPES`` (PLAN-continuous §5.2)."""
        continuous = self.mode == "continuous"
        checks = {
            "continuous section": self.continuous is not None,
            'interaction.direction "continuous"': self.interaction.direction == "continuous",
            "interaction.type continuous|drag": (
                self.interaction.type in CONTINUOUS_INTERACTION_TYPES
            ),
        }
        for what, present in checks.items():
            if present != continuous:
                raise ValueError(
                    f"mode {self.mode!r} {'requires' if continuous else 'forbids'} {what}"
                )
        if not continuous:
            if not self.segments:
                raise ValueError("transition mode needs at least one segment")
            return self
        assert self.continuous is not None
        c = self.continuous
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"continuous mode needs schema_version {SCHEMA_VERSION!r}")
        if self.segments or self.transitions or self.relationships:
            raise ValueError("continuous mode needs empty segments, transitions, relationships")
        scroller = next((e for e in self.elements if e.id == c.element_id), None)
        if scroller is None or scroller.kind != "scroller":
            raise ValueError("continuous.element_id must be an element with kind 'scroller'")
        if self.interaction.target_element_id != c.element_id:
            raise ValueError("continuous mode: interaction.target_element_id must be the scroller")
        total = self.interaction.total_duration_ms
        if total.reverse is not None or total.forward != c.span_ms.end_ms - c.span_ms.start_ms:
            raise ValueError(
                "continuous mode: total_duration_ms.forward = analysed span, reverse = null"
            )
        if (self.interaction.type == "drag") != (c.behavior.drag is not None):
            raise ValueError('interaction.type "drag" <=> behavior.drag is set')
        return self

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
        expected_segs: set[str] = {
            "forward": {"fwd"},
            "forward_reverse": {"fwd", "rev"},
            "round_trip": {"rt_in", "rt_out"},
            "continuous": set(),
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
    "CONTINUOUS_INTERACTION_TYPES",
    "CONTINUOUS_ROLES",
    "CONTINUOUS_SAMPLE_MAX",
    "CSS_KEYWORD_BEZIER",
    "DISCLAIMER",
    "PHASE_FIT_MODELS",
    "PHASE_KINDS",
    "PROPERTY_VALUE_KIND",
    "SCHEMA_VERSION",
    "SIGN_CONVENTION",
    "UNCERTAIN_BELOW",
    "Autoplay",
    "Behavior",
    "Box",
    "CardScale",
    "ColorValue",
    "Confidence",
    "ConstantFit",
    "ContinuousMotion",
    "Cursor",
    "CursorEvent",
    "DragBehavior",
    "Easing",
    "ExponentialFit",
    "InertiaBehavior",
    "LoopInfo",
    "MotionElement",
    "ElementStatic",
    "Interaction",
    "Interpretation",
    "MeasuredColor",
    "MeasuredNumber",
    "MeasuredShadow",
    "Meta",
    "MotionSpec",
    "PauseBehavior",
    "Phase",
    "PhaseFit",
    "PxValue",
    "RampFit",
    "RatioValue",
    "Relationship",
    "ResumeBehavior",
    "Scene",
    "Segment",
    "Shadow",
    "ShadowValue",
    "Size",
    "SnapBehavior",
    "Source",
    "Span",
    "SpecWarning",
    "TextValue",
    "TotalDuration",
    "Transition",
    "TransitionConfidence",
    "Trigger",
    "TweenFit",
    "Value",
    "band_for",
]
