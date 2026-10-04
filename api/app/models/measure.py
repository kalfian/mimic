"""Internal data passed between pipeline stages (PLAN §6.0). FROZEN CONTRACT after Phase 0.

These are plain dataclasses (not serialized, not part of the API). They are the contract
between Track M1 (CV, produces most of them), Track M2 (pure math, consumes series/features)
and Track I / INT (assemble).

Unit conventions (differ from the IR on purpose):

* Times are **float seconds** on the absolute video timeline (``*_s`` fields, ``times`` arrays).
  ``assemble`` converts to integer ms for the IR.
* Geometry is **CSS px** (``*_css`` / :class:`Rect`) in full-frame coordinates unless the field
  says "analysis px". Use :class:`Scale` to convert.
* Images are numpy ``uint8`` arrays: BGR ``(H, W, 3)`` or gray ``(H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np

from app.models.ir import (
    Axis,
    Behavior,
    Direction,
    Easing,
    ElementKind,
    InteractionType,
    PhaseEvidence,
    PhaseFit,
    PhaseKind,
    PixelRatio,
    PixelRatioSource,
    Property,
    ReverseTriggerKind,
    SegmentId,
    SpecWarning,
    TransitionConfidence,
    TriggerKind,
    Value,
    ValueKind,
)

CursorState = Literal["moving", "stationary", "unknown"]


# --------------------------------------------------------------------------------------------
# Geometry / scale
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned box ``(x, y, w, h)``; CSS px unless stated otherwise. IR equivalent: ``Box``."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def intersection(self, other: Rect) -> Rect | None:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def union(self, other: Rect) -> Rect:
        x1, y1 = min(self.x, other.x), min(self.y, other.y)
        x2, y2 = max(self.x2, other.x2), max(self.y2, other.y2)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: Rect) -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        denom = self.area + other.area - inter.area
        return inter.area / denom if denom > 0 else 0.0

    def contained_fraction(self, outer: Rect) -> float:
        """Fraction of ``self``'s area that lies inside ``outer`` (hierarchy tests)."""
        inter = self.intersection(outer)
        return inter.area / self.area if inter is not None and self.area > 0 else 0.0

    def padded(self, pad: float) -> Rect:
        return Rect(self.x - pad, self.y - pad, self.w + 2 * pad, self.h + 2 * pad)

    def clamped(self, width: float, height: float) -> Rect:
        """Clamp to ``[0, width] x [0, height]``."""
        x1, y1 = min(max(self.x, 0.0), width), min(max(self.y, 0.0), height)
        x2, y2 = min(max(self.x2, 0.0), width), min(max(self.y2, 0.0), height)
        return Rect(x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True, slots=True)
class Scale:
    """Pixel-space conversions (PLAN P4, §6.1 step 5).

    * source frame px --(``/ pixel_ratio``)--> CSS px
    * ``analysis_scale`` = analysis px per source frame px (pass 2 decode resolution).
      ``k = analysis_scale * pixel_ratio`` = analysis px per CSS px.

    Any other image (e.g. the 960-wide pass-1 stream) is described by its own
    ``image_scale`` = image px per source frame px.
    """

    pixel_ratio: PixelRatio
    pixel_ratio_source: PixelRatioSource
    analysis_scale: float

    @property
    def k(self) -> float:
        """Analysis px per CSS px."""
        return self.analysis_scale * self.pixel_ratio

    def to_css(self, px: float, image_scale: float | None = None) -> float:
        """Length in an image (default: analysis image) -> CSS px."""
        s = self.analysis_scale if image_scale is None else image_scale
        return px / (s * self.pixel_ratio)

    def from_css(self, css: float, image_scale: float | None = None) -> float:
        """CSS px -> length in an image (default: analysis image)."""
        s = self.analysis_scale if image_scale is None else image_scale
        return css * s * self.pixel_ratio

    def area_to_css(self, px2: float, image_scale: float | None = None) -> float:
        """Area in image px² -> CSS px² (used for UI energy)."""
        s = self.analysis_scale if image_scale is None else image_scale
        return px2 / (s * self.pixel_ratio) ** 2

    def rect_to_css(self, r: Rect, image_scale: float | None = None) -> Rect:
        f = self.to_css(1.0, image_scale)
        return Rect(r.x * f, r.y * f, r.w * f, r.h * f)


# --------------------------------------------------------------------------------------------
# A0 probe
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ProbeInfo:
    """ffprobe facts. ``width``/``height`` are post-rotation source px."""

    container: str
    codec: str
    width: int
    height: int
    rotation: int  # degrees as reported (0, ±90, 180)
    duration_s: float
    frame_ts: np.ndarray  # float64 (N,), display-order seconds, monotonic
    fps_nominal: float
    fps_effective: float
    is_vfr: bool
    has_audio: bool
    timestamps_estimated: bool = False  # True -> warning `timestamps_estimated`


# --------------------------------------------------------------------------------------------
# A2 scan (pass 1) + cursor
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ActiveSegment:
    """A run of UI energy above the hysteresis thresholds."""

    start_s: float
    end_s: float
    roi_css: Rect  # union of non-cursor component boxes over the run (unpadded)
    peak_energy: float  # CSS px²
    is_global: bool = False  # scroll / page-level geometric motion


@dataclass(slots=True)
class StableSegment:
    """Gap between active segments that is >= ``min_stable`` long."""

    start_s: float
    end_s: float


@dataclass(slots=True)
class PrimaryInteraction:
    """The one interaction analysed per video (PLAN P5).

    ``round_trip`` = a single segment that returns to its start state (press/pulse);
    then ``reverse`` is None and A5 splits ``forward`` at the peak into ``rt_in``/``rt_out``.
    """

    forward: ActiveSegment
    reverse: ActiveSegment | None = None
    round_trip: bool = False


class CursorSample(NamedTuple):
    t_s: float
    x_css: float  # hotspot estimate (±15 px)
    y_css: float
    state: CursorState


@dataclass(slots=True)
class CursorTrack:
    visible: bool
    samples: list[CursorSample] = field(default_factory=list)
    #: Per pass-1 frame (aligned with ``ScanResult.frame_times``): inflated cursor box or None.
    boxes_css: list[Rect | None] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class ScanResult:
    energy: np.ndarray  # float64 (N,), CSS px² of non-cursor change per pass-1 frame
    frame_times: np.ndarray  # float64 (N,), seconds
    segments: list[ActiveSegment]
    stable: list[StableSegment]
    thumbs: np.ndarray  # uint8 (N, h, thumb_width) gray thumbnails
    primary: PrimaryInteraction
    cursor: CursorTrack
    warnings: list[SpecWarning] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# A3 windowed decode (pass 2)
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class WindowFrames:
    """Cropped analysis-resolution frames for one primary segment window.

    All windows of one interaction share ``crop_css`` so coordinates are comparable.
    """

    segment_id: SegmentId
    times: np.ndarray  # float64 (N,), seconds
    crops: list[np.ndarray]  # BGR uint8 (h, w, 3), analysis px
    crop_css: Rect  # crop rectangle in CSS px, full-frame coords
    scale: Scale
    is_dup: np.ndarray  # bool (N,), duplicate of previous frame -> excluded from fits
    cursor_masks: list[np.ndarray | None]  # uint8 (h, w), 255 = cursor pixels; None = no cursor

    @property
    def crop_origin_css(self) -> tuple[float, float]:
        return (self.crop_css.x, self.crop_css.y)


# --------------------------------------------------------------------------------------------
# A4 regions
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ElementCandidate:
    """A changed region decomposed into an element (ids ``e1..eN`` by change energy)."""

    id: str
    bbox_a: Rect  # state A, CSS px full-frame
    bbox_b: Rect  # state B
    parent_id: str | None
    kind: ElementKind
    change_energy: float
    text_like: bool = False
    #: Dominant A->B 2x3 affine in CSS px, relative to the parent (None if not estimated).
    warp_ab: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)  # e.g. "caused_by_resize"


# --------------------------------------------------------------------------------------------
# A5 per-frame series, A6 fits
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class PropertySeries:
    """Per-frame measurement of one property of one element in one segment.

    ``values`` is the measured scalar (px / ratio) for geometric and opacity properties, or the
    blend coefficient α(t) (``v_start=0``, ``v_end=1``) for color / shadow / backdrop / content.
    ``from_value``/``to_value`` always carry the reportable endpoints in IR form.
    """

    element_id: str
    property: Property
    segment: SegmentId
    times: np.ndarray  # float64 (N,), seconds, non-duplicate frames only
    values: np.ndarray  # float64 (N,)
    quality: np.ndarray  # float64 (N,), ECC rho or blend R² per frame (0..1)
    v_start: float  # median over pre-stable frames
    v_end: float  # median over post-stable frames
    unit: ValueKind
    from_value: Value
    to_value: Value
    stable_std: float = 0.0  # σ of values over stable frames (SNR in confidence)
    transform_origin: str | None = None  # scale only
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FittedTransition:
    """PropertySeries + timing/easing fit (Track M2) + confidences (``confidence.py``)."""

    id: str  # "t1".., unique per spec; assigned in fit order (segment, then t0)
    series: PropertySeries
    t0_s: float
    duration_s: float
    easing: Easing
    confidence: TransitionConfidence
    progress_rmse: float = 0.0


# --------------------------------------------------------------------------------------------
# A8 classification inputs/outputs
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ClassifyFeatures:
    """Abstract features for ``classify.py`` (PLAN §6.10). Produced by M1/INT, consumed by M2."""

    cursor_visible: bool
    enter_before_onset: bool  # cursor entered target bbox within [-400, +50] ms of onset
    stationary_at_onset: bool  # speed < 2 px/frame for >= 3 frames at onset
    leave_before_reverse: bool
    has_appear: bool
    appear_area_frac: float  # largest appear region / frame area
    has_backdrop: bool
    appear_below_trigger: bool
    resize_pattern: bool
    round_trip: bool
    forward_reverse: bool
    forward_duration_s: float
    properties: frozenset[str] = frozenset()  # IR property names present in forward
    #: Sign of the forward scale change on the target: -1 shrinks, +1 grows, 0 none / unknown.
    #: Rule 4 (press) needs "scale < 1"; 0 falls back to "has a scale property" (Phase 4).
    scale_direction: int = 0


@dataclass(slots=True)
class HeuristicInteraction:
    """Rule-based interaction guess (``classify.py``)."""

    type: InteractionType
    trigger: TriggerKind
    reverse_trigger: ReverseTriggerKind
    direction: Direction
    confidence: float
    rule: int  # matched rule number from the PLAN §6.10 table (1..8)


# --------------------------------------------------------------------------------------------
# Continuous mode (PLAN-continuous §3-§4): regime detection + scroller measurement
# --------------------------------------------------------------------------------------------

#: Why an ambient region is not a measurable scroller (selects the
#: ``continuous_motion_unsupported`` message variant, PLAN-continuous §3.5).
NotScrollerReason = Literal["two_axis", "untrackable", "opposing_unsplit"]
#: Phase kinds that mean the user interacted (``decide_mode``: scroller with interaction).
INTERACTION_PHASE_KINDS: frozenset[str] = frozenset(
    {"decelerate", "paused", "drag", "inertia", "snap", "stop", "resume"}
)


@dataclass(slots=True)
class AmbientRegion:
    """Cells active from the start of the recording (§3.2), cursor removed."""

    rect_css: Rect  # bounding box of the connected cell component, CSS px full-frame
    cells: int  # number of active cells in the component
    lead_frac: float  # mean leading-window activity fraction over its cells (0..1)
    #: Several pieces of one scroller row merged (``regime.merge_row_pieces``): the tracking box
    #: then spans every changed column along the axis, gaps included.
    merged: bool = False


@dataclass(slots=True)
class RegimeReport:
    """Output of ``continuous.regime.detect_regime`` (§3.2). No ambient region -> transition.

    The cursor track is not repeated here; it is ``ScanResult.cursor`` / ``CursorTrack``.
    """

    ambient: list[AmbientRegion]  # largest first, <= ContinuousParams.max_ambient_regions
    cell_css: float  # grid cell size used, CSS px
    grid_shape: tuple[int, int]  # (rows, cols) of the activity grid
    lead_window_s: tuple[float, float]  # [start, end] seconds of the leading window

    @property
    def has_ambient(self) -> bool:
        return bool(self.ambient)


@dataclass(slots=True)
class DisplacementSeries:
    """Per real (non-duplicate) frame displacement of the scroller content (§4.3).

    ``pos`` is signed along ``axis`` (+ = content moves right / down), ``pos[0] == 0``.
    Fits use ``pos``; velocities are derived for labelling/UI only.
    """

    times: np.ndarray  # float64 (N,), seconds (VFR-aware)
    pos: np.ndarray  # float64 (N,), CSS px
    quality: np.ndarray  # float64 (N,), 0..1 (NCC peak or ECC rho)
    axis: Axis
    region: Rect  # refined scroller viewport, CSS px full-frame
    region_confidence: float
    dt_median: float  # seconds


@dataclass(slots=True)
class PhaseCandidate:
    """One labelled phase before IR assembly (§4.5). Velocities CSS px/s, signed."""

    kind: PhaseKind
    start_s: float
    end_s: float
    v_start: float
    v_end: float
    v_peak: float  # signed velocity with the largest magnitude
    displacement: float  # CSS px, signed
    fit: PhaseFit | None = None  # IR fit model (values carry their own confidence)
    interrupted: bool = False
    evidence: PhaseEvidence = "velocity"
    confidence: float = 0.0  # label confidence
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScrollerAnalysis:
    """Everything measured for one ambient region (``continuous.measure.analyse_ambient``).

    ``is_scroller`` false -> only ``region``/``reason``/``coherence`` are meaningful.
    """

    region: AmbientRegion
    is_scroller: bool
    reason: NotScrollerReason | None = None
    axis: Axis | None = None
    coherence: float = 0.0  # fraction of coherence-window frames with response >= threshold
    series: DisplacementSeries | None = None
    autoplay_velocity: float | None = None  # CSS px/s signed; None = no autoplay
    autoplay_confidence: float = 0.0
    loop_period_px: float | None = None  # None = not observed (never guessed)
    loop_confidence: float = 0.0
    pitch_px: float | None = None
    pitch_confidence: float = 0.0
    gap_px: float | None = None
    gap_confidence: float = 0.0
    #: Card census (``continuous.cards``, P2b), set when it supplied pitch / gap: card size
    #: along / across the axis (CSS px; the centre card when the cards scale with their
    #: position) and the relative size change per CSS px² of distance from the scroller centre
    #: (None = rigid cards); with it, the farthest measured card-centre distance (CSS px) and
    #: the scale fit's confidence (``CensusResult.zoom_confidence``, uncapped).
    card_len_px: float | None = None
    card_cross_px: float | None = None
    card_zoom_per_px2: float | None = None
    card_zoom_reach_px: float | None = None
    card_zoom_confidence: float = 0.0
    phases: list[PhaseCandidate] = field(default_factory=list)
    behavior: Behavior | None = None  # IR model, aggregated by ``kinematics.aggregate``
    warnings: list[SpecWarning] = field(default_factory=list)

    @property
    def has_interaction(self) -> bool:
        """Any of decelerate / paused / drag / inertia / snap / stop / resume (§3.3)."""
        return any(p.kind in INTERACTION_PHASE_KINDS for p in self.phases)


@dataclass(slots=True)
class ContinuousMeasurement:
    """Layer A result of a continuous-mode job; ``Measurement.continuous`` (P2 integration)."""

    regime: RegimeReport
    scroller: ScrollerAnalysis  # the analysed (chosen) scroller
    extra_scrollers: int = 0  # other scrollers ignored -> warning extra_scrollers_ignored
