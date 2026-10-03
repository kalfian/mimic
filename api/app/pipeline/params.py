"""Every measurement threshold in one place (PLAN Appendix A, §6). FROZEN SHAPE after P0.

Values are *starting defaults* to calibrate against the synthetic suite in Phase 4; changing a
value is fine, renaming/removing a field needs a coordinated edit. Units are in each name
(``_css`` = CSS px, ``_s`` = seconds, ``_ms`` = milliseconds, ``_px2`` = CSS px²).
Stages receive ``params: MeasureParams`` and read their own group, e.g. ``params.scan.fps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProbeParams:
    """A0 (§6.1)."""

    vfr_cv_threshold: float = 0.15  # std(dt)/median(dt) above -> is_vfr
    auto_retina_min_width: int = 2560  # pixel_ratio=auto -> 2 if width >= this ...
    auto_retina_min_height: int = 1600  # ... or height >= this
    max_analysis_long_side: int = 1920  # analysis px; downscale further above this
    low_fps_threshold: float = 24.0  # fps_effective below -> warning low_fps_source


@dataclass(frozen=True, slots=True)
class ScanParams:
    """A2 pass 1: diff energy, cursor, segmentation, pairing (§6.3)."""

    fps: int = 30
    width: int = 960
    thumb_width: int = 240
    blur_ksize: int = 3
    diff_threshold: int = 12  # gray levels after blur
    open_ksize: int = 2
    dilate_css: float = 3.0
    # cursor
    cursor_max_css: float = 56.0
    cursor_link_max_jump_css: float = 250.0  # per 1/30 s
    cursor_min_track_frames: int = 3
    cursor_min_disp_css: float = 10.0
    cursor_visible_min_frames: int = 5
    cursor_mask_inflate_css: float = 12.0
    # hysteresis
    hi_mad_k: float = 8.0
    lo_mad_k: float = 3.0
    energy_min_hi_px2: float = 40.0
    energy_min_lo_px2: float = 15.0
    energy_min_total_px2: float = 400.0  # drop runs with ΣE below (compression blips)
    settle_frames: int = 3
    merge_gap_s: float = 0.150
    min_stable_s: float = 0.150
    # global motion (scroll / page transition)
    global_area_frac: float = 0.30
    global_shift_css: float = 2.0
    global_response: float = 0.3
    # pairing
    pair_ratio: float = 0.25
    pair_roi_iou: float = 0.3
    representative_thumbs: int = 5


@dataclass(frozen=True, slots=True)
class DecodeParams:
    """A3 pass 2 windowed decode (§6.4)."""

    window_pad_s: float = 0.400
    min_stable_side_s: float = 0.150
    crop_pad_min_css: float = 48.0
    crop_pad_frac: float = 0.15
    max_fps: float = 60.0
    max_window_frames: int = 150
    dup_max_abs_diff: int = 1


@dataclass(frozen=True, slots=True)
class RegionParams:
    """A4 state frames + element decomposition (§6.5)."""

    state_median_frames: int = 9
    change_threshold: int = 14
    open_ksize: int = 3
    close_ksize: int = 9
    min_area_px2: float = 24.0
    merge_gap_css: float = 12.0
    region_pad_css: float = 16.0
    appear_bg_std_max: float = 4.0
    appear_fg_std_min: float = 10.0
    backdrop_min_frac: float = 0.5
    backdrop_ratio_std_max: float = 0.05
    orb_nfeatures: int = 500
    orb_fast_threshold: int = 5
    ransac_reproj_px: float = 1.5
    min_inliers: int = 12
    min_inlier_ratio: float = 0.4
    residual_rigid_ratio: float = 0.1
    identity_translate_px: float = 0.3
    identity_scale: float = 0.005
    content_change_rho: float = 0.6
    resize_translate_tol_css: float = 3.0
    max_elements: int = 8
    max_depth: int = 2
    containment_parent: float = 0.90
    text_fill_ratio_max: float = 0.35
    text_stroke_max_css: float = 4.0


@dataclass(frozen=True, slots=True)
class TrackParams:
    """A4/A5 geometric tracking (§6.5, §6.6)."""

    ecc_iters: int = 100
    ecc_eps: float = 1e-5
    ecc_gauss_filt: int = 3
    ecc_min_rho: float = 0.7
    template_pad_css: float = 4.0
    inner_child_inset_frac: float = 0.10
    search_pad_css: float = 24.0
    scale_xy_split: float = 0.01  # report scaleX/scaleY if |sx - sy| above
    rotation_warn_deg: float = 1.0
    origin_min_scale_delta: float = 0.01
    origin_snap: float = 0.2


@dataclass(frozen=True, slots=True)
class PhotometricParams:
    """A5 color / opacity / backdrop / shadow / radius (§6.6)."""

    fill_erode_css: float = 2.0
    text_top_frac: float = 0.20
    min_delta_e: float = 3.0
    opacity_kappa_r2: float = 0.9
    shadow_ring_css: float = 40.0
    shadow_min_darkening: float = 2.0
    shadow_min_coverage: float = 0.30
    shadow_profile_frac: float = 0.10
    shadow_alpha_max: float = 0.6
    radius_lab_dist: float = 10.0
    radius_corner_agree_css: float = 2.0
    radius_min_delta_css: float = 2.0


@dataclass(frozen=True, slots=True)
class FitParams:
    """A6 timing + easing (§6.7)."""

    # significance gates: drop the property if |Δ| below
    gate_translate_css: float = 0.75
    gate_scale: float = 0.008
    gate_delta_e: float = 3.0
    gate_opacity: float = 0.05
    gate_radius_css: float = 2.0
    gate_height_css: float = 2.0
    # coarse crossings
    t_lo: float = 0.05
    t_hi: float = 0.95
    # joint grid
    t0_step_ms: float = 2.0
    d_step_ms: float = 4.0
    t0_back_frac: float = 0.6
    d_min_frac: float = 0.8
    d_max_frac: float = 2.5
    fit_pre_s: float = 0.150
    fit_post_s: float = 0.200
    # bezier
    lut_points: int = 513
    lut_points_final: int = 2001
    free_min_samples: int = 10
    occam_margin: float = 0.01
    refine_rounds: int = 2
    overshoot_p: float = 1.04
    overshoot_min_samples: int = 2


@dataclass(frozen=True, slots=True)
class ConfidenceParams:
    """§6.8 starting formulas, penalties and caps. Bands live in ``models.ir.band_for``."""

    rho_lo: float = 0.80
    rho_span: float = 0.18
    snr_full: float = 20.0
    snr_floor_px: float = 0.05
    snr_floor_scale: float = 0.002
    snr_floor_delta_e: float = 1.0
    fit_rmse_zero: float = 0.08
    n_min: int = 3
    n_span: int = 12
    sep_base: float = 0.4
    sep_span: float = 0.03
    easing_max: float = 0.95
    # penalties
    pen_timestamps_estimated: float = 0.15
    pen_cursor_over_element: float = 0.05
    pen_vfr: float = 0.05
    # caps
    cap_all: float = 0.95
    cap_color: float = 0.85
    cap_opacity: float = 0.6
    cap_opacity_appear: float = 0.8
    cap_backdrop: float = 0.7
    cap_shadow: float = 0.45
    cap_radius: float = 0.5
    cap_height: float = 0.75
    cap_content_change: float = 0.4
    cap_origin: float = 0.7
    # rule confidences (§6.10)
    classify_with_cursor: float = 0.85
    classify_shape_only: float = 0.55
    classify_no_cursor_hover: float = 0.4
    classify_unknown: float = 0.3  # rule 8 (nothing matched)
    #: q_track when a series has no usable per-frame quality samples.
    q_track_unknown: float = 0.5
    #: Strong ease-out tails (Phase 4 decision): the last 5 % of progress hides in the noise, so
    #: the fitted duration is uncertain. ``x95`` = fraction of the duration at which the fitted
    #: curve reaches 95 %; timing confidence is scaled by
    #: ``clip((x95 - tail_x95_zero) / (tail_x95_full - tail_x95_zero), tail_min_factor, 1)``.
    tail_x95_full: float = 0.80
    tail_x95_zero: float = 0.35
    tail_min_factor: float = 0.4


@dataclass(frozen=True, slots=True)
class RelationshipParams:
    """A7 (§6.9)."""

    tau_min_ms: float = 20.0
    stagger_min_count: int = 3
    stagger_area_tol: float = 0.30
    stagger_std_frac: float = 0.25


@dataclass(frozen=True, slots=True)
class ClassifyParams:
    """A8 (§6.10)."""

    enter_window_ms: tuple[float, float] = (-400.0, 50.0)
    stationary_speed_css: float = 2.0  # px per pass-1 frame
    stationary_min_frames: int = 3
    dropdown_max_area_frac: float = 0.40
    appear_below_gap_css: float = 24.0
    press_max_duration_ms: float = 500.0


@dataclass(frozen=True, slots=True)
class KeyframeParams:
    """A9 (§6.11)."""

    max_long_side: int = 1600
    max_element_crops: int = 8


@dataclass(frozen=True, slots=True)
class MeasureParams:
    """All thresholds. Pass one instance through the pipeline; never read module globals."""

    probe: ProbeParams = field(default_factory=ProbeParams)
    scan: ScanParams = field(default_factory=ScanParams)
    decode: DecodeParams = field(default_factory=DecodeParams)
    regions: RegionParams = field(default_factory=RegionParams)
    track: TrackParams = field(default_factory=TrackParams)
    photometric: PhotometricParams = field(default_factory=PhotometricParams)
    fit: FitParams = field(default_factory=FitParams)
    confidence: ConfidenceParams = field(default_factory=ConfidenceParams)
    relationships: RelationshipParams = field(default_factory=RelationshipParams)
    classify: ClassifyParams = field(default_factory=ClassifyParams)
    keyframes: KeyframeParams = field(default_factory=KeyframeParams)


DEFAULT_PARAMS = MeasureParams()
