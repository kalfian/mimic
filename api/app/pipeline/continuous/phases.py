"""Phase segmentation of a scroller's displacement profile (PLAN-continuous §4.5).

Input: a :class:`DisplacementSeries` (positions, CSS px, signed). Output: chronological,
contiguous :class:`PhaseCandidate` list + autoplay velocity + aggregated IR ``Behavior``.

Pipeline (step numbers = §4.5):

1. **Autoplay speed** from the leading window (low cv, |v| < ``autoplay_max_speed``), refined
   by a position-domain constant fit over the autoplay runs. Leading window not steady →
   ``autoplay = None``.
2. **Coarse labels** on the smoothed velocity: ``A`` (≈ v_auto), ``Z`` (≈ 0), ``M`` (other).
   Short runs are absorbed (``A`` runs also need ≥ ``MIN_AUTOPLAY_RUN_S`` so an inertia decay
   crossing the autoplay band is not mistaken for autoplay).
3. Each ``M`` run is classified by context: pure in-band transitions are **decelerate** /
   **resume** ramps (step 6); otherwise an optional decelerate prefix (cut by the press →
   ``interrupted``), an optional resume suffix, and a **drag region** in between. The drag
   region is split at re-grab valleys of |v|; every sub-run is split into ``drag`` + release
   tail by minimising ``tail χ² + k·ln n + DRAG_COST_PER_SAMPLE · n_head`` over the release
   index and the models {exponential (v_∞ ∈ {0, v_auto}), eased tween, none}.
4. Abrupt ends (> ``stop_from_speed`` to ≈ 0 within ``stop_max_frames``) → ``stop``.
5. **Snap evidence** from rest positions on the pitch grid relabels stops / tween releases as
   ``snap``; overshoot flagged.
7. Resume delays (after rest / after release).
8. Cursor fusion when a cursor track is given (pause trigger, drag follows pointer).
9. Aggregation (``kinematics.aggregate``).

Sharp boundaries (drag start, stop end) are placed where the observed positions leave the
neighbouring phase's fitted model by more than ``ONSET_SIGMAS`` noise σ (position domain).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.models.ir import (
    Behavior,
    ConstantFit,
    ExponentialFit,
    MeasuredNumber,
    PauseTrigger,
    PhaseEvidence,
    PhaseFit,
    PhaseKind,
    RampFit,
    SpecWarning,
    TweenFit,
)
from app.models.measure import CursorSample, DisplacementSeries, PhaseCandidate, Rect
from app.pipeline.continuous import confidence as cconf
from app.pipeline.continuous import kinematics as kin
from app.pipeline.continuous.displacement import robust_velocity, smoothed_velocity
from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams

# ---- module-local constants (TODO(calibrate P2): candidates for ContinuousParams) ----------
#: χ² cost per unmodelled drag sample in the release split (≈ 2.8σ per sample): a parametric
#: tail wins wherever it explains the data; the release kink itself is pinned by the tail
#: misfit + the velocity-continuity term, not by this constant.
DRAG_COST_PER_SAMPLE = 8.0
#: The last ``HEAD_LOCAL`` drag samples before a release candidate are modelled by a local
#: quadratic (smooth pointer motion) instead of the flat per-sample cost, so the release
#: lands on the velocity kink rather than drifting into the drag.
HEAD_LOCAL = 6
#: An in-band dip of autoplay counts as decelerate + resume only if the speed drops below
#: ``|v_auto| − DIP_MIN_TOLS · tol_a``; shallower dips are noise.
DIP_MIN_TOLS = 2.0
#: Release-velocity continuity tolerance (pointer velocity at release = content velocity):
#: σ² = var(head) + (CONT_REL · |v|)² + CONT_ABS².
CONT_REL = 0.1
CONT_ABS = 20.0
#: A release model whose initial speed exceeds the drag speed by this factor is rejected.
CONT_MAX_RATIO = 1.25
#: Golden-section iterations of the continuous tween start (``_refine_tween_start``).
TWEEN_START_ITERS = 12
#: Samples of the local linear fit that estimates the drag velocity at a release candidate.
HEAD_VEL = 8
#: Position-noise model: σ_i² = σ0² + (k_v · v_i)² (see ``noise_sigma``); defaults for k_v when
#: there are too few fast samples to estimate it; ``FAST_SPEED`` px/s = "fast" sample.
SIGMA0_MIN_PX = 0.03
SIGMA0_MAX_PX = 2.0
SIGMA_T_CFR_S = 0.0003
SIGMA_T_VFR_S = 0.0015
Q_FLOOR = 0.25
FAST_SPEED = 300.0
MOVING_MIN_SPEED = 3.0
#: Boundary onset: positions leave the neighbouring model by this many σ for 2 samples.
ONSET_SIGMAS = 3.0
#: The drag onset is searched in ``[run start − 6, run start + ONSET_AHEAD)`` samples, then up to
#: twice that far ahead.
ONSET_AHEAD = 8
#: The drag onset is measured against the autoplay line fitted over this long before the run;
#: a departure must also exceed ``ONSET_MIN_PX`` (a slowly varying on-screen speed, e.g. cards
#: that scale with position, drifts off any straight line by a few tenths of a px per 100 ms
#: while the noise σ of a clean video is a few hundredths).
ONSET_LINE_S = 0.5
ONSET_MIN_PX = 0.3
#: Minimum durations (frames): a drag after its refined onset, and a rest between two motions.
MIN_DRAG_FRAMES = 2
MIN_REST_FRAMES = 2
#: A slowdown cut by a press within this many frames is the press itself (the press stops
#: autoplay at once; a drag that then starts slowly, e.g. one that ends at rest while still
#: pressed, shows a frame or two of near standstill first).
PRESS_DECEL_FRAMES = 3
#: Autoplay runs shorter than this are absorbed (an inertia decay crossing the band).
MIN_AUTOPLAY_RUN_S = 0.2
#: Leading window used for the autoplay speed (s) and its minimum steady speed (px/s).
LEAD_WINDOW_S = 0.6
#: Autoplay band (P2b): at least AUTO_NOISE_SIGMAS × the leading autoplay's velocity noise,
#: but never wider than AUTO_TOL_MAX_FRAC of the autoplay speed.
AUTO_NOISE_SIGMAS = 3.0
AUTO_TOL_MAX_FRAC = 0.3
#: A non-autoplay run shorter than this at the very start (recorder start-up glitch) is
#: folded into the leading autoplay run.
LEAD_GLITCH_S = 0.15
#: ``A M A`` with all M positions within max(px, k·σ) of one straight line is a velocity blip.
#: 2 px (acceptance run): on replicas with flat placeholder cards that scale with their position
#: the tracker wobbles by up to ≈1.5 px for ≲ 1 s while autoplay runs steadily (the cards move
#: at different on-screen speeds and few features carry the match); any real slowdown, pause or
#: drag leaves the autoplay line by several px.
BLIP_MAX_PX = 2.0
BLIP_SIGMAS = 5.0
#: Re-grab valley: |v| must drop by this fraction of the preceding peak (and by
#: ``REGRAB_MIN_DROP`` px/s) and rise again by the same amounts.
REGRAB_REL = 0.3
REGRAB_MIN_DROP = 40.0
#: An ``M`` run is a pure decelerate / resume ramp when at least this fraction of its samples lie
#: in the autoplay band and none is beyond ``RAMP_MAX_FACTOR`` × the autoplay speed (tracking
#: noise on low-texture content spikes single samples out of the band; drags are far faster).
RAMP_BAND_FRAC = 0.85
RAMP_MAX_FACTOR = 2.0
#: Ramp fit context windows (s) before / after an M run.
RAMP_PRE_S = 0.4
RAMP_POST_S = 0.4
RAMP_DUR_MAX_S = 2.0
#: Momentum stop threshold plausibility: the fitted stop speed must be ≤ min(px/s, frac · |v0|).
STOP_V_MAX = 60.0
STOP_V_FRAC = 0.25
#: Travel-proportional systematic error floor of release tails (fraction of distance).
SYS_FRAC = 0.002
#: Tween (snap) durations searched, s.
TWEEN_DUR_MAX_S = 1.0
#: Extension of a release tail into the following rest / autoplay run (s).
TAIL_EXT_MAX_S = 1.5
#: Trailing samples of an inertia tail that define the rest it ends in (``_observed_rest_end``).
REST_END_SAMPLES = 4
#: A sample interval longer than this many frame intervals is a gap of skipped duplicates.
DUP_GAP_FRAMES = 1.5
#: A pointer leave counts for the hover resume delay up to this long after the resume onset
#: (the leave time is ± one pass-1 frame).
LEAVE_AFTER_ONSET_S = 0.05
#: Inertia decaying to rest ends at this speed (CSS px/s) unless it is observed to be cut at
#: ≥ ``CUT_MIN_PX_S`` (``_inertia_end``). Same convention as the synthetic truth (10 px/s).
INERTIA_END_PX_S = 10.0
CUT_MIN_PX_S = 15.0
#: An observed cut must beat the uncut decay by this Δχ² (``_cut_evidence``).
CUT_MIN_DCHI2 = 25.0
#: Monotone check tolerance for exponential tails: max(abs px/s, rel · |v0|).
MONO_TOL_ABS = 10.0
MONO_TOL_REL = 0.05
#: Fixed label confidences for kinds without a model comparison.
STOP_LABEL_CONF = 0.45
NOTE_STOP = "pointer likely stopped before release, or hard snap (ambiguous without cursor)"
NOTE_DEGRADED = "tracking degraded during fast motion"
NOTE_REGRAB = "re-grabbed before coming to rest"
NOTE_CUT = "cut short by a drag before reaching zero"
NOTE_MOMENTUM_STOP = "momentum stops abruptly once it has slowed down (minimum-speed threshold)"
NOTE_OVERSHOOT = "spring-like overshoot; spring parameters not reconstructed"
NOTE_POINTER_DECEL = (
    "slows down with the pointer before letting go (decay too unlike this scroller's momentum)"
)
#: ``_merge_inconsistent_inertia``: τ outside [ref / TAU_RATIO, ref · TAU_RATIO] is not momentum.
TAU_RATIO = 2.0
#: Momentum blending into autoplay through zero (``_merge_blend``): refit reduced χ² limit, τ
#: agreement with the inertia's own fit, longest standstill at the zero crossing (s).
BLEND_CHI2 = 4.0
BLEND_TAU_REL = 0.3
BLEND_MAX_REST_S = 0.15
NOTE_BLEND = "momentum blends back into autoplay (passes through a standstill on the way)"
#: ``_merge_pointer_stops``: a glide-to-rest (tween release) that stops within this fraction of
#: the time the scroller's exponential momentum needs to slow from the same speed to
#: ``INERTIA_END_PX_S`` is not momentum: the pointer stopped while still pressed. Needs
#: ``POINTER_STOP_MIN_EXP`` exponential instances whose τ agree within ``TAU_RATIO``.
POINTER_STOP_FRAC = 0.5
POINTER_STOP_MIN_EXP = 2
NOTE_POINTER_STOP = (
    "the pointer slows to a stop before letting go (stops far sooner than this scroller's "
    "momentum would)"
)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Ph:
    kind: PhaseKind
    start: float
    end: float
    start_fixed: bool = False  # boundary from a model / onset (wins over run boundaries)
    end_fixed: bool = False
    fit: PhaseFit | None = None
    raw: kin.Fit | None = None
    v_start: float | None = None
    v_end: float | None = None
    interrupted: bool = False
    label_conf: float = 0.5
    notes: list[str] = field(default_factory=list)
    evidence: PhaseEvidence = "velocity"
    stop_px_s: float | None = None  # inertia decaying to rest: speed at its end (_inertia_end)


@dataclass(slots=True)
class Segmentation:
    """Result of :func:`segment`."""

    phases: list[PhaseCandidate]
    autoplay_velocity: float | None
    autoplay_confidence: float
    behavior: Behavior
    velocity: np.ndarray  # smoothed, per sample (UI / samples)
    sigma: np.ndarray  # position noise model per sample
    rests: list[float]  # rest positions used for snap evidence
    snap_resultant: float | None
    warnings: list[SpecWarning] = field(default_factory=list)


@dataclass(slots=True)
class _Ctx:
    t: np.ndarray
    x: np.ndarray
    q: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    w: np.ndarray
    dt: float
    v_auto: float | None
    tol_a: float
    tol_z: float
    params: ContinuousParams
    is_vfr: bool
    ts_est: bool
    rv: np.ndarray | None = None  # robust_v() cache

    def qt(self, a: int, b: int) -> float:
        return cconf.q_track(self.q[max(a, 0) : max(b, a + 1)], self.params)

    def robust_v(self) -> np.ndarray:
        """Peak-speed velocity (``displacement.robust_velocity``), computed once."""
        if self.rv is None:
            self.rv = robust_velocity(self.t, self.x, self.q, q_min=self.params.degraded_q)
        return self.rv


# --------------------------------------------------------------------------------------------
# Noise model, autoplay, coarse labels
# --------------------------------------------------------------------------------------------


def _robust_sigma(e: np.ndarray, var_factor: float = 1.5) -> float:
    """σ from interpolation residuals ``e`` (var(e) = ``var_factor`` σ² for white noise)."""
    if e.size < 5:
        return math.nan
    mad = float(np.median(np.abs(e - np.median(e))))
    return 1.4826 * mad / math.sqrt(var_factor)


def noise_sigma(
    t: np.ndarray, x: np.ndarray, v: np.ndarray, q: np.ndarray, *, is_vfr: bool
) -> np.ndarray:
    """Per-sample position σ (CSS px), estimated from the data.

    ``σ_i² = σ0² + (k_v · v_i)²``. Residual ``e_i`` = ``x_i`` minus the straight line through
    its neighbours **at the recorded times** (so timestamp jitter shows up as velocity-scaled
    noise), detrended by a running median (smooth acceleration of inertia / ramps). ``σ0`` from
    slow samples, ``k_v`` (seconds: timing jitter / blur) from fast samples
    (``|v| > FAST_SPEED``); without enough fast samples ``k_v`` defaults to ``SIGMA_T_VFR_S`` /
    ``SIGMA_T_CFR_S``. Samples with very low tracking quality (``q < Q_FLOOR``) get ×2.
    """
    n = len(t)
    s0 = math.nan
    kv = SIGMA_T_VFR_S if is_vfr else SIGMA_T_CFR_S
    if n >= 7:
        frac = (t[1:-1] - t[:-2]) / np.maximum(t[2:] - t[:-2], 1e-9)
        e = x[1:-1] - (x[:-2] + (x[2:] - x[:-2]) * frac)
        trend = np.array([np.median(e[max(0, i - 2) : i + 3]) for i in range(len(e))])
        e = e - trend
        sp = np.abs(v[1:-1])
        slow_cap = max(30.0, float(np.percentile(np.abs(v), 40)))
        # moving-but-slow samples (autoplay): includes sub-pixel rendering / interpolation
        # error that a resting image does not show
        slow = (sp >= MOVING_MIN_SPEED) & (sp < slow_cap)
        if np.count_nonzero(slow) < 5:
            slow = sp < slow_cap
        s0 = _robust_sigma(e[slow]) if np.count_nonzero(slow) >= 5 else _robust_sigma(e)
        fast = sp > FAST_SPEED
        if np.count_nonzero(fast) >= 8 and math.isfinite(s0):
            sf = _robust_sigma(e[fast])
            vf = float(np.median(sp[fast]))
            if math.isfinite(sf) and vf > 0:
                kv = max(math.sqrt(max(sf * sf - s0 * s0, 0.0)) / vf, SIGMA_T_CFR_S)
    if not math.isfinite(s0):
        s0 = SIGMA0_MIN_PX
    s0 = float(np.clip(s0, SIGMA0_MIN_PX, SIGMA0_MAX_PX))
    sig = np.sqrt(s0 * s0 + (kv * v) ** 2)
    return np.where(q < Q_FLOOR, 2.0 * sig, sig)


def _runs(labels: Sequence[str]) -> list[list]:
    out: list[list] = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1][2] = i + 1
        else:
            out.append([lab, i, i + 1])
    return out


def lead_velocity_noise(t: np.ndarray, v: np.ndarray, v_lead: float | None) -> float:
    """Robust σ of the smoothed velocity around ``v_lead`` in the leading window (0 = none).

    Autoplay on a capture grid that is not the render rate (C12/C13: 75 Hz stamps, 60 Hz
    frames) or content that is not perfectly rigid (cards scaling with position, C13) wobbles
    by several px/s around a slow autoplay; the autoplay band must not be narrower than that."""
    if v_lead is None or len(t) < 4:
        return 0.0
    lead = t <= t[0] + LEAD_WINDOW_S
    if np.count_nonzero(lead) < 4:
        return 0.0
    return 1.4826 * float(np.median(np.abs(v[lead] - v_lead)))


def leading_autoplay(t: np.ndarray, v: np.ndarray, params: ContinuousParams) -> float | None:
    """Median smoothed velocity of the leading window if it is steady autoplay, else None."""
    if len(t) < 4:
        return None
    lead = t <= t[0] + LEAD_WINDOW_S
    if np.count_nonzero(lead) < 4:
        lead = np.zeros(len(t), bool)
        lead[: min(len(t), 6)] = True
    vl = v[lead]
    med = float(np.median(vl))
    if abs(med) < params.label_zero_tol or abs(med) >= params.autoplay_max_speed:
        return None
    # robust cv (MAD): recordings often start with a few glitchy frames
    cv = float(1.4826 * np.median(np.abs(vl - med)) / abs(med))
    return med if cv < params.autoplay_cv_max else None


def coarse_labels(v: np.ndarray, v_auto: float | None, tol_a: float, tol_z: float) -> list[str]:
    out = []
    for vi in v:
        if v_auto is not None and abs(vi - v_auto) <= tol_a:
            out.append("A")
        elif abs(vi) <= tol_z:
            out.append("Z")
        else:
            out.append("M")
    return out


def absorb_short(runs: list[list], t: np.ndarray, min_samples: int) -> list[list]:
    """Absorb short runs (§4.5 step 2) and merge equal neighbours.

    One run at a time, shortest first, merging equal neighbours after every change: relabelling
    in sequence order let a one-sample blip that had just been relabelled make the next short
    autoplay run look isolated (C13: ``A … M A(0.2 s) M`` lost the autoplay before a press)."""
    runs = [list(r) for r in runs]

    def merge(rs: list[list]) -> list[list]:
        out: list[list] = []
        for r in rs:
            if out and out[-1][0] == r[0]:
                out[-1][2] = r[2]
            else:
                out.append(list(r))
        return out

    def relabel(i: int) -> str | None:
        lab, a, b = runs[i]
        short = b - a < min_samples
        if lab == "A" and not short:
            short = (t[b - 1] - t[a]) < MIN_AUTOPLAY_RUN_S and 0 < i < len(runs) - 1
        if not short:
            return None
        prev_lab = runs[i - 1][0] if i > 0 else None
        next_lab = runs[i + 1][0] if i + 1 < len(runs) else None
        if lab == "M" and prev_lab is not None and next_lab is not None and prev_lab != next_lab:
            return None  # quick transition between two different steady states: keep
        if prev_lab is not None and prev_lab == next_lab:
            new = prev_lab
        elif lab in ("A", "Z") and "M" in (prev_lab, next_lab):
            new = "M"
        else:
            new = prev_lab or next_lab
        return None if new is None or new == lab else new

    runs = merge(runs)
    while len(runs) > 1:
        cands = [(runs[i][2] - runs[i][1], i, new) for i in range(len(runs))
                 if (new := relabel(i)) is not None]  # fmt: skip
        if not cands:
            break
        _, i, new = min(cands, key=lambda z: (z[0], z[1]))
        runs[i][0] = new
        runs = merge(runs)
    return runs


def _local_sigma(x: np.ndarray) -> float:
    """Position noise (px) from second differences (robust MAD; a constant or slowly varying
    velocity contributes nothing): the tracking noise of this stretch, whatever the global
    noise model says."""
    if len(x) < 5:
        return 0.0
    d2 = np.diff(x, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6.0))


def _absorb_blips(
    t: np.ndarray, x: np.ndarray, sigma: np.ndarray, runs: list[list], min_samples: int
) -> list[list]:
    """``A M A`` where the positions never leave one straight line (velocity blips from
    timestamp jitter or tracking noise, not motion) → one autoplay run (position-domain test);
    likewise a trailing ``A M`` that stays on the line of the autoplay before it (the recording
    ends there). The tube is ``max(BLIP_MAX_PX, BLIP_SIGMAS·σ)`` with σ the larger of the noise
    model and the stretch's own second-difference noise (low-texture content tracks noisier
    than the leading window suggests)."""
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs)):
            lab, a, b = runs[i]
            trailing = i == len(runs) - 1
            if lab != "M" or runs[i - 1][0] != "A" or (not trailing and runs[i + 1][0] != "A"):
                continue
            lo = max(runs[i - 1][1], int(np.searchsorted(t, t[a] - 1.0)))
            if trailing:
                f = (
                    kin.fit_constant(t[lo:a], x[lo:a], 1.0 / sigma[lo:a] ** 2)
                    if a - lo >= 3
                    else None
                )
                win = slice(lo, b)
            else:
                hi = min(runs[i + 1][2], int(np.searchsorted(t, t[b - 1] + 1.0)))
                f = kin.fit_constant(t[lo:hi], x[lo:hi], 1.0 / sigma[lo:hi] ** 2)
                win = slice(lo, hi)
            if f is None:
                continue
            line = f.params["x0"] + f.params["v"] * (t[a:b] - f.params["t_ref"])
            res = np.abs(x[a:b] - line)
            sig = max(float(np.median(sigma[a:b])), _local_sigma(x[win]))
            if float(res.max()) <= max(BLIP_MAX_PX, BLIP_SIGMAS * sig):
                if trailing:
                    runs[i - 1][2] = b
                    del runs[i]
                else:
                    runs[i - 1][2] = runs[i + 1][2]
                    del runs[i : i + 2]
                changed = True
                break
    del min_samples
    return runs


# --------------------------------------------------------------------------------------------
# Boundary helpers
# --------------------------------------------------------------------------------------------


def _onset_forward(c: _Ctx, model, i0: int, i1: int) -> int | None:  # type: ignore[no-untyped-def]
    """First index ``i`` in ``[i0, i1)`` where ``x`` leaves ``model`` for 2 samples."""
    i0, i1 = max(i0, 0), min(i1, len(c.t))
    if i1 - i0 < 2:
        return None
    thr = np.maximum(ONSET_SIGMAS * c.sigma[i0:i1], ONSET_MIN_PX)
    res = np.abs(c.x[i0:i1] - model(c.t[i0:i1])) > thr
    for j in range(len(res) - 1):
        if res[j] and res[j + 1]:
            return i0 + j
    return None


def _onset_backward(c: _Ctx, model, i0: int, i1: int) -> int | None:  # type: ignore[no-untyped-def]
    """Last index ``i`` in ``[i0, i1)`` (scanning backwards) that is off ``model`` together
    with its predecessor; i.e. the motion ends after sample ``i``."""
    i0, i1 = max(i0, 0), min(i1, len(c.t))
    if i1 - i0 < 2:
        return None
    res = np.abs(c.x[i0:i1] - model(c.t[i0:i1])) > ONSET_SIGMAS * c.sigma[i0:i1]
    for j in range(len(res) - 1, 0, -1):
        if res[j] and res[j - 1]:
            return i0 + j
    return None


def _idx_at(c: _Ctx, t: float) -> int:
    return int(np.clip(np.searchsorted(c.t, t), 0, len(c.t) - 1))


def _window(
    c: _Ctx, a: int, b: int, pre_s: float, post_s: float, lo: int, hi: int
) -> tuple[int, int]:
    """Sample window ``[a − pre, b + post)`` clamped to ``[lo, hi)``."""
    ta = c.t[a] - pre_s
    tb = c.t[min(b, len(c.t)) - 1] + post_s
    i0 = max(lo, int(np.searchsorted(c.t, ta)))
    i1 = min(hi, int(np.searchsorted(c.t, tb, side="right")))
    return i0, max(i1, i0 + 1)


# --------------------------------------------------------------------------------------------
# Ramps (decelerate / resume)
# --------------------------------------------------------------------------------------------


def _ramp_phase(
    c: _Ctx,
    kind: PhaseKind,
    i0: int,
    i1: int,
    v_a: float,
    v_b: float,
    t0_lo: float,
    t0_hi: float,
    *,
    interrupted: bool = False,
) -> _Ph | None:
    if (
        0 < i0 < len(c.t)
        and c.t[i0] - c.t[i0 - 1] > DUP_GAP_FRAMES * c.dt
        and (t0_lo <= c.t[i0] + 1e-9)
    ):
        # skipped duplicate frames before the window: the content was still (or moved less than
        # a grey level) there, so an eased onset may lie inside the gap
        t0_lo = max(float(c.t[i0 - 1]), t0_lo - RAMP_PRE_S)
        i0 -= 1
    t, x, w = c.t[i0:i1], c.x[i0:i1], c.w[i0:i1]
    if len(t) < 5:
        return None
    dur_lo = max(2.0 * c.dt, 0.02)
    fit = kin.fit_ramp(
        t, x, v_a, v_b, w, t0_range=(t0_lo, max(t0_hi, t0_lo)),
        dur_range=(dur_lo, RAMP_DUR_MAX_S), frame_dt=c.dt,
    )  # fmt: skip
    if fit is None:
        return None
    t0, dur = fit.params["t0"], fit.params["duration"]
    qt = c.qt(i0, i1)
    # label: ramp vs a straight constant-velocity line over the same window
    lin = kin.fit_constant(t, x, w)
    dbic = lin.bic() - fit.bic()
    ff = cconf.fit_factor(fit.progress_rmse)
    d_conf = (
        cconf.duration_confidence(c.dt, dur, ff, is_vfr=c.is_vfr, timestamps_estimated=c.ts_est)
        * qt
    )
    if fit.at_bound and not interrupted:
        d_conf *= 0.5
    if interrupted:
        # the ramp end was not observed: D is an extrapolation
        d_conf *= float(np.clip((float(c.t[i1 - 1]) - t0) / dur, 0.0, 1.0)) if dur > 0 else 0.0
    easing = kin.fit_easing(fit)
    assert easing is not None
    end = t0 + dur
    if interrupted:
        end = min(end, float(c.t[i1 - 1]))
    ph = _Ph(
        kind=kind,
        start=t0,
        end=end,
        start_fixed=True,
        end_fixed=not interrupted,
        fit=RampFit(
            from_px_s=round(v_a, 1),
            to_px_s=round(v_b, 1),
            duration_ms=MeasuredNumber(
                value=round(dur * 1000.0), confidence=cconf.ir_confidence(d_conf)
            ),
            easing=easing,
        ),
        raw=fit,
        v_start=v_a,
        v_end=float(v_a + (v_b - v_a) * _ease_at(fit, end)),
        interrupted=interrupted,
        label_conf=min(cconf.label_confidence(dbic, c.params), 0.4 + 0.6 * qt),
    )
    if interrupted:
        ph.notes.append(NOTE_CUT)
    return ph


def _ease_at(fit: kin.Fit, t: float) -> float:
    from app.pipeline.easing import eval_curve

    assert fit.curve is not None
    u = (t - fit.params["t0"]) / fit.params["duration"]
    return float(eval_curve(fit.curve.bezier, np.clip(u, 0.0, 1.0)))


# --------------------------------------------------------------------------------------------
# Drag + release split (§4.5 step 3)
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Release:
    r: int  # first tail sample
    model: str  # "exponential" | "tween" | "none"
    fit: kin.Fit | None
    cost: float
    second_cost: float


def _monotone_ok(c: _Ctx, i0: int, i1: int, v_inf: float, v0: float) -> bool:
    """|v − v_∞| decreases over the tail (≤ ``exp_max_monotone_violations`` rises). The
    smoothing half-window at both ends is skipped: the smoothed velocity there mixes in the
    drag before / the re-grab after."""
    h = c.params.velocity_half_window
    i0, i1 = i0 + h, i1 - h
    if i1 - i0 < 3:
        return True
    rel = np.abs(c.v[i0:i1] - v_inf)
    tol = max(MONO_TOL_ABS, MONO_TOL_REL * abs(v0 - v_inf))
    viol = int(np.count_nonzero(np.diff(rel) > tol))
    return viol <= c.params.exp_max_monotone_violations


def _tail_fits(c: _Ctx, r: int, e: int, allow_tween: bool) -> list[tuple[str, kin.Fit]]:
    p = c.params
    t, x = c.t[r:e], c.x[r:e]
    if len(t) == 0:
        return []
    # systematic tracking error grows with the distance travelled (a 0.2 % scale error is
    # 0.6 px after a 300 px fling — huge against the sub-0.1 px noise of the rest that
    # follows), so the tail's weights get a travel-proportional floor
    sys_err = SYS_FRAC * np.abs(x - x[0])
    w = 1.0 / (c.sigma[r:e] ** 2 + sys_err**2)
    out: list[tuple[str, kin.Fit]] = []
    v_infs = [0.0] + ([c.v_auto] if c.v_auto is not None else [])
    if e - r >= p.exp_min_samples:
        for v_inf in v_infs:
            f = kin.fit_exponential(t, x, float(t[0]), v_inf, w, tau_min=p.tau_min_s,
                                    tau_max=p.tau_max_s, cutoff=v_inf == 0.0)  # fmt: skip
            if (
                f is not None
                and math.isfinite(f.params["s_stop"])
                and f.params["v_stop"] > min(STOP_V_MAX, STOP_V_FRAC * abs(f.params["v0"]))
            ):
                # a "momentum stop" at high speed is just a drag that stopped abruptly
                f = kin.fit_exponential(t, x, float(t[0]), v_inf, w, tau_min=p.tau_min_s,
                                        tau_max=p.tau_max_s)  # fmt: skip
            if f is None or f.at_bound:
                continue
            v0 = f.params["v0"]
            if min(abs(v0 - v_inf), abs(v0)) < p.exp_v0_zero_mult * c.tol_z:
                continue  # no momentum to speak of (also: a creep toward autoplay is a resume)
            if not _monotone_ok(c, r, e, v_inf, v0):
                continue
            out.append(("exponential", f))
    if allow_tween and e - r >= 4:
        f = kin.fit_tween(t, x, float(t[0]), w, dur_min=max(2.0 * c.dt, 0.03),
                          dur_max=TWEEN_DUR_MAX_S, include_overshoot=True)  # fmt: skip
        # a tween's defining property is that it ends at rest after T: it must be observed
        # (an interrupted tail that is still moving at the re-grab is not a tween)
        if f is not None and not f.at_bound and float(t[0]) + f.params["duration"] <= float(t[-1]):
            out.append(("tween", f))
    return out


def _next_cost(c: _Ctx, b: int, e: int) -> float:
    """Cost of the extension samples ``[b, e)`` under the following steady phase."""
    if e - b < 1:
        return 0.0
    t, x, w = c.t[b:e], c.x[b:e], c.w[b:e]
    if e - b < 3:
        return float(np.sum(w)) * 0.0 + 2.0 * (e - b)
    f = kin.fit_constant(t, x, w)
    rest = kin.fit_rest(t, x, w)
    return min(f.bic(), rest.bic())


def _head_velocity(c: _Ctx, a: int, r: int) -> tuple[float | None, float]:
    """Drag velocity just before ``r``: local linear fit over the last ``HEAD_VEL`` samples
    (velocity estimate + variance). None with fewer than 3 samples."""
    m = min(r - a, HEAD_VEL)
    if m < 3:
        return None, math.inf
    lo = r - m
    tt = c.t[lo:r] - c.t[r - 1]
    w = c.w[lo:r]
    f = kin.fit_constant(c.t[lo:r], c.x[lo:r], w)
    tm = float(np.sum(w * tt) / np.sum(w))
    s_tt = float(np.sum(w * (tt - tm) ** 2))
    var = 1.0 / s_tt if s_tt > 0 else math.inf
    return float(f.params["v"]), var


def _head_cost(c: _Ctx, a: int, r: int) -> tuple[float, float | None, float]:
    """Cost of the drag samples ``[a, r)``: local quadratic over the last ``HEAD_LOCAL`` samples
    (χ² + 3 ln n) plus ``DRAG_COST_PER_SAMPLE`` for the earlier ones; plus the head velocity
    (:func:`_head_velocity`) for the release-velocity continuity term."""
    n = r - a
    if n <= 0:
        return 0.0, None, math.inf
    v_h, var = _head_velocity(c, a, r)
    m = min(n, HEAD_LOCAL)
    if m < 4:
        return DRAG_COST_PER_SAMPLE * n, v_h, var
    lo = r - m
    tt = c.t[lo:r] - c.t[r - 1]
    _, sse = kin._lstsq([np.ones(m), tt, tt * tt], c.x[lo:r], c.w[lo:r])
    return DRAG_COST_PER_SAMPLE * (n - m) + sse + 3.0 * math.log(m), v_h, var


def _release_velocity(c: _Ctx, model: str, f: kin.Fit, r: int) -> float:
    """Model velocity at the release instant (midpoint between samples r − 1 and r)."""
    t_m = 0.5 * (c.t[r - 1] + c.t[r]) if r > 0 else c.t[r]
    if model == "exponential":
        p = f.params
        return float(p["v_inf"] + (p["v0"] - p["v_inf"]) * math.exp((p["t_r"] - t_m) / p["tau"]))
    assert f.curve is not None
    from app.pipeline.easing import eval_curve

    du = 1e-3
    slope = float(eval_curve(f.curve.bezier, np.array([du]))[0]) / du
    return float(f.params["distance"] * slope / f.params["duration"])


def _continuity_cost(v_model: float, v_h: float | None, var_h: float) -> float:
    """χ² of the release-velocity mismatch between the drag head and the tail model."""
    if v_h is None or not math.isfinite(var_h):
        return 0.0
    tol2 = var_h + (CONT_REL * abs(v_h)) ** 2 + CONT_ABS**2
    return float((v_model - v_h) ** 2 / tol2)


def split_release(
    c: _Ctx, a: int, b: int, e: int, *, allow_tween: bool = True, t_min: float | None = None
) -> _Release:
    """Best release index / tail model for the sub-run ``[a, b)`` with tail extension ``e``.

    ``t_min``: earliest allowed release time (a drag lasts ≥ ``MIN_DRAG_FRAMES`` frames after
    its refined onset)."""
    n = b - a
    cost_none = _head_cost(c, a, b)[0] + _next_cost(c, b, e)
    best = _Release(r=b, model="none", fit=None, cost=cost_none, second_cost=math.inf)
    if n < 2:
        return best
    # candidates: once the drag is under way (|v| ≥ (1 − release_search_frac) · peak). A pure
    # "last 70 % of the run" bound misses releases when the inertia tail is longer than the drag.
    sp = np.abs(c.v[a:b])
    started = np.nonzero(sp >= (1.0 - c.params.release_search_frac) * float(sp.max()))[0]
    r_lo = a + (int(started[0]) if started.size else 0)
    cands = list(range(max(r_lo, a + 2), b))
    if t_min is not None:
        cands = [r for r in cands if c.t[r] >= t_min - 1e-9]
    if not cands:
        return best
    results: dict[int, list[tuple[str, kin.Fit, float]]] = {}

    def evaluate(r: int) -> None:
        if r in results:
            return
        fits = _tail_fits(c, r, e, allow_tween)
        head, v_h, var_h = _head_cost(c, a, r)
        out: list[tuple[str, kin.Fit, float]] = []
        for m, f in fits:
            v_m = _release_velocity(c, m, f, r)
            if v_h is not None and abs(v_m) > CONT_MAX_RATIO * abs(v_h) + CONT_ABS:
                continue  # content cannot speed up when the pointer lets go
            out.append((m, f, head + f.bic() + _continuity_cost(v_m, v_h, var_h)))
        results[r] = out

    coarse = cands[::2] if len(cands) > 12 else cands
    for r in coarse:
        evaluate(r)
    # refine around the coarse optimum of every model (the global coarse winner may hide a
    # model whose optimum fell between coarse candidates)
    by_model: dict[str, tuple[float, int]] = {}
    for r, lst in list(results.items()):
        for m, _, cost in lst:
            if m not in by_model or cost < by_model[m][0]:
                by_model[m] = (cost, r)
    for _, r_best in by_model.values():
        for r in range(max(r_best - 2, cands[0]), min(r_best + 3, b)):
            evaluate(r)
    all_c: list[tuple[float, int, str, kin.Fit]] = [
        (cost, r, m, f) for r, lst in results.items() for (m, f, cost) in lst
    ]
    all_c.append((cost_none, b, "none", None))  # type: ignore[arg-type]
    all_c.sort(key=lambda z: z[0])
    top = all_c[0]
    second = next((z[0] for z in all_c[1:] if z[2] != top[2]), math.inf)
    return _Release(r=top[1], model=top[2], fit=top[3], cost=top[0], second_cost=second)


def regrab_valleys(c: _Ctx, a: int, b: int) -> list[int]:
    """Indices of |v| valleys inside ``[a, b)`` that separate drag/release pairs."""
    s = np.abs(c.v[a:b])
    n = len(s)
    if n < 7:
        return []
    out: list[int] = []
    peak = s[0]
    valley_i, valley = 0, s[0]
    falling = False
    for i in range(1, n):
        if not falling:
            if s[i] > peak:
                peak = s[i]
            drop = peak - s[i]
            if drop >= max(REGRAB_MIN_DROP, REGRAB_REL * peak):
                falling = True
                valley_i, valley = i, s[i]
        else:
            if s[i] < valley:
                valley_i, valley = i, s[i]
            rise = s[i] - valley
            if rise >= max(REGRAB_MIN_DROP, REGRAB_REL * max(s[i], peak * 0.5)):
                out.append(a + valley_i)
                falling = False
                peak = s[i]
    # sign reversals with a valley near zero are valleys too (already caught by |v|)
    return [i for i in out if a + 2 <= i <= b - 3]


# --------------------------------------------------------------------------------------------
# Phase construction
# --------------------------------------------------------------------------------------------


def _in_band(c: _Ctx, i: int) -> bool:
    if c.v_auto is None:
        return False
    s = 1.0 if c.v_auto > 0 else -1.0
    return -c.tol_z <= s * c.v[i] <= abs(c.v_auto) + c.tol_a


def _exp_end(c: _Ctx, f: kin.Fit, hi_t: float) -> float:
    """End of an inertia phase: the momentum stop (if fitted) or where the decay comes within
    the zero / autoplay tolerance of ``v_∞``."""
    v0, tau, v_inf = f.params["v0"], f.params["tau"], f.params["v_inf"]
    tol = c.tol_z if v_inf == 0.0 else c.tol_a
    span = abs(v0 - v_inf)
    dt_end = tau * math.log(span / tol) if span > tol else 0.0
    dt_end = min(dt_end, f.params.get("s_stop", math.inf))
    return float(min(f.params["t_r"] + dt_end, hi_t))


def _observed_rest_end(c: _Ctx, f: kin.Fit, r: int, e: int) -> float | None:
    """Where an inertia tail is observed to come to rest (PLAN-continuous P2 definition).

    The inertia phase ends when the content is at rest within tracking noise: the first
    ``REST_END_SAMPLES`` consecutive samples of the tail extension ``[r, e)`` that stay within
    ``ONSET_SIGMAS`` σ of their mean define the rest position (the extension may already
    reach into the next resume), and the end lies between the last sample still off it (with
    its predecessor) and the next one. This is the same instant whether the motion decays into
    the noise or a library cuts it at a minimum speed (the synthetic truth cuts at 10 px/s), so
    resume delays measured from it compare like for like. None when the tail never rests.
    """
    n_rest = REST_END_SAMPLES
    if f.params["v_inf"] != 0.0 or e - r < n_rest + 2:
        return None
    for j in range(r + 2, e - n_rest + 1):
        seg = slice(j, j + n_rest)
        x0 = kin.fit_rest(c.t[seg], c.x[seg], c.w[seg]).params["x0"]
        if np.all(np.abs(c.x[seg] - x0) <= ONSET_SIGMAS * c.sigma[seg]):
            break
    else:
        return None

    def model(tt: np.ndarray) -> np.ndarray:
        return np.full(len(tt), x0)

    hi = j + n_rest
    i = _onset_backward(c, model, r, hi)
    if i is None or i + 1 >= hi:
        return None
    return 0.5 * float(c.t[i] + c.t[i + 1])


def _inertia_end(c: _Ctx, f: kin.Fit, r: int, e: int, t_rel: float) -> tuple[float, float, bool]:
    """``(end, stop speed, cut)`` of a final inertia phase decaying to rest (P2 definition).

    The decay ends where its speed falls to :data:`INERTIA_END_PX_S` (the convention the
    synthetic truth uses as well: a pure exponential never ends), unless the content is
    observed at rest earlier while the fitted decay is still ≥ :data:`CUT_MIN_PX_S` (a library
    dropping the momentum at a minimum speed): then that observed rest is the end and its speed
    the stop speed (``cut``). Resume delays are measured from this end, and the JS driver stops
    the momentum at the reported speed, so both run on the same clock.
    """
    t_r, v0, tau = f.params["t_r"], f.params["v0"], f.params["tau"]

    def speed(tt: float) -> float:
        return abs(float(kin.exp_velocity(np.array([tt - t_r]), v0, tau, 0.0)[0]))

    t_conv = (
        t_r + tau * math.log(abs(v0) / INERTIA_END_PX_S) if abs(v0) > INERTIA_END_PX_S else t_rel
    )
    rest = _observed_rest_end(c, f, r, e)
    if rest is not None and rest < t_conv and speed(rest) >= CUT_MIN_PX_S:
        if _cut_evidence(c, f, rest, t_conv) >= CUT_MIN_DCHI2:
            return rest, speed(rest), True
    return t_conv, INERTIA_END_PX_S, False


def _cut_evidence(c: _Ctx, f: kin.Fit, t_cut: float, t_conv: float) -> float:
    """Δχ² of "the content stops at ``t_cut``" over "it keeps decaying to the 10 px/s end",
    on the samples between them (+2 frames). A slow tail hidden in the position noise must not
    pass for a cut: both models get their own offset, only their shapes compete."""
    j0 = int(np.searchsorted(c.t, t_cut))
    j1 = int(np.searchsorted(c.t, t_conv + 2.0 * c.dt, side="right"))
    if j1 - j0 < 3:
        return 0.0
    t, x, w = c.t[j0:j1], c.x[j0:j1], c.w[j0:j1]
    p = f.params
    decay = kin.exp_position(t - p["t_r"], 0.0, p["v0"], p["tau"], 0.0)

    def sse(model: np.ndarray) -> float:
        off = float(np.sum(w * (x - model)) / np.sum(w))
        return float(np.sum(w * (x - model - off) ** 2))

    return sse(decay) - sse(np.zeros(len(t)))


def _model_velocity(model: str, f: kin.Fit, t: np.ndarray) -> np.ndarray:
    p = f.params
    if model == "exponential":
        return kin.exp_velocity(
            t - p["t_r"], p["v0"], p["tau"], p["v_inf"], p.get("s_stop", math.inf)
        )
    from app.pipeline.easing import eval_curve

    assert f.curve is not None
    du = 1e-3
    u = np.clip((t - p["t_r"]) / p["duration"], 0.0, 1.0 - du)
    slope = (eval_curve(f.curve.bezier, u + du) - eval_curve(f.curve.bezier, u)) / du
    return p["distance"] * slope / p["duration"]


def _release_instant(c: _Ctx, a: int, r: int, model: str, f: kin.Fit) -> tuple[float, float]:
    """Release instant + velocity: where the drag head's local-quadratic velocity meets the
    tail model's velocity, searched between samples ``r − 2`` and ``r + 3`` (falls back to the
    midpoint between ``r − 1`` and ``r``). Removes the ±½-frame bias of ``v0`` when the split
    lands one sample off."""
    t_mid = 0.5 * (c.t[r - 1] + c.t[r]) if r > a else float(c.t[r])
    v_head, _ = _head_velocity(c, a, r)
    if v_head is None:
        return float(t_mid), float(_model_velocity(model, f, np.array([t_mid]))[0])
    t_lo = float(c.t[max(r - 2, a)])
    t_hi = float(c.t[min(r + 3, len(c.t) - 1)])
    grid = np.linspace(t_lo, t_hi, 61)
    v_h = np.full(len(grid), v_head)
    v_m = _model_velocity(model, f, grid)
    d = np.abs(v_m) - np.abs(v_h)  # tail decelerates: |v_model| falls below |v_head| after
    cross = np.nonzero((d[:-1] > 0) & (d[1:] <= 0))[0]
    if cross.size:
        j = int(cross[-1])
        w = d[j] / (d[j] - d[j + 1]) if d[j] != d[j + 1] else 0.5
        t_rel = float(grid[j] + w * (grid[j + 1] - grid[j]))
    else:
        t_rel = float(t_mid)
    return t_rel, float(_model_velocity(model, f, np.array([t_rel]))[0])


def _refine_tween_start(c: _Ctx, a: int, r: int, e: int, f: kin.Fit) -> tuple[float, kin.Fit]:
    """Continuous tween start in ``[t[r−1], t[r]]`` (golden search on the tail SSE).

    A tween is defined by its start, not by velocity continuity: a snap's initial speed can
    exceed the drag speed, so the head/tail velocity crossing (:func:`_release_instant`) would
    land late. The split pins the start to sample ``r``; the true start lies before it.
    """
    lo = float(c.t[max(r - 1, a)])
    hi = float(c.t[r])
    if hi - lo <= 1e-6:
        return hi, f
    t, x = c.t[r:e], c.x[r:e]
    w = 1.0 / (c.sigma[r:e] ** 2 + (SYS_FRAC * np.abs(x - x[0])) ** 2)
    names = (f.curve.name,) if f.curve is not None else kin.TWEEN_CURVES
    fits: dict[float, kin.Fit | None] = {}

    def fit_at(t0: float) -> kin.Fit | None:
        if t0 not in fits:
            fits[t0] = kin.fit_tween(t, x, t0, w, dur_min=max(2.0 * c.dt, 0.03),
                                     dur_max=TWEEN_DUR_MAX_S, curve_names=names,
                                     include_overshoot=False)  # fmt: skip
        return fits[t0]

    def cost(t0: float) -> float:
        ft = fit_at(t0)
        return ft.sse if ft is not None else math.inf

    best_t = kin._golden(cost, lo, hi, TWEEN_START_ITERS)  # noqa: SLF001
    cands = [(cost(tt), tt) for tt in (lo, best_t, hi)]
    sse, t0 = min(cands)
    ft = fit_at(t0)
    if ft is None or not math.isfinite(sse) or sse > f.sse:
        return hi, f
    return t0, ft


def _release_phases(
    c: _Ctx,
    a: int,
    b: int,
    e: int,
    *,
    last: bool,
    next_lab: str | None,
    t_min: float | None = None,
) -> tuple[list[_Ph], _Release]:
    """Drag (+ release) phases for sub-run ``[a, b)``; ``e`` = tail extension end."""
    p = c.params
    rel = split_release(c, a, b, e, t_min=t_min)
    phases: list[_Ph] = []
    qt_all = c.qt(a, max(b, a + 1))
    if rel.model == "tween" and rel.fit is not None:
        if rel.fit.params["duration"] <= (p.stop_max_frames + 1) * c.dt * 1.05:
            # a "tween" over <= stop_max_frames + 1 frames is an abrupt stop (§4.5 step 4)
            rel = _Release(r=b, model="none", fit=None, cost=rel.cost, second_cost=rel.second_cost)
    if rel.model == "none":
        drag = _Ph("drag", float(c.t[a]), float(c.t[min(b, len(c.t) - 1)]),
                   label_conf=0.4 + 0.5 * qt_all)  # fmt: skip
        phases.append(drag)
        # abrupt stop into rest?
        if last and next_lab == "Z" and b < len(c.t):
            stop = _stop_phase(c, a, b, e)
            if stop is not None:
                drag.end = stop.start
                drag.end_fixed = True
                phases.append(stop)
        return phases, rel

    f = rel.fit
    assert f is not None
    if rel.model == "tween":
        t_rel, f = _refine_tween_start(c, a, rel.r, e, f)
        v_rel = float(_model_velocity("tween", f, np.array([t_rel]))[0])
    else:
        t_rel, v_rel = _release_instant(c, a, rel.r, rel.model, f)
    if rel.r - a >= 1:
        drag = _Ph("drag", float(c.t[a]), t_rel, end_fixed=True, label_conf=0.4 + 0.5 * qt_all)
        if rel.model == "tween":
            # a tween's start speed is not the pointer's: the drag ends at its own velocity
            drag.v_end = _head_velocity(c, a, rel.r)[0]
        phases.append(drag)
    qt = c.qt(rel.r, e)
    sep = cconf.label_confidence(rel.second_cost - rel.cost, p)
    hi_t = float(c.t[min(e, len(c.t)) - 1])
    if rel.model == "exponential":
        tau, v_inf = f.params["tau"], f.params["v_inf"]
        rel_se = f.se.get("tau", math.inf) / tau if tau > 0 else math.inf
        n_tau = int(np.count_nonzero(c.t[rel.r : e] <= t_rel + 3.0 * tau))
        tau_c = cconf.tau_confidence(qt, rel_se, n_tau, p)
        v0_c = cconf.release_confidence(qt, c.dt, tau, p)
        stop_px: float | None = None
        cut = False
        if last and v_inf == 0.0:
            end_t, stop_px, cut = _inertia_end(c, f, rel.r, e, t_rel)
            end_t = min(max(end_t, t_rel + c.dt), hi_t)
        else:
            end_t = _exp_end(c, f, hi_t) if last else float(c.t[min(b, len(c.t) - 1)])
        ph = _Ph(
            kind="inertia",
            start=t_rel,
            end=max(end_t, t_rel + c.dt),
            start_fixed=True,
            end_fixed=last,
            fit=ExponentialFit(
                tau_ms=MeasuredNumber(
                    value=round(tau * 1000.0), confidence=cconf.ir_confidence(tau_c, p.cap_tau)
                ),
                v0_px_s=MeasuredNumber(
                    value=round(v_rel, 1), confidence=cconf.ir_confidence(v0_c, p.cap_release)
                ),
                v_inf_px_s=round(v_inf, 1),
            ),  # fmt: skip
            raw=f,
            v_start=v_rel,
            interrupted=not last,
            label_conf=min(sep, 0.4 + 0.6 * qt),
        )
        if not last:
            ph.notes.append(NOTE_REGRAB)
        if stop_px is not None:
            ph.stop_px_s = stop_px
            ph.v_end = float(np.sign(f.params["v0"]) * stop_px)
            if cut:
                ph.notes.append(NOTE_MOMENTUM_STOP)
        else:
            ph.v_end = float(_model_velocity("exponential", f, np.array([ph.end]))[0])
        if ph.stop_px_s is not None and isinstance(ph.fit, ExponentialFit):
            ph.fit = ph.fit.model_copy(update={"stop_px_s": round(ph.stop_px_s, 1)})
        phases.append(ph)
    else:  # tween
        dist, dur = f.params["distance"], f.params["duration"]
        ff = cconf.fit_factor(f.progress_rmse)
        d_conf = (
            cconf.duration_confidence(c.dt, dur, ff, is_vfr=c.is_vfr, timestamps_estimated=c.ts_est)
            * qt
        )
        easing = kin.fit_easing(f)
        assert easing is not None
        t_r = float(f.params["t_r"])
        ph = _Ph(
            kind="inertia",
            start=t_rel,
            end=min(t_r + dur, hi_t) if last else float(c.t[min(b, len(c.t) - 1)]),
            start_fixed=True,
            end_fixed=last,
            fit=TweenFit(
                distance_px=MeasuredNumber(
                    value=round(dist, 1), confidence=cconf.ir_confidence(qt * ff)
                ),
                duration_ms=MeasuredNumber(
                    value=round(dur * 1000.0), confidence=cconf.ir_confidence(d_conf)
                ),
                easing=easing,
            ),  # fmt: skip
            raw=f,
            v_start=v_rel,
            v_end=0.0 if last else None,
            interrupted=not last,
            label_conf=min(sep, 0.4 + 0.6 * qt),
        )
        if not last:
            ph.notes.append(NOTE_REGRAB)
        phases.append(ph)
    return phases, rel


def _stop_phase(c: _Ctx, a: int, b: int, e: int) -> _Ph | None:
    """Abrupt stop at the end of ``[a, b)`` into the rest run starting at ``b`` (§4.5 step 4)."""
    p = c.params
    rest_hi = min(e, len(c.t))
    if rest_hi - b < 2:
        return None
    rest = kin.fit_rest(c.t[b:rest_hi], c.x[b:rest_hi], c.w[b:rest_hi])

    def model(tt: np.ndarray) -> np.ndarray:
        return np.full(len(tt), rest.params["x0"])

    last_off = _onset_backward(c, model, a, b + 2)
    end_i = (last_off + 1) if last_off is not None else b
    end_i = min(max(end_i, a + 1), len(c.t) - 1)
    # fast step into the stop: last sample before end_i with step |v| > stop_from_speed
    sv = np.abs(np.diff(c.x[a : end_i + 1]) / np.maximum(np.diff(c.t[a : end_i + 1]), 1e-9))
    fast = np.nonzero(sv > p.stop_from_speed)[0]
    if fast.size == 0:
        return None
    start_i = a + int(fast[-1])  # step start_i -> start_i+1 is still fast
    if end_i - start_i > p.stop_max_frames + 1:
        return None
    if start_i <= a:
        return None
    v_from = float((c.x[start_i] - c.x[start_i - 1]) / max(c.t[start_i] - c.t[start_i - 1], 1e-9))
    return _Ph(
        kind="stop",
        start=float(c.t[start_i]),
        end=float(c.t[end_i]),
        start_fixed=True,
        end_fixed=True,
        v_start=v_from,
        v_end=0.0,
        label_conf=STOP_LABEL_CONF * (0.5 + 0.5 * c.qt(start_i, end_i + 1)),
        notes=[NOTE_STOP],
    )


def _drag_start(c: _Ctx, prev: _Ph | None, a: int, lo: int) -> float | None:
    """Refine the drag start from the previous phase's model (position-domain onset)."""
    if prev is None:
        return None
    i0 = max(lo, a - 6)
    if prev.kind == "autoplay" and isinstance(prev.raw, kin.Fit):
        f = prev.raw
        # the autoplay line just before the run (the whole run's line drifts by a few tenths of
        # a px when the on-screen speed varies slowly, e.g. cards that scale with position,
        # and the onset test is in units of a sub-0.1 px noise σ)
        j0 = max(_idx_at(c, prev.start), int(np.searchsorted(c.t, c.t[i0] - ONSET_LINE_S)))
        if i0 - j0 >= 5:
            f = kin.fit_constant(c.t[j0:i0], c.x[j0:i0], c.w[j0:i0])

        def model(tt: np.ndarray) -> np.ndarray:
            return f.params["x0"] + f.params["v"] * (tt - f.params["t_ref"])

    elif prev.kind == "paused" and isinstance(prev.raw, kin.Fit):
        x0 = prev.raw.params["x0"]

        def model(tt: np.ndarray) -> np.ndarray:
            return np.full(len(tt), x0)

    elif prev.kind == "decelerate" and isinstance(prev.raw, kin.Fit):
        f = prev.raw
        assert f.curve is not None

        def model(tt: np.ndarray) -> np.ndarray:
            return kin.ramp_position(tt, f.params["x0"], f.params["t0"], f.params["duration"],
                                     f.params["v_a"], f.params["v_b"], f.curve.bezier)  # type: ignore[union-attr]  # fmt: skip

    else:
        return None
    on = _onset_forward(c, model, i0, a + ONSET_AHEAD)
    if on is None:
        # a slowly accelerating pointer (or timestamp jitter widening the smoothed run) can put
        # the run start well before the departure: look further ahead before giving up
        on = _onset_forward(c, model, a + ONSET_AHEAD - 1, a + 2 * ONSET_AHEAD)
    if on is None:
        return None
    return float(c.t[max(on - 1, lo)])


def _steady_phase(c: _Ctx, kind: PhaseKind, a: int, b: int) -> _Ph:
    t, x, w = c.t[a:b], c.x[a:b], c.w[a:b]
    qt = c.qt(a, b)
    if kind == "autoplay":
        f = kin.fit_constant(t, x, w)
        v = f.params["v"]
        rel_se = f.se["v"] / abs(v) if v else math.inf
        sc = cconf.speed_confidence(qt, float(t[-1] - t[0]), rel_se, c.params)
        return _Ph(
            kind="autoplay", start=float(t[0]), end=float(t[-1]),
            fit=ConstantFit(velocity_px_s=MeasuredNumber(
                value=round(v, 1), confidence=cconf.ir_confidence(sc, c.params.cap_speed))),
            raw=f, v_start=v, v_end=v, label_conf=max(sc, 0.4 * qt),
        )  # fmt: skip
    f = kin.fit_rest(t, x, w)
    dur = float(t[-1] - t[0])
    return _Ph(kind="paused", start=float(t[0]), end=float(t[-1]), raw=f, v_start=0.0,
               v_end=0.0, label_conf=qt * float(np.clip(dur / 0.2, 0.3, 1.0)))  # fmt: skip


def _m_run(c: _Ctx, runs: list[list], i: int, prev: _Ph | None) -> list[_Ph]:
    """Phases for the ``M`` run ``runs[i]``."""
    _, a, b = runs[i]
    n_all = len(c.t)
    prev_lab = runs[i - 1][0] if i > 0 else None
    next_lab = runs[i + 1][0] if i + 1 < len(runs) else None
    next_b = runs[i + 1][2] if i + 1 < len(runs) else n_all
    prev_a = runs[i - 1][1] if i > 0 else 0
    va = c.v_auto
    out: list[_Ph] = []

    def next_autoplay_v() -> float:
        if next_lab == "A":
            nb = min(next_b, a + (b - a) + int(1.0 / max(c.dt, 1e-3)))
            f = kin.fit_constant(c.t[b:nb], c.x[b:nb], c.w[b:nb]) if nb - b >= 3 else None
            if f is not None:
                return float(f.params["v"])
        return float(va or 0.0)

    band = [_in_band(c, j) for j in range(a, b)]
    ramp_like = False
    tol_n = c.tol_a
    if va is not None and band:
        sv_ = (1.0 if va > 0 else -1.0) * c.v[a:b]
        # the run's own velocity noise (smoothed over ±half_window samples): low-texture content
        # can track noisier than the leading autoplay the band was sized on
        hw = max(c.params.velocity_half_window, 1)
        tol_n = max(c.tol_a, 3.0 * _local_sigma(c.x[a:b]) / (hw * max(c.dt, 1e-3)))
        in_noisy_band = (sv_ >= -c.tol_z - (tol_n - c.tol_a)) & (sv_ <= abs(va) + tol_n)
        ramp_like = all(band) or (
            float(np.mean(in_noisy_band)) >= RAMP_BAND_FRAC
            and bool(np.all(sv_ >= -abs(va) - c.tol_z))
            and bool(np.all(sv_ <= RAMP_MAX_FACTOR * abs(va) + tol_n))
        )
    if va is not None and ramp_like:
        s = 1.0 if va > 0 else -1.0
        sv = s * c.v[a:b]
        if prev_lab == "A" and next_lab == "A":
            if float(np.min(sv)) > abs(va) - DIP_MIN_TOLS * c.tol_a:
                return [_steady_phase(c, "autoplay", a, b)]  # noise dip: still autoplay
            m = a + int(np.argmin(sv))
            i0, _ = _window(c, a, m, RAMP_PRE_S, 0.0, prev_a, n_all)
            dec = _ramp_phase(c, "decelerate", i0, m + 1, va, 0.0, float(c.t[i0]), float(c.t[m]),
                              interrupted=True)  # fmt: skip
            _, i1 = _window(c, m, b, 0.0, RAMP_POST_S, 0, next_b)
            res = _ramp_phase(c, "resume", m, i1, 0.0, next_autoplay_v(), float(c.t[m]),
                              float(c.t[b - 1]))  # fmt: skip
            if dec is not None:
                dec.notes = ["autoplay slowed without stopping, then resumed"]
                out.append(dec)
            if res is not None:
                out.append(res)
            return out
        third = max(len(sv) // 3, 1)
        decreasing = float(np.mean(sv[-third:])) < float(np.mean(sv[:third]))
        if next_lab is None and float(np.min(sv)) > abs(va) - DIP_MIN_TOLS * c.tol_a:
            return [_steady_phase(c, "autoplay", a, b)]  # end-of-recording wobble, not a slowdown
        if decreasing or next_lab == "Z":
            i0, i1 = _window(c, a, b, RAMP_PRE_S, RAMP_POST_S, prev_a, next_b)
            dec = _ramp_phase(c, "decelerate", i0, i1, va, 0.0, float(c.t[i0]),
                              float(c.t[b - 1]))  # fmt: skip
            return [dec] if dec is not None else []
        # a noisy run can stay "M" long after the speed is back at autoplay (single samples
        # spiking out of the band): end the ramp where the speed reaches the band for good
        tail: _Ph | None = None
        w_n = max(int(round(MIN_AUTOPLAY_RUN_S / max(c.dt, 1e-3))), 3)
        if b - a > 2 * w_n:
            reached = [
                k for k in range(a, b - w_n)
                if float(np.mean(sv[k - a :])) >= abs(va) - tol_n
                and float(np.mean(sv[k - a : k - a + w_n])) >= abs(va) - tol_n
            ]  # fmt: skip
            if reached and b - reached[0] >= w_n:
                tail = _steady_phase(c, "autoplay", reached[0], b)
                b = reached[0]
        i0, i1 = _window(c, a, b, RAMP_PRE_S, RAMP_POST_S, prev_a, next_b)
        res = _ramp_phase(c, "resume", i0, i1, 0.0, next_autoplay_v(), float(c.t[i0]),
                          float(c.t[b - 1]))  # fmt: skip
        out = [res] if res is not None else []
        return out + ([tail] if tail is not None else [])

    # ---- drag region with optional decelerate prefix / resume suffix ----------------------
    p0, p1 = a, b
    if va is not None and prev_lab == "A":
        s = 1.0 if va > 0 else -1.0
        run_min = math.inf
        rise = 1.5 * c.tol_a
        j = a
        while j < b and _in_band(c, j) and s * c.v[j] <= run_min + rise:
            run_min = min(run_min, s * c.v[j])
            j += 1
        off_line = True
        if prev is not None and prev.kind == "autoplay" and isinstance(prev.raw, kin.Fit):
            f_a = prev.raw
            line = f_a.params["x0"] + f_a.params["v"] * (c.t[a:j] - f_a.params["t_ref"])
            # a real slowdown leaves the autoplay line; tracking wobble (as noisy as the
            # autoplay run before it) stays on it
            i_a = max(_idx_at(c, prev.start), a - int(round(1.0 / max(c.dt, 1e-3))))
            sig = np.maximum(c.sigma[a:j], _local_sigma(c.x[i_a:a]))
            off_line = bool(np.any(np.abs(c.x[a:j] - line) > ONSET_SIGMAS * sig))
        if (
            off_line
            and j - a >= c.params.min_run_samples
            and s * c.v[max(j - 1, a)] < abs(va) - c.tol_a
        ):
            i0, _ = _window(c, a, j, RAMP_PRE_S, 0.0, prev_a, n_all)
            dec = _ramp_phase(c, "decelerate", i0, j, va, 0.0, float(c.t[i0]),
                              float(c.t[j - 1]), interrupted=True)  # fmt: skip
            if dec is not None:
                out.append(dec)
                prev = dec
                p0 = j
    suffix: _Ph | None = None
    if va is not None and next_lab == "A":
        s = 1.0 if va > 0 else -1.0
        j = b - 1
        while j > p0 and _in_band(c, j) and s * c.v[j - 1] <= s * c.v[j] + c.tol_a:
            j -= 1
        if b - j >= c.params.min_run_samples and s * c.v[j] <= c.tol_z * 2.0:
            _, i1 = _window(c, j, b, 0.0, RAMP_POST_S, 0, next_b)
            suffix = _ramp_phase(c, "resume", max(j - 3, p0), i1, 0.0, next_autoplay_v(),
                                 float(c.t[max(j - 3, p0)]), float(c.t[b - 1]))  # fmt: skip
            if suffix is not None:
                p1 = j
    if p1 - p0 < 2:
        if suffix is not None:
            out.append(suffix)
        return out

    # drag start refinement
    start_t = _drag_start(c, prev, p0, prev_a)
    valleys = regrab_valleys(c, p0, p1)
    bounds = [p0, *valleys, p1]
    ext_end = p1
    if suffix is None and next_lab in ("Z", "A"):
        ext_end = min(next_b, int(np.searchsorted(c.t, c.t[p1 - 1] + TAIL_EXT_MAX_S)))
        ext_end = max(ext_end, p1)
    for k in range(len(bounds) - 1):
        sa, sb = bounds[k], bounds[k + 1]
        last = k == len(bounds) - 2
        # an interrupted tail stops a smoothing half-window before the re-grab valley: the
        # samples right at the valley already carry the new drag
        e = ext_end if last else max(sb - c.params.velocity_half_window, sa + 1)
        t_min = None
        if k == 0 and start_t is not None:
            t_min = start_t + MIN_DRAG_FRAMES * c.dt  # no one-frame "drag" after the onset
        phs, _ = _release_phases(c, sa, sb, e, last=last, next_lab=next_lab if last else "M",
                                 t_min=t_min)  # fmt: skip
        if k == 0 and phs and phs[0].kind == "drag" and start_t is not None:
            phs[0].start = min(start_t, phs[0].end - 1e-3)
            phs[0].start_fixed = True
        out.extend(phs)
    if suffix is not None:
        out.append(suffix)
    return out


def _reconcile(c: _Ctx, phs: list[_Ph]) -> list[_Ph]:
    """Make phases contiguous over ``[t0, tN]``; merge same-kind neighbours; drop empties."""
    if not phs:
        return phs
    phs.sort(key=lambda p: p.start)
    for k in range(len(phs) - 1):
        a, b = phs[k], phs[k + 1]
        if a.end_fixed and not b.start_fixed:
            b.start = a.end
        elif b.start_fixed and not a.end_fixed:
            a.end = b.start
        elif a.end_fixed and b.start_fixed:
            m = 0.5 * (a.end + b.start)
            a.end = b.start = m
        else:
            a.end = b.start
    phs[0].start = float(c.t[0])
    phs[-1].end = float(c.t[-1])
    # ensure monotone boundaries (a fitted boundary may overrun a short neighbour)
    for k in range(1, len(phs)):
        if phs[k].start < phs[k - 1].start:
            phs[k].start = phs[k - 1].start
        phs[k - 1].end = phs[k].start
    # a rest shorter than MIN_REST_FRAMES frames between two motions is a re-grab, not a pause;
    # a slowdown cut by a press within MIN_DRAG_FRAMES frames is the press itself
    for k in range(1, len(phs) - 1):
        ph = phs[k]
        dur = ph.end - ph.start
        short_rest = ph.kind == "paused" and dur < MIN_REST_FRAMES * c.dt - 1e-9
        short_decel = (
            ph.kind == "decelerate" and ph.interrupted and dur < PRESS_DECEL_FRAMES * c.dt - 1e-9
        )
        if short_rest or short_decel:
            phs[k - 1].end = ph.end
            ph.end = ph.start
    out: list[_Ph] = []
    for ph in phs:
        if ph.end - ph.start <= 1e-6:
            continue
        if out and out[-1].kind == ph.kind and ph.kind in ("drag", "paused", "autoplay"):
            prev = out[-1]
            prev.end = ph.end
            if ph.kind == "autoplay":
                i0, i1 = _idx_at(c, prev.start), _idx_at(c, prev.end) + 1
                merged = _steady_phase(c, "autoplay", i0, max(i1, i0 + 3))
                merged.start, merged.end = prev.start, prev.end
                out[-1] = merged
            continue
        out.append(ph)
    return out


# --------------------------------------------------------------------------------------------
# Snap evidence, cursor fusion, finishing
# --------------------------------------------------------------------------------------------


def _merge_blend(c: _Ctx, phs: list[_Ph]) -> list[_Ph]:
    """Momentum that blends into autoplay through zero (P2b): a fling *against* the autoplay
    direction decays towards ``v_auto``, so its velocity crosses zero on the way and the coarse
    labels split it into inertia → (short) paused → resume → autoplay.

    When one exponential with ``v_∞ = v_auto``, refitted from the inertia's release to the end of
    the resume, explains those samples (reduced χ² ≤ :data:`BLEND_CHI2`, τ within
    :data:`BLEND_TAU_REL` of the inertia's own) and the standstill is shorter than
    :data:`BLEND_MAX_REST_S`, it is one inertia phase ending where it comes within the autoplay
    band (``_exp_end``); the following autoplay starts there."""
    va = c.v_auto
    if va is None:
        return phs
    out: list[_Ph] = []
    k = 0
    while k < len(phs):
        ph = phs[k]
        nxt = phs[k + 1 : k + 3]
        j = k + 1
        if nxt and nxt[0].kind == "paused":
            j += 1
        merged = None
        if (
            ph.kind == "inertia"
            and isinstance(ph.raw, kin.Fit)
            and isinstance(ph.fit, ExponentialFit)
            and abs(ph.raw.params.get("v_inf", 0.0) - va) <= 1e-6
            and j < len(phs)
            and phs[j].kind == "resume"
            and (j == k + 1 or phs[k + 1].end - phs[k + 1].start <= BLEND_MAX_REST_S)
        ):
            merged = _blend_fit(c, ph, phs[j].end)
        if merged is None:
            out.append(ph)
            k += 1
            continue
        out.append(merged)
        k = j + 1
        if k < len(phs) and phs[k].kind == "autoplay":
            i0, i1 = _idx_at(c, merged.end), _idx_at(c, phs[k].end) + 1
            if i1 - i0 >= 3:
                ap = _steady_phase(c, "autoplay", i0, i1)
                ap.start, ap.end = merged.end, phs[k].end
                out.append(ap)
                k += 1
            else:
                merged.end = phs[k].end
                k += 1
    return out


def _blend_fit(c: _Ctx, ph: _Ph, until: float) -> _Ph | None:
    """``ph`` (an inertia towards ``v_auto``) refitted up to ``until`` (:func:`_merge_blend`)."""
    assert ph.raw is not None and isinstance(ph.fit, ExponentialFit) and c.v_auto is not None
    p = c.params
    t_r = float(ph.raw.params["t_r"])
    i0 = max(_idx_at(c, ph.start), 0)
    i1 = min(_idx_at(c, until) + 1, len(c.t))
    if i1 - i0 < p.exp_min_samples:
        return None
    t, x = c.t[i0:i1], c.x[i0:i1]
    sys_err = SYS_FRAC * np.abs(x - x[0])
    w = 1.0 / (c.sigma[i0:i1] ** 2 + sys_err**2)
    f = kin.fit_exponential(t, x, t_r, c.v_auto, w, tau_min=p.tau_min_s, tau_max=p.tau_max_s)
    if f is None or f.at_bound:
        return None
    tau0 = float(ph.raw.params["tau"])
    tau = float(f.params["tau"])
    if f.sse / max(f.n - f.k, 1) > BLEND_CHI2 or abs(tau - tau0) > BLEND_TAU_REL * tau0:
        return None
    end = _exp_end(c, f, float(c.t[-1]))
    v_end = float(f.params["v_inf"] + (f.params["v0"] - f.params["v_inf"])
                  * math.exp(-(end - t_r) / tau))  # fmt: skip
    fit = ph.fit.model_copy(update={
        "tau_ms": MeasuredNumber(value=round(tau * 1000.0), confidence=ph.fit.tau_ms.confidence),
    })  # fmt: skip
    return _Ph(
        kind="inertia", start=ph.start, end=end, start_fixed=ph.start_fixed, end_fixed=True,
        fit=fit, raw=f, v_start=ph.v_start, v_end=v_end, interrupted=False,
        label_conf=ph.label_conf, notes=[*ph.notes, NOTE_BLEND], evidence=ph.evidence,
    )  # fmt: skip


def _merge_inconsistent_inertia(phs: list[_Ph]) -> list[_Ph]:
    """One scroller has one momentum model: an exponential "inertia" whose τ is off the
    scroller's τ (weighted median over the instances) by more than ``TAU_RATIO`` is the pointer
    slowing down (or stopping) before letting go, and is merged into the drag before it
    (consecutive drags merge too). Needs ≥ 3 exponential instances so a reference exists."""
    exps = [ph for ph in phs if ph.kind == "inertia" and isinstance(ph.fit, ExponentialFit)]
    if len(exps) < 3:
        return phs
    taus = [ph.fit.tau_ms.value for ph in exps]  # type: ignore[union-attr]
    confs = [max(ph.fit.tau_ms.confidence.value, 1e-3) for ph in exps]  # type: ignore[union-attr]
    ref = kin.weighted_median(taus, confs)
    out: list[_Ph] = []
    for ph in phs:
        prev = out[-1] if out else None
        odd = (
            ph.kind == "inertia"
            and isinstance(ph.fit, ExponentialFit)
            and not (ref / TAU_RATIO <= ph.fit.tau_ms.value <= ref * TAU_RATIO)
        )
        if prev is not None and prev.kind == "drag" and (odd or ph.kind == "drag"):
            prev.end, prev.end_fixed = ph.end, ph.end_fixed
            prev.v_end = ph.v_end
            if odd and NOTE_POINTER_DECEL not in prev.notes:
                prev.notes.append(NOTE_POINTER_DECEL)
            continue
        out.append(ph)
    return out


#: ``_unify_momentum_target``: an interrupted decay refitted towards the autoplay velocity is
#: accepted when its BIC is at most this much worse than the decay to zero.
UNIFY_MAX_DBIC = 6.0


def _unify_momentum_target(c: _Ctx, phs: list[_Ph]) -> list[_Ph]:
    """One scroller, one momentum law: when some release visibly decays towards the autoplay
    velocity, a momentum that was re-grabbed before it settled (a short stretch that cannot tell
    "towards autoplay" from "towards zero with a longer τ") is refitted towards the autoplay
    velocity and keeps that fit unless it is clearly worse (:data:`UNIFY_MAX_DBIC`). Decays
    observed to come to rest are left alone (a scroller may do both, e.g. C13)."""
    va = c.v_auto
    if va is None:
        return phs
    exps = [
        ph
        for ph in phs
        if ph.kind == "inertia"
        and isinstance(ph.fit, ExponentialFit)
        and isinstance(ph.raw, kin.Fit)
    ]
    towards = [ph.raw.params.get("v_inf", 0.0) for ph in exps if ph.raw is not None]
    if not any(abs(v - va) <= 1e-6 for v in towards):
        return phs
    p = c.params
    for ph in exps:
        raw = ph.raw
        assert raw is not None and isinstance(ph.fit, ExponentialFit)
        if not ph.interrupted or raw.params.get("v_inf", 0.0) != 0.0:
            continue
        i0 = max(_idx_at(c, ph.start), 0)
        i1 = min(_idx_at(c, ph.end) + 1, len(c.t))
        if i1 - i0 < p.exp_min_samples:
            continue
        t, x = c.t[i0:i1], c.x[i0:i1]
        w = 1.0 / (c.sigma[i0:i1] ** 2 + (SYS_FRAC * np.abs(x - x[0])) ** 2)
        t_r = float(raw.params["t_r"])
        f0 = kin.fit_exponential(t, x, t_r, 0.0, w, tau_min=p.tau_min_s, tau_max=p.tau_max_s)
        f = kin.fit_exponential(t, x, t_r, va, w, tau_min=p.tau_min_s, tau_max=p.tau_max_s)
        if f0 is None or f is None or f.at_bound or f.bic() > f0.bic() + UNIFY_MAX_DBIC:
            continue
        tau = float(f.params["tau"])
        ph.raw = f
        ph.fit = ph.fit.model_copy(update={
            "tau_ms": ph.fit.tau_ms.model_copy(update={"value": round(tau * 1000.0)}),
            "v_inf_px_s": round(va, 1),
        })  # fmt: skip
        ph.v_end = float(_model_velocity("exponential", f, np.array([ph.end]))[0])
    return phs


def _momentum_tau(phs: list[_Ph]) -> float | None:
    """The scroller's exponential momentum τ (s): weighted median of ≥
    :data:`POINTER_STOP_MIN_EXP` exponential inertia instances whose τ agree within
    :data:`TAU_RATIO`; ``None`` otherwise."""
    exps = [ph for ph in phs if ph.kind == "inertia" and isinstance(ph.fit, ExponentialFit)]
    if len(exps) < POINTER_STOP_MIN_EXP:
        return None
    taus = [ph.fit.tau_ms.value for ph in exps]  # type: ignore[union-attr]
    if min(taus) <= 0 or max(taus) > TAU_RATIO * min(taus):
        return None
    confs = [max(ph.fit.tau_ms.confidence.value, 1e-3) for ph in exps]  # type: ignore[union-attr]
    return kin.weighted_median(taus, confs) / 1000.0


def _merge_pointer_stops(phs: list[_Ph]) -> list[_Ph]:
    """One scroller has one momentum model (acceptance run on the motivating recording): when
    its releases decay exponentially (≥ 2 consistent instances), a tween-shaped "glide to rest"
    after a drag that stops much sooner than that momentum would from the same speed
    (:data:`POINTER_STOP_FRAC`) is the pointer itself slowing to a stop while still pressed —
    recorded fast drags end that way (deceleration time constants of ≈ 40–70 ms, velocity
    sometimes still rising after the supposed release), not a second release behaviour. It is
    merged into the drag before it, which then ends at rest; the rest that follows is the
    pointer held still / released without a fling."""
    tau = _momentum_tau(phs)
    if tau is None:
        return phs
    out: list[_Ph] = []
    for ph in phs:
        prev = out[-1] if out else None
        v0 = abs(ph.v_start or 0.0)
        stops_early = (
            ph.kind == "inertia"
            and isinstance(ph.fit, TweenFit)
            and v0 > math.e * INERTIA_END_PX_S
            and (ph.end - ph.start) < POINTER_STOP_FRAC * tau * math.log(v0 / INERTIA_END_PX_S)
        )
        if prev is not None and prev.kind == "drag" and stops_early:
            prev.end, prev.end_fixed = ph.end, ph.end_fixed
            prev.v_end = ph.v_end
            if NOTE_POINTER_STOP not in prev.notes:
                prev.notes.append(NOTE_POINTER_STOP)
            continue
        out.append(ph)
    return out


def _rest_after(c: _Ctx, phs: list[_Ph], k: int) -> float | None:
    """Rest position after phase ``k`` if the next phase is a ≥ rest_min_s pause."""
    if k + 1 >= len(phs) or phs[k + 1].kind != "paused":
        return None
    nxt = phs[k + 1]
    if nxt.end - nxt.start < c.params.rest_min_s:
        return None
    i0, i1 = _idx_at(c, nxt.start), _idx_at(c, nxt.end)
    if i1 <= i0:
        return None
    return float(np.median(c.x[i0 : i1 + 1]))


def _snap_evidence(
    c: _Ctx, phs: list[_Ph], pitch: float | None, grid_phases: Sequence[float]
) -> tuple[bool, float, float | None, list[float]]:
    """``(on_grid, confidence, resultant, rests)`` and relabels stops / tween releases."""
    if pitch is None or pitch <= 0:
        return False, 0.0, None, []
    p = c.params
    idx = [k for k, ph in enumerate(phs) if ph.kind in ("inertia", "stop", "snap")]
    rests = [(k, r) for k in idx if (r := _rest_after(c, phs, k)) is not None]
    if not rests:
        return False, 0.0, None, []
    ang = np.array([2.0 * math.pi * (r % pitch) / pitch for _, r in rests])
    resultant = float(abs(np.mean(np.exp(1j * ang))))
    on_grid = False
    conf = 0.0
    if len(rests) >= 2 and resultant >= p.snap_min_resultant:
        on_grid = True
        conf = min(p.cap_snap_step, resultant * 0.85)
    elif len(rests) == 1 and grid_phases:
        k, r = rests[0]
        if isinstance(phs[k].fit, TweenFit) or phs[k].kind == "stop":
            for g in grid_phases:
                off = ((r - g + pitch / 2.0) % pitch) - pitch / 2.0
                if abs(off) <= p.snap_single_tol_px:
                    on_grid, conf = True, p.cap_snap_single
                    break
    if on_grid:
        for k, _ in rests:
            ph = phs[k]
            if ph.kind == "stop" or (ph.kind == "inertia" and isinstance(ph.fit, TweenFit)):
                ph.kind = "snap"
                ph.notes = [n for n in ph.notes if n != NOTE_STOP]
                ph.label_conf = min(max(ph.label_conf, conf), conf if len(rests) == 1 else 1.0)
    return on_grid, conf, resultant, [r for _, r in rests]


def _overshoot(c: _Ctx, phs: list[_Ph], k: int) -> bool:
    r = _rest_after(c, phs, k)
    ph = phs[k]
    if r is None:
        return False
    i0, i1 = _idx_at(c, ph.start), _idx_at(c, ph.end)
    if i1 - i0 < 2:
        return False
    s = 1.0 if c.x[i1] - c.x[i0] >= 0 else -1.0
    return bool(np.max(s * (c.x[i0 : i1 + 1] - r)) > c.params.overshoot_px)


def _cursor_xy(cursor: Sequence[CursorSample], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ct = np.array([s.t_s for s in cursor])
    cx = np.array([s.x_css for s in cursor])
    cy = np.array([s.y_css for s in cursor])
    ok = np.isfinite(cx) & np.isfinite(cy)
    return np.interp(t, ct[ok], cx[ok]), np.interp(t, ct[ok], cy[ok])


def _fuse_cursor(
    c: _Ctx,
    phs: list[_Ph],
    cursor: Sequence[CursorSample] | None,
    region: Rect | None,
    axis: str,
) -> tuple[PauseTrigger, float, bool | None, float | None, float]:
    """Pause trigger + drag-follows-pointer from a visible cursor (§4.5 step 8)."""
    p = c.params
    if not cursor or region is None or len(cursor) < 2:
        return "unknown", p.cap_trigger_no_cursor * 0.875, None, None, 0.0
    on: PauseTrigger = "unknown"
    on_conf = p.cap_trigger_no_cursor * 0.875
    lo_ms, hi_ms = p.hover_enter_window_ms
    ct = np.array([s.t_s for s in cursor])
    inside = np.array(
        [region.x <= s.x_css <= region.x2 and region.y <= s.y_css <= region.y2 for s in cursor]
    )
    enters = [float(ct[j]) for j in range(1, len(ct)) if inside[j] and not inside[j - 1]]
    for k, ph in enumerate(phs):
        instant = ph.kind == "drag" and k > 0 and phs[k - 1].kind == "autoplay"
        if ph.kind != "decelerate" and not instant:
            continue
        t0 = ph.start
        j = int(np.clip(np.searchsorted(ct, t0), 0, len(ct) - 1))
        if instant:
            # autoplay stopped at once by the press that starts the drag; the pointer has been
            # over the scroller for longer than a hover reaction, so hovering did not pause it
            # (``pause_on_press`` evidence; without a cursor the trigger stays unknown)
            entered = [te for te in enters if te <= t0]
            if inside[j] and entered and (t0 - entered[-1]) * 1000.0 > hi_ms:
                on, on_conf = "press", p.conf_pause_press
        # pointer entered the region within [lo, hi] ms of the deceleration onset
        elif any(lo_ms <= (te - t0) * 1000.0 <= hi_ms for te in enters):
            on, on_conf = "hover", p.conf_pause_hover
        elif inside[j] and cursor[j].state == "stationary":
            on, on_conf = "press", p.conf_pause_press
        ph.evidence = "velocity+cursor" if on != "unknown" else ph.evidence
        break
    # drag follows pointer
    follows: bool | None = None
    ratio: float | None = None
    ratio_conf = 0.0
    drags = [ph for ph in phs if ph.kind == "drag"]
    if drags:
        cx, cy = _cursor_xy(cursor, c.t)
        cpos = cx if axis == "x" else cy
        cv = smoothed_velocity(c.t, cpos, p.velocity_half_window)
        ratios: list[float] = []
        for ph in drags:
            i0, i1 = _idx_at(c, ph.start), _idx_at(c, ph.end)
            for j in range(i0, i1 + 1):
                if abs(c.v[j]) > 50.0 and abs(cv[j]) > 1e-6:
                    ratios.append(float(c.v[j] / cv[j]))
        if ratios:
            r = np.asarray(ratios)
            frac = float(np.mean(np.abs(r - 1.0) <= p.pointer_ratio_tol))
            follows = frac >= p.pointer_ratio_min_frac
            ratio = float(np.median(r))
            ratio_conf = frac * c.qt(0, len(c.t))
            for ph in drags:
                ph.evidence = "velocity+cursor"
    return on, on_conf, follows, ratio, ratio_conf


def _leave_delays(
    c: _Ctx, phs: list[_Ph], cursor: Sequence[CursorSample], region: Rect
) -> list[tuple[float, float]]:
    """Hover pause: ``(resume onset − last pointer leave before it, confidence)`` per resume
    that follows a pause (the pointer left during the preceding decelerate / paused phases)."""
    ct = np.array([s.t_s for s in cursor])
    inside = np.array(
        [region.x <= s.x_css <= region.x2 and region.y <= s.y_css <= region.y2 for s in cursor]
    )
    # leave = first sample outside after one inside (± one pass-1 frame)
    leaves = [float(ct[j]) for j in range(1, len(ct)) if inside[j - 1] and not inside[j]]
    out: list[tuple[float, float]] = []
    for k, ph in enumerate(phs):
        if ph.kind != "resume":
            continue
        j = k - 1
        while j >= 0 and phs[j].kind in ("paused", "decelerate"):
            j -= 1
        lo = phs[j + 1].start if j + 1 < k else ph.start
        cand = [tl for tl in leaves if lo <= tl <= ph.start + LEAVE_AFTER_ONSET_S]
        if not cand:
            continue
        # the pointer is seen at the pass-1 rate: ± one frame on the leave time
        dt_cursor = float(np.median(np.diff(ct))) if len(ct) > 1 else c.dt
        conf = c.params.conf_pause_hover * float(
            np.clip(1.0 - dt_cursor / max(ph.start - cand[-1], dt_cursor), 0.3, 1.0)
        )
        out.append((max(ph.start - cand[-1], 0.0), conf))
    return out


def _to_candidate(c: _Ctx, ph: _Ph, degraded: bool) -> PhaseCandidate:
    i0, i1 = _idx_at(c, ph.start), _idx_at(c, ph.end)
    x0 = float(np.interp(ph.start, c.t, c.x))
    x1 = float(np.interp(ph.end, c.t, c.x))
    seg = c.v[i0 : i1 + 1] if i1 >= i0 else c.v[i0 : i0 + 1]
    v_peak = float(seg[int(np.argmax(np.abs(seg)))]) if seg.size else 0.0
    v_start = ph.v_start if ph.v_start is not None else float(np.interp(ph.start, c.t, c.v))
    v_end = ph.v_end if ph.v_end is not None else float(np.interp(ph.end, c.t, c.v))
    if ph.kind == "drag":
        # robust (Theil–Sen, quality-filtered) velocity: a single glitch or jittered timestamp
        # must not become the peak (``_continuity`` adds a fitted release velocity)
        rv = c.robust_v()[i0 : i1 + 1]
        if rv.size:
            v_peak = float(rv[int(np.argmax(np.abs(rv)))])
    elif ph.kind in ("inertia", "snap"):
        v_peak = max([v_start, v_end], key=abs)  # model: the release / tween start is the peak
    elif ph.kind == "stop":
        v_peak = max([v_start, v_end, v_peak], key=abs)
    elif ph.kind == "autoplay" and ph.v_start is not None:
        v_peak = ph.v_start  # the fitted constant velocity
    if ph.kind == "paused":
        v_start = v_end = v_peak = 0.0
    conf = ph.label_conf
    notes = list(ph.notes)
    fit = ph.fit
    if degraded:
        conf *= c.params.degraded_conf_factor
        if NOTE_DEGRADED not in notes:
            notes.append(NOTE_DEGRADED)
        fit = _scale_fit(fit, c.params.degraded_conf_factor)
    return PhaseCandidate(
        kind=ph.kind,
        start_s=float(ph.start),
        end_s=float(ph.end),
        v_start=float(v_start),
        v_end=float(v_end),
        v_peak=float(v_peak),
        displacement=x1 - x0,
        fit=fit,
        interrupted=ph.interrupted,
        evidence=ph.evidence,
        confidence=float(np.clip(conf, 0.0, 1.0)),
        notes=notes,
    )


def _continuity(cands: list[PhaseCandidate]) -> None:
    """Velocity continuity across boundaries (smoothed velocities blur kinks): a drag starts at
    the previous phase's end velocity and ends at the release velocity of the next phase."""
    for k, p in enumerate(cands):
        if p.kind in ("decelerate", "resume"):
            p.v_peak = max(p.v_start, p.v_end, key=abs)
        if p.kind != "drag":
            continue
        if k > 0:
            p.v_start = cands[k - 1].v_end
        nxt = cands[k + 1] if k + 1 < len(cands) else None
        # momentum continues the pointer velocity; a tween (snap / tween inertia) need not
        if nxt is not None and (
            nxt.kind == "stop" or (nxt.kind == "inertia" and not isinstance(nxt.fit, TweenFit))
        ):
            p.v_end = nxt.v_start
        # the peak: the robust in-phase estimate, or a fitted release velocity (a stop's
        # one-step speed carries the full timestamp jitter and is not a peak estimate)
        fitted_end = (
            nxt is not None and nxt.kind == "inertia" and isinstance(nxt.fit, ExponentialFit)
        )
        p.v_peak = max(p.v_peak, p.v_end if fitted_end else 0.0, key=abs)


def _scale_fit(fit: PhaseFit | None, factor: float) -> PhaseFit | None:
    if fit is None:
        return None

    def sc(m: MeasuredNumber) -> MeasuredNumber:
        return MeasuredNumber(
            value=m.value, confidence=cconf.ir_confidence(m.confidence.value * factor)
        )

    if isinstance(fit, ConstantFit):
        return ConstantFit(velocity_px_s=sc(fit.velocity_px_s))
    if isinstance(fit, ExponentialFit):
        return ExponentialFit(tau_ms=sc(fit.tau_ms), v0_px_s=sc(fit.v0_px_s),
                              v_inf_px_s=fit.v_inf_px_s)  # fmt: skip
    if isinstance(fit, RampFit):
        return RampFit(from_px_s=fit.from_px_s, to_px_s=fit.to_px_s,
                       duration_ms=sc(fit.duration_ms), easing=fit.easing)  # fmt: skip
    return TweenFit(distance_px=sc(fit.distance_px), duration_ms=sc(fit.duration_ms),
                    easing=fit.easing)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def segment(
    series: DisplacementSeries,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
    *,
    is_vfr: bool = False,
    timestamps_estimated: bool = False,
    pitch_px: float | None = None,
    pitch_confidence: float = 0.0,
    grid_phases: Sequence[float] = (),
    cursor: Sequence[CursorSample] | None = None,
) -> Segmentation:
    """Label the displacement profile (see module docstring)."""
    t = np.asarray(series.times, dtype=np.float64)
    x = np.asarray(series.pos, dtype=np.float64)
    q = np.asarray(series.quality, dtype=np.float64)
    n = len(t)
    if n < 4:
        raise ValueError("segment needs at least 4 samples")
    v = smoothed_velocity(t, x, params.velocity_half_window)
    sigma = noise_sigma(t, x, v, q, is_vfr=is_vfr)
    dt = float(np.median(np.diff(t)))

    # 1. autoplay speed
    v_lead = leading_autoplay(t, v, params)
    v_noise = lead_velocity_noise(t, v, v_lead)

    def tols(va: float | None) -> tuple[float, float]:
        a = max(params.label_auto_tol, params.label_auto_tol_frac * abs(va or 0.0))
        if va is not None:
            # the band also covers the autoplay's own velocity noise (P2b), within reason
            a = max(a, min(AUTO_NOISE_SIGMAS * v_noise, AUTO_TOL_MAX_FRAC * abs(va)))
        z = max(params.label_zero_tol, params.label_zero_tol_frac * abs(va or 0.0))
        return a, z

    tol_a, tol_z = tols(v_lead)
    v_auto = v_lead
    runs: list[list] = []
    for _ in range(2):
        labels = coarse_labels(v, v_auto, tol_a, tol_z)
        runs = absorb_short(_runs(labels), t, params.min_run_samples)
        if v_auto is None:
            break
        # refine from the leading autoplay run (position domain)
        if runs and runs[0][0] == "A" and runs[0][2] - runs[0][1] >= 3:
            a0, b0 = runs[0][1], runs[0][2]
            f = kin.fit_constant(t[a0:b0], x[a0:b0], 1.0 / sigma[a0:b0] ** 2)
            v_new = float(f.params["v"])
            if abs(v_new - v_auto) < 1e-6:
                break
            v_auto = v_new
            tol_a, tol_z = tols(v_auto)
        else:
            break
    if v_auto is not None and len(runs) > 1 and runs[0][0] != "A" and runs[1][0] == "A":
        if t[runs[0][2] - 1] - t[0] < LEAD_GLITCH_S:
            # a few glitchy frames at the very start of the recording: still autoplay
            runs[1][1] = 0
            runs = runs[1:]
    if v_auto is not None and not (runs and runs[0][0] == "A"):
        v_auto = None  # the leading window was not an autoplay run after all
        tol_a, tol_z = tols(None)
        runs = absorb_short(_runs(coarse_labels(v, None, tol_a, tol_z)), t,
                            params.min_run_samples)  # fmt: skip

    if v_auto is not None:
        runs = _absorb_blips(t, x, sigma, runs, params.min_run_samples)

    c = _Ctx(t=t, x=x, q=q, v=v, sigma=sigma, w=1.0 / sigma**2, dt=dt, v_auto=v_auto,
             tol_a=tol_a, tol_z=tol_z, params=params, is_vfr=is_vfr,
             ts_est=timestamps_estimated)  # fmt: skip

    # 3. phases per run
    phs: list[_Ph] = []
    for i, (lab, a, b) in enumerate(runs):
        prev = phs[-1] if phs else None
        if lab == "A":
            phs.append(_steady_phase(c, "autoplay", a, b))
        elif lab == "Z":
            phs.append(_steady_phase(c, "paused", a, b))
        else:
            phs.extend(_m_run(c, runs, i, prev))
    phs = _reconcile(c, phs)
    phs = _merge_blend(c, phs)
    phs = _unify_momentum_target(c, phs)
    phs = _merge_inconsistent_inertia(phs)
    phs = _merge_pointer_stops(phs)

    # 5. snap evidence, overshoot
    on_grid, grid_conf, resultant, rests = _snap_evidence(c, phs, pitch_px, grid_phases)
    overshoot = False
    for k, ph in enumerate(phs):
        if ph.kind in ("snap", "inertia") and _overshoot(c, phs, k):
            overshoot = True
            if NOTE_OVERSHOOT not in ph.notes:
                ph.notes.append(NOTE_OVERSHOOT)

    # 7. resume delays
    delays: list[tuple[float, float | None, float]] = []
    for k, ph in enumerate(phs):
        if ph.kind != "resume":
            continue
        rest_end = None
        release = None
        for j in range(k - 1, -1, -1):
            pk = phs[j].kind
            if rest_end is None and pk in ("inertia", "snap", "stop", "decelerate"):
                rest_end = phs[j].end
            if pk == "drag":
                # a drag that ended at rest (no momentum between it and the rest): the pointer
                # stopped there, but when it was let go is not visible -> no release delay
                release = phs[j].end if rest_end is not None else None
                break
            if pk == "autoplay":
                break
        if rest_end is None:
            rest_end = phs[k - 1].start if k > 0 else ph.start
        conf = min(ph.label_conf, 0.4 + 0.5 * c.qt(_idx_at(c, rest_end), _idx_at(c, ph.end)))
        after_release = None if release is None else max(ph.start - release, 0.0)
        delays.append((max(ph.start - rest_end, 0.0), after_release, conf))

    # 8. cursor fusion
    pause_on, pause_conf, follows, ratio, ratio_conf = _fuse_cursor(
        c, phs, cursor, series.region, series.axis
    )
    leave_delays = (
        _leave_delays(c, phs, cursor, series.region) if pause_on == "hover" and cursor else []
    )

    # degraded tracking per phase
    warnings: list[SpecWarning] = []
    degraded_ids: list[int] = []
    low_frac_all: list[float] = []
    cands: list[PhaseCandidate] = []
    for k, ph in enumerate(phs):
        i0, i1 = _idx_at(c, ph.start), _idx_at(c, ph.end)
        frac = cconf.degraded_fraction(q[i0 : i1 + 1], params)
        deg = frac > params.degraded_max_frac
        if deg:
            degraded_ids.append(k + 1)
            low_frac_all.append(frac)
        cands.append(_to_candidate(c, ph, deg))
    _continuity(cands)
    if degraded_ids:
        pct = int(round(100.0 * float(np.mean(q < params.degraded_q))))
        names = ", ".join(f"p{k}" for k in degraded_ids)
        warnings.append(SpecWarning(
            code="tracking_degraded", severity="warn",
            message=f"About {pct}% of frames could not be tracked precisely ({names}); their "
                    "values are less certain.",
        ))  # fmt: skip

    # autoplay confidence (from the autoplay phases)
    ap_conf = 0.0
    if v_auto is not None:
        aps = [p for p in cands if p.kind == "autoplay" and isinstance(p.fit, ConstantFit)]
        if aps:
            ap_conf = max(p.fit.velocity_px_s.confidence.value for p in aps)  # type: ignore[union-attr]
            # report the leading autoplay run's fit (most context, before any interaction)
            v_auto = float(aps[0].fit.velocity_px_s.value)  # type: ignore[union-attr]

    # drag peak: as good as the tracking around the peak sample (fast drags blur)
    peak_conf = None
    drag_c = [p for p in cands if p.kind == "drag"]
    if drag_c:
        pk = max(drag_c, key=lambda p: abs(p.v_peak))
        i0, i1 = _idx_at(c, pk.start_s), _idx_at(c, pk.end_s)
        rv = c.robust_v()
        j = i0 + int(np.argmax(np.abs(rv[i0 : i1 + 1]))) if i1 >= i0 else i0
        half = max(c.params.velocity_half_window, 3)
        peak_conf = min(pk.confidence, c.qt(j - half, j + half + 1))

    ctx = kin.BehaviorContext(
        autoplay_velocity=v_auto,
        pitch_px=pitch_px,
        pitch_confidence=pitch_confidence,
        snap_grid=on_grid,
        snap_grid_confidence=grid_conf,
        pause_on=pause_on,
        pause_on_confidence=pause_conf,
        follows_pointer=follows,
        pointer_ratio=ratio,
        pointer_ratio_confidence=ratio_conf,
        overshoot=overshoot,
        resume_delays=delays,
        resume_leave_delays=leave_delays,
        drag_peak_confidence=peak_conf,
    )
    behavior = kin.aggregate(cands, ctx, params)
    return Segmentation(
        phases=cands,
        autoplay_velocity=v_auto,
        autoplay_confidence=ap_conf,
        behavior=behavior,
        velocity=v,
        sigma=sigma,
        rests=rests,
        snap_resultant=resultant,
        warnings=warnings,
    )


__all__ = [
    "DRAG_COST_PER_SAMPLE",
    "Segmentation",
    "absorb_short",
    "coarse_labels",
    "leading_autoplay",
    "noise_sigma",
    "regrab_valleys",
    "segment",
    "split_release",
]
