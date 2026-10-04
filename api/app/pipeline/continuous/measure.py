"""Continuous-scroller orchestrator (PLAN-continuous §4.7): one ambient region → ScrollerAnalysis.

:func:`analyse_ambient` streams each ambient region once (``decode.iter_region``) and runs:
coherence / axis test (+ opposing sub-scroller split) on the first ≤ ``coherence_window_s`` →
streaming displacement tracking + panorama → phase segmentation → behaviour. No IR is built here
(``continuous.assemble`` does that).

:func:`analyse_stream` is the I/O-free core (frames from any iterable), used by tests.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from app.models.ir import SpecWarning
from app.models.measure import (
    AmbientRegion,
    CursorSample,
    CursorTrack,
    ProbeInfo,
    Rect,
    RegimeReport,
    Scale,
    ScrollerAnalysis,
)
from app.pipeline import decode
from app.pipeline.continuous import cards as cards_mod
from app.pipeline.continuous import displacement as disp
from app.pipeline.continuous import panorama as pano
from app.pipeline.continuous.phases import segment
from app.pipeline.continuous.pointer import occlusion_corrected
from app.pipeline.params import DEFAULT_PARAMS, ContinuousParams, MeasureParams
from app.pipeline.scan import ambient_mask_rects

log = logging.getLogger(__name__)

#: A scroller must show at least one phase of these kinds (a region that only rests is not one).
MOVING_KINDS = frozenset({"autoplay", "drag", "inertia", "snap", "stop", "decelerate", "resume"})
#: Minimum frames for the coherence test / for a usable series.
MIN_FRAMES = 6
#: The reported region is refined from changed pixels over the whole stream; the change
#: reference frame is refreshed this often (s).
REGION_REF_REFRESH_S = 0.5
#: Panorama: paste a frame only after the content moved this far (analysis px) since the last
#: pasted frame (bounded cost; ≤ pano_max_contrib per column anyway).
PANO_MIN_STEP_PX = 2.0


def _no_pad(params: MeasureParams) -> MeasureParams:
    return dataclasses.replace(
        params,
        decode=dataclasses.replace(params.decode, crop_pad_min_css=0.0, crop_pad_frac=0.0),
    )


def region_crop(
    region: AmbientRegion,
    probe: ProbeInfo,
    scale: Scale,
    params: MeasureParams,
    cell_css: float | None = None,
) -> tuple[Rect, tuple[int, int, int, int]]:
    """Crop for one ambient region: rect grown by ``region_grow_cells`` cells, clamped, snapped
    to analysis px exactly like ``decode.iter_region`` does."""
    c = params.continuous
    grow = c.region_grow_cells * (cell_css if cell_css is not None else c.cell_css)
    fw, fh = decode.frame_size_css(probe, scale)
    r = region.rect_css.padded(grow).clamped(fw, fh)
    return decode.crop_rect_css([r], probe, scale, _no_pad(params))


def _cursor_samples(cursor: CursorTrack | None) -> list[CursorSample] | None:
    if cursor is None or not cursor.visible or len(cursor.samples) < 2:
        return None
    return list(cursor.samples)


def _not_scroller(
    region: AmbientRegion, reason: str | None, coherence: float, warnings: list[SpecWarning]
) -> ScrollerAnalysis:
    return ScrollerAnalysis(
        region=region,
        is_scroller=False,
        reason=reason or "untrackable",  # type: ignore[arg-type]
        coherence=coherence,
        warnings=warnings,
    )


def _narrow_box(
    box: tuple[int, int, int, int], axis: str, strips: np.ndarray, n_strips: int
) -> tuple[int, int, int, int]:
    """Restrict ``box`` (crop px) to the strips (perpendicular to ``axis``) in ``strips``."""
    x0, y0, x1, y1 = box
    if axis == "x":
        edges = np.linspace(y0, y1, n_strips + 1).astype(int)
        return x0, int(edges[strips.min()]), x1, int(edges[strips.max() + 1])
    edges = np.linspace(x0, x1, n_strips + 1).astype(int)
    return int(edges[strips.min()]), y0, int(edges[strips.max() + 1]), y1


def _crop(frames: Sequence[np.ndarray], box: tuple[int, int, int, int]) -> list[np.ndarray]:
    x0, y0, x1, y1 = box
    return [np.ascontiguousarray(f[y0:y1, x0:x1], dtype=np.float32) for f in frames]


def _split_opposing(
    sub: list[np.ndarray],
    times: list[float],
    box: tuple[int, int, int, int],
    k: float,
    params: ContinuousParams,
    axes: Sequence[str],
) -> tuple[tuple[int, int, int, int], int, disp.CoherenceResult | None, int]:
    """Opposing sub-scrollers (§4.2): strips one cell thick, clustered by velocity.

    Returns ``(box, extra_groups, coherence of the chosen group or None, groups_seen)``.
    """
    strip_px = max(4, int(round(params.cell_css * k)))
    best: tuple[tuple[int, int, int, int], int, disp.CoherenceResult | None, int] = (
        box, 0, None, 0,
    )  # fmt: skip
    for axis in axes:
        vel, _ = disp.strip_velocities(sub, times, axis, strip_px, k)  # type: ignore[arg-type]
        groups = disp.cluster_strips(vel, params)
        if not groups:
            continue
        n_strips = len(vel)
        nb = _narrow_box(box, axis, groups[0], n_strips)
        if nb == box and len(groups) == 1:
            continue
        coh = disp.coherence_test(_crop_rel(sub, box, nb), times, k, params)
        if coh.is_scroller:
            return nb, len(groups) - 1, coh, len(groups)
        best = (box, 0, None, max(best[3], len(groups)))
    return best


def _crop_rel(
    sub: list[np.ndarray], box: tuple[int, int, int, int], nb: tuple[int, int, int, int]
) -> list[np.ndarray]:
    """Crop already-boxed frames to the narrower box ``nb`` (both in crop px)."""
    rel = (nb[0] - box[0], nb[1] - box[1], nb[2] - box[0], nb[3] - box[1])
    return _crop(sub, rel)


@dataclasses.dataclass(frozen=True, slots=True)
class CardLayout:
    """Pitch / gap (CSS px) and where they came from (panorama or card census)."""

    pitch_px: float | None
    pitch_confidence: float
    gap_px: float | None
    gap_confidence: float
    grid_phases: list[float]
    card_len_px: float | None = None
    card_cross_px: float | None = None
    card_zoom_per_px2: float | None = None
    card_zoom_reach_px: float | None = None
    card_zoom_confidence: float = 0.0


#: Census vs panorama: a panorama pitch within this fraction of a whole multiple (≥ 2) of a
#: confident census pitch is a multiple (alternate cards looking alike), not the pitch.
MULTIPLE_REL = 0.02
CENSUS_MIN_CONF = 0.5


def _is_multiple(p: float, base: float) -> bool:
    m = round(p / base)
    return m >= 2 and abs(p - m * base) <= MULTIPLE_REL * p


def card_layout(pres: pano.PanoramaResult, census: cards_mod.CensusResult | None) -> CardLayout:
    """The panorama's pitch / gap unless the card census must take over (P2b): no periodic
    pitch in the panorama (photo-filled cards, tiny gaps), cards that scale with their
    position (the panorama assumes rigid content; its pitch is then meaningless), or a
    panorama pitch that is a whole multiple of a confident census pitch."""
    multiple = (
        census is not None
        and pres.pitch_px is not None
        and census.pitch_confidence >= CENSUS_MIN_CONF
        and _is_multiple(pres.pitch_px, census.pitch)
    )
    if census is None or (pres.pitch_px is not None and not census.zoomed and not multiple):
        return CardLayout(pres.pitch_px, pres.pitch_confidence, pres.gap_px,
                          pres.gap_confidence, list(pres.grid_phases))  # fmt: skip
    if pres.loop_period_px is not None and census.pitch >= pres.loop_period_px - 2.0:
        return CardLayout(None, 0.0, None, 0.0, [])
    return CardLayout(
        pitch_px=round(census.pitch, 2),
        pitch_confidence=census.pitch_confidence,
        gap_px=round(census.gap, 2),
        gap_confidence=census.gap_confidence,
        grid_phases=list(census.grid_phases),
        card_len_px=round(census.card_len, 2),
        card_cross_px=round(census.card_cross, 2),
        card_zoom_per_px2=census.zoom_per_px2,
        card_zoom_reach_px=round(census.zoom_extent, 2) if census.zoomed else None,
        card_zoom_confidence=census.zoom_confidence if census.zoomed else 0.0,
    )


def analyse_stream(
    frames: Iterable[tuple[float, np.ndarray]],
    region: AmbientRegion,
    crop_css: Rect,
    k: float,
    params: ContinuousParams = DEFAULT_PARAMS.continuous,
    *,
    is_vfr: bool = False,
    timestamps_estimated: bool = False,
    cursor: Sequence[CursorSample] | None = None,
    cursor_hidden_in: Rect | None = None,
) -> ScrollerAnalysis:
    """Analyse one region from a time-ordered stream of gray crops of ``crop_css``.

    ``k`` = analysis px per CSS px of the crops. ``cursor_hidden_in``: the masked area the
    cursor track cannot see into (``pointer.occlusion_corrected`` places the unseen samples).
    """
    it: Iterator[tuple[float, np.ndarray]] = iter(frames)
    warnings: list[SpecWarning] = []

    # ---- buffer the coherence window ---------------------------------------------------------
    buf_t: list[float] = []
    buf_f: list[np.ndarray] = []
    compact: bool | None = None  # integral gray values (decoded video) -> buffer as uint8
    for t, f in it:
        f = np.asarray(f)
        if compact is None:
            compact = bool(np.array_equal(f, np.round(f))) and float(f.max(initial=0)) <= 255
        buf_t.append(float(t))
        buf_f.append(f.astype(np.uint8) if compact else f.astype(np.float32))
        if buf_t[-1] - buf_t[0] >= params.coherence_window_s and len(buf_t) >= MIN_FRAMES:
            break
    if len(buf_t) < 3:
        return _not_scroller(region, "untrackable", 0.0, warnings)
    f32 = [b.astype(np.float32) for b in buf_f]
    h, w = f32[0].shape
    box = disp.refine_box(f32)
    sub = _crop(f32, box)
    coh = disp.coherence_test(sub, buf_t, k, params)
    extra = 0
    if coh.is_scroller:
        nb, extra_g, coh2, _ = _split_opposing(sub, buf_t, box, k, params, [coh.axis or "x"])
        if coh2 is not None and extra_g > 0:
            box, extra, coh = nb, extra_g, coh2
    else:
        nb, extra_g, coh2, groups_seen = _split_opposing(sub, buf_t, box, k, params, ["x", "y"])
        if coh2 is None:
            reason = "opposing_unsplit" if groups_seen >= 2 else coh.reason
            return _not_scroller(region, reason, coh.coherence, warnings)
        box, extra, coh = nb, extra_g, coh2
    axis = coh.axis
    assert axis is not None
    if region.merged and not extra:
        # pieces of one row: track the whole row (a box limited to one piece sees few,
        # differently magnified features and loses fast drags)
        box = disp.changed_span(f32, box, axis)

    # ---- streaming tracking + panorama -------------------------------------------------------
    tracker = disp.DisplacementTracker(axis=axis, k=k, box=box, params=params,
                                       seed_velocity=coh.velocity)  # fmt: skip
    builder = pano.PanoramaBuilder(k, params)
    census = cards_mod.CardCensus(k)
    acc = np.zeros((h, w), np.float32)
    ref = f32[0]
    ref_t = buf_t[0]
    last_paste: float | None = None
    x0, y0, x1, y1 = box

    def consume(t: float, f: np.ndarray) -> None:
        nonlocal ref, ref_t, last_paste
        n_before = len(tracker.times)
        tracker.push(t, f)
        if len(tracker.times) == n_before:
            return  # duplicate
        np.maximum(acc, cv2.absdiff(f, ref), out=acc)
        if t - ref_t >= REGION_REF_REFRESH_S:
            ref, ref_t = f, t
        pos = float(np.sum(tracker.d_css)) if tracker.d_css else 0.0
        if last_paste is None or abs(pos - last_paste) * k >= PANO_MIN_STEP_PX:
            builder.add(disp.axis_major(f[y0:y1, x0:x1], axis), pos)
            # the census sees the tracking box's rows over the whole crop along the axis (the
            # box can be narrower than the scroller where low-contrast content barely changed
            # during the coherence window)
            strip = f[y0:y1, :] if axis == "x" else f[:, x0:x1]
            census.add(disp.axis_major(strip, axis), cards_mod.band_margins(f, box, axis), pos)
            last_paste = pos

    for t, f in zip(buf_t, f32, strict=True):
        consume(t, f)
    del f32, buf_f, sub
    for t, f in it:
        consume(float(t), np.asarray(f, dtype=np.float32))
    tracker.finish()
    if len(tracker.times) < MIN_FRAMES:
        return _not_scroller(region, "untrackable", coh.coherence, warnings)

    # ---- reported region: union of changed pixels over the stream ---------------------------
    # unpadded: the reported box is the changed pixels themselves (the round trip compares it
    # at ±2 px; the padded box measured every scroller 4 px too wide and too tall)
    rb = disp.refine_box([np.zeros((h, w), np.float32), acc], pad=0)
    reg = Rect(
        crop_css.x + rb[0] / k, crop_css.y + rb[1] / k, (rb[2] - rb[0]) / k, (rb[3] - rb[1]) / k
    )
    region_conf = min(params.cap_region, 0.5 + 0.5 * coh.coherence)
    series = tracker.series(reg, region_conf)

    # ---- panorama + phases -------------------------------------------------------------------
    if cursor is not None and cursor_hidden_in is not None:
        cursor = occlusion_corrected(cursor, reg, cursor_hidden_in)
    pres = pano.analyse(builder, params)
    # zoom centre = the scroller's centre (the reported region) along the axis, crop px
    lo_c, hi_c = (rb[0], rb[2]) if axis == "x" else (rb[1], rb[3])
    cres = census.result((lo_c, hi_c))
    cards = card_layout(pres, cres)
    if cres is not None:
        log.info(
            "card census: card %.1f x %.1f gap %.1f pitch %.1f (conf %.2f) zoom x%.3f at %.0f px "
            "(%d cards / %d frames); panorama pitch %s -> using %s",
            cres.card_len, cres.card_cross, cres.gap, cres.pitch, cres.pitch_confidence,
            cres.zoom_observed, cres.zoom_extent, cres.cards, cres.frames, pres.pitch_px,
            "census" if cards.card_len_px is not None else "panorama",
        )  # fmt: skip
    seg = segment(
        series,
        params,
        is_vfr=is_vfr,
        timestamps_estimated=timestamps_estimated,
        # snap evidence compares rest positions (tracker units) with the card grid; when the
        # cards scale with position the tracker's displacement is an average over differently
        # magnified content, so that grid does not exist in its units: no snap from the pitch
        pitch_px=cards.pitch_px if cards.card_zoom_per_px2 is None else None,
        pitch_confidence=cards.pitch_confidence,
        grid_phases=cards.grid_phases,
        cursor=cursor,
    )
    if not any(ph.kind in MOVING_KINDS for ph in seg.phases):
        # coherent but it never actually moves (only rest): not a scroller
        return _not_scroller(region, "untrackable", coh.coherence, warnings)
    warnings.extend(seg.warnings)
    loop_px = pres.loop_period_px
    if seg.autoplay_velocity is not None and loop_px is None:
        warnings.append(SpecWarning(
            code="loop_period_not_observed", severity="info",
            message="The content never repeated within the recording, so the loop length is "
                    "not known.",
        ))  # fmt: skip
    if extra:
        warnings.append(SpecWarning(
            code="extra_scrollers_ignored", severity="info",
            message=f"{extra} other scrolling row(s) in the same area were ignored; only the "
                    "largest one is described.",
        ))  # fmt: skip
    return ScrollerAnalysis(
        region=region,
        is_scroller=True,
        axis=axis,
        coherence=coh.coherence,
        series=series,
        autoplay_velocity=seg.autoplay_velocity,
        autoplay_confidence=seg.autoplay_confidence,
        loop_period_px=loop_px,
        loop_confidence=pres.loop_confidence if loop_px is not None else 0.0,
        pitch_px=cards.pitch_px,
        pitch_confidence=cards.pitch_confidence,
        gap_px=cards.gap_px,
        gap_confidence=cards.gap_confidence,
        card_len_px=cards.card_len_px,
        card_cross_px=cards.card_cross_px,
        card_zoom_per_px2=cards.card_zoom_per_px2,
        card_zoom_reach_px=cards.card_zoom_reach_px,
        card_zoom_confidence=cards.card_zoom_confidence,
        phases=seg.phases,
        behavior=seg.behavior,
        warnings=warnings,
    )


def analyse_ambient(
    video: Path,
    info: ProbeInfo,
    scale: Scale,
    regime: RegimeReport,
    params: MeasureParams = DEFAULT_PARAMS,
    *,
    cursor: CursorTrack | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[ScrollerAnalysis]:
    """Coherence, displacement, panorama and phases for every ambient region (no IR)."""
    out: list[ScrollerAnalysis] = []
    samples = _cursor_samples(cursor)
    for region in regime.ambient:
        crop_css, crop_px = region_crop(region, info, scale, params, regime.cell_css)
        k = crop_px[2] / crop_css.w if crop_css.w > 0 else scale.k
        stream_warnings: list[SpecWarning] = []
        frames = decode.iter_region(
            video, info, scale, crop_css, params, params.continuous.region_max_fps,
            warnings=stream_warnings, ffmpeg_bin=ffmpeg_bin,
        )  # fmt: skip
        hidden = ambient_mask_rects([region], decode.frame_size_css(info, scale), params)
        sa = analyse_stream(
            frames,
            region,
            crop_css,
            k,
            params.continuous,
            is_vfr=info.is_vfr,
            timestamps_estimated=info.timestamps_estimated or bool(stream_warnings),
            cursor=samples,
            cursor_hidden_in=hidden[0] if hidden else None,
        )
        sa.warnings = stream_warnings + sa.warnings
        log.info(
            "ambient region %s: scroller=%s axis=%s phases=%s",
            region.rect_css, sa.is_scroller, sa.axis, [p.kind for p in sa.phases],
        )  # fmt: skip
        out.append(sa)
    return out


__all__ = ["analyse_ambient", "analyse_stream", "region_crop"]
