"""A2 scan, PLAN-continuous Track R: activity grid, ambient masking, unchanged default path."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.errors import ErrorCode, PipelineError
from app.models.measure import AmbientRegion, CursorTrack, Rect, Scale, ScanResult
from app.pipeline.cursor import Blob
from app.pipeline.params import DEFAULT_PARAMS as P
from app.pipeline.scan import (
    Pass1Accumulator,
    Pass1Stats,
    _ThumbCtx,
    activity_grid_shape,
    ambient_mask_rects,
    build_scan_result,
    subtract_masks,
)
from tests.unit.m1_helpers import base_canvas, composite, scale1, solid_rect_sprite, translate

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FPS = 30.0
FRAME = (1280.0, 800.0)
STRIP = Rect(100.0, 300.0, 760.0, 200.0)
CARD = Rect(950.0, 100.0, 120.0, 90.0)
STRADDLE = Rect(STRIP.x2 - 30, STRIP.y + 50, 50, 40)


# ---- recorded S1 pass-1 stats: the default path is byte-identical ----------------------------
#
# ``fixtures/scan_s1_pass1.npz`` = pass-1 stats of the synthetic S1 video
# (``card_hover_translate``) and a digest of the ``ScanResult`` that ``build_scan_result``
# produced from them *before* Track R touched ``scan.py``. Recorded once, no video needed.


def _r(x: float) -> str:
    return repr(float(x))


def _rect(b: Rect | None) -> list[str] | None:
    return None if b is None else [_r(b.x), _r(b.y), _r(b.w), _r(b.h)]


def _seg(s) -> list[Any] | None:
    if s is None:
        return None
    return [_r(s.start_s), _r(s.end_s), _rect(s.roi_css), _r(s.peak_energy), s.is_global]


def _sha(a: np.ndarray, dtype=None) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def scan_digest(res: ScanResult) -> dict[str, Any]:
    """Exact (repr-level) digest of every ``ScanResult`` field."""
    c = res.cursor
    return {
        "energy": _sha(res.energy, np.float64),
        "times": _sha(res.frame_times),
        "thumbs": _sha(res.thumbs),
        "segments": [_seg(s) for s in res.segments],
        "stable": [[_r(s.start_s), _r(s.end_s)] for s in res.stable],
        "primary": [_seg(res.primary.forward), _seg(res.primary.reverse), res.primary.round_trip],
        "cursor": [
            c.visible,
            _r(c.confidence),
            [[_r(s.t_s), _r(s.x_css), _r(s.y_css), s.state] for s in c.samples],
            [_rect(b) for b in c.boxes_css],
        ],
        "warnings": [[w.code, w.severity, w.message] for w in res.warnings],
    }


def _blob_lists(rows: np.ndarray, n: int) -> list[list[Blob]]:
    out: list[list[Blob]] = [[] for _ in range(n)]
    for i, x, y, w, h, area, top_x in rows:
        out[int(i)].append(Blob(float(x), float(y), float(w), float(h), float(area), float(top_x)))
    return out


def load_s1() -> tuple[Pass1Stats, Scale, dict[str, Any]]:
    z = np.load(FIXTURES / "scan_s1_pass1.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = int(meta["n"])
    thumbs = np.frombuffer(lzma.decompress(z["thumbs_xz"].tobytes()), dtype=np.uint8)
    stats = Pass1Stats(
        times=z["times"],
        blobs=_blob_lists(z["blobs"], n),
        slow_blobs=_blob_lists(z["slow_blobs"], n),
        changed_frac=z["changed_frac"],
        shift_css=z["shift_css"],
        response=z["response"],
        thumbs=thumbs.reshape(meta["thumbs_shape"]),
        image_scale=float(meta["image_scale"]),
        thumb_scale=float(meta["thumb_scale"]),
        frame_css=(float(meta["frame_css"][0]), float(meta["frame_css"][1])),
    )
    scale = Scale(
        pixel_ratio=meta["pixel_ratio"],
        pixel_ratio_source=meta["pixel_ratio_source"],
        analysis_scale=float(meta["analysis_scale"]),
    )
    return stats, scale, meta["expected"]


def test_default_path_byte_identical_on_recorded_s1() -> None:
    stats, scale, expected = load_s1()
    assert len(expected["segments"]) == 2  # sanity: S1 is a hover with forward + reverse
    assert scan_digest(build_scan_result(stats, scale, P)) == expected
    assert scan_digest(build_scan_result(stats, scale, P, ambient=None)) == expected
    assert scan_digest(build_scan_result(stats, scale, P, ambient=[])) == expected


def test_far_ambient_region_leaves_s1_segments() -> None:
    """A mask over an area where nothing changes does not alter S1's segmentation."""
    stats, scale, expected = load_s1()
    far = AmbientRegion(Rect(0.0, stats.frame_css[1] - 40.0, 200.0, 40.0), cells=10, lead_frac=1)
    res = build_scan_result(stats, scale, P, ambient=[far])
    assert [_seg(s) for s in res.segments] == expected["segments"]
    assert [_seg(res.primary.forward), _seg(res.primary.reverse), res.primary.round_trip] == (
        expected["primary"]
    )


# ---- subtract_masks / mask rects (pure) ------------------------------------------------------


def test_subtract_masks_cases() -> None:
    r = Rect(0, 0, 100, 50)
    assert subtract_masks(r, []) == (1.0, r)
    assert subtract_masks(r, [Rect(200, 0, 10, 10)]) == (1.0, r)
    assert subtract_masks(r, [Rect(-10, -10, 200, 100)]) == (0.0, None)
    keep, box = subtract_masks(r, [Rect(50, -5, 100, 100)])  # right half masked
    assert keep == pytest.approx(0.5) and box == Rect(0, 0, 50, 50)
    # two overlapping masks: the overlap is not counted twice
    keep, box = subtract_masks(r, [Rect(0, 0, 60, 50), Rect(40, 0, 40, 50)])
    assert keep == pytest.approx(0.2) and box == Rect(80, 0, 20, 50)
    # hole in the middle: bbox of the remainder is still the whole rect
    keep, box = subtract_masks(r, [Rect(40, 20, 20, 10)])
    assert keep == pytest.approx(1 - 200 / 5000) and box == r


def test_ambient_mask_rects_grow_and_clamp() -> None:
    a = AmbientRegion(Rect(0, 10, 100, 20), cells=8, lead_frac=0.9)
    (m,) = ambient_mask_rects([a], (90.0, 200.0), P)
    pad = P.continuous.ambient_mask_dilate_css
    assert m == Rect(0, 10 - pad, 90, 20 + 2 * pad)
    assert ambient_mask_rects([Rect(1, 1, 5, 5)], (90.0, 200.0), P)[0].x == 0.0  # Rect accepted


# ---- masked segmentation on synthetic energy (no video) --------------------------------------


def _stats(blobs: list[list[Blob]], changed: np.ndarray | None = None, **kw: Any) -> Pass1Stats:
    n = len(blobs)
    fw, fh = FRAME
    return Pass1Stats(
        times=np.arange(n) / FPS,
        blobs=blobs,
        slow_blobs=kw.get("slow", [[] for _ in range(n)]),
        changed_frac=np.zeros(n) if changed is None else changed,
        shift_css=kw.get("shift", np.zeros(n)),
        response=kw.get("response", np.zeros(n)),
        thumbs=np.zeros((n, 150, 240), np.uint8),
        image_scale=0.75,
        thumb_scale=240 / 1280,
        frame_css=FRAME,
    )


def _blob(r: Rect, area: float | None = None) -> Blob:
    return Blob(r.x, r.y, r.w, r.h, r.area if area is None else area, r.x + r.w / 2)


def _strip_and_card(n: int = 90, card: range = range(36, 46)) -> list[list[Blob]]:
    """Ambient strip energy every frame (noisy, ~30k px²) + a card change in ``card`` frames."""
    rng = np.random.default_rng(7)
    out: list[list[Blob]] = [[]]
    for i in range(1, n):
        fb = [_blob(STRIP, 30_000.0 + float(rng.normal(0, 4_000)))]
        # straddles the strip edge (leakage, compression noise): 30 of its 50 px inside
        fb.append(_blob(STRADDLE, float(rng.uniform(1_600.0, 2_000.0))))
        if i in card:
            fb.append(_blob(CARD))
        out.append(fb)
    return out


AMBIENT = [AmbientRegion(STRIP, cells=600, lead_frac=0.95)]


def test_ambient_strip_breaks_the_unmasked_path() -> None:
    """The motivating failure: the strip's energy hides the card change (§0.1)."""
    with pytest.raises(PipelineError) as ei:
        build_scan_result(_stats(_strip_and_card()), scale1(), P)
    assert ei.value.code in (ErrorCode.NO_MOTION_DETECTED, ErrorCode.NO_STABLE_STATE)


def test_masked_segmentation_finds_the_card_run() -> None:
    res = build_scan_result(_stats(_strip_and_card()), scale1(), P, ambient=AMBIENT)
    assert len(res.segments) == 1
    seg = res.segments[0]
    assert seg.start_s == pytest.approx(35 / FPS) and seg.end_s == pytest.approx(45 / FPS)
    # masked parts never enter the ROI: card ∪ the unmasked sliver of the straddling blob
    (mask,) = ambient_mask_rects(AMBIENT, FRAME, P)
    sliver = Rect(mask.x2, STRADDLE.y, STRADDLE.x2 - mask.x2, STRADDLE.h)
    assert seg.roi_css == CARD.union(sliver)
    assert seg.roi_css.intersection(mask) is None
    assert not seg.is_global
    # the edge-straddling blob only contributes its unmasked part (no strip energy at all)
    pad = P.continuous.ambient_mask_dilate_css
    keep = 1 - (30 + pad) / 50
    blobs = _strip_and_card()
    assert res.energy[5] == pytest.approx(blobs[5][1].area * keep)
    assert res.energy[40] == pytest.approx(blobs[40][1].area * keep + CARD.area)
    # Rect input == AmbientRegion input
    res2 = build_scan_result(_stats(_strip_and_card()), scale1(), P, ambient=[STRIP])
    assert np.array_equal(res.energy, res2.energy)


def test_masked_slow_blobs_are_masked_too() -> None:
    n = 90
    slow: list[list[Blob]] = [[] for _ in range(n)]
    for i in range(n):
        slow[i].append(_blob(STRIP, 25_000.0))
    for i in range(50, 60):
        slow[i].append(_blob(CARD))
    blobs: list[list[Blob]] = [[] for _ in range(n)]
    res = build_scan_result(_stats(blobs, slow=slow), scale1(), P, ambient=AMBIENT)
    assert len(res.segments) == 1 and res.segments[0].roi_css == CARD


def test_masked_global_check_ignores_the_ambient_area() -> None:
    """Full-frame changed fraction > 30 % only because of the strip → not page motion."""
    n = 90
    big = Rect(0, 250, 1280, 400)  # 50 % of the frame, ambient
    blobs: list[list[Blob]] = [[]] + [[_blob(big)] for _ in range(1, n)]
    changed = np.zeros(n)
    changed[1:] = big.area / (FRAME[0] * FRAME[1])
    for i in range(36, 46):
        blobs[i].append(_blob(CARD))
        changed[i] += CARD.area / (FRAME[0] * FRAME[1])
    shift = np.full(n, 5.0)  # phase correlation locked onto the moving strip
    response = np.full(n, 0.6)
    region = [AmbientRegion(big, cells=2000, lead_frac=1.0)]
    stats = _stats(blobs, changed, shift=shift, response=response)
    res = build_scan_result(stats, scale1(), P, ambient=region)
    assert len(res.segments) == 1 and not res.segments[0].is_global
    # same run with the full-frame fraction (no masking of the fraction) would be global
    blobs_card_only: list[list[Blob]] = [
        [b for b in fb if b.rect == CARD] for fb in blobs
    ]  # energy = card only, but changed_frac still includes the strip
    with pytest.raises(PipelineError) as ei:
        build_scan_result(_stats(blobs_card_only, changed, shift=shift, response=response),
                          scale1(), P)  # fmt: skip
    assert ei.value.code == ErrorCode.UNSUPPORTED_MOTION


def test_masked_mode_drops_content_tracks_from_cursor_tracking() -> None:
    """Glyph-sized blobs drifting inside the scroller are not a pointer once masked."""
    n = 90
    blobs: list[list[Blob]] = [[]]
    for i in range(1, n):
        x = STRIP.x + 20 + 2.0 * i  # 60 px/s, small "glyph" blob
        fb = [_blob(Rect(x, STRIP.y + 40, 14, 18))]
        if i in range(36, 46):
            fb.append(_blob(CARD))
        blobs.append(fb)
    res = build_scan_result(_stats(blobs), scale1(), P, ambient=AMBIENT)
    assert not res.cursor.visible
    assert len(res.segments) == 1 and res.segments[0].roi_css == CARD


def test_thumb_ctx_masks_ambient_pixels() -> None:
    ctx = _ThumbCtx(
        thumbs=np.zeros((3, 150, 240), np.uint8),
        times=np.arange(3) / FPS,
        k=240 / 1280,
        cursor=CursorTrack(visible=False, boxes_css=[None] * 3),
        ambient=[STRIP],
    )
    valid = ctx.cursor_valid([0, 1])
    k = 240 / 1280
    assert not valid[int(STRIP.y * k) + 2, int(STRIP.x * k) + 2]
    assert valid[5, 5]
    ctx.ambient = []
    assert ctx.cursor_valid([0]).all()


# ---- activity grid (Pass1Accumulator) --------------------------------------------------------


def _accumulate(frames: list[np.ndarray]) -> Pass1Stats:
    h, w = frames[0].shape[:2]
    acc = Pass1Accumulator(params=P, scale=scale1(), image_scale=1.0, frame_css=(w, h))
    for i, f in enumerate(frames):
        acc.add(i / FPS, f.mean(axis=2).astype(np.uint8))
    return acc.finish()


def test_activity_grid_shape_and_cells() -> None:
    assert activity_grid_shape((1280.0, 800.0), 16.0) == (50, 80)
    assert activity_grid_shape((904.0, 786.0), 16.0) == (50, 57)
    card = solid_rect_sprite(48, 32, (90, 90, 200))
    frames = []
    for i in range(20):
        img = base_canvas(320, 240)
        composite(img, card, translate(160 + (3 * i if i >= 10 else 0), 96))
        frames.append(img)
    stats = _accumulate(frames)
    act = stats.activity
    assert act is not None and act.shape == (20, 15, 20) and act.dtype == bool
    assert not act[:10].any()  # static (frame 0 has no diff)
    assert act[10:].any(axis=0)[6:8, 10:].any()  # card rows 96..128 → cell rows 6–7
    assert not act[10:].any(axis=0)[:5].any()  # nothing above the card
    # existing per-frame outputs are unaffected by the grid (same as before Track R)
    assert len(stats.blobs) == len(stats.slow_blobs) == len(stats.times) == 20
