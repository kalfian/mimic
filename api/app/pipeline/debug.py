"""``MIMIC_DEBUG`` dumps (PLAN §4.2 ``debug/``) + an M1-only inspection harness.

:class:`DebugDump` writes, when enabled (no-op otherwise):

* ``energy.csv`` — per pass-1 frame: t, UI energy, changed fraction, phase shift/response,
  cursor sample
* ``segments.json`` — segments, stable gaps, primary interaction, windows, crop, warnings
* ``elements.json`` — element candidates (CSS boxes, kinds, parents, A→B warps)
* ``masks/*.png`` — state A/B, change mask, per-element masks
* ``tracks/<element>_<segment>_<property>.csv`` — per-frame series
* continuous / ambient regions (PLAN-continuous §9 P2): ``masks/activity_mean.png`` (mean
  pass-1 cell activity), ``activity.csv`` (active cells per frame), ``regime.json``,
  ``scroller_<i>.json`` (phases, behaviour) + ``displacement_<i>.csv`` per analysed region,
  ``layout.json`` (rest frame time, scroller / card / title boxes)

``python -m app.pipeline.debug VIDEO [--out DIR] [--pixel-ratio auto|1|2|3] [--keyframes]``
runs the M1 stages only (scan → decode → regions → track → photometric → shadow → radius
[→ keyframes]) and writes the dumps plus ``summary.json`` with every series' endpoints.
It is a development tool, not the production pipeline (``run.py`` owns the wiring).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.measure import (
    ElementCandidate,
    ProbeInfo,
    PropertySeries,
    Rect,
    Scale,
    ScanResult,
    ScrollerAnalysis,
    WindowFrames,
)
from app.pipeline.params import DEFAULT_PARAMS, MeasureParams


def _rect(r: Rect | None) -> dict[str, float] | None:
    if r is None:
        return None
    return {"x": round(r.x, 2), "y": round(r.y, 2), "w": round(r.w, 2), "h": round(r.h, 2)}


def _num(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), 5)


class DebugDump:
    """Writes debug artifacts under ``directory``; every method is a no-op when disabled."""

    def __init__(self, directory: Path | None) -> None:
        self.dir = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _path(self, rel: str) -> Path:
        assert self.dir is not None
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, data: Any) -> None:
        if not self.enabled:
            return
        self._path(rel).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def write_csv(self, rel: str, header: list[str], rows: list[list[Any]]) -> None:
        if not self.enabled:
            return
        with self._path(rel).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def energy(self, scan: ScanResult, stats: Any | None = None) -> None:
        if not self.enabled:
            return
        with self._path("energy.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "energy_px2", "changed_frac", "shift_css", "response",
                        "cursor_x", "cursor_y", "cursor_state"])  # fmt: skip
            for i, t in enumerate(scan.frame_times):
                s = scan.cursor.samples[i] if i < len(scan.cursor.samples) else None
                w.writerow([
                    f"{t:.4f}", f"{scan.energy[i]:.1f}",
                    f"{stats.changed_frac[i]:.4f}" if stats is not None else "",
                    f"{stats.shift_css[i]:.2f}" if stats is not None else "",
                    f"{stats.response[i]:.3f}" if stats is not None else "",
                    "" if s is None or not math.isfinite(s.x_css) else f"{s.x_css:.1f}",
                    "" if s is None or not math.isfinite(s.y_css) else f"{s.y_css:.1f}",
                    "" if s is None else s.state,
                ])  # fmt: skip

    def segments(self, scan: ScanResult, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        def seg(s):
            if s is None:
                return None
            return {"start_s": round(s.start_s, 4), "end_s": round(s.end_s, 4),
                    "roi_css": _rect(s.roi_css), "peak_energy": round(s.peak_energy, 1),
                    "is_global": s.is_global}  # fmt: skip

        data = {
            "segments": [seg(s) for s in scan.segments],
            "stable": [
                {"start_s": round(s.start_s, 4), "end_s": round(s.end_s, 4)} for s in scan.stable
            ],  # fmt: skip
            "primary": {
                "forward": seg(scan.primary.forward),
                "reverse": seg(scan.primary.reverse),
                "round_trip": scan.primary.round_trip,
            },
            "cursor": {"visible": scan.cursor.visible, "confidence": scan.cursor.confidence},
            "warnings": [w.model_dump() for w in scan.warnings],
        }
        data.update(extra or {})
        self.write_json("segments.json", data)

    def elements(self, elements: list[ElementCandidate]) -> None:
        self.write_json("elements.json", [element_json(e) for e in elements])

    def image(self, name: str, img: np.ndarray) -> None:
        if not self.enabled:
            return
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        cv2.imwrite(str(self._path(f"masks/{name}.png")), img)

    def series(self, series: list[PropertySeries]) -> None:
        if not self.enabled:
            return
        for s in series:
            rel = f"tracks/{s.element_id}_{s.segment}_{s.property}.csv"
            with self._path(rel).open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "value", "quality"])
                for t, v, q in zip(s.times, s.values, s.quality, strict=True):
                    w.writerow([f"{t:.4f}", f"{v:.5f}", f"{q:.4f}"])


def scroller_json(sa: ScrollerAnalysis) -> dict[str, Any]:
    """One analysed ambient region (continuous mode debug dump)."""
    return {
        "region": _rect(sa.region.rect_css), "is_scroller": sa.is_scroller,
        "reason": sa.reason, "axis": sa.axis, "coherence": _num(sa.coherence),
        "viewport": _rect(sa.series.region) if sa.series is not None else None,
        "autoplay_velocity": _num(sa.autoplay_velocity) if sa.autoplay_velocity else None,
        "loop_period_px": _num(sa.loop_period_px) if sa.loop_period_px else None,
        "pitch_px": _num(sa.pitch_px) if sa.pitch_px else None,
        "gap_px": _num(sa.gap_px) if sa.gap_px else None,
        # card census (P2b): set when it supplied pitch / gap
        "card_census": {
            "card_len_px": _num(sa.card_len_px), "card_cross_px": _num(sa.card_cross_px),
            "zoom_per_px2": sa.card_zoom_per_px2, "zoom_reach_px": _num(sa.card_zoom_reach_px),
            "zoom_confidence": _num(sa.card_zoom_confidence),
        } if sa.card_len_px is not None else None,
        "phases": [
            {"kind": p.kind, "start_s": _num(p.start_s), "end_s": _num(p.end_s),
             "v_start": _num(p.v_start), "v_end": _num(p.v_end), "v_peak": _num(p.v_peak),
             "displacement": _num(p.displacement), "interrupted": p.interrupted,
             "fit": p.fit.model_dump() if p.fit is not None else None,
             "confidence": _num(p.confidence), "notes": p.notes}
            for p in sa.phases
        ],
        "behavior": sa.behavior.model_dump() if sa.behavior is not None else None,
        "warnings": [w.model_dump() for w in sa.warnings],
    }  # fmt: skip


def element_json(e: ElementCandidate) -> dict[str, Any]:
    return {
        "id": e.id, "kind": e.kind, "parent_id": e.parent_id,
        "bbox_a": _rect(e.bbox_a), "bbox_b": _rect(e.bbox_b),
        "change_energy": round(e.change_energy, 1), "text_like": e.text_like,
        "warp_ab": None if e.warp_ab is None else np.round(e.warp_ab, 4).tolist(),
        "notes": e.notes,
    }  # fmt: skip


def series_json(s: PropertySeries) -> dict[str, Any]:
    return {
        "element_id": s.element_id, "property": s.property, "segment": s.segment,
        "v_start": _num(s.v_start), "v_end": _num(s.v_end), "stable_std": _num(s.stable_std),
        "from": s.from_value.model_dump(), "to": s.to_value.model_dump(),
        "n": int(len(s.times)), "mean_quality": _num(float(np.mean(s.quality)))
        if len(s.quality) else None,
        "transform_origin": s.transform_origin, "notes": s.notes,
    }  # fmt: skip


# --------------------------------------------------------------------------------------------
# M1 harness
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class M1Result:
    probe: ProbeInfo
    scale: Scale
    scan: ScanResult
    windows: list[WindowFrames]
    elements: list[ElementCandidate]
    series: list[PropertySeries]
    warnings: list[Any] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def run_m1(
    video: Path,
    out_dir: Path | None,
    *,
    pixel_ratio: str = "auto",
    params: MeasureParams = DEFAULT_PARAMS,
    keyframes: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> M1Result:
    """Run the M1 measurement stages on one video and (optionally) dump everything."""
    from app.config import get_settings
    from app.pipeline import probe as probe_mod
    from app.pipeline.decode import (
        clipped_sides,
        crop_rect_css,
        decode_windows,
        expand_crop,
        frame_size_css,
        plan_windows,
    )
    from app.pipeline.keyframes import write_keyframes
    from app.pipeline.photometric import measure_photometric, refine_appear_endpoints
    from app.pipeline.radius import measure_radii
    from app.pipeline.regions import detect_elements, soft_change_mask, state_frames
    from app.pipeline.scan import scan_video
    from app.pipeline.shadow import measure_shadows
    from app.pipeline.track import derive_height_series, measure_geometry, segment_spans

    dump = DebugDump(out_dir)
    timings: dict[str, float] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(now - t, 3)
        t = now

    settings = get_settings()
    info = probe_mod.probe(video, settings, params)
    scale, warnings = probe_mod.resolve_scale(info, pixel_ratio, params)
    lap("probe")
    scan, stats = scan_video(video, info, scale, params, ffmpeg_bin)
    warnings += scan.warnings
    lap("scan")
    specs, w = plan_windows(scan, info, params)
    warnings += w
    prim = scan.primary
    rois = [prim.forward.roi_css] + ([prim.reverse.roi_css] if prim.reverse else [])
    crop_css, crop_px = crop_rect_css(rois, info, scale, params)
    wfs, w = decode_windows(video, info, scale, specs, crop_css, crop_px, params,
                            cursor=scan.cursor, pass1_times=scan.frame_times,
                            ffmpeg_bin=ffmpeg_bin)  # fmt: skip
    warnings += w
    lap("decode")
    fwd_wf = wfs[0]
    st = state_frames(fwd_wf, prim.forward.start_s, prim.forward.end_s, params,
                      round_trip=prim.round_trip)  # fmt: skip
    # faint elements can extend past the pass-1 ROI: grow the crop once and decode again
    sides = clipped_sides(soft_change_mask(st.a, st.b, st.valid, params), crop_css, info, scale)
    if sides:
        crop_css, crop_px = expand_crop(crop_css, sides, info, scale, params)
        wfs, _ = decode_windows(video, info, scale, specs, crop_css, crop_px, params,
                                cursor=scan.cursor, pass1_times=scan.frame_times,
                                ffmpeg_bin=ffmpeg_bin)  # fmt: skip
        fwd_wf = wfs[0]
        st = state_frames(fwd_wf, prim.forward.start_s, prim.forward.end_s, params,
                          round_trip=prim.round_trip)  # fmt: skip
        timings["crop_expanded"] = 1.0
    rr = detect_elements(fwd_wf, st, frame_size_css(info, scale), params)
    warnings += rr.warnings
    lap("regions")
    windows = []
    for wf in wfs:
        if prim.round_trip:
            peak_s = float(fwd_wf.times[st.peak_idx]) if st.peak_idx is not None else None
            windows.append((wf, segment_spans(wf, prim.forward.start_s, prim.forward.end_s,
                                              peak_s=peak_s)))  # fmt: skip
        else:
            seg = prim.forward if wf.segment_id == "fwd" else prim.reverse
            windows.append((wf, segment_spans(wf, seg.start_s, seg.end_s)))
    geo = measure_geometry(rr.elements, rr.boxes_px, rr.warps_px, st.a, st.b, st.valid,
                           windows, params)  # fmt: skip
    lap("track")
    pho = measure_photometric(rr.elements, rr.element_masks, rr.boxes_px, rr.warps_px,
                              rr.overlay, st.a, st.b, st.valid, windows, params,
                              tracks=geo.tracks)  # fmt: skip
    lap("photometric")
    sh = measure_shadows(rr.elements, rr.boxes_px, rr.warps_px, rr.element_masks, st.a, st.b,
                         windows, params, tracks=geo.tracks)  # fmt: skip
    rad = measure_radii(rr.elements, rr.boxes_px, rr.warps_px, st.a, st.b, windows, params,
                        tracks=geo.tracks)  # fmt: skip
    lap("shadow_radius")
    refine_appear_endpoints(geo.series + pho.series)
    heights = derive_height_series(rr.resize_links, geo.series)
    series = geo.series + heights + pho.series + sh.series + rad.series

    kf_files = []
    if keyframes and out_dir is not None:
        stable_mid = [(s.start_s + s.end_s) / 2 for s in scan.stable]
        a_s = max([m for m in stable_mid if m < prim.forward.start_s], default=0.0)
        b_s = min([m for m in stable_mid if m > prim.forward.end_s], default=prim.forward.end_s)
        f = prim.forward
        mid = {p: f.start_s + (f.end_s - f.start_s) * p / 100 for p in (25, 50, 75)}
        kf_files = write_keyframes(video, out_dir / "keyframes", info, scale, state_a_s=a_s,
                                   state_b_s=b_s, mid_s=mid, elements=rr.elements,
                                   params=params, ffmpeg_bin=ffmpeg_bin)  # fmt: skip
        lap("keyframes")

    extras = {
        "shadow_static": {k: (v.model_dump() if v else None) for k, v in sh.static.items()},
        "shadow_change": {k: {"darkening": round(v.darkening, 2), "coverage": round(v.coverage, 3),
                              "increases": v.increases} for k, v in sh.changes.items()},
        "radius_static": {k: {"radius_css": round(v.radius_css, 2), "corners": v.corners}
                          for k, v in rad.static.items()},
        "colors": {k: {"a": v.hex_a, "b": v.hex_b, "delta_e": round(v.delta_e, 2)}
                   for k, v in pho.colors.items()},
        "geometry_hints": geo.warnings,
        "keyframes": [f.name for f in kf_files],
    }  # fmt: skip
    if dump.enabled:
        dump.energy(scan, stats)
        dump.segments(scan, {
            "windows": [{"segment_id": wf.segment_id, "n": len(wf.times),
                         "t0": round(float(wf.times[0]), 4), "t1": round(float(wf.times[-1]), 4),
                         "dups": int(wf.is_dup.sum())} for wf in wfs],
            "crop_css": _rect(crop_css),
            "scale": {"pixel_ratio": scale.pixel_ratio, "source": scale.pixel_ratio_source,
                      "analysis_scale": scale.analysis_scale},
            "warnings_all": [w.model_dump() for w in warnings],
            "timings_s": timings,
        })  # fmt: skip
        dump.elements(rr.elements)
        dump.image("state_a", st.a)
        dump.image("state_b", st.b)
        dump.image("change_mask", rr.change_mask)
        for eid, m in rr.element_masks.items():
            dump.image(f"el_{eid}", m)
        dump.series(series)
        dump.write_json("summary.json", {
            "elements": [element_json(e) for e in rr.elements],
            "series": [series_json(s) for s in series],
            **extras,
        })  # fmt: skip
    return M1Result(info, scale, scan, wfs, rr.elements, series, warnings, timings, extras)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="debug dir (default: <video>.debug)")
    ap.add_argument("--pixel-ratio", default="auto", choices=["auto", "1", "2", "3"])
    ap.add_argument("--keyframes", action="store_true")
    ns = ap.parse_args(argv)
    out = ns.out or ns.video.with_suffix(ns.video.suffix + ".debug")
    res = run_m1(ns.video, out, pixel_ratio=ns.pixel_ratio, keyframes=ns.keyframes)
    p = res.scan.primary
    print(f"segments: {[(round(s.start_s, 3), round(s.end_s, 3)) for s in res.scan.segments]}")
    print(f"primary: fwd={p.forward.start_s:.3f}-{p.forward.end_s:.3f} "
          f"rev={'yes' if p.reverse else 'no'} round_trip={p.round_trip}")  # fmt: skip
    for e in res.elements:
        print(f"  {e.id} {e.kind:<15} parent={e.parent_id} A={_rect(e.bbox_a)} "
              f"text={e.text_like}")  # fmt: skip
    for s in res.series:
        print(f"  {s.element_id} {s.segment:<6} {s.property:<16} {s.v_start:9.4f} -> "
              f"{s.v_end:9.4f}  n={len(s.times)}")  # fmt: skip
    print(f"warnings: {[w.code for w in res.warnings]}")
    print(f"timings: {res.timings}  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
