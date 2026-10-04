"""Round-trip harness (PLAN-continuous §14): replica HTML + source IR → replica video → replica IR.

    uv run python scripts/roundtrip.py INDEX.html SOURCE.json [--out DIR]
        [--capture virtual|screencast] [--fps 60] [--warmup-ms 500] [--draw-cursor]
        [--keyframes] [--no-compare] [--plan-only]

1. **Plan** (:func:`build_plan`, pure): viewport (CSS px) and device pixel ratio from the source IR,
   capture length = source duration, and a mouse script derived from the IR:
   * continuous — the pointer rests outside the scroller; for each ``drag`` phase it presses inside
     the scroller at the phase start, moves along the axis so the content (which follows the
     pointer 1:1) travels the phase's displacement, and releases at the phase end moving at the
     phase's release velocity ``v_end`` (velocity profile: smoothstep acceleration then a constant
     ``v_end`` tail, the synthetic suite's drag model; a general hump + constant tail when that
     model can't reach ``v_end``). After release it leaves at once unless a hover pause holds it.
     A ``decelerate`` phase (pause on hover / unknown trigger) is replayed as the pointer entering
     at the phase start and leaving at the next resume start minus the measured
     ``delay_after_leave_ms`` (0 if unknown); with a ``press`` trigger the button goes down there.
   * transition — wait through the stable lead; at the forward trigger (``trigger_event_ms``, else
     the onset) enter the target's measured box (hover), click it, or press it; at the reverse
     trigger leave / click again / release.
   All input lands on frame boundaries (60 fps → 16.67 ms grid), like a real 60 Hz pointer.
2. **Capture** (``tools/roundtrip/capture.mjs``, Node built-ins only): launches headless Chrome
   (Playwright's cached chrome-headless-shell, or ``$MIMIC_CHROME``) on a free DevTools port with a
   throw-away profile, opens the page at the viewport/DPR, replays the input and writes one PNG
   per frame. Default ``virtual`` capture is deterministic (virtual time + per-frame rAF and CSS
   animation clock, see capture.mjs); ``screencast`` is the real-time fallback (VFR timestamps).
3. **Encode** with ffmpeg: H.264 CRF 18, constant 60 fps for virtual capture, real PTS (concat
   demuxer durations) for screencast.
4. **Analyze** the replica with the same pipeline as ``scripts/analyze.py`` (pixel ratio = the
   source's) → ``replica.ir.json``; then **compare** with ``scripts/compare_roundtrip.py``.

Everything goes to ``--out`` (default ``/tmp/mimic-roundtrip/<html stem>-<time>``): plan.json,
frames/, replica.mp4, replica.ir.json, compare.json. Exit 0 = compare PASS, 1 = FAIL,
2 = usage/setup error, 3 = capture/encode/analysis failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
API_DIR = SCRIPTS_DIR.parent
REPO_ROOT = API_DIR.parent
CAPTURE_JS = REPO_ROOT / "tools" / "roundtrip" / "capture.mjs"
DEFAULT_OUT_ROOT = Path("/tmp/mimic-roundtrip")

FPS = 60.0
WARMUP_MS = 500.0
#: Inset (CSS px) kept between pointer positions and the scroller / target edges.
EDGE_INSET = 12.0
#: Distance (CSS px) of the parked pointer from the scroller / target box.
PARK_GAP = 48.0
#: Constant-velocity tail before release when the smoothstep drag model can't reach v_end.
TAIL_S = 0.08
#: A drag whose release velocity is below this (px/s) ends with the pointer stopped (the content
#: comes to rest with it): replayed as a smooth hump ending at rest, then held still.
STOP_RELEASE_PX_S = 30.0
#: Before letting go of a drag that ended at rest, the pointer is held still this long (when the
#: following rest is that long), longer than the release-velocity window the prompt asks for
#: (half the momentum τ, ≈130 ms on the motivating recording), so the release carries no velocity.
RELEASE_HOLD_MS = 150.0


# --------------------------------------------------------------------------------------------
# Plan (pure)
# --------------------------------------------------------------------------------------------


def spec_of(data: dict[str, Any]) -> dict[str, Any]:
    if "error" in data and isinstance(data["error"], dict):
        raise ValueError(
            f"source IR is an error body ({data['error'].get('code')}); nothing to replay"
        )
    return data["spec"] if "spec" in data else data


def viewport_of(spec: dict[str, Any]) -> tuple[int, int, int]:
    """``(w, h, dpr)`` in CSS px from ``scene.viewport_css`` (else source size / pixel ratio)."""
    src = spec["source"]
    dpr = int(src.get("pixel_ratio") or 1)
    vp = (spec.get("scene") or {}).get("viewport_css")
    if vp:
        return int(round(vp["w"])), int(round(vp["h"])), dpr
    return int(round(src["width"] / dpr)), int(round(src["height"] / dpr)), dpr


def frame_t(k: int, fps: float = FPS) -> float:
    """Time (ms) of frame ``k``; same µs rounding as capture.mjs."""
    return round(k * 1_000_000 / fps) / 1000.0


def frame_of(t_ms: float, fps: float = FPS) -> int:
    return int(round(t_ms * fps / 1000.0))


def _inside(p: tuple[float, float], w: float, h: float, margin: float = 4.0) -> bool:
    return margin <= p[0] <= w - margin and margin <= p[1] <= h - margin


def park_point(box: tuple[float, float, float, float], vw: float, vh: float) -> tuple[float, float]:
    """A pointer rest position outside ``box`` (x, y, w, h) but inside the viewport."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    for p in (
        (cx, y + h + PARK_GAP),
        (cx, y - PARK_GAP),
        (x + w + PARK_GAP, cy),
        (x - PARK_GAP, cy),
        (vw - 6.0, vh - 6.0),
        (6.0, vh - 6.0),
    ):
        outside = not (x - 2 <= p[0] <= x + w + 2 and y - 2 <= p[1] <= y + h + 2)
        if outside and _inside(p, vw, vh):
            return (round(p[0], 1), round(p[1], 1))
    return (round(min(vw - 2.0, x + w + 2), 1), round(min(vh - 2.0, y + h + 2), 1))


def _smoothstep_int(q: float) -> float:
    """∫_0^q smoothstep(s) ds (unit interval)."""
    q = min(max(q, 0.0), 1.0)
    return q**3 - q**4 / 2.0


def drag_offsets(
    distance: float, v_release: float, dur_s: float, times_s: list[float], *, stop: bool = False
) -> list[float]:
    """Pointer offsets (px) at ``times_s`` (0..dur_s) of a drag covering ``distance`` and ending at
    ``v_release`` px/s.

    Preferred model (= the synthetic suite's ``Drag``): smoothstep acceleration over ``a·T`` then
    constant ``v_release``, ``a = 2 (1 − D / (v T))``; valid for ``|D|/T ≤ |v| ≤ 2|D|/T`` with the
    same sign. Otherwise: a constant ``v_release`` tail of :data:`TAIL_S` (≤ 30 % of T) and before
    it ``v_release·smoothstep(u/Th) + B·sin²(πu/Th)`` with ``B`` chosen so the total is ``D``.
    ``stop``: the pointer comes to rest exactly at ``T`` (a ``2D/T · sin²(πu/T)`` hump, no tail):
    a drag that ends with the pointer stopped.
    """
    T, D, v = dur_s, distance, v_release
    out: list[float] = []
    if T <= 0:
        return [D for _ in times_s]
    if stop:
        return [
            D * (q - math.sin(2.0 * math.pi * q) / (2.0 * math.pi))
            for q in (min(max(t, 0.0), T) / T for t in times_s)
        ]
    feasible = v != 0 and D != 0 and math.copysign(1, D) == math.copysign(1, v)
    a = 2.0 * (1.0 - D / (v * T)) if feasible else -1.0
    if feasible and 0.0 < a <= 1.0:
        aT = a * T
        for t in times_s:
            u = min(max(t, 0.0), T)
            if u < aT:
                out.append(v * aT * _smoothstep_int(u / aT))
            else:
                out.append(v * (aT / 2.0 + (u - aT)))
        return out
    h = min(TAIL_S, 0.3 * T)
    th = T - h
    head = D - v * h
    b = 2.0 * head / th - v
    for t in times_s:
        u = min(max(t, 0.0), T)
        if u <= th:
            q = u / th
            hump = u / 2.0 - th / (4.0 * math.pi) * math.sin(2.0 * math.pi * q)
            out.append(v * th * _smoothstep_int(q) + b * hump)
        else:
            out.append(head + v * (u - th))
    return out


#: Sample-driven drags: the IR's ``samples`` positions are smoothed with a centred moving
#: average of ± this many ms (the ends keep their values) before they become pointer positions;
#: samples tracked with a quality below ``SAMPLE_MIN_Q`` are left out (fast-motion blur: a
#: recorder clock beating against the page's frames zig-zags single samples).
SAMPLE_SMOOTH_MS = 30.0
SAMPLE_MIN_Q = 0.5
#: Drags the source flags as degraded ("tracking degraded during fast motion") are smoothed
#: over ± this many ms: their fine velocity structure is tracking noise, not the hand.
SAMPLE_SMOOTH_DEGRADED_MS = 60.0
NOTE_DEGRADED = "tracking degraded during fast motion"
#: Frames over which the replayed speed of a drag that ends at rest tapers to zero.
STOP_TAPER_FRAMES = 3


def sample_offsets(
    samples: list[list[float]],
    t0_ms: float,
    t1_ms: float,
    times_ms: list[float],
    distance: float,
    *,
    smooth_ms: float = SAMPLE_SMOOTH_MS,
) -> list[float] | None:
    """Pointer offsets at ``times_ms`` replaying the content's own measured path between
    ``t0_ms`` and ``t1_ms`` (IR ``continuous.samples`` = ``[t_ms, v, pos, q]``): while pressed the
    content follows the pointer, so its displacement profile is the pointer's (velocity build-up,
    the slowdown before letting go, the release speed). Rescaled so the total is ``distance``
    (the phase's displacement). ``None`` when the samples do not cover the drag densely enough.
    """
    near = [s for s in samples if t0_ms - 40.0 <= s[0] <= t1_ms + 40.0]
    pts = [
        (float(s[0]), float(s[2]))
        for k, s in enumerate(near)
        if k in (0, len(near) - 1) or float(s[3]) >= SAMPLE_MIN_Q
    ]
    if len(pts) < 4 or pts[0][0] > t0_ms + 20.0 or pts[-1][0] < t1_ms - 20.0:
        return None
    ts = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    sm = []
    for ti in ts:
        h = min(smooth_ms, ti - ts[0], ts[-1] - ti)  # symmetric: the ends stay put
        win = [xs[j] for j in range(len(ts)) if abs(ts[j] - ti) <= h + 1e-9]
        sm.append(sum(win) / len(win))

    def at(t: float) -> float:
        if t <= ts[0]:
            return sm[0]
        for i in range(1, len(ts)):
            if t <= ts[i]:
                f = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
                return sm[i - 1] + f * (sm[i] - sm[i - 1])
        return sm[-1]

    total = at(t1_ms) - at(t0_ms)
    if abs(total) < 1e-6 or total * distance <= 0:
        return None
    # per-frame steps along the drag direction, made unimodal (speed rises to one peak, then
    # only falls): the IR describes ONE drag, so dips inside it (hand jitter, or tracking noise
    # the source analysis did not split on) are not replayed as extra release / re-grab pairs
    sign = 1.0 if total > 0 else -1.0
    grid = [min(max(t, t0_ms), t1_ms) for t in times_ms]
    prev = [t0_ms, *grid[:-1]]
    steps = [max(sign * (at(b) - at(a)), 0.0) for a, b in zip(prev, grid, strict=True)]
    steps = unimodal(steps)
    # the drag ends with the pointer at rest: the measured phase end is where the content's
    # speed fell into the noise, a frame or two before it truly stops; taper the last frames
    # to zero so the replica slows down instead of stopping dead (a dead stop reads as an
    # abrupt "stop" phase)
    for k in range(1, min(STOP_TAPER_FRAMES, len(steps) - 1) + 1):
        steps[-k] *= k / (STOP_TAPER_FRAMES + 1)
    acc, out = 0.0, []
    norm = sum(steps)
    if norm <= 0:
        return None
    for st in steps:
        acc += st
        out.append(sign * acc / norm * abs(distance))
    return out


def _pava(y: list[float]) -> list[float]:
    """Non-decreasing least-squares fit (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [sum, count]
    for v in y:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s1, n1 = blocks.pop()
            blocks[-1][0] += s1
            blocks[-1][1] += n1
    out: list[float] = []
    for s_, n_ in blocks:
        out += [s_ / n_] * int(n_)
    return out


def unimodal(y: list[float]) -> list[float]:
    """Least-squares unimodal (rise, then fall) fit with the mode at the largest value."""
    if len(y) < 3:
        return list(y)
    m = max(range(len(y)), key=lambda i: y[i])
    up = _pava(y[: m + 1])
    down = _pava(y[m:][::-1])[::-1]
    return up[:-1] + [max(up[-1], down[0])] + down[1:]


class _Script:
    """Accumulates timed mouse events and tracks the pointer state."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.events: list[dict[str, Any]] = []
        self._seq = 0

    def add(self, k_or_t: int | float, kind: str, p: tuple[float, float], *, buttons: int = 0,
            exact_t: bool = False) -> None:  # fmt: skip
        t = float(k_or_t) if exact_t else frame_t(int(k_or_t), self.fps)
        self._seq += 1
        self.events.append(
            {
                "t_ms": round(t, 3),
                "type": kind,
                "x": round(p[0], 2),
                "y": round(p[1], 2),
                "buttons": buttons,
                "_seq": self._seq,
            }
        )

    def finish(self) -> list[dict[str, Any]]:
        evs = sorted(self.events, key=lambda e: (e["t_ms"], e["_seq"]))
        for e in evs:
            e.pop("_seq", None)
        return evs


def _num(x: Any) -> float | None:
    if x is None:
        return None
    return float(x["value"]) if isinstance(x, dict) else float(x)


def plan_continuous(spec: dict[str, Any], fps: float, warmup_ms: float) -> tuple[list, list[str]]:
    c = spec["continuous"]
    vw, vh, _ = viewport_of(spec)
    axis = c["axis"]
    reg = c["region"]
    rbox = (reg["x"], reg["y"], reg["w"], reg["h"])
    park = park_point(rbox, vw, vh)
    a0, alen = (reg["x"], reg["w"]) if axis == "x" else (reg["y"], reg["h"])
    cross = reg["y"] + reg["h"] / 2 if axis == "x" else reg["x"] + reg["w"] / 2
    lo, hi = a0 + EDGE_INSET, a0 + alen - EDGE_INSET
    center_along = a0 + alen / 2

    def pt(along: float) -> tuple[float, float]:
        return (along, cross) if axis == "x" else (cross, along)

    phases = sorted(c["phases"], key=lambda p: p["start_ms"])
    samples = c.get("samples") or []
    beh = c.get("behavior") or {}
    pause_on = (beh.get("pause") or {}).get("on")
    leave_delay = _num((beh.get("resume") or {}).get("delay_after_leave_ms"))
    notes: list[str] = []
    s = _Script(fps)
    s.add(-warmup_ms, "move", park, exact_t=True)

    # drags: press point, frames, offsets. A drag that ends at rest (|v_end| below
    # STOP_RELEASE_PX_S: the pointer stopped before letting go) is replayed as a hump ending at
    # rest; when a rest follows it the pointer stays pressed and still: through the rest into the
    # next drag (one continuous press), else for RELEASE_HOLD_MS (at most the rest) before release
    drags = []
    for i, ph in enumerate(phases):
        if ph["kind"] != "drag":
            continue
        k0 = frame_of(ph["start_ms"], fps)
        k1 = max(frame_of(ph["end_ms"], fps), k0 + 2)
        D = float(ph["displacement_px"])
        v_end = float(ph["v_end_px_s"])
        stop = abs(v_end) < STOP_RELEASE_PX_S
        nxt = phases[i + 1] if i + 1 < len(phases) else None
        after = phases[i + 2] if i + 2 < len(phases) else None
        k_up, hold_into = k1, False
        if stop and nxt is not None and nxt["kind"] == "paused":
            if after is not None and after["kind"] == "drag":
                hold_into = True
            else:
                rest_ms = nxt["end_ms"] - nxt["start_ms"]
                k_up = k1 + max(frame_of(min(RELEASE_HOLD_MS, rest_ms - 1000.0 / fps), fps), 0)
        prev = drags[-1] if drags else None
        if prev is not None and prev["hold_into"]:
            p0 = prev["p0"] + prev["D"]  # the same press continues where the pointer rests
        elif abs(D) <= hi - lo:
            p0 = min(max(center_along - D / 2.0, lo), hi)
            p0 = min(max(p0, lo - min(D, 0.0)), hi - max(D, 0.0))
        else:
            p0 = lo if D > 0 else hi
        if not lo - 1e-6 <= p0 + D <= hi + 1e-6 or not lo - 1e-6 <= p0 <= hi + 1e-6:
            notes.append(f"drag {ph['id']}: {D:+.0f} px from {p0:.0f}: the pointer leaves the "
                         "scroller before release (pointer capture assumed)")  # fmt: skip
        drags.append({"ph": ph, "k0": k0, "k1": k1, "D": D, "v": v_end, "p0": p0, "stop": stop,
                      "k_up": k_up, "hold_into": hold_into,
                      "continued": prev is not None and prev["hold_into"]})  # fmt: skip

    # hover / press-hold windows from decelerate phases
    windows = []  # (k_enter, k_leave | None, point, press)
    for i, ph in enumerate(phases):
        if ph["kind"] != "decelerate":
            continue
        k_enter = frame_of(ph["start_ms"], fps)
        nxt_resume = next((q for q in phases[i + 1 :] if q["kind"] == "resume"), None)
        inner = [d for d in drags if d["k0"] >= k_enter and (
            nxt_resume is None or d["ph"]["start_ms"] < nxt_resume["start_ms"])]  # fmt: skip
        point = pt(inner[0]["p0"]) if inner else pt(center_along)
        if nxt_resume is None:
            k_leave = None
        else:
            delay = leave_delay if (pause_on == "hover" and leave_delay is not None) else 0.0
            k_leave = frame_of(nxt_resume["start_ms"] - delay, fps)
            if inner:
                k_leave = max(k_leave, inner[-1]["k1"] + 1)
        press = pause_on == "press"
        windows.append({"k_enter": k_enter, "k_leave": k_leave, "point": point, "press": press,
                        "drags": inner})  # fmt: skip
        if pause_on in (None, "unknown"):
            notes.append(f"{ph['id']} decelerate with trigger '{pause_on or 'none'}': replayed as "
                         "hover (pointer enters at the phase start)")  # fmt: skip

    def window_at(k: int) -> dict[str, Any] | None:
        for w in windows:
            if w["k_enter"] <= k and (w["k_leave"] is None or k < w["k_leave"]):
                return w
        return None

    for w in windows:
        if w["press"]:
            # enter and press on the same frame: a page that pauses on hover must not see a
            # hover frame before the press the source recorded
            s.add(w["k_enter"], "move", w["point"])
            s.add(w["k_enter"], "down", w["point"], buttons=1)
            if not w["drags"] and w["k_leave"] is not None:
                s.add(w["k_leave"], "up", w["point"])
                s.add(w["k_leave"] + 1, "move", park)
        else:
            s.add(w["k_enter"], "move", w["point"])
            if w["k_leave"] is not None:
                s.add(w["k_leave"], "move", park)

    for d in drags:
        w = window_at(d["k0"])
        held = (w is not None and w["press"] and d is w["drags"][0]) or d["continued"]
        start = pt(d["p0"])
        if not held:
            # enter and press on the same frame (no hover frame before the press, see above)
            s.add(d["k0"], "move", start)
            s.add(d["k0"], "down", start, buttons=1)
        ks = list(range(d["k0"] + 1, d["k1"] + 1))
        dur_s = (frame_t(d["k1"], fps) - frame_t(d["k0"], fps)) / 1000.0
        ts = [(frame_t(k, fps) - frame_t(d["k0"], fps)) / 1000.0 for k in ks]
        offs = None
        # a drag that ends with the pointer stopped is replayed from the content's own measured
        # path (its slowdown is part of the drag, the parametric hump only approximates it);
        # flings keep the validated parametric model (smoothstep + release-speed tail)
        if samples and d["stop"]:
            t0 = frame_t(d["k0"], fps)
            degraded = NOTE_DEGRADED in (d["ph"].get("notes") or [])
            offs = sample_offsets(
                samples, t0, frame_t(d["k1"], fps), [t0 + u * 1000.0 for u in ts], d["D"],
                smooth_ms=SAMPLE_SMOOTH_DEGRADED_MS if degraded else SAMPLE_SMOOTH_MS,
            )  # fmt: skip
        if offs is None:
            offs = drag_offsets(d["D"], d["v"], dur_s, ts, stop=d["stop"])
        last = start
        for k, off in zip(ks, offs, strict=True):
            last = pt(d["p0"] + off)
            s.add(k, "move", last, buttons=1)
        if d["hold_into"]:
            continue  # still pressed: the next drag continues this press
        s.add(d["k_up"], "up", last)
        if window_at(d["k_up"] + 1) is None:
            s.add(d["k_up"] + 1, "move", park)
    if pause_on == "hover" and not windows:
        notes.append("pause on hover reported but no decelerate phase: nothing to replay")
    return s.finish(), notes


def plan_transition(spec: dict[str, Any], fps: float, warmup_ms: float) -> tuple[list, list[str]]:
    vw, vh, _ = viewport_of(spec)
    inter = spec["interaction"]
    els = {e["id"]: e for e in spec.get("elements") or []}
    target = els.get(inter.get("target_element_id") or "") or next(iter(els.values()), None)
    notes: list[str] = []
    if target is None:
        raise ValueError("source IR has no elements to interact with")
    bi, ba = target["bbox_initial"], target["bbox_active"]
    x0, y0 = min(bi["x"], ba["x"]), min(bi["y"], ba["y"])
    x1 = max(bi["x"] + bi["w"], ba["x"] + ba["w"])
    y1 = max(bi["y"] + bi["h"], ba["y"] + ba["h"])
    # a point inside both boxes when possible (stays hovered through the transform)
    ix0, iy0 = max(bi["x"], ba["x"]), max(bi["y"], ba["y"])
    ix1, iy1 = min(bi["x"] + bi["w"], ba["x"] + ba["w"]), min(bi["y"] + bi["h"], ba["y"] + ba["h"])
    if ix1 - ix0 > 2 and iy1 - iy0 > 2:
        point = ((ix0 + ix1) / 2, (iy0 + iy1) / 2)
    else:
        point = (bi["x"] + bi["w"] / 2, bi["y"] + bi["h"] / 2)
    park = park_point((x0, y0, x1 - x0, y1 - y0), vw, vh)
    segs = {sg["id"]: sg for sg in spec.get("segments") or []}

    def trig(sg: dict[str, Any] | None) -> float | None:
        if sg is None:
            return None
        t = sg.get("trigger_event_ms")
        return float(t if t is not None else sg["onset_ms"])

    fwd = segs.get("fwd") or segs.get("rt_in")
    rev = segs.get("rev") or segs.get("rt_out")
    t_f, t_r = trig(fwd), trig(rev)
    if t_f is None:
        raise ValueError("source IR has no forward segment to replay")
    itype, tk, rk = inter["type"], inter["trigger"]["kind"], inter["trigger"]["reverse_kind"]
    if itype == "press":
        kind = "press"
    elif itype in ("click", "expand_collapse", "dropdown", "modal") or tk == "click":
        kind = "click"
    else:
        kind = "hover"
        if tk == "unknown":
            notes.append(f"trigger unknown ({itype}): replayed as hover")
    s = _Script(fps)
    s.add(-warmup_ms, "move", park, exact_t=True)
    kf = frame_of(t_f, fps)
    if kind == "hover":
        s.add(kf, "move", point)
        if t_r is not None and rk != "none":
            s.add(frame_of(t_r, fps), "move", park)
    elif kind == "click":
        enters = [ev["t_ms"] for ev in (spec.get("cursor") or {}).get("events") or []
                  if ev["kind"] == "enter" and ev["t_ms"] <= t_f]  # fmt: skip
        k_in = frame_of(enters[-1], fps) if enters else kf - 1
        s.add(min(k_in, kf - 1), "move", point)
        s.add(kf, "down", point, buttons=1)
        s.add(kf, "up", point)
        if t_r is not None:
            kr = frame_of(t_r, fps)
            if rk == "click":
                s.add(kr, "down", point, buttons=1)
                s.add(kr, "up", point)
                notes.append("reverse click replayed on the same point as the forward click")
            elif rk == "pointer_leave":
                s.add(kr, "move", park)
    else:  # press
        s.add(kf - 1, "move", point)
        s.add(kf, "down", point, buttons=1)
        if t_r is not None:
            s.add(frame_of(t_r, fps), "up", point)
            s.add(frame_of(t_r, fps) + 30, "move", park)
    return s.finish(), notes


def build_plan(
    source: dict[str, Any], *, fps: float = FPS, warmup_ms: float = WARMUP_MS,
    draw_cursor: bool = False, capture: str = "virtual",
) -> dict[str, Any]:  # fmt: skip
    """Capture plan (without browser/url/output paths) for a source IR envelope or spec."""
    spec = spec_of(source)
    vw, vh, dpr = viewport_of(spec)
    mode = spec.get("mode", "transition")
    if mode == "continuous":
        events, notes = plan_continuous(spec, fps, warmup_ms)
    else:
        events, notes = plan_transition(spec, fps, warmup_ms)
    dur = float(spec["source"]["duration_ms"])
    return {
        "mode": mode,
        "viewport": {"w": vw, "h": vh},
        "dpr": dpr,
        "fps": fps,
        "frames": max(int(dur * fps / 1000.0), 2),
        "warmup_ms": warmup_ms,
        "capture": capture,
        "draw_cursor": draw_cursor,
        "events": events,
        "notes": notes,
    }


# --------------------------------------------------------------------------------------------
# Browser / ffmpeg / pipeline
# --------------------------------------------------------------------------------------------


def find_chrome() -> tuple[str, str | None]:
    """``(binary, headless flag)``: ``$MIMIC_CHROME``, else Playwright's cached
    chrome-headless-shell (headless by design), else Playwright Chromium / Chrome for Testing /
    Google Chrome in new headless mode."""
    env = os.environ.get("MIMIC_CHROME")
    if env:
        flag = None if "headless-shell" in env or "headless_shell" in env else "--headless=new"
        return env, flag
    caches = [
        Path.home() / "Library" / "Caches" / "ms-playwright",
        Path.home() / ".cache" / "ms-playwright",
    ]
    for root in caches:
        shells = sorted(
            glob.glob(str(root / "chromium_headless_shell-*" / "*" / "chrome-headless-shell"))
        )
        if shells:
            return shells[-1], None
    for root in caches:
        for pat in ("chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/"
                    "Google Chrome for Testing",
                    "chromium-*/chrome-linux*/chrome"):  # fmt: skip
            found = sorted(glob.glob(str(root / pat)))
            if found:
                return found[-1], "--headless=new"
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium"):  # fmt: skip
        if Path(p).exists():
            return p, "--headless=new"
    for name in ("chromium", "chromium-browser", "google-chrome"):
        p = shutil.which(name)
        if p:
            return p, "--headless=new"
    raise FileNotFoundError("no headless Chrome found (install Playwright's chrome-headless-shell "
                            "or set MIMIC_CHROME)")  # fmt: skip


def run_capture(plan: dict[str, Any], html: Path, out_dir: Path, *, timeout_s: float = 900) -> dict:
    node = shutil.which("node")
    if node is None:
        raise FileNotFoundError("node is required (Node >= 22 with built-in WebSocket)")
    chrome, flag = find_chrome()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    full = {**plan, "chrome": chrome, "headless_flag": flag, "url": html.resolve().as_uri(),
            "out_dir": str(frames_dir), "tmp_dir": "/tmp"}  # fmt: skip
    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(full, indent=1) + "\n", "utf-8")
    proc = subprocess.run([node, str(CAPTURE_JS), str(plan_path)], capture_output=True, text=True,
                          timeout=timeout_s)  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(f"capture failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    meta = json.loads((frames_dir / "frames.json").read_text("utf-8"))
    meta["stdout"] = proc.stdout.strip()
    return meta


def encode(frames_dir: Path, meta: dict[str, Any], out_mp4: Path) -> None:
    """PNG frames → H.264 MP4: constant fps for virtual capture, real PTS for screencast."""
    frames = meta["frames"]
    if not frames:
        raise RuntimeError("no frames captured")
    # same encoder settings and BT.709 tagging as the synthetic suite / macOS screen recordings
    vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    common = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
              "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
              "-color_range", "tv", "-vf", vf, "-movflags", "+faststart"]  # fmt: skip
    if meta.get("capture") == "virtual":
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{meta['fps']:g}",
               "-i", str(frames_dir / "f_%06d.png"), *common, "-r", f"{meta['fps']:g}",
               str(out_mp4)]  # fmt: skip
    else:
        lines = ["ffconcat version 1.0"]
        for a, b in zip(frames, [*frames[1:], None], strict=True):
            dur = ((b["t_ms"] - a["t_ms"]) if b else 1000.0 / meta["fps"]) / 1000.0
            lines += [f"file '{a['file']}'", f"duration {max(dur, 0.001):.6f}"]
        lines.append(f"file '{frames[-1]['file']}'")
        lst = frames_dir / "frames.ffconcat"
        lst.write_text("\n".join(lines) + "\n", "utf-8")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            *common,
            "-fps_mode",
            "vfr",
            "-video_track_timescale",
            "90000",
            str(out_mp4),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")


def analyze_replica(video: Path, dpr: int, *, keyframes_dir: Path | None) -> dict[str, Any]:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from analyze import analyze_file  # noqa: PLC0415

    data, _ = analyze_file(
        video, pixel_ratio=str(dpr), keyframes_dir=keyframes_dir,
        write_keyframes=keyframes_dir is not None,
    )  # fmt: skip
    return data


def default_out(html: Path) -> Path:
    return DEFAULT_OUT_ROOT / f"{html.stem}-{time.strftime('%Y%m%d-%H%M%S')}"


def roundtrip(
    html: Path, source_path: Path, out_dir: Path, *, fps: float = FPS,
    warmup_ms: float = WARMUP_MS, capture: str = "virtual", draw_cursor: bool = False,
    keyframes: bool = False, keep_frames: bool = True, do_compare: bool = True,
    log=print,
) -> dict[str, Any]:  # fmt: skip
    """Run the whole round trip; returns paths, timings and (optionally) the compare result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text("utf-8"))
    timings: dict[str, float] = {}
    t = time.perf_counter()
    plan = build_plan(source, fps=fps, warmup_ms=warmup_ms, draw_cursor=draw_cursor,
                      capture=capture)  # fmt: skip
    for n in plan["notes"]:
        log(f"  note: {n}")
    meta = run_capture(plan, html, out_dir)
    timings["capture_s"] = round(time.perf_counter() - t, 1)
    if meta.get("exceptions"):
        log(f"  page exceptions: {meta['exceptions'][:3]}")
    t = time.perf_counter()
    video = out_dir / "replica.mp4"
    encode(out_dir / "frames", meta, video)
    timings["encode_s"] = round(time.perf_counter() - t, 1)
    if not keep_frames:
        for f in (out_dir / "frames").glob("f_*.png"):
            f.unlink()
    t = time.perf_counter()
    kf = out_dir / "keyframes" if keyframes else None
    replica = analyze_replica(video, plan["dpr"], keyframes_dir=kf)
    timings["analyze_s"] = round(time.perf_counter() - t, 1)
    ir_path = out_dir / "replica.ir.json"
    ir_path.write_text(json.dumps(replica, indent=2, ensure_ascii=False) + "\n", "utf-8")
    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "plan": str(out_dir / "plan.json"),
        "frames": str(out_dir / "frames"),
        "video": str(video),
        "replica_ir": str(ir_path),
        "frames_captured": len(meta["frames"]),
        "capture": meta.get("capture"),
        "events_sent": meta.get("events_sent"),
        "page_exceptions": meta.get("exceptions", []),
        "timings": timings,
    }
    if do_compare:
        import compare_roundtrip as cr  # noqa: PLC0415

        res = cr.compare(source, replica)
        cmp_path = out_dir / "compare.json"
        cmp_path.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n", "utf-8")
        result["compare"] = str(cmp_path)
        result["verdict"] = cr.PASS if res.ok else cr.FAIL
        result["failed"] = res.failed()
        result["table"] = cr.format_table(
            res, title=f"round trip: {source_path.name} vs {html.name} ({res.mode})"
        )
    (out_dir / "roundtrip.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                            "utf-8")  # fmt: skip
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("html", type=Path, help="the replica page (index.html)")
    ap.add_argument("source", type=Path, help="source IR JSON (from scripts/analyze.py --out)")
    ap.add_argument("--out", type=Path, help="output directory (default /tmp/mimic-roundtrip/...)")
    ap.add_argument("--capture", choices=["virtual", "screencast"], default="virtual")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--warmup-ms", type=float, default=WARMUP_MS)
    ap.add_argument("--draw-cursor", action="store_true", help="draw a pointer sprite into frames")
    ap.add_argument("--keyframes", action="store_true", help="keep the replica's keyframe PNGs")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="print the input plan and exit")
    ns = ap.parse_args(argv)
    if not ns.html.is_file() or not ns.source.is_file():
        ap.error("both the HTML file and the source IR must exist")
    if ns.plan_only:
        plan = build_plan(
            json.loads(ns.source.read_text("utf-8")),
            fps=ns.fps,
            warmup_ms=ns.warmup_ms,
            draw_cursor=ns.draw_cursor,
            capture=ns.capture,
        )
        print(json.dumps(plan, indent=1))
        return 0
    out = ns.out or default_out(ns.html)
    t = time.perf_counter()
    try:
        res = roundtrip(ns.html, ns.source, out, fps=ns.fps, warmup_ms=ns.warmup_ms,
                        capture=ns.capture, draw_cursor=ns.draw_cursor, keyframes=ns.keyframes,
                        do_compare=not ns.no_compare)  # fmt: skip
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if "error" in json.loads(Path(res["replica_ir"]).read_text("utf-8")):
        print(f"replica analysis returned an error: see {res['replica_ir']}", file=sys.stderr)
    if "table" in res:
        print(res["table"])
    print(f"total {time.perf_counter() - t:.1f}s {res['timings']}")
    for key in ("plan", "frames", "video", "replica_ir", "compare"):
        if key in res:
            print(f"  {key:10} {res[key]}")
    if ns.no_compare:
        return 0
    return 0 if res.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
