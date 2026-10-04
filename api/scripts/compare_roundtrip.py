"""Round-trip compare: source IR vs replica IR → pass/fail table (PLAN-continuous §14).

    uv run python scripts/compare_roundtrip.py SOURCE.json REPLICA.json [--json OUT.json]

``SOURCE`` is the IR of the original recording, ``REPLICA`` the IR of the replica page
re-recorded by ``scripts/roundtrip.py``. Both may be a ResultEnvelope (``{"spec": ...}``), a
bare MotionSpec, or an error body (then every row fails). Exit status 0 = PASS, 1 = FAIL,
2 = usage / unreadable input.

Criteria (§14). Continuous mode: phase-kind sequence identical (phases shorter than 2 frames
dropped, then consecutive phases of one kind merged); autoplay ±5 % and same direction; inertia τ
±20 %; inertia mode (decay to zero / decay towards autoplay / glide) identical; resume delay
±100 ms; resume ramp and pause slowdown: the time to reach 90 % of the speed change ±100 ms (an
eased ramp's end is not identifiable: easeOutQuart over 1080 ms ≈ material-decelerate over
815 ms); paused durations ±100 ms; snap presence identical; scroller box, card size, pitch, gap
±2 px when measured on both; card scaling (``card.scale``): the replica's scale at the source's
reference distance ±0.03, scaling on one side only fails, and then card size / pitch are not
compared (the centre card vs a uniform size); page / scroller / card / title colours ΔE76 ≤ 5
where measured on both. Transition mode: same segments; per source transition: present in the
replica, duration ±max(20 ms, 10 %), value within PLAN §11.4 (translate 0.5 px when |Δ| ≥ 4 px,
scale 0.01, opacity 0.1, colour ΔE 5 / text 10, shadow direction only), easing family identical
(``ease`` ≈ ``ease-in-out``, PLAN Appendix B; judged for transitions ≥ 150 ms whose source easing
confidence is ≥ 0.3, never for shadows).

A criterion the **source** could not measure is reported ``n/a`` (never ``PASS``); geometry and
colour rows are ``n/a`` unless both sides measured them. Pure functions on plain dicts; no
pipeline imports (the CLI validates inputs against the IR model before comparing); ΔE uses
OpenCV like the synthetic evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PASS, FAIL, NA, INFO = "PASS", "FAIL", "n/a", "info"

#: §14 tolerances (continuous).
SPEED_REL = 0.05
TAU_REL = 0.20
RESUME_MS = 100.0
PAUSE_MS = 100.0
GEOMETRY_PX = 2.0
#: Card scaling (P2c): the replica's scale at the source's reference distance (same tolerance as
#: the synthetic evaluator's C13 check).
CARD_SCALE_TOL = 0.03
COLOR_DE = 5.0
#: Phases shorter than this many frames may merge into their neighbours (§14).
SHORT_PHASE_FRAMES = 2

#: §14 transition tolerances: duration ±max(20 ms, 10 %); values per PLAN §11.4.
DURATION_ABS_MS = 20.0
DURATION_REL = 0.10
TRANSLATE_PX = 0.5
TRANSLATE_MIN_DELTA = 4.0
SCALE_TOL = 0.01
OPACITY_TOL = 0.1
PX_TOL = 0.5
COLOR_FILL_DE = 5.0
COLOR_TEXT_DE = 10.0
FAMILY_MIN_DURATION_MS = 150
FAMILY_EQUIV = {frozenset({"ease", "ease-in-out"})}
#: An easing whose source confidence is below this is "uncertain" (IR ``UNCERTAIN_BELOW``; the
#: generators leave it out of the main text), i.e. not a measured family: reported n/a.
EASING_UNCERTAIN_BELOW = 0.3
ELEMENT_IOU = 0.3


@dataclass
class Row:
    criterion: str
    source: str
    replica: str
    tolerance: str
    status: str
    detail: str = ""


@dataclass
class Result:
    mode: str
    rows: list[Row] = field(default_factory=list)

    @property
    def judged(self) -> list[Row]:
        return [r for r in self.rows if r.status in (PASS, FAIL)]

    @property
    def ok(self) -> bool:
        """PASS: at least one judged row and none failed."""
        return bool(self.judged) and all(r.status == PASS for r in self.judged)

    def failed(self) -> list[str]:
        return [r.criterion for r in self.rows if r.status == FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "verdict": PASS if self.ok else FAIL,
            "rows": [asdict(r) for r in self.rows],
        }


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def spec_of(data: dict[str, Any]) -> dict[str, Any] | None:
    """The MotionSpec dict of an envelope / bare spec; ``None`` for an error body."""
    if "error" in data and isinstance(data["error"], dict):
        return None
    return data["spec"] if "spec" in data else data


def _num(x: Any) -> float | None:
    """Value of a MeasuredNumber dict or a plain number; ``None`` when absent."""
    if x is None:
        return None
    if isinstance(x, dict):
        v = x.get("value")
        return None if v is None else float(v)
    return float(x)


def _g(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _fmt(v: float | None, unit: str = "", nd: int = 1) -> str:
    if v is None:
        return "—"
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"
    return f"{s}{unit}"


def delta_e76(a: str, b: str) -> float:
    """CIE76 ΔE of two ``#RRGGBB`` colours, computed exactly like the synthetic evaluator
    (``tests/synth/evaluate.py``: OpenCV float RGB → Lab)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def lab(h: str) -> np.ndarray:
        rgb = np.array([[[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]]], np.float32) / 255
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0].astype(np.float64)

    return float(np.linalg.norm(lab(a) - lab(b)))


def _frame_ms(spec: dict[str, Any]) -> float:
    src = spec.get("source") or {}
    fps = src.get("fps_effective") or src.get("fps_nominal")
    if fps:
        return 1000.0 / float(fps)
    return float(src.get("timing_resolution_ms") or 17)


def _box(b: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not b:
        return None
    return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------------------------
# Continuous
# --------------------------------------------------------------------------------------------


def merged_phases(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Phases with those shorter than :data:`SHORT_PHASE_FRAMES` frames dropped and consecutive
    phases of one kind merged. Each item: ``{kind, start_ms, end_ms, phases: [...]}``."""
    c = spec.get("continuous") or {}
    min_ms = SHORT_PHASE_FRAMES * _frame_ms(spec) - 1e-6
    out: list[dict[str, Any]] = []
    for ph in c.get("phases") or []:
        if ph["end_ms"] - ph["start_ms"] < min_ms:
            continue
        if out and out[-1]["kind"] == ph["kind"]:
            out[-1]["end_ms"] = ph["end_ms"]
            out[-1]["phases"].append(ph)
        else:
            out.append(
                {"kind": ph["kind"], "start_ms": ph["start_ms"], "end_ms": ph["end_ms"],
                 "phases": [ph]}
            )  # fmt: skip
    return out


def match_merged(
    a: list[dict[str, Any]], b: list[dict[str, Any]], kind: str
) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Pairs of ``kind`` phases (in order when both sequences agree, else greedy time-IoU) and
    the unmatched phases of ``a``."""
    aa = [p for p in a if p["kind"] == kind]
    bb = [p for p in b if p["kind"] == kind]
    if [p["kind"] for p in a] == [p["kind"] for p in b]:
        return list(zip(aa, bb, strict=True)), []
    cands = sorted(
        (
            (_time_iou(p["start_ms"], p["end_ms"], q["start_ms"], q["end_ms"]), i, j)
            for i, p in enumerate(aa)
            for j, q in enumerate(bb)
        ),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for iou, i, j in cands:
        if iou > 0 and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    pairs.sort()
    return [(aa[i], bb[j]) for i, j in pairs], [p for i, p in enumerate(aa) if i not in used_a]


def inertia_mode(phase: dict[str, Any]) -> str:
    """``decay-to-zero`` / ``decay-towards-autoplay`` / ``glide`` (tween) / ``unfitted``."""
    fit = phase.get("fit") or {}
    model = fit.get("model")
    if model == "tween":
        return "glide"
    if model == "exponential":
        return (
            "decay-to-zero"
            if abs(float(fit.get("v_inf_px_s") or 0.0)) < 1e-9
            else ("decay-towards-autoplay")
        )
    return "unfitted"


def _inertia_modes(merged: list[dict[str, Any]]) -> list[str]:
    return [inertia_mode(p) for m in merged if m["kind"] == "inertia" for p in m["phases"][:1]]


def _scroller_parts(spec: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Scroller, card and card-title elements of a continuous spec."""
    c = spec.get("continuous") or {}
    els = spec.get("elements") or []
    sid = c.get("element_id")
    scroller = next((e for e in els if e["id"] == sid), None)
    card = next(
        (e for e in els if e.get("role") == "card" and e.get("parent_id") == sid),
        None,
    ) or next((e for e in els if e.get("parent_id") == sid and not e.get("text_like")), None)
    title = None
    if card is not None:
        title = next(
            (e for e in els if e.get("parent_id") == card["id"] and e.get("text_like")), None
        )
    return {"scroller": scroller, "card": card, "title": title}


def _color_row(name: str, src: Any, rep: Any) -> Row:
    s, r = _g(src, "value"), _g(rep, "value")
    if s is None or r is None:
        why = "not measured on the source" if s is None else "not measured on the replica"
        return Row(name, s or "—", r or "—", f"ΔE ≤ {COLOR_DE:g}", NA, why)
    de = delta_e76(s, r)
    return Row(name, s, r, f"ΔE ≤ {COLOR_DE:g}", PASS if de <= COLOR_DE else FAIL, f"ΔE {de:.1f}")


def _abs_row(
    name: str,
    s: float | None,
    r: float | None,
    tol: float,
    unit: str,
    *,
    need_replica: bool = True,
    src_missing: str = "not measured on the source",
) -> Row:
    if s is None:
        return Row(name, "—", _fmt(r, unit), f"±{tol:g} {unit}".strip(), NA, src_missing)
    if r is None:
        if need_replica:
            return Row(name, _fmt(s, unit), "—", f"±{tol:g} {unit}".strip(), FAIL,
                       "not measured on the replica")  # fmt: skip
        return Row(name, _fmt(s, unit), "—", f"±{tol:g} {unit}".strip(), NA,
                   "not measured on the replica")  # fmt: skip
    d = r - s
    ok = abs(d) <= tol + 1e-9
    return Row(name, _fmt(s, unit), _fmt(r, unit), f"±{tol:g} {unit}".strip(), PASS if ok else FAIL,
               f"Δ {d:+.1f}")  # fmt: skip


#: Eased velocity ramps are compared by the time they take to reach this fraction of their
#: speed change: a strongly ease-out ramp's end is not identifiable (an easeOutQuart over
#: 1080 ms and a material-decelerate over 815 ms reach 90 % within 25 ms of each other).
RAMP_REACH = 0.9


def ramp_reach_ms(
    duration_ms: float, easing: dict[str, Any] | None, frac: float = RAMP_REACH
) -> float:
    """Time (ms) at which an eased ramp has covered ``frac`` of its velocity change."""
    bz = (easing or {}).get("cubic_bezier")
    if not bz:
        return duration_ms * frac
    x1, y1, x2, y2 = (float(v) for v in bz)
    lo, hi = 0.0, 1.0
    for _ in range(60):  # bisection on the curve parameter for y(t) = frac
        t = (lo + hi) / 2.0
        y = 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t**3
        lo, hi = (t, hi) if y < frac else (lo, t)
    t = (lo + hi) / 2.0
    return duration_ms * (3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t**3)


def _ramp_row(
    name: str, s_b: dict[str, Any], r_b: dict[str, Any] | None, key: str, tol: float
) -> Row:
    """Ramp durations compared by the time to reach :data:`RAMP_REACH` of the speed change."""
    s_d, r_d = _num(s_b.get(key)), _num(_g(r_b, key))
    if s_d is None or r_d is None or r_b is None:
        return _abs_row(name, s_d, r_d, tol, "ms")
    s_t = ramp_reach_ms(s_d, s_b.get("easing"))
    r_t = ramp_reach_ms(r_d, r_b.get("easing"))
    d = r_t - s_t
    pct = f"{RAMP_REACH:.0%}"
    return Row(name, _fmt(s_d, " ms"), _fmt(r_d, " ms"), f"±{tol:g} ms to {pct} speed",
               PASS if abs(d) <= tol + 1e-9 else FAIL,
               f"{pct} reached at {s_t:.0f} vs {r_t:.0f} ms (Δ {d:+.0f})")  # fmt: skip


def scale_at(cs: dict[str, Any], d: float) -> float:
    """Card scale at distance ``d`` from the scroller centre (IR ``CardScale``, quadratic)."""
    ref = float(cs["reference_distance_px"])
    return 1.0 + (float(cs["scale_at_reference"]) - 1.0) * (d / ref) ** 2


def _card_scale_row(
    s_cs: dict[str, Any] | None, r_cs: dict[str, Any] | None, *, rigid_source: bool, judged: bool
) -> Row:
    """``card.scale``: the replica's scale at the source's reference distance ±CARD_SCALE_TOL;
    scaling on one side only fails (a rigid source counts only when it measured a card)."""
    tol = f"±{CARD_SCALE_TOL:g} at the source's reference"

    def show(cs: dict[str, Any] | None) -> str:
        if cs is None:
            return "rigid"
        return (
            f"x{float(cs['scale_at_reference']):.3f} @ {float(cs['reference_distance_px']):.0f} px"
        )

    if s_cs is None:
        if r_cs is not None and rigid_source and judged:
            return Row("card.scale", "rigid", show(r_cs), tol, FAIL,
                       "cards scale on the replica only")  # fmt: skip
        why = "rigid cards on both" if r_cs is None else "no card measured on the source"
        return Row("card.scale", "rigid" if rigid_source else "—", show(r_cs), tol, NA, why)
    if r_cs is None:
        return Row("card.scale", show(s_cs), "rigid" if judged else "—", tol, FAIL,
                   "no card scaling measured on the replica")  # fmt: skip
    d = float(s_cs["reference_distance_px"])
    want, got = float(s_cs["scale_at_reference"]), scale_at(r_cs, d)
    ok = abs(got - want) <= CARD_SCALE_TOL + 1e-9
    return Row("card.scale", show(s_cs), show(r_cs), tol, PASS if ok else FAIL,
               f"replica x{got:.3f} at {d:.0f} px (Δ {got - want:+.3f})")  # fmt: skip


def compare_continuous(src: dict[str, Any], rep: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    sc, rc = src["continuous"], rep.get("continuous") if rep.get("mode") == "continuous" else None
    rc = rc or {}
    rows.append(Row("mode", "continuous", rep.get("mode", "—"), "identical",
                    PASS if rep.get("mode") == "continuous" else FAIL))  # fmt: skip
    rows.append(Row("axis", sc["axis"], rc.get("axis", "—"), "identical",
                    PASS if rc.get("axis") == sc["axis"] else FAIL))  # fmt: skip

    # phase sequence
    sm, rm = merged_phases(src), merged_phases(rep) if rc else []
    sk, rk = [m["kind"] for m in sm], [m["kind"] for m in rm]
    rows.append(Row("phase sequence", " ".join(sk), " ".join(rk) or "—",
                    f"identical (phases < {SHORT_PHASE_FRAMES} frames merged)",
                    PASS if sk == rk else FAIL))  # fmt: skip

    # autoplay
    sa, ra = sc.get("autoplay"), rc.get("autoplay")
    if sa is None:
        rows.append(Row("autoplay.direction", "none", (ra or {}).get("direction", "none"),
                        "identical", NA, "no autoplay on the source"))  # fmt: skip
        rows.append(Row("autoplay.speed", "—", _fmt(_num(_g(ra, "speed_px_s")), " px/s"),
                        f"±{SPEED_REL:.0%}", NA, "no autoplay on the source"))  # fmt: skip
    else:
        s_speed = _num(sa["speed_px_s"])
        r_dir = (ra or {}).get("direction", "none")
        rows.append(Row("autoplay.direction", sa["direction"], r_dir, "identical",
                        PASS if r_dir == sa["direction"] else FAIL))  # fmt: skip
        r_speed = _num(_g(ra, "speed_px_s"))
        if r_speed is None:
            rows.append(Row("autoplay.speed", _fmt(s_speed, " px/s"), "—", f"±{SPEED_REL:.0%}",
                            FAIL, "no autoplay on the replica"))  # fmt: skip
        else:
            rel = abs(r_speed - s_speed) / s_speed if s_speed else abs(r_speed)
            rows.append(Row("autoplay.speed", _fmt(s_speed, " px/s"), _fmt(r_speed, " px/s"),
                            f"±{SPEED_REL:.0%}", PASS if rel <= SPEED_REL + 1e-9 else FAIL,
                            f"{rel:+.1%}" if r_speed >= s_speed else f"-{rel:.1%}"))  # fmt: skip

    # inertia
    sb, rb = sc.get("behavior") or {}, rc.get("behavior") or {}
    si, ri = sb.get("inertia"), rb.get("inertia")
    s_tau, r_tau = _num(_g(si, "tau_ms")), _num(_g(ri, "tau_ms"))
    if s_tau is None:
        why = "no inertia on the source" if si is None else f"source inertia is a {si['model']}"
        rows.append(Row("inertia.tau", "—", _fmt(r_tau, " ms"), f"±{TAU_REL:.0%}", NA, why))
    elif r_tau is None:
        rows.append(Row("inertia.tau", _fmt(s_tau, " ms"), "—", f"±{TAU_REL:.0%}", FAIL,
                        "no exponential inertia on the replica"))  # fmt: skip
    else:
        rel = (r_tau - s_tau) / s_tau
        rows.append(Row("inertia.tau", _fmt(s_tau, " ms"), _fmt(r_tau, " ms"), f"±{TAU_REL:.0%}",
                        PASS if abs(rel) <= TAU_REL + 1e-9 else FAIL, f"{rel:+.0%}"))  # fmt: skip
    s_modes, r_modes = _inertia_modes(sm), _inertia_modes(rm)
    if not s_modes:
        rows.append(Row("inertia.mode", "—", " ".join(r_modes) or "—", "identical", NA,
                        "no inertia phase on the source"))  # fmt: skip
    else:
        rows.append(Row("inertia.mode", " ".join(s_modes), " ".join(r_modes) or "—", "identical",
                        PASS if s_modes == r_modes else FAIL))  # fmt: skip

    # resume
    sr, rr = sb.get("resume"), rb.get("resume")
    for name, key in (("resume.delay", "delay_after_rest_ms"), ("resume.ramp", "ramp_ms")):
        if sr is None:
            rows.append(Row(name, "—", _fmt(_num(_g(rr, key)), " ms"), f"±{RESUME_MS:g} ms", NA,
                            "no resume on the source"))  # fmt: skip
        elif key == "ramp_ms":
            rows.append(_ramp_row(name, sr, rr, key, RESUME_MS))
        else:
            rows.append(_abs_row(name, _num(sr[key]), _num(_g(rr, key)), RESUME_MS, "ms"))

    # pause
    sp, rp = sb.get("pause"), rb.get("pause")
    s_dec = _num(_g(sp, "decel_ms"))
    if s_dec is None:
        why = "no pause on the source" if sp is None else "source pause stops at once (no decel)"
        rows.append(Row("pause.decel", "—", _fmt(_num(_g(rp, "decel_ms")), " ms"),
                        f"±{PAUSE_MS:g} ms", NA, why))  # fmt: skip
    else:
        rows.append(_ramp_row("pause.decel", sp, rp, "decel_ms", PAUSE_MS))
    pairs, missing = match_merged(sm, rm, "paused")
    if not pairs and not missing:
        rows.append(Row("paused.durations", "—", "—", f"±{PAUSE_MS:g} ms", NA,
                        "no paused phase on the source"))  # fmt: skip
    else:
        diffs = [(q["end_ms"] - q["start_ms"]) - (p["end_ms"] - p["start_ms"]) for p, q in pairs]
        worst = max(diffs, key=abs) if diffs else 0
        ok = not missing and all(abs(d) <= PAUSE_MS for d in diffs)
        s_d = ", ".join(
            str(p["end_ms"] - p["start_ms"]) for p, _ in pairs + [(m, m) for m in missing]
        )  # noqa: E501
        r_d = ", ".join(str(q["end_ms"] - q["start_ms"]) for _, q in pairs) or "—"
        det = f"worst Δ {worst:+d} ms" + (f"; {len(missing)} unmatched" if missing else "")
        rows.append(Row("paused.durations", s_d + " ms", r_d + (" ms" if pairs else ""),
                        f"±{PAUSE_MS:g} ms", PASS if ok else FAIL, det))  # fmt: skip

    # snap
    ss, rs = sb.get("snap"), rb.get("snap")
    s_kind, r_kind = (ss or {}).get("kind", "none"), (rs or {}).get("kind", "none")
    rows.append(Row("snap.presence", s_kind, r_kind if rc else "—", "identical presence",
                    PASS if (ss is None) == (rs is None) and rc else FAIL))  # fmt: skip

    # geometry
    s_reg, r_reg = _box(sc.get("region")), _box(rc.get("region"))
    if s_reg is None or r_reg is None:
        rows.append(Row("scroller.box", "—", "—", f"±{GEOMETRY_PX:g} px", NA, "not measured"))
    else:
        d = max(abs(a - b) for a, b in zip(s_reg, r_reg, strict=True))
        rows.append(
            Row(
                "scroller.box",
                "{:g},{:g} {:g}x{:g}".format(*s_reg),
                "{:g},{:g} {:g}x{:g}".format(*r_reg),
                f"±{GEOMETRY_PX:g} px",
                PASS if d <= GEOMETRY_PX + 1e-9 else FAIL,
                f"max Δ {d:.1f} px",
            )
        )
    sp_parts, rp_parts = _scroller_parts(src), _scroller_parts(rep) if rc else {}
    s_card, r_card = sp_parts["card"], rp_parts.get("card")
    # with cards that scale with their position the card box and the pitch are the centre
    # card's; a rigid measurement is a different quantity (mean spacing): only like-for-like
    s_cs, r_cs = sc.get("card_scale"), rc.get("card_scale")
    unlike = rc and (s_cs is None) != (r_cs is None) and s_card is not None and r_card is not None
    rows.append(_card_scale_row(s_cs, r_cs, rigid_source=s_card is not None, judged=bool(rc)))
    if unlike:
        side = "the source" if s_cs is not None else "the replica"
        why = f"cards scale only on {side}: centre size vs uniform size (see card.scale)"
        for name, sv, rv, unit in (
            ("card.size", "{w:g}x{h:g}".format(**s_card["bbox_initial"]),
             "{w:g}x{h:g}".format(**r_card["bbox_initial"]), ""),
            ("pitch", _fmt(_num(sc.get("pitch_px")), " px"), _fmt(_num(rc.get("pitch_px")), " px"),
             ""),
        ):  # fmt: skip
            rows.append(Row(name, sv, rv + unit, f"±{GEOMETRY_PX:g} px", NA, why))
    elif s_card is None or r_card is None:
        rows.append(
            Row(
                "card.size",
                "—" if s_card is None else "{w:g}x{h:g}".format(**s_card["bbox_initial"]),
                "—" if r_card is None else "{w:g}x{h:g}".format(**r_card["bbox_initial"]),
                f"±{GEOMETRY_PX:g} px",
                NA,
                "card not measured on " + ("the source" if s_card is None else "the replica"),
            )
        )
    else:
        sb_, rb_ = s_card["bbox_initial"], r_card["bbox_initial"]
        d = max(abs(sb_["w"] - rb_["w"]), abs(sb_["h"] - rb_["h"]))
        rows.append(Row("card.size", f"{sb_['w']:g}x{sb_['h']:g}", f"{rb_['w']:g}x{rb_['h']:g}",
                        f"±{GEOMETRY_PX:g} px", PASS if d <= GEOMETRY_PX + 1e-9 else FAIL,
                        f"max Δ {d:.1f} px"))  # fmt: skip
    for name, key in (("pitch", "pitch_px"), ("gap", "gap_px")):
        if name == "pitch" and unlike:
            continue
        s_v, r_v = _num(sc.get(key)), _num(rc.get(key))
        if s_v is None or r_v is None:
            why = "not measured on " + ("the source" if s_v is None else "the replica")
            rows.append(Row(name, _fmt(s_v, " px"), _fmt(r_v, " px"), f"±{GEOMETRY_PX:g} px", NA,
                            why))  # fmt: skip
        else:
            rows.append(_abs_row(name, s_v, r_v, GEOMETRY_PX, "px"))

    # colours
    rows.append(_color_row("colour.page_background", _g(src, "scene", "page_background"),
                           _g(rep, "scene", "page_background")))  # fmt: skip
    for label, part, key in (
        ("colour.scroller_background", "scroller", "background_color"),
        ("colour.card_background", "card", "background_color"),
        ("colour.card_text", "title", "text_color"),
    ):
        rows.append(_color_row(label, _g(sp_parts[part], "static", key),
                               _g(rp_parts.get(part), "static", key)))  # fmt: skip

    # informational
    sd, rd = sb.get("drag"), rb.get("drag")
    rows.append(Row("info.drag_count", str((sd or {}).get("count", 0)),
                    str((rd or {}).get("count", 0)), "—", INFO))  # fmt: skip
    if si is not None or ri is not None:

        def rel_range(b: Any) -> str:
            if b is None:
                return "—"
            return f"{b['release_speed_min_px_s']:.0f}–{b['release_speed_max_px_s']:.0f} px/s"

        rows.append(Row("info.release_speeds", rel_range(si), rel_range(ri), "—", INFO))
    return rows


# --------------------------------------------------------------------------------------------
# Transition
# --------------------------------------------------------------------------------------------


def match_elements(src: dict[str, Any], rep: dict[str, Any]) -> dict[str, str]:
    """source element id → replica element id (greedy by box IoU ≥ 0.3, one-to-one)."""
    pairs = []
    for se in src.get("elements") or []:
        for re_ in rep.get("elements") or []:
            iou = max(
                _iou(_box(se["bbox_initial"]), _box(re_["bbox_initial"])),  # type: ignore[arg-type]
                _iou(_box(se["bbox_active"]), _box(re_["bbox_active"])),  # type: ignore[arg-type]
            )
            if iou >= ELEMENT_IOU:
                pairs.append((iou, se["id"], re_["id"]))
    out: dict[str, str] = {}
    used: set[str] = set()
    for _, sid, rid in sorted(pairs, reverse=True):
        if sid not in out and rid not in used:
            out[sid] = rid
            used.add(rid)
    return out


def _shadow_strength(v: dict[str, Any]) -> float:
    s = v.get("shadow")
    if not s:
        return 0.0
    a = float(s["rgba"][3])
    return float(s["blur"]) * a + abs(float(s["y"])) * a


def _value_row(name: str, prop: str, st: dict, rt: dict, text_like: bool) -> Row:
    f0, t0, f1, t1 = st["from"], st["to"], rt["from"], rt["to"]
    kind = f0["kind"]

    def show(v: dict) -> str:
        if v["kind"] in ("px", "ratio"):
            return f"{v['number']:g}"
        if v["kind"] == "color":
            return v["color"]
        if v["kind"] == "shadow":
            s = v.get("shadow")
            return "none" if not s else f"{s['x']:g} {s['y']:g} {s['blur']:g} a{s['rgba'][3]:g}"
        return str(v.get("text"))

    sv, rv = f"{show(f0)}→{show(t0)}", f"{show(f1)}→{show(t1)}"
    if kind == "px":
        err = max(abs(f1["number"] - f0["number"]), abs(t1["number"] - t0["number"]))
        delta = abs(t0["number"] - f0["number"])
        if prop in ("translateX", "translateY"):
            if delta < TRANSLATE_MIN_DELTA:
                return Row(name, sv, rv, f"±{TRANSLATE_PX:g} px", NA,
                           f"|Δ| {delta:g} px < {TRANSLATE_MIN_DELTA:g} px (§11.4)")  # fmt: skip
            tol = TRANSLATE_PX
        else:
            tol = PX_TOL
        return Row(name, sv, rv, f"±{tol:g} px", PASS if err <= tol + 1e-9 else FAIL,
                   f"max error {err:.2f} px")  # fmt: skip
    if kind == "ratio":
        tol = OPACITY_TOL if prop == "opacity" else SCALE_TOL
        err = max(abs(f1["number"] - f0["number"]), abs(t1["number"] - t0["number"]))
        return Row(name, sv, rv, f"±{tol:g}", PASS if err <= tol + 1e-9 else FAIL,
                   f"max error {err:.3f}")  # fmt: skip
    if kind == "color":
        tol = COLOR_TEXT_DE if text_like else COLOR_FILL_DE
        de = max(delta_e76(f0["color"], f1["color"]), delta_e76(t0["color"], t1["color"]))
        return Row(name, sv, rv, f"ΔE ≤ {tol:g}", PASS if de <= tol else FAIL, f"ΔE {de:.1f}")
    if kind == "shadow":
        want = math.copysign(1, _shadow_strength(t0) - _shadow_strength(f0))
        got = math.copysign(1, _shadow_strength(t1) - _shadow_strength(f1))
        word = {1.0: "increases", -1.0: "decreases"}
        return Row(name, word[want], word[got], "same direction (§11.4)",
                   PASS if want == got else FAIL)  # fmt: skip
    return Row(name, sv, rv, "—", NA, f"{kind} values are not a §11.4 target")


def compare_transition(src: dict[str, Any], rep: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    rows.append(Row("mode", "transition", rep.get("mode", "—"), "identical",
                    PASS if rep.get("mode", "transition") == "transition" else FAIL))  # fmt: skip
    s_segs = sorted(s["id"] for s in src.get("segments") or [])
    r_segs = sorted(s["id"] for s in rep.get("segments") or [])
    rows.append(Row("segments", " ".join(s_segs), " ".join(r_segs) or "—", "identical",
                    PASS if s_segs == r_segs else FAIL))  # fmt: skip
    eid = match_elements(src, rep)
    s_els = {e["id"]: e for e in src.get("elements") or []}
    used: set[str] = set()
    s_onsets = {s["id"]: s["onset_ms"] for s in src.get("segments") or []}
    r_onsets = {s["id"]: s["onset_ms"] for s in rep.get("segments") or []}
    for st in src.get("transitions") or []:
        el = s_els.get(st["element_id"], {})
        tag = f"{st['segment_id']} {st['element_id']}({el.get('role', '?')}).{st['property']}"
        rid = eid.get(st["element_id"])
        rt = next(
            (t for t in rep.get("transitions") or []
             if rid is not None and t["element_id"] == rid and t["property"] == st["property"]
             and t["segment_id"] == st["segment_id"]),
            None,
        )  # fmt: skip
        if rt is None:
            why = "no matching replica element" if rid is None else "transition missing"
            rows.append(Row(f"{tag} present", "yes", "no", "present", FAIL, why))
            continue
        used.add(rt["id"])
        shadow = st["property"] == "box-shadow"
        s_dur, r_dur = float(st["duration_ms"]), float(rt["duration_ms"])
        tol = max(DURATION_ABS_MS, DURATION_REL * s_dur)
        if shadow:
            rows.append(
                Row(
                    f"{tag} duration",
                    _fmt(s_dur, " ms"),
                    _fmt(r_dur, " ms"),
                    f"±{tol:.0f} ms",
                    NA,
                    "shadow judged on direction only (§11.4)",
                )
            )
        else:
            d = r_dur - s_dur
            rows.append(Row(f"{tag} duration", _fmt(s_dur, " ms"), _fmt(r_dur, " ms"),
                            f"±{tol:.0f} ms", PASS if abs(d) <= tol + 1e-9 else FAIL,
                            f"Δ {d:+.0f} ms"))  # fmt: skip
        rows.append(_value_row(f"{tag} value", st["property"], st, rt, bool(el.get("text_like"))))
        s_fam, r_fam = st["easing"]["family"], rt["easing"]["family"]
        same = s_fam == r_fam or frozenset({s_fam, r_fam}) in FAMILY_EQUIV
        s_econf = float((st.get("confidence") or {}).get("easing", 1.0))
        if shadow or s_dur < FAMILY_MIN_DURATION_MS or s_econf < EASING_UNCERTAIN_BELOW:
            if shadow:
                why = "shadow judged on direction only (§11.4)"
            elif s_dur < FAMILY_MIN_DURATION_MS:
                why = f"source duration < {FAMILY_MIN_DURATION_MS} ms (§11.4 eligibility)"
            else:
                why = (
                    f"source easing uncertain (confidence {s_econf:.2f} "
                    f"< {EASING_UNCERTAIN_BELOW:g})"
                )
            rows.append(Row(f"{tag} easing", s_fam, r_fam, "same family", NA, why))
        else:
            rows.append(Row(f"{tag} easing", s_fam, r_fam, "same family",
                            PASS if same else FAIL, st["easing"]["nearest_named"] + " / " +
                            rt["easing"]["nearest_named"]))  # fmt: skip
        s_delay = st["start_ms"] - s_onsets.get(st["segment_id"], st["start_ms"])
        r_delay = rt["start_ms"] - r_onsets.get(rt["segment_id"], rt["start_ms"])
        rows.append(Row(f"{tag} delay", _fmt(s_delay, " ms", 0), _fmt(r_delay, " ms", 0), "—",
                        INFO, "relative to the segment onset"))  # fmt: skip
    for rt in rep.get("transitions") or []:
        if rt["id"] not in used:
            rows.append(
                Row(
                    f"extra {rt['segment_id']} {rt['element_id']}.{rt['property']}",
                    "—",
                    f"{rt['duration_ms']} ms",
                    "—",
                    INFO,
                    "replica-only transition",
                )
            )
    return rows


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------


def compare(source: dict[str, Any], replica: dict[str, Any]) -> Result:
    """Compare two IRs (envelopes, bare specs or error bodies)."""
    src, rep = spec_of(source), spec_of(replica)
    if src is None:
        code = source["error"].get("code")
        return Result("—", [Row("source", f"error {code}", "—", "an analysable recording", NA,
                                "the source IR is an error; nothing to compare")])  # fmt: skip
    mode = src.get("mode", "transition")
    if rep is None:
        code = replica["error"].get("code")
        return Result(mode, [Row("replica", "result", f"error {code}", "a result", FAIL,
                                 replica["error"].get("message", ""))])  # fmt: skip
    rows = compare_continuous(src, rep) if mode == "continuous" else compare_transition(src, rep)
    return Result(mode, rows)


def format_table(res: Result, *, title: str = "") -> str:
    hdr = ("criterion", "source", "replica", "tolerance", "result", "detail")
    data = [(r.criterion, r.source, r.replica, r.tolerance, r.status, r.detail) for r in res.rows]
    widths = [max(len(h), *(len(d[i]) for d in data)) if data else len(h)
              for i, h in enumerate(hdr)]  # fmt: skip
    widths = [min(w, 48) for w in widths]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c[:w].ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    out = []
    if title:
        out.append(title)
    out += [line(hdr), line(tuple("-" * w for w in widths))]
    out += [line(d) for d in data]
    judged = res.judged
    n_pass = sum(r.status == PASS for r in judged)
    n_na = sum(r.status == NA for r in res.rows)
    out.append(
        f"verdict: {PASS if res.ok else FAIL}  ({n_pass}/{len(judged)} judged rows pass, "
        f"{n_na} n/a)"
    )
    return "\n".join(out)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(data: dict[str, Any]) -> None:
    """Raise if a (non-error) document is not a valid MotionSpec (needs the api package)."""
    spec = spec_of(data)
    if spec is None:
        return
    api_dir = Path(__file__).resolve().parents[1]
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))
    from app.models.ir import MotionSpec  # noqa: PLC0415

    MotionSpec.model_validate(spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("replica", type=Path)
    ap.add_argument("--json", type=Path, help="also write the rows as JSON here")
    ap.add_argument("--no-validate", action="store_true", help="skip IR model validation")
    ns = ap.parse_args(argv)
    try:
        source, replica = load_json(ns.source), load_json(ns.replica)
        if not ns.no_validate:
            validate(source)
            validate(replica)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    res = compare(source, replica)
    print(
        format_table(res, title=f"round trip: {ns.source.name} vs {ns.replica.name} ({res.mode})")
    )
    if ns.json:
        ns.json.parent.mkdir(parents=True, exist_ok=True)
        ns.json.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n", "utf-8")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
