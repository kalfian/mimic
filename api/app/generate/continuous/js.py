"""JS tab, continuous mode only (PLAN-continuous §6.4): a deterministic suggested driver.

One ``requestAnimationFrame`` loop owns the track position; a small state machine
(``autoplay | decel | paused | drag | inertia | snap | waiting | resume``) switches between
autoplay, the hover/press slowdown, pointer dragging (Pointer Events + capture), momentum
(``v *= exp(-dt / τ)``, frame-rate independent; or ``v = vAuto + (v - vAuto) · exp(-dt / τ)``
when the IR's momentum decays towards the autoplay velocity, which then blends into autoplay
without a rest / resume), grid snapping and the resume ramp. The wrap period is the distance
from the first card of copy A to the first card of copy B (the gap between the copies
included). Behaviours
that were not observed drop their constants and code paths. Cards that scale with their position
(``continuous.card_scale``, P2c) get ``scaleCards`` in the same loop, and the on-screen speeds /
pointer distances are divided by ``SPEED_SCALE`` (the IR's mean magnification) to move the
unscaled track.

Number provenance: the constants block holds IR values only (display-rounded); the template
itself uses just ``0``, ``1``, ``2`` and ``1000`` (structural literals / ms <-> s), allow-listed
by the test next to ``CSS_IDENTITY``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.generate import phrasing as ph
from app.generate.continuous import phrasing as cp
from app.models.ir import Easing, MeasuredNumber, MotionSpec, band_for

HEADER = "// Suggested driver — estimated from a screen recording, not the original code."

#: Literals the template may print besides IR values (structural, ms <-> s).
TEMPLATE_LITERALS = frozenset({0.0, 1.0, 2.0, 1000.0})


@dataclass(frozen=True)
class _Flags:
    autoplay: bool
    pause: bool
    pause_hover: bool  # hover or unknown trigger (implemented as hover)
    pause_press: bool
    pause_ramp: bool  # slowdown duration measured
    drag: bool
    ratio: bool  # content does not follow the pointer 1:1 and the ratio is measured
    inertia_exp: bool
    inertia_tween: bool
    inertia_stop: bool  # measured momentum stop speed (else: rest when < 1 px is left)
    inertia_blend: bool  # momentum decays towards the autoplay velocity (blends into autoplay)
    leave_delay: bool  # hover pause: resume delay counted from the pointer leaving
    snap: bool  # grid snap
    snap_ramp: bool  # snap duration measured
    resume: bool
    resume_ease: bool  # resume easing measured (else linear)
    card_scale: bool  # cards scale with their distance from the scroller centre (P2c)
    rest_release: bool  # drags that end at rest: a release without velocity rests, then resumes

    @property
    def inertia(self) -> bool:
        return self.inertia_exp or self.inertia_tween

    @property
    def eased(self) -> bool:
        return self.pause_ramp or self.resume or self.snap_ramp or self.inertia_tween

    @property
    def presses(self) -> bool:
        return self.drag or self.pause_press


def _flags(spec: MotionSpec) -> _Flags:
    b = cp.cont(spec).behavior
    pz, d, i, s = b.pause, b.drag, b.inertia, b.snap
    snap = s is not None and s.kind == "grid" and s.step_px is not None
    return _Flags(
        autoplay=cp.cont(spec).autoplay is not None,
        pause=pz is not None,
        pause_hover=cp.pauses_on_hover(spec),
        pause_press=cp.pause_trigger(spec) == "press",
        pause_ramp=pz is not None and pz.decel_ms is not None and pz.easing is not None,
        drag=d is not None,
        ratio=d is not None and d.follows_pointer is False and d.pointer_ratio is not None,
        inertia_exp=i is not None and i.model == "exponential" and i.tau_ms is not None,
        inertia_stop=i is not None
        and i.model == "exponential"
        and i.tau_ms is not None
        and i.stop_px_s is not None,
        inertia_blend=i is not None
        and i.model == "exponential"
        and i.tau_ms is not None
        and (cp.cont(spec).autoplay is not None or b.resume is not None)
        and cp.inertia_blends_into_autoplay(spec),
        leave_delay=cp.pauses_on_hover(spec)
        and b.resume is not None
        and b.resume.delay_after_leave_ms is not None,
        inertia_tween=i is not None
        and i.model == "tween"
        and i.duration_ms is not None
        and i.easing is not None,
        snap=snap,
        snap_ramp=snap and s.duration_ms is not None and s.easing is not None,  # type: ignore[union-attr]
        resume=b.resume is not None,
        resume_ease=b.resume is not None and b.resume.easing is not None,
        card_scale=cp.card_scale(spec) is not None,
        rest_release=d is not None
        and i is not None
        and i.model == "exponential"
        and i.tau_ms is not None
        and b.resume is not None
        and bool(cp.drag_stops(spec)),
    )


# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------


def _conf_comment(score: float) -> str:
    if cp.is_uncertain_score(score):
        return f"uncertain ({ph.pct(score)} confidence)"
    return f"{band_for(score)} confidence"


def _const(name: str, value: str, comment: str) -> str:
    return f"const {name} = {value}; // {comment}"


def _signed(v: float) -> str:
    n = cp.speed_num(v)
    return f"-{n}" if v < 0 and n != "0" else n


def _ease(e: Easing) -> str:
    return "[" + ", ".join(ph.bezier_num(x) for x in e.cubic_bezier) + "]"


def _ease_comment(e: Easing, conf: float) -> str:
    return ph.easing_text(e, conf)


def _ms(m: MeasuredNumber) -> str:
    return cp.ms_num(m.value)


def _constants(spec: MotionSpec, f: _Flags) -> list[str]:
    c = cp.cont(spec)
    b = c.behavior
    out: list[str] = []
    if c.autoplay is not None:
        sp = c.autoplay.speed_px_s
        out.append(
            _const(
                "AUTOPLAY_PX_S",
                _signed(c.autoplay.velocity_px_s),
                f"linear autoplay {cp.DIRECTION_PHRASE[c.autoplay.direction]}, "
                f"{_conf_comment(sp.confidence.value)}",
            )
        )
    if b.pause is not None and f.pause_ramp:
        m = b.pause.decel_ms
        assert m is not None and b.pause.easing is not None
        trigger = {
            "hover": "on hover",
            "press": "while pressed",
            "unknown": "on hover (trigger not visible)",
        }[cp.pause_trigger(spec) or b.pause.on]
        out += [
            _const(
                "PAUSE_DECEL_MS",
                _ms(m),
                f"slowdown to a stop {trigger}, {_conf_comment(m.confidence.value)}",
            ),
            _const(
                "PAUSE_EASE",
                _ease(b.pause.easing),
                _ease_comment(b.pause.easing, m.confidence.value),
            ),
        ]
    if f.ratio:
        r = b.drag.pointer_ratio  # type: ignore[union-attr]
        assert r is not None
        out.append(
            _const(
                "POINTER_RATIO",
                cp.ratio_num(r.value),
                f"content / pointer movement, {_conf_comment(r.confidence.value)}",
            )
        )
    i = b.inertia
    if f.inertia_exp:
        assert i is not None and i.tau_ms is not None
        towards = " (towards the autoplay velocity)" if f.inertia_blend else ""
        out.append(
            _const(
                "INERTIA_TAU_MS",
                _ms(i.tau_ms),
                f"momentum decay time constant{towards}, "
                f"{_conf_comment(i.tau_ms.confidence.value)}",
            )
        )
    if f.inertia_stop and not f.inertia_blend:
        assert i is not None and i.stop_px_s is not None
        out.append(
            _const(
                "INERTIA_STOP_PX_S",
                cp.speed_num(i.stop_px_s.value),
                f"momentum counts as at rest below this speed, "
                f"{_conf_comment(i.stop_px_s.confidence.value)}",
            )
        )
    if f.inertia_tween:
        assert i is not None and i.duration_ms is not None and i.easing is not None
        out += [
            _const(
                "INERTIA_MS",
                _ms(i.duration_ms),
                f"glide after release, {_conf_comment(i.duration_ms.confidence.value)}",
            ),
            _const(
                "INERTIA_EASE",
                _ease(i.easing),
                _ease_comment(i.easing, i.duration_ms.confidence.value),
            ),
        ]
    s = b.snap
    if f.snap:
        assert s is not None and s.step_px is not None
        out.append(
            _const(
                "SNAP_STEP_PX",
                ph.num_px(s.step_px.value),
                f"card pitch, {_conf_comment(s.step_px.confidence.value)}",
            )
        )
        if f.snap_ramp:
            assert s.duration_ms is not None and s.easing is not None
            out += [
                _const(
                    "SNAP_MS",
                    _ms(s.duration_ms),
                    f"settle on the nearest card, {_conf_comment(s.duration_ms.confidence.value)}",
                ),
                _const(
                    "SNAP_EASE",
                    _ease(s.easing),
                    _ease_comment(s.easing, s.duration_ms.confidence.value),
                ),
            ]
    r = b.resume
    if r is not None:
        target = cp.resume_velocity(spec)
        assert target is not None
        same = "same direction" if r.direction_preserved else "opposite direction"
        if f.leave_delay:
            assert r.delay_after_leave_ms is not None
            out.append(
                _const(
                    "RESUME_AFTER_LEAVE_MS",
                    _ms(r.delay_after_leave_ms),
                    "after the pointer leaves, "
                    f"{_conf_comment(r.delay_after_leave_ms.confidence.value)}",
                )
            )
        out += [
            _const(
                "RESUME_DELAY_MS",
                _ms(r.delay_after_rest_ms),
                "after the motion comes to rest, "
                f"{_conf_comment(r.delay_after_rest_ms.confidence.value)}",
            ),
            _const(
                "RESUME_RAMP_MS",
                _ms(r.ramp_ms),
                f"ramp back to autoplay speed, {_conf_comment(r.ramp_ms.confidence.value)}",
            ),
        ]
        if r.easing is not None:
            out.append(
                _const(
                    "RESUME_EASE",
                    _ease(r.easing),
                    _ease_comment(r.easing, r.ramp_ms.confidence.value),
                )
            )
        out.append(
            _const("RESUME_PX_S", _signed(target), f"autoplay velocity after resuming, {same}")
        )
    cs = cp.card_scale(spec)
    if cs is not None:
        out += [
            _const(
                "SCALE_REF_PX",
                ph.num_px(cs.reference_distance_px),
                "farthest measured card centre from the scroller centre",
            ),
            _const(
                "SCALE_AT_REF",
                cp.scale_factor(cs.scale_at_reference),
                f"card scale there (1 at the centre), {_conf_comment(cs.confidence.value)}",
            ),
            _const(
                "SPEED_SCALE",
                cp.scale_factor(cs.mean_scale),
                "mean on-screen magnification: the speeds above are on screen, the unscaled "
                "track moves this many times slower",
            ),
        ]
    return out


# --------------------------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------------------------


class _Lines:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, *lines: str, when: bool = True) -> None:
        if when:
            self.lines.extend(lines)


def _holds(f: _Flags) -> str | None:
    """JS expression that is true while the user still holds the scroller."""
    parts = []
    if f.presses:
        parts.append("pressed")
    if f.pause_hover:
        parts.append("hovered")
    return " || ".join(parts) or None


def _pause_start(f: _Flags, indent: str) -> list[str]:
    if f.pause_ramp:
        return [f"{indent}start('decel', v, 0, PAUSE_DECEL_MS, PAUSE_EASE);"]
    return [f"{indent}state = 'paused';", f"{indent}v = 0;"]


def _resume_start(f: _Flags, from_v: str) -> str:
    """``start('resume', …)`` ramping from ``from_v`` (linear when the easing is unknown)."""
    curve = "RESUME_EASE" if f.resume_ease else "[0, 0, 1, 1]"
    return f"start('resume', {from_v}, RESUME_PX_S, RESUME_RAMP_MS, {curve});"


def _card_scaling(x_axis: bool) -> list[str]:
    """Per-frame card scaling (P2c): ``scale = 1 + k d²`` by the on-screen distance ``d`` of a
    card's centre from the scroller centre, cards packed outward from the centre with their
    layout gaps (the gaps are not magnified). Layout positions come from ``offsetLeft`` /
    ``offsetTop`` (transforms excluded), so the scroller and the track are made positioned."""
    off, size, client = (
        ("offsetLeft", "offsetWidth", "clientWidth")
        if x_axis
        else ("offsetTop", "offsetHeight", "clientHeight")
    )
    move = "`${at[i] - u[i]}px 0`" if x_axis else "`0 ${at[i] - u[i]}px`"
    return [
        "// Card scaling: scale = 1 + k·d² by the on-screen distance d of a card's centre from the",
        "// scroller centre; the gaps keep their size, so the cards are packed outward from there",
        "const scaleK = (SCALE_AT_REF - 1) / SCALE_REF_PX ** 2;",
        "const cards = [...track.children];",
        "for (const el of [scroller, track]) {",
        "  // card offsets below are read relative to these two",
        "  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';",
        "}",
        "// screen centre of a card of length len whose inner edge is e >= 0 px from the centre",
        "const place = (e, len) =>",
        "  (2 * e + len) / (1 + Math.sqrt(Math.max(1 - scaleK * len * (2 * e + len), 0)));",
        "function scaleCards(offset) {",
        f"  const half = scroller.{client} / 2;",
        f"  const lead = cards.map((c) => track.{off} + offset + c.{off} - half);",
        f"  const len = cards.map((c) => c.{size});",
        "  const u = lead.map((p, i) => p + len[i] / 2); // unscaled centres, from the centre",
        "  const at = []; // scaled (on-screen) centres",
        "  const s = (i) => 1 + scaleK * at[i] ** 2;",
        "  let j = u.findIndex((c, i) => c + len[i] / 2 >= 0); // first card reaching the centre",
        "  if (j < 0) j = cards.length - 1;",
        "  if (lead[j] <= 0) {",
        "    // the card spans the centre: magnified about the centre by its own scale",
        "    at[j] = (2 * u[j]) / (1 + Math.sqrt(Math.max(1 - scaleK * (2 * u[j]) ** 2, 0)));",
        "  } else {",
        "    at[j] = place(lead[j], len[j]); // the centre lies in the gap before it",
        "  }",
        "  for (let i = j + 1; i < cards.length; i++) {",
        "    const gap = lead[i] - (lead[i - 1] + len[i - 1]);",
        "    at[i] = place(at[i - 1] + (len[i - 1] * s(i - 1)) / 2 + gap, len[i]);",
        "  }",
        "  for (let i = j - 1; i >= 0; i--) {",
        "    const gap = lead[i + 1] - (lead[i] + len[i]);",
        "    at[i] = -place(-(at[i + 1] - (len[i + 1] * s(i + 1)) / 2 - gap), len[i]);",
        "  }",
        "  cards.forEach((c, i) => {",
        "    const away = Math.abs(u[i]) - len[i] / 2 > half; // off screen: leave it unscaled",
        f"    c.style.translate = away ? '' : {move};",
        "    c.style.scale = away ? '' : `${s(i)}`;",
        "  });",
        "}",
        "",
    ]


def _body(spec: MotionSpec, f: _Flags, names: dict[str, str]) -> list[str]:
    c = cp.cont(spec)
    x_axis = c.axis == "x"
    sc = cp.scroller(spec)
    scroller = names[sc.id]
    out = _Lines()
    holds = _holds(f)

    out.add(
        f"const scroller = document.querySelector('.{scroller}');",
        f"const track = scroller.querySelector('.{scroller}__track');",
        "const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');",
        "",
        "let x = 0; // track offset in px, unwrapped (wrapped only when rendering)",
        "let v = 0; // velocity in px/s",
    )
    out.add(
        f"let autoV = {'AUTOPLAY_PX_S' if f.autoplay else 'RESUME_PX_S'}; "
        "// current autoplay velocity",
        when=f.autoplay or f.resume,
    )
    out.add(
        f"let state = '{'autoplay' if f.autoplay else 'paused'}'; "
        "// autoplay | decel | paused | drag | inertia | snap | waiting | resume",
        "let last = performance.now();",
    )
    out.add(
        "let ramp = null; // velocity ramp or position tween: { from, to, ms, curve, start }",
        when=f.eased,
    )
    out.add("let restAt = 0;", when=f.resume)
    out.add("let restDelay = RESUME_DELAY_MS;", when=f.resume and f.leave_delay)
    out.add("let pressed = false;", when=f.presses)
    out.add("let hovered = false;", when=f.pause_hover)
    out.add("let dragged = false;", "let startP = 0;", "let lastP = 0;", when=f.drag)
    out.add("let movedAt = 0; // time of the last pointer movement", when=f.rest_release)
    out.add(
        "const recent = []; // [time, pointer position] pairs for the release velocity",
        when=f.drag and f.inertia,
    )
    out.add("")
    out.add(
        "const lerp = (a, b, t) => a + (b - a) * t;",
        "const bez = (p1, p2, t) =>",
        "  lerp(lerp(lerp(0, p1, t), lerp(p1, p2, t), t), "
        "lerp(lerp(p1, p2, t), lerp(p2, 1, t), t), t);",
        "// cubic-bezier easing: find the curve parameter for progress p by bisection",
        "function ease([x1, y1, x2, y2], p) {",
        "  let lo = 0;",
        "  let hi = 1;",
        "  while (hi - lo > 1 / 1000) {",
        "    const mid = (lo + hi) / 2;",
        "    if (bez(x1, x2, mid) < p) lo = mid;",
        "    else hi = mid;",
        "  }",
        "  return bez(y1, y2, (lo + hi) / 2);",
        "}",
        "const progress = (now) => "
        "Math.min(Math.max((now - ramp.start) / Math.max(ramp.ms, 1), 0), 1);",
        when=f.eased,
    )
    out.add(
        "const canAutoplay = () => !reducedMotion.matches && !scroller.matches(':focus-visible');",
        when=f.autoplay or f.resume,
    )
    # One copy (the wrap period) = from the first card of copy A to the first card of copy B:
    # it includes the gap between the copies (half the track length is half a gap short, so the
    # track would jump by half a gap on every loop).
    if f.card_scale:
        out.add(
            "// one copy = first card of copy A -> first card of copy B, gap between the copies",
            "// included; layout offsets (offset*), the scaled cards' transforms must not count",
            "const copySize = () => {",
            "  const kids = track.children;",
            "  const b = kids[Math.floor(kids.length / 2)];",
            f"  return kids.length > 1 ? b.{'offsetLeft' if x_axis else 'offsetTop'} - "
            f"kids[0].{'offsetLeft' if x_axis else 'offsetTop'} : 0;",
            "};",
        )
    else:
        edge = "left" if x_axis else "top"
        out.add(
            "// one copy = first card of copy A -> first card of copy B, gap between the copies",
            "// included (both move with the track, so its translate cancels out)",
            "const copySize = () => {",
            "  const kids = track.children;",
            "  const b = kids[Math.floor(kids.length / 2)];",
            "  return kids.length > 1",
            f"    ? b.getBoundingClientRect().{edge} - kids[0].getBoundingClientRect().{edge}",
            "    : 0;",
            "};",
        )
    out.add(
        f"const pointer = (e) => e.{'clientX' if x_axis else 'clientY'};",
        when=f.drag,
    )
    out.add("")
    out.add(
        "function start(kind, from, to, ms, curve) {",
        "  state = kind;",
        "  ramp = { from, to, ms, curve, start: performance.now() };",
        "}",
        "",
        when=f.eased,
    )
    # rest(): the motion has come to rest (after a release, a snap or a hover / press ends)
    if f.drag or (f.pause and f.resume):
        if not f.resume:
            settle = ["  state = 'paused'; // resuming autoplay was not observed in the recording"]
        elif holds:
            settle = ["  restAt = performance.now();", f"  state = {holds} ? 'paused' : 'waiting';"]
        else:
            settle = ["  restAt = performance.now();", "  state = 'waiting';"]
        if f.leave_delay and f.resume:
            out.add("function rest(delay = RESUME_DELAY_MS) {", "  v = 0;", "  restDelay = delay;",
                    *settle, "}", "")  # fmt: skip
        else:
            out.add("function rest() {", "  v = 0;", *settle, "}", "")
    if f.snap:
        out.add(
            "function snapTo(target) {",
            "  const to = Math.round(target / SNAP_STEP_PX) * SNAP_STEP_PX;",
            "  v = 0;",
        )
        if f.snap_ramp:
            out.add("  start('snap', x, to, SNAP_MS, SNAP_EASE);")
        else:
            out.add("  x = to; // snap duration not measured: settle at once", "  rest();")
        out.add("}", "")
    if f.resume and (f.pause or f.presses):
        if f.leave_delay:
            out.add(
                "function letGo(delay = RESUME_DELAY_MS) {",
                f"  if (state === 'decel' && canAutoplay()) {_resume_start(f, 'v')}",
                "  else if (state === 'decel' || state === 'paused') rest(delay);",
                "}",
                "",
                when=f.pause,
            )
        else:
            out.add(
                "function letGo() {",
                f"  if (state === 'decel' && canAutoplay()) {_resume_start(f, 'v')}",
                "  else if (state === 'decel' || state === 'paused') rest();",
                "}",
                "",
                when=f.pause,
            )

    if f.card_scale:
        out.add(*_card_scaling(x_axis))

    # frame loop
    out.add("function frame(now) {", "  const dt = (now - last) / 1000;", "  last = now;")
    branches: list[list[str]] = []
    if f.autoplay or f.resume:
        branches.append(["state === 'autoplay'", "    v = canAutoplay() ? autoV : 0;"])
    if f.pause_ramp:
        branches.append(
            [
                "state === 'decel'",
                "    const p = progress(now);",
                "    v = lerp(ramp.from, ramp.to, ease(ramp.curve, p));",
                "    if (p === 1) {",
                "      v = 0;",
                "      state = 'paused';",
                "    }",
            ]
        )
    if f.resume:
        branches.append(
            [
                "state === 'resume'",
                "    const p = progress(now);",
                "    v = canAutoplay() ? lerp(ramp.from, ramp.to, ease(ramp.curve, p)) : 0;",
                "    if (p === 1) {",
                "      autoV = ramp.to;",
                "      state = 'autoplay';",
                "    }",
            ]
        )
    if f.inertia_blend:
        lines = [
            "state === 'inertia'",
            "    // decays towards the autoplay velocity (zero while autoplay is off): a release",
            "    // blends back into autoplay, which keeps running (no rest, no separate resume)",
            "    const vEnd = canAutoplay() ? autoV : 0;",
            "    v = vEnd + (v - vEnd) * Math.exp((-dt * 1000) / INERTIA_TAU_MS); "
            "// frame-rate independent",
            "    const left = ((v - vEnd) * INERTIA_TAU_MS) / 1000; "
            "// distance still to travel beyond autoplay",
        ]
        if f.snap:
            target = "x + left / SPEED_SCALE" if f.card_scale else "x + left"
            lines.append(f"    if (Math.abs(left) < SNAP_STEP_PX / 2) snapTo({target});")
        else:
            lines += [
                "    if (Math.abs(left) < 1) {",
                "      v = vEnd;",
                "      if (vEnd === 0) rest();",
                "      else state = 'autoplay';",
                "    }",
            ]
        branches.append(lines)
    elif f.inertia_exp:
        lines = [
            "state === 'inertia'",
            "    v *= Math.exp((-dt * 1000) / INERTIA_TAU_MS); // frame-rate independent decay",
            "    const left = (v * INERTIA_TAU_MS) / 1000; // distance still to travel",
        ]
        if f.snap:
            target = "x + left / SPEED_SCALE" if f.card_scale else "x + left"
            lines.append(f"    if (Math.abs(left) < SNAP_STEP_PX / 2) snapTo({target});")
        elif f.inertia_stop:
            lines.append("    if (Math.abs(v) < INERTIA_STOP_PX_S) rest();")
        else:
            lines.append("    if (Math.abs(left) < 1) rest();")
        branches.append(lines)
    if f.inertia_tween:
        branches.append(
            [
                "state === 'inertia'",
                "    const p = progress(now);",
                "    x = lerp(ramp.from, ramp.to, ease(ramp.curve, p));",
                "    if (p === 1) {",
                "      x = ramp.to;",
                "      " + ("snapTo(x);" if f.snap else "rest();"),
                "    }",
            ]
        )
    if f.snap_ramp:
        branches.append(
            [
                "state === 'snap'",
                "    const p = progress(now);",
                "    x = lerp(ramp.from, ramp.to, ease(ramp.curve, p));",
                "    if (p === 1) {",
                "      x = ramp.to; // land exactly on the grid",
                "      rest();",
                "    }",
            ]
        )
    if f.resume:
        branches.append(
            [
                "state === 'waiting'",
                "    if (now - restAt >= "
                + ("restDelay" if f.leave_delay else "RESUME_DELAY_MS")
                + " && canAutoplay()) {",
                f"      {_resume_start(f, '0')}",
                "    }",
            ]
        )
    for k, (cond, *body) in enumerate(branches):
        head = "  if" if k == 0 else "  } else if"
        out.add(f"{head} ({cond}) {{", *body)
    if branches:
        out.add("  }")
    if f.card_scale:
        out.add(
            "  // v is on screen (zero while dragging, snapping, gliding or paused); the unscaled",
            "  // track moves SPEED_SCALE times slower",
            "  x += (v * dt) / SPEED_SCALE;",
            "  const size = copySize();",
            "  if (size > 0) {",
            "    const offset = ((x % size) - size) % size; // wrap into one copy",
            f"    track.style.translate = {'`${offset}px 0`' if x_axis else '`0 ${offset}px`'};",
            "    scaleCards(offset);",
            "  }",
            "  requestAnimationFrame(frame);",
            "}",
            "",
        )
    else:
        out.add(
            "  x += v * dt; // v is zero while dragging, snapping, gliding or paused",
            "  const size = copySize();",
        )
        pos = "${((x % size) - size) % size}px"
        translate = f"`{pos} 0`" if x_axis else f"`0 {pos}`"
        out.add(
            f"  if (size > 0) track.style.translate = {translate}; // wrap into one copy",
            "  requestAnimationFrame(frame);",
            "}",
            "",
        )

    # pointer input
    if f.presses:
        out.add(
            "scroller.addEventListener('pointerdown', (e) => {",
            "  if (e.button !== 0) return;",
            "  scroller.setPointerCapture(e.pointerId);",
            "  pressed = true;",
        )
        out.add(
            "  dragged = false;",
            "  startP = pointer(e);",
            "  lastP = startP;",
            when=f.drag,
        )
        out.add(
            "  recent.length = 0;",
            "  recent.push([e.timeStamp, lastP]);",
            when=f.drag and f.inertia,
        )
        if f.pause_press:
            out.add("  if (state === 'autoplay' || state === 'resume') {")
            out.add(*_pause_start(f, "    "))
            out.add(
                "  } else {",
                "    state = 'paused'; // pressing stops any coasting at once",
                "    v = 0;",
                "  }",
            )
        else:
            out.add(
                "  state = 'drag'; // pressing takes over from autoplay and momentum", "  v = 0;"
            )
        out.add("});", "")
    if f.drag:
        out.add(
            "scroller.addEventListener('pointermove', (e) => {",
            "  if (!pressed) return;",
            "  const p = pointer(e);",
        )
        if f.pause_press:
            out.add(
                "  if (state !== 'drag') {",
                "    if (Math.abs(p - startP) <= 2) return;",
                "    state = 'drag';",
                "    v = 0;",
                "  }",
            )
        moved = "(p - lastP) * POINTER_RATIO" if f.ratio else "p - lastP"
        if f.card_scale:
            moved = f"({moved}) / SPEED_SCALE; // 1:1 on screen on average (centre: a bit less)"
        else:
            moved += ";"
        out.add(f"  x += {moved}")
        out.add("  if (p !== lastP) movedAt = e.timeStamp;", when=f.rest_release)
        out.add(
            "  lastP = p;",
            "  dragged = dragged || Math.abs(p - startP) > 2;",
        )
        if f.inertia:
            window = "INERTIA_TAU_MS" if f.inertia_exp else "INERTIA_MS"
            out.add(
                "  recent.push([e.timeStamp, p]);",
                "  while (recent.length > 2 && "
                f"e.timeStamp - recent[0][0] > {window}) recent.shift();",
            )
        out.add("});", "")
    if f.presses:
        out.add("function release(e) {", "  if (!pressed) return;", "  pressed = false;")
        if f.drag:
            out.add("  if (state === 'drag') {")
            if f.inertia:
                window = "INERTIA_TAU_MS / 2" if f.inertia_exp else "INERTIA_MS / 2"
                ratio = " * POINTER_RATIO" if f.ratio else ""
                out.add(
                    "    // release velocity from the most recent pointer movement only",
                    f"    const fresh = recent.filter(([t]) => e.timeStamp - t <= {window});",
                    "    const [t0, p0] = fresh[0] || [e.timeStamp, lastP];",
                    "    const span = e.timeStamp - t0;",
                    f"    const vRelease = span > 0 ? ((lastP - p0) / span) * 1000{ratio} : 0;",
                )
                if f.inertia_exp:
                    out.add(
                        "    if (vRelease === 0) {",
                        "      // the pointer stopped before letting go: no momentum; the content "
                        "rests",
                        "      // since its last movement (the resume delay counts from there)",
                        "      rest();",
                        "      restAt = movedAt;",
                        "      return;",
                        "    }",
                        when=f.rest_release,
                    )
                    out.add("    v = vRelease;", "    state = 'inertia';")
                else:
                    out.add(
                        "    // glide distance: release speed x duration / 2 "
                        "(speed falling to zero)",
                        "    start('inertia', x, x + (vRelease * INERTIA_MS) / 1000 / 2"
                        + (" / SPEED_SCALE" if f.card_scale else "")
                        + ", INERTIA_MS, INERTIA_EASE);",
                        "    v = 0;",
                    )
            elif f.snap:
                out.add("    snapTo(x);")
            else:
                out.add("    rest();")
            out.add("    return;", "  }")
        if f.resume and f.pause:
            out.add("  letGo();")
        elif f.pause_press:
            out.add("  // resuming autoplay was not observed in the recording")
        out.add("}", "")
        out.add(
            "scroller.addEventListener('pointerup', release);",
            "scroller.addEventListener('pointercancel', release);",
        )
    if f.drag:
        out.add(
            "scroller.addEventListener(",
            "  'click',",
            "  (e) => {",
            "    if (!dragged) return;",
            "    e.preventDefault(); // a drag is not a click on a card",
            "    e.stopPropagation();",
            "  },",
            "  true,",
            ");",
            "scroller.addEventListener('dragstart', (e) => e.preventDefault());",
        )
    if f.pause_hover:
        if f.drag or f.presses:
            out.add("")
        out.add(
            "scroller.addEventListener('pointerenter', (e) => {",
            "  if (e.pointerType !== 'mouse') return; // touch has no hover",
            "  hovered = true;",
            "  if (state === 'autoplay' || state === 'resume') {",
            *_pause_start(f, "    "),
            "  }",
            "});",
            "scroller.addEventListener('pointerleave', (e) => {",
            "  if (e.pointerType !== 'mouse') return;",
            "  hovered = false;",
        )
        if f.resume:
            go = "letGo(RESUME_AFTER_LEAVE_MS);" if f.leave_delay else "letGo();"
            out.add(f"  if (!pressed) {go}" if f.presses else f"  {go}")
        else:
            out.add(
                "  // resuming autoplay after the pointer leaves was not observed in the recording"
            )
        out.add("});")
    if out.lines[-1] != "":
        out.add("")
    out.add(
        "// requestAnimationFrame stops while the tab is hidden: "
        "restart the clock so nothing jumps",
        "document.addEventListener('visibilitychange', () => {",
        "  last = performance.now();",
        "});",
        "requestAnimationFrame(frame);",
    )
    return out.lines


def _notes(spec: MotionSpec) -> list[str]:
    c = cp.cont(spec)
    b = c.behavior
    out = []
    if b.snap is not None and b.snap.kind == "abrupt_ambiguous":
        out.append(
            "// Abrupt stops were seen but not aligned to the cards (snap or the pointer stopping"
            " before release); no snapping is implemented."
        )
    if b.snap is not None and b.snap.overshoot:
        out.append("// The snap overshot slightly (spring-like); the spring is not reconstructed.")
    if cp.pause_by_press_inferred(spec):
        out.append(
            "// Pause trigger not visible in the recording: every pause was a drag starting, so it"
            " is implemented as a press."
        )
    elif b.pause is not None and b.pause.on == "unknown":
        out.append("// Pause trigger not visible in the recording: implemented as hover.")
    return out


def render(spec: MotionSpec) -> str:
    c = cp.cont(spec)
    f = _flags(spec)
    names = ph.class_names(spec)
    scroller = names[cp.scroller(spec).id]
    head: Sequence[str] = [
        HEADER,
        f"// All values are approximate ({ph.pixel_ratio_sentence(spec)}).",
        f"// Markup: .{scroller} > .{scroller}__track holding the cards twice (two identical "
        "copies back to back).",
        f"// Signed velocities: {c.sign_convention}.",
        *_notes(spec),
    ]
    out = [*head, "", *_constants(spec, f), "", *_body(spec, f, names)]
    return "\n".join(out) + "\n"
