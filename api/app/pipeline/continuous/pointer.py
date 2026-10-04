"""Where the pointer is while moving content hides it (PLAN-continuous §4.5 step 8, P2).

The pass-1 cursor tracker cannot follow a pointer over a moving scroller (every frame there is
"busy"), so the continuous path tracks it with the ambient regions masked out
(``scan.masked_cursor_track``): the pointer is seen up to the region edge and again once it
comes out. For the frames in between, ``cursor.build_cursor_track`` reports it "resting" at the
last / next sighting, which is wrong when it crossed into the region unseen.

:func:`occlusion_corrected` rewrites those unseen samples: a gap between two sightings is
**inside** the region when the pointer was last seen at the masked area's edge heading into it
(or already inside the region), and it stays inside until the next sighting. The corrected
samples sit at the entry point clamped into the region, with state ``stationary`` (their
position inside is unknown, so they never contribute a pointer velocity).

In-band pointer detection (motion-compensated residual) is P4; until then enter / leave times
are ±1 pass-1 frame and pointer positions inside the region are not measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.models.measure import CursorSample, Rect

#: The pointer vanishes once its whole blob is inside the masked area: its last sighting lies
#: within roughly one pointer size of that area.
EDGE_CSS = 56.0
#: The heading at the last sighting comes from an earlier sighting at least this far away and
#: at most this long before.
HEADING_MIN_CSS = 2.0
HEADING_MAX_S = 0.25


def _dist(r: Rect, x: float, y: float) -> float:
    dx = max(r.x - x, 0.0, x - r.x2)
    dy = max(r.y - y, 0.0, y - r.y2)
    return math.hypot(dx, dy)


def _inside(r: Rect, x: float, y: float) -> bool:
    return r.x <= x <= r.x2 and r.y <= y <= r.y2


def _clamp(r: Rect, x: float, y: float) -> tuple[float, float]:
    return min(max(x, r.x + 1.0), r.x2 - 1.0), min(max(y, r.y + 1.0), r.y2 - 1.0)


def occlusion_corrected(
    samples: Sequence[CursorSample], region: Rect, hidden_in: Rect, edge_css: float = EDGE_CSS
) -> list[CursorSample]:
    """``samples`` with the frames the pointer spent unseen over ``region`` placed inside it.

    ``hidden_in`` = the masked area (ambient region grown by the mask dilation) inside which the
    pointer cannot be seen. Observed samples (state ``moving``) are kept as they are.
    """
    out = list(samples)
    seen = [
        i
        for i, s in enumerate(samples)
        if s.state == "moving" and math.isfinite(s.x_css) and math.isfinite(s.y_css)
    ]
    if not seen:
        return out
    for a, b in zip(seen, [*seen[1:], None], strict=True):
        if b is not None and b - a <= 1:
            continue
        sa = samples[a]
        # direction from the latest earlier sighting that moved (a blob cut by the mask edge
        # can repeat the previous position)
        prev = next(
            (
                samples[j]
                for j in reversed(seen)
                if j < a
                and sa.t_s - samples[j].t_s <= HEADING_MAX_S
                and math.hypot(sa.x_css - samples[j].x_css, sa.y_css - samples[j].y_css)
                >= HEADING_MIN_CSS
            ),
            None,
        )
        heading_in = False
        if prev is not None:
            vx, vy = sa.x_css - prev.x_css, sa.y_css - prev.y_css
            heading_in = _dist(hidden_in, sa.x_css + vx, sa.y_css + vy) < _dist(
                hidden_in, sa.x_css, sa.y_css
            ) or _inside(hidden_in, sa.x_css + vx, sa.y_css + vy)
        at_edge = _dist(hidden_in, sa.x_css, sa.y_css) <= edge_css
        if not (_inside(region, sa.x_css, sa.y_css) or (at_edge and heading_in)):
            continue
        x, y = _clamp(region, sa.x_css, sa.y_css)
        stop = b if b is not None else len(samples)
        for j in range(a + 1, stop):
            out[j] = CursorSample(samples[j].t_s, x, y, "stationary")
    return out


__all__ = ["EDGE_CSS", "occlusion_corrected"]
