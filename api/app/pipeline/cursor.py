"""Cursor track from pass-1 diff blobs (PLAN §6.3 "Cursor vs UI blobs").

The pass-1 diff of a moving pointer produces small blobs: one at the old position and one at
the new position (or a single merged blob when the motion is slow). We link small blobs over
consecutive frames into tracks; a track that lives ≥ ``cursor_min_track_frames`` and travels
≥ ``cursor_min_disp_css`` is the cursor. Small blobs that never join such a track are UI changes
(e.g. an icon color flip) and stay in the UI energy.

Precision is ≈ ±1 blob (≈ ±15 CSS px): enough for enter/leave ordering, not for exact timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.models.measure import CursorSample, CursorTrack, Rect
from app.pipeline.params import MeasureParams

#: Nominal pointer extent around the hotspot (CSS px) used for the mask while it is stationary
#: and no blob outlines it: arrow ≈ 12×19, hand ≈ 17×22 with the hotspot near the top.
_STATIONARY_BOX = (-10.0, -4.0, 30.0, 32.0)  # dx, dy, w, h relative to the hotspot
#: Tracks whose frames mostly coincide with a large UI change are not the pointer.
#: TODO(calibrate): module-local; candidate for ``ScanParams``.
MAX_BIG_FRAC = 0.5


@dataclass(slots=True)
class Blob:
    """One connected component of a pass-1 diff, in CSS px (full-frame coordinates)."""

    x: float
    y: float
    w: float
    h: float
    area: float  # CSS px²
    top_x: float  # mean x of the topmost row (≈ arrow tip of a pointer blob)

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(slots=True)
class _Track:
    obs: list[tuple[int, list[int], int]] = field(default_factory=list)  # (frame, blobs, pos blob)
    pos: tuple[float, float] = (0.0, 0.0)
    vel: tuple[float, float] = (0.0, 0.0)
    last_i: int = -1
    first_pos: tuple[float, float] = (0.0, 0.0)
    max_disp: float = 0.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_candidate(b: Blob, cmax: float) -> bool:
    return b.w <= cmax and b.h <= cmax


def candidate_indices(frame: list[Blob], cmax: float) -> set[int]:
    """Small blobs that are not part of a larger change in the same frame.

    A small blob overlapping the box of a large blob is a fragment of that UI change (glyphs of
    a moving card); while the pointer crosses a large change it is simply not observed.
    """
    big = [b.rect.padded(2.0) for b in frame if not _is_candidate(b, cmax)]
    out: set[int] = set()
    for j, b in enumerate(frame):
        if not _is_candidate(b, cmax):
            continue
        r = b.rect
        if any(r.intersection(g) is not None for g in big):
            continue
        out.add(j)
    return out


def link_tracks(blobs: list[list[Blob]], params: MeasureParams, dt_s: float) -> list[_Track]:
    """Greedy nearest-neighbour linking of small blobs into tracks."""
    p = params.scan
    jump = p.cursor_link_max_jump_css * max(dt_s * 30.0, 1e-3)  # param is per 1/30 s
    # A stationary pointer resumes exactly where it stopped: the first diff blob of the resumed
    # motion is the pointer leaving its old spot, so its centre is within a few px.
    resume_r = p.cursor_max_css / 4.0
    tracks: list[_Track] = []
    for i, frame in enumerate(blobs):
        free = candidate_indices(frame, p.cursor_max_css)
        if not free:
            continue
        # most recently updated, longest tracks claim blobs first
        for tr in sorted(tracks, key=lambda t: (-t.last_i, -len(t.obs))):
            if not free:
                break
            gap = i - tr.last_i
            if gap <= 0:
                continue
            centers = {j: frame[j].center for j in free}
            pred = (tr.pos[0] + tr.vel[0], tr.pos[1] + tr.vel[1])
            speed = math.hypot(*tr.vel)
            # "old" blob = the pointer leaving its previous spot (≈ last position)
            old = min(free, key=lambda j: _dist(centers[j], tr.pos))
            d_old = _dist(centers[old], tr.pos)
            if gap > 1:
                if d_old > resume_r:
                    continue
                claimed, pos_blob = [old], old
                rest = [j for j in free if j != old]
                if rest:
                    new = min(rest, key=lambda j: _dist(centers[j], tr.pos))
                    if _dist(centers[new], tr.pos) <= jump:
                        claimed.append(new)
                        pos_blob = new
            else:
                has_old = d_old <= p.cursor_max_css / 2
                cands = [j for j in free if not (has_old and j == old)]
                # "new" blob must be where the motion predicts it (no hopping to UI fragments);
                # young tracks have no velocity yet and may take any blob within the jump.
                young = len(tr.obs) <= 2
                radius = jump if young else p.cursor_max_css + 0.75 * speed
                new = min(cands, key=lambda j: _dist(centers[j], pred)) if cands else None
                if new is not None and (
                    _dist(centers[new], pred) > radius or _dist(centers[new], tr.pos) > jump
                ):
                    new = None
                if new is None and not has_old:
                    continue
                claimed = ([old] if has_old else []) + ([new] if new is not None else [])
                pos_blob = new if new is not None else old
            for j in claimed:
                free.discard(j)
            c = centers[pos_blob]
            steps = max(gap, 1)
            tr.vel = ((c[0] - tr.pos[0]) / steps, (c[1] - tr.pos[1]) / steps)
            tr.pos = c
            tr.last_i = i
            tr.obs.append((i, claimed, pos_blob))
            tr.max_disp = max(tr.max_disp, _dist(c, tr.first_pos))
        for j in free:  # unclaimed blobs start new tracks
            c = frame[j].center
            tracks.append(_Track(obs=[(i, [j], j)], pos=c, last_i=i, first_pos=c))
    return tracks


def _hotspot(b: Blob, vel: tuple[float, float], dilate_css: float) -> tuple[float, float]:
    """Pointer tip estimate from its diff blob (topmost row, shifted toward the leading edge)."""
    x, y = b.top_x, b.y + dilate_css
    vx, vy = vel
    merged = b.w > 2 * abs(vx) and b.h > 2 * abs(vy)  # old+new overlap in one blob
    if not merged:
        return x, y
    if vy > 0.5:
        return x + vx, y + vy
    if vy < -0.5:
        return x, y
    return x + vx / 2.0, y


def build_cursor_track(
    blobs: list[list[Blob]],
    times: np.ndarray,
    params: MeasureParams,
) -> tuple[CursorTrack, list[set[int]]]:
    """Cursor track + per-frame set of blob indices that belong to the cursor.

    ``samples`` and ``boxes_css`` are aligned with ``times`` (one per pass-1 frame). Samples
    before the first sighting have ``state="unknown"`` and NaN coordinates.
    """
    p = params.scan
    n = len(blobs)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / p.fps
    tracks = link_tracks(blobs, params, dt)
    qualifying = [
        t
        for t in tracks
        if len(t.obs) >= p.cursor_min_track_frames and t.max_disp >= p.cursor_min_disp_css
    ]
    cursor_sets: list[set[int]] = [set() for _ in range(n)]
    if not qualifying:
        samples = [CursorSample(float(t), math.nan, math.nan, "unknown") for t in times]
        return CursorTrack(visible=False, samples=samples, boxes_css=[None] * n), cursor_sets

    # samples: longest track + other qualifying tracks that do not overlap it in time and look
    # like the same pointer (similar blob size, real travel). Fragments of a UI change hopping
    # between glyphs can "qualify" on paper; they fail the size/travel test.
    def big_frac(tr: _Track) -> float:
        """Share of the track's frames that also contain a large (non-pointer) change."""
        hits = sum(
            any(not _is_candidate(b, p.cursor_max_css) for b in blobs[i]) for i, _, _ in tr.obs
        )
        return hits / max(1, len(tr.obs))

    # A pointer moves through quiet frames too; a "track" that only exists while a large UI
    # change is on screen is fragments of that change (no-cursor recordings).
    qualifying = [t for t in qualifying if big_frac(t) < MAX_BIG_FRAC]
    if not qualifying:
        samples = [CursorSample(float(t), math.nan, math.nan, "unknown") for t in times]
        return CursorTrack(visible=False, samples=samples, boxes_css=[None] * n), cursor_sets
    qualifying.sort(key=lambda t: -len(t.obs))
    main = qualifying[0]

    # Secondary tracks (pointer re-appearing after a pause it was not seen in) must travel and
    # move through quiet frames: hopping between fragments of a UI change fails both.
    chosen: list[_Track] = [main]
    spans: list[tuple[int, int]] = [(main.obs[0][0], main.obs[-1][0])]
    for tr in qualifying[1:]:
        a, b = tr.obs[0][0], tr.obs[-1][0]
        if not all(b < s0 or a > s1 for s0, s1 in spans):
            continue
        if tr.max_disp >= 3 * p.cursor_min_disp_css and big_frac(tr) < MAX_BIG_FRAC:
            chosen.append(tr)
            spans.append((a, b))
    chosen.sort(key=lambda t: t.obs[0][0])
    for tr in chosen:
        for i, js, _ in tr.obs:
            cursor_sets[i].update(js)
    per_frame: dict[int, tuple[int, list[int], int]] = {}
    for tr in chosen:
        for ob in tr.obs:
            per_frame[ob[0]] = ob

    def rest_blob(tr: _Track) -> tuple[Blob, int, int]:
        """Blob of the pointer at rest just before ``tr`` started (the leaving diff)."""
        i0, js, pj = tr.obs[0]
        first = blobs[i0][pj]
        best_j, best_d = None, math.inf
        for j, b in enumerate(blobs[i0]):
            if j in js or not _is_candidate(b, p.cursor_max_css):
                continue
            dd = _dist(b.center, first.center)
            if dd < best_d and dd <= 2 * p.cursor_max_css:
                best_j, best_d = j, dd
        if best_j is None:
            return first, i0, pj
        cursor_sets[i0].add(best_j)
        # of the two blobs of the resume frame, the rest spot is the one *behind* the motion
        ahead_i, _, ahead_j = tr.obs[min(2, len(tr.obs) - 1)]
        ahead = blobs[ahead_i][ahead_j].center
        other = blobs[i0][best_j]
        if _dist(other.center, ahead) > _dist(first.center, ahead):
            return other, i0, best_j
        return first, i0, pj

    # where the pointer rests before each chosen track (except the first) resumes motion
    resume_at: list[tuple[int, Blob]] = [(tr.obs[0][0], rest_blob(tr)[0]) for tr in chosen[1:]]

    inflate = p.cursor_mask_inflate_css
    samples: list[CursorSample] = []
    boxes: list[Rect | None] = []
    last_hot: tuple[float, float] | None = None
    last_box: Rect | None = None
    prev_c: tuple[float, float] | None = None
    seen = 0
    for i in range(n):
        t = float(times[i])
        ob = per_frame.get(i)
        if ob is not None:
            _, js, pj = ob
            b = blobs[i][pj]
            c = b.center
            vel = (0.0, 0.0) if prev_c is None else (c[0] - prev_c[0], c[1] - prev_c[1])
            prev_c = c
            hx, hy = _hotspot(b, vel, p.dilate_css)
            last_hot = (hx, hy)
            box = blobs[i][js[0]].rect
            for j in js[1:]:
                box = box.union(blobs[i][j].rect)
            last_box = b.rect
            boxes.append(box.padded(inflate))
            samples.append(CursorSample(t, hx, hy, "moving"))
            seen += 1
        elif last_hot is not None:
            # Pointer not seen: it rests where the next chosen track resumes from (it may have
            # crossed a changing region unseen), else where it was last seen.
            hot = last_hot
            nxt = next(((i0, rb) for i0, rb in resume_at if i0 > i), None)
            far = nxt is not None and (
                last_box is None or _dist(nxt[1].center, last_box.center) > p.cursor_max_css / 4
            )
            if far:
                hot = _hotspot(nxt[1], (0.0, 0.0), p.dilate_css)
            dx, dy, bw, bh = _STATIONARY_BOX
            box = Rect(hot[0] + dx, hot[1] + dy, bw, bh)
            # the resume blob outlines the resting pointer exactly; keep the last sighting too
            # unless the pointer clearly moved on (unseen) from there
            if nxt is not None:
                box = box.union(nxt[1].rect)
            if last_box is not None and not far:
                box = box.union(last_box)
            boxes.append(box.padded(inflate))
            samples.append(CursorSample(t, hot[0], hot[1], "stationary"))
            prev_c = None
        else:
            boxes.append(None)
            samples.append(CursorSample(t, math.nan, math.nan, "unknown"))
    visible = seen >= p.cursor_visible_min_frames
    conf = 0.0
    if visible:
        conf = float(min(0.9, 0.5 + 0.02 * seen))
    # Pointer-like blobs stay out of the UI energy even when the track is too short to call
    # the cursor "visible"; the masks/samples are still produced for the frames we saw.
    return CursorTrack(visible=visible, samples=samples, boxes_css=boxes, confidence=conf), (
        cursor_sets
    )
