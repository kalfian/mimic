"""Cursor track from pass-1 blobs (Track M1)."""

from __future__ import annotations

import math

import numpy as np

from app.pipeline.cursor import Blob, build_cursor_track
from app.pipeline.params import DEFAULT_PARAMS as P


def pointer(x: float, y: float) -> Blob:
    return Blob(x=x, y=y, w=18, h=25, area=300.0, top_x=x + 4)


def times(n: int) -> np.ndarray:
    return np.arange(n) / 30.0


def test_moving_pointer_is_tracked_then_stationary() -> None:
    n = 30
    frames: list[list[Blob]] = [[] for _ in range(n)]
    pos = [(20.0 + 15 * k, 200.0 - 5 * k) for k in range(12)]
    for k in range(1, 12):  # frame k: old position k-1 and new position k
        frames[2 + k] = [pointer(*pos[k - 1]), pointer(*pos[k])]
    track, sets = build_cursor_track(frames, times(n), P)
    assert track.visible
    assert all(s.state == "unknown" and math.isnan(s.x_css) for s in track.samples[:3])
    assert track.samples[8].state == "moving"
    assert track.samples[20].state == "stationary"
    # stationary at the final position (hotspot ≈ blob top-left, ±15 px)
    assert abs(track.samples[20].x_css - pos[-1][0]) < 15
    assert track.boxes_css[20] is not None
    assert all(len(sets[2 + k]) == 2 for k in range(2, 12))


def test_stationary_small_ui_blob_is_not_a_cursor() -> None:
    n = 20
    frames = [[Blob(100, 100, 16, 16, 200.0, 108)] if 5 <= i < 12 else [] for i in range(n)]
    track, sets = build_cursor_track(frames, times(n), P)
    assert not track.visible
    assert all(not s for s in sets)


def test_fragments_of_large_change_are_not_a_cursor() -> None:
    n = 20
    frames: list[list[Blob]] = [[] for _ in range(n)]
    for i in range(5, 12):  # small blobs hopping around while a big change is on screen
        frames[i] = [Blob(0, 0, 300, 200, 40000.0, 150), Blob(400 + 20 * i, 50, 14, 10, 90, 405)]
    track, _ = build_cursor_track(frames, times(n), P)
    assert not track.visible


def test_resumed_pointer_rest_position_from_next_track() -> None:
    n = 40
    frames: list[list[Blob]] = [[] for _ in range(n)]
    for k in range(1, 8):  # first approach
        frames[k] = [pointer(10 + 20 * (k - 1), 300), pointer(10 + 20 * k, 300)]
    # pointer crosses a changing region unseen and rests at (400, 100); resumes at frame 30
    rest = (400.0, 100.0)
    for k in range(6):
        a = (rest[0] + 20 * k, rest[1])
        frames[30 + k] = [pointer(*a), pointer(a[0] + 20, a[1])]
    track, _ = build_cursor_track(frames, times(n), P)
    s = track.samples[20]
    assert s.state == "stationary"
    assert abs(s.x_css - rest[0]) < 20 and abs(s.y_css - rest[1]) < 20
