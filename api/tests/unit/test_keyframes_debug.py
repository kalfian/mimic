"""A9 keyframes + MIMIC_DEBUG dumps + the M1 harness end to end (Track M1)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.api.schemas import KEYFRAME_NAME_RE
from app.models.measure import ElementCandidate, Rect
from app.pipeline.debug import DebugDump, run_m1
from app.pipeline.keyframes import annotate, element_crop, keyframe_size, write_keyframes
from app.pipeline.params import DEFAULT_PARAMS as P
from tests.unit.m1_helpers import HoverScene, make_probe, render_video, requires_ffmpeg, scale1


@pytest.fixture(scope="module")
def hover(tmp_path_factory: pytest.TempPathFactory):
    scene = HoverScene()
    path = tmp_path_factory.mktemp("kf") / "hover.mp4"
    probe, frames = render_video(scene, path, 3.2)
    return scene, path, probe, frames


def test_keyframe_size_caps_long_side() -> None:
    assert keyframe_size(make_probe(10, 30, 3200, 2000), P) == (1600, 1000)
    assert keyframe_size(make_probe(10, 30, 480, 320), P) == (480, 320)


def test_annotate_and_crop_shapes() -> None:
    img = np.full((100, 200, 3), 200, np.uint8)
    out = annotate(img, [("e1", Rect(10, 10, 50, 40))], 1.0)
    assert out.shape == img.shape and not np.array_equal(out, img)
    crop = element_crop(img, img, Rect(10, 10, 50, 40), 1.0, 1600)
    assert crop.shape == (62, 2 * 72 + 4, 3)  # box ±12 px clamped at the image edge, A|sep|B


@requires_ffmpeg
def test_write_keyframes(hover, tmp_path: Path) -> None:
    _, path, probe, frames = hover
    el = ElementCandidate("e1", Rect(140, 70, 180, 200), Rect(140, 62, 180, 200), None,
                          "transform", 1.0)  # fmt: skip
    files = write_keyframes(path, tmp_path, probe, scale1(), state_a_s=0.8, state_b_s=1.8,
                            mid_s={25: 1.25, 50: 1.3, 75: 1.35}, elements=[el],
                            params=P)  # fmt: skip
    names = [f.name for f in files]
    assert names == ["state_a.png", "state_b.png", "mid_25.png", "mid_50.png", "mid_75.png",
                     "annotated_a.png", "annotated_b.png", "el_e1.png"]  # fmt: skip
    assert all(KEYFRAME_NAME_RE.match(n) for n in names)
    a = cv2.imread(str(tmp_path / "state_a.png"))
    assert a.shape == (320, 480, 3)
    assert np.abs(a.astype(int) - frames[24].astype(int)).mean() < 3  # t = 0.8 s
    assert files[-1].kind == "element" and files[-1].element_id == "e1" and files[-1].t_s is None


def test_debug_dump_disabled_is_noop(tmp_path: Path) -> None:
    d = DebugDump(None)
    assert not d.enabled
    d.write_json("x.json", {"a": 1})
    d.image("m", np.zeros((2, 2), bool))
    assert list(tmp_path.iterdir()) == []


@requires_ffmpeg
def test_run_m1_end_to_end(hover, tmp_path: Path, settings) -> None:
    scene, path, _, _ = hover
    res = run_m1(path, tmp_path / "dbg", pixel_ratio="1", keyframes=True)
    assert res.scan.primary.reverse is not None
    (card,) = [e for e in res.elements if e.kind == "transform"]
    assert card.bbox_a.y == pytest.approx(scene.card_xy[1], abs=1.5)
    by = {(s.segment, s.property): s for s in res.series if s.element_id == card.id}
    assert by[("fwd", "translateY")].v_end == pytest.approx(scene.dy, abs=0.25)
    assert by[("rev", "translateY")].v_end == pytest.approx(0.0, abs=0.25)
    assert abs(by[("fwd", "translateX")].v_end) < 0.25
    out = tmp_path / "dbg"
    for rel in ("energy.csv", "segments.json", "elements.json", "summary.json",
                "masks/state_a.png", "masks/change_mask.png", "keyframes/state_a.png"):  # fmt: skip
        assert (out / rel).exists(), rel
    assert any((out / "tracks").glob(f"{card.id}_fwd_translateY.csv"))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["elements"][0]["id"] == card.id
