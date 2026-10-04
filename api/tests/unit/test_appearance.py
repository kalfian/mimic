"""Track A: measured appearance on rendered UI with known truth (PLAN-continuous §13).

Scenes are rendered in CSS px times ``k`` (1x and 2x), with anti-aliased rounded cards, drop
shadows and real glyphs (OpenCV's built-in ``sans`` font; ``size`` = CSS font-size in px), then
optionally round-tripped through H.264 (yuv420p, BT.709 tags, CRF 18 — what the synthetic suite
and macOS recordings use) and decoded the way the pipeline decodes (ffmpeg → bgr24).

DoD (Track A): flat-fill colours ΔE76 ≤ 3 before the codec and ≤ 6 after; font size ±15 %.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.ir import ElementStatic, MeasuredNumber, MotionSpec
from app.models.measure import ElementCandidate, Rect
from app.pipeline import assemble as asm
from app.pipeline import persist
from app.pipeline.appearance import (
    CAP_HEIGHT_EM,
    FONT_SIZE_CONF_CAP,
    AppearanceElement,
    AppearanceImage,
    AppearanceResult,
    ColorEstimate,
    ElementAppearance,
    FontSizeEstimate,
    analyse_text,
    background_color,
    dominant_color,
    elements_from_candidates,
    font_size,
    measure_appearance,
    page_background,
)
from app.pipeline.confidence import confidence as make_confidence
from app.pipeline.photometric import bgr_to_lab, delta_e76
from tests.unit.test_assemble import _assemble, _el, _ft, _interp, _measurement, _series

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
FONT = cv2.FontFace("sans")

#: DoD thresholds (ΔE76).
FILL_DE_RAW = 3.0
FILL_DE_H264 = 6.0
FONT_TOL = 0.15


# --------------------------------------------------------------------------------------------
# Rendering helpers (truth in CSS px)
# --------------------------------------------------------------------------------------------


def _bgr(hex_: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_[i : i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


def _lab(hex_: str) -> np.ndarray:
    return bgr_to_lab(np.array([[_bgr(hex_)]], np.uint8))[0, 0]


def de(est: ColorEstimate | None, hex_: str) -> float:
    assert est is not None, f"no estimate for {hex_}"
    return delta_e76(np.asarray(est.lab), _lab(hex_))


def _rounded_alpha(w: int, h: int, r: float) -> np.ndarray:
    """Anti-aliased rounded-rect coverage (h, w) float32 via 4x supersampling."""
    s = 4
    big = np.zeros((h * s, w * s), np.uint8)
    rr = int(round(r * s))
    if rr <= 0:
        big[:] = 255
    else:
        cv2.rectangle(big, (rr, 0), (w * s - 1 - rr, h * s - 1), 255, -1)
        cv2.rectangle(big, (0, rr), (w * s - 1, h * s - 1 - rr), 255, -1)
        for cx, cy in ((rr, rr), (w * s - 1 - rr, rr), (rr, h * s - 1 - rr),
                       (w * s - 1 - rr, h * s - 1 - rr)):  # fmt: skip
            cv2.circle(big, (cx, cy), rr, 255, -1)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


@dataclass
class Scene:
    """Painter over a page in CSS px; ``k`` image px per CSS px."""

    w: int
    h: int
    k: int
    page: str
    img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.img = np.zeros((self.h * self.k, self.w * self.k, 3), np.uint8)
        self.img[:] = _bgr(self.page)

    def box(self, x: float, y: float, w: float, h: float, color: str, radius: float = 0.0,
            shadow: tuple[float, float, float] | None = None) -> Rect:  # fmt: skip
        k = self.k
        X, Y, W, H = (int(round(v * k)) for v in (x, y, w, h))
        if shadow is not None:  # (offset_y, blur, alpha)
            oy, blur, alpha = shadow
            sh = np.zeros(self.img.shape[:2], np.float32)
            sh[Y + int(oy * k) : Y + int(oy * k) + H, X : X + W] = 1.0
            sh = cv2.GaussianBlur(sh, (0, 0), blur * k / 2)
            self.img = (self.img * (1 - alpha * sh[..., None])).round().astype(np.uint8)
        a = _rounded_alpha(W, H, radius * k)[..., None]
        roi = self.img[Y : Y + H, X : X + W].astype(np.float32)
        col = np.array(_bgr(color), np.float32)
        self.img[Y : Y + H, X : X + W] = (roi * (1 - a) + col * a).round().astype(np.uint8)
        return Rect(x, y, w, h)

    def text(self, x: float, baseline: float, text: str, size: float, color: str,
             weight: int = 400, pad: float = 2.0) -> Rect:  # fmt: skip
        """Draw ``text``; returns its ink box in CSS px grown by ``pad`` (a detector's box)."""
        k = self.k
        layer = np.zeros(self.img.shape[:2] + (3,), np.uint8)
        cv2.putText(layer, text, (int(round(x * k)), int(round(baseline * k))), (255, 255, 255),
                    FONT, int(round(size * k)), weight)  # fmt: skip
        a = layer[..., :1].astype(np.float32) / 255.0
        col = np.array(_bgr(color), np.float32)
        self.img = (self.img * (1 - a) + col * a).round().astype(np.uint8)
        ys, xs = np.nonzero(layer[..., 0] > 0)
        return Rect(xs.min() / k - pad, ys.min() / k - pad, (xs.max() + 1 - xs.min()) / k + 2 * pad,
                    (ys.max() + 1 - ys.min()) / k + 2 * pad)  # fmt: skip

    def image(self, img: np.ndarray | None = None) -> AppearanceImage:
        return AppearanceImage(bgr=self.img if img is None else img, k=float(self.k))


def h264_roundtrip(img: np.ndarray, tmp: Path, crf: int = 18, frames: int = 6) -> np.ndarray:
    """Encode a still as H.264 yuv420p (BT.709 tv, like the synth suite) and decode one
    mid-GOP frame back to BGR with ffmpeg, as ``decode.py`` does."""
    assert FFMPEG is not None
    h, w = img.shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    img = np.ascontiguousarray(img[:h2, :w2])
    out = tmp / "still.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
         "-r", "30", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv", str(out)],
        input=img.tobytes() * frames, check=True,
    )  # fmt: skip
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-nostdin", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    n = len(raw) // (w2 * h2 * 3)
    assert n >= frames // 2
    return np.frombuffer(raw, np.uint8).reshape(n, h2, w2, 3)[n // 2].copy()


# --------------------------------------------------------------------------------------------
# A product card scene (truth)
# --------------------------------------------------------------------------------------------

PAGE = "#F3F4F6"
CARD = "#FFFFFF"
MEDIA = "#C7D2FE"
TITLE = "#111827"
BODY = "#6B7280"
BUTTON = "#4F46E5"
BUTTON_TEXT = "#FFFFFF"
BADGE = "#FEF3C7"
BADGE_TEXT = "#92400E"


@dataclass
class CardScene:
    scene: Scene
    elements: list[AppearanceElement]
    font_px: dict[str, float]


def card_scene(k: int = 1, title_px: float = 20, body_px: float = 14) -> CardScene:
    s = Scene(640, 420, k, PAGE)
    card = s.box(160, 40, 320, 340, CARD, radius=12, shadow=(4, 12, 0.10))
    media = s.box(160, 40, 320, 140, MEDIA)
    badge = s.box(176, 196, 72, 22, BADGE, radius=11)
    badge_t = s.text(186, 212, "NEW", 11, BADGE_TEXT, weight=700, pad=1)
    title = s.text(176, 250, "Card title quality", title_px, TITLE, weight=600)
    body = s.text(176, 280, "Description copy goes here", body_px, BODY)
    button = s.box(176, 316, 120, 40, BUTTON, radius=8)
    button_t = s.text(200, 341, "Get started", 14, BUTTON_TEXT, weight=600)
    els = [
        AppearanceElement("e1", card, "transform"),
        AppearanceElement("e2", media, "transform", parent_id="e1"),
        AppearanceElement("e3", badge, "transform", parent_id="e1"),
        AppearanceElement("e4", badge_t, "photometric", text_like=True, parent_id="e3"),
        AppearanceElement("e5", title, "photometric", text_like=True, parent_id="e1"),
        AppearanceElement("e6", body, "transform", text_like=True, parent_id="e1"),
        AppearanceElement("e7", button, "transform", parent_id="e1"),
        AppearanceElement("e8", button_t, "transform", text_like=True, parent_id="e7"),
    ]
    return CardScene(s, els, {"e5": title_px, "e6": body_px, "e8": 14})


FILLS = {"e1": CARD, "e2": MEDIA, "e3": BADGE, "e7": BUTTON}
INKS = {"e5": TITLE, "e6": BODY, "e8": BUTTON_TEXT}


def _report(res: AppearanceResult, cs: CardScene) -> dict[str, float]:
    out = {"page": de(res.page_background, PAGE)}
    for eid, hex_ in FILLS.items():
        out[f"{eid}.bg"] = de(res.elements[eid].background, hex_)
    for eid, hex_ in INKS.items():
        out[f"{eid}.ink"] = de(res.elements[eid].text, hex_)
    for eid, truth in cs.font_px.items():
        fs = res.elements[eid].font_size
        assert fs is not None, eid
        out[f"{eid}.font_err"] = fs.px / truth - 1
    return out


# --------------------------------------------------------------------------------------------
# Accuracy (rendered truth)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
def test_card_scene_before_codec(k: int) -> None:
    cs = card_scene(k)
    res = measure_appearance(cs.scene.image(), cs.elements, (640, 420))
    rep = _report(res, cs)
    for key, v in rep.items():
        if key.endswith(".font_err"):
            assert abs(v) <= FONT_TOL, (key, v)
        elif key == "page" or key.endswith(".bg"):
            assert v <= FILL_DE_RAW, (key, v)
        else:  # ink: anti-aliased glyphs, still near-exact before the codec
            assert v <= FILL_DE_RAW, (key, v)
    # text boxes sit on their parent's fill: transparent, no background reported
    for eid in ("e4", "e5", "e6", "e8"):
        assert res.elements[eid].background is None, eid


@requires_ffmpeg
@pytest.mark.parametrize("k", [1, 2])
def test_card_scene_after_h264(k: int, tmp_path: Path) -> None:
    cs = card_scene(k)
    decoded = h264_roundtrip(cs.scene.img, tmp_path)
    res = measure_appearance(cs.scene.image(decoded), cs.elements, (640, 420))
    rep = _report(res, cs)
    for key, v in rep.items():
        if key.endswith(".font_err"):
            assert abs(v) <= FONT_TOL, (key, v)
        elif key == "page" or key.endswith(".bg"):
            assert v <= FILL_DE_H264, (key, v)
        else:
            # thin glyph strokes lose chroma in 4:2:0; neutral / dark ink stays close
            assert v <= 8.0, (key, v)


@requires_ffmpeg
@pytest.mark.parametrize(
    "fill",
    ["#FFFFFF", "#111827", "#4F46E5", "#EF4444", "#10B981", "#F59E0B", "#0EA5E9", "#E5E7EB",
     "#7C3AED", "#FDE68A"],
)  # fmt: skip
def test_flat_fill_palette(fill: str, tmp_path: Path) -> None:
    """Saturated and neutral fills next to a contrasting page, raw and after H.264."""
    s = Scene(240, 160, 1, "#F8FAFC" if fill != "#FFFFFF" else "#1F2937")
    box = s.box(40, 30, 160, 100, fill, radius=10)
    raw = background_color(s.image(), box)
    assert de(raw, fill) <= FILL_DE_RAW
    dec = background_color(s.image(h264_roundtrip(s.img, tmp_path)), box)
    assert de(dec, fill) <= FILL_DE_H264
    assert raw is not None and raw.purity > 0.9


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("size", [12, 14, 16, 20, 28, 40])
@pytest.mark.parametrize("text", ["Card title quality", "Description goes here", "HELLO 2026"])
def test_font_size_within_tolerance(k: int, size: int, text: str) -> None:
    if k == 1 and size < 14:
        pytest.skip("cap height < 10 image px: ±1 px is already ±10 %")
    s = Scene(600, 120, k, "#FFFFFF")
    box = s.text(20, 70, text, size, "#111111")
    ta = analyse_text(s.image(), box)
    assert ta is not None
    fs = font_size(ta, float(k))
    assert fs is not None
    assert abs(fs.px / size - 1) <= FONT_TOL, (fs.px, size)
    assert fs.score <= FONT_SIZE_CONF_CAP


def test_font_size_multiline_and_x_height_only_is_low() -> None:
    s = Scene(600, 160, 2, "#FFFFFF")
    a = s.text(20, 50, "First line of copy", 16, "#333333")
    b = s.text(20, 74, "second line here", 16, "#333333")
    box = Rect(a.x, a.y, max(a.w, b.w), b.y2 - a.y)
    fs = font_size(analyse_text(s.image(), box), 2.0)  # type: ignore[arg-type]
    assert fs is not None and fs.lines == 2
    assert abs(fs.px / 16 - 1) <= FONT_TOL
    # "ease" has no ascenders/descenders: reads ~x-height, reported with a lower score
    s2 = Scene(300, 100, 2, "#FFFFFF")
    box2 = s2.text(20, 60, "ease neon", 24, "#111111")
    fs2 = font_size(analyse_text(s2.image(), box2), 2.0)  # type: ignore[arg-type]
    assert fs2 is not None and fs2.px < 24 * 0.85 and fs2.score < fs.score


# --------------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------------


def test_dominant_color_ignores_minority_and_reports_purity() -> None:
    rng = np.random.default_rng(3)
    main = np.tile(_lab("#3366CC"), (900, 1)) + rng.normal(0, 0.8, (900, 3))
    other = np.tile(_lab("#000000"), (300, 1))
    est = dominant_color(np.vstack([main, other]).astype(np.float32))
    assert est is not None and est.hex == "#3366CC"
    assert est.purity == pytest.approx(0.75, abs=0.01)
    assert dominant_color(np.zeros((5, 3), np.float32)) is None


def test_background_excludes_children_and_text() -> None:
    s = Scene(300, 200, 1, "#FFFFFF")
    card = s.box(20, 20, 260, 160, "#E0F2FE", radius=8)
    child = s.box(30, 30, 240, 30, "#0F172A")  # covers most of the card's top ring
    s.text(40, 150, "Some text right along the inner edge", 12, "#000000")
    est = background_color(s.image(), card, [child])
    assert de(est, "#E0F2FE") <= FILL_DE_RAW


def test_textured_fill_is_not_reported() -> None:
    rng = np.random.default_rng(7)
    s = Scene(200, 160, 1, "#FFFFFF")
    s.img[30:130, 30:170] = rng.integers(0, 255, (100, 140, 3), dtype=np.uint8)
    assert background_color(s.image(), Rect(30, 30, 140, 100)) is None


def test_page_background_and_crop_origin() -> None:
    cs = card_scene(2)
    full = cs.scene.image()
    assert de(page_background(full, [e.box for e in cs.elements]), PAGE) <= FILL_DE_RAW
    # a crop placed by origin_css: same answers as the full frame
    crop = AppearanceImage(bgr=cs.scene.img[60:800, 280:1000], origin_css=(140.0, 30.0), k=2.0)
    assert de(background_color(crop, cs.elements[6].box), BUTTON) <= FILL_DE_RAW
    ta = analyse_text(crop, cs.elements[4].box)
    assert ta is not None and delta_e76(np.asarray(ta.color.lab), _lab(TITLE)) <= FILL_DE_RAW


def test_valid_mask_excludes_cursor_pixels() -> None:
    s = Scene(200, 160, 1, "#FFFFFF")
    card = s.box(20, 20, 160, 120, "#22C55E")
    s.img[20:140, 20:60] = 0  # a "cursor" smear over the left part of the card
    valid = np.full(s.img.shape[:2], 255, np.uint8)
    valid[20:140, 20:60] = 0
    img = AppearanceImage(bgr=s.img, valid=valid)
    assert de(background_color(img, card), "#22C55E") <= FILL_DE_RAW


def test_hidden_in_state_a_and_empty_elements_skipped() -> None:
    cs = card_scene(1)
    els = [*cs.elements, AppearanceElement("e9", Rect(0, 0, 640, 420), "backdrop"),
           AppearanceElement("e10", Rect(10, 10, 0, 0), "appear")]  # fmt: skip
    res = measure_appearance(cs.scene.image(), els, (640, 420))
    assert "e9" not in res.elements and "e10" not in res.elements
    assert list(res.elements) == [e.id for e in cs.elements if e.id in res.elements]


def test_deterministic() -> None:
    cs = card_scene(1)
    a = measure_appearance(cs.scene.image(), cs.elements, (640, 420))
    b = measure_appearance(cs.scene.image(cs.scene.img.copy()), cs.elements, (640, 420))
    assert a == b


def test_elements_from_candidates() -> None:
    c = ElementCandidate("e1", Rect(1, 2, 3, 4), Rect(1, 0, 3, 4), None, "transform", 1.0,
                         text_like=True)  # fmt: skip
    (el,) = elements_from_candidates([c])
    assert el == AppearanceElement("e1", Rect(1, 2, 3, 4), "transform", True, None)


def test_cap_height_ratio_documented() -> None:
    assert 0.68 <= CAP_HEIGHT_EM <= 0.76 and FONT_SIZE_CONF_CAP < 0.8


# --------------------------------------------------------------------------------------------
# assemble mapping (IR) + persistence
# --------------------------------------------------------------------------------------------


def _estimate(hex_: str, score: float) -> ColorEstimate:
    return ColorEstimate(hex=hex_, lab=(0.0, 0.0, 0.0), purity=0.9, spread_de=1.0, n=500,
                         score=score)  # fmt: skip


def _appearance() -> AppearanceResult:
    return AppearanceResult(
        viewport_css=(1440.0, 900.0),
        page_background=_estimate("#F3F4F6", 0.97),
        elements={
            "e1": ElementAppearance(background=_estimate("#FFFFFF", 0.99)),
            "e3": ElementAppearance(
                text=_estimate("#111111", 0.62),
                font_size=FontSizeEstimate(px=17.84, cap_height_css=12.85, lines=1, score=0.9),
            ),
            "e9": ElementAppearance(background=_estimate("#000000", 0.05)),  # too weak: dropped
        },
    )


def _m() -> asm.Measurement:
    els = [_el("e1"), _el("e3", parent="e1", box=Rect(120, 120, 200, 24))]
    fts = [_ft("t1", _series("e1", "translateY")), _ft("t2", _series("e3", "translateX"))]
    return _measurement(els, fts)


def test_apply_appearance_maps_into_statics_and_scene() -> None:
    m = _m()
    radius = ElementStatic(
        border_radius_px=MeasuredNumber(value=12.0, confidence=make_confidence(0.45))
    )
    m.statics = {"e1": radius}
    asm.apply_appearance(m, _appearance())
    e1 = m.statics["e1"]
    assert e1.border_radius_px == radius.border_radius_px  # existing values kept
    assert e1.background_color is not None and e1.background_color.value == "#FFFFFF"
    assert e1.background_color.confidence.value == 0.85  # colour cap
    e3 = m.statics["e3"]
    assert e3.text_color is not None and e3.text_color.confidence.value == 0.62
    assert e3.font_size_px is not None and e3.font_size_px.value == 17.8
    assert e3.font_size_px.confidence.value == FONT_SIZE_CONF_CAP
    assert "e9" not in m.statics
    assert m.scene is not None and m.scene.viewport_css.w == 1440.0
    assert m.scene.page_background is not None
    assert m.scene.page_background.confidence.value == 0.85


def test_assemble_without_appearance_is_unchanged_and_with_it_carries_scene(
    tmp_path: Path,
) -> None:
    plain = _m()
    dumped = _assemble(plain, _interp(plain, tmp_path)).model_dump(mode="json")
    assert "scene" not in dumped
    assert all(
        not {"background_color", "text_color", "font_size_px"} & set(e["static"])
        for e in dumped["elements"]
    )
    m = _m()
    asm.apply_appearance(m, _appearance())
    data = _assemble(m, _interp(m, tmp_path)).model_dump(mode="json")
    assert data["scene"]["viewport_css"] == {"w": 1440.0, "h": 900.0}
    e1 = next(e for e in data["elements"] if e["id"] == "e1")
    assert e1["static"]["background_color"]["value"] == "#FFFFFF"
    e3 = next(e for e in data["elements"] if e["id"] == "e3")
    assert e3["static"]["font_size_px"]["confidence"]["band"] == "medium"
    MotionSpec.model_validate(data)


def test_measurement_with_appearance_round_trips_persist() -> None:
    m = _m()
    asm.apply_appearance(m, _appearance())
    doc = persist.dump_measurement(m, [])
    m2, _ = persist.load_measurement(doc)
    assert m2.scene == m.scene and m2.statics == m.statics
    # a stored file written before ``Measurement.scene`` existed still loads
    del doc["measurement"]["scene"]
    m3, _ = persist.load_measurement(doc)
    assert m3.scene is None
