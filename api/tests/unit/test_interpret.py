"""Track I — interpretation: fake CLI modes, OpenAI-compatible backend, fallback, schema, check.

All images here are synthetic (drawn with OpenCV). Live tests are opt-in:

* ``uv run pytest -m claude_live tests/unit/test_interpret.py`` — real ``claude`` CLI.
* ``MIMIC_LIVE_LLM=1`` + ``MIMIC_LLM_BASE_URL/API_KEY/MODEL`` — real OpenAI-compatible gateway.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from app.config import Settings
from app.interpret import (
    ClaudeCliInterpreter,
    FallbackInterpreter,
    OpenAICompatInterpreter,
    get_interpreter,
)
from app.interpret import openai_compat as oc
from app.interpret.base import ElementSummary, InterpretationInput, VideoMeta
from app.interpret.check import check_interpreter
from app.interpret.claude_cli import parse_envelope
from app.interpret.fallback import build_fallback
from app.interpret.payload import RAW_FILENAME, extract_json_object
from app.interpret.prompt import select_images
from app.interpret.schema import ROLES, build_schema
from app.models.ir import Box

FAKE_CLAUDE = Path(__file__).resolve().parents[1] / "fixtures" / "fake_claude.py"
IDS = ["e1", "e2", "e3"]
TOKEN = "tok-test-1234567890-secret"  # fake value for redaction tests
BASE_URL = "http://llm.test/v1"


# ---- synthetic keyframes -------------------------------------------------------------------------


def _page() -> np.ndarray:
    img = np.full((560, 640, 3), 242, np.uint8)
    cv2.rectangle(img, (0, 0), (640, 40), (225, 225, 225), -1)  # top bar
    cv2.putText(img, "Shop", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    return img


def _texture(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    tex = np.zeros((h, w, 3), np.uint8)
    tex[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    tex[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    tex[..., 2] = 180
    tex = cv2.add(tex, rng.integers(0, 60, (h, w, 3), dtype=np.uint8))
    cv2.circle(tex, (w // 3, h // 2), min(h, w) // 4, (30, 200, 240), -1)
    cv2.rectangle(tex, (w // 2, h // 4), (w - 20, h - 20), (40, 40, 200), -1)
    return tex


def _card(img: np.ndarray, dy: int, scale: float, title_color: tuple[int, int, int]) -> None:
    x, y = 100, 80 + dy
    cv2.rectangle(img, (x, y), (x + 320, y + 400), (255, 255, 255), -1)
    tex = _texture(180, 288)
    if scale != 1.0:
        big = cv2.resize(tex, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        oy, ox = (big.shape[0] - 180) // 2, (big.shape[1] - 288) // 2
        tex = big[oy : oy + 180, ox : ox + 288]
    img[y + 16 : y + 196, x + 16 : x + 304] = tex
    cv2.putText(
        img, "Product title", (x + 16, y + 236), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2,
        cv2.LINE_AA,
    )  # fmt: skip
    cv2.putText(
        img, "$ 49", (x + 16, y + 276), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA
    )


def make_keyframes(job_dir: Path) -> dict[str, Path]:
    """Synthetic card-hover keyframe set (state A/B, mid, annotated, element crops)."""
    kf = job_dir / "keyframes"
    kf.mkdir(parents=True, exist_ok=True)
    a, b, mid = _page(), _page(), _page()
    _card(a, 0, 1.0, (20, 20, 20))
    _card(b, -8, 1.06, (255, 80, 80))
    _card(mid, -4, 1.03, (140, 50, 50))
    ann = a.copy()
    boxes = {"e1": (100, 80, 320, 400), "e2": (116, 96, 288, 180), "e3": (116, 292, 200, 32)}
    for el_id, (x, y, w, h) in boxes.items():
        cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(ann, el_id, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out = {"state_a.png": a, "state_b.png": b, "mid_50.png": mid, "annotated_a.png": ann}
    for el_id, (x, y, w, h) in boxes.items():
        y0 = max(y - 12, 0)
        out[f"el_{el_id}.png"] = np.hstack([a[y0 : y + h, x : x + w], b[y0 : y + h, x : x + w]])
    paths = {}
    for name, img in out.items():
        cv2.imwrite(str(kf / name), img)
        paths[name] = kf / name
    return paths


def make_input(job_dir: Path, keyframes: dict[str, Path]) -> InterpretationInput:
    return InterpretationInput(
        job_dir=job_dir,
        keyframe_paths=keyframes,
        elements=[
            ElementSummary(
                id="e1", bbox=Box(x=100, y=80, w=320, h=400), kind="transform",
                change_summary="moves up ~8px",
            ),
            ElementSummary(
                id="e2", bbox=Box(x=116, y=96, w=288, h=180), kind="transform", parent_id="e1",
                change_summary="scales up ~6%",
            ),
            ElementSummary(
                id="e3", bbox=Box(x=116, y=292, w=200, h=32), kind="photometric",
                parent_id="e1", text_like=True, change_summary="text color changes",
            ),
        ],
        heuristic_type="hover",
        heuristic_trigger="pointer_enter",
        cursor_summary="enters e1 at 1000 ms, moving at onset",
        video_meta=VideoMeta(width=640, height=560, duration_ms=4000, pixel_ratio=1),
    )  # fmt: skip


@pytest.fixture
def job(tmp_path: Path) -> InterpretationInput:
    job_dir = tmp_path / "jobs" / ("a" * 32)
    job_dir.mkdir(parents=True)
    return make_input(job_dir, make_keyframes(job_dir))


@pytest.fixture
def make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    for var in list(os.environ):
        if var.startswith("MIMIC_"):
            monkeypatch.delenv(var, raising=False)

    def _make(**kw: Any) -> Settings:
        kw.setdefault("data_dir", tmp_path / "data")
        return Settings(_env_file=None, **kw)  # type: ignore[call-arg]

    return _make


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """Executable wrapper so the fake runs under this interpreter (``exec`` keeps the pid)."""
    wrapper = tmp_path / "bin" / "claude"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_CLAUDE}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


@pytest.fixture
def cli(
    make_settings: Callable[..., Settings], fake_bin: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., tuple[ClaudeCliInterpreter, Path]]:  # fmt: skip
    record = tmp_path / "record.json"

    def _make(mode: str, timeout_s: float = 20.0, **kw: Any) -> tuple[ClaudeCliInterpreter, Path]:
        monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
        monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(record))
        s = make_settings(
            interpreter="claude_cli", claude_bin=str(fake_bin), claude_timeout_s=timeout_s
        )
        return ClaudeCliInterpreter(s, **kw), record

    return _make


@pytest.fixture(autouse=True)
def _clear_mode_cache() -> None:
    oc._MODE_CACHE.clear()


def _raw(job: InterpretationInput) -> dict[str, Any]:
    return json.loads((job.job_dir / RAW_FILENAME).read_text())


def assert_complete(result: Any, ids: list[str] = IDS) -> None:
    """Every input id labeled, nothing else; interaction present; valid roles."""
    assert sorted(result.elements) == sorted(ids)
    for el in result.elements.values():
        assert el.label and len(el.label) <= 60
        assert el.role in ROLES
        assert el.parent_id is None or el.parent_id in ids
    assert result.interaction.type_confirmation
    t = result.interaction.target_element_id
    assert t is None or t in ids


# ---- schema --------------------------------------------------------------------------------------


def _number_fields(node: Any, name: str = "") -> list[str]:
    out = []
    if isinstance(node, dict):
        t = node.get("type")
        types = t if isinstance(t, list) else [t]
        if "number" in types or "integer" in types:
            out.append(name)
        for k, v in node.get("properties", {}).items():
            out += _number_fields(v, k)
        if "items" in node:
            out += _number_fields(node["items"], name)
    return out


def test_schema_enums_and_no_numbers() -> None:
    s = build_schema(IDS)
    item = s["properties"]["elements"]["items"]["properties"]
    assert item["id"]["enum"] == IDS
    assert item["parent_id"]["enum"] == [*IDS, None]
    assert item["role"]["enum"] == list(ROLES)
    assert s["properties"]["interaction"]["properties"]["target_element_id"]["enum"] == [
        *IDS,
        None,
    ]
    assert sorted(_number_fields(s)) == ["confidence", "type_confidence"]
    strict = oc.to_strict_schema(s)
    el = strict["properties"]["elements"]["items"]
    assert set(el["required"]) == set(el["properties"])
    assert "maxLength" not in json.dumps(strict)
    assert sorted(_number_fields(strict)) == ["confidence", "type_confidence"]


def test_schema_without_elements() -> None:
    s = build_schema([])
    assert s["properties"]["elements"]["maxItems"] == 0
    assert s["properties"]["interaction"]["properties"]["target_element_id"] == {"type": "null"}


# ---- parsing helpers -----------------------------------------------------------------------------


def test_parse_envelope_paths() -> None:
    p = {"elements": [], "interaction": {}, "structure": []}
    env = {"type": "result", "subtype": "success", "is_error": False}
    assert parse_envelope(json.dumps({**env, "structured_output": p})).path == "structured_output"
    assert parse_envelope(json.dumps({**env, "result": json.dumps(p)})).path == "result_json"
    prose = {**env, "result": 'Sure! {"a": "}"} then ' + json.dumps(p)}
    out = parse_envelope(json.dumps(prose))
    assert out.path == "result_braces" and out.payload == {"a": "}"}
    assert parse_envelope("not json").path == "no_envelope"
    assert parse_envelope(json.dumps({**env, "is_error": True})).payload is None
    assert parse_envelope("log line\n" + json.dumps({**env, "structured_output": p})).payload == p


def test_extract_json_object() -> None:
    assert extract_json_object('{"a": 1}') == ({"a": 1}, "json")
    assert extract_json_object('Here:\n```json\n{"a": 2}\n```') == ({"a": 2}, "fenced")
    assert extract_json_object('x {"a": 3} y') == ({"a": 3}, "braces")
    assert extract_json_object("nothing") == (None, "none")


# ---- fallback ------------------------------------------------------------------------------------


def test_disabled_by_default(job: InterpretationInput, make_settings: Callable) -> None:
    s = make_settings()  # interpreter=claude_cli by default, but the job did not opt in
    interp = get_interpreter(s, use_interpreter=False)
    assert isinstance(interp, FallbackInterpreter)
    r = interp.interpret(job)
    assert r.status == "disabled" and r.provider == "none" and r.model is None
    assert_complete(r)
    assert not (job.job_dir / RAW_FILENAME).exists()  # nothing was sent anywhere


def test_none_mode_is_disabled_even_when_opted_in(make_settings: Callable) -> None:
    interp = get_interpreter(make_settings(interpreter="none"), use_interpreter=True)
    assert isinstance(interp, FallbackInterpreter) and interp.status == "disabled"


def test_fallback_output_complete(job: InterpretationInput) -> None:
    r = build_fallback(job, status="disabled")
    assert_complete(r)
    assert {k: v.role for k, v in r.elements.items()} == {"e1": "card", "e2": "image", "e3": "text"}
    assert r.elements["e1"].label == "Card (e1)"
    assert r.elements["e2"].parent_id == "e1"
    assert r.elements["e2"].description == "scales up ~6%"
    assert r.interaction.target_element_id == "e1"  # from the cursor summary
    assert r.interaction.target_description == "Card (e1)"
    assert r.interaction.type_confirmation == "hover"
    assert r.interaction.trigger_description == "Pointer enters the target"
    assert r.structure == ["image", "text"]
    assert r.notes and "heuristic" in r.notes[0]
    assert build_fallback(job, status="disabled") == r  # deterministic


def _el(el_id: str, x: float, y: float, w: float, h: float, kind: str, **kw: Any) -> ElementSummary:
    return ElementSummary(
        id=el_id, bbox=Box(x=x, y=y, w=w, h=h), kind=kind, change_summary=kw.pop("cs", ""), **kw
    )


def _inp(tmp_path: Path, els: list[ElementSummary], **kw: Any) -> InterpretationInput:
    return InterpretationInput(
        job_dir=tmp_path, keyframe_paths={}, elements=els,
        heuristic_type=kw.get("type", "unknown"), heuristic_trigger="unknown",
        cursor_summary=kw.get("cursor", "cursor not visible"),
        video_meta=VideoMeta(width=2560, height=1600, duration_ms=3000, pixel_ratio=2),
    )  # fmt: skip


def test_fallback_role_rules(tmp_path: Path) -> None:
    # frame is 1280x800 CSS (2x source)
    modal = _inp(tmp_path, [
        _el("e1", 0, 0, 1280, 800, "backdrop"),
        _el("e2", 390, 200, 500, 400, "appear"),
    ], type="modal")  # fmt: skip
    r = build_fallback(modal, status="fallback")
    assert r.elements["e1"].role == "backdrop" and r.elements["e2"].role == "modal_panel"

    dropdown = _inp(tmp_path, [
        _el("e1", 100, 100, 120, 40, "photometric", cs="background color changes"),
        _el("e2", 100, 146, 200, 160, "appear"),
    ])  # fmt: skip
    r = build_fallback(dropdown, status="fallback")
    assert r.elements["e1"].role == "button" and r.elements["e2"].role == "dropdown_menu"
    assert r.structure == []  # e1 (largest... no transform) has no children

    accordion = _inp(tmp_path, [
        _el("e1", 100, 100, 400, 120, "resize"),
        _el("e2", 100, 240, 400, 300, "transform"),
    ])  # fmt: skip
    r = build_fallback(accordion, status="fallback")
    assert r.elements["e1"].role == "accordion_panel" and r.elements["e2"].role == "other"
    assert r.interaction.target_element_id == "e2"  # largest transform element

    assert (
        build_fallback(_inp(tmp_path, []), status="fallback").interaction.target_element_id is None
    )


def test_select_images_stays_inside_keyframes_dir(job: InterpretationInput, tmp_path: Path) -> None:
    outside = tmp_path / "secret.png"
    cv2.imwrite(str(outside), np.zeros((4, 4, 3), np.uint8))
    paths = {**job.keyframe_paths, "state_b.png": outside, "el_e1.png": Path("../../secret.png")}
    inp = job.model_copy(update={"keyframe_paths": paths})
    images = select_images(inp, job.job_dir / "keyframes")
    names = [p.name for p in images]
    assert "secret.png" not in names and "state_b.png" not in names
    assert names[:3] == ["annotated_a.png", "state_a.png", "mid_50.png"]
    assert all(p.parent == (job.job_dir / "keyframes").resolve() for p in images)


# ---- claude_cli with the fake CLI ----------------------------------------------------------------


def test_cli_ok(job: InterpretationInput, cli: Callable) -> None:
    interp, record = cli("ok")
    r = interp.interpret(job)
    assert r.status == "ok" and r.provider == "claude_cli" and r.model == "claude-fake-sonnet"
    assert_complete(r)
    assert r.elements["e1"].label == "Fake card e1" and r.elements["e1"].role == "card"
    assert r.elements["e2"].parent_id == "e1"
    assert r.interaction.type_confirmation == "hover" and r.interaction.type_confidence == 0.85
    assert r.interaction.target_element_id == "e1"
    assert r.structure == ["image", "title", "price"]
    assert r.duration_ms is not None and r.duration_ms >= 0

    rec = json.loads(record.read_text())
    argv, kf = rec["argv"], str((job.job_dir / "keyframes").resolve())
    assert Path(rec["cwd"]).resolve() == Path(kf)
    flag = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert flag["--tools"] == "Read" and flag["--allowedTools"] == "Read"
    assert flag["--permission-mode"] == "dontAsk"
    assert flag["--add-dir"] == kf and flag["--output-format"] == "json"
    assert {"--strict-mcp-config", "--no-session-persistence", "--safe-mode"} <= set(argv)
    schema = json.loads(flag["--json-schema"])
    assert schema["properties"]["elements"]["items"]["properties"]["id"]["enum"] == IDS
    prompt = argv[argv.index("-p") + 1]
    assert f"{kf}/annotated_a.png" in prompt and f"{kf}/el_e3.png" in prompt
    assert "Do not output any motion numbers" in prompt

    raw = _raw(job)
    assert raw["status"] == "ok" and raw["parse_path"] == "structured_output"
    assert raw["exit_code"] == 0 and raw["envelope"]["type"] == "result"
    assert "<prompt:" in " ".join(raw["argv"]) and prompt not in json.dumps(raw)


@pytest.mark.parametrize(
    ("mode", "path"), [("envelope_result_only", "result_json"), ("braces", "result_braces")]
)
def test_cli_parser_fallback_paths(
    job: InterpretationInput, cli: Callable, mode: str, path: str
) -> None:
    r = cli(mode)[0].interpret(job)
    assert r.status == "ok"
    assert_complete(r)
    assert _raw(job)["parse_path"] == path


def test_cli_invalid_ids_sanitized(job: InterpretationInput, cli: Callable) -> None:
    r = cli("bad_ids")[0].interpret(job)
    assert r.status == "ok"
    assert_complete(r)
    e1 = r.elements["e1"]
    assert e1.role == "other"  # "spaceship" is not a role
    assert e1.parent_id is None  # "e77" unknown -> CV parent (none)
    assert e1.label.startswith("Bad label") and len(e1.label) <= 60
    assert not any(ord(c) < 32 for c in e1.label)
    assert r.interaction.target_element_id == "e1"  # "e42" unknown -> heuristic target
    assert r.interaction.type_confirmation == "unknown" and r.interaction.type_confidence == 0
    assert r.structure == ["image", "y" * 40]
    assert any(n.startswith("AI output repaired") for n in r.notes)
    issues = " ".join(_raw(job)["issues"])
    assert "'e99'" in issues and "'e42'" in issues and "'e77'" in issues


@pytest.mark.parametrize("mode", ["garbage", "error", "is_error"])
def test_cli_failures_return_complete_fallback(
    job: InterpretationInput, cli: Callable, mode: str
) -> None:
    r = cli(mode)[0].interpret(job)
    assert r.status == "error" and r.provider == "claude_cli"
    assert_complete(r)
    assert r.elements["e1"].label == "Card (e1)"  # heuristic labels
    assert r.interaction.type_confidence == 0.0
    raw = _raw(job)
    assert raw["status"] == "error" and raw["error"]
    if mode == "error":
        assert raw["exit_code"] == 1 and "fake failure" in raw["stderr_tail"]
    if mode == "garbage":
        assert raw["parse_path"] == "no_envelope" and "not json" in raw["stdout_text"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_cli_timeout_kills_process_group(
    job: InterpretationInput, cli: Callable, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pidfile = tmp_path / "pids"
    monkeypatch.setenv("FAKE_CLAUDE_PIDFILE", str(pidfile))
    timeout = 1.5
    interp, _ = cli("timeout", timeout_s=timeout)
    t0 = time.monotonic()
    r = interp.interpret(job)
    elapsed = time.monotonic() - t0
    assert r.status == "timeout" and r.provider == "claude_cli"
    assert_complete(r)
    assert elapsed < timeout + 5.0
    assert _raw(job)["timed_out"] is True

    child, grandchild = (int(x) for x in pidfile.read_text().split())
    deadline = time.monotonic() + 3.0
    while (_alive(child) or _alive(grandchild)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child), "CLI process survived the timeout"
    assert not _alive(grandchild), "process-group member survived the timeout"


def test_cli_missing_binary(job: InterpretationInput, make_settings: Callable) -> None:
    s = make_settings(interpreter="claude_cli", claude_bin="/nonexistent/claude-xyz")
    interp = get_interpreter(s, use_interpreter=True)
    assert isinstance(interp, FallbackInterpreter) and interp.status == "fallback"
    r = interp.interpret(job)
    assert r.status == "fallback" and r.provider == "none"
    assert_complete(r)
    # Constructed directly (binary vanished after selection): error + fallback, no raise.
    r2 = ClaudeCliInterpreter(s).interpret(job)
    assert r2.status == "error" and "could not start" in " ".join(r2.notes)
    assert_complete(r2)


def test_cli_progress_ticks(
    job: InterpretationInput, cli: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks: list[float] = []
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "1.3")
    interp, _ = cli("ok", timeout_s=20, on_progress=ticks.append)
    assert interp.interpret(job).status == "ok"
    assert ticks and all(0 < t <= 1 for t in ticks)


def test_cli_no_keyframes(job: InterpretationInput, cli: Callable) -> None:
    r = cli("ok")[0].interpret(job.model_copy(update={"keyframe_paths": {}}))
    assert r.status == "fallback"
    assert_complete(r)


def test_get_interpreter_selects_cli(make_settings: Callable, fake_bin: Path) -> None:
    s = make_settings(interpreter="claude_cli", claude_bin=str(fake_bin))
    assert isinstance(get_interpreter(s, use_interpreter=True), ClaudeCliInterpreter)


# ---- openai_compat with httpx.MockTransport ------------------------------------------------------


def _payload(ids: list[str] = IDS) -> dict[str, Any]:
    return {
        "elements": [
            {"id": i, "label": f"Label {i}", "role": "card" if i == "e1" else "image",
             "parent_id": None if i == "e1" else "e1", "confidence": 0.8}
            for i in ids
        ],
        "interaction": {"type": "hover", "type_confidence": 0.9, "target_element_id": "e1"},
        "structure": ["image", "title"],
    }  # fmt: skip


def _chat(content: str, model: str = "gw-model-1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        },
    )  # fmt: skip


def _unsupported(what: str) -> httpx.Response:
    return httpx.Response(400, json={"error": {"message": f"response_format {what} not supported"}})


class Gateway:
    """Programmable fake gateway. ``chat`` = list of responses/exceptions, consumed in order."""

    def __init__(self, chat: list[Any], models: Any = None):
        self.chat = list(chat)
        self.models = (
            models
            if models is not None
            else httpx.Response(
                200, json={"object": "list", "data": [{"id": "gw-model-1"}, {"id": "other"}]}
            )
        )
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.models if request.url.path.endswith("/models") else self.chat.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def chat_bodies(self) -> list[dict[str, Any]]:
        return [
            json.loads(r.content) for r in self.requests if r.url.path.endswith("/chat/completions")
        ]


@pytest.fixture
def llm_settings(make_settings: Callable) -> Callable[..., Settings]:
    def _make(**kw: Any) -> Settings:
        base = {
            "interpreter": "openai_compat", "llm_base_url": BASE_URL, "llm_api_key": TOKEN,
            "llm_model": "gw-model-1", "llm_timeout_s": 10,
        }  # fmt: skip
        return make_settings(**{**base, **kw})

    return _make


def _llm(settings: Settings, gw: Gateway) -> OpenAICompatInterpreter:
    return OpenAICompatInterpreter(settings, transport=httpx.MockTransport(gw))


def test_llm_json_schema_ok(job: InterpretationInput, llm_settings: Callable) -> None:
    gw = Gateway([_chat(json.dumps(_payload()))])
    r = _llm(llm_settings(), gw).interpret(job)
    assert r.status == "ok" and r.provider == "openai_compat" and r.model == "gw-model-1"
    assert_complete(r)
    assert r.elements["e2"].label == "Label e2" and r.elements["e2"].parent_id == "e1"

    req = gw.requests[0]
    assert str(req.url) == f"{BASE_URL}/chat/completions"
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    body = gw.chat_bodies[0]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    ids_enum = rf["json_schema"]["schema"]["properties"]["elements"]["items"]["properties"]["id"]
    assert ids_enum["enum"] == IDS
    parts = body["messages"][1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 7  # annotated_a, state_a, state_b, mid_50, el_e1..e3
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in images)
    assert (
        "attached images" in parts[0]["text"] and '"additionalProperties"' not in parts[0]["text"]
    )

    raw_text = (job.job_dir / RAW_FILENAME).read_text()
    assert TOKEN not in raw_text and "base64" not in raw_text
    assert json.loads(raw_text)["mode"] == "json_schema"


def test_llm_downgrades_to_json_object(job: InterpretationInput, llm_settings: Callable) -> None:
    gw = Gateway([_unsupported("json_schema"), _chat(json.dumps(_payload()))])
    s = llm_settings()
    r = _llm(s, gw).interpret(job)
    assert r.status == "ok"
    bodies = gw.chat_bodies
    assert [b["response_format"]["type"] for b in bodies] == ["json_schema", "json_object"]
    assert '"additionalProperties"' in bodies[1]["messages"][1]["content"][0]["text"]
    raw = _raw(job)
    assert raw["mode"] == "json_object" and len(raw["attempts"]) == 2

    # The working mode is cached: the next job starts at json_object directly.
    gw2 = Gateway([_chat(json.dumps(_payload()))])
    assert _llm(s, gw2).interpret(job).status == "ok"
    assert gw2.chat_bodies[0]["response_format"]["type"] == "json_object"


def test_llm_prompt_only_fenced(job: InterpretationInput, llm_settings: Callable) -> None:
    fenced = "Sure, here you go:\n```json\n" + json.dumps(_payload()) + "\n```"
    gw = Gateway([_unsupported("json_schema"), _unsupported("json_object"), _chat(fenced)])
    r = _llm(llm_settings(), gw).interpret(job)
    assert r.status == "ok"
    assert_complete(r)
    assert "response_format" not in gw.chat_bodies[2]
    raw = _raw(job)
    assert raw["mode"] == "prompt_only" and raw["parse_path"] == "fenced"


def test_llm_forced_json_schema_does_not_downgrade(
    job: InterpretationInput, llm_settings: Callable
) -> None:
    gw = Gateway([_unsupported("json_schema")])
    r = _llm(llm_settings(llm_supports_json_schema=True), gw).interpret(job)
    assert r.status == "error" and len(gw.chat_bodies) == 1
    assert_complete(r)


def test_llm_invalid_ids_sanitized(job: InterpretationInput, llm_settings: Callable) -> None:
    p = _payload()
    p["elements"].append({"id": "e9", "label": "Ghost", "role": "card", "confidence": 2})
    p["elements"][0]["role"] = "hero"
    p["interaction"]["target_element_id"] = "e9"
    p["notes"] = None  # strict-mode style null
    r = _llm(llm_settings(), Gateway([_chat(json.dumps(p))])).interpret(job)
    assert r.status == "ok"
    assert_complete(r)
    assert r.elements["e1"].role == "other"
    assert r.interaction.target_element_id == "e1"


@pytest.mark.parametrize(
    ("response", "status", "code"),
    [
        (httpx.Response(401, json={"error": {"message": f"bad key {TOKEN}"}}), "error",
         "unauthorized"),
        (httpx.ReadTimeout("slow"), "timeout", "timeout"),
        (httpx.ConnectError("Name or service not known"), "error", "unreachable"),
        (httpx.Response(500, text="upstream exploded"), "error", "bad_response"),
        (_chat("I cannot help with that."), "error", "bad_response"),
        (httpx.Response(404, json={"error": {"message": "model gw-model-1 not found"}}), "error",
         "model_not_found"),
    ],
)  # fmt: skip
def test_llm_failures_return_complete_fallback(
    job: InterpretationInput, llm_settings: Callable, response: Any, status: str, code: str
) -> None:
    r = _llm(llm_settings(), Gateway([response])).interpret(job)
    assert r.status == status and r.provider == "openai_compat"
    assert_complete(r)
    assert r.elements["e1"].label == "Card (e1)"
    assert any(n.startswith(code) for n in r.notes)
    dumped = r.model_dump_json() + (job.job_dir / RAW_FILENAME).read_text()
    assert TOKEN not in dumped


def test_llm_secret_not_in_repr(llm_settings: Callable) -> None:
    s = llm_settings()
    interp = _llm(s, Gateway([]))
    assert TOKEN not in repr(interp) and TOKEN not in repr(interp.client) and TOKEN not in repr(s)


def test_get_interpreter_selects_openai(llm_settings: Callable) -> None:
    assert isinstance(
        get_interpreter(llm_settings(), use_interpreter=True), OpenAICompatInterpreter
    )
    assert isinstance(get_interpreter(llm_settings(), use_interpreter=False), FallbackInterpreter)
    unconfigured = get_interpreter(llm_settings(llm_api_key=None), use_interpreter=True)
    assert isinstance(unconfigured, FallbackInterpreter) and unconfigured.status == "fallback"
    assert "MIMIC_LLM_API_KEY" in (unconfigured.reason or "")


# ---- check_interpreter ---------------------------------------------------------------------------


PROBE_OK = json.dumps({"shape": "rectangle", "color": "blue"})


def _check(settings: Settings, gw: Gateway) -> Any:
    return check_interpreter(settings, transport=httpx.MockTransport(gw))


def test_check_openai_ok(llm_settings: Callable) -> None:
    gw = Gateway([_chat(PROBE_OK)])
    c = _check(llm_settings(), gw)
    assert c.ok and c.mode == "openai_compat" and c.error is None
    assert c.model_listed is True and c.supports_images is True
    assert c.structured_output == "json_schema" and isinstance(c.latency_ms, int)
    assert c.checked_at.tzinfo is not None
    assert gw.requests[0].url.path == "/v1/models"
    parts = gw.chat_bodies[0]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_check_openai_json_object(llm_settings: Callable) -> None:
    c = _check(llm_settings(), Gateway([_unsupported("json_schema"), _chat(PROBE_OK)]))
    assert c.ok and c.structured_output == "json_object"


def test_check_openai_not_configured(llm_settings: Callable) -> None:
    gw = Gateway([])
    c = _check(llm_settings(llm_base_url=None, llm_model=None), gw)
    assert not c.ok and c.error.code == "not_configured"
    assert "MIMIC_LLM_BASE_URL" in c.error.message and "MIMIC_LLM_MODEL" in c.error.message
    assert gw.requests == []  # no network call


@pytest.mark.parametrize(
    ("models", "chat", "code"),
    [
        (httpx.Response(401, json={"error": f"invalid token {TOKEN}"}), [], "unauthorized"),
        (httpx.Response(403, text="forbidden"), [], "unauthorized"),
        (httpx.ConnectError("[Errno 8] nodename nor servname provided"), [], "unreachable"),
        (httpx.ConnectTimeout("connect timed out"), [], "timeout"),
        (None, [httpx.ReadTimeout("read timed out")], "timeout"),
        (None, [_chat(json.dumps({"shape": "none", "color": "other"}))], "no_image_support"),
        (None, [httpx.Response(400, json={"error": {"message": "image input is not supported"}})],
         "no_image_support"),
        (None, [_chat("garbage")], "bad_response"),
    ],
)  # fmt: skip
def test_check_openai_errors(llm_settings: Callable, models: Any, chat: list, code: str) -> None:
    c = _check(llm_settings(), Gateway(chat, models=models))
    assert not c.ok and c.error is not None and c.error.code == code
    assert TOKEN not in c.model_dump_json()


def test_check_openai_model_not_listed(llm_settings: Callable) -> None:
    models = httpx.Response(200, json={"data": [{"id": "something-else"}]})
    gw = Gateway([httpx.Response(404, json={"error": {"message": "model does not exist"}})], models)
    c = _check(llm_settings(), gw)
    assert not c.ok and c.error.code == "model_not_found" and c.model_listed is False
    # Listed under another alias but the chat works: still ok, model_listed reports False.
    c2 = _check(llm_settings(), Gateway([_chat(PROBE_OK)], models))
    assert c2.ok and c2.model_listed is False


def test_check_openai_without_models_endpoint(llm_settings: Callable) -> None:
    c = _check(llm_settings(), Gateway([_chat(PROBE_OK)], models=httpx.Response(404, text="")))
    assert c.ok and c.model_listed is None and isinstance(c.latency_ms, int)


def test_check_none_and_claude(make_settings: Callable, fake_bin: Path) -> None:
    c = check_interpreter(make_settings(interpreter="none"))
    assert c.mode == "none" and not c.ok and c.error.code == "not_configured"

    missing = check_interpreter(make_settings(claude_bin="/nonexistent/claude-xyz"))
    assert missing.mode == "claude_cli" and not missing.ok
    assert missing.error.code == "not_configured"

    s = make_settings(interpreter="claude_cli", claude_bin=str(fake_bin))
    c = check_interpreter(s)
    assert c.ok and c.model == "sonnet" and c.supports_images is None
    deep = check_interpreter(s, deep=True)
    assert deep.ok and deep.supports_images is True and deep.structured_output == "json_schema"


# ---- live (opt-in) -------------------------------------------------------------------------------


def _assert_live_result(r: Any) -> None:
    assert r.status == "ok", r.notes
    assert_complete(r)
    labels = [el.label.lower() for el in r.elements.values()]
    assert all(2 <= len(lbl) <= 60 for lbl in labels)
    assert not any(re.search(r"\d\s*(px|ms|%)", lbl) for lbl in labels)  # no motion numbers
    assert r.interaction.target_element_id in IDS


@pytest.mark.claude_live
def test_live_claude_cli(job: InterpretationInput, make_settings: Callable) -> None:
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")
    for var in ("HOME", "PATH"):  # make_settings cleared only MIMIC_*; auth comes from $HOME
        assert os.environ.get(var)
    s = make_settings(interpreter="claude_cli", claude_model="sonnet", claude_timeout_s=180)
    r = ClaudeCliInterpreter(s).interpret(job)
    print(json.dumps(r.model_dump(), indent=2))
    _assert_live_result(r)
    assert r.provider == "claude_cli"


@pytest.mark.skipif(
    os.environ.get("MIMIC_LIVE_LLM") != "1"
    or not all(os.environ.get(f"MIMIC_LLM_{v}") for v in ("BASE_URL", "API_KEY", "MODEL")),
    reason="set MIMIC_LIVE_LLM=1 and MIMIC_LLM_BASE_URL/API_KEY/MODEL to run",
)  # fmt: skip
def test_live_openai_compat(job: InterpretationInput) -> None:
    s = Settings(_env_file=None, interpreter="openai_compat")  # type: ignore[call-arg]
    check = check_interpreter(s)
    print(check.model_dump_json(indent=2))
    assert check.ok, check.error
    _assert_live_result(OpenAICompatInterpreter(s).interpret(job))
