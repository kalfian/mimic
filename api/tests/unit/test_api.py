"""HTTP API (Track A): upload validation codes, lifecycle with the real pipeline, artifacts,
path traversal, pipeline failures, shutdown + startup recovery, health, interpreter check.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import routes_interpreter
from app.api.routes_health import interpreter_status
from app.api.routes_jobs import clean_filename
from app.api.schemas import (
    ErrorBody,
    Health,
    JobCreated,
    JobOptions,
    JobSource,
    JobStatus,
    ResultEnvelope,
)
from app.config import Settings, get_settings
from app.core.errors import ErrorCode, PipelineError
from app.core.jobstore import SqliteJobStore
from app.core.runner import JobContext, ProgressReporter
from app.core.stages import JobState, Stage
from app.core.storage import LocalJobStorage
from app.main import create_app
from tests.conftest import SAMPLE_RESULT_PATH
from tests.unit.media_clips import build_clip_set, requires_ffmpeg

pytestmark = requires_ffmpeg

TERMINAL = {"succeeded", "failed"}


# --------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_clip_set(tmp_path_factory.mktemp("api-clips"))


def reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for k, v in env.items():
        monkeypatch.setenv(f"MIMIC_{k.upper()}", v)
    get_settings.cache_clear()
    return get_settings()


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client(settings: Settings) -> Iterator[ClientFactory]:
    """``make_client(pipeline=None, settings_=None, reinterpret=None)`` -> started TestClient
    (lifespan entered)."""
    opened: list[TestClient] = []

    def factory(
        pipeline: Any = None, settings_: Settings | None = None, reinterpret: Any = None
    ) -> TestClient:
        client = TestClient(
            create_app(settings_ or settings, pipeline=pipeline, reinterpret=reinterpret)
        )
        client.__enter__()
        opened.append(client)
        return client

    yield factory
    for c in opened:
        c.__exit__(None, None, None)


def upload(
    client: TestClient,
    path: Path,
    *,
    filename: str | None = None,
    data: dict[str, str] | None = None,
) -> Any:
    with path.open("rb") as f:
        return client.post(
            "/api/jobs",
            files={"file": (filename or path.name, f, "application/octet-stream")},
            data=data or {},
        )


def wait_terminal(client: TestClient, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in TERMINAL:
            return body
        if time.monotonic() > deadline:
            raise AssertionError(f"job did not finish: {body}")
        time.sleep(0.05)


def assert_error(resp: Any, status: int, code: ErrorCode | str) -> ErrorBody:
    assert resp.status_code == status, resp.text
    body = ErrorBody.model_validate(resp.json())
    assert body.error.code == ErrorCode(code)
    assert body.error.message
    return body


def fixture_pipeline(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
    """Fast fake pipeline: the contract fixture with this job's id, no media work."""
    reporter.enter(Stage.GENERATING)
    data = json.loads(SAMPLE_RESULT_PATH.read_text(encoding="utf-8"))
    data["job_id"] = data["spec"]["job_id"] = ctx.job_id
    data["artifacts"] = {"video_url": None, "keyframes": []}
    return ResultEnvelope.model_validate(data)


def assert_no_jobs(settings: Settings) -> None:
    jobs_dir = settings.jobs_dir
    assert not jobs_dir.exists() or list(jobs_dir.iterdir()) == [], "job dir left behind"
    store = SqliteJobStore(settings.db_path)
    assert all(store.list_by_status(s) == [] for s in JobState), "job row left behind"


# --------------------------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------------------------


def test_health_interpreter_none(make_client: ClientFactory) -> None:
    body = Health.model_validate(make_client().get("/api/health").json())
    assert body.status == "ok"
    assert body.ffmpeg is True
    assert (body.interpreter.mode, body.interpreter.available) == ("none", False)
    assert body.interpreter.model is None


def test_health_claude_cli(monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory) -> None:
    s = reload_settings(monkeypatch, interpreter="claude_cli", claude_bin=sys.executable)
    body = make_client(settings_=s).get("/api/health").json()
    assert body["interpreter"] == {"mode": "claude_cli", "available": True, "model": "sonnet"}

    s = reload_settings(monkeypatch, claude_bin="/definitely/not/claude")
    body = make_client(settings_=s).get("/api/health").json()
    assert body["interpreter"]["available"] is False


def test_health_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory) -> None:
    s = reload_settings(monkeypatch, ffprobe_bin="/nope/ffprobe")
    assert make_client(settings_=s).get("/api/health").json()["ffmpeg"] is False


def test_interpreter_status_openai_compat_never_leaks() -> None:
    secret_key = "sk-test-" + uuid.uuid4().hex
    base = SimpleNamespace(
        interpreter="openai_compat",
        claude_bin="claude",
        claude_model="sonnet",
        llm_base_url=f"https://user:{secret_key}@router.local/v1",
        llm_api_key=secret_key,
        llm_model="gpt-x",
    )
    status = interpreter_status(base)  # type: ignore[arg-type]
    assert (status.available, status.model) == (True, "gpt-x")
    assert secret_key not in status.model_dump_json()
    for missing in ("llm_base_url", "llm_api_key", "llm_model"):
        partial = SimpleNamespace(**{**vars(base), missing: "  "})
        assert interpreter_status(partial).available is False  # type: ignore[arg-type]


def test_health_openai_compat_real_settings(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    secret_key = "sk-test-" + uuid.uuid4().hex
    s = reload_settings(
        monkeypatch,
        interpreter="openai_compat",
        llm_base_url="http://localhost:20128/v1",
        llm_api_key=secret_key,
        llm_model="cc/claude-sonnet",
    )
    with caplog.at_level(logging.DEBUG):
        resp = make_client(settings_=s).get("/api/health")
    assert resp.json()["interpreter"] == {
        "mode": "openai_compat",
        "available": True,
        "model": "cc/claude-sonnet",
    }
    assert secret_key not in resp.text and secret_key not in caplog.text
    assert "localhost:20128" not in resp.text

    s = reload_settings(monkeypatch, llm_api_key="")
    body = make_client(settings_=s).get("/api/health").json()
    assert body["interpreter"]["available"] is False


# --------------------------------------------------------------------------------------------
# lifecycle with the real pipeline
# --------------------------------------------------------------------------------------------


def test_upload_poll_result_video_keyframes(
    clips: dict[str, Path], make_client: ClientFactory, settings: Settings
) -> None:
    client = make_client()
    resp = upload(client, clips["ui_mp4"], filename="Card Move.MP4", data={"pixel_ratio": "1"})
    assert resp.status_code == 202, resp.text
    created = JobCreated.model_validate(resp.json())
    job_id = created.id

    first = JobStatus.model_validate(client.get(f"/api/jobs/{job_id}").json())
    assert first.options == JobOptions(pixel_ratio="1", use_interpreter=False)
    assert first.source == JobSource(
        filename="Card Move.MP4", width=320, height=240, fps=60.0, duration_s=2.5
    )

    final = JobStatus.model_validate(wait_terminal(client, job_id, timeout=60))
    assert final.status == "succeeded", final.error
    assert (final.stage, final.stage_label, final.progress) == ("done", "Done", 1.0)
    assert final.error is None

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    assert result_resp.headers["content-type"] == "application/json"
    env = ResultEnvelope.model_validate(result_resp.json())
    assert env.job_id == env.spec.job_id == job_id
    spec = env.spec
    assert spec.source.filename == "Card Move.MP4"
    assert spec.source.pixel_ratio == 1 and spec.source.pixel_ratio_source == "user"
    assert spec.interpretation.status == "disabled"
    assert all(e.label_source == "heuristic" for e in spec.elements)
    moves = [t for t in spec.transitions if t.property == "translateY"]
    assert moves and moves[0].delta == pytest.approx(-16, abs=1)
    for text in (env.outputs.technical, env.outputs.llm_prompt, env.outputs.css):
        assert text.strip()
    assert json.loads(env.outputs.model_dump()["json"])["job_id"] == job_id

    assert env.artifacts.video_url == f"/api/jobs/{job_id}/video"
    kf = {k.name: k for k in env.artifacts.keyframes}
    assert {"state_a.png", "state_b.png", "mid_25.png", "mid_50.png", "mid_75.png",
            "annotated_a.png", "annotated_b.png"} <= set(kf)  # fmt: skip
    assert any(k.kind == "element" and k.element_id for k in kf.values())
    assert kf["state_a.png"].t_ms < 1000 < kf["state_b.png"].t_ms

    # video: full + Range (Safari seeking)
    full = client.get(env.artifacts.video_url)
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    part = client.get(env.artifacts.video_url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{len(full.content)}"
    assert part.content == full.content[:100]
    assert part.content[4:8] == b"ftyp"

    for k in env.artifacts.keyframes:
        img = client.get(k.url)
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        assert img.content.startswith(b"\x89PNG\r\n\x1a\n")

    storage = LocalJobStorage(settings.jobs_dir)
    assert storage.read_json(job_id, "probe.json")["frame_count"] == 150
    assert storage.exists(job_id, "result.json")


def test_real_pipeline_reports_unanalysable_video(
    clips: dict[str, Path], make_client: ClientFactory
) -> None:
    """``testsrc`` has no stable UI states / UI-like motion: a pipeline-time error, not a crash."""
    client = make_client()
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    status = wait_terminal(client, job_id, timeout=60)
    assert status["status"] == "failed"
    assert status["error"]["code"] in {
        "no_motion_detected",
        "unsupported_motion",
        "no_stable_state",
    }


def test_interpreter_on_with_fake_claude(
    clips: dict[str, Path],
    make_client: ClientFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = Path(__file__).resolve().parents[1] / "fixtures" / "fake_claude.py"
    wrapper = tmp_path / "claude"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    s = reload_settings(monkeypatch, interpreter="claude_cli", claude_bin=str(wrapper))
    client = make_client(settings_=s)
    job_id = upload(client, clips["ui_mp4"], data={"use_interpreter": "true"}).json()["id"]
    assert wait_terminal(client, job_id, timeout=60)["status"] == "succeeded"
    env = ResultEnvelope.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert env.spec.interpretation.provider == "claude_cli"
    assert env.spec.interpretation.status == "ok"
    assert all(e.label_source == "interpreter" for e in env.spec.elements)
    assert all(e.label.startswith("Fake ") for e in env.spec.elements)
    # numbers never come from the interpreter
    assert all(t.property for t in env.spec.transitions)
    raw = LocalJobStorage(s.jobs_dir).read_json(job_id, "interpretation_raw.json")
    assert raw


@pytest.mark.parametrize("key", ["ok_mov", "ok_webm"])
def test_other_containers_accepted(
    clips: dict[str, Path], make_client: ClientFactory, key: str
) -> None:
    client = make_client(pipeline=fixture_pipeline)
    resp = upload(client, clips[key], data={"use_interpreter": "true"})
    assert resp.status_code == 202, resp.text
    status = wait_terminal(client, resp.json()["id"])
    assert status["status"] == "succeeded"
    assert status["options"] == {"pixel_ratio": "auto", "use_interpreter": True}


def test_m4v_extension_accepted(
    clips: dict[str, Path], make_client: ClientFactory, settings: Settings
) -> None:
    client = make_client(pipeline=fixture_pipeline)
    resp = upload(client, clips["ok_mp4"], filename="clip.m4v")
    assert resp.status_code == 202, resp.text
    assert LocalJobStorage(settings.jobs_dir).exists(resp.json()["id"], "input.m4v")


# --------------------------------------------------------------------------------------------
# upload validation: every code, and no residue on failure
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "filename", "status", "code"),
    [
        ("ok_mp4", "clip.avi", 415, ErrorCode.UNSUPPORTED_FORMAT),
        ("ok_mp4", "clip", 415, ErrorCode.UNSUPPORTED_FORMAT),
        ("not_video", "clip.mp4", 415, ErrorCode.UNSUPPORTED_FORMAT),
        ("too_short", None, 422, ErrorCode.TOO_SHORT),
        ("too_long", None, 422, ErrorCode.TOO_LONG),
        ("corrupt", None, 422, ErrorCode.DECODE_FAILED),
        ("audio_only", None, 422, ErrorCode.DECODE_FAILED),
    ],
)
def test_upload_validation_codes(
    clips: dict[str, Path],
    make_client: ClientFactory,
    settings: Settings,
    key: str,
    filename: str | None,
    status: int,
    code: ErrorCode,
) -> None:
    client = make_client(pipeline=fixture_pipeline)
    assert_error(upload(client, clips[key], filename=filename), status, code)
    assert_no_jobs(settings)


def test_too_large_by_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    s = reload_settings(monkeypatch, max_upload_mb="1")
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (1024 * 1024 + 200 * 1024))

    def must_not_read(*_: object, **__: object) -> None:
        raise AssertionError("body must be rejected before it is stored")

    monkeypatch.setattr(LocalJobStorage, "save_upload", must_not_read)
    body = assert_error(upload(make_client(settings_=s), big), 413, ErrorCode.FILE_TOO_LARGE)
    assert "1 MB" in body.error.message
    assert_no_jobs(s)


def test_too_large_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """Content-Length within the multipart allowance, file bytes still over the limit."""
    s = reload_settings(monkeypatch, max_upload_mb="1")
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (1024 * 1024))
    assert_error(upload(make_client(settings_=s), big), 413, ErrorCode.FILE_TOO_LARGE)
    assert_no_jobs(s)


@pytest.mark.parametrize(
    "data",
    [
        {"pixel_ratio": "4"},
        {"pixel_ratio": "2x"},
        {"use_interpreter": "maybe"},
    ],
)
def test_invalid_form_fields(
    clips: dict[str, Path], make_client: ClientFactory, settings: Settings, data: dict[str, str]
) -> None:
    client = make_client(pipeline=fixture_pipeline)
    assert_error(upload(client, clips["ok_mp4"], data=data), 422, ErrorCode.INVALID_REQUEST)
    assert_no_jobs(settings)


def test_missing_file_and_wrong_content_type(make_client: ClientFactory) -> None:
    client = make_client(pipeline=fixture_pipeline)
    assert_error(
        client.post("/api/jobs", data={"pixel_ratio": "1"}), 422, ErrorCode.INVALID_REQUEST
    )
    assert_error(client.post("/api/jobs", json={"file": "x"}), 422, ErrorCode.INVALID_REQUEST)
    assert_error(
        client.post(
            "/api/jobs",
            content=b"--x\r\nnot a valid part",
            headers={"content-type": "multipart/form-data; boundary=x"},
        ),
        422,
        ErrorCode.INVALID_REQUEST,
    )


def test_form_values_are_normalized(clips: dict[str, Path], make_client: ClientFactory) -> None:
    client = make_client(pipeline=fixture_pipeline)
    resp = upload(client, clips["ok_mp4"], data={"pixel_ratio": " AUTO ", "use_interpreter": "0"})
    assert resp.status_code == 202, resp.text
    assert client.get(f"/api/jobs/{resp.json()['id']}").json()["options"] == {
        "pixel_ratio": "auto",
        "use_interpreter": False,
    }


def test_filename_is_sanitized(clips: dict[str, Path], make_client: ClientFactory) -> None:
    client = make_client(pipeline=fixture_pipeline)
    resp = upload(client, clips["ok_mp4"], filename="../../etc/my clip.mp4")
    assert resp.status_code == 202, resp.text
    status = client.get(f"/api/jobs/{resp.json()['id']}").json()
    assert status["source"]["filename"] == "my clip.mp4"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("clip.mp4", "clip.mp4"),
        ("C:\\Users\\me\\Desktop\\rec.mov", "rec.mov"),
        ("/tmp/../x/evil\x07\x00name.webm", "evilname.webm"),
        ("  spaced.mp4  ", "spaced.mp4"),
        ("", "upload"),
        (None, "upload"),
        ("a" * 300 + ".mp4", "a" * 196 + ".mp4"),
    ],
)
def test_clean_filename(raw: str | None, expected: str) -> None:
    assert clean_filename(raw) == expected


# --------------------------------------------------------------------------------------------
# routing errors, result readiness, traversal
# --------------------------------------------------------------------------------------------


def test_unknown_and_malformed_job_ids(make_client: ClientFactory) -> None:
    client = make_client(pipeline=fixture_pipeline)
    unknown = uuid.uuid4().hex
    for path in ("", "/result", "/video", "/keyframes/mid_50.png"):
        assert_error(client.get(f"/api/jobs/{unknown}{path}"), 404, ErrorCode.NOT_FOUND)
    for bad in ("nope", unknown.upper(), unknown[:-1] + "g", "..%2F..%2Fetc"):
        assert_error(client.get(f"/api/jobs/{bad}"), 404, ErrorCode.NOT_FOUND)
    assert_error(client.get("/api/nothing-here"), 404, ErrorCode.NOT_FOUND)
    assert_error(client.delete(f"/api/jobs/{unknown}"), 405, ErrorCode.INVALID_REQUEST)


def test_result_not_ready_then_ready(clips: dict[str, Path], make_client: ClientFactory) -> None:
    release = threading.Event()

    def blocking(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.MEASURING, 0.5)
        assert release.wait(10)
        return fixture_pipeline(ctx, reporter)

    client = make_client(pipeline=blocking)
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    deadline = time.monotonic() + 5
    while client.get(f"/api/jobs/{job_id}").json()["stage"] != "measuring":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "processing"
    assert status["stage_label"] == "Measuring motion"
    assert status["progress"] == pytest.approx(0.62)
    assert_error(client.get(f"/api/jobs/{job_id}/result"), 409, ErrorCode.NOT_READY)
    release.set()
    assert wait_terminal(client, job_id)["status"] == "succeeded"
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200


def test_keyframe_whitelist_and_missing_artifacts(
    clips: dict[str, Path], make_client: ClientFactory, settings: Settings
) -> None:
    client = make_client(pipeline=fixture_pipeline)
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    assert wait_terminal(client, job_id)["status"] == "succeeded"
    # a file that exists in the job dir but is not a keyframe must never be served
    (settings.jobs_dir / job_id / "keyframes").mkdir(parents=True, exist_ok=True)
    (settings.jobs_dir / job_id / "keyframes" / "secret.png").write_bytes(b"x")
    for name in (
        "secret.png",
        "state_a.jpg",
        "el_e0.png",
        "..%2Fresult.json",
        "%2E%2E%2Finput.mp4",
        "mid_50.png%00",
    ):
        assert_error(client.get(f"/api/jobs/{job_id}/keyframes/{name}"), 404, "not_found")
    # whitelisted name, but not produced by this (fake) pipeline
    assert_error(client.get(f"/api/jobs/{job_id}/keyframes/state_a.png"), 404, "not_found")
    # fake pipeline made no preview
    assert_error(client.get(f"/api/jobs/{job_id}/video"), 404, "not_found")


# --------------------------------------------------------------------------------------------
# pipeline failures
# --------------------------------------------------------------------------------------------


def test_pipeline_error_is_reported_with_stage(
    clips: dict[str, Path], make_client: ClientFactory
) -> None:
    def failing(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.SCANNING, 0.5)
        raise PipelineError(ErrorCode.NO_MOTION_DETECTED)

    client = make_client(pipeline=failing)
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    status = JobStatus.model_validate(wait_terminal(client, job_id))
    assert status.status == "failed" and status.stage == "scanning"
    assert status.error is not None and status.error.code == "no_motion_detected"
    assert_error(client.get(f"/api/jobs/{job_id}/result"), 409, ErrorCode.NOT_READY)


def test_unexpected_exception_is_internal_error(
    clips: dict[str, Path], make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    def crashing(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.FITTING)
        raise ZeroDivisionError("boom")

    client = make_client(pipeline=crashing)
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    with caplog.at_level(logging.ERROR):
        status = wait_terminal(client, job_id)
    assert status["status"] == "failed" and status["stage"] == "fitting"
    assert status["error"]["code"] == "internal_error"
    assert "boom" not in status["error"]["message"]


def test_missing_preview_is_a_warning(
    clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    from app.pipeline import run as run_mod

    monkeypatch.setattr(run_mod, "make_preview", lambda *a, **k: False)
    client = make_client()
    job_id = upload(client, clips["ui_mp4"]).json()["id"]
    assert wait_terminal(client, job_id, timeout=60)["status"] == "succeeded"
    env = ResultEnvelope.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert env.artifacts.video_url is None
    assert "preview_unavailable" in {w.code for w in env.spec.warnings}


# --------------------------------------------------------------------------------------------
# shutdown + startup recovery
# --------------------------------------------------------------------------------------------


def test_shutdown_interrupts_running_job(clips: dict[str, Path], settings: Settings) -> None:
    started = threading.Event()

    def endless(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.MEASURING)
        started.set()
        while True:
            reporter.sleep(0.05)

    client = TestClient(create_app(settings, pipeline=endless))
    with client:
        job_id = upload(client, clips["ok_mp4"]).json()["id"]
        assert started.wait(5)
    rec = SqliteJobStore(settings.db_path).get(job_id)
    assert rec is not None
    assert (rec.status, rec.error_code, rec.stage) == (
        JobState.FAILED,
        ErrorCode.INTERRUPTED,
        Stage.MEASURING,
    )


def test_startup_recovery(
    clips: dict[str, Path], settings: Settings, make_client: ClientFactory
) -> None:
    storage = LocalJobStorage(settings.jobs_dir)
    storage.init()
    store = SqliteJobStore(settings.db_path)
    store.init()
    opts = JobOptions()

    def seed(with_input: bool) -> str:
        job_id = uuid.uuid4().hex
        store.create(job_id, original_filename="clip.mp4", ext="mp4", options=opts)
        if with_input:
            dst = storage.artifact_path(job_id, "input.mp4", create_parents=True)
            shutil.copyfile(clips["ok_mp4"], dst)
        return job_id

    was_processing = seed(with_input=True)
    store.update_progress(was_processing, Stage.MEASURING, 0.6)
    queued_ok = seed(with_input=True)
    queued_lost = seed(with_input=False)
    done = seed(with_input=True)
    store.mark_succeeded(done)

    client = make_client(pipeline=fixture_pipeline)

    s1 = client.get(f"/api/jobs/{was_processing}").json()
    assert (s1["status"], s1["stage"], s1["error"]["code"]) == (
        "failed",
        "measuring",
        "interrupted",
    )
    assert wait_terminal(client, queued_ok)["status"] == "succeeded"
    assert client.get(f"/api/jobs/{queued_ok}/result").status_code == 200
    s3 = client.get(f"/api/jobs/{queued_lost}").json()
    assert (s3["status"], s3["error"]["code"]) == ("failed", "interrupted")
    assert client.get(f"/api/jobs/{done}").json()["status"] == "succeeded"


def test_mimic_debug_writes_dumps(
    clips: dict[str, Path], make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = reload_settings(monkeypatch, debug="1")
    client = make_client(settings_=s)
    job_id = upload(client, clips["ui_mp4"]).json()["id"]
    assert wait_terminal(client, job_id, timeout=60)["status"] == "succeeded"
    debug = s.jobs_dir / job_id / "debug"
    for rel in ("energy.csv", "segments.json", "elements.json", "summary.json",
                "masks/state_a.png", "masks/state_b.png", "masks/change_mask.png"):  # fmt: skip
        assert (debug / rel).is_file(), rel
    assert list((debug / "tracks").glob("e1_fwd_*.csv"))
    summary = json.loads((debug / "summary.json").read_text())
    assert {"elements", "series", "features", "heuristic", "target"} <= set(summary)


# --------------------------------------------------------------------------------------------
# re-run interpretation (Phase 5)
# --------------------------------------------------------------------------------------------


def _fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    fake = Path(__file__).resolve().parents[1] / "fixtures" / "fake_claude.py"
    wrapper = tmp_path / "claude"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    return reload_settings(monkeypatch, interpreter="claude_cli", claude_bin=str(wrapper))


def rerun(client: TestClient, job_id: str, body: Any = None) -> Any:
    return client.post(
        f"/api/jobs/{job_id}/interpret",
        json={"use_interpreter": True} if body is None else body,
    )


def test_rerun_interpretation_reuses_measurement(
    clips: dict[str, Path],
    make_client: ClientFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.pipeline import run as run_mod

    s = _fake_claude(tmp_path, monkeypatch)
    client = make_client(settings_=s)
    job_id = upload(client, clips["ui_mp4"]).json()["id"]  # AI off at upload (default)
    assert wait_terminal(client, job_id, timeout=60)["status"] == "succeeded"
    first = ResultEnvelope.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert first.spec.interpretation.status == "disabled"
    storage = LocalJobStorage(s.jobs_dir)
    assert storage.exists(job_id, "measurement.json")

    def no_measuring(*a: Any, **k: Any) -> None:
        raise AssertionError("re-run must not measure again")

    monkeypatch.setattr(run_mod, "measure", no_measuring)
    resp = rerun(client, job_id)
    assert resp.status_code == 202, resp.text
    queued = JobStatus.model_validate(resp.json())
    assert queued.status in ("queued", "processing")
    assert queued.options.use_interpreter is True
    final = JobStatus.model_validate(wait_terminal(client, job_id, timeout=60))
    assert (final.status, final.stage, final.error) == ("succeeded", "done", None)
    assert final.options == JobOptions(pixel_ratio="auto", use_interpreter=True)

    again = ResultEnvelope.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert again.spec.interpretation.provider == "claude_cli"
    assert again.spec.interpretation.status == "ok"
    assert all(e.label.startswith("Fake ") for e in again.spec.elements)
    # numbers and artifacts are the stored ones
    dump = lambda e: [t.model_dump(exclude={"id"}) for t in e.spec.transitions]  # noqa: E731
    assert dump(again) == dump(first)
    assert again.artifacts == first.artifacts
    assert again.outputs.llm_prompt != first.outputs.llm_prompt  # new labels in the text

    # and back to heuristic labels (explicit false)
    assert rerun(client, job_id, {"use_interpreter": False}).status_code == 202
    assert wait_terminal(client, job_id, timeout=60)["options"]["use_interpreter"] is False
    back = ResultEnvelope.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert back.spec.interpretation.status == "disabled"


def _pipeline_with_measurement(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
    env = fixture_pipeline(ctx, reporter)
    ctx.storage.write_json(ctx.job_id, "measurement.json", {"format": -1})
    return env


def test_rerun_conflicts_and_validation(clips: dict[str, Path], make_client: ClientFactory) -> None:
    release = threading.Event()

    def blocking_rerun(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.INTERPRETING)
        assert release.wait(10)
        return fixture_pipeline(ctx, reporter)

    def failing(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        raise PipelineError(ErrorCode.NO_MOTION_DETECTED)

    client = make_client(pipeline=_pipeline_with_measurement, reinterpret=blocking_rerun)
    assert_error(rerun(client, uuid.uuid4().hex), 404, ErrorCode.NOT_FOUND)
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    assert wait_terminal(client, job_id)["status"] == "succeeded"

    for body in ({}, {"use_interpreter": "maybe"}, {"use_interpreter": True, "x": 1}):
        assert_error(rerun(client, job_id, body), 422, ErrorCode.INVALID_REQUEST)
    no_body = client.post(f"/api/jobs/{job_id}/interpret")
    assert_error(no_body, 422, ErrorCode.INVALID_REQUEST)

    assert rerun(client, job_id).status_code == 202
    busy = assert_error(rerun(client, job_id), 409, ErrorCode.ALREADY_RUNNING)  # double click
    assert "still being processed" in busy.error.message
    assert_error(client.get(f"/api/jobs/{job_id}/result"), 409, ErrorCode.NOT_READY)
    release.set()
    assert wait_terminal(client, job_id)["status"] == "succeeded"

    # failed job, and a result without measurement.json (made before Phase 5)
    failed = make_client(pipeline=failing)
    fid = upload(failed, clips["ok_mp4"]).json()["id"]
    assert wait_terminal(failed, fid)["status"] == "failed"
    assert_error(rerun(failed, fid), 409, ErrorCode.NOT_READY)
    old = make_client(pipeline=fixture_pipeline)
    oid = upload(old, clips["ok_mp4"]).json()["id"]
    assert wait_terminal(old, oid)["status"] == "succeeded"
    body = assert_error(rerun(old, oid), 409, ErrorCode.NOT_READY)
    assert "Upload the video again" in body.error.message


@pytest.mark.parametrize("mode", ["crash", "bad_measurement"])
def test_failed_rerun_keeps_previous_result(
    clips: dict[str, Path], make_client: ClientFactory, mode: str
) -> None:
    def crashing(ctx: JobContext, reporter: ProgressReporter) -> ResultEnvelope:
        reporter.enter(Stage.INTERPRETING)
        raise ZeroDivisionError("secret detail")

    # bad_measurement: the real reinterpret() on a measurement.json of another format
    client = make_client(
        pipeline=_pipeline_with_measurement, reinterpret=crashing if mode == "crash" else None
    )
    job_id = upload(client, clips["ok_mp4"]).json()["id"]
    assert wait_terminal(client, job_id)["status"] == "succeeded"
    before = client.get(f"/api/jobs/{job_id}/result").json()

    assert rerun(client, job_id).status_code == 202
    status = JobStatus.model_validate(wait_terminal(client, job_id))
    assert (status.status, status.stage) == ("succeeded", "done")
    assert status.options.use_interpreter is False  # restored: the result was made without AI
    assert status.error is not None
    if mode == "crash":
        assert status.error.code == "internal_error"
        assert "secret" not in status.error.message
    else:
        assert status.error.code == "not_ready"
        assert "Upload the video again" in status.error.message
    assert client.get(f"/api/jobs/{job_id}/result").json() == before


def test_recovery_restores_interrupted_rerun(
    clips: dict[str, Path], settings: Settings, make_client: ClientFactory
) -> None:
    storage = LocalJobStorage(settings.jobs_dir)
    storage.init()
    store = SqliteJobStore(settings.db_path)
    store.init()
    data = json.loads(SAMPLE_RESULT_PATH.read_text(encoding="utf-8"))

    def seed(state: JobState) -> str:
        job_id = uuid.uuid4().hex
        store.create(job_id, original_filename="clip.mp4", ext="mp4", options=JobOptions())
        data["job_id"] = data["spec"]["job_id"] = job_id
        storage.write_json(job_id, "result.json", data)
        store.mark_succeeded(job_id)
        assert store.claim_rerun(job_id, JobOptions(use_interpreter=False))
        if state is JobState.PROCESSING:
            store.update_progress(job_id, Stage.INTERPRETING, 0.8)
        return job_id

    running, pending = seed(JobState.PROCESSING), seed(JobState.QUEUED)
    client = make_client(pipeline=fixture_pipeline)
    for job_id in (running, pending):
        st = client.get(f"/api/jobs/{job_id}").json()
        assert (st["status"], st["stage"], st["error"]) == ("succeeded", "done", None)
        # the fixture result was produced by the interpreter (status ok) -> option restored
        assert st["options"]["use_interpreter"] is (data["spec"]["interpretation"]["status"]
                                                    != "disabled")  # fmt: skip
        assert client.get(f"/api/jobs/{job_id}/result").status_code == 200


# --------------------------------------------------------------------------------------------
# CORS + interpreter check
# --------------------------------------------------------------------------------------------


def test_cors_preflight(make_client: ClientFactory) -> None:
    resp = make_client(pipeline=fixture_pipeline).options(
        "/api/jobs",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_interpreter_check_returns_result(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory, settings: Settings
) -> None:
    from pydantic import BaseModel

    class FakeCheck(BaseModel):
        mode: str
        ok: bool
        latency_ms: int

    seen: list[Settings] = []

    def fake_check(s: Settings) -> FakeCheck:
        seen.append(s)
        assert "AnyIO worker" in threading.current_thread().name  # off the event loop
        return FakeCheck(mode="none", ok=False, latency_ms=0)

    monkeypatch.setattr(routes_interpreter, "check_interpreter", fake_check)
    resp = make_client(pipeline=fixture_pipeline).post("/api/interpreter/check")
    assert resp.status_code == 200
    assert resp.json() == {"mode": "none", "ok": False, "latency_ms": 0}
    assert seen == [settings]


def test_interpreter_check_never_leaks_exception_text(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sk-live-" + uuid.uuid4().hex

    def leaky(_: Settings) -> Any:
        raise ConnectionError(f"failed to reach https://user:{secret}@router.local/v1")

    monkeypatch.setattr(routes_interpreter, "check_interpreter", leaky)
    with caplog.at_level(logging.DEBUG):
        resp = make_client(pipeline=fixture_pipeline).post("/api/interpreter/check")
    assert_error(resp, 500, ErrorCode.INTERNAL_ERROR)
    assert secret not in resp.text
    assert secret not in caplog.text
    assert "ConnectionError" in caplog.text


def test_interpreter_check_real_module_mode_none(make_client: ClientFactory) -> None:
    """End-to-end with Track I's real ``check_interpreter`` (no network for mode ``none``)."""
    from app.api.schemas import InterpreterCheck

    resp = make_client(pipeline=fixture_pipeline).post("/api/interpreter/check")
    assert resp.status_code == 200, resp.text
    check = InterpreterCheck.model_validate(resp.json())
    assert (check.mode, check.ok) == ("none", False)
    assert check.error is not None and check.error.code == "not_configured"
