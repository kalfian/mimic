"""LocalJobStorage: id validation, artifact whitelist, path traversal, upload limit (Track A)."""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import pytest

from app.core.errors import ErrorCode, PipelineError
from app.core.storage import JobStorage, LocalJobStorage, is_allowed_artifact, is_valid_job_id


@pytest.fixture
def storage(tmp_path: Path) -> LocalJobStorage:
    s = LocalJobStorage(tmp_path / "jobs")
    s.init()
    return s


@pytest.fixture
def job_id() -> str:
    return uuid.uuid4().hex


def test_implements_protocol(storage: LocalJobStorage) -> None:
    assert isinstance(storage, JobStorage)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "3F2B9C1D8E7A4B6C9D0E1F2A3B4C5D6E",  # uppercase
        "3f2b9c1d-8e7a-4b6c-9d0e-1f2a3b4c5d6e",  # dashed uuid
        "../" + "a" * 29,
        "3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e/..",
        "3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e\n",
    ],
)
def test_invalid_job_ids_are_not_found(storage: LocalJobStorage, bad: str) -> None:
    assert not is_valid_job_id(bad)
    with pytest.raises(PipelineError) as exc:
        storage.job_dir(bad)
    assert exc.value.code is ErrorCode.NOT_FOUND
    assert storage.exists(bad) is False


@pytest.mark.parametrize(
    "rel",
    [
        "input.mp4",
        "input.mov",
        "input.webm",
        "input.m4v",
        "probe.json",
        "preview.mp4",
        "result.json",
        "interpretation_raw.json",
        "keyframes/state_a.png",
        "keyframes/mid_50.png",
        "keyframes/annotated_b.png",
        "keyframes/el_e12.png",
        "debug/energy.csv",
        "debug/masks/e1_change.png",
        "debug/tracks/e1.csv",
    ],
)
def test_whitelisted_artifacts(storage: LocalJobStorage, job_id: str, rel: str) -> None:
    assert is_allowed_artifact(rel)
    path = storage.artifact_path(job_id, rel)
    assert path.is_relative_to(storage.root / job_id)


@pytest.mark.parametrize(
    "rel",
    [
        "",
        "..",
        "../result.json",
        "../../etc/passwd",
        "/etc/passwd",
        "keyframes/../result.json",
        "keyframes/../../x.png",
        "keyframes/state_a.jpg",
        "keyframes/el_e0.png",
        "keyframes/secret.png",
        "keyframes",
        "keyframes/",
        "keyframes\\state_a.png",
        "debug",
        "debug/../result.json",
        "debug/.hidden",
        "debug/a/../../result.json",
        "debug/a/b/c/d/e/f.csv",
        "input.exe",
        "input.MP4",
        "result.json\x00.png",
        "Result.json",
        "./result.json",
        "result.json/",
    ],
)
def test_rejected_artifacts(storage: LocalJobStorage, job_id: str, rel: str) -> None:
    assert not is_allowed_artifact(rel)
    with pytest.raises(PipelineError) as exc:
        storage.artifact_path(job_id, rel)
    assert exc.value.code is ErrorCode.NOT_FOUND
    assert storage.exists(job_id, rel) is False


def test_symlink_escape_is_rejected(storage: LocalJobStorage, job_id: str, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state_a.png").write_bytes(b"secret")
    job_dir = storage.job_dir(job_id)
    job_dir.mkdir()
    os.symlink(outside, job_dir / "keyframes")
    with pytest.raises(PipelineError):
        storage.artifact_path(job_id, "keyframes/state_a.png")
    assert storage.exists(job_id, "keyframes/state_a.png") is False


def test_create_parents(storage: LocalJobStorage, job_id: str) -> None:
    p = storage.artifact_path(job_id, "keyframes/mid_25.png", create_parents=True)
    assert p.parent.is_dir() and not p.exists()


def test_save_upload_streams_and_limits(storage: LocalJobStorage, job_id: str) -> None:
    data = os.urandom(3 * 1024 * 1024 + 7)
    path = storage.save_upload(job_id, io.BytesIO(data), "MOV", max_bytes=len(data))
    assert path == storage.input_path(job_id, "mov")
    assert path.read_bytes() == data

    other = uuid.uuid4().hex
    with pytest.raises(PipelineError) as exc:
        storage.save_upload(other, io.BytesIO(data), "mp4", max_bytes=len(data) - 1)
    assert exc.value.code is ErrorCode.FILE_TOO_LARGE
    assert not storage.input_path(other, "mp4").exists(), "partial upload must be removed"


def test_save_upload_rejects_extension(storage: LocalJobStorage, job_id: str) -> None:
    with pytest.raises(PipelineError) as exc:
        storage.save_upload(job_id, io.BytesIO(b"x"), "avi")
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT


def test_json_roundtrip_is_atomic(storage: LocalJobStorage, job_id: str) -> None:
    payload = {"a": [1, 2.5, None], "label": "Kartu ✓"}
    path = storage.write_json(job_id, "result.json", payload)
    assert storage.read_json(job_id, "result.json") == payload
    assert path.read_text(encoding="utf-8").endswith("\n")
    storage.write_json(job_id, "result.json", {"v": 2})
    assert storage.read_json(job_id, "result.json") == {"v": 2}
    leftovers = [p.name for p in storage.job_dir(job_id).iterdir() if p.name != "result.json"]
    assert leftovers == []


def test_read_json_missing(storage: LocalJobStorage, job_id: str) -> None:
    with pytest.raises(FileNotFoundError):
        storage.read_json(job_id, "result.json")


def test_exists_and_delete(storage: LocalJobStorage, job_id: str) -> None:
    assert storage.exists(job_id) is False
    storage.write_json(job_id, "probe.json", {})
    assert storage.exists(job_id) is True
    assert storage.exists(job_id, "probe.json") is True
    assert storage.exists(job_id, "result.json") is False
    assert list(storage.job_ids()) == [job_id]
    storage.delete_job(job_id)
    assert storage.exists(job_id) is False
    storage.delete_job(job_id)  # idempotent
