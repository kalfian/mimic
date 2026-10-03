"""Job file storage (PLAN §4.2): ``<data_dir>/jobs/<job_id>/...`` on local disk.

Every path handed out by :class:`LocalJobStorage` is derived from a validated job id and a
whitelisted relative artifact name, and is checked to resolve inside the job directory. Route
handlers and pipeline stages must go through :meth:`JobStorage.artifact_path` instead of
joining paths themselves.

Layout::

    jobs/<job_id>/
      input.<mp4|mov|webm|m4v>
      probe.json  preview.mp4  interpretation_raw.json  result.json
      measurement.json                  (Layer A output, for re-running interpretation)
      keyframes/<KEYFRAME_NAME_RE>
      debug/<safe relative path>        (MIMIC_DEBUG=1)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

from app.api.schemas import ALLOWED_EXTENSIONS, JOB_ID_RE, KEYFRAME_NAME_RE
from app.core.errors import ErrorCode, PipelineError

#: Copy buffer for uploads.
CHUNK_BYTES = 1024 * 1024

#: Layer A output kept for re-running interpretation (``app.pipeline.persist``).
MEASUREMENT_FILE = "measurement.json"
#: Fixed top-level artifact names (besides ``input.<ext>``).
_FIXED_ARTIFACTS: frozenset[str] = frozenset(
    {"probe.json", "preview.mp4", "result.json", "interpretation_raw.json", MEASUREMENT_FILE}
)
_INPUT_RE = re.compile(r"^input\.(?:" + "|".join(sorted(ALLOWED_EXTENSIONS)) + r")$")
#: One path segment below ``debug/``: no leading dot (blocks ``.``/``..``/hidden files).
_DEBUG_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_DEBUG_MAX_DEPTH = 4


def is_valid_job_id(job_id: str) -> bool:
    """``uuid4().hex`` format (``JOB_ID_RE``)."""
    return isinstance(job_id, str) and JOB_ID_RE.fullmatch(job_id) is not None


def is_allowed_artifact(rel: str) -> bool:
    """Whether ``rel`` (POSIX-style, relative to the job dir) is a whitelisted artifact name."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        return False
    if rel in _FIXED_ARTIFACTS or _INPUT_RE.fullmatch(rel):
        return True
    parts = rel.split("/")
    if len(parts) == 2 and parts[0] == "keyframes":
        return KEYFRAME_NAME_RE.fullmatch(parts[1]) is not None
    if parts[0] == "debug" and 2 <= len(parts) <= _DEBUG_MAX_DEPTH + 1:
        return all(_DEBUG_SEGMENT_RE.fullmatch(p) for p in parts[1:])
    return False


@runtime_checkable
class JobStorage(Protocol):
    """Per-job file storage. Swappable later (D1); keep it to these methods."""

    def job_dir(self, job_id: str) -> Path:
        """Directory of a job (not created). Raises ``not_found`` for an invalid id."""
        ...

    def save_upload(
        self, job_id: str, stream: BinaryIO, ext: str, *, max_bytes: int | None = None
    ) -> Path:
        """Stream an upload to ``input.<ext>``; ``file_too_large`` once ``max_bytes`` is passed."""
        ...

    def artifact_path(self, job_id: str, rel: str, *, create_parents: bool = False) -> Path:
        """Absolute path of a whitelisted artifact inside the job dir (may not exist yet)."""
        ...

    def input_path(self, job_id: str, ext: str) -> Path:
        """Path of the uploaded video."""
        ...

    def write_json(self, job_id: str, rel: str, data: Any) -> Path:
        """Atomically write JSON (UTF-8, 2-space indent)."""
        ...

    def read_json(self, job_id: str, rel: str) -> Any:
        """Parse a JSON artifact. ``FileNotFoundError`` if missing."""
        ...

    def exists(self, job_id: str, rel: str | None = None) -> bool:
        """Job dir (``rel=None``) or artifact exists. False for invalid ids/names."""
        ...

    def delete_job(self, job_id: str) -> None:
        """Remove the job dir (validation failure cleanup). Missing dir is not an error."""
        ...


class LocalJobStorage:
    """:class:`JobStorage` on the local filesystem under ``jobs_dir``."""

    def __init__(self, jobs_dir: Path) -> None:
        self.root = Path(jobs_dir).resolve()

    def init(self) -> None:
        """Create the jobs root (idempotent)."""
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths ------------------------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        if not is_valid_job_id(job_id):
            raise PipelineError(ErrorCode.NOT_FOUND)
        return self.root / job_id

    def artifact_path(self, job_id: str, rel: str, *, create_parents: bool = False) -> Path:
        base = self.job_dir(job_id)
        if not is_allowed_artifact(rel):
            raise PipelineError(ErrorCode.NOT_FOUND, f"Unknown artifact {rel!r}.")
        path = base.joinpath(*rel.split("/"))
        # Defence in depth: symlinks inside the job dir must not lead outside it.
        if not path.resolve().is_relative_to(base.resolve()):
            raise PipelineError(ErrorCode.NOT_FOUND, f"Unknown artifact {rel!r}.")
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def input_path(self, job_id: str, ext: str) -> Path:
        return self.artifact_path(job_id, f"input.{ext.lower()}")

    # ---- writes -----------------------------------------------------------------------------

    def save_upload(
        self, job_id: str, stream: BinaryIO, ext: str, *, max_bytes: int | None = None
    ) -> Path:
        ext = ext.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT)
        dest = self.artifact_path(job_id, f"input.{ext}", create_parents=True)
        written = 0
        try:
            with dest.open("wb") as out:
                while chunk := stream.read(CHUNK_BYTES):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise PipelineError(
                            ErrorCode.FILE_TOO_LARGE,
                            f"The file is larger than {max_bytes // (1024 * 1024)} MB. "
                            "Trim it or export it at a lower resolution.",
                        )
                    out.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return dest

    def write_json(self, job_id: str, rel: str, data: Any) -> Path:
        dest = self.artifact_path(job_id, rel, create_parents=True)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        _atomic_write_text(dest, text)
        return dest

    def delete_job(self, job_id: str) -> None:
        d = self.job_dir(job_id)
        if d.is_symlink():
            d.unlink()
        elif d.exists():
            shutil.rmtree(d)

    # ---- reads ------------------------------------------------------------------------------

    def read_json(self, job_id: str, rel: str) -> Any:
        return json.loads(self.artifact_path(job_id, rel).read_text(encoding="utf-8"))

    def exists(self, job_id: str, rel: str | None = None) -> bool:
        try:
            path = self.job_dir(job_id) if rel is None else self.artifact_path(job_id, rel)
        except PipelineError:
            return False
        return path.is_dir() if rel is None else path.is_file()

    def job_ids(self) -> Iterable[str]:
        """Job ids that have a directory (diagnostics/tests)."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if is_valid_job_id(p.name))


def _atomic_write_text(dest: Path, text: str) -> None:
    """Write via a temp file in the same dir + ``os.replace`` so readers never see partial JSON."""
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
