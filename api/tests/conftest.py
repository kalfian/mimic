"""Shared fixtures: isolated data dir + settings, contract fixture loader."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.config import REPO_ROOT, Settings, get_settings

CONTRACT_DIR: Path = REPO_ROOT / "docs" / "contract"
SAMPLE_RESULT_PATH: Path = CONTRACT_DIR / "sample-result.json"


def load_sample_result() -> dict[str, Any]:
    """Fresh copy of ``docs/contract/sample-result.json`` (safe to mutate)."""
    return json.loads(SAMPLE_RESULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """The shared ResultEnvelope fixture as a plain dict (fresh copy per test)."""
    return copy.deepcopy(load_sample_result())


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def settings(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings pointing at a temp data dir, interpreter disabled, no ``.env`` influence.

    Override more fields with ``monkeypatch.setenv("MIMIC_...", ...)`` then
    ``get_settings.cache_clear()`` inside the test.
    """
    monkeypatch.setenv("MIMIC_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("MIMIC_INTERPRETER", "none")
    monkeypatch.setenv("MIMIC_DEBUG", "0")
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()
