"""Shared fixtures: isolated data dir + settings, contract fixture loader, fast scrypt."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.auth import policy as auth_policy
from app.config import REPO_ROOT, Settings, get_settings

CONTRACT_DIR: Path = REPO_ROOT / "docs" / "contract"
SAMPLE_RESULT_PATH: Path = CONTRACT_DIR / "sample-result.json"
SAMPLE_CONTINUOUS_RESULT_PATH: Path = CONTRACT_DIR / "sample-continuous-result.json"


#: scrypt cost used by tests (PLAN-auth §9): 2^10 instead of 2^15. Hashes encode their own
#: parameters, so verification is unaffected.
TEST_SCRYPT_LOG2_N = 10


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_scrypt: use the production scrypt cost (one test only; slow)"
    )


@pytest.fixture(autouse=True)
def _fast_scrypt(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower the scrypt cost for every test unless it is marked ``real_scrypt``."""
    if request.node.get_closest_marker("real_scrypt") is None:
        monkeypatch.setattr(
            auth_policy, "SCRYPT_PARAMS", auth_policy.ScryptParams(log2_n=TEST_SCRYPT_LOG2_N)
        )


def load_sample_result() -> dict[str, Any]:
    """Fresh copy of ``docs/contract/sample-result.json`` (safe to mutate)."""
    return json.loads(SAMPLE_RESULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """The shared ResultEnvelope fixture as a plain dict (fresh copy per test)."""
    return copy.deepcopy(load_sample_result())


def load_sample_continuous_result() -> dict[str, Any]:
    """Fresh copy of ``docs/contract/sample-continuous-result.json`` (continuous mode)."""
    return json.loads(SAMPLE_CONTINUOUS_RESULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sample_continuous_result() -> dict[str, Any]:
    """The continuous-mode ResultEnvelope fixture (PLAN-continuous §5.3), fresh copy per test."""
    return load_sample_continuous_result()


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
