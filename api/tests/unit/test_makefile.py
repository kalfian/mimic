"""Makefile wiring that the CLI tests can't see (PLAN-auth P3 E2E finding)."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.config import REPO_ROOT

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")


def _dry_run(target: str, **env: str) -> str:
    base = {k: v for k, v in os.environ.items() if k != "MIMIC_DATA_DIR"}
    out = subprocess.run(
        ["make", "-n", target],
        cwd=REPO_ROOT,
        env={**base, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_reset_data_uses_the_default_data_dir() -> None:
    out = _dry_run("reset-data")
    assert "rm -rf data/jobs data/mimic.db data/mimic.db-wal data/mimic.db-shm" in out


def test_reset_data_follows_mimic_data_dir(tmp_path) -> None:
    # Before the fix, reset-data always wiped <repo>/data, even when the API and the CLI
    # (clean-jobs, create-admin) used MIMIC_DATA_DIR, so the configured store survived a "wipe".
    out = _dry_run("reset-data", MIMIC_DATA_DIR=str(tmp_path))
    assert f"rm -rf {tmp_path}/jobs {tmp_path}/mimic.db" in out
    assert " data/jobs" not in out
