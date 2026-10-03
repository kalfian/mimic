"""Runtime settings (PLAN §4.1). Every variable uses the ``MIMIC_`` env prefix.

Settings are read from the process environment and, if present, from ``<repo>/.env``.
Use :func:`get_settings` everywhere; tests override via env vars + ``get_settings.cache_clear()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: ``<repo>`` (this file lives at ``<repo>/api/app/config.py``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

InterpreterMode = Literal["claude_cli", "openai_compat", "none"]


class Settings(BaseSettings):
    """Process-wide configuration. Immutable after load."""

    model_config = SettingsConfigDict(
        env_prefix="MIMIC_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = Field(
        default=REPO_ROOT / "data",
        description="Root for mimic.db and jobs/<id>/. Resolved to an absolute path.",
    )
    max_upload_mb: int = Field(default=200, gt=0)
    max_duration_s: float = Field(default=15.0, gt=0)
    min_duration_s: float = Field(default=0.5, ge=0)
    duration_tolerance_s: float = Field(
        default=0.5, ge=0, description="Added to max_duration_s during upload validation."
    )

    interpreter: InterpreterMode = "claude_cli"
    claude_bin: str = Field(default="claude", description="Tests point this at fake_claude.py.")
    claude_model: str = Field(default="sonnet", description="Alias passed to `claude --model`.")
    claude_timeout_s: float = Field(default=120.0, gt=0, description="Hard kill after this.")

    # ---- openai_compat interpreter (any OpenAI-compatible /v1 gateway) -----------------------
    llm_base_url: str | None = Field(
        default=None, description="e.g. http://localhost:20128/v1 (no trailing /chat/...)."
    )
    llm_api_key: SecretStr | None = Field(
        default=None, description="Bearer token. Never logged, never written to job files."
    )
    llm_model: str | None = Field(default=None, description="Model id sent to the gateway.")
    llm_timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout.")
    llm_supports_json_schema: bool | Literal["auto"] = Field(
        default="auto",
        description='"auto" tries response_format json_schema, then json_object, then prompt-only.',
    )

    workers: int = Field(default=1, ge=1, description="JobRunner executor size.")
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Comma-separated list in the env var.",
    )
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    debug: bool = Field(default=False, description="MIMIC_DEBUG=1 writes jobs/<id>/debug/ dumps.")

    @field_validator("data_dir", mode="after")
    @classmethod
    def _absolute_data_dir(cls, v: Path) -> Path:
        v = v.expanduser()
        if not v.is_absolute():
            v = REPO_ROOT / v
        return v.resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ---- derived paths / limits -------------------------------------------------------------

    @property
    def jobs_dir(self) -> Path:
        """``<data_dir>/jobs``; one sub-directory per job id."""
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        """SQLite job store file."""
        return self.data_dir / "mimic.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_duration_with_tolerance_s(self) -> float:
        return self.max_duration_s + self.duration_tolerance_s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Tests: change env, then ``get_settings.cache_clear()``."""
    return Settings()
