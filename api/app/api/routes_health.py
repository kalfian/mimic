"""``GET /api/health`` (PLAN §5). Must stay cheap: no subprocess and no network calls."""

from __future__ import annotations

import shutil
from typing import Any

from fastapi import APIRouter

from app import __version__
from app.api.deps import ServicesDep
from app.api.schemas import Health, HealthInterpreter, HealthLimits
from app.config import Settings

router = APIRouter(tags=["health"])


def _configured(value: Any) -> bool:
    """Non-empty setting; unwraps ``SecretStr`` locally (the value never leaves this function)."""
    if value is None:
        return False
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return bool(str(value).strip())


def interpreter_status(settings: Settings) -> HealthInterpreter:
    """Configured interpreter and whether it *could* run, from local facts only.

    * ``claude_cli``: the binary is on PATH (``claude`` itself is never invoked here).
    * ``openai_compat``: base URL, API key and model are all configured (no request is made;
      use ``POST /api/interpreter/check`` for a live check). Neither value is ever returned.
    * ``none``: unavailable.
    """
    mode = settings.interpreter
    if mode == "claude_cli":
        available = shutil.which(settings.claude_bin) is not None
        model: str | None = settings.claude_model
    elif mode == "openai_compat":
        available = all(
            _configured(getattr(settings, name, None))
            for name in ("llm_base_url", "llm_api_key", "llm_model")
        )
        model = getattr(settings, "llm_model", None) or None
    else:
        available, model = False, None
    return HealthInterpreter(mode=mode, available=available, model=model)


@router.get("/api/health", response_model=Health)
def health(services: ServicesDep) -> Health:
    s = services.settings
    ffmpeg_ok = shutil.which(s.ffmpeg_bin) is not None and shutil.which(s.ffprobe_bin) is not None
    limits = HealthLimits(
        max_upload_mb=s.max_upload_mb,
        max_duration_s=s.max_duration_s,
        min_duration_s=s.min_duration_s,
        duration_tolerance_s=s.duration_tolerance_s,
    )
    return Health(
        version=__version__, ffmpeg=ffmpeg_ok, interpreter=interpreter_status(s), limits=limits
    )
