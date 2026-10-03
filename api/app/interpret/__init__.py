"""Layer B — interpretation (PLAN §7). Labels only; never motion numbers (D4).

``get_interpreter(settings, use_interpreter)`` picks the implementation for one job:

* ``use_interpreter=False`` (the default, PLAN P9) or ``MIMIC_INTERPRETER=none``
  -> :class:`FallbackInterpreter` with status ``disabled``. Nothing leaves the machine.
* ``claude_cli`` and the binary is on PATH -> :class:`ClaudeCliInterpreter`.
* ``openai_compat`` and base URL + key + model are configured -> :class:`OpenAICompatInterpreter`.
* Opted in but the backend is unavailable -> :class:`FallbackInterpreter` with status ``fallback``.

Every interpreter returns a complete :class:`InterpretationResult` and never raises for
backend failures (timeouts / errors come back as the fallback result with that status).
"""

from __future__ import annotations

import shutil

from app.config import Settings
from app.interpret.base import (
    InterpretationInput,
    InterpretationResult,
    Interpreter,
)
from app.interpret.claude_cli import ClaudeCliInterpreter, ProgressCallback
from app.interpret.fallback import FallbackInterpreter, build_fallback
from app.interpret.openai_compat import OpenAICompatInterpreter, missing_settings


def get_interpreter(
    settings: Settings,
    use_interpreter: bool,
    *,
    on_progress: ProgressCallback | None = None,
) -> Interpreter:
    """Interpreter for one job. ``on_progress(fraction_of_timeout)`` ticks while waiting."""
    if not use_interpreter or settings.interpreter == "none":
        return FallbackInterpreter(status="disabled")
    if settings.interpreter == "claude_cli":
        if shutil.which(settings.claude_bin) is None:
            return FallbackInterpreter(status="fallback", reason="claude CLI not found on PATH")
        return ClaudeCliInterpreter(settings, on_progress=on_progress)
    if settings.interpreter == "openai_compat":
        missing = missing_settings(settings)
        if missing:
            return FallbackInterpreter(
                status="fallback", reason=f"LLM gateway not configured ({', '.join(missing)})"
            )
        return OpenAICompatInterpreter(settings, on_progress=on_progress)
    return FallbackInterpreter(status="disabled")


__all__ = [
    "ClaudeCliInterpreter",
    "FallbackInterpreter",
    "InterpretationInput",
    "InterpretationResult",
    "Interpreter",
    "OpenAICompatInterpreter",
    "build_fallback",
    "get_interpreter",
]
