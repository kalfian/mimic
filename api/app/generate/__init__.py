"""Output generators (PLAN §9). ``render_all`` is FROZEN after Phase 0.

Each generator module (Track G) exposes exactly::

    def render(spec: MotionSpec) -> str

and must be pure: no I/O, no LLM, no clock, no randomness. Same spec -> byte-identical text.

Modules: ``technical``, ``llm_prompt``, ``css``, ``export_json`` (+ shared ``phrasing``).
"""

from __future__ import annotations

import importlib

from pydantic import BaseModel, ConfigDict, Field

from app.models.ir import MotionSpec

#: Generator module name per ``Outputs`` key (serialized name).
GENERATOR_MODULES: dict[str, str] = {
    "technical": "technical",
    "llm_prompt": "llm_prompt",
    "css": "css",
    "json": "export_json",
}


class Outputs(BaseModel):
    """The four text tabs. Serialized keys: ``technical``, ``llm_prompt``, ``css``, ``json``."""

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        json_schema_serialization_defaults_required=True,
    )

    technical: str
    llm_prompt: str
    css: str
    json_: str = Field(alias="json", description="Pretty IR export without transition samples.")


def render_all(spec: MotionSpec) -> Outputs:
    """Render every output tab from the IR.

    Raises ``NotImplementedError`` if a generator module is missing (all four exist since
    Track G landed).
    """
    rendered: dict[str, str] = {}
    for key, module_name in GENERATOR_MODULES.items():
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{module_name}":
                raise NotImplementedError(f"generator {module_name!r} not implemented") from exc
            raise
        rendered[key] = module.render(spec)
    return Outputs.model_validate(rendered)
