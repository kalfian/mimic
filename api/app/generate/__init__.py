"""Output generators (PLAN §9, PLAN-continuous §6). ``render_all`` is FROZEN after Phase 0.

Each generator module exposes exactly::

    def render(spec: MotionSpec) -> str

and must be pure: no I/O, no LLM, no clock, no randomness. Same spec -> byte-identical text.

``render_all`` dispatches on ``spec.mode``:

* ``transition``: :data:`GENERATOR_MODULES` (``technical``, ``llm_prompt``, ``css``,
  ``export_json`` + shared ``phrasing``). ``Outputs.js`` stays null (omitted from the JSON).
* ``continuous``: :data:`CONTINUOUS_GENERATOR_MODULES` (``continuous.*`` from Track G, plus the
  shared ``export_json``) and the extra ``js`` driver output.
"""

from __future__ import annotations

import importlib

from pydantic import BaseModel, ConfigDict, Field

from app.models.ir import MotionSpec

#: Generator module name per ``Outputs`` key (serialized name), transition mode.
GENERATOR_MODULES: dict[str, str] = {
    "technical": "technical",
    "llm_prompt": "llm_prompt",
    "css": "css",
    "json": "export_json",
}

#: Continuous mode (PLAN-continuous §6). The JSON export is mode-agnostic.
CONTINUOUS_GENERATOR_MODULES: dict[str, str] = {
    "technical": "continuous.technical",
    "llm_prompt": "continuous.llm_prompt",
    "css": "continuous.css",
    "json": "export_json",
    "js": "continuous.js",
}


def _omit_if_none(v: object) -> bool:
    return v is None


class Outputs(BaseModel):
    """The text tabs. Serialized keys: ``technical``, ``llm_prompt``, ``css``, ``json`` and, in
    continuous mode only, ``js`` (omitted when null, so transition outputs are unchanged)."""

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
    json_: str = Field(
        alias="json", description="Pretty IR export without transition / continuous samples."
    )
    js: str | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Suggested JS driver (continuous mode only; absent in transition mode).",
    )


def generator_modules(spec: MotionSpec) -> dict[str, str]:
    """``Outputs`` key -> module (relative to this package) for the spec's mode."""
    return CONTINUOUS_GENERATOR_MODULES if spec.mode == "continuous" else GENERATOR_MODULES


def render_all(spec: MotionSpec) -> Outputs:
    """Render every output tab from the IR.

    Raises ``NotImplementedError`` if a generator module is missing or still a stub (the
    continuous generators until Track G lands).
    """
    rendered: dict[str, str] = {}
    for key, module_name in generator_modules(spec).items():
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{module_name}":
                raise NotImplementedError(f"generator {module_name!r} not implemented") from exc
            raise
        rendered[key] = module.render(spec)
    return Outputs.model_validate(rendered)
