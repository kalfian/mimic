"""Per-job JSON schema for ``claude --json-schema`` + the lenient payload model (PLAN §7.4).

The schema is rebuilt for every job so element ids are a closed ``enum`` (the model cannot
invent ids). It has **no numeric fields except ``confidence`` / ``type_confidence``** (D4):
the interpreter labels things, it never reports motion numbers.

``InterpretationPayload`` is deliberately lenient (unknown keys ignored, loose scalars coerced
to strings): the CLI already validates against the schema, and anything that still slips
through is repaired by :func:`app.interpret.claude_cli.sanitize_payload` (unknown ids dropped,
strings clipped, unknown roles -> ``other``) instead of failing the whole interpretation.
"""

from __future__ import annotations

import math
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.interpret.base import DESCRIPTION_MAX, LABEL_MAX, STRUCTURE_ITEM_MAX
from app.models.ir import InteractionType, Role

ROLES: tuple[str, ...] = get_args(Role)
INTERACTION_TYPES: tuple[str, ...] = get_args(InteractionType)

MAX_ELEMENTS = 8
MAX_STRUCTURE = 12
MAX_NOTES = 5
NOTE_MAX = DESCRIPTION_MAX


def _nullable_id(ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {"type": "null"}
    return {"type": ["string", "null"], "enum": [*ids, None]}


def build_schema(element_ids: list[str]) -> dict[str, Any]:
    """JSON schema for one job; ``element_ids`` become closed enums (PLAN §7.4)."""
    ids = list(dict.fromkeys(element_ids))  # dedupe, keep order
    if ids:
        id_prop: dict[str, Any] = {"type": "string", "enum": ids}
        elements: dict[str, Any] = {
            "type": "array",
            "maxItems": min(MAX_ELEMENTS, len(ids)),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "role", "parent_id", "confidence"],
                "properties": {
                    "id": id_prop,
                    "label": {"type": "string", "maxLength": LABEL_MAX},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "parent_id": _nullable_id(ids),
                    "description": {"type": "string", "maxLength": DESCRIPTION_MAX},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    else:
        elements = {"type": "array", "maxItems": 0}

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["elements", "interaction", "structure"],
        "properties": {
            "elements": elements,
            "interaction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "type_confidence", "target_element_id"],
                "properties": {
                    "type": {"type": "string", "enum": list(INTERACTION_TYPES)},
                    "type_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "target_element_id": _nullable_id(ids),
                    "target_description": {"type": "string", "maxLength": DESCRIPTION_MAX},
                    "trigger_description": {"type": "string", "maxLength": DESCRIPTION_MAX},
                },
            },
            "structure": {
                "type": "array",
                "maxItems": MAX_STRUCTURE,
                "items": {"type": "string", "maxLength": STRUCTURE_ITEM_MAX},
            },
            "notes": {
                "type": "array",
                "maxItems": MAX_NOTES,
                "items": {"type": "string", "maxLength": NOTE_MAX},
            },
        },
    }


# ---- lenient payload model -----------------------------------------------------------------------


def _to_str(v: Any) -> Any:
    """Loose scalar -> str (numbers/bools become text); None stays None; containers untouched."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, int | float | bool):
        return str(v)
    return v


def _to_score(v: Any) -> float:
    """Anything -> confidence in [0, 1]; unparseable / NaN -> 0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return min(max(f, 0.0), 1.0)


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PayloadElement(_Loose):
    id: str
    label: str = ""
    role: str = "other"
    parent_id: str | None = None
    description: str | None = None
    confidence: float = 0.0

    @field_validator("id", "label", "role", "parent_id", "description", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> Any:
        return _to_str(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> float:
        return _to_score(v)


class PayloadInteraction(_Loose):
    type: str = "unknown"
    type_confidence: float = 0.0
    target_element_id: str | None = None
    target_description: str | None = None
    trigger_description: str | None = None

    @field_validator(
        "type", "target_element_id", "target_description", "trigger_description", mode="before"
    )
    @classmethod
    def _coerce_str(cls, v: Any) -> Any:
        return _to_str(v)

    @field_validator("type_confidence", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> float:
        return _to_score(v)


class InterpretationPayload(_Loose):
    """What the model returned, before sanitizing. ``elements``/``interaction`` are required."""

    elements: list[PayloadElement]
    interaction: PayloadInteraction
    structure: list[Any] = Field(default_factory=list)
    notes: list[Any] = Field(default_factory=list)
