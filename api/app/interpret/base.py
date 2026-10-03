"""Layer B interface (PLAN §7.1). FROZEN CONTRACT after Phase 0.

An interpreter only *labels* things (D4): labels, roles, hierarchy hints, interaction-type
confirmation and a structure inventory. It never produces motion numbers.

Rules every implementation must follow:

* ``interpret()`` never raises for expected failures. ``ClaudeCliInterpreter`` returns the
  deterministic fallback result with ``status`` = ``timeout`` / ``error`` instead.
* ``elements`` keys must be a subset of the input element ids (unknown ids are dropped).
* The result is always *complete*: every input element has an entry and ``interaction`` is set,
  so ``assemble`` never has to special-case missing data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.models.ir import (
    Box,
    ElementKind,
    InteractionType,
    InterpretationStatus,
    InterpreterProvider,
    LabelSource,
    PixelRatio,
    Role,
    TriggerKind,
)

LABEL_MAX = 60
DESCRIPTION_MAX = 200
STRUCTURE_ITEM_MAX = 40


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- input ---------------------------------------------------------------------------------------


class ElementSummary(_Model):
    """What the interpreter is told about one CV element."""

    id: str
    bbox: Box = Field(description="State A box, CSS px.")
    kind: ElementKind
    parent_id: str | None = None
    text_like: bool = False
    change_summary: str = Field(description='Measured change, e.g. "moves up ~8px; appears".')


class VideoMeta(_Model):
    width: int
    height: int
    duration_ms: int
    pixel_ratio: PixelRatio


class InterpretationInput(_Model):
    """Everything Layer B may see. ``keyframe_paths`` keys are artifact names (``state_a.png``)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    job_dir: Path
    keyframe_paths: dict[str, Path]
    elements: list[ElementSummary]
    heuristic_type: InteractionType
    heuristic_trigger: TriggerKind
    cursor_summary: str = Field(description='e.g. "enters e1 at 1120 ms, stationary at onset".')
    video_meta: VideoMeta


# ---- output --------------------------------------------------------------------------------------


class InterpretedElement(_Model):
    """One element's labels. ``label_source`` says whether the model (``interpreter``) or the
    deterministic fallback (``heuristic``) produced them; an ``ok`` result can mix both when the
    model omitted elements."""

    label: str = Field(max_length=LABEL_MAX)
    role: Role
    parent_id: str | None = None
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    confidence: float = Field(ge=0.0, le=1.0)
    label_source: LabelSource = "interpreter"


class InterpretedInteraction(_Model):
    type_confirmation: InteractionType
    type_confidence: float = Field(ge=0.0, le=1.0)
    target_element_id: str | None = None
    target_description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    trigger_description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)


class InterpretationResult(_Model):
    """Layer B output merged by ``assemble`` according to PLAN §7.5."""

    status: InterpretationStatus
    provider: InterpreterProvider
    model: str | None = None
    duration_ms: int | None = None
    elements: dict[str, InterpretedElement]
    interaction: InterpretedInteraction
    structure: list[str] = Field(default_factory=list, max_length=12)
    notes: list[str] = Field(default_factory=list, max_length=5)


# ---- protocol ------------------------------------------------------------------------------------


@runtime_checkable
class Interpreter(Protocol):
    """Implemented by ``ClaudeCliInterpreter`` and ``FallbackInterpreter`` (Track I).

    Selected by ``app.interpret.get_interpreter(settings, use_interpreter)``.
    """

    provider: InterpreterProvider

    def interpret(self, inp: InterpretationInput) -> InterpretationResult:
        """Label the elements. Must not raise for timeout / CLI / parse failures."""
        ...
