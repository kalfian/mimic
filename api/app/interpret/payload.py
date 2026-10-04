"""Backend-independent handling of a model's labeling answer.

Shared by ``claude_cli`` and ``openai_compat``: defensive JSON extraction, payload sanitizing
(PLAN §7.2: unknown ids dropped, strings clipped, control characters stripped, unknown roles ->
``other``) and the best-effort ``interpretation_raw.json`` writer with secret redaction.

Continuous mode (PLAN-continuous §4.7): the model only labels. Elements of kind ``scroller``
always get role ``scroller`` (the schema never offers it, so Layer B cannot put it on another
element), and the interaction type, its confidence, the target (= the scroller) and the trigger
text are kept from the heuristic baseline; only ``target_description`` comes from the model.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.interpret.base import (
    DESCRIPTION_MAX,
    LABEL_MAX,
    STRUCTURE_ITEM_MAX,
    InterpretationInput,
    InterpretationResult,
    InterpretedElement,
    InterpretedInteraction,
)
from app.interpret.prompt import is_continuous
from app.interpret.schema import (
    INTERACTION_TYPES,
    MAX_NOTES,
    MAX_STRUCTURE,
    NOTE_MAX,
    ROLES,
    InterpretationPayload,
)
from app.models.ir import InterpreterProvider

RAW_FILENAME = "interpretation_raw.json"
REDACTED = "<redacted>"

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


# ---- JSON extraction -----------------------------------------------------------------------------


def first_balanced_object(text: str) -> dict[str, Any] | None:
    """First ``{...}`` substring of ``text`` that parses as a JSON object (string-aware)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Object from free model text: whole text, then code fences, then first balanced ``{}``.

    Returns ``(obj, path)`` with path ``json`` | ``fenced`` | ``braces`` | ``none``.
    """
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, "json"
    except json.JSONDecodeError:
        pass
    for m in _FENCE_RE.finditer(stripped):
        try:
            obj = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, "fenced"
    obj = first_balanced_object(stripped)
    return (obj, "braces") if obj is not None else (None, "none")


# ---- sanitizing ----------------------------------------------------------------------------------


def clean_text(value: str | None, max_len: int) -> str:
    """Strip control/format characters, collapse whitespace, clip to ``max_len``."""
    if not value:
        return ""
    s = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in value)
    return " ".join(s.split())[:max_len].rstrip()


def _cycle_members(parents: dict[str, str | None]) -> set[str]:
    bad: set[str] = set()
    for start in parents:
        seen: list[str] = []
        node: str | None = start
        while node is not None and node not in seen:
            seen.append(node)
            node = parents.get(node)
        if node is not None:  # walked back onto the path -> node is on a cycle
            bad.update(seen[seen.index(node) :])
    return bad


def break_cycles(parents: dict[str, str | None], cv_parents: dict[str, str | None]) -> list[str]:
    """Revert parents on cycles to the CV hierarchy (in place). Returns the reverted ids."""
    reverted: list[str] = []
    for _ in range(len(parents) + 1):
        bad = _cycle_members(parents)
        if not bad:
            return reverted
        changed = False
        for el_id in sorted(bad):
            if parents[el_id] != cv_parents.get(el_id):
                parents[el_id] = cv_parents.get(el_id)
                reverted.append(el_id)
                changed = True
        if not changed:
            break
    for el_id in parents:  # only reachable if the CV hierarchy itself is cyclic
        parents[el_id] = None
    return reverted


def sanitize_payload(
    payload: InterpretationPayload,
    inp: InterpretationInput,
    base: InterpretationResult,
    *,
    provider: InterpreterProvider,
    model: str | None,
    duration_ms: int | None,
) -> tuple[InterpretationResult | None, list[str]]:
    """Payload -> complete ``status="ok"`` result, gaps filled from the fallback ``base``.

    Returns ``(None, issues)`` when nothing usable came back (no valid element label).
    """
    issues: list[str] = []
    order = [e.id for e in inp.elements]
    known = set(order)
    cv_parents = {e.id: e.parent_id for e in inp.elements}
    continuous = is_continuous(inp)
    scroller_ids = {e.id for e in inp.elements if e.kind == "scroller"}

    got: dict[str, InterpretedElement] = {}
    parents: dict[str, str | None] = {}
    for pe in payload.elements:
        if pe.id not in known:
            issues.append(f"dropped unknown element id {clean_text(pe.id, 20)!r}")
            continue
        if pe.id in got:
            issues.append(f"dropped duplicate entry for {pe.id}")
            continue
        label = clean_text(pe.label, LABEL_MAX)
        if not label:
            label = base.elements[pe.id].label
            issues.append(f"{pe.id}: empty label")
        role = pe.role if pe.role in ROLES else "other"
        if pe.id in scroller_ids:
            role = "scroller"  # measured kind; the model was told to answer "container"
        elif role != pe.role:
            issues.append(f"{pe.id}: role {clean_text(pe.role, 30)!r} -> other")
        parent = pe.parent_id
        if parent is not None and (parent not in known or parent == pe.id):
            issues.append(f"{pe.id}: invalid parent {clean_text(parent, 20)!r} -> CV parent")
            parent = cv_parents[pe.id]
        parents[pe.id] = parent
        got[pe.id] = InterpretedElement(
            label=label,
            role=role,  # type: ignore[arg-type]
            parent_id=parent,
            description=clean_text(pe.description, DESCRIPTION_MAX) or None,
            confidence=pe.confidence,
        )

    if order and not got:
        return None, [*issues, "no valid element labels"]

    missing = [i for i in order if i not in got]
    for el_id in missing:
        parents[el_id] = base.elements[el_id].parent_id
    if missing:
        issues.append(f"missing from AI output (heuristic labels kept): {', '.join(missing)}")
    for el_id in break_cycles(parents, cv_parents):
        issues.append(f"{el_id}: parent cycle -> CV parent")

    elements: dict[str, InterpretedElement] = {}
    for el_id in order:
        src = got.get(el_id) or base.elements[el_id]
        elements[el_id] = src.model_copy(update={"parent_id": parents[el_id]})

    pi = payload.interaction
    if continuous:
        interaction = _continuous_interaction(payload, base, elements, issues)
    else:
        itype = pi.type if pi.type in INTERACTION_TYPES else "unknown"
        type_conf = pi.type_confidence if itype == pi.type else 0.0
        if itype != pi.type:
            issues.append(f"interaction type {clean_text(pi.type, 30)!r} -> unknown")
        target = pi.target_element_id
        if target is not None and target not in known:
            issues.append(f"unknown target {clean_text(target, 20)!r} -> heuristic target")
            target = base.interaction.target_element_id
        target_desc = clean_text(pi.target_description, DESCRIPTION_MAX) or (
            elements[target].label if target is not None else None
        )
        interaction = InterpretedInteraction(
            type_confirmation=itype,  # type: ignore[arg-type]
            type_confidence=type_conf,
            target_element_id=target,
            target_description=target_desc,
            trigger_description=clean_text(pi.trigger_description, DESCRIPTION_MAX)
            or base.interaction.trigger_description,
        )

    structure: list[str] = []
    for item in payload.structure:
        s = clean_text(item if isinstance(item, str) else None, STRUCTURE_ITEM_MAX)
        if s and s.lower() not in {x.lower() for x in structure}:
            structure.append(s)
    notes = [clean_text(n, NOTE_MAX) for n in payload.notes if isinstance(n, str)]
    notes = [n for n in notes if n]
    if issues:
        repaired = clean_text("AI output repaired: " + "; ".join(issues), NOTE_MAX)
        notes = [*notes[: MAX_NOTES - 1], repaired]

    result = InterpretationResult(
        status="ok",
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        elements=elements,
        interaction=interaction,
        structure=structure[:MAX_STRUCTURE],
        notes=notes[:MAX_NOTES],
    )
    return result, issues


def _continuous_interaction(
    payload: InterpretationPayload,
    base: InterpretationResult,
    elements: dict[str, InterpretedElement],
    issues: list[str],
) -> InterpretedInteraction:
    """Continuous mode: type / confidence / target / trigger stay heuristic (Layer B cannot see
    velocities, PLAN-continuous §4.7); the model's ``type`` is ignored without a note."""
    pi = payload.interaction
    target = base.interaction.target_element_id
    if pi.target_element_id is not None and pi.target_element_id != target:
        issues.append(
            f"target {clean_text(pi.target_element_id, 20)!r} -> scroller {target or '-'}"
        )
    target_desc = clean_text(pi.target_description, DESCRIPTION_MAX) or (
        elements[target].label if target is not None and target in elements else None
    )
    return base.interaction.model_copy(update={"target_description": target_desc})


# ---- raw record ----------------------------------------------------------------------------------


def redact(value: Any, secrets: Iterable[str]) -> Any:
    """Recursively replace every occurrence of each secret in strings with ``<redacted>``."""
    secrets = [s for s in secrets if s]
    if not secrets:
        return value
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, REDACTED)
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v, secrets) for v in value]
    return value


def write_raw(job_dir: Path, data: dict[str, Any], secrets: Iterable[str] = ()) -> None:
    """Best-effort atomic write of ``interpretation_raw.json``. Never raises; redacts secrets.

    For a real job directory (``<jobs>/<job_id>``) the write goes through
    :class:`~app.core.storage.LocalJobStorage` (artifact whitelist + path checks). Other
    directories (CLI runs, tests) get a plain atomic write.
    """
    if not job_dir.is_dir():
        return
    try:
        safe = redact(data, list(secrets))
        from app.core.storage import LocalJobStorage, is_valid_job_id

        if is_valid_job_id(job_dir.name):
            plain = json.loads(json.dumps(safe, default=str))  # JSON-safe before the write
            LocalJobStorage(job_dir.parent).write_json(job_dir.name, RAW_FILENAME, plain)
            return
        text = json.dumps(safe, indent=2, ensure_ascii=False, default=str)
        tmp = job_dir / f".{RAW_FILENAME}.tmp"
        tmp.write_text(text, "utf-8")
        tmp.replace(job_dir / RAW_FILENAME)
    except Exception:  # noqa: BLE001 - the raw record is diagnostics only
        pass
