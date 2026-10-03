"""Persist Layer A's :class:`~app.pipeline.assemble.Measurement` per job (Phase 5).

``POST /api/jobs/{id}/interpret`` re-runs only Layer B + assemble + render. Everything it needs
from Layer A is stored next to ``result.json`` as ``measurement.json``, so the video is never
decoded or measured again.

Format: plain JSON with explicit type tags, decoded through an allow-list. No pickle, so a job
file can never execute code, and a renamed/removed class fails loudly instead of silently
loading the wrong thing.

* dataclass  → ``{"__dc__": "<Name>", <field>: <value>, ...}`` (names in :data:`DATACLASSES`)
* Pydantic   → ``{"__pm__": "<Name>", "data": <model_dump>}`` (IR models in :data:`MODELS`)
* ndarray    → ``{"__nd__": "<dtype>", "shape": [...], "data": [...]}``
* tuple / frozenset → ``{"__tuple__": [...]}`` / ``{"__frozenset__": [...]}``
* floats keep ``inf``/``nan`` (``json`` writes ``Infinity``/``NaN``); NumPy scalars become
  Python scalars.

Bump :data:`FORMAT_VERSION` when the stored shape changes incompatibly; older files then raise
:class:`MeasurementFormatError` and the route answers "upload the video again".
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from pydantic import BaseModel

from app import PIPELINE_VERSION
from app.core.storage import MEASUREMENT_FILE
from app.models import ir
from app.models.ir import SpecWarning
from app.models.measure import (
    ElementCandidate,
    FittedTransition,
    HeuristicInteraction,
    ProbeInfo,
    PropertySeries,
    Rect,
    Scale,
)
from app.pipeline.assemble import Measurement, SegmentWindow

#: Stored as ``format``; bump on incompatible changes.
FORMAT_VERSION = 1

DATACLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Measurement,
        SegmentWindow,
        ProbeInfo,
        Scale,
        Rect,
        ElementCandidate,
        FittedTransition,
        PropertySeries,
        HeuristicInteraction,
    )
}
#: Every Pydantic model defined in the IR module may appear (values, easing, warnings, ...).
MODELS: dict[str, type[BaseModel]] = {
    name: obj
    for name, obj in vars(ir).items()
    if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == ir.__name__
}


class MeasurementFormatError(ValueError):
    """The stored file is missing pieces, from another format version, or names unknown types."""


# --------------------------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------------------------


def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, int | float):
        return obj
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.dtype.str, "shape": list(obj.shape), "data": obj.ravel().tolist()}
    if isinstance(obj, BaseModel):
        name = type(obj).__name__
        if MODELS.get(name) is not type(obj):
            raise TypeError(f"cannot persist model {name}")
        return {"__pm__": name, "data": _encode(obj.model_dump(mode="python", by_alias=True))}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        name = type(obj).__name__
        if DATACLASSES.get(name) is not type(obj):
            raise TypeError(f"cannot persist dataclass {name}")
        out: dict[str, Any] = {"__dc__": name}
        for f in dataclasses.fields(obj):
            out[f.name] = _encode(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v) for v in obj]}
    if isinstance(obj, frozenset | set):
        return {"__frozenset__": sorted((_encode(v) for v in obj), key=repr)}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("only str dict keys can be persisted")
        if any(k.startswith("__") for k in obj):
            raise TypeError("dict keys starting with '__' are reserved for type tags")
        return {k: _encode(v) for k, v in obj.items()}
    raise TypeError(f"cannot persist {type(obj).__name__}")


def dump_measurement(m: Measurement, extra_warnings: list[SpecWarning]) -> dict[str, Any]:
    """JSON-ready document for ``measurement.json``.

    ``extra_warnings`` are the job-level warnings passed to ``assemble`` besides the
    measurement's own (e.g. ``preview_unavailable``), so a re-run reproduces them.
    """
    return {
        "format": FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "measurement": _encode(m),
        "extra_warnings": _encode(list(extra_warnings)),
    }


# --------------------------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------------------------


def _decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__nd__" in obj:
        arr = np.asarray(obj["data"], dtype=np.dtype(obj["__nd__"]))
        return arr.reshape(obj["shape"])
    if "__tuple__" in obj:
        return tuple(_decode(v) for v in obj["__tuple__"])
    if "__frozenset__" in obj:
        return frozenset(_decode(v) for v in obj["__frozenset__"])
    if "__pm__" in obj:
        model = MODELS.get(obj["__pm__"])
        if model is None:
            raise MeasurementFormatError(f"unknown model {obj['__pm__']!r}")
        return model.model_validate(_decode(obj["data"]))
    if "__dc__" in obj:
        cls = DATACLASSES.get(obj["__dc__"])
        if cls is None:
            raise MeasurementFormatError(f"unknown dataclass {obj['__dc__']!r}")
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: _decode(v) for k, v in obj.items() if k != "__dc__"}
        if set(kwargs) - names:
            raise MeasurementFormatError(f"{cls.__name__}: unknown fields {set(kwargs) - names}")
        try:
            return cls(**kwargs)
        except TypeError as exc:  # missing required field
            raise MeasurementFormatError(f"{cls.__name__}: {exc}") from exc
    return {k: _decode(v) for k, v in obj.items()}


def load_measurement(doc: Any) -> tuple[Measurement, list[SpecWarning]]:
    """Inverse of :func:`dump_measurement`. Raises :class:`MeasurementFormatError`."""
    if not isinstance(doc, dict) or doc.get("format") != FORMAT_VERSION:
        got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise MeasurementFormatError(f"unsupported measurement format {got!r}")
    try:
        m = _decode(doc["measurement"])
        extra = _decode(doc.get("extra_warnings", []))
    except MeasurementFormatError:
        raise
    except (KeyError, TypeError, ValueError) as exc:  # incl. pydantic ValidationError
        raise MeasurementFormatError(str(exc)) from exc
    if not isinstance(m, Measurement) or not all(isinstance(w, SpecWarning) for w in extra):
        raise MeasurementFormatError("measurement.json does not contain a Measurement")
    return m, extra


__all__ = [
    "FORMAT_VERSION",
    "MEASUREMENT_FILE",
    "MeasurementFormatError",
    "dump_measurement",
    "load_measurement",
]
