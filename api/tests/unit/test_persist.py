"""Phase 5: ``measurement.json`` round trip (re-run interpretation without re-measuring)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.models.ir import CursorEvent, ElementStatic, MeasuredNumber, SpecWarning
from app.pipeline import assemble as asm
from app.pipeline.confidence import static_confidence
from app.pipeline.persist import (
    FORMAT_VERSION,
    MeasurementFormatError,
    dump_measurement,
    load_measurement,
)
from tests.unit.test_assemble import _el, _ft, _interp, _measurement, _series


def _rich_measurement() -> asm.Measurement:
    els = [_el("e1"), _el("e2", "appear", parent="e1")]
    els[0].warp_ab = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -8.0]])
    els[1].notes.append("caused_by_resize")
    fts = [_ft("t1", _series("e1", "translateY")), _ft("t2", _series("e2", "opacity", v0=0, v1=1))]
    fts[1].series.notes.append("timing shared with opacity")
    m = _measurement(els, fts)
    m.probe.timestamps_estimated = True
    m.cursor_events = [CursorEvent(t_ms=1000, kind="enter", element_id="e1")]
    m.statics = {
        "e1": ElementStatic(
            border_radius_px=MeasuredNumber(
                value=12.0, confidence=static_confidence(0.4, "border-radius")
            )
        )
    }
    m.warnings = [SpecWarning(code="vfr_source", severity="info", message="VFR")]
    m.segments[0].trigger_event_s = 0.97
    return m


def _roundtrip(m: asm.Measurement, extra: list[SpecWarning]):
    text = json.dumps(dump_measurement(m, extra))  # what storage.write_json does
    return load_measurement(json.loads(text))


def test_roundtrip_reassembles_identically(tmp_path: Path) -> None:
    m = _rich_measurement()
    extra = [SpecWarning(code="preview_unavailable", severity="warn", message="no preview")]
    m2, extra2 = _roundtrip(m, extra)
    assert extra2 == extra
    assert m2.probe.frame_ts.dtype == np.float64
    assert np.array_equal(m2.probe.frame_ts, m.probe.frame_ts)
    assert np.array_equal(m2.elements[0].warp_ab, m.elements[0].warp_ab)
    assert m2.elements[1].warp_ab is None and m2.elements[1].notes == ["caused_by_resize"]
    assert m2.transitions[0].series.from_value == m.transitions[0].series.from_value
    assert m2.scale == m.scale and m2.frame_css == m.frame_css

    interp = _interp(m, tmp_path)
    kw = {"job_id": "a" * 32, "filename": "x.mp4", "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
          "pipeline_version": "t", "use_interpreter": False}  # fmt: skip
    first = asm.assemble(m, interp, extra_warnings=extra, **kw)
    again = asm.assemble(m2, interp, extra_warnings=extra2, **kw)
    assert again.model_dump(mode="json") == first.model_dump(mode="json")


def test_non_finite_floats_survive() -> None:
    m = _rich_measurement()
    m.transitions[0].series.values[3] = np.nan
    m.transitions[0].series.stable_std = float("inf")
    m2, _ = _roundtrip(m, [])
    assert np.isnan(m2.transitions[0].series.values[3])
    assert m2.transitions[0].series.stable_std == float("inf")


def test_rejects_other_versions_and_unknown_types() -> None:
    doc = dump_measurement(_rich_measurement(), [])
    with pytest.raises(MeasurementFormatError, match="format"):
        load_measurement({**doc, "format": FORMAT_VERSION + 1})
    with pytest.raises(MeasurementFormatError):
        load_measurement([])
    evil = json.loads(json.dumps(doc))
    evil["measurement"]["__dc__"] = "Popen"
    with pytest.raises(MeasurementFormatError, match="unknown dataclass"):
        load_measurement(evil)
    evil = json.loads(json.dumps(doc))
    evil["measurement"]["surprise"] = 1
    with pytest.raises(MeasurementFormatError, match="unknown fields"):
        load_measurement(evil)
    evil = json.loads(json.dumps(doc))
    evil["measurement"]["transitions"][0]["__dc__"] = "ElementCandidate"
    with pytest.raises(MeasurementFormatError):
        load_measurement(evil)
