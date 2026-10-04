"""JSON tab (PRD §20, PLAN §9.5): the IR without per-transition / continuous ``samples``.

Key order is the model's field order (stable), 2-space indent, UTF-8 kept as-is. The full IR
including samples stays available from ``GET /api/jobs/{id}/result``.
"""

from __future__ import annotations

import json

from app.models.ir import MotionSpec

EXCLUDE = {"transitions": {"__all__": {"samples"}}}
#: Continuous mode also drops the velocity-profile samples (PLAN-continuous §5.3).
EXCLUDE_CONTINUOUS = {**EXCLUDE, "continuous": {"samples"}}


def render(spec: MotionSpec) -> str:
    exclude = EXCLUDE_CONTINUOUS if spec.continuous is not None else EXCLUDE
    data = spec.model_dump(mode="json", exclude=exclude)
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
