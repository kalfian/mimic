"""Mimic backend: screen recording of a UI interaction -> MotionSpec IR -> text outputs."""

__version__ = "0.1.0"

#: Recorded in ``MotionSpec.meta.pipeline_version``. Bump when measurement behaviour changes.
#: 0.2.0 (PLAN-continuous P2): regime detection + continuous (scroller) mode, measured
#: appearance (§13), IR 0.2.
PIPELINE_VERSION = "0.2.0"
