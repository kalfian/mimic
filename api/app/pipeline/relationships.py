"""Timing relationships between the transitions of one segment (PLAN §6.9).

Pure function of fitted start times / durations. Output is IR ``Relationship`` objects whose
``transition_ids[0]`` is the reference (primary) transition:

* ``simultaneous`` — every transition with ``|t0 - t0_primary| ≤ τ`` (one relationship,
  ``offset_ms = 0``).
* ``stagger`` — ≥ 3 transitions of the same property on elements of similar area (±30 %) and
  the same role, whose consecutive start offsets are roughly constant
  (``std ≤ max(τ, 0.25·mean)`` and ``mean > τ``) → ``interval_ms = mean``. The reference of a
  stagger relationship is its first member.
* ``sequential`` / ``delayed`` — remaining transitions, clustered by start time (members within
  τ of the cluster's first start share one relationship). ``sequential`` if every member starts
  at or after the primary's end (``t0 ≥ t0_p + D_p − τ``), else ``delayed``;
  ``offset_ms`` = mean start offset from the primary.

Members of a stagger group are not repeated in ``delayed``/``sequential`` relationships (the
stagger already describes their offsets); they still join ``simultaneous`` if they start with
the primary. τ = ``max(timing_resolution, tau_min_ms)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from app.models.ir import Property, Relationship, SegmentId
from app.models.measure import FittedTransition
from app.pipeline.params import DEFAULT_PARAMS, RelationshipParams


@dataclass(frozen=True, slots=True)
class TimedTransition:
    """The minimum a relationship needs to know about a fitted transition."""

    id: str
    segment: SegmentId
    element_id: str
    property: Property
    t0_s: float
    duration_s: float
    area: float | None = None  # element area (CSS px²), for stagger similarity
    role: str | None = None  # element role, for stagger similarity (None matches None)

    @property
    def end_s(self) -> float:
        return self.t0_s + self.duration_s


def from_fitted(
    ft: FittedTransition, *, area: float | None = None, role: str | None = None
) -> TimedTransition:
    """Adapter from the ``FittedTransition`` contract type."""
    s = ft.series
    return TimedTransition(
        id=ft.id,
        segment=s.segment,
        element_id=s.element_id,
        property=s.property,
        t0_s=ft.t0_s,
        duration_s=ft.duration_s,
        area=area,
        role=role,
    )


def tau_s(
    timing_resolution_s: float, params: RelationshipParams = DEFAULT_PARAMS.relationships
) -> float:
    """τ = max(timing_resolution, tau_min_ms) in seconds."""
    return max(float(timing_resolution_s), params.tau_min_ms / 1000.0)


def pick_primary(
    transitions: Sequence[TimedTransition], target_element_id: str | None
) -> TimedTransition:
    """The target element's earliest transition; fallback: the earliest overall."""
    if not transitions:
        raise ValueError("no transitions")
    pool = [t for t in transitions if t.element_id == target_element_id] or list(transitions)
    return min(pool, key=lambda t: (t.t0_s, t.id))


def find_staggers(
    transitions: Sequence[TimedTransition],
    tau: float,
    params: RelationshipParams = DEFAULT_PARAMS.relationships,
) -> list[tuple[list[TimedTransition], float]]:
    """Stagger groups ``[(members sorted by t0, interval_s)]`` within one segment."""
    groups: dict[tuple[str, str | None], list[TimedTransition]] = {}
    for t in transitions:
        groups.setdefault((t.property, t.role), []).append(t)

    out: list[tuple[list[TimedTransition], float]] = []
    for members in groups.values():
        if len(members) < params.stagger_min_count:
            continue
        similar = _similar_area(members, params.stagger_area_tol)
        if len(similar) < params.stagger_min_count:
            continue
        similar.sort(key=lambda t: (t.t0_s, t.id))
        offsets = np.diff([t.t0_s for t in similar])
        mean = float(np.mean(offsets))
        std = float(np.std(offsets))
        if mean > tau and std <= max(tau, params.stagger_std_frac * mean):
            out.append((similar, mean))
    out.sort(key=lambda g: (g[0][0].t0_s, g[0][0].id))
    return out


def _similar_area(members: list[TimedTransition], tol: float) -> list[TimedTransition]:
    """Members whose area is within ±tol of the median area (all, if areas are unknown)."""
    areas = [t.area for t in members]
    if any(a is None for a in areas):
        return list(members) if all(a is None for a in areas) else []
    med = float(np.median(np.asarray(areas, dtype=np.float64)))
    if med <= 0:
        return []
    return [t for t in members if abs(float(t.area) - med) <= tol * med]  # type: ignore[arg-type]


def segment_relationships(
    transitions: Sequence[TimedTransition],
    *,
    target_element_id: str | None,
    timing_resolution_s: float,
    params: RelationshipParams = DEFAULT_PARAMS.relationships,
) -> list[Relationship]:
    """Relationships of one segment (all ``transitions`` must share a segment)."""
    if len(transitions) < 2:
        return []
    segs = {t.segment for t in transitions}
    if len(segs) != 1:
        raise ValueError(f"transitions span several segments: {sorted(segs)}")
    seg = next(iter(segs))
    tau = tau_s(timing_resolution_s, params)
    primary = pick_primary(transitions, target_element_id)
    ordered = sorted(transitions, key=lambda t: (t.t0_s, t.id))
    out: list[Relationship] = []

    simultaneous = [t for t in ordered if t.id != primary.id and abs(t.t0_s - primary.t0_s) <= tau]
    if simultaneous:
        out.append(
            Relationship(
                kind="simultaneous",
                segment_id=seg,
                transition_ids=[primary.id, *(t.id for t in simultaneous)],
                offset_ms=0,
            )
        )

    staggered: set[str] = set()
    for members, interval in find_staggers(ordered, tau, params):
        staggered.update(t.id for t in members)
        out.append(
            Relationship(
                kind="stagger",
                segment_id=seg,
                transition_ids=[t.id for t in members],
                offset_ms=_ms(members[0].t0_s - primary.t0_s),
                interval_ms=_ms(interval),
            )
        )

    handled = {primary.id, *(t.id for t in simultaneous), *staggered}
    rest = [t for t in ordered if t.id not in handled]
    for cluster in _clusters(rest, tau):
        offset = float(np.mean([t.t0_s for t in cluster])) - primary.t0_s
        sequential = all(t.t0_s >= primary.end_s - tau for t in cluster)
        out.append(
            Relationship(
                kind="sequential" if sequential else "delayed",
                segment_id=seg,
                transition_ids=[primary.id, *(t.id for t in cluster)],
                offset_ms=_ms(offset),
            )
        )
    return out


def relationships(
    transitions: Iterable[TimedTransition],
    *,
    target_element_id: str | None,
    timing_resolution_s: float,
    params: RelationshipParams = DEFAULT_PARAMS.relationships,
) -> list[Relationship]:
    """Relationships for all segments, in segment order fwd, rev, rt_in, rt_out."""
    by_seg: dict[str, list[TimedTransition]] = {}
    for t in transitions:
        by_seg.setdefault(t.segment, []).append(t)
    out: list[Relationship] = []
    for seg in ("fwd", "rev", "rt_in", "rt_out"):
        if seg in by_seg:
            out.extend(
                segment_relationships(
                    by_seg[seg],
                    target_element_id=target_element_id,
                    timing_resolution_s=timing_resolution_s,
                    params=params,
                )
            )
    return out


def _clusters(ordered: list[TimedTransition], tau: float) -> list[list[TimedTransition]]:
    """Group start-sorted transitions whose t0 is within τ of the cluster's first member."""
    clusters: list[list[TimedTransition]] = []
    for t in ordered:
        if clusters and t.t0_s - clusters[-1][0].t0_s <= tau:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return clusters


def _ms(seconds: float) -> int:
    return round(seconds * 1000.0)


__all__ = [
    "TimedTransition",
    "find_staggers",
    "from_fitted",
    "pick_primary",
    "relationships",
    "segment_relationships",
    "tau_s",
]
