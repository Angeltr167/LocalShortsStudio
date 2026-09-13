"""Five-stage deterministic timing for cartoon story templates.

This is intentionally a tiny timing policy, not a general keyframe or animation
authoring system. Template drawing remains authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass


PHASE_NAMES = ("enter", "establish", "action", "emphasis", "resolve")


@dataclass(frozen=True)
class ChoreographyPhase:
    name: str
    progress: float


def phase_at(progress: float, duration: float) -> ChoreographyPhase:
    """Return an absolute-order stage; short beats compress setup first."""
    value = max(0.0, min(1.0, float(progress)))
    short = float(duration) < 1.6
    weights = (0.08, 0.15, 0.50, 0.15, 0.12) if short else (0.12, 0.18, 0.40, 0.15, 0.15)
    cursor = 0.0
    for name, weight in zip(PHASE_NAMES, weights):
        end = cursor + weight
        if value < end or name == PHASE_NAMES[-1]:
            return ChoreographyPhase(name, max(0.0, min(1.0, (value - cursor) / weight)))
        cursor = end
    return ChoreographyPhase("resolve", 1.0)


def stage_at(scene, absolute_time: float) -> ChoreographyPhase:
    duration = max(0.01, scene.duration)
    progress = max(0.0, min(1.0, (absolute_time - scene.start) / duration))
    return phase_at(progress, duration)
