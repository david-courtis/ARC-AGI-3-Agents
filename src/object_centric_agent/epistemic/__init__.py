"""Epistemic state: continuous confidence tracking for object–action interactions."""

from .knowledge_state import (
    EpistemicState,
    TransitionEffect,
    TransitionRecord,
    ObservationContext,
)

__all__ = [
    "EpistemicState",
    "TransitionEffect",
    "TransitionRecord",
    "ObservationContext",
]
