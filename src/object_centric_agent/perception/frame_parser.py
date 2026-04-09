"""
Frame parser: orchestrates the full perception pipeline.

    raw RGB frame
        │
        ▼
    FragmentDetector.detect()        → List[Fragment]
        │
        ▼
    SpriteRegistry.match()           → List[SpriteInstance]  (track_id = -1)
        │
        ▼
    ObjectTracker.update()           → List[SpriteInstance]  (track_id assigned)
        │
        ▼
    WorldState                       → structured frame snapshot

The FrameParser also feeds transitions into the ComovementTracker to
discover multi-color sprites over time, and triggers SpriteRegistry merges
when reliable co-movement evidence is found.

This is the single entry point for perception. The agent calls:
    world_state = parser.parse(frame, frame_index)
and gets back a fully structured WorldState.
"""

from __future__ import annotations

import logging

import numpy as np

from ..state.object_state import Fragment, SpriteInstance, WorldState
from .fragment_detector import FragmentDetector
from .comovement_tracker import ComovementTracker
from .sprite_registry import SpriteRegistry
from .object_tracker import ObjectTracker

logger = logging.getLogger(__name__)


class FrameParser:
    """
    Orchestrates fragment detection → sprite matching → object tracking.

    Stateful: maintains the SpriteRegistry, ComovementTracker, and ObjectTracker
    across frames. Each call to parse() advances the internal state.

    Usage:
        parser = FrameParser()
        ws0 = parser.parse(frame_0, frame_index=0)  # bootstrap
        ws1 = parser.parse(frame_1, frame_index=1, action_id=1)  # after ACTION1
        ws2 = parser.parse(frame_2, frame_index=2, action_id=3)  # after ACTION3
    """

    def __init__(self, min_fragment_size: int = 1):
        self.fragment_detector = FragmentDetector(min_fragment_size=min_fragment_size)
        self.sprite_registry = SpriteRegistry()
        self.comovement_tracker = ComovementTracker()
        self.object_tracker = ObjectTracker()

        self._frame_count: int = 0
        self._background_color: tuple[int, int, int] | None = None
        self._previous_fragments: list[Fragment] | None = None

        # How often to check for co-movement merges
        self._comovement_check_interval: int = 3
        self._transitions_since_last_check: int = 0

    @property
    def is_bootstrapped(self) -> bool:
        return self.sprite_registry._bootstrapped

    @property
    def background_color(self) -> tuple[int, int, int]:
        return self._background_color or (0, 0, 0)

    def parse(
        self,
        frame: np.ndarray,
        frame_index: int,
        action_id: int | None = None,
    ) -> WorldState:
        """
        Parse a single frame into a structured WorldState.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            frame_index: Monotonically increasing frame counter.
            action_id: The action that was taken to produce this frame (None for
                the initial frame). Used only for logging context.

        Returns:
            WorldState with all detected and tracked objects.
        """
        # Step 1: Detect fragments
        fragments, bg = self.fragment_detector.detect(
            frame, background_color=self._background_color
        )
        self._background_color = bg

        # Step 2: Bootstrap or match
        if not self.is_bootstrapped:
            # First frame: bootstrap registry
            self.sprite_registry.bootstrap(fragments, frame_index)
            logger.info(
                f"[FrameParser] Bootstrap: {len(fragments)} fragments → "
                f"{self.sprite_registry.num_types} sprite types"
            )

        # Step 3: Feed co-movement tracker (if we have a previous frame)
        if self._previous_fragments is not None and action_id is not None:
            groups = self.comovement_tracker.observe_transition(
                self._previous_fragments, fragments
            )
            self._transitions_since_last_check += 1

            if groups:
                moving = [g for g in groups if g.moved]
                if moving:
                    logger.debug(
                        f"[FrameParser] Co-movement: {len(moving)} moving groups "
                        f"after ACTION{action_id}"
                    )

            # Periodically check for merges
            if self._transitions_since_last_check >= self._comovement_check_interval:
                new_types = self.sprite_registry.merge_from_comovement(
                    self.comovement_tracker
                )
                if new_types:
                    logger.info(
                        f"[FrameParser] Merged {len(new_types)} new multi-fragment types"
                    )
                self._transitions_since_last_check = 0

        # Step 4: Match fragments to sprite types
        instances = self.sprite_registry.match(fragments, frame_index)

        # Step 5: Track across frames
        instances = self.object_tracker.update(instances, frame_index)

        # Step 6: Build WorldState
        world_state = WorldState(
            frame=frame,
            frame_index=frame_index,
            fragments=fragments,
            sprites=instances,
            composites=[],
            background_color=bg,
        )

        # Update internal state
        self._previous_fragments = fragments
        self._frame_count += 1

        if self._frame_count <= 3 or self._frame_count % 10 == 0:
            logger.info(
                f"[FrameParser] Frame {frame_index}: "
                f"{len(fragments)} fragments → {len(instances)} sprites "
                f"({self.sprite_registry.num_types} types known)"
            )

        return world_state

    def get_registry_summary(self) -> str:
        """Get human-readable summary of known sprite types."""
        return self.sprite_registry.describe()

    def mark_sprite_type(
        self,
        type_id: int,
        *,
        is_static: bool | None = None,
        is_player: bool | None = None,
        is_selectable: bool | None = None,
        name: str | None = None,
    ) -> None:
        """
        Update behavioral metadata on a sprite type.

        Called by the exploration layer when it discovers object roles.
        For example, after observing that sprite_type 3 moves in response
        to directional actions, mark it as the player.
        """
        sprite_type = self.sprite_registry.get_type(type_id)
        if sprite_type is None:
            return

        if is_static is not None:
            sprite_type.is_static = is_static
        if is_player is not None:
            sprite_type.is_player = is_player
        if is_selectable is not None:
            sprite_type.is_selectable = is_selectable
        if name is not None:
            sprite_type.name = name

    def request_merge(self, colors: list[tuple[int, int, int]]) -> None:
        """
        LLM refinement: request merging sprite types by color.

        Called when the LLM observes that fragments of different colors are
        actually parts of the same logical object.
        """
        result = self.sprite_registry.refine_from_hint(merge_colors=colors)
        if result is not None:
            logger.info(f"[FrameParser] LLM-requested merge: {colors} → {result}")

    def request_split(self, type_id: int) -> None:
        """
        LLM refinement: request splitting a multi-fragment type.

        Called when the LLM observes that a previously-merged sprite's parts
        actually move independently.
        """
        result = self.sprite_registry.refine_from_hint(split_type_id=type_id)
        if result:
            logger.info(f"[FrameParser] LLM-requested split of type {type_id}")
