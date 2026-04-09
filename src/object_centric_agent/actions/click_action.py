"""
ACTION6 click targeting: compute (x, y) coordinates for clicking on sprites.

ARC-AGI-3 ACTION6 requires {x, y} data where x and y are pixel coordinates
in the 0-63 range. This module computes click targets from the perception
layer's SpriteInstances.

Usage:
    planner = ClickPlanner()
    target = planner.target_sprite(sprite_instance)
    # target.x, target.y → pass to env.step(ACTION6, data={"x": target.x, "y": target.y})

The ClickPlanner also supports:
    - Clicking on a sprite type (targets the nearest instance)
    - Clicking on a specific pixel position
    - Systematic click survey (click each type once to discover selectability)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..state.object_state import SpriteInstance, WorldState


@dataclass(frozen=True)
class ClickTarget:
    """
    A resolved click target with x, y coordinates for ACTION6.

    ARC-AGI-3 uses (x, y) where x = column, y = row (screen coordinates).
    Our internal representation uses (row, col), so we convert here.
    """
    x: int  # column (0-63)
    y: int  # row (0-63)
    reason: str = ""
    target_sprite_track_id: int | None = None
    target_sprite_type_id: int | None = None

    @staticmethod
    def from_row_col(row: int, col: int, **kwargs) -> ClickTarget:
        """Convert from (row, col) internal coords to (x=col, y=row) API coords."""
        return ClickTarget(
            x=max(0, min(63, col)),
            y=max(0, min(63, row)),
            **kwargs,
        )

    @property
    def as_data(self) -> dict[str, int]:
        """Format for env.step(ACTION6, data=...)."""
        return {"x": self.x, "y": self.y}


@dataclass
class ClickSurveyState:
    """Tracks which sprite types have been click-tested."""
    tested_type_ids: set[int] = field(default_factory=set)
    results: dict[int, bool] = field(default_factory=dict)  # type_id → had_effect

    def is_tested(self, type_id: int) -> bool:
        return type_id in self.tested_type_ids

    def record(self, type_id: int, had_effect: bool) -> None:
        self.tested_type_ids.add(type_id)
        self.results[type_id] = had_effect

    @property
    def selectable_types(self) -> set[int]:
        return {tid for tid, eff in self.results.items() if eff}

    @property
    def untested_count(self) -> int:
        return 0  # computed externally based on known types


class ClickPlanner:
    """
    Plans ACTION6 click targets based on the current WorldState.

    Responsibilities:
    - Target a specific sprite instance (click its center).
    - Target a sprite type (find nearest instance, click its center).
    - Generate a click survey plan (test each type once).
    - Track which types are selectable (respond to clicks).
    """

    def __init__(self):
        self.survey = ClickSurveyState()

    def target_sprite(self, sprite: SpriteInstance) -> ClickTarget:
        """Click on the center of a specific sprite instance."""
        row = int(sprite.center[0])
        col = int(sprite.center[1])
        return ClickTarget.from_row_col(
            row, col,
            reason=f"click on {sprite.type_name} (track {sprite.track_id})",
            target_sprite_track_id=sprite.track_id,
            target_sprite_type_id=sprite.type_id,
        )

    def target_type(
        self,
        type_id: int,
        world: WorldState,
        prefer_near: tuple[int, int] | None = None,
    ) -> ClickTarget | None:
        """
        Click on the nearest instance of a sprite type.

        Args:
            type_id: Which sprite type to click on.
            world: Current world state (to find instances).
            prefer_near: If given, prefer the instance closest to this (row, col).

        Returns:
            ClickTarget, or None if no instance of this type exists.
        """
        instances = world.get_sprites_of_type(type_id)
        if not instances:
            return None

        if prefer_near is not None:
            # Sort by distance to preferred position
            instances = sorted(
                instances,
                key=lambda s: (
                    (s.center[0] - prefer_near[0]) ** 2
                    + (s.center[1] - prefer_near[1]) ** 2
                ),
            )

        return self.target_sprite(instances[0])

    def target_position(self, row: int, col: int, reason: str = "") -> ClickTarget:
        """Click on a specific pixel position."""
        return ClickTarget.from_row_col(row, col, reason=reason)

    def next_survey_target(
        self,
        world: WorldState,
    ) -> ClickTarget | None:
        """
        Get the next untested sprite type to click on for the survey.

        Returns None if all types have been tested.
        """
        for sprite in world.sprites:
            if not self.survey.is_tested(sprite.type_id):
                return self.target_sprite(sprite)
        return None

    def record_click_result(
        self,
        type_id: int,
        had_effect: bool,
    ) -> None:
        """Record whether clicking a sprite type had an effect."""
        self.survey.record(type_id, had_effect)

    def get_selectable_types(self) -> set[int]:
        """Get sprite type IDs that are known to respond to clicks."""
        return self.survey.selectable_types

    def untested_types(self, known_type_ids: set[int]) -> set[int]:
        """Get sprite type IDs that haven't been click-tested yet."""
        return known_type_ids - self.survey.tested_type_ids
