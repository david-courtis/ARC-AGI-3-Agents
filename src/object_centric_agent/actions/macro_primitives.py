"""
Macro primitives: composed multi-step actions for object interaction.

The key primitive is "Move to sprite then act" — compose directional actions
(ACTION1-4) to navigate a cursor/player to a target sprite, then execute
an action on it (ACTION5 interact, ACTION6 click, etc).

This is essential for ARC-3 where most games require:
    1. Navigate to an object (using directional movement)
    2. Select/interact with it (click or ACTION5)
    3. Observe what changes

The macro planner uses the perception layer to:
    - Identify the player/cursor sprite (the one that moves with directional actions)
    - Compute a path to the target sprite using BFS
    - Emit a sequence of atomic actions to execute the path

BFS pathfinding operates on the grid using known wall/obstacle positions
from the epistemic state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..state.object_state import SpriteInstance, SpriteType, WorldState


class MacroStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MacroAction:
    """
    A high-level action that decomposes into a sequence of atomic actions.

    The agent loop pops actions from `remaining_steps` one at a time.
    """
    name: str
    description: str
    target_type_id: int | None = None
    target_track_id: int | None = None

    # The decomposed atomic action sequence.
    # Each entry is (action_id, data_dict_or_None).
    # action_id: 1-7, data: {"x": int, "y": int} for ACTION6, else None.
    remaining_steps: list[tuple[int, dict | None]] = field(default_factory=list)
    total_steps: int = 0

    status: MacroStatus = MacroStatus.PENDING

    @property
    def is_done(self) -> bool:
        return not self.remaining_steps or self.status in (
            MacroStatus.COMPLETED, MacroStatus.FAILED
        )

    def pop_next(self) -> tuple[int, dict | None] | None:
        """Pop and return the next atomic action, or None if done."""
        if not self.remaining_steps:
            self.status = MacroStatus.COMPLETED
            return None
        if self.status == MacroStatus.PENDING:
            self.status = MacroStatus.IN_PROGRESS
        return self.remaining_steps.pop(0)


@dataclass
class MoveToSprite:
    """
    The core macro: move the player/cursor to a target sprite, then act.

    Decomposes into:
        [directional moves to reach target] + [interact action]

    The path is computed by BFS on the grid. Obstacles are sprite types
    marked as static/blocking in the epistemic state.
    """
    target_type_id: int
    target_track_id: int | None = None
    interact_action: int | None = None  # ACTION5 (5) or ACTION6 (6), or None for just navigate
    interact_data: dict | None = None   # {"x": ..., "y": ...} if ACTION6


# Direction constants: action_id → (dr, dc)
DIRECTION_MAP = {
    1: (-1, 0),   # ACTION1 = up
    2: (1, 0),    # ACTION2 = down
    3: (0, -1),   # ACTION3 = left
    4: (0, 1),    # ACTION4 = right
}

# Reverse: (dr, dc) → action_id
REVERSE_DIRECTION = {v: k for k, v in DIRECTION_MAP.items()}


class MacroPlanner:
    """
    Plans and manages macro actions.

    Usage:
        planner = MacroPlanner()

        # Set up knowledge of which types block movement
        planner.set_blocking_types({wall_type_id, border_type_id})
        planner.set_player_type(player_type_id)

        # Plan a move-to-sprite
        macro = planner.plan_move_to(
            target_type_id=goal_type_id,
            world=current_world_state,
            interact_action=5,  # ACTION5 after arriving
        )

        # Execute step by step
        while not macro.is_done:
            step = macro.pop_next()
            # ... execute step in environment ...
    """

    def __init__(self):
        self._player_type_id: int | None = None
        self._blocking_type_ids: set[int] = set()
        self._current_macro: MacroAction | None = None

        # Movement step size (how many pixels the player moves per action).
        # Discovered from epistemic state. Default assumes 1-cell grid movement.
        self._step_size: tuple[int, int] = (1, 1)  # (row_step, col_step)

        # Grid cell size (for converting pixel positions to grid coords).
        # If the game uses a visible grid, this is the cell size in pixels.
        # None means we operate in raw pixel coordinates.
        self._cell_size: tuple[int, int] | None = None

    @property
    def has_active_macro(self) -> bool:
        return self._current_macro is not None and not self._current_macro.is_done

    @property
    def current_macro(self) -> MacroAction | None:
        return self._current_macro

    def set_player_type(self, type_id: int) -> None:
        self._player_type_id = type_id

    def set_blocking_types(self, type_ids: set[int]) -> None:
        self._blocking_type_ids = type_ids

    def set_step_size(self, dr: int, dc: int) -> None:
        """Set the player's movement step size (from epistemic state)."""
        self._step_size = (abs(dr) if dr != 0 else 1, abs(dc) if dc != 0 else 1)

    def set_cell_size(self, cell_h: int, cell_w: int) -> None:
        """Set grid cell size for coordinate conversion."""
        self._cell_size = (cell_h, cell_w)

    def plan_move_to(
        self,
        target_type_id: int,
        world: WorldState,
        target_track_id: int | None = None,
        interact_action: int | None = None,
        interact_data: dict | None = None,
    ) -> MacroAction | None:
        """
        Plan a macro to move the player to a target sprite and optionally interact.

        Args:
            target_type_id: Sprite type to navigate to.
            world: Current world state.
            target_track_id: Specific instance to target (if multiple exist).
            interact_action: Action to execute after arriving (5 or 6), or None.
            interact_data: Data for ACTION6 (computed automatically if None).

        Returns:
            MacroAction with the step sequence, or None if no path exists.
        """
        if self._player_type_id is None:
            return None

        # Find player
        player_sprites = world.get_sprites_of_type(self._player_type_id)
        if not player_sprites:
            return None
        player = player_sprites[0]

        # Find target
        if target_track_id is not None:
            target = world.get_sprite_by_track(target_track_id)
        else:
            targets = world.get_sprites_of_type(target_type_id)
            if not targets:
                return None
            # Pick closest target
            target = min(
                targets,
                key=lambda s: (
                    (s.center[0] - player.center[0]) ** 2
                    + (s.center[1] - player.center[1]) ** 2
                ),
            )

        if target is None:
            return None

        # Convert to grid coordinates if cell size is known
        start = self._to_grid(player.position)
        goal = self._to_grid(target.position)

        # Build obstacle map
        obstacles = self._build_obstacle_set(world)

        # BFS pathfinding
        path = self._bfs(start, goal, obstacles, world.frame.shape[:2])

        if path is None:
            # No path — try to get adjacent instead of exact position
            path = self._bfs_adjacent(start, goal, obstacles, world.frame.shape[:2])

        if path is None:
            return MacroAction(
                name=f"move_to_{target.type_name}",
                description=f"No path to {target.type_name} at {target.position}",
                target_type_id=target_type_id,
                target_track_id=target.track_id,
                status=MacroStatus.FAILED,
            )

        # Convert path to action sequence
        steps: list[tuple[int, dict | None]] = []
        for i in range(1, len(path)):
            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]
            # Normalize to unit direction
            dr_norm = (1 if dr > 0 else -1) if dr != 0 else 0
            dc_norm = (1 if dc > 0 else -1) if dc != 0 else 0
            action_id = REVERSE_DIRECTION.get((dr_norm, dc_norm))
            if action_id is not None:
                steps.append((action_id, None))

        # Add interact action at the end
        if interact_action is not None:
            if interact_action == 6 and interact_data is None:
                # Auto-compute click coordinates from target center
                interact_data = {
                    "x": max(0, min(63, int(target.center[1]))),
                    "y": max(0, min(63, int(target.center[0]))),
                }
            steps.append((interact_action, interact_data))

        macro = MacroAction(
            name=f"move_to_{target.type_name}",
            description=(
                f"Navigate to {target.type_name} at {target.position} "
                f"({len(steps)} steps)"
            ),
            target_type_id=target_type_id,
            target_track_id=target.track_id,
            remaining_steps=steps,
            total_steps=len(steps),
        )

        self._current_macro = macro
        return macro

    def plan_click_on(
        self,
        target_type_id: int,
        world: WorldState,
        target_track_id: int | None = None,
    ) -> MacroAction | None:
        """
        Plan a direct ACTION6 click on a target (no movement, just click).

        Useful when no player navigation is needed (e.g., point-and-click games).
        """
        if target_track_id is not None:
            target = world.get_sprite_by_track(target_track_id)
        else:
            targets = world.get_sprites_of_type(target_type_id)
            target = targets[0] if targets else None

        if target is None:
            return None

        data = {
            "x": max(0, min(63, int(target.center[1]))),
            "y": max(0, min(63, int(target.center[0]))),
        }

        macro = MacroAction(
            name=f"click_{target.type_name}",
            description=f"Click on {target.type_name} at ({data['x']}, {data['y']})",
            target_type_id=target_type_id,
            target_track_id=target.track_id,
            remaining_steps=[(6, data)],
            total_steps=1,
        )

        self._current_macro = macro
        return macro

    def cancel(self) -> None:
        """Cancel the current macro."""
        if self._current_macro is not None:
            self._current_macro.status = MacroStatus.FAILED
            self._current_macro = None

    # =========================================================================
    # Pathfinding internals
    # =========================================================================

    def _to_grid(self, position: tuple[int, int]) -> tuple[int, int]:
        """Convert pixel position to grid coordinates."""
        if self._cell_size is not None:
            return (
                position[0] // self._cell_size[0],
                position[1] // self._cell_size[1],
            )
        return position

    def _build_obstacle_set(self, world: WorldState) -> set[tuple[int, int]]:
        """Build set of grid cells occupied by blocking sprites."""
        obstacles: set[tuple[int, int]] = set()
        for sprite in world.sprites:
            if sprite.sprite_type.type_id in self._blocking_type_ids:
                if self._cell_size is not None:
                    # Add the grid cell of the sprite's position
                    obstacles.add(self._to_grid(sprite.position))
                else:
                    # Add all pixel positions
                    for r, c in sprite.all_pixels:
                        obstacles.add((r, c))
        return obstacles

    def _bfs(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
        frame_shape: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """BFS from start to goal, avoiding obstacles. Returns path or None."""
        if start == goal:
            return [start]

        max_r, max_c = frame_shape
        if self._cell_size is not None:
            max_r = max_r // self._cell_size[0]
            max_c = max_c // self._cell_size[1]

        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
        queue.append((start, [start]))
        visited: set[tuple[int, int]] = {start}

        # Limit search depth to prevent explosion
        max_depth = max_r * max_c

        while queue and len(visited) < max_depth:
            pos, path = queue.popleft()

            for dr, dc in DIRECTION_MAP.values():
                nr, nc = pos[0] + dr, pos[1] + dc

                if nr < 0 or nr >= max_r or nc < 0 or nc >= max_c:
                    continue
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in obstacles:
                    continue

                new_path = path + [(nr, nc)]

                if (nr, nc) == goal:
                    return new_path

                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))

        return None

    def _bfs_adjacent(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
        frame_shape: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """BFS to any cell adjacent to goal (for when goal itself is occupied)."""
        max_r, max_c = frame_shape
        if self._cell_size is not None:
            max_r = max_r // self._cell_size[0]
            max_c = max_c // self._cell_size[1]

        adjacent_goals = set()
        for dr, dc in DIRECTION_MAP.values():
            nr, nc = goal[0] + dr, goal[1] + dc
            if 0 <= nr < max_r and 0 <= nc < max_c and (nr, nc) not in obstacles:
                adjacent_goals.add((nr, nc))

        if not adjacent_goals:
            return None

        # BFS to any adjacent cell
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
        queue.append((start, [start]))
        visited: set[tuple[int, int]] = {start}
        max_depth = max_r * max_c

        while queue and len(visited) < max_depth:
            pos, path = queue.popleft()

            if pos in adjacent_goals:
                return path

            for dr, dc in DIRECTION_MAP.values():
                nr, nc = pos[0] + dr, pos[1] + dc

                if nr < 0 or nr >= max_r or nc < 0 or nc >= max_c:
                    continue
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in obstacles and (nr, nc) not in adjacent_goals:
                    continue

                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

        return None
