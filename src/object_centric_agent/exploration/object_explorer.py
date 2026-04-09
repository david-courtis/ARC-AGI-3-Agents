"""
Object-centric exploration: phased testing of object interactions.

Phases:
  0 DETECT:            Parse first frame, bootstrap sprite registry
  1 IDENTIFY_PLAYER:   Test directional actions, find what moves
  2 CLICK_SURVEY:      If ACTION6 available, click each type once
  3 EXPLORE:           Priority-driven testing — prefer high exploration_priority cells
  4 SYNTHESIS_READY:   Enough confidence to attempt world model synthesis

The explorer uses the continuous-confidence epistemic model to decide what to test.
Each (type, action) cell has an exploration_priority score. The explorer picks the
highest-priority cell and constructs an action (possibly with repositioning macro)
to test it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..state.object_state import SpriteInstance, WorldState
from ..perception.frame_parser import FrameParser
from ..epistemic.knowledge_state import (
    EpistemicState,
    TransitionRecord,
    ObservationContext,
)
from ..actions.click_action import ClickPlanner
from ..actions.macro_primitives import MacroPlanner, MacroAction

logger = logging.getLogger(__name__)


class ExplorationPhase(Enum):
    DETECT = auto()
    IDENTIFY_PLAYER = auto()
    CLICK_SURVEY = auto()
    EXPLORE = auto()
    SYNTHESIS_READY = auto()


@dataclass
class ActionRequest:
    """What the explorer wants the agent loop to execute next."""
    action_id: int
    data: dict | None = None
    reason: str = ""
    macro: MacroAction | None = None
    is_macro_step: bool = False

    @staticmethod
    def atomic(action_id: int, reason: str = "", data: dict | None = None) -> ActionRequest:
        return ActionRequest(action_id=action_id, reason=reason, data=data)

    @staticmethod
    def from_macro_step(action_id: int, data: dict | None, macro: MacroAction) -> ActionRequest:
        return ActionRequest(
            action_id=action_id, data=data,
            reason=macro.description, macro=macro, is_macro_step=True,
        )


class ObjectExplorer:
    """
    Drives exploration through structured phases using continuous confidence.

    Each turn the agent calls:
        request = explorer.next_action(world_state)
    Then after executing:
        explorer.observe_result(before_world, action_id, after_world)
    """

    # When mean confidence exceeds this, synthesis is viable
    SYNTHESIS_CONFIDENCE_THRESHOLD = 0.4

    def __init__(
        self,
        frame_parser: FrameParser,
        available_action_ids: set[int] | None = None,
    ):
        self.parser = frame_parser
        self.epistemic = EpistemicState()
        self.click_planner = ClickPlanner()
        self.macro_planner = MacroPlanner()

        self._phase = ExplorationPhase.DETECT
        self._available_actions: set[int] = available_action_ids or {1, 2, 3, 4, 5}
        self._has_action6 = 6 in self._available_actions

        # Phase 1 state
        self._directions_to_test: list[int] = []
        self._direction_test_index: int = 0
        self._player_candidates: dict[int, list[tuple[int, int]]] = {}

        # Phase 2 state
        self._click_survey_queue: list[int] = []

        # General
        self._frame_count: int = 0
        self._last_world: WorldState | None = None
        self._pending_macro: MacroAction | None = None

    @property
    def phase(self) -> ExplorationPhase:
        return self._phase

    @property
    def is_synthesis_ready(self) -> bool:
        return self._phase == ExplorationPhase.SYNTHESIS_READY

    def update_available_actions(self, action_ids: set[int]) -> None:
        self._available_actions = action_ids
        self._has_action6 = 6 in action_ids

    def next_action(self, world: WorldState) -> ActionRequest:
        self._last_world = world

        # Continue macro if in progress
        if self._pending_macro and not self._pending_macro.is_done:
            step = self._pending_macro.pop_next()
            if step is not None:
                return ActionRequest.from_macro_step(step[0], step[1], self._pending_macro)

        if self._phase == ExplorationPhase.DETECT:
            return self._handle_detect(world)
        elif self._phase == ExplorationPhase.IDENTIFY_PLAYER:
            return self._handle_identify_player(world)
        elif self._phase == ExplorationPhase.CLICK_SURVEY:
            return self._handle_click_survey(world)
        elif self._phase == ExplorationPhase.EXPLORE:
            return self._handle_explore(world)
        else:
            return self._handle_explore(world)  # SYNTHESIS_READY still explores

    def observe_result(
        self,
        before: WorldState,
        action_id: int,
        after: WorldState,
    ) -> None:
        """Process action result. Builds context-rich TransitionRecords."""
        self._frame_count += 1

        # Find player position for context
        player_pos = None
        player_tid = self.macro_planner._player_type_id
        if player_tid is not None:
            for s in before.sprites:
                if s.type_id == player_tid:
                    player_pos = s.position
                    break

        for after_sprite in after.sprites:
            before_sprite = before.get_sprite_by_track(after_sprite.track_id)
            if before_sprite is None:
                continue

            displacement = after_sprite.displacement_from(before_sprite)
            had_effect = displacement != (0, 0)

            if not had_effect:
                had_effect = after_sprite.colors != before_sprite.colors

            # Build context snapshot
            nearby = []
            for other in before.sprites:
                if other.track_id == before_sprite.track_id:
                    continue
                dist = _distance(other.center, before_sprite.center)
                if dist < 15:  # within interaction range
                    nearby.append({
                        "type_id": other.type_id,
                        "position": other.position,
                    })

            context = ObservationContext.from_world_snapshot(
                target_position=before_sprite.position,
                nearby_sprites=nearby,
                selected_type_id=(
                    before.selected_sprite.type_id if before.selected_sprite else None
                ),
                player_position=player_pos if before_sprite.type_id != player_tid else None,
            )

            record = TransitionRecord(
                frame_index=after.frame_index,
                action_id=action_id,
                sprite_type_id=after_sprite.type_id,
                track_id=after_sprite.track_id,
                before_position=before_sprite.position,
                after_position=after_sprite.position,
                displacement=displacement,
                had_effect=had_effect,
                context=context,
                nearby_types=[s["type_id"] for s in nearby],
                selected_type_id=(
                    before.selected_sprite.type_id if before.selected_sprite else None
                ),
            )

            self.epistemic.record_transition(record)

        # Phase-specific processing
        if self._phase == ExplorationPhase.IDENTIFY_PLAYER:
            self._process_player_identification(before, action_id, after)
        elif self._phase == ExplorationPhase.CLICK_SURVEY:
            self._process_click_result(before, action_id, after)

        # Check if we should promote to synthesis-ready
        self._check_synthesis_readiness()

    def get_epistemic_summary(self) -> str:
        type_names = {tid: s.name for tid, s in self.parser.sprite_registry.types.items()}
        return self.epistemic.describe(type_names)

    # =========================================================================
    # Phase handlers
    # =========================================================================

    def _handle_detect(self, world: WorldState) -> ActionRequest:
        type_ids = list(self.parser.sprite_registry.types.keys())
        action_ids = [a for a in self._available_actions if a <= 5]
        self.epistemic.initialize_for_types(type_ids, action_ids)

        self._directions_to_test = [a for a in [1, 2, 3, 4] if a in self._available_actions]
        self._direction_test_index = 0
        self._phase = ExplorationPhase.IDENTIFY_PLAYER

        logger.info(
            f"[Explorer] DETECT: {len(type_ids)} types × {len(action_ids)} actions "
            f"= {len(type_ids) * len(action_ids)} cells"
        )
        return self._handle_identify_player(world)

    def _handle_identify_player(self, world: WorldState) -> ActionRequest:
        if self._direction_test_index >= len(self._directions_to_test):
            self._resolve_player()
            self._advance_to_click_or_explore(world)
            return self.next_action(world)

        action_id = self._directions_to_test[self._direction_test_index]
        self._direction_test_index += 1
        return ActionRequest.atomic(
            action_id, reason=f"Phase 1: test ACTION{action_id} for player identification",
        )

    def _handle_click_survey(self, world: WorldState) -> ActionRequest:
        if not self._has_action6 or not self._click_survey_queue:
            self._phase = ExplorationPhase.EXPLORE
            return self.next_action(world)

        type_id = self._click_survey_queue[0]
        target = self.click_planner.target_type(type_id, world)
        if target is None:
            self._click_survey_queue.pop(0)
            return self.next_action(world)

        return ActionRequest.atomic(6, data=target.as_data,
                                    reason=f"Phase 2: click-test type {type_id}")

    def _handle_explore(self, world: WorldState) -> ActionRequest:
        """
        Priority-driven exploration: pick the (type, action) cell with
        highest exploration_priority and construct an action to test it.
        """
        targets = self.epistemic.get_exploration_targets()

        if not targets:
            return self._random_action(world)

        # Pick highest priority that we can actually test
        for type_id, action_id, priority in targets:
            if priority < 0.1:
                break  # everything is well-understood

            if action_id not in self._available_actions:
                continue

            # For the player type: just execute the action directly
            if type_id == self.macro_planner._player_type_id:
                return ActionRequest.atomic(
                    action_id,
                    reason=f"explore: ACTION{action_id} on player "
                           f"(priority={priority:.2f})",
                )

            # For non-player types: only navigate to them if they're NOT static.
            # Static types (walls, borders) don't have proximity interactions —
            # their observations come passively from being in the frame.
            sprite_type = self.parser.sprite_registry.get_type(type_id)
            is_static = sprite_type and sprite_type.is_static

            if (
                not is_static
                and self.macro_planner._player_type_id is not None
                and action_id in (1, 2, 3, 4, 5)
            ):
                targets_of_type = world.get_sprites_of_type(type_id)
                if targets_of_type:
                    macro = self.macro_planner.plan_move_to(
                        target_type_id=type_id,
                        world=world,
                        interact_action=action_id if action_id == 5 else None,
                    )
                    if macro and not macro.is_done:
                        self._pending_macro = macro
                        step = macro.pop_next()
                        if step:
                            return ActionRequest.from_macro_step(step[0], step[1], macro)

            # Fallback: just fire the action
            return ActionRequest.atomic(
                action_id,
                reason=f"explore: ACTION{action_id} for type {type_id} "
                       f"(priority={priority:.2f})",
            )

        return self._random_action(world)

    def _random_action(self, world: WorldState) -> ActionRequest:
        """Fallback: least-tested action."""
        action_counts: dict[int, int] = {}
        for eff in self.epistemic.effects.values():
            action_counts[eff.action_id] = (
                action_counts.get(eff.action_id, 0) + eff.observation_count
            )
        valid = [a for a in self._available_actions if a <= 5]
        if not valid:
            return ActionRequest.atomic(1, reason="fallback")
        least = min(valid, key=lambda a: action_counts.get(a, 0))
        return ActionRequest.atomic(least, reason="fallback: least tested")

    # =========================================================================
    # Phase result processors
    # =========================================================================

    def _process_player_identification(
        self, before: WorldState, action_id: int, after: WorldState,
    ) -> None:
        for after_sprite in after.sprites:
            before_sprite = before.get_sprite_by_track(after_sprite.track_id)
            if before_sprite is None:
                continue
            disp = after_sprite.displacement_from(before_sprite)
            if disp != (0, 0):
                tid = after_sprite.type_id
                if tid not in self._player_candidates:
                    self._player_candidates[tid] = []
                self._player_candidates[tid].append(disp)

    def _resolve_player(self) -> None:
        if not self._player_candidates:
            logger.info("[Explorer] No sprite moved — no player detected")
            return

        best_type = max(
            self._player_candidates,
            key=lambda tid: len(self._player_candidates[tid]),
        )
        displacements = self._player_candidates[best_type]

        logger.info(
            f"[Explorer] Player: type {best_type} "
            f"(moved {len(displacements)}x: {displacements})"
        )

        self.parser.mark_sprite_type(best_type, is_player=True, name="player")
        self.macro_planner.set_player_type(best_type)

        for dr, dc in displacements:
            if dr != 0 or dc != 0:
                self.macro_planner.set_step_size(dr, dc)
                break

        # Mark non-moving types as static (informational, not epistemic override)
        for tid in self.parser.sprite_registry.types:
            if tid != best_type and tid not in self._player_candidates:
                self.parser.mark_sprite_type(tid, is_static=True)

    def _process_click_result(
        self, before: WorldState, action_id: int, after: WorldState,
    ) -> None:
        if not self._click_survey_queue:
            return

        type_id = self._click_survey_queue[0]
        had_effect = False

        for after_sprite in after.sprites:
            before_sprite = before.get_sprite_by_track(after_sprite.track_id)
            if before_sprite is None:
                had_effect = True
                break
            if after_sprite.displacement_from(before_sprite) != (0, 0):
                had_effect = True
                break
            if after_sprite.colors != before_sprite.colors:
                had_effect = True
                break

        if not had_effect:
            diff_count = int(np.sum(np.any(before.frame != after.frame, axis=-1)))
            had_effect = diff_count > 0

        self.click_planner.record_click_result(type_id, had_effect)
        if had_effect:
            self.parser.mark_sprite_type(type_id, is_selectable=True)
            logger.info(f"[Explorer] Type {type_id} is SELECTABLE")

        self._click_survey_queue.pop(0)

    # =========================================================================
    # Phase transitions
    # =========================================================================

    def _advance_to_click_or_explore(self, world: WorldState) -> None:
        if self._has_action6:
            player_type = self.macro_planner._player_type_id
            self._click_survey_queue = [
                tid for tid in self.parser.sprite_registry.types
                if tid != player_type
            ]
            self._phase = ExplorationPhase.CLICK_SURVEY
            logger.info(
                f"[Explorer] → CLICK_SURVEY: {len(self._click_survey_queue)} types"
            )
        else:
            self._phase = ExplorationPhase.EXPLORE
            logger.info("[Explorer] → EXPLORE (priority-driven)")

    def _check_synthesis_readiness(self) -> None:
        """Promote to SYNTHESIS_READY when we have enough confidence."""
        if self._phase == ExplorationPhase.SYNTHESIS_READY:
            return
        if self._phase in (ExplorationPhase.DETECT, ExplorationPhase.IDENTIFY_PLAYER):
            return

        mean_conf = self.epistemic.mean_confidence()
        coverage = self.epistemic.coverage_ratio()

        if coverage >= 0.8 and mean_conf >= self.SYNTHESIS_CONFIDENCE_THRESHOLD:
            self._phase = ExplorationPhase.SYNTHESIS_READY
            logger.info(
                f"[Explorer] → SYNTHESIS_READY "
                f"(coverage={coverage:.0%}, confidence={mean_conf:.0%})"
            )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
