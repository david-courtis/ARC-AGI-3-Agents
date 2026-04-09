"""
Object-Centric Agent v2 — perception-driven, epistemic exploration with level-gating.

Level-gating policy:
    The agent does NOT advance to the next level until:
    1. Sufficient epistemic coverage (mean confidence > threshold)
    2. The synthesized world model correctly predicts ALL observed transitions

    If the agent accidentally completes a level (e.g., by pushing a box onto
    a goal during exploration), the level is RESET. The agent is informed that
    the level was completed but is being reset to continue learning.

    Only when both conditions are met does the agent allow level advancement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from arcengine import FrameDataRaw, GameAction, GameState
from src.shared.agent_base import SynthesisAgent
from src.shared import ExplorationEngine
from src.shared.frame_utils import extract_grid
from src.shared.models import ActionID
from src.shared.run_logger import RunLogger

from .perception.frame_parser import FrameParser
from .exploration.object_explorer import ObjectExplorer, ExplorationPhase, ActionRequest
from .epistemic.knowledge_state import EpistemicState
from .synth_loop import (
    ClaudeCodeBackend,
    ReflexionLoop,
    Transition,
)
from .world_model import compute_diff
from .shared_world_model import SharedWorldModel

logger = logging.getLogger(__name__)

INT_TO_GAME_ACTION = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
    4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}

ACTION_ID_MAP = {
    ActionID.ACTION1: 1, ActionID.ACTION2: 2, ActionID.ACTION3: 3,
    ActionID.ACTION4: 4, ActionID.ACTION5: 5, ActionID.ACTION6: 6,
    ActionID.ACTION7: 7,
}


@dataclass
class LevelState:
    """Per-level tracking state."""
    level_index: int
    replay_buffer: list[Transition] = field(default_factory=list)
    action_counts: dict[int, int] = field(default_factory=dict)
    synthesis_attempts: int = 0
    model_is_perfect: bool = False
    level_completed_count: int = 0

    @property
    def total_observations(self) -> int:
        return sum(self.action_counts.values())

    @property
    def best_accuracy(self) -> float:
        # Delegated to the synthesis loop's workspace
        return 0.0  # overridden by the synthesis loop

    def record_transition(
        self, before_frame: np.ndarray, action_id: int, after_frame: np.ndarray,
    ) -> None:
        self.replay_buffer.append(Transition(
            before_frame=before_frame, action_id=action_id,
            after_frame=after_frame, timestep=len(self.replay_buffer),
        ))
        self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1


class ObjectCentricAgentV2(SynthesisAgent):
    """
    Object-centric agent with level-gating.

    The agent stays on each level until:
    - Enough epistemic coverage AND
    - Synthesized world model achieves 100% on observed transitions

    If a level is accidentally completed, it resets.
    """

    MAX_ACTIONS = 500  # generous budget for multi-level games

    # Level-gating thresholds
    MIN_CONFIDENCE_TO_ADVANCE = 0.5
    MIN_TRANSITIONS_FOR_SYNTHESIS = 50   # need enough diversity before attempting synthesis
    MAX_SYNTHESIS_ROUNDS_PER_LEVEL = 10
    MAX_RESETS_PER_LEVEL = 8

    def __init__(
        self,
        *,
        synthesis_backend: str = "claude_code",  # "claude_code" or custom SynthesisBackend
        synthesis_model: str | None = "opus",
        synthesis_max_turns: int = 30,
        exploration_provider: str = "clewdr",
        exploration_model: str = "claude-sonnet-4-6-thinking",
        exploration_reasoning: bool = False,
        results_dir: str = "results",
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self._results_dir = results_dir

        # Perception
        self.frame_parser = FrameParser(min_fragment_size=1)
        self.explorer: ObjectExplorer | None = None

        # LLM exploration engine (for analysis + logging)
        self.engine = ExplorationEngine(
            results_dir=results_dir,
            llm_provider=exploration_provider,
            llm_model=exploration_model,
            reasoning=exploration_reasoning,
            verbose=verbose,
        )

        # Synthesis backend (modular — can be swapped)
        if isinstance(synthesis_backend, str) and synthesis_backend == "claude_code":
            self._backend = ClaudeCodeBackend(
                max_turns=synthesis_max_turns,
                model=synthesis_model,
            )
        elif isinstance(synthesis_backend, str):
            raise ValueError(f"Unknown backend: {synthesis_backend}")
        else:
            self._backend = synthesis_backend  # custom SynthesisBackend instance

        self._synth_loop: ReflexionLoop | None = None  # created per level

        # Shared world model — the single source of truth
        self.world_model: SharedWorldModel | None = None  # created on init

        # Level-gating state
        self._current_level_idx: int = 0
        self._level_state = LevelState(level_index=0)
        self._level_ready_to_advance: bool = False
        self._needs_reset: bool = False  # flag to reset on next choose_action
        self._total_levels_mastered: int = 0

        # Game metadata
        self._win_levels: int = 0  # total levels to win (from API)
        self._goal_reached_this_level: bool = False  # set when levels_completed increases

        # Frame tracking
        self._previous_frame: np.ndarray | None = None
        self._previous_world = None
        self._pending_action_id: int | None = None
        self._pending_action_data: dict | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._frame_index: int = 0
        self._last_levels_completed: int = 0  # to detect level transitions

        self._initialized: bool = False
        self._available_action_ids: set[int] = {1, 2, 3, 4, 5}

    @property
    def run_logger(self) -> RunLogger | None:
        return self.engine.run_logger if self._initialized else None

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.engine.initialize_run(
                disable_logging=bool(os.environ.get("ARC_DISABLE_RUN_LOGGING")),
            )
            # Create shared world model in the results dir
            run_dir = Path(self._results_dir) / f"level_{self._current_level_idx}"
            self.world_model = SharedWorldModel(run_dir)
            self._initialized = True

    def is_done(self, obs: FrameDataRaw) -> bool:
        if obs.state is GameState.WIN:
            logger.info("[Agent] GAME WON!")
            return True
        if self.action_counter >= self.MAX_ACTIONS:
            logger.info(f"[Agent] MAX_ACTIONS ({self.MAX_ACTIONS}) reached")
            return True
        return False

    def choose_action(self, obs: FrameDataRaw) -> GameAction:
        self._ensure_initialized()
        self.pending_action_data = None

        # === Handle reset/game-over states ===
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._previous_frame = None
            self._previous_world = None
            self._pending_action_id = None
            return GameAction.RESET

        # === If we flagged a reset (level completed but not ready) ===
        if self._needs_reset:
            self._needs_reset = False
            self._previous_frame = None
            self._previous_world = None
            self._pending_action_id = None
            logger.info(
                f"[Agent] Resetting level {self._current_level_idx} "
                f"(completed {self._level_state.level_completed_count}x but "
                f"model not ready — accuracy={self._level_state.best_accuracy:.0%})"
            )
            return GameAction.RESET

        # === Update available actions ===
        if obs.available_actions:
            self._update_available_actions(obs.available_actions)
            self.engine.update_available_actions(obs.available_actions)

        # === Extract frame ===
        current_frame = extract_grid(obs.frame)
        if current_frame is None:
            return GameAction.RESET
        if self._frame_shape is None:
            self._frame_shape = current_frame.shape

        # === Track game metadata on first obs ===
        if self._win_levels == 0 and hasattr(obs, 'win_levels') and obs.win_levels > 0:
            self._win_levels = obs.win_levels
            logger.info(f"[Agent] Game has {self._win_levels} levels to win")

        # === Detect level/goal completion ===
        score = obs.levels_completed
        if score > self._last_levels_completed:
            # GOAL REACHED — the level was just completed
            self._goal_reached_this_level = True

            # Record the goal-reaching transition BEFORE handling level change
            if self._pending_action_id is not None and self._previous_frame is not None:
                self._level_state.record_transition(
                    before_frame=self._previous_frame,
                    action_id=self._pending_action_id,
                    after_frame=current_frame,
                )
                logger.info(
                    f"[Agent] GOAL REACHED on level {self._current_level_idx}! "
                    f"(ACTION{self._pending_action_id} caused level completion, "
                    f"score: {self._last_levels_completed} → {score})"
                )

            # Inform the world model about goal achievement
            if self.world_model:
                self.world_model.record_goal_reached(
                    level_index=self._current_level_idx,
                    action_id=self._pending_action_id,
                    transition_count=len(self._level_state.replay_buffer),
                    previous_world=self._previous_world,
                )

            self._last_levels_completed = score
            self._handle_level_completed(score)
            if self._needs_reset:
                return GameAction.RESET

        # === Perception ===
        current_world = self.frame_parser.parse(
            current_frame,
            frame_index=self._frame_index,
            action_id=self._pending_action_id,
        )
        self._frame_index += 1

        # === Initialize explorer on first frame ===
        if self.explorer is None:
            self.explorer = ObjectExplorer(
                frame_parser=self.frame_parser,
                available_action_ids=self._available_action_ids,
            )
            logger.info(
                f"[Agent] Level {self._current_level_idx} — Explorer initialized: "
                f"{current_world.num_sprites} sprites, "
                f"{len(self.frame_parser.sprite_registry.types)} types"
            )

        # === Record transition (skip if goal was just reached — that's a reset frame) ===
        if (
            self._pending_action_id is not None
            and self._previous_frame is not None
            and not self._goal_reached_this_level
        ):
            diff = compute_diff(self._previous_frame, current_frame)
            self._level_state.record_transition(
                before_frame=self._previous_frame,
                action_id=self._pending_action_id,
                after_frame=current_frame,
            )

            if self._previous_world is not None:
                self.explorer.observe_result(
                    self._previous_world,
                    self._pending_action_id,
                    current_world,
                )

            # Sync exploration knowledge to shared world model
            if self.world_model and self.engine.state:
                self.world_model.update_from_exploration(
                    action_knowledge=self.engine.state.action_knowledge,
                    environment=self.engine.state.environment,
                    sprite_registry=self.frame_parser.sprite_registry,
                    epistemic_state=self.explorer.epistemic if self.explorer else None,
                )

            if self.verbose and self._frame_index % 5 == 0:
                logger.info(
                    f"[Agent] L{self._current_level_idx} "
                    f"T#{len(self._level_state.replay_buffer)} "
                    f"ACTION{self._pending_action_id}: {diff.count}px changed"
                )

        # Clear goal flag after processing
        self._goal_reached_this_level = False

        # === Pass win_levels to world model if known ===
        if self.world_model and self._win_levels > 0:
            self.world_model.win_levels = self._win_levels

        # === Attempt synthesis if ready ===
        self._maybe_synthesize(current_frame)

        # === Get next action ===
        request = self.explorer.next_action(current_world)
        game_action = INT_TO_GAME_ACTION.get(request.action_id, GameAction.ACTION1)

        if request.action_id == 6 and request.data:
            self.pending_action_data = request.data

        if not request.is_macro_step:
            try:
                action_id_enum = ActionID(f"ACTION{request.action_id}")
                self.engine.prepare_for_action(action_id_enum, current_frame)
                self.engine.increment_action_count()
            except (ValueError, KeyError):
                pass

        self._previous_frame = current_frame
        self._previous_world = current_world
        self._pending_action_id = request.action_id

        if self.verbose and self._frame_index % 10 == 0:
            hi, lo, un = self.explorer.epistemic.coverage()
            logger.info(
                f"[Agent] L{self._current_level_idx} phase={self.explorer.phase.name} "
                f"ACTION{request.action_id} | "
                f"epistemic: {hi}hi/{lo}lo/{un}un "
                f"conf={self.explorer.epistemic.mean_confidence():.0%} | "
                f"synth: {self._level_state.best_accuracy:.0%} | "
                f"{request.reason[:50]}"
            )

        return game_action

    # =========================================================================
    # Level gating
    # =========================================================================

    def _handle_level_completed(self, new_score: int) -> None:
        """
        Called when the game reports a higher score (level completed).

        Decision: advance or reset?
        """
        ls = self._level_state
        ls.level_completed_count += 1

        logger.info(
            f"[Agent] === LEVEL {self._current_level_idx} COMPLETED "
            f"(attempt #{ls.level_completed_count}) ==="
        )

        # Check if we should advance
        confidence = (
            self.explorer.epistemic.mean_confidence()
            if self.explorer else 0.0
        )
        model_perfect = ls.model_is_perfect
        best_acc = self._synth_loop.best_accuracy if self._synth_loop else 0.0
        enough_obs = len(ls.replay_buffer) >= self.MIN_TRANSITIONS_FOR_SYNTHESIS
        too_many_resets = ls.level_completed_count >= self.MAX_RESETS_PER_LEVEL

        # Require world model convergence: goal understood + model perfect
        wm_converged = self.world_model.is_converged if self.world_model else False
        should_advance = wm_converged or (model_perfect and confidence >= self.MIN_CONFIDENCE_TO_ADVANCE)
        if too_many_resets:
            should_advance = True
            logger.info(
                f"[Agent] Forcing advancement — {ls.level_completed_count} resets "
                f"without achieving perfect model"
            )

        if should_advance:
            # Advance to next level
            self._current_level_idx = new_score
            self._level_state = LevelState(level_index=new_score)
            self._total_levels_mastered += 1
            self._previous_frame = None
            self._previous_world = None
            self._pending_action_id = None

            # Reset perception and synthesis for new level
            self.frame_parser = FrameParser(min_fragment_size=1)
            self.explorer = None
            self._synth_loop = None  # new workspace for new level

            logger.info(
                f"[Agent] >>> ADVANCING to level {self._current_level_idx} "
                f"(model accuracy={best_acc:.0%}, "
                f"confidence={confidence:.0%}, "
                f"mastered={self._total_levels_mastered}) <<<"
            )
        else:
            # Reset: stay on this level, keep learning
            self._needs_reset = True
            self._previous_frame = None
            self._previous_world = None
            self._pending_action_id = None

            reason_parts = []
            if not model_perfect:
                reason_parts.append(
                    f"model accuracy {best_acc:.0%} (need 100%)"
                )
            if confidence < self.MIN_CONFIDENCE_TO_ADVANCE:
                reason_parts.append(
                    f"confidence {confidence:.0%} (need {self.MIN_CONFIDENCE_TO_ADVANCE:.0%})"
                )
            if not enough_obs:
                reason_parts.append(
                    f"only {len(ls.replay_buffer)} transitions (need {self.MIN_TRANSITIONS_FOR_SYNTHESIS})"
                )

            logger.info(
                f"[Agent] Level completed but NOT advancing — "
                f"resetting to continue learning. "
                f"Reason: {'; '.join(reason_parts)}"
            )

            # Trigger synthesis if we haven't already
            if not model_perfect and enough_obs:
                logger.info("[Agent] Triggering synthesis before reset...")
                self._force_synthesize()

    # =========================================================================
    # Synthesis
    # =========================================================================

    def _maybe_synthesize(self, current_frame: np.ndarray) -> None:
        ls = self._level_state
        if ls.model_is_perfect:
            return
        if ls.synthesis_attempts >= self.MAX_SYNTHESIS_ROUNDS_PER_LEVEL:
            return

        ready = False
        if self.explorer and self.explorer.is_synthesis_ready:
            ready = True
        elif (
            len(ls.replay_buffer) >= self.MIN_TRANSITIONS_FOR_SYNTHESIS
            and self.explorer
            and self.explorer.epistemic.coverage_ratio() >= 0.5
        ):
            ready = True

        if not ready:
            return

        self._run_synthesis(current_frame)

    def _force_synthesize(self) -> None:
        """Force a synthesis attempt (called on accidental level completion)."""
        if self._previous_frame is not None:
            self._run_synthesis(self._previous_frame)

    def _run_synthesis(self, current_frame: np.ndarray) -> None:
        ls = self._level_state

        # Create synthesis loop for this level if needed
        if self._synth_loop is None:
            workspace_dir = (
                Path(self._results_dir)
                / f"level_{self._current_level_idx}"
                / "synthesis"
            )
            self._synth_loop = ReflexionLoop(
                backend=self._backend,
                workspace_dir=workspace_dir,
            )

        ls.synthesis_attempts += 1

        logger.info(
            f"[Agent] Synthesis run #{ls.synthesis_attempts} "
            f"for level {self._current_level_idx}, "
            f"{len(ls.replay_buffer)} transitions"
        )

        structured_analysis = self._build_structured_analysis(current_frame)

        # Get the initial frame for perceive() bootstrapping
        initial_frame = ls.replay_buffer[0].before_frame if ls.replay_buffer else current_frame

        success = self._synth_loop.run(
            replay_buffer=ls.replay_buffer,
            structured_analysis=structured_analysis,
            initial_frame=initial_frame,
        )

        best_acc = self._synth_loop.best_accuracy

        if success:
            ls.model_is_perfect = True
            logger.info(
                f"[Agent] PERFECT model for level {self._current_level_idx}! "
                f"(accuracy=100%, code at {self._synth_loop.workspace.code_path})"
            )
            # Write definitive feedback to world model
            if self.world_model:
                code = self._synth_loop.current_code or ""
                self.world_model.update_from_synthesis(
                    accuracy=1.0,
                    confirmed_facts=[
                        "All observed transitions are correctly predicted by the code.",
                    ],
                    code_summary=f"Transition rules ({len(code)} chars) covering "
                                 f"{len(ls.replay_buffer)} transitions",
                )
        else:
            logger.info(
                f"[Agent] Synthesis: best {best_acc:.0%}, "
                f"code at {self._synth_loop.workspace.code_path}"
            )
            # Write tentative feedback + questions to world model
            if self.world_model:
                self.world_model.update_from_synthesis(
                    accuracy=best_acc,
                    tentative_findings=[
                        f"Code achieves {best_acc:.0%} accuracy on {len(ls.replay_buffer)} transitions",
                    ],
                    questions=[
                        f"Synthesis stuck at {best_acc:.0%} — need more diverse observations "
                        f"to disambiguate transition rules",
                    ],
                )

    def _build_structured_analysis(self, current_frame: np.ndarray) -> str:
        """
        Build context for the synthesis backend.

        Uses the shared world model as primary context (contains both
        LLM exploration insights AND deterministic observations).
        Appends sample transitions with object-level diffs.
        """
        parts = [f"Frame shape: {current_frame.shape}"]

        # Primary context: the shared world model
        if self.world_model:
            parts.append(self.world_model.to_markdown())
        else:
            # Fallback: raw registry + epistemic
            parts.append(f"\n== DETECTED OBJECT TYPES ==")
            parts.append(self.frame_parser.get_registry_summary())

            if self.explorer:
                parts.append(f"\n== EPISTEMIC STATE ==")
                parts.append(self.explorer.get_epistemic_summary())

        # Add sample transitions with OBJECT-LEVEL state changes
        # This is the key data: actual before/after object positions
        ls = self._level_state
        if ls.replay_buffer:
            parts.append(f"\n== SAMPLE TRANSITIONS (object states) ==")
            sample_parser = FrameParser(min_fragment_size=1)
            for trans in ls.replay_buffer[:12]:
                before_ws = sample_parser.parse(trans.before_frame, trans.timestep * 2)
                after_ws = sample_parser.parse(
                    trans.after_frame, trans.timestep * 2 + 1,
                    action_id=trans.action_id,
                )
                parts.append(f"\n  T{trans.timestep} ACTION{trans.action_id}:")
                parts.append(f"    Before objects:")
                for s in before_ws.sprites:
                    parts.append(
                        f"      {s.type_name} (track={s.track_id}) "
                        f"at row={s.position[0]}, col={s.position[1]}"
                    )
                parts.append(f"    After objects:")
                for s in after_ws.sprites:
                    b = before_ws.get_sprite_by_track(s.track_id)
                    moved = ""
                    if b:
                        d = s.displacement_from(b)
                        if d != (0, 0):
                            moved = f" *** MOVED by ({d[0]},{d[1]}) ***"
                    parts.append(
                        f"      {s.type_name} (track={s.track_id}) "
                        f"at row={s.position[0]}, col={s.position[1]}{moved}"
                    )

        parts.append(f"\nBackground color: RGB{self.frame_parser.background_color}")
        return "\n".join(parts)

    def _update_available_actions(self, api_actions: list[Any]) -> None:
        ids = set()
        for ga in api_actions:
            value = ga.value if hasattr(ga, "value") else ga
            if isinstance(value, int) and value >= 1:
                ids.add(value)
        if ids:
            self._available_action_ids = ids
            if self.explorer:
                self.explorer.update_available_actions(ids)
