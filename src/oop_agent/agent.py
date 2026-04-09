"""
OOP World Model Agent — the primary research agent.

This agent synthesizes polymorphic, object-centric world models from raw pixel
observations using LLM-driven exploration + CEGIS program synthesis.

Pipeline
--------
The agent uses the shared ExplorationEngine for LLM-driven exploration:

1. **Explore**: LLM analyzes actions (before/after frames), understands the
   environment, and suggests the next action. All transitions are recorded
   into a replay buffer.

2. **Synthesize**: When enough transitions are collected, call the synthesis LLM
   to generate a SynthesizedDomain (object classes + action classes + perceive).
   Verify against the replay buffer via symbolic execution.
   The verification accuracy is the FEEDBACK SIGNAL — it replaces asking
   an LLM whether the representation is accurate, because the code can
   be executed deterministically against every transition.

   If errors, refine via CEGIS (Counterexample-Guided Inductive Synthesis):
   feed mismatches back to the LLM and repeat. Then explore more and
   re-synthesize with a larger replay buffer.

The agent does NOT attempt to solve games or reach goals. Its sole purpose
is to build an accurate world model. Success = 100% verification accuracy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arcengine import FrameDataRaw, GameAction, GameState
from src.shared.agent_base import SynthesisAgent
from src.shared import ExplorationEngine
from src.shared.frame_utils import extract_grid
from src.shared.models import ActionID
from src.shared.run_logger import RunLogger
from .synthesis import (
    DomainSynthesizer,
    Transition,
    VerificationResult,
    diagnose_persistent_errors,
    verify_domain,
)
from .world_model import (
    Domain,
    compute_diff,
    find_unique_colors,
    most_common_color,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Synthesis-specific state (replay buffer, domain, etc.)
# ============================================================================

@dataclass
class SynthesisState:
    """Tracks synthesis-specific state: replay buffer and CEGIS results."""
    action_counts: dict[int, int] = field(default_factory=lambda: {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    })
    action_observations: dict[int, list[str]] = field(default_factory=lambda: {
        1: [], 2: [], 3: [], 4: [], 5: [],
    })
    replay_buffer: list[Transition] = field(default_factory=list)

    synthesis_attempts: int = 0
    last_verification: VerificationResult | None = None
    best_accuracy: float = 0.0

    current_domain: Domain | None = None
    current_code: str | None = None
    model_is_perfect: bool = False

    min_observations_per_action: int = 3
    max_observations_per_action: int = 10

    current_score: int = 0
    max_score: int = 0
    score_history: list[int] = field(default_factory=list)

    available_action_ids: set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5})

    @property
    def total_observations(self) -> int:
        return sum(self.action_counts.values())

    @property
    def least_observed_action(self) -> int:
        valid = {k: v for k, v in self.action_counts.items()
                 if k in self.available_action_ids}
        return min(valid, key=valid.get)

    @property
    def has_minimum_coverage(self) -> bool:
        return all(
            self.action_counts.get(a, 0) >= self.min_observations_per_action
            for a in self.available_action_ids
        )

    def record_transition(
        self, before_frame: np.ndarray, action_id: int,
        after_frame: np.ndarray, observation_text: str,
    ) -> None:
        self.replay_buffer.append(Transition(
            before_frame=before_frame, action_id=action_id,
            after_frame=after_frame, timestep=len(self.replay_buffer),
        ))
        self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1
        if action_id not in self.action_observations:
            self.action_observations[action_id] = []
        self.action_observations[action_id].append(observation_text)


# Keep for backward compatibility
ExplorationState = SynthesisState


def choose_exploration_action(state: SynthesisState) -> int:
    """Deterministic exploration fallback: untested > under-observed > least-observed."""
    untested = [a for a in sorted(state.available_action_ids)
                if state.action_counts.get(a, 0) == 0]
    if untested:
        return untested[0]

    under_observed = [a for a in sorted(state.available_action_ids)
                      if state.action_counts.get(a, 0) < state.min_observations_per_action]
    if under_observed:
        return min(under_observed, key=lambda a: state.action_counts.get(a, 0))

    return state.least_observed_action


def summarize_transition(before: np.ndarray, after: np.ndarray, action_id: int) -> str:
    diff = compute_diff(before, after)
    if diff.count == 0:
        return f"ACTION{action_id}: no visible change"

    parts = [f"ACTION{action_id}: {diff.count} pixels changed"]
    if diff.bbox:
        parts.append(f"  region: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")

    color_changes: dict[tuple, int] = {}
    for bc, ac in zip(diff.before_colors[:50], diff.after_colors[:50]):
        color_changes[(bc, ac)] = color_changes.get((bc, ac), 0) + 1
    for (from_c, to_c), count in sorted(color_changes.items(), key=lambda x: -x[1])[:5]:
        parts.append(f"  {count}x: {from_c} -> {to_c}")

    return "\n".join(parts)


def analyze_frame(frame: np.ndarray) -> str:
    parts = [f"Frame shape: {frame.shape}"]

    colors = find_unique_colors(frame)
    parts.append(f"Unique colors: {len(colors)}")
    if len(colors) <= 20:
        for c in colors:
            if frame.ndim == 3:
                mask = np.all(frame == np.array(c, dtype=np.uint8), axis=-1)
            else:
                mask = frame == c[0]
            count = int(np.sum(mask))
            parts.append(f"  RGB{c}: {count} pixels ({count / frame.shape[0] / frame.shape[1] * 100:.1f}%)")

    bg = most_common_color(frame)
    parts.append(f"Most common color (likely background): RGB{bg}")

    if frame.ndim == 3:
        top_row = frame[0, :, :]
        bottom_row = frame[-1, :, :]
        left_col = frame[:, 0, :]
        right_col = frame[:, -1, :]

        borders = []
        if np.all(top_row == top_row[0]):
            borders.append(f"  Top: RGB{tuple(int(x) for x in top_row[0])}")
        if np.all(bottom_row == bottom_row[0]):
            borders.append(f"  Bottom: RGB{tuple(int(x) for x in bottom_row[0])}")
        if np.all(left_col == left_col[0]):
            borders.append(f"  Left: RGB{tuple(int(x) for x in left_col[0])}")
        if np.all(right_col == right_col[0]):
            borders.append(f"  Right: RGB{tuple(int(x) for x in right_col[0])}")
        if borders:
            parts.append("Border structure:")
            parts.extend(borders)

    return "\n".join(parts)


# ============================================================================
# Conversion helpers
# ============================================================================

ACTION_ID_MAP = {
    ActionID.ACTION1: 1, ActionID.ACTION2: 2, ActionID.ACTION3: 3,
    ActionID.ACTION4: 4, ActionID.ACTION5: 5, ActionID.ACTION6: 6,
    ActionID.ACTION7: 7,
}

INT_TO_ACTION_ID = {v: k for k, v in ACTION_ID_MAP.items()}

INT_TO_GAME_ACTION = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
    4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}

ACTION_ID_TO_GAME_ACTION = {
    ActionID.ACTION1: GameAction.ACTION1,
    ActionID.ACTION2: GameAction.ACTION2,
    ActionID.ACTION3: GameAction.ACTION3,
    ActionID.ACTION4: GameAction.ACTION4,
    ActionID.ACTION5: GameAction.ACTION5,
    ActionID.ACTION6: GameAction.ACTION6,
    ActionID.ACTION7: GameAction.ACTION7,
}


class OOPAgent(SynthesisAgent):
    """
    Primary research agent: synthesizes polymorphic OOP world models.

    Uses the shared ExplorationEngine for LLM-driven exploration (action analysis,
    environment analysis, next action suggestion) and adds program synthesis with
    CEGIS verification as a feedback signal.
    """

    MAX_ACTIONS = 30

    def __init__(
        self,
        *,
        llm_base_url: str = "https://clewdr.wavycats.com/code/v1",
        llm_api_key: str = "LkY56yDjUVYkfL9BPm7VLTpMs7kM6gbXPMp6PV2QysqcAvpr8PAdyjPYYbbgTwgH",
        llm_model: str = "claude-sonnet-4-6-thinking",
        max_refinements: int = 3,
        exploration_provider: str = "clewdr",
        exploration_model: str = "claude-sonnet-4-6-thinking",
        exploration_reasoning: bool = False,
        results_dir: str = "results",
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.synthesis = SynthesisState()

        # The shared exploration engine handles all LLM-driven exploration
        self.engine = ExplorationEngine(
            results_dir=results_dir,
            llm_provider=exploration_provider,
            llm_model=exploration_model,
            reasoning=exploration_reasoning,
            verbose=verbose,
        )

        # Synthesis LLM (separate from exploration LLM)
        self.synthesizer = DomainSynthesizer(
            base_url=llm_base_url, api_key=llm_api_key, model=llm_model,
            max_refinements=max_refinements,
        )

        self._previous_frame: np.ndarray | None = None
        self._pending_action_id: int | None = None
        self._frame_shape: tuple[int, ...] | None = None

        self._phase: str = "explore"
        self._initialized: bool = False

        self._min_transitions_for_synthesis = 10
        self._max_synthesis_rounds = 10

    # Keep backward compatibility
    @property
    def exploration(self) -> SynthesisState:
        return self.synthesis

    @property
    def run_logger(self) -> RunLogger | None:
        return self.engine.run_logger if self._initialized else None

    def _ensure_initialized(self) -> None:
        """Initialize the exploration engine on first use."""
        if not self._initialized:
            self.engine.initialize_run(
                disable_logging=bool(os.environ.get("ARC_DISABLE_RUN_LOGGING")),
            )
            self._initialized = True

    def is_done(self, obs: FrameDataRaw) -> bool:
        done = False
        outcome = "INCOMPLETE"

        if obs.state is GameState.WIN:
            done, outcome = True, "WIN"
        elif self.synthesis.model_is_perfect:
            logger.info(
                f"[OOP] World model is PERFECT at {self.synthesis.best_accuracy:.0%} "
                f"accuracy after {self.synthesis.synthesis_attempts} synthesis rounds "
                f"and {self.synthesis.total_observations} observations"
            )
            done, outcome = True, "PERFECT_MODEL"
        elif self.action_counter >= self.MAX_ACTIONS:
            done, outcome = True, "MAX_ACTIONS"

        if done:
            self.engine.generate_final_report()

        return done

    def choose_action(self, obs: FrameDataRaw) -> GameAction:
        # Initialize on first call
        self._ensure_initialized()

        # Handle game states that require reset
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._previous_frame = None
            self._pending_action_id = None
            return GameAction.RESET

        # Update available actions
        if obs.available_actions:
            self._update_available_actions(obs.available_actions)
            self.engine.update_available_actions(obs.available_actions)

        # Extract frame
        current_frame = extract_grid(obs.frame)
        if current_frame is None:
            return GameAction.RESET

        if self._frame_shape is None:
            self._frame_shape = current_frame.shape

        # Track score (levels_completed replaces the old score field)
        score = obs.levels_completed
        self.synthesis.current_score = score
        self.synthesis.max_score = max(self.synthesis.max_score, score)
        self.synthesis.score_history.append(score)
        self.engine.record_score(score)

        # ================================================================
        # Record transition into replay buffer (for synthesis)
        # ================================================================
        if self._pending_action_id is not None and self._previous_frame is not None:
            obs_text = summarize_transition(
                self._previous_frame, current_frame, self._pending_action_id
            )
            self.synthesis.record_transition(
                before_frame=self._previous_frame,
                action_id=self._pending_action_id,
                after_frame=current_frame,
                observation_text=obs_text,
            )
            if self.verbose:
                logger.info(
                    f"[OOP] Transition #{len(self.synthesis.replay_buffer)} "
                    f"(ACTION{self._pending_action_id}): {obs_text.split(chr(10))[0]}"
                )

        # ================================================================
        # PHASE 1: Complete pending action analysis (LLM exploration)
        # ================================================================
        if self.engine.has_pending_analysis():
            analysis = self.engine.complete_action_analysis(current_frame)
            # Sync synthesis accuracy into engine state
            if self.engine.state:
                self.engine.state.synthesis_attempts = self.synthesis.synthesis_attempts
                self.engine.state.synthesis_accuracy = self.synthesis.best_accuracy
                self.engine.state.model_is_perfect = self.synthesis.model_is_perfect

        # ================================================================
        # PHASE 2: Handle pending setup sequence
        # ================================================================
        if self.engine.has_pending_setup():
            next_setup, target = self.engine.get_pending_setup_action()

            if next_setup is not None:
                # Execute setup action
                game_action = ACTION_ID_TO_GAME_ACTION[next_setup]
                self.engine.advance_setup()
                self.engine.increment_action_count()

                self._previous_frame = current_frame
                self._pending_action_id = ACTION_ID_MAP[next_setup]

                return game_action
            else:
                # Setup complete, execute target
                game_action = ACTION_ID_TO_GAME_ACTION[target]

                self.engine.prepare_for_action(target, current_frame)
                self.engine.advance_setup()
                self.engine.increment_action_count()

                self._previous_frame = current_frame
                self._pending_action_id = ACTION_ID_MAP[target]

                return game_action

        # ================================================================
        # PHASE 3: Check if synthesis should run
        # ================================================================
        self._update_phase()

        if self._phase == "synthesize":
            self._do_synthesis(current_frame)
            # After synthesis, fall through to get next exploration action

        # ================================================================
        # PHASE 4: Get next action from exploration engine (LLM-driven)
        # ================================================================
        suggestion = self.engine.suggest_next_action(current_frame)

        # Handle setup sequences
        if suggestion.setup_sequence:
            # First setup action
            setup_action = suggestion.setup_sequence[0]
            game_action = ACTION_ID_TO_GAME_ACTION[setup_action]

            self.engine.increment_action_count()
            self._previous_frame = current_frame
            self._pending_action_id = ACTION_ID_MAP[setup_action]

            return game_action

        # Execute target action
        target = suggestion.target_action
        game_action = ACTION_ID_TO_GAME_ACTION[target]

        # Prepare for analysis on next turn
        self.engine.prepare_for_action(target, current_frame)
        self.engine.increment_action_count()

        self._previous_frame = current_frame
        self._pending_action_id = ACTION_ID_MAP[target]

        return game_action

    def _update_phase(self) -> None:
        if self.synthesis.model_is_perfect:
            return

        # Require at least 3 new transitions since last synthesis attempt
        replay_size = len(self.synthesis.replay_buffer)
        last_size = getattr(self, '_last_synthesis_replay_size', 0)
        has_new_data = replay_size >= last_size + 3

        if (
            has_new_data
            and self.synthesis.has_minimum_coverage
            and replay_size >= self._min_transitions_for_synthesis
            and self.synthesis.synthesis_attempts < self._max_synthesis_rounds
        ):
            self._phase = "synthesize"
            return

        self._phase = "explore"

    def _do_synthesis(self, current_frame: np.ndarray) -> None:
        if self.verbose:
            logger.info(
                f"[OOP] Synthesis attempt {self.synthesis.synthesis_attempts + 1}, "
                f"{len(self.synthesis.replay_buffer)} transitions"
            )

        frame_analysis = analyze_frame(current_frame)

        # Use the shared run_logger for synthesis logging
        run_logger = self.engine.run_logger

        domain, code, result = self.synthesizer.synthesize(
            replay_buffer=self.synthesis.replay_buffer,
            frame_analysis=frame_analysis,
            action_observations=self.synthesis.action_observations,
            frame_shape=self._frame_shape,
            run_logger=run_logger,
            agent_state=self.engine.state,
        )

        self.synthesis.synthesis_attempts += 1
        self._last_synthesis_replay_size = len(self.synthesis.replay_buffer)

        if domain is not None and result is not None:
            self.synthesis.current_domain = domain
            self.synthesis.current_code = code
            self.synthesis.last_verification = result
            if result.accuracy > self.synthesis.best_accuracy:
                self.synthesis.best_accuracy = result.accuracy

            # Sync to engine state
            if self.engine.state:
                self.engine.state.synthesis_attempts = self.synthesis.synthesis_attempts
                self.engine.state.synthesis_accuracy = self.synthesis.best_accuracy
                self.engine.state.model_is_perfect = result.is_perfect

            if result.is_perfect:
                self.synthesis.model_is_perfect = True
            else:
                diagnosis = diagnose_persistent_errors(
                    self.synthesis.replay_buffer, domain,
                    self.synthesis.synthesis_attempts,
                )
                if diagnosis.is_stuck:
                    self.synthesis.min_observations_per_action += 2

        self._phase = "explore"

    def _update_available_actions(self, api_actions: list[Any]) -> None:
        """Update available action IDs from the API response.

        api_actions can be list[int] (new arc-agi) or list[GameAction] (old vendor).
        """
        ids = set()
        for ga in api_actions:
            value = ga.value if hasattr(ga, "value") else ga
            if isinstance(value, int) and value >= 1 and value != 0:  # 0 is RESET
                ids.add(value)
        if ids:
            self.synthesis.available_action_ids = ids
            for aid in ids:
                if aid not in self.synthesis.action_counts:
                    self.synthesis.action_counts[aid] = 0
                if aid not in self.synthesis.action_observations:
                    self.synthesis.action_observations[aid] = []
