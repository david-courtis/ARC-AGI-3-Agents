"""
OOP World Model Agent - Main Agent Implementation.

This agent learns game mechanics by synthesizing world models from raw observations:
1. Explores the environment by taking actions and recording frame transitions
2. Analyzes frames to build a structural understanding (colors, regions, diffs)
3. Synthesizes a WorldModel subclass via LLM (CEGIS loop)
4. Verifies the model against the replay buffer
5. Uses the verified model for planning / score-maximizing play

The agent assumes NOTHING about the domain. Frames are raw 64x64 RGB pixel grids.
Actions are anonymous integers (ACTION1-5). Everything else is inferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...agent import Agent
from ...structs import FrameData, GameAction, GameState
from .synthesis import (
    CounterexampleDiagnosis,
    ModelSynthesizer,
    Transition,
    VerificationResult,
    diagnose_persistent_errors,
    verify_model,
)
from .world_model import (
    WorldModel,
    IdentityModel,
    compute_diff,
    find_unique_colors,
    most_common_color,
    extract_grid,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exploration policy
# =============================================================================

@dataclass
class ExplorationState:
    """Tracks exploration progress across the agent's lifetime."""

    # Per-action observation counts
    action_counts: dict[int, int] = field(default_factory=lambda: {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    })

    # Per-action textual observations (for synthesis prompts)
    action_observations: dict[int, list[str]] = field(default_factory=lambda: {
        1: [], 2: [], 3: [], 4: [], 5: [],
    })

    # Replay buffer of (before, action, after) transitions
    replay_buffer: list[Transition] = field(default_factory=list)

    # Synthesis tracking
    synthesis_attempts: int = 0
    last_verification: VerificationResult | None = None
    best_accuracy: float = 0.0

    # Model state
    current_model: WorldModel | None = None
    current_code: str | None = None
    model_is_perfect: bool = False

    # Exploration phase control
    min_observations_per_action: int = 3
    max_observations_per_action: int = 10

    # Score tracking
    current_score: int = 0
    max_score: int = 0
    score_history: list[int] = field(default_factory=list)

    # Available actions (populated from API)
    available_action_ids: set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5})

    @property
    def total_observations(self) -> int:
        return sum(self.action_counts.values())

    @property
    def least_observed_action(self) -> int:
        """Return the action ID with the fewest observations."""
        valid = {k: v for k, v in self.action_counts.items()
                 if k in self.available_action_ids}
        return min(valid, key=valid.get)

    @property
    def has_minimum_coverage(self) -> bool:
        """True if every available action has been observed at least min times."""
        return all(
            self.action_counts.get(a, 0) >= self.min_observations_per_action
            for a in self.available_action_ids
        )

    def record_transition(
        self,
        before_frame: np.ndarray,
        action_id: int,
        after_frame: np.ndarray,
        observation_text: str,
    ) -> None:
        """Record a transition and its textual observation."""
        self.replay_buffer.append(Transition(
            before_frame=before_frame,
            action_id=action_id,
            after_frame=after_frame,
            timestep=len(self.replay_buffer),
        ))
        self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1
        if action_id not in self.action_observations:
            self.action_observations[action_id] = []
        self.action_observations[action_id].append(observation_text)


def choose_exploration_action(state: ExplorationState) -> int:
    """
    Decide which action to take next during exploration.

    Priority:
    1. Untested actions (0 observations)
    2. Under-observed actions (below minimum threshold)
    3. Cycle through all available actions (round-robin on least-observed)
    """
    untested = [
        a for a in sorted(state.available_action_ids)
        if state.action_counts.get(a, 0) == 0
    ]
    if untested:
        return untested[0]

    under_observed = [
        a for a in sorted(state.available_action_ids)
        if state.action_counts.get(a, 0) < state.min_observations_per_action
    ]
    if under_observed:
        return min(under_observed, key=lambda a: state.action_counts.get(a, 0))

    return state.least_observed_action


# =============================================================================
# Observation summarization (domain-agnostic)
# =============================================================================


def summarize_transition(
    before: np.ndarray, after: np.ndarray, action_id: int
) -> str:
    """Produce a human-readable summary of what changed between two frames."""
    diff = compute_diff(before, after)

    if diff.count == 0:
        return f"ACTION{action_id}: no visible change"

    parts = [f"ACTION{action_id}: {diff.count} pixels changed"]
    if diff.bbox:
        parts.append(
            f"  region: rows {diff.bbox[0]}-{diff.bbox[2]}, "
            f"cols {diff.bbox[1]}-{diff.bbox[3]}"
        )

    # Summarize color changes
    color_changes: dict[tuple, int] = {}
    for bc, ac in zip(diff.before_colors[:50], diff.after_colors[:50]):
        key = (bc, ac)
        color_changes[key] = color_changes.get(key, 0) + 1

    for (from_c, to_c), count in sorted(color_changes.items(), key=lambda x: -x[1])[:5]:
        parts.append(f"  {count}x: {from_c} -> {to_c}")

    return "\n".join(parts)


def analyze_frame(frame: np.ndarray) -> str:
    """
    Produce a domain-agnostic structural analysis of a frame.

    This gives the LLM context about the visual structure without
    assuming anything about what the colors or regions mean.
    """
    parts = [f"Frame shape: {frame.shape}"]

    # Unique colors
    colors = find_unique_colors(frame)
    parts.append(f"Unique colors: {len(colors)}")
    if len(colors) <= 20:
        for c in colors:
            # Count pixels of this color
            if frame.ndim == 3:
                mask = np.all(frame == np.array(c, dtype=np.uint8), axis=-1)
            else:
                mask = frame == c[0]
            count = int(np.sum(mask))
            parts.append(f"  RGB{c}: {count} pixels ({count / frame.shape[0] / frame.shape[1] * 100:.1f}%)")

    # Background (most common color)
    bg = most_common_color(frame)
    parts.append(f"Most common color (likely background): RGB{bg}")

    # Check for border structure (common in ARC-AGI-3 games)
    if frame.ndim == 3:
        top_row = frame[0, :, :]
        bottom_row = frame[-1, :, :]
        left_col = frame[:, 0, :]
        right_col = frame[:, -1, :]

        top_uniform = np.all(top_row == top_row[0])
        bottom_uniform = np.all(bottom_row == bottom_row[0])
        left_uniform = np.all(left_col == left_col[0])
        right_uniform = np.all(right_col == right_col[0])

        if top_uniform or bottom_uniform or left_uniform or right_uniform:
            parts.append("Border structure detected:")
            if top_uniform:
                parts.append(f"  Top row: uniform RGB{tuple(int(x) for x in top_row[0])}")
            if bottom_uniform:
                parts.append(f"  Bottom row: uniform RGB{tuple(int(x) for x in bottom_row[0])}")
            if left_uniform:
                parts.append(f"  Left col: uniform RGB{tuple(int(x) for x in left_col[0])}")
            if right_uniform:
                parts.append(f"  Right col: uniform RGB{tuple(int(x) for x in right_col[0])}")

    return "\n".join(parts)


# =============================================================================
# Main agent
# =============================================================================


class OOPAgent(Agent):
    """
    An agent that learns game mechanics by synthesizing world models.

    Assumes nothing about the domain. Frames are raw RGB pixels, actions
    are anonymous integers. Everything else is inferred from observations
    and synthesized by the LLM.
    """

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        llm_provider = kwargs.pop("llm_provider", "openrouter")
        llm_model = kwargs.pop("llm_model", "google/gemini-2.5-flash")
        max_refinements = kwargs.pop("max_refinements", 3)
        verbose = kwargs.pop("verbose", True)

        super().__init__(*args, **kwargs)

        self.verbose = verbose

        # Exploration state
        self.exploration = ExplorationState()

        # Synthesis engine
        self.synthesizer = ModelSynthesizer(
            provider=llm_provider,
            model=llm_model,
            max_refinements=max_refinements,
        )

        # Frame tracking
        self._previous_frame: np.ndarray | None = None
        self._pending_action_id: int | None = None
        self._frame_shape: tuple[int, ...] | None = None

        # Phase control
        self._phase: str = "explore"  # explore | synthesize | exploit
        self._exploit_action_queue: list[int] = []

        # Synthesis trigger thresholds
        self._min_transitions_for_synthesis = 10
        self._max_synthesis_rounds = 5

    @property
    def name(self) -> str:
        return f"OOPAgent.{self.MAX_ACTIONS}"

    # -----------------------------------------------------------------
    # Agent interface
    # -----------------------------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            logger.info("Game won!")
            return True

        if self.action_counter >= self.MAX_ACTIONS:
            logger.info("Reached MAX_ACTIONS limit")
            return True

        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Main decision loop: explore -> synthesize -> exploit."""

        # Handle game reset states
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._previous_frame = None
            self._pending_action_id = None
            action = GameAction.RESET
            action.reasoning = "Game not started or over"
            return action

        # Update available actions from API
        if latest_frame.available_actions:
            self._update_available_actions(latest_frame.available_actions)

        # Extract current frame as numpy array
        current_frame = extract_grid(latest_frame.frame)
        if current_frame is None:
            action = GameAction.RESET
            action.reasoning = "Could not extract frame"
            return action

        # Record frame shape once
        if self._frame_shape is None:
            self._frame_shape = current_frame.shape

        # Track score
        self.exploration.current_score = latest_frame.score
        self.exploration.max_score = max(self.exploration.max_score, latest_frame.score)
        self.exploration.score_history.append(latest_frame.score)

        # Record previous transition if we have a pending action
        if self._pending_action_id is not None and self._previous_frame is not None:
            obs_text = summarize_transition(
                self._previous_frame, current_frame, self._pending_action_id
            )
            self.exploration.record_transition(
                before_frame=self._previous_frame,
                action_id=self._pending_action_id,
                after_frame=current_frame,
                observation_text=obs_text,
            )

            if self.verbose:
                logger.info(
                    f"[OOP] Recorded transition #{len(self.exploration.replay_buffer)} "
                    f"(ACTION{self._pending_action_id}): "
                    f"{obs_text.split(chr(10))[0]}"
                )

        # Phase transitions
        self._update_phase()

        # Act based on current phase
        if self._phase == "explore":
            action_id = choose_exploration_action(self.exploration)
        elif self._phase == "synthesize":
            action_id = self._do_synthesis(current_frame)
        elif self._phase == "exploit":
            action_id = self._do_exploitation(current_frame)
        else:
            action_id = choose_exploration_action(self.exploration)

        # Convert to GameAction
        game_action = self._action_id_to_game_action(action_id)
        game_action.reasoning = self._build_reasoning(action_id)

        # Remember for next iteration
        self._previous_frame = current_frame
        self._pending_action_id = action_id

        return game_action

    # -----------------------------------------------------------------
    # Phase management
    # -----------------------------------------------------------------

    def _update_phase(self) -> None:
        """Determine current phase based on exploration state."""
        if self.exploration.model_is_perfect:
            self._phase = "exploit"
            return

        if (
            self.exploration.has_minimum_coverage
            and len(self.exploration.replay_buffer) >= self._min_transitions_for_synthesis
            and self.exploration.synthesis_attempts < self._max_synthesis_rounds
        ):
            self._phase = "synthesize"
            return

        if self.exploration.synthesis_attempts >= self._max_synthesis_rounds:
            if self.exploration.current_model is not None:
                self._phase = "exploit"
            else:
                self._phase = "explore"
            return

        self._phase = "explore"

    # -----------------------------------------------------------------
    # Synthesis
    # -----------------------------------------------------------------

    def _do_synthesis(self, current_frame: np.ndarray) -> int:
        """Run synthesis, then return an exploration action."""
        if self.verbose:
            logger.info(
                f"[OOP] Triggering synthesis (attempt {self.exploration.synthesis_attempts + 1}, "
                f"{len(self.exploration.replay_buffer)} transitions)"
            )

        # Build frame analysis (domain-agnostic)
        frame_analysis = analyze_frame(current_frame)

        # Run CEGIS loop
        model, code, result = self.synthesizer.synthesize(
            replay_buffer=self.exploration.replay_buffer,
            frame_analysis=frame_analysis,
            action_observations=self.exploration.action_observations,
            frame_shape=self._frame_shape,
        )

        self.exploration.synthesis_attempts += 1

        if model is not None and result is not None:
            self.exploration.current_model = model
            self.exploration.current_code = code
            self.exploration.last_verification = result

            if result.accuracy > self.exploration.best_accuracy:
                self.exploration.best_accuracy = result.accuracy

            if result.is_perfect:
                self.exploration.model_is_perfect = True
                if self.verbose:
                    logger.info("[OOP] Perfect model synthesized!")
                self._phase = "exploit"
            else:
                if self.verbose:
                    logger.info(
                        f"[OOP] Synthesis result: {result.accuracy:.1%} accuracy "
                        f"({result.correct}/{result.correct + result.incorrect})"
                    )

                # Diagnose persistent errors
                diagnosis = diagnose_persistent_errors(
                    replay_buffer=self.exploration.replay_buffer,
                    model=model,
                    synthesis_attempts=self.exploration.synthesis_attempts,
                )
                if diagnosis.is_stuck:
                    if self.verbose:
                        logger.info(
                            f"[OOP] Model stuck: "
                            f"{len(diagnosis.failed_transitions)} persistent failures. "
                            f"Gathering more observations."
                        )
                    self.exploration.min_observations_per_action += 2
                    self._phase = "explore"
                else:
                    self._phase = "explore"
        else:
            if self.verbose:
                logger.info("[OOP] Synthesis failed entirely, continuing exploration")
            self._phase = "explore"

        return choose_exploration_action(self.exploration)

    # -----------------------------------------------------------------
    # Exploitation
    # -----------------------------------------------------------------

    def _do_exploitation(self, current_frame: np.ndarray) -> int:
        """Use the synthesized model to pick actions that maximize change."""
        model = self.exploration.current_model

        if self._exploit_action_queue:
            return self._exploit_action_queue.pop(0)

        if model is None:
            return choose_exploration_action(self.exploration)

        # One-step lookahead: try each action, pick the one that
        # produces the most pixel change (heuristic for progress)
        best_action = None
        best_change = -1

        for action_id in sorted(self.exploration.available_action_ids):
            try:
                predicted = model.predict(current_frame, action_id)
                diff = compute_diff(current_frame, predicted)
                if diff.count > best_change:
                    best_change = diff.count
                    best_action = action_id
            except Exception:
                continue

        if best_action is not None and best_change > 0:
            return best_action

        return choose_exploration_action(self.exploration)

    # -----------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------

    def _update_available_actions(self, api_actions: list[GameAction]) -> None:
        """Update available action IDs from the API response."""
        ids = set()
        for ga in api_actions:
            if ga.value >= 1 and ga != GameAction.RESET:
                ids.add(ga.value)
        if ids:
            self.exploration.available_action_ids = ids
            for aid in ids:
                if aid not in self.exploration.action_counts:
                    self.exploration.action_counts[aid] = 0
                if aid not in self.exploration.action_observations:
                    self.exploration.action_observations[aid] = []

    def _action_id_to_game_action(self, action_id: int) -> GameAction:
        """Convert an integer action ID to a GameAction enum."""
        try:
            return GameAction.from_id(action_id)
        except ValueError:
            return GameAction.ACTION1

    def _build_reasoning(self, action_id: int) -> str:
        """Build a reasoning string for the action."""
        parts = [f"phase={self._phase}"]
        parts.append(f"action=ACTION{action_id}")
        parts.append(f"observations={self.exploration.total_observations}")

        if self.exploration.current_model is not None:
            parts.append(f"model_accuracy={self.exploration.best_accuracy:.1%}")

        if self.exploration.model_is_perfect:
            parts.append("model=PERFECT")

        return " | ".join(parts)
