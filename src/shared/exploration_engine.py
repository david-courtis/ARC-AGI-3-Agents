"""
ExplorationEngine — composable LLM-driven exploration extracted from learning_agent.

This class encapsulates the full exploration loop:
1. Action analysis (LLM Call 1): observe before/after frames, update action knowledge
2. Environment analysis (LLM Call 3): understand the game world structure
3. Next action suggestion (LLM Call 2): decide what to do next
4. Action guard: prevent stuck loops by blocking no-effect actions

It is designed to be COMPOSED (not inherited) by synthesis agents. Each synthesis
agent creates an ExplorationEngine and delegates action selection to it, while
adding its own program synthesis on top.

Usage:
    engine = ExplorationEngine(
        results_dir="results",
        llm_provider="clewdr",
        llm_model="claude-sonnet-4-6-thinking",
    )
    engine.initialize_run()

    # Each turn:
    engine.record_score(latest_frame.score)
    if engine.has_pending_analysis():
        engine.complete_action_analysis(current_frame)
    suggestion = engine.suggest_next_action(current_frame)
    engine.prepare_for_action(target_action, current_frame)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .diff import FrameDiffer, compute_sequential_diffs, create_differ
from .knowledge import KnowledgeManager
from .llm_agents import LLMAgent, SyncLLMAgent, create_agent
from .models import (
    ActionAnalysisResult,
    ActionID,
    AgentState,
    NextActionSuggestion,
)
from .run_logger import ConsoleLogger, RunLogger
from .vision import FrameCapture, GridFrameRenderer


class ExplorationEngine:
    """
    Composable LLM-driven exploration engine.

    Encapsulates the learning_agent's full exploration pipeline:
    - Action analysis with before/after frame comparison
    - Environment analysis for world understanding
    - Next action suggestion with strategic reasoning
    - Action guard to prevent stuck loops

    Synthesis agents compose this rather than inherit from it.
    """

    def __init__(
        self,
        results_dir: str = "results",
        llm_provider: str = "clewdr",
        llm_model: str = "claude-sonnet-4-6-thinking",
        reasoning: bool = False,
        verbose: bool = True,
    ):
        self.results_dir = Path(results_dir)
        self.verbose = verbose

        # Initialize components
        self.knowledge_manager = KnowledgeManager(base_dir=str(self.results_dir))
        self.differ: FrameDiffer = create_differ("smart")
        self.renderer = GridFrameRenderer()
        self.console = ConsoleLogger(verbose=self.verbose)

        # Initialize LLM agent
        self._async_llm_agent = create_agent(
            provider=llm_provider, model=llm_model, reasoning=reasoning
        )
        self.llm_agent = SyncLLMAgent(self._async_llm_agent)

        # State (initialized on first call to initialize_run)
        self.state: AgentState | None = None
        self.run_logger: RunLogger | None = None
        self.frame_capture: FrameCapture | None = None

        # Pending analysis tracking
        self._pending_analysis: dict | None = None

        # Hard guard: prevent repeated no-effect actions
        self._last_action_had_effect: bool = True
        self._consecutive_no_effect_count: int = 0
        self._blocked_actions: set[ActionID] = set()
        self._last_no_effect_action: ActionID | None = None

        # Setup sequence tracking
        self._pending_setup: dict | None = None

        # Available actions from API
        self._available_actions: set[ActionID] | None = None

        # Whether disk logging is disabled (for tests)
        self._logging_disabled: bool = False

    def initialize_run(self, disable_logging: bool = False) -> AgentState:
        """Initialize a new exploration run. Returns the fresh AgentState."""
        if not disable_logging:
            self.state = self.knowledge_manager.create_new_run()
            run_dir = self.results_dir / self.state.run_id
            self.run_logger = RunLogger(run_dir)
            self.frame_capture = FrameCapture(
                renderer=self.renderer,
                output_dir=run_dir / "frames",
            )
        else:
            # No-disk mode: create state in memory, frame_capture writes to temp
            import tempfile
            from datetime import datetime
            self._logging_disabled = True
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.state = AgentState.initialize(run_id)
            tmp = Path(tempfile.mkdtemp())
            self.run_logger = None
            self.frame_capture = FrameCapture(
                renderer=self.renderer,
                output_dir=tmp / "frames",
            )

        self.console.info(f"Starting new run: {self.state.run_id}")
        self.console.separator()

        return self.state

    def update_available_actions(self, api_available_actions: list[Any]) -> None:
        """
        Update available actions based on API response.

        Args:
            api_available_actions: List of GameAction enums or ints from the API.
                New arc-agi package returns list[int], old vendor returned list[GameAction].
        """
        # Map GameAction values to ActionID
        self._available_actions = set()
        for game_action in api_available_actions:
            value = game_action.value if hasattr(game_action, "value") else game_action
            if isinstance(value, int) and 1 <= value <= 7:
                try:
                    action_id = ActionID(f"ACTION{value}")
                    self._available_actions.add(action_id)
                except ValueError:
                    pass

        available_names = sorted([a.value for a in self._available_actions])
        self.console.info(f"Available actions from API: {available_names}")

        if self.state:
            self.state.available_actions = available_names

            # Remove knowledge for unavailable actions
            unavailable = set(ActionID) - self._available_actions
            for action_id in unavailable:
                if action_id.value in self.state.action_knowledge:
                    del self.state.action_knowledge[action_id.value]
                    self.console.info(f"   Removed {action_id.value} (not available)")

    def record_score(self, score: int) -> bool:
        """
        Update score and detect stage transitions.

        Returns True if a stage transition occurred.
        """
        stage_transitioned = self.state.update_score(score)
        if stage_transitioned:
            self.console.info(
                f"STAGE TRANSITION! Now on stage {self.state.current_stage + 1} "
                f"(score: {self.state.current_score}/254)"
            )
            self.console.info("New stage may have new objects, mechanics, or constraints!")
        return stage_transitioned

    def has_pending_analysis(self) -> bool:
        """Check if there's a pending action analysis to complete."""
        return self._pending_analysis is not None

    def has_pending_setup(self) -> bool:
        """Check if there's an in-progress setup sequence."""
        return self._pending_setup is not None

    def get_pending_setup_action(self) -> tuple[ActionID | None, ActionID | None]:
        """
        Get the next setup action and target, if any.

        Returns:
            (next_setup_action, target_action) or (None, None) if no setup pending.
        """
        if self._pending_setup is None:
            return None, None

        remaining = self._pending_setup["remaining"]
        target = self._pending_setup["target"]

        if remaining:
            return remaining[0], target
        else:
            return None, target

    def advance_setup(self) -> None:
        """Advance the setup sequence by one step."""
        if self._pending_setup is None:
            return

        remaining = self._pending_setup["remaining"]
        if remaining:
            self._pending_setup["remaining"] = remaining[1:]
        else:
            self._pending_setup = None

    def complete_action_analysis(self, current_frame: Any) -> ActionAnalysisResult | None:
        """
        Complete analysis of the previous action using the current frame as "after".

        Returns the analysis result, or None if analysis fails.
        """
        from .diff import compute_sequential_diffs

        if self._pending_analysis is None:
            return None

        analysis_info = self._pending_analysis
        self._pending_analysis = None

        action_id = analysis_info["action_id"]
        before_path = analysis_info["before_path"]
        before_frame = analysis_info["before_frame"]

        # Capture after frame
        after_path = self.frame_capture.capture(
            current_frame,
            f"after_{self.state.action_count:04d}"
        )

        # Compute diff
        diff = self.differ.compute_diff(before_frame, current_frame)

        self.console.llm_call("action_analysis", action_id)

        knowledge = self.state.get_action_knowledge(action_id)

        start_time = time.time()

        # Extract animation frames
        animation_frames = self._extract_animation_frames(current_frame)
        num_anim_frames = len(animation_frames)

        before_b64 = self.frame_capture.to_base64(before_frame)

        if num_anim_frames > 1:
            animation_b64_list = self.frame_capture.frames_to_base64_list(current_frame)
            sequential_diffs = compute_sequential_diffs(animation_frames)
        else:
            animation_b64_list = [self.frame_capture.to_base64(current_frame)]
            sequential_diffs = []

        prompt = self._build_analysis_prompt(action_id, knowledge, diff)
        state_before = self.state.model_copy(deep=True)

        try:
            analysis = self.llm_agent.analyze_action(
                before_image_b64=before_b64,
                after_image_b64=animation_b64_list[-1],
                action_id=action_id,
                diff=diff,
                action_knowledge=knowledge,
                environment=self.state.environment,
                all_action_knowledge=self.state.action_knowledge,
                animation_frames_b64=animation_b64_list,
                sequential_diffs=sequential_diffs,
                stage_context=self.state.get_stage_context(),
            )

            duration_ms = (time.time() - start_time) * 1000

            # Update knowledge
            self.state = self.knowledge_manager.update_from_analysis(
                self.state, action_id, analysis, diff, before_path, after_path,
            )

            # Log the call
            if self.run_logger:
                self.run_logger.log_action_analysis(
                    action_id=action_id,
                    before_frame_path=before_path,
                    after_frame_path=after_path,
                    diff=diff,
                    prompt=prompt,
                    result=analysis,
                    state_before=state_before,
                    state_after=self.state,
                    duration_ms=duration_ms,
                    full_messages=self.llm_agent.last_messages,
                )

            self.state.llm_call_count += 1

            # Report result
            effect = "HAD EFFECT" if analysis.had_effect else "NO EFFECT"
            consistent = (
                "consistent"
                if analysis.is_consistent_with_previous
                else "new info"
                if analysis.is_consistent_with_previous is not None
                else "first obs"
            )
            self.console.result(
                f"{effect} | {consistent} | {analysis.interpretation[:60]}..."
            )

            # Update no-effect tracking for hard guard
            if analysis.had_effect:
                self._last_action_had_effect = True
                self._consecutive_no_effect_count = 0
                self._blocked_actions.clear()
                self._last_no_effect_action = None
            else:
                self._last_action_had_effect = False
                self._consecutive_no_effect_count += 1
                self._blocked_actions.add(action_id)
                self._last_no_effect_action = action_id

                if knowledge.is_exhausted and knowledge.consecutive_no_effects == 8:
                    self.console.info(
                        f"ACTION EXHAUSTED: {action_id.value} hit 8 consecutive no-effects."
                    )
                else:
                    self.console.info(
                        f"GUARD: {action_id.value} blocked (no effect #{knowledge.consecutive_no_effects}). "
                        f"Blocked: {[a.value for a in self._blocked_actions]}"
                    )

            # Environment analysis after every action (once we've seen a state change)
            has_seen_state_change = any(
                obs.had_effect
                for k in self.state.action_knowledge.values()
                for obs in k.observations
            )

            if has_seen_state_change:
                self._analyze_environment(
                    current_frame,
                    action_context=(
                        f"{action_id.value} had {'effect' if analysis.had_effect else 'NO effect'}: "
                        f"{analysis.interpretation[:100]}"
                    ),
                    diff=diff,
                    action_knowledge=self.state.action_knowledge,
                    had_state_change=analysis.had_effect,
                    action_analysis=analysis,
                )

            # Save state
            if not self._logging_disabled:
                self.knowledge_manager.save_state(self.state)

            return analysis

        except Exception as e:
            self.console.error(f"Action analysis failed: {e}")
            return None

    def suggest_next_action(self, current_frame: Any) -> NextActionSuggestion:
        """
        Get LLM suggestion for next action.

        Returns a NextActionSuggestion with target_action and optional setup_sequence.
        """
        current_path = self.frame_capture.capture(
            current_frame,
            f"current_{self.state.action_count:04d}"
        )
        current_b64 = self.frame_capture.to_base64(current_frame)

        self.console.llm_call("next_action_suggestion")

        prompt = self._build_suggestion_prompt()

        start_time = time.time()

        try:
            suggestion = self.llm_agent.suggest_next_action(
                current_frame_b64=current_b64,
                state=self.state,
            )

            duration_ms = (time.time() - start_time) * 1000

            if self.run_logger:
                self.run_logger.log_next_action_suggestion(
                    current_frame_path=current_path,
                    prompt=prompt,
                    result=suggestion,
                    state=self.state,
                    duration_ms=duration_ms,
                    full_messages=self.llm_agent.last_messages,
                )

            self.state.llm_call_count += 1

            self.console.result(
                f"Target: {suggestion.target_action.value} | "
                f"Setup: {[a.value for a in suggestion.setup_sequence]} | "
                f"{suggestion.reasoning[:40]}..."
            )

            # Apply hard guard
            suggestion = self._apply_action_guard(suggestion)

            # Set up pending setup if needed
            if suggestion.setup_sequence:
                self._pending_setup = {
                    "remaining": suggestion.setup_sequence[1:],
                    "target": suggestion.target_action,
                }

            return suggestion

        except Exception as e:
            self.console.error(f"Next action suggestion failed: {e}")
            unverified = self.state.get_unverified_actions()
            if unverified:
                return NextActionSuggestion(
                    target_action=unverified[0],
                    setup_sequence=[],
                    reasoning="Fallback due to LLM error",
                    expected_information_gain="Unknown",
                    current_board_assessment="Unknown",
                )
            return NextActionSuggestion(
                target_action=ActionID.ACTION1,
                setup_sequence=[],
                reasoning="Fallback - all actions verified",
                expected_information_gain="None",
                current_board_assessment="Unknown",
            )

    def prepare_for_action(self, action_id: ActionID, current_frame: Any) -> None:
        """
        Prepare to analyze an action on the next turn.

        Captures the before frame and stores pending analysis info.
        Call this before executing the target action.
        """
        before_path = self.frame_capture.capture(
            current_frame,
            f"before_{self.state.action_count:04d}"
        )

        self._pending_analysis = {
            "action_id": action_id,
            "before_path": before_path,
            "before_frame": current_frame,
        }

    def increment_action_count(self) -> None:
        """Increment the action counter."""
        self.state.action_count += 1

    def should_terminate(self) -> bool:
        """Check if exploration is complete (all actions verified or exhausted)."""
        return self.state.should_terminate()

    def log_setup_action(self, action_id: ActionID, current_frame: Any, reason: str) -> None:
        """Log a setup action that is executed without analysis."""
        if self.run_logger:
            frame_path = self.frame_capture.capture(
                current_frame, f"setup_{self.state.action_count:04d}"
            )
            self.run_logger.log_setup_action(action_id, frame_path, reason)

    def generate_final_report(self) -> str | None:
        """Generate the final exploration report."""
        if self.state and self.run_logger:
            report_path = self.run_logger.generate_final_report(self.state)
            self.console.info(f"Final report: {report_path}")

            if not self._logging_disabled:
                self.knowledge_manager.save_state(self.state)
            self.console.info("Final state saved")

            self.console.separator()
            self.console.info("Exploration complete!")
            self.console.info(f"Total actions: {self.state.action_count}")
            self.console.info(f"Total LLM calls: {self.state.llm_call_count}")

            verified = [
                k for k, v in self.state.action_knowledge.items() if v.is_verified
            ]
            self.console.info(f"Verified actions: {verified}")

            exhausted = [
                k for k, v in self.state.action_knowledge.items() if v.is_exhausted
            ]
            if exhausted:
                self.console.info(f"Exhausted actions (8+ no-effects): {exhausted}")

            return report_path
        return None

    # ================================================================
    # Internal methods
    # ================================================================

    def _analyze_environment(
        self,
        current_frame: Any,
        action_context: str,
        diff: Any = None,
        action_knowledge: dict | None = None,
        had_state_change: bool = False,
        action_analysis: ActionAnalysisResult | None = None,
    ) -> None:
        """Analyze the environment to understand its structure."""
        self.console.llm_call("environment_analysis")

        current_b64 = self.frame_capture.to_base64(current_frame)

        start_time = time.time()

        try:
            analysis = self.llm_agent.analyze_environment(
                current_frame_b64=current_b64,
                environment=self.state.environment,
                action_context=action_context,
                diff=diff,
                action_knowledge=action_knowledge,
                had_state_change=had_state_change,
                stage_context=self.state.get_stage_context(),
                action_analysis=action_analysis,
            )

            duration_ms = (time.time() - start_time) * 1000

            self.state = self.knowledge_manager.update_environment_from_analysis(
                self.state, analysis,
            )

            self.state.llm_call_count += 1

            if analysis.breakthroughs:
                for breakthrough in analysis.breakthroughs:
                    self.console.info(f"BREAKTHROUGH: {breakthrough}")

            if analysis.movement_constraints:
                constraints_preview = "; ".join(analysis.movement_constraints[:3])
                self.console.result(
                    f"Environment analyzed | Constraints: {constraints_preview[:80]}..."
                )

            if analysis.suggested_action_updates:
                for update in analysis.suggested_action_updates:
                    self.console.info(
                        f"ACTION UPDATE: {update.action_id} -> "
                        f'"{update.suggested_definition}" ({update.reasoning[:50]}...)'
                    )

            if self.run_logger:
                self.run_logger.log_environment_analysis(
                    analysis=analysis,
                    action_context=action_context,
                    duration_ms=duration_ms,
                    full_messages=self.llm_agent.last_messages,
                )

        except Exception as e:
            self.console.error(f"Environment analysis failed: {e}")

    def _build_analysis_prompt(
        self, action_id: ActionID, knowledge: Any, diff: Any
    ) -> str:
        """Build the prompt for action analysis (for logging)."""
        context = self.knowledge_manager.format_for_action_analysis(
            action_id, knowledge, self.state.environment, diff
        )

        is_first = len(knowledge.observations) == 0

        if is_first:
            history_section = "This is the FIRST time observing this action."
        else:
            history_section = f"""YOUR CURRENT DEFINITION OF {action_id.value}:
"{knowledge.current_definition}"

PREVIOUS OBSERVATIONS ({len(knowledge.observations)} total):
{context['observation_history']}"""

        return f"""Analyze the effect of action {action_id.value}.

{history_section}

ENVIRONMENT UNDERSTANDING:
{context['environment']}

FRAME CHANGES:
{context['diff_summary']}

{f"Changed pixels: {context['pixel_changes']}" if context['pixel_changes'] else ""}

Analyze the BEFORE and AFTER frames and respond with your structured analysis."""

    def _build_suggestion_prompt(self) -> str:
        """Build the prompt for next action suggestion (for logging)."""
        context = self.knowledge_manager.format_for_next_action(
            self.state.action_knowledge, self.state.environment
        )

        return f"""Decide what action to test next.

CURRENT ACTION KNOWLEDGE STATUS:
{context['action_status']}

ENVIRONMENT UNDERSTANDING:
{context['environment']}

VERIFIED ACTIONS (can use for setup): {context['verified_actions']}
PENDING ACTIONS (need testing): {context['pending_actions']}

Suggest what action to test next and any setup sequence needed."""

    def _apply_action_guard(self, suggestion: NextActionSuggestion) -> NextActionSuggestion:
        """
        Apply hard guard rules to prevent repeated no-effect actions.

        If the suggested action is blocked (had no effect), modify the suggestion.
        """
        target = suggestion.target_action

        if target not in self._blocked_actions:
            return suggestion

        self.console.info(
            f"GUARD: LLM suggested blocked action {target.value}. Overriding..."
        )

        # If there's already a setup sequence, allow it
        if suggestion.setup_sequence:
            self.console.info("GUARD: Setup sequence exists, allowing")
            return suggestion

        # Strategy 1: Try a different unblocked action
        unblocked_unverified = [
            a for a in self.state.get_unverified_actions()
            if a not in self._blocked_actions
        ]

        if unblocked_unverified:
            new_target = unblocked_unverified[0]
            self.console.info(f"GUARD: Switching to unblocked action {new_target.value}")
            return NextActionSuggestion(
                target_action=new_target,
                setup_sequence=[],
                reasoning=f"Guard override: {target.value} was blocked, trying {new_target.value}",
                expected_information_gain=suggestion.expected_information_gain,
                current_board_assessment=suggestion.current_board_assessment,
            )

        # Strategy 2: Force a setup sequence using verified movement actions
        verified = self.state.get_verified_actions()
        movement_actions = [a for a in verified if a in [
            ActionID.ACTION1, ActionID.ACTION2, ActionID.ACTION3, ActionID.ACTION4
        ]]

        if movement_actions:
            setup = movement_actions[:2] if len(movement_actions) >= 2 else movement_actions
            self.console.info(
                f"GUARD: Forcing setup sequence {[a.value for a in setup]} before {target.value}"
            )
            return NextActionSuggestion(
                target_action=target,
                setup_sequence=setup,
                reasoning=f"Guard override: forced movement before blocked action {target.value}",
                expected_information_gain=suggestion.expected_information_gain,
                current_board_assessment=suggestion.current_board_assessment,
            )

        # Strategy 3: Clear blocks after too many consecutive no-effects
        if self._consecutive_no_effect_count >= 3:
            self.console.info(
                f"GUARD: {self._consecutive_no_effect_count} consecutive no-effects. Clearing blocks."
            )
            self._blocked_actions.clear()
            self._consecutive_no_effect_count = 0

            available_actions = self._get_testable_actions()
            for action in available_actions:
                if action != target:
                    return NextActionSuggestion(
                        target_action=action,
                        setup_sequence=[],
                        reasoning="Guard reset: trying different action after repeated failures",
                        expected_information_gain="Unknown - resetting exploration",
                        current_board_assessment=suggestion.current_board_assessment,
                    )

        return suggestion

    def _get_testable_actions(self) -> list[ActionID]:
        """Get the list of actions that can be tested."""
        if self._available_actions is not None:
            return list(self._available_actions)
        return list(ActionID)

    def _extract_animation_frames(self, frame_data: Any) -> list:
        """Extract individual animation frames from frame data."""
        if isinstance(frame_data, np.ndarray):
            if frame_data.ndim == 3 and frame_data.shape[-1] == 3:
                # Single (H, W, 3) RGB frame — not multiple animation frames
                return [frame_data]
            if frame_data.ndim == 3:
                # (N, H, W) palette-indexed animation — split into N frames
                return [frame_data[i] for i in range(len(frame_data))]
            return [frame_data]

        if isinstance(frame_data, list):
            if len(frame_data) > 0 and isinstance(frame_data[0], list):
                if len(frame_data[0]) > 0 and isinstance(frame_data[0][0], list):
                    return frame_data
                return [frame_data]

        return [frame_data]
