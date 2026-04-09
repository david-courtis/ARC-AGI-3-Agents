"""
Learning Agent - Main Agent Implementation.

This module contains the main LearningAgent class that orchestrates
the exploration loop using the two-phase LLM architecture.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...agent import Agent
from ...structs import FrameData, GameAction, GameState
from .diff import FrameDiffer, create_differ
from .knowledge import KnowledgeManager
from .llm_agents import LLMAgent, SyncLLMAgent, create_agent
from .models import ActionAnalysisResult, ActionID, AgentState, NextActionSuggestion, StageInfo
from .run_logger import ConsoleLogger, RunLogger
from .vision import FrameCapture, GridFrameRenderer


class LearningAgent(Agent):
    """
    A learning agent that discovers game mechanics from scratch.

    This agent:
    1. Takes actions and observes state transitions
    2. Builds iterative understanding of what actions do
    3. Builds understanding of the environment
    4. Validates hypotheses through repeated observations

    No game-specific knowledge is baked in.
    """

    MAX_ACTIONS = 200  # Maximum actions before forced termination

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract LearningAgent-specific kwargs before passing to parent
        llm_provider = kwargs.pop("llm_provider", "openrouter")
        llm_model = kwargs.pop("llm_model", "google/gemini-2.5-flash")
        reasoning = kwargs.pop("reasoning", False)
        results_dir = kwargs.pop("results_dir", "results")
        verbose = kwargs.pop("verbose", True)

        super().__init__(*args, **kwargs)

        # Configuration
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

        # State (initialized on first action)
        # Note: Using _agent_state to avoid conflict with parent Agent.state property
        self._agent_state: AgentState | None = None
        self.run_logger: RunLogger | None = None
        self.frame_capture: FrameCapture | None = None

        # Track previous frame for diff
        self._previous_frame: Any = None
        self._pending_analysis: dict | None = None

        # Hard guard: Track consecutive no-effect actions to prevent loops
        self._last_action_had_effect: bool = True
        self._consecutive_no_effect_count: int = 0
        self._blocked_actions: set[ActionID] = set()  # Actions blocked until state changes
        self._last_no_effect_action: ActionID | None = None

        # Setup sequence tracking
        self._pending_setup: dict | None = None  # {"remaining": [...], "target": ActionID}

        # Available actions from API (populated on first frame)
        self._available_actions: set[ActionID] | None = None  # None = not yet determined

    @property
    def name(self) -> str:
        return f"LearningAgent.{self.MAX_ACTIONS}"

    def _initialize_run(self) -> None:
        """Initialize a new exploration run."""
        self._agent_state = self.knowledge_manager.create_new_run()
        run_dir = self.results_dir / self._agent_state.run_id

        self.run_logger = RunLogger(run_dir)
        self.frame_capture = FrameCapture(
            renderer=self.renderer,
            output_dir=run_dir / "frames",
        )

        self.console.info(f"Starting new run: {self._agent_state.run_id}")
        self.console.separator()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Check if the agent is done exploring."""
        # Check for win
        if latest_frame.state is GameState.WIN:
            return True

        # Check if we've reached max actions
        if self._agent_state and self._agent_state.action_count >= self.MAX_ACTIONS:
            self.console.info("Reached maximum actions limit")
            return True

        # Check if all actions are verified or maxed out
        if self._agent_state and self._agent_state.should_terminate():
            self.console.info("All actions verified or maxed out")
            return True

        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """
        Choose the next action using the two-phase LLM architecture.

        Phase 1: If we just took an action, analyze what happened
        Phase 2: Get suggestion for next action
        """
        # Initialize on first call
        if self._agent_state is None:
            self._initialize_run()

        # Handle game states that require reset
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._previous_frame = None
            self._pending_analysis = None
            action = GameAction.RESET
            action.reasoning = "Game not started or over - need to reset"
            return action

        # Capture available actions from API on first valid frame
        if self._available_actions is None and latest_frame.available_actions:
            self._update_available_actions(latest_frame.available_actions)

        # Track score for stage detection
        stage_transitioned = self._agent_state.update_score(latest_frame.score)
        if stage_transitioned:
            self.console.info(
                f"🎯 STAGE TRANSITION! Now on stage {self._agent_state.current_stage + 1} "
                f"(score: {self._agent_state.current_score}/254)"
            )
            self.console.info(
                "⚠️  New stage may have new objects, mechanics, or constraints!"
            )

        # Get the current frame data
        current_frame = latest_frame.frame

        # Debug: Log frame structure to understand animation frames
        if isinstance(current_frame, list) and len(current_frame) > 0:
            if isinstance(current_frame[0], list) and len(current_frame[0]) > 0:
                if isinstance(current_frame[0][0], list):
                    # 3D structure - multiple animation frames
                    num_frames = len(current_frame)
                    frame_size = f"{len(current_frame[0])}x{len(current_frame[0][0])}"
                    # Check if first and last frames differ
                    first_frame = np.array(current_frame[0])
                    last_frame = np.array(current_frame[-1])
                    frames_differ = not np.array_equal(first_frame, last_frame)
                    diff_pixels = np.sum(first_frame != last_frame) if frames_differ else 0
                    self.console.info(
                        f"📊 Frame data: {num_frames} animation frame(s), grid size {frame_size}, "
                        f"first!=last: {frames_differ} ({diff_pixels} pixels differ)"
                    )
                else:
                    # 2D structure - single frame
                    self.console.info(f"📊 Frame data: 1 frame (2D), grid size {len(current_frame)}x{len(current_frame[0])}")

        # ================================================================
        # PHASE 1: Analyze previous action (if any)
        # ================================================================
        if self._pending_analysis is not None:
            self._complete_action_analysis(current_frame)

        # ================================================================
        # PHASE 2: Continue pending setup sequence (if any)
        # ================================================================
        if self._pending_setup is not None:
            remaining = self._pending_setup["remaining"]
            target = self._pending_setup["target"]

            if remaining:
                # Execute next setup action
                setup_action = remaining[0]
                action = self._convert_to_game_action(setup_action)

                self.console.action(
                    setup_action,
                    f"Continuing setup (remaining: {[a.value for a in remaining]})"
                )

                # Update state
                self._agent_state.action_count += 1
                self._pending_setup = {
                    "remaining": remaining[1:],
                    "target": target,
                }

                return action
            else:
                # Setup complete - now execute the target action
                self.console.info(f"Setup complete, executing target: {target.value}")

                action = self._convert_to_game_action(target)

                # Capture before frame for analysis on next iteration
                before_path = self.frame_capture.capture(
                    current_frame,
                    f"before_{self._agent_state.action_count:04d}"
                )

                # Store pending analysis info
                self._pending_analysis = {
                    "action_id": target,
                    "before_path": before_path,
                    "before_frame": current_frame,
                }

                # Clear the pending setup
                self._pending_setup = None

                # Update state
                self._previous_frame = current_frame
                self._agent_state.action_count += 1

                return action

        # ================================================================
        # PHASE 3: Get next action suggestion
        # ================================================================
        suggestion = self._get_next_action_suggestion(current_frame)

        # ================================================================
        # PHASE 4: Execute setup sequence (if any)
        # ================================================================
        # For setup actions, we execute them without full analysis
        # They use verified actions so we trust them
        if suggestion.setup_sequence:
            # Execute first setup action and return
            # The rest will be handled in subsequent calls
            setup_action = suggestion.setup_sequence[0]
            action = self._convert_to_game_action(setup_action)

            self.console.action(
                setup_action,
                f"Setup action (sequence: {[a.value for a in suggestion.setup_sequence]})"
            )

            # Update state for next iteration
            self._agent_state.action_count += 1

            # Store remaining setup and target
            self._pending_setup = {
                "remaining": suggestion.setup_sequence[1:],
                "target": suggestion.target_action,
            }

            return action

        # ================================================================
        # PHASE 5: Execute target action
        # ================================================================
        target_action = suggestion.target_action
        action = self._convert_to_game_action(target_action)

        self.console.action(
            target_action,
            f"Target action - {suggestion.reasoning[:50]}..."
        )

        # Capture before frame for analysis on next iteration
        before_path = self.frame_capture.capture(
            current_frame,
            f"before_{self._agent_state.action_count:04d}"
        )

        # Store pending analysis info
        self._pending_analysis = {
            "action_id": target_action,
            "before_path": before_path,
            "before_frame": current_frame,
        }

        # Update state
        self._previous_frame = current_frame
        self._agent_state.action_count += 1

        return action

    def _complete_action_analysis(self, current_frame: Any) -> None:
        """Complete the analysis of the previous action."""
        from .diff import compute_sequential_diffs

        analysis_info = self._pending_analysis
        self._pending_analysis = None

        action_id = analysis_info["action_id"]
        before_path = analysis_info["before_path"]
        before_frame = analysis_info["before_frame"]

        # Capture after frame (uses last animation frame)
        after_path = self.frame_capture.capture(
            current_frame,
            f"after_{self._agent_state.action_count:04d}"
        )

        # Compute diff (between before state and final state)
        diff = self.differ.compute_diff(before_frame, current_frame)

        self.console.llm_call("action_analysis", action_id)

        # Get action knowledge
        knowledge = self._agent_state.get_action_knowledge(action_id)

        # Call LLM for analysis
        start_time = time.time()

        # Extract animation frames from current_frame
        animation_frames = self._extract_animation_frames(current_frame)
        num_anim_frames = len(animation_frames)

        # Get base64 images for all frames
        before_b64 = self.frame_capture.to_base64(before_frame)

        # If we have multiple animation frames, send them all
        if num_anim_frames > 1:
            # Get all animation frame images
            animation_b64_list = self.frame_capture.frames_to_base64_list(current_frame)
            # Compute sequential diffs between animation frames
            sequential_diffs = compute_sequential_diffs(animation_frames)
        else:
            animation_b64_list = [self.frame_capture.to_base64(current_frame)]
            sequential_diffs = []

        # Build prompt for logging
        prompt = self._build_analysis_prompt(action_id, knowledge, diff)

        state_before = self._agent_state.model_copy(deep=True)

        try:
            analysis = self.llm_agent.analyze_action(
                before_image_b64=before_b64,
                after_image_b64=animation_b64_list[-1],  # Last frame for compatibility
                action_id=action_id,
                diff=diff,
                action_knowledge=knowledge,
                environment=self._agent_state.environment,
                all_action_knowledge=self._agent_state.action_knowledge,
                animation_frames_b64=animation_b64_list,  # All animation frames
                sequential_diffs=sequential_diffs,  # Diffs between consecutive frames
                stage_context=self._agent_state.get_stage_context(),  # Stage info
            )

            duration_ms = (time.time() - start_time) * 1000

            # Update knowledge
            self._agent_state = self.knowledge_manager.update_from_analysis(
                self._agent_state,
                action_id,
                analysis,
                diff,
                before_path,
                after_path,
            )

            # Log the call with full conversation history
            self.run_logger.log_action_analysis(
                action_id=action_id,
                before_frame_path=before_path,
                after_frame_path=after_path,
                diff=diff,
                prompt=prompt,
                result=analysis,
                state_before=state_before,
                state_after=self._agent_state,
                duration_ms=duration_ms,
                full_messages=self.llm_agent.last_messages,
            )

            self._agent_state.llm_call_count += 1

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
                # State changed - clear blocked actions
                self._last_action_had_effect = True
                self._consecutive_no_effect_count = 0
                self._blocked_actions.clear()
                self._last_no_effect_action = None
            else:
                # No effect - block this action until state changes
                self._last_action_had_effect = False
                self._consecutive_no_effect_count += 1
                self._blocked_actions.add(action_id)
                self._last_no_effect_action = action_id

                # Check if action just became exhausted
                if knowledge.is_exhausted and knowledge.consecutive_no_effects == 8:
                    self.console.info(
                        f"⛔ ACTION EXHAUSTED: {action_id.value} hit 8 consecutive no-effects. "
                        f"Will no longer test this action."
                    )
                else:
                    self.console.info(
                        f"GUARD: {action_id.value} blocked (no effect #{knowledge.consecutive_no_effects}). "
                        f"Blocked: {[a.value for a in self._blocked_actions]}"
                    )

            # ================================================================
            # ENVIRONMENT ANALYSIS: Run after EVERY action to continuously
            # refine understanding based on new evidence
            # ================================================================
            # Only skip the very first analysis until we've seen a state change
            # (avoid uneducated guesses without evidence)
            has_seen_state_change = any(
                obs.had_effect
                for k in self._agent_state.action_knowledge.values()
                for obs in k.observations
            )

            if has_seen_state_change:
                self._analyze_environment(
                    current_frame,
                    action_context=f"{action_id.value} had {'effect' if analysis.had_effect else 'NO effect'}: {analysis.interpretation[:100]}",
                    diff=diff,
                    action_knowledge=self._agent_state.action_knowledge,
                    had_state_change=analysis.had_effect,
                    action_analysis=analysis,  # Pass full action analysis result
                )

        except Exception as e:
            self.console.error(f"Action analysis failed: {e}")
            # Continue without analysis update

        # Save state after analysis
        self.knowledge_manager.save_state(self._agent_state)

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
                environment=self._agent_state.environment,
                action_context=action_context,
                diff=diff,
                action_knowledge=action_knowledge,
                had_state_change=had_state_change,
                stage_context=self._agent_state.get_stage_context(),  # Stage info
                action_analysis=action_analysis,  # Full action analysis insights
            )

            duration_ms = (time.time() - start_time) * 1000

            # Update environment knowledge
            self._agent_state = self.knowledge_manager.update_environment_from_analysis(
                self._agent_state,
                analysis,
            )

            self._agent_state.llm_call_count += 1

            # Report breakthroughs
            if analysis.breakthroughs:
                for breakthrough in analysis.breakthroughs:
                    self.console.info(f"🔍 BREAKTHROUGH: {breakthrough}")

            # Report new movement constraints
            if analysis.movement_constraints:
                constraints_preview = "; ".join(analysis.movement_constraints[:3])
                self.console.result(
                    f"Environment analyzed | Constraints: {constraints_preview[:80]}..."
                )

            # Report suggested action updates
            if analysis.suggested_action_updates:
                for update in analysis.suggested_action_updates:
                    self.console.info(
                        f"📝 ACTION UPDATE: {update.action_id} -> \"{update.suggested_definition}\" ({update.reasoning[:50]}...)"
                    )

            # Log the call with full conversation history
            if self.run_logger:
                self.run_logger.log_environment_analysis(
                    analysis=analysis,
                    action_context=action_context,
                    duration_ms=duration_ms,
                    full_messages=self.llm_agent.last_messages,
                )

        except Exception as e:
            self.console.error(f"Environment analysis failed: {e}")
            # Continue without environment update

    def _get_next_action_suggestion(self, current_frame: Any) -> NextActionSuggestion:
        """Get LLM suggestion for next action."""
        # Capture current frame for the suggestion
        current_path = self.frame_capture.capture(
            current_frame,
            f"current_{self._agent_state.action_count:04d}"
        )

        current_b64 = self.frame_capture.to_base64(current_frame)

        self.console.llm_call("next_action_suggestion")

        # Build prompt for logging
        prompt = self._build_suggestion_prompt()

        start_time = time.time()

        try:
            suggestion = self.llm_agent.suggest_next_action(
                current_frame_b64=current_b64,
                state=self._agent_state,
            )

            duration_ms = (time.time() - start_time) * 1000

            # Log the call with full conversation history
            self.run_logger.log_next_action_suggestion(
                current_frame_path=current_path,
                prompt=prompt,
                result=suggestion,
                state=self._agent_state,
                duration_ms=duration_ms,
                full_messages=self.llm_agent.last_messages,
            )

            self._agent_state.llm_call_count += 1

            self.console.result(
                f"Target: {suggestion.target_action.value} | "
                f"Setup: {[a.value for a in suggestion.setup_sequence]} | "
                f"{suggestion.reasoning[:40]}..."
            )

            # HARD GUARD: Enforce blocked action rules
            suggestion = self._apply_action_guard(suggestion)

            return suggestion

        except Exception as e:
            self.console.error(f"Next action suggestion failed: {e}")
            # Fallback to random unverified action
            unverified = self._agent_state.get_unverified_actions()
            if unverified:
                return NextActionSuggestion(
                    target_action=unverified[0],
                    setup_sequence=[],
                    reasoning="Fallback due to LLM error",
                    expected_information_gain="Unknown",
                    current_board_assessment="Unknown",
                )
            else:
                # All verified, pick any
                return NextActionSuggestion(
                    target_action=ActionID.ACTION1,
                    setup_sequence=[],
                    reasoning="Fallback - all actions verified",
                    expected_information_gain="None",
                    current_board_assessment="Unknown",
                )

    def _build_analysis_prompt(
        self, action_id: ActionID, knowledge: Any, diff: Any
    ) -> str:
        """Build the prompt for action analysis (for logging)."""
        context = self.knowledge_manager.format_for_action_analysis(
            action_id, knowledge, self._agent_state.environment, diff
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
            self._agent_state.action_knowledge, self._agent_state.environment
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

        If the suggested action is blocked (had no effect), this method will:
        1. Force a setup sequence to move to a different state
        2. Or pick a different unblocked action
        """
        target = suggestion.target_action

        # If target is not blocked, allow it
        if target not in self._blocked_actions:
            return suggestion

        # Target is blocked - we need to modify the suggestion
        self.console.info(
            f"GUARD: LLM suggested blocked action {target.value}. Overriding..."
        )

        # If there's already a setup sequence, it should change state - allow it
        if suggestion.setup_sequence:
            self.console.info(
                f"GUARD: Setup sequence exists, allowing (will clear blocked after movement)"
            )
            return suggestion

        # No setup sequence and action is blocked - we need to force a change
        # Strategy 1: Try a different unblocked action
        unblocked_unverified = [
            a for a in self._agent_state.get_unverified_actions()
            if a not in self._blocked_actions
        ]

        if unblocked_unverified:
            new_target = unblocked_unverified[0]
            self.console.info(
                f"GUARD: Switching to unblocked action {new_target.value}"
            )
            return NextActionSuggestion(
                target_action=new_target,
                setup_sequence=[],
                reasoning=f"Guard override: {target.value} was blocked, trying {new_target.value}",
                expected_information_gain=suggestion.expected_information_gain,
                current_board_assessment=suggestion.current_board_assessment,
            )

        # Strategy 2: Force a setup sequence using verified movement actions
        verified = self._agent_state.get_verified_actions()
        movement_actions = [a for a in verified if a in [
            ActionID.ACTION1, ActionID.ACTION2, ActionID.ACTION3, ActionID.ACTION4
        ]]

        if movement_actions:
            # Create a simple movement sequence to change state
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

        # Strategy 3: If we have too many consecutive no-effects, try any action
        if self._consecutive_no_effect_count >= 3:
            # Clear blocked and try again - something is wrong
            self.console.info(
                f"GUARD: {self._consecutive_no_effect_count} consecutive no-effects. "
                f"Clearing blocks and trying different approach."
            )
            self._blocked_actions.clear()
            self._consecutive_no_effect_count = 0

            # Pick a random different action from available actions
            available_actions = self._get_testable_actions()
            for action in available_actions:
                if action != target:
                    return NextActionSuggestion(
                        target_action=action,
                        setup_sequence=[],
                        reasoning=f"Guard reset: trying different action after repeated failures",
                        expected_information_gain="Unknown - resetting exploration",
                        current_board_assessment=suggestion.current_board_assessment,
                    )

        # Fallback: allow the original suggestion
        return suggestion

    def _update_available_actions(self, api_available_actions: list[GameAction]) -> None:
        """
        Update the set of available actions based on API response.

        This filters out actions that the game doesn't support, avoiding
        wasted attempts on unavailable actions (e.g., ACTION5 when only 1-4 are available).
        """
        # Map GameAction to ActionID for simple actions we care about
        game_to_action = {
            GameAction.ACTION1: ActionID.ACTION1,
            GameAction.ACTION2: ActionID.ACTION2,
            GameAction.ACTION3: ActionID.ACTION3,
            GameAction.ACTION4: ActionID.ACTION4,
            GameAction.ACTION5: ActionID.ACTION5,
        }

        # Filter to only simple actions that are available
        self._available_actions = set()
        for game_action in api_available_actions:
            if game_action in game_to_action:
                self._available_actions.add(game_to_action[game_action])

        # Log what's available
        available_names = sorted([a.value for a in self._available_actions])
        self.console.info(f"📋 Available actions from API: {available_names}")

        # Update agent state
        if self._agent_state:
            # Store available actions in state for prompts
            self._agent_state.available_actions = available_names

            # Remove action knowledge for unavailable actions
            unavailable = set(ActionID) - self._available_actions
            for action_id in unavailable:
                if action_id.value in self._agent_state.action_knowledge:
                    del self._agent_state.action_knowledge[action_id.value]
                    self.console.info(f"   Removed {action_id.value} (not available in this game)")

    def _get_testable_actions(self) -> list[ActionID]:
        """Get the list of actions that can be tested."""
        if self._available_actions is not None:
            return list(self._available_actions)
        # Default to all actions if not yet determined
        return list(ActionID)

    def _convert_to_game_action(self, action_id: ActionID) -> GameAction:
        """Convert ActionID to GameAction."""
        action_map = {
            ActionID.ACTION1: GameAction.ACTION1,
            ActionID.ACTION2: GameAction.ACTION2,
            ActionID.ACTION3: GameAction.ACTION3,
            ActionID.ACTION4: GameAction.ACTION4,
            ActionID.ACTION5: GameAction.ACTION5,
        }
        action = action_map[action_id]
        action.reasoning = f"Learning agent exploring {action_id.value}"
        return action

    def _extract_animation_frames(self, frame_data: Any) -> list:
        """
        Extract individual animation frames from frame data.

        Args:
            frame_data: 3D frame data [num_frames, height, width] or 2D single frame

        Returns:
            List of 2D frames
        """
        if isinstance(frame_data, np.ndarray):
            if frame_data.ndim == 3:
                return [frame_data[i] for i in range(len(frame_data))]
            return [frame_data]

        if isinstance(frame_data, list):
            if len(frame_data) > 0 and isinstance(frame_data[0], list):
                if len(frame_data[0]) > 0 and isinstance(frame_data[0][0], list):
                    # 3D list: multiple animation frames
                    return frame_data
                # 2D list: single frame
                return [frame_data]

        return [frame_data]

    def cleanup(self, scorecard: Any = None) -> None:
        """Cleanup when the agent is done."""
        if self._agent_state and self.run_logger:
            # Generate final report
            report_path = self.run_logger.generate_final_report(self._agent_state)
            self.console.info(f"Final report: {report_path}")

            # Save final state
            self.knowledge_manager.save_state(self._agent_state)
            self.console.info(f"Final state saved")

            self.console.separator()
            self.console.info("Exploration complete!")
            self.console.info(f"Total actions: {self._agent_state.action_count}")
            self.console.info(f"Total LLM calls: {self._agent_state.llm_call_count}")

            # Summary of verified actions
            verified = [
                k for k, v in self._agent_state.action_knowledge.items() if v.is_verified
            ]
            self.console.info(f"Verified actions: {verified}")

            # Summary of exhausted actions
            exhausted = [
                k for k, v in self._agent_state.action_knowledge.items() if v.is_exhausted
            ]
            if exhausted:
                self.console.info(f"Exhausted actions (8+ no-effects): {exhausted}")
