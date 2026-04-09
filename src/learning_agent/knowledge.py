"""
Knowledge management for the Learning Agent.

This module handles state persistence, knowledge updates, and formatting
knowledge for LLM prompts.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .models import (
    ActionAnalysisResult,
    ActionID,
    ActionKnowledge,
    ActionObservation,
    AgentState,
    DiffResult,
    EnvironmentAnalysisResult,
    EnvironmentKnowledge,
    SuggestedActionUpdate,
)


# ============================================================================
# Abstract Base Classes / Protocols
# ============================================================================


class StateStore(Protocol):
    """Protocol for state persistence."""

    def save(self, state: AgentState) -> None:
        """Save the agent state."""
        ...

    def load(self, run_id: str) -> AgentState | None:
        """Load agent state, returns None if not found."""
        ...

    def exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        ...


class KnowledgeFormatter(ABC):
    """Abstract base for formatting knowledge for different purposes."""

    @abstractmethod
    def format_action_knowledge(self, knowledge: ActionKnowledge) -> str:
        """Format action knowledge for display/prompts."""
        ...

    @abstractmethod
    def format_environment(self, environment: EnvironmentKnowledge) -> str:
        """Format environment knowledge for display/prompts."""
        ...

    @abstractmethod
    def format_observation_history(
        self, observations: list[ActionObservation], limit: int = 5
    ) -> str:
        """Format recent observation history."""
        ...


# ============================================================================
# Implementations
# ============================================================================


class JSONStateStore:
    """JSON-based state persistence."""

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)

    def _get_run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def _get_state_path(self, run_id: str) -> Path:
        return self._get_run_dir(run_id) / "state.json"

    def save(self, state: AgentState) -> None:
        """Save agent state to JSON file."""
        run_dir = self._get_run_dir(state.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        state_path = self._get_state_path(state.run_id)
        with open(state_path, "w") as f:
            f.write(state.model_dump_json(indent=2))

    def load(self, run_id: str) -> AgentState | None:
        """Load agent state from JSON file."""
        state_path = self._get_state_path(run_id)
        if not state_path.exists():
            return None

        with open(state_path) as f:
            return AgentState.model_validate_json(f.read())

    def exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return self._get_state_path(run_id).exists()


class PromptKnowledgeFormatter(KnowledgeFormatter):
    """Format knowledge for LLM prompts."""

    def format_action_knowledge(self, knowledge: ActionKnowledge) -> str:
        """Format action knowledge for LLM context."""
        if knowledge.is_verified:
            status = "VERIFIED"
        elif len(knowledge.observations) == 0:
            status = "NO OBSERVATIONS"
        else:
            consistent = knowledge._count_recent_consistent()
            status = f"{consistent}/3 consistent, {knowledge.verification_attempts}/8 attempts"

        lines = [
            f"Action: {knowledge.action_id.value}",
            f"Status: {status}",
        ]

        if knowledge.current_definition:
            lines.append(f"Definition: {knowledge.current_definition}")

        return "\n".join(lines)

    def format_all_action_knowledge(
        self, action_knowledge: dict[str, ActionKnowledge]
    ) -> str:
        """Format all action knowledge for prompts."""
        lines = []
        for action_id, knowledge in action_knowledge.items():
            if knowledge.is_verified:
                status = f"VERIFIED - {knowledge.current_definition}"
            elif knowledge.is_exhausted:
                status = f"EXHAUSTED (8+ consecutive no-effects) - {knowledge.current_definition or 'no definition'}"
            elif len(knowledge.observations) == 0:
                status = "NO OBSERVATIONS YET"
            else:
                consistent = knowledge._count_recent_consistent()
                no_effect_warn = ""
                if knowledge.consecutive_no_effects >= 3:
                    no_effect_warn = f" [WARNING: {knowledge.consecutive_no_effects} consecutive no-effects]"
                status = f"{consistent}/3 consistent, {knowledge.verification_attempts}/8 attempts{no_effect_warn}"
                if knowledge.current_definition:
                    status += f" | Def: {knowledge.current_definition}"
            lines.append(f"  {action_id}: {status}")

        return "\n".join(lines)

    def format_environment(self, environment: EnvironmentKnowledge) -> str:
        """Format environment knowledge for LLM context."""
        sections = []

        # Add uncertainty header if we have any environment understanding
        if environment.analysis_count > 0:
            sections.append(
                "⚠️ NOTE: This understanding is TENTATIVE and may be WRONG. "
                "If observations contradict it, the model should be UPDATED."
            )

        # === STRUCTURED ENVIRONMENT ANALYSIS (from dedicated LLM calls) ===
        # This is the primary source of environment understanding

        # Background and boundaries
        if environment.background_color:
            sections.append(f"BACKGROUND: {environment.background_color}")

        if environment.has_border:
            border_info = f"BORDER: Yes ({environment.border_color or 'unknown color'})"
            if environment.border_description:
                border_info += f"\n  Description: {environment.border_description}"
            sections.append(border_info)

        # Spatial structure (critical for understanding the play area)
        if environment.spatial_structure:
            sections.append(f"SPATIAL STRUCTURE:\n  {environment.spatial_structure}")

        # High-level domain description
        if environment.domain_description:
            sections.append(f"DOMAIN DESCRIPTION:\n  {environment.domain_description}")

        # Key breakthroughs (most important discoveries)
        if environment.breakthroughs:
            breakthrough_lines = "\n".join(f"  - {b}" for b in environment.breakthroughs)
            sections.append(f"KEY BREAKTHROUGHS:\n{breakthrough_lines}")

        # Unexplored elements (areas needing more investigation)
        if environment.unexplored_elements:
            unexplored_lines = "\n".join(f"  - {e}" for e in environment.unexplored_elements)
            sections.append(f"UNEXPLORED ELEMENTS (need investigation):\n{unexplored_lines}")

        # Movement constraints (critical for action understanding)
        if environment.movement_constraints:
            constraint_lines = "\n".join(f"  - {c}" for c in environment.movement_constraints)
            sections.append(f"MOVEMENT CONSTRAINTS:\n{constraint_lines}")

        # Internal walls/obstacles
        if environment.internal_walls:
            wall_lines = "\n".join(f"  - {w}" for w in environment.internal_walls)
            sections.append(f"INTERNAL WALLS/OBSTACLES:\n{wall_lines}")

        # Identified objects (structured) - TENTATIVE, may be wrong!
        if environment.identified_objects:
            obj_lines = []
            for obj in environment.identified_objects:
                name = obj.get('name', 'Unknown')
                color = obj.get('color', '?')
                shape = obj.get('shape', '?')
                role = obj.get('role_hypothesis', 'Unknown')
                evidence = obj.get('evidence_for_role', '')
                obj_line = f"  - {name}: {color} {shape} (Role hypothesis: {role})"
                if evidence:
                    obj_line += f"\n      Evidence: {evidence}"
                obj_lines.append(obj_line)
            sections.append(f"IDENTIFIED OBJECTS (TENTATIVE - may be wrong!):\n" + "\n".join(obj_lines))

        # Open questions
        if environment.open_questions:
            question_lines = "\n".join(f"  - {q}" for q in environment.open_questions)
            sections.append(f"OPEN QUESTIONS:\n{question_lines}")

        # === LEGACY FIELDS (from action analysis side effects) ===
        if environment.objects:
            obj_lines = []
            for obj in environment.objects:
                obj_lines.append(f"  - {obj.name}: {obj.description}")
            sections.append("Legacy Objects:\n" + "\n".join(obj_lines))

        if environment.spatial_rules:
            rules = "\n".join(f"  - {rule}" for rule in environment.spatial_rules)
            sections.append(f"Spatial Rules:\n{rules}")

        if environment.general_observations:
            # Only show last 5 observations
            recent = environment.general_observations[-5:]
            obs = "\n".join(f"  - {o}" for o in recent)
            sections.append(f"Recent Observations:\n{obs}")

        if not sections:
            return "No environment knowledge yet."

        return "\n\n".join(sections)

    def format_observation_history(
        self, observations: list[ActionObservation], limit: int = 5
    ) -> str:
        """Format recent observation history."""
        if not observations:
            return "No observations yet."

        recent = observations[-limit:]
        lines = []
        for i, obs in enumerate(recent, 1):
            effect_str = "HAD EFFECT" if obs.had_effect else "NO EFFECT"
            consistent_str = ""
            if obs.was_consistent is not None:
                consistent_str = " (consistent)" if obs.was_consistent else " (NEW INFO)"

            lines.append(
                f"  [{i}] {effect_str}{consistent_str}\n"
                f"      Context: {obs.context_description}\n"
                f"      Interpretation: {obs.llm_interpretation}"
            )

        return "\n".join(lines)


class KnowledgeManager:
    """
    Central manager for all knowledge operations.

    Handles:
    - State persistence
    - Knowledge updates
    - Formatting for prompts
    """

    def __init__(
        self,
        store: StateStore | None = None,
        formatter: KnowledgeFormatter | None = None,
        base_dir: str = "results",
    ):
        self.store = store or JSONStateStore(base_dir)
        self.formatter = formatter or PromptKnowledgeFormatter()
        self.base_dir = Path(base_dir)

    def create_new_run(self) -> AgentState:
        """Create a new run with fresh state."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state = AgentState.initialize(run_id)

        # Create run directory structure
        run_dir = self.base_dir / run_id
        (run_dir / "frames").mkdir(parents=True, exist_ok=True)
        (run_dir / "calls").mkdir(parents=True, exist_ok=True)

        # Save initial state
        self.store.save(state)

        return state

    def load_or_create_run(self, run_id: str | None = None) -> AgentState:
        """Load existing run or create new one."""
        if run_id and self.store.exists(run_id):
            state = self.store.load(run_id)
            if state:
                return state

        return self.create_new_run()

    def save_state(self, state: AgentState) -> None:
        """Save current state."""
        self.store.save(state)

    def update_from_analysis(
        self,
        state: AgentState,
        action_id: ActionID,
        analysis: ActionAnalysisResult,
        diff: DiffResult,
        before_path: str,
        after_path: str,
    ) -> AgentState:
        """
        Update state based on action analysis result.

        This is the main knowledge update method called after each action.
        """
        knowledge = state.get_action_knowledge(action_id)

        # Create observation record with full context
        observation = ActionObservation(
            before_frame_path=before_path,
            after_frame_path=after_path,
            diff_summary=diff.change_summary,
            llm_interpretation=analysis.interpretation,
            context_description=analysis.context_description,
            had_effect=analysis.had_effect,
            context_that_caused_outcome=analysis.context_that_caused_this_outcome,
            object_changes=diff.object_changes,
        )

        # Only update definition if LLM chose to update it
        new_definition = None
        if analysis.update_definition and analysis.new_definition:
            new_definition = analysis.new_definition

        # Update action knowledge
        knowledge.add_observation(
            observation=observation,
            new_definition=new_definition,
            is_consistent=analysis.is_consistent_with_previous,
        )

        # Update environment knowledge
        for env_update in analysis.environment_updates:
            state.environment.add_observation(env_update)

        # Update objects if mentioned
        for obj_name in analysis.objects_involved:
            state.environment.add_object(
                name=obj_name, description=f"Involved in {action_id.value} action"
            )

        return state

    def build_conversation_history(
        self,
        action_knowledge: dict[str, ActionKnowledge],
        max_tokens: int = 200000,
    ) -> list[dict]:
        """
        Build full conversation history for in-context learning.

        Returns a list of message dicts (role: user/assistant) containing
        all previous action analyses, sorted by timestamp.
        This enables the LLM to learn from the entire session history.

        Args:
            action_knowledge: Dict of all action knowledge
            max_tokens: Approximate token limit (chars / 4)

        Returns:
            List of message dicts for conversation history
        """
        # Collect all observations with their full context
        all_observations = []
        for action_id_str, knowledge in action_knowledge.items():
            for obs in knowledge.observations:
                all_observations.append({
                    "action_id": action_id_str,
                    "timestamp": obs.timestamp,
                    "observation": obs,
                    "definition_at_time": knowledge.current_definition,
                })

        if not all_observations:
            return []

        # Sort by timestamp (oldest first for conversation order)
        all_observations.sort(key=lambda x: x["timestamp"])

        # Build conversation messages
        messages = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars from tokens

        for item in all_observations:
            obs = item["observation"]
            action_id = item["action_id"]

            # User message: what was the situation
            user_content = f"""[Previous Analysis] Action: {action_id}

Context: {obs.context_description}
Changes: {obs.diff_summary}
Objects changed: {obs.object_changes if obs.object_changes else 'Not recorded'}"""

            # Assistant message: what the LLM concluded
            effect_str = "HAD EFFECT" if obs.had_effect else "NO EFFECT"
            assistant_content = f"""[My Analysis] {effect_str}

Interpretation: {obs.llm_interpretation}
Why this outcome: {obs.context_that_caused_outcome if obs.context_that_caused_outcome else 'Not recorded'}
Consistent: {obs.was_consistent}"""

            # Check if adding these would exceed limit
            msg_chars = len(user_content) + len(assistant_content)
            if total_chars + msg_chars > max_chars:
                break

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
            total_chars += msg_chars

        return messages

    def format_for_action_analysis(
        self,
        action_id: ActionID,
        knowledge: ActionKnowledge,
        environment: EnvironmentKnowledge,
        diff: DiffResult,
    ) -> dict[str, str]:
        """Format knowledge for action analysis prompt."""
        is_first = len(knowledge.observations) == 0

        return {
            "action_id": action_id.value,
            "is_first_observation": str(is_first),
            "current_definition": knowledge.current_definition or "None",
            "observation_history": self.formatter.format_observation_history(
                knowledge.observations, limit=3
            ),
            "environment": self.formatter.format_environment(environment),
            "diff_summary": diff.change_summary if diff.has_changes else "NO CHANGES",
            "pixel_changes": self._format_pixel_changes(diff) if diff.has_changes else "",
            # ASCII grid representations
            "before_ascii": diff.before_ascii,
            "after_ascii": diff.after_ascii,
            "diff_ascii": diff.diff_ascii,
            # Object-level analysis
            "before_objects": diff.before_objects,
            "after_objects": diff.after_objects,
            "object_changes": diff.object_changes,
        }

    def format_for_next_action(
        self,
        action_knowledge: dict[str, ActionKnowledge],
        environment: EnvironmentKnowledge,
    ) -> dict[str, str]:
        """Format knowledge for next action suggestion prompt."""
        # Track recent no-effect observations to help avoid repeating
        no_effect_info = self._format_recent_no_effects(action_knowledge)

        # Format recent action history for better context
        action_history = self._format_recent_action_history(action_knowledge)

        # Format no-op avoidance warning (only for actions with 3+ attempts)
        no_op_avoidance = self._format_no_op_avoidance_warning(action_knowledge)

        return {
            "action_status": self.formatter.format_all_action_knowledge(
                action_knowledge
            ),
            "environment": self.formatter.format_environment(environment),
            "verified_actions": ", ".join(
                k for k, v in action_knowledge.items() if v.is_verified
            )
            or "None",
            "pending_actions": ", ".join(
                k for k, v in action_knowledge.items() if v.needs_more_observations()
            )
            or "None",
            "recent_no_effects": no_effect_info,
            "recent_action_history": action_history,
            "no_op_avoidance_warning": no_op_avoidance,
        }

    def _format_recent_action_history(
        self, action_knowledge: dict[str, ActionKnowledge], limit: int = 10
    ) -> str:
        """Format recent action history across all actions for context."""
        # Collect all observations with timestamps
        all_observations = []
        for action_id, knowledge in action_knowledge.items():
            for obs in knowledge.observations:
                all_observations.append({
                    "action_id": action_id,
                    "timestamp": obs.timestamp,
                    "had_effect": obs.had_effect,
                    "interpretation": obs.llm_interpretation[:100] if obs.llm_interpretation else "Unknown",
                })

        if not all_observations:
            return "No actions taken yet."

        # Sort by timestamp (most recent first) and take last N
        all_observations.sort(key=lambda x: x["timestamp"], reverse=True)
        recent = all_observations[:limit]

        lines = []
        for i, obs in enumerate(reversed(recent), 1):  # Show oldest first
            effect = "✓ EFFECT" if obs["had_effect"] else "✗ NO EFFECT"
            lines.append(
                f"  {i}. {obs['action_id']}: {effect} - {obs['interpretation'][:60]}..."
            )

        return "\n".join(lines)

    def _format_recent_no_effects(
        self, action_knowledge: dict[str, ActionKnowledge]
    ) -> str:
        """Format recent no-effect observations to help avoid repeating."""
        no_effects = []
        for action_id, knowledge in action_knowledge.items():
            # Check last 3 observations for no-effect
            recent_no_effect = 0
            for obs in knowledge.observations[-3:]:
                if not obs.had_effect:
                    recent_no_effect += 1

            if recent_no_effect > 0:
                no_effects.append(
                    f"{action_id}: {recent_no_effect} recent no-effect attempts"
                )

        if not no_effects:
            return "None"

        return "; ".join(no_effects)

    def _format_no_op_avoidance_warning(
        self, action_knowledge: dict[str, ActionKnowledge]
    ) -> str:
        """
        Format no-op avoidance warning for actions with 3+ attempts.

        Only includes this warning after 3 attempts for an action, as requested.
        This helps the LLM avoid suggesting actions that are likely to be no-ops
        in the current context.
        """
        warnings = []
        for action_id, knowledge in action_knowledge.items():
            # Only include warning after 3+ verification attempts
            if knowledge.verification_attempts < 3:
                continue

            # Check if this action has recent no-effects
            if knowledge.consecutive_no_effects > 0:
                # Get the last no-effect reason if available
                last_no_effect_reason = None
                for obs in reversed(knowledge.observations):
                    if not obs.had_effect:
                        # Try to get the context that caused no-op
                        if obs.context_that_caused_outcome:
                            last_no_effect_reason = obs.context_that_caused_outcome
                        break

                warning = f"{action_id}: {knowledge.consecutive_no_effects} consecutive no-effects"
                if last_no_effect_reason:
                    warning += f" (last reason: {last_no_effect_reason[:80]}...)"
                warnings.append(warning)

        if not warnings:
            return ""

        # Only return a full warning section if there are warnings
        return (
            "AVOID EXPECTED NO-OPS: If you believe an action will cause a no-op "
            "in the current state (based on your understanding of the environment), "
            "please choose a different action or use a setup sequence to change position first.\n"
            "Actions with recent no-ops:\n  " + "\n  ".join(warnings)
        )

    def _format_pixel_changes(self, diff: DiffResult, limit: int = 30) -> str:
        """Format pixel changes for prompt."""
        if not diff.changed_pixels:
            return ""

        changes = diff.changed_pixels[:limit]
        formatted = [
            f"({c.row},{c.col}): {c.old_value}->{c.new_value}" for c in changes
        ]

        result = ", ".join(formatted)
        if len(diff.changed_pixels) > limit:
            result += f" ... and {len(diff.changed_pixels) - limit} more"

        return result

    def format_environment_for_analysis(
        self, environment: EnvironmentKnowledge
    ) -> str:
        """Format environment knowledge for the environment analysis prompt."""
        sections = []

        # Background and boundaries
        if environment.background_color:
            sections.append(f"Background: {environment.background_color}")

        if environment.has_border:
            border_info = f"Border: Yes ({environment.border_color or 'unknown color'})"
            if environment.border_description:
                border_info += f" - {environment.border_description}"
            sections.append(border_info)
        else:
            sections.append("Border: Unknown or none detected")

        # Internal walls
        if environment.internal_walls:
            sections.append("Internal walls/obstacles:")
            for wall in environment.internal_walls:
                sections.append(f"  - {wall}")

        # Movement constraints
        if environment.movement_constraints:
            sections.append("Known movement constraints:")
            for constraint in environment.movement_constraints:
                sections.append(f"  - {constraint}")

        # Identified objects
        if environment.identified_objects:
            sections.append("Identified objects:")
            for obj in environment.identified_objects:
                obj_info = f"  - {obj.get('name', 'Unknown')}: {obj.get('color', '?')} {obj.get('shape', '?')}"
                if obj.get('role_hypothesis'):
                    obj_info += f" (Role: {obj['role_hypothesis']})"
                sections.append(obj_info)

        # Spatial structure
        if environment.spatial_structure:
            sections.append(f"Spatial structure: {environment.spatial_structure}")

        if not sections:
            return "No environment understanding yet. This is your first analysis."

        return "\n".join(sections)

    def build_environment_history(
        self,
        environment: EnvironmentKnowledge,
        max_tokens: int = 100000,
    ) -> list[dict]:
        """
        Build conversation history for environment analysis.

        Returns user/assistant message pairs representing previous
        environment analyses and breakthroughs.
        """
        messages = []
        total_chars = 0
        max_chars = max_tokens * 4

        # Include breakthroughs as key learning moments
        for i, breakthrough in enumerate(environment.breakthroughs):
            user_msg = f"[Previous Environment Analysis {i+1}]"
            assistant_msg = f"BREAKTHROUGH: {breakthrough}"

            msg_chars = len(user_msg) + len(assistant_msg)
            if total_chars + msg_chars > max_chars:
                break

            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
            total_chars += msg_chars

        # Include movement constraints as learned facts
        for constraint in environment.movement_constraints:
            user_msg = "[Environment observation]"
            assistant_msg = f"Movement constraint discovered: {constraint}"

            msg_chars = len(user_msg) + len(assistant_msg)
            if total_chars + msg_chars > max_chars:
                break

            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
            total_chars += msg_chars

        return messages

    def update_environment_from_analysis(
        self,
        state: "AgentState",
        analysis: "EnvironmentAnalysisResult",
    ) -> "AgentState":
        """Update environment knowledge from an environment analysis result."""
        env = state.environment

        # Update basic info
        if analysis.background_color:
            env.background_color = analysis.background_color

        # Update boundary info
        env.has_border = analysis.boundaries.has_border
        env.border_color = analysis.boundaries.border_color
        env.border_description = analysis.boundaries.border_description

        # Merge internal walls (don't duplicate)
        for wall in analysis.boundaries.internal_walls:
            if wall not in env.internal_walls:
                env.internal_walls.append(wall)

        # Merge movement constraints
        for constraint in analysis.movement_constraints:
            env.add_movement_constraint(constraint)

        # Update identified objects (replace with latest understanding)
        if analysis.objects_identified:
            env.identified_objects = [
                obj.model_dump() for obj in analysis.objects_identified
            ]

        # Update spatial structure
        if analysis.spatial_structure:
            env.spatial_structure = analysis.spatial_structure

        # Record breakthroughs
        for breakthrough in analysis.breakthroughs:
            env.add_breakthrough(breakthrough)

        # Update open questions
        env.open_questions = analysis.open_questions

        # Update domain description (always take latest for evolving understanding)
        if analysis.domain_description:
            env.domain_description = analysis.domain_description

        # Update unexplored elements (replace with latest assessment)
        env.unexplored_elements = analysis.unexplored_elements

        # Apply suggested action updates
        for update in analysis.suggested_action_updates:
            action_id_str = update.action_id
            if action_id_str in state.action_knowledge:
                action_knowledge = state.action_knowledge[action_id_str]
                # Only update if we have a new suggested definition
                if update.suggested_definition:
                    action_knowledge.current_definition = update.suggested_definition

        # Increment analysis count
        env.analysis_count += 1

        return state
