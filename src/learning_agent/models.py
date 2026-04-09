"""
Data models for the Learning Agent.

This module defines all Pydantic models used throughout the learning agent,
following a clean separation of concerns and enabling easy serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionID(str, Enum):
    """Available actions in the game (ACTION1-5 only for learning agent)."""

    ACTION1 = "ACTION1"  # Semantically: up
    ACTION2 = "ACTION2"  # Semantically: down
    ACTION3 = "ACTION3"  # Semantically: left
    ACTION4 = "ACTION4"  # Semantically: right
    ACTION5 = "ACTION5"  # Semantically: interact/select/rotate


# Semantic meanings for LLM context
ACTION_SEMANTICS = {
    ActionID.ACTION1: "Move UP (decrease row/y)",
    ActionID.ACTION2: "Move DOWN (increase row/y)",
    ActionID.ACTION3: "Move LEFT (decrease column/x)",
    ActionID.ACTION4: "Move RIGHT (increase column/x)",
    ActionID.ACTION5: "INTERACT/SELECT/ROTATE at current position",
}


# ============================================================================
# Observation Models
# ============================================================================


class ActionObservation(BaseModel):
    """A single observation of what an action did."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    before_frame_path: str
    after_frame_path: str
    diff_summary: str
    llm_interpretation: str
    context_description: str
    had_effect: bool
    was_consistent: Optional[bool] = None  # None for first observation
    context_that_caused_outcome: str = ""  # Why this outcome in this context
    object_changes: str = ""  # Object-level changes description

    class Config:
        frozen = False


class ActionKnowledge(BaseModel):
    """Everything we know about a single action."""

    action_id: ActionID
    current_definition: Optional[str] = None
    observations: list[ActionObservation] = Field(default_factory=list)
    verification_attempts: int = 0  # Only counts observations with effect
    is_verified: bool = False
    consecutive_no_effects: int = 0  # Track consecutive no-effect observations
    is_exhausted: bool = False  # True if we've given up on this action (8+ consecutive no-effects)

    def add_observation(
        self,
        observation: ActionObservation,
        new_definition: Optional[str],
        is_consistent: Optional[bool],
    ) -> None:
        """Add an observation and update state."""
        observation.was_consistent = is_consistent
        self.observations.append(observation)

        # Only update definition if a new one was provided
        if new_definition is not None:
            self.current_definition = new_definition

        # Always count this as an attempt (every observation counts)
        self.verification_attempts += 1

        # Track consecutive no-effects (for exhaustion detection)
        if observation.had_effect:
            # Reset consecutive no-effect counter
            self.consecutive_no_effects = 0
        else:
            # Increment consecutive no-effect counter
            self.consecutive_no_effects += 1
            # Mark as exhausted if we've hit 8 consecutive no-effects
            if self.consecutive_no_effects >= 8:
                self.is_exhausted = True

        # Check for verification: 3 consecutive consistent observations (including expected no-ops)
        if is_consistent:
            if self._count_recent_consistent() >= 3:
                self.is_verified = True

    def _count_recent_consistent(self) -> int:
        """Count recent consecutive consistent observations (including expected no-ops)."""
        count = 0
        for obs in reversed(self.observations):
            # Count ALL observations that the LLM marked as consistent
            # This includes expected no-ops (e.g., hitting a wall again is consistent behavior)
            if obs.was_consistent:
                count += 1
            elif obs.was_consistent is False:
                break  # Reset on explicit inconsistency
            # Note: was_consistent=None (first observation) doesn't break the streak
        return count

    def needs_more_observations(self) -> bool:
        """Check if this action needs more observations."""
        if self.is_verified:
            return False
        if self.is_exhausted:
            return False  # Stop testing after 8 consecutive no-effects
        return self.verification_attempts < 8


# ============================================================================
# Environment Models
# ============================================================================


class ObjectHypothesis(BaseModel):
    """A hypothesis about an object in the environment."""

    name: str
    description: str
    first_seen: str = Field(default_factory=lambda: datetime.now().isoformat())
    observations: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    def update(self, new_observation: str, new_confidence: float) -> None:
        """Update the hypothesis with new information."""
        self.observations.append(new_observation)
        self.confidence = new_confidence


class EnvironmentKnowledge(BaseModel):
    """Everything we know about the game environment."""

    objects: list[ObjectHypothesis] = Field(default_factory=list)
    spatial_rules: list[str] = Field(default_factory=list)
    general_observations: list[str] = Field(default_factory=list)
    iteration_history: list[str] = Field(default_factory=list)

    # Structured environment analysis (from dedicated LLM calls)
    background_color: str = ""
    has_border: bool = False
    border_color: Optional[str] = None
    border_description: str = ""
    internal_walls: list[str] = Field(default_factory=list)
    movement_constraints: list[str] = Field(default_factory=list)
    identified_objects: list[dict] = Field(default_factory=list)  # Stores IdentifiedObject as dicts
    spatial_structure: str = ""
    breakthroughs: list[str] = Field(default_factory=list)  # Key discoveries
    open_questions: list[str] = Field(default_factory=list)
    analysis_count: int = 0  # How many environment analyses have been done

    # High-level domain understanding
    domain_description: str = ""  # Conceptual description of game mechanics
    unexplored_elements: list[str] = Field(default_factory=list)  # Colors/sections not yet analyzed

    def add_observation(self, observation: str) -> None:
        """Add a general observation about the environment."""
        if observation not in self.general_observations:
            self.general_observations.append(observation)
            self.iteration_history.append(
                f"[{datetime.now().isoformat()}] {observation}"
            )

    def add_object(self, name: str, description: str) -> None:
        """Add or update an object hypothesis."""
        for obj in self.objects:
            if obj.name.lower() == name.lower():
                obj.observations.append(description)
                return
        self.objects.append(ObjectHypothesis(name=name, description=description))

    def add_spatial_rule(self, rule: str) -> None:
        """Add a spatial rule if not already present."""
        if rule not in self.spatial_rules:
            self.spatial_rules.append(rule)

    def add_movement_constraint(self, constraint: str) -> None:
        """Add a movement constraint if not already present."""
        if constraint not in self.movement_constraints:
            self.movement_constraints.append(constraint)

    def add_breakthrough(self, breakthrough: str) -> None:
        """Add a breakthrough discovery."""
        if breakthrough not in self.breakthroughs:
            self.breakthroughs.append(breakthrough)
            self.iteration_history.append(
                f"[{datetime.now().isoformat()}] BREAKTHROUGH: {breakthrough}"
            )


# ============================================================================
# Diff Models
# ============================================================================


class PixelChange(BaseModel):
    """A single pixel change between frames."""

    row: int
    col: int
    old_value: int
    new_value: int


class DiffResult(BaseModel):
    """Result of comparing two frames."""

    changed_pixels: list[PixelChange] = Field(default_factory=list)
    change_summary: str = ""
    has_changes: bool = False
    change_regions: list[str] = Field(default_factory=list)  # Human-readable regions

    # ASCII grid representations for better spatial understanding
    before_ascii: str = ""  # ASCII grid of before frame
    after_ascii: str = ""   # ASCII grid of after frame
    diff_ascii: str = ""    # ASCII grid highlighting changes (. for unchanged, X for changed)

    # Object-level analysis (Gestalt grouping)
    before_objects: str = ""  # Description of objects before action
    after_objects: str = ""   # Description of objects after action
    object_changes: str = ""  # Object-level diff (appeared, disappeared, moved, etc.)


# ============================================================================
# LLM Response Models
# ============================================================================


class ActionAnalysisResult(BaseModel):
    """Result of analyzing what an action did (LLM Call 1)."""

    interpretation: str = Field(description="What happened when the action was taken")
    update_definition: bool = Field(
        default=True, description="Whether to update the definition"
    )
    new_definition: Optional[str] = Field(
        default=None, description="New definition if update_definition is True"
    )
    is_consistent_with_previous: Optional[bool] = Field(
        default=None, description="None if first observation, else bool"
    )
    context_that_caused_this_outcome: str = Field(
        default="", description="What context caused this specific outcome"
    )
    objects_involved: list[str] = Field(
        default_factory=list, description="Objects that were affected"
    )
    context_description: str = Field(
        description="Description of the game state context"
    )
    had_effect: bool = Field(description="Whether the action changed anything")
    no_effect_reason: Optional[str] = Field(
        default=None, description="If no effect, why?"
    )
    environment_updates: list[str] = Field(
        default_factory=list, description="New observations about the environment"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

    @property
    def action_definition(self) -> Optional[str]:
        """Get the definition to use (for backwards compatibility)."""
        return self.new_definition if self.update_definition else None


class NextActionSuggestion(BaseModel):
    """LLM's suggestion for what action to take next (LLM Call 2)."""

    target_action: ActionID = Field(description="The action to test")
    setup_sequence: list[ActionID] = Field(
        default_factory=list, description="Actions to execute first"
    )
    reasoning: str = Field(description="Why this action/sequence")
    expected_information_gain: str = Field(description="What we hope to learn")
    current_board_assessment: str = Field(description="Assessment of current state")


class BoundaryInfo(BaseModel):
    """Information about environment boundaries."""

    has_border: bool = False
    border_color: Optional[str] = None
    border_description: str = ""
    internal_walls: list[str] = Field(default_factory=list)


class IdentifiedObject(BaseModel):
    """An object identified in the environment."""

    name: str
    color: str
    shape: str
    role_hypothesis: str = ""
    evidence_for_role: str = ""


class SuggestedActionUpdate(BaseModel):
    """A suggested update to an action definition based on environment understanding."""

    action_id: str  # ACTION1-5
    suggested_definition: str  # Simple, conceptual definition
    reasoning: str  # Why this update makes sense given environment breakthroughs


class EnvironmentAnalysisResult(BaseModel):
    """Result of analyzing the environment (LLM Call 3)."""

    background_color: str = Field(description="Background/empty space color")
    boundaries: BoundaryInfo = Field(default_factory=BoundaryInfo)
    objects_identified: list[IdentifiedObject] = Field(default_factory=list)
    spatial_structure: str = Field(default="", description="Spatial layout description")
    movement_constraints: list[str] = Field(
        default_factory=list, description="Observed movement constraints"
    )
    breakthroughs: list[str] = Field(
        default_factory=list, description="New discoveries this observation"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Questions to investigate"
    )
    suggested_action_updates: list[SuggestedActionUpdate] = Field(
        default_factory=list,
        description="Suggested updates to action definitions based on environment understanding"
    )
    domain_description: str = Field(
        default="",
        description="High-level conceptual description of the game domain and mechanics"
    )
    unexplored_elements: list[str] = Field(
        default_factory=list,
        description="Colors, sections, or features not yet thoroughly analyzed"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class EnvironmentObservation(BaseModel):
    """A single environment analysis observation."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    frame_path: str = ""
    analysis: EnvironmentAnalysisResult = Field(default_factory=EnvironmentAnalysisResult)
    action_context: str = ""  # What action just happened (for context)


# ============================================================================
# Agent State
# ============================================================================


class StageInfo(BaseModel):
    """Information about a game stage/level."""

    stage_number: int  # 0-indexed stage number
    started_at_action: int  # Which action count this stage started
    entry_score: int  # Score when entering this stage
    observations_count: int = 0  # How many observations in this stage
    key_discoveries: list[str] = Field(default_factory=list)  # Notable discoveries in this stage


class AgentState(BaseModel):
    """Complete state of the learning agent - fully serializable."""

    run_id: str
    action_knowledge: dict[str, ActionKnowledge] = Field(default_factory=dict)
    environment: EnvironmentKnowledge = Field(default_factory=EnvironmentKnowledge)
    current_frame_path: Optional[str] = None
    previous_frame_path: Optional[str] = None
    action_count: int = 0
    llm_call_count: int = 0
    phase: str = "exploration"  # exploration | complete

    # Stage/level tracking
    current_score: int = 0  # Current game score (0-254)
    previous_score: int = 0  # Score from last frame
    current_stage: int = 0  # Current stage number (0-indexed)
    stage_history: list[StageInfo] = Field(default_factory=list)  # History of all stages
    actions_since_stage_change: int = 0  # Actions taken in current stage
    just_transitioned_stage: bool = False  # True if we JUST entered a new stage

    # Available actions (from API)
    available_actions: list[str] = Field(default_factory=list)  # Empty = all actions available

    @classmethod
    def initialize(cls, run_id: str) -> "AgentState":
        """Create a fresh agent state with empty knowledge for all actions."""
        state = cls(run_id=run_id)
        for action_id in ActionID:
            state.action_knowledge[action_id.value] = ActionKnowledge(
                action_id=action_id
            )
        # Initialize first stage
        state.stage_history.append(StageInfo(
            stage_number=0,
            started_at_action=0,
            entry_score=0,
        ))
        return state

    def update_score(self, new_score: int) -> bool:
        """
        Update the score and detect stage transitions.

        Returns True if a stage transition occurred.
        """
        self.previous_score = self.current_score
        self.current_score = new_score
        self.just_transitioned_stage = False

        # Detect stage transition: score increased
        if new_score > self.previous_score:
            self.current_stage += 1
            self.actions_since_stage_change = 0
            self.just_transitioned_stage = True

            # Record new stage
            self.stage_history.append(StageInfo(
                stage_number=self.current_stage,
                started_at_action=self.action_count,
                entry_score=new_score,
            ))

            return True

        self.actions_since_stage_change += 1
        return False

    def get_stage_context(self) -> str:
        """Get a formatted string describing current stage context."""
        if not self.stage_history:
            return "Stage tracking not initialized."

        current_stage_info = self.stage_history[-1]
        lines = [
            f"=== STAGE/LEVEL INFO ===",
            f"Current Stage: {self.current_stage + 1} (score: {self.current_score}/254)",
            f"Actions in this stage: {self.actions_since_stage_change}",
        ]

        if self.just_transitioned_stage:
            lines.append("")
            lines.append("⚠️  **JUST ENTERED NEW STAGE** ⚠️")
            lines.append("WARNING: This new stage may introduce:")
            lines.append("  - New object types or colors not seen before")
            lines.append("  - Different mechanics or rules")
            lines.append("  - New movement constraints or interactions")
            lines.append("  - Modified behavior for existing actions")
            lines.append("IMPORTANT: Re-evaluate your understanding of the environment!")
            lines.append("Previous assumptions may no longer be valid.")

        if len(self.stage_history) > 1:
            lines.append("")
            lines.append("Stage History:")
            for stage in self.stage_history[-5:]:  # Last 5 stages
                lines.append(
                    f"  Stage {stage.stage_number + 1}: started at action {stage.started_at_action}, "
                    f"score={stage.entry_score}"
                )

        return "\n".join(lines)

    def get_action_knowledge(self, action_id: ActionID) -> ActionKnowledge:
        """Get knowledge for a specific action."""
        return self.action_knowledge[action_id.value]

    def should_terminate(self) -> bool:
        """Check if exploration phase is complete."""
        for knowledge in self.action_knowledge.values():
            if knowledge.needs_more_observations():
                return False
        return True

    def get_unverified_actions(self) -> list[ActionID]:
        """Get list of actions that still need verification (excludes exhausted)."""
        return [
            ActionID(action_id)
            for action_id, knowledge in self.action_knowledge.items()
            if knowledge.needs_more_observations()
        ]

    def get_exhausted_actions(self) -> list[ActionID]:
        """Get list of exhausted actions (8+ consecutive no-effects)."""
        return [
            ActionID(action_id)
            for action_id, knowledge in self.action_knowledge.items()
            if knowledge.is_exhausted
        ]

    def get_verified_actions(self) -> list[ActionID]:
        """Get list of verified actions (available for setup sequences)."""
        return [
            ActionID(action_id)
            for action_id, knowledge in self.action_knowledge.items()
            if knowledge.is_verified
        ]
