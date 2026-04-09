"""
Shared World Model: the single source of truth about the game world.

This is the central document that BOTH the exploration LLM and the
synthesis backend read and write. It lives on disk as a structured
markdown file (world_model.md) in the run directory.

The world model contains:
  - Goal hypothesis (what we think the objective is)
  - Object catalog (detected types with roles and relationships)
  - Action effects (what each action does to each object)
  - Object relationships (spatial, causal, conditional)
  - Environment structure (boundaries, layout, spatial rules)
  - Open questions (what we still need to figure out)
  - Synthesis feedback (what the code synthesizer discovered)

Two writers:
  1. Exploration LLM: updates after each action analysis and environment analysis.
     Writes observations, hypotheses, relationship descriptions, goal guesses.
  2. Synthesis backend: writes back ONLY at the end of a reflexion run.
     - If 100% accuracy: definitive updates (confirmed facts).
     - If stuck: tentative updates + new open questions to guide exploration.

The document format is structured markdown so both LLMs and humans can
read it naturally.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ObjectEntry:
    """One entry in the object catalog."""
    type_id: int
    name: str
    colors: list[str]                # e.g. ["RGB(255,0,0)"]
    size: str                        # e.g. "3x3"
    role: str = ""                   # e.g. "player", "wall", "ball", "goal"
    role_confidence: float = 0.0
    behavior_summary: str = ""       # e.g. "moves right 1px per action"
    relationship_to_goal: str = ""   # e.g. "must reach this to complete level"
    observations: list[str] = field(default_factory=list)
    is_static: bool = False
    is_player: bool = False


@dataclass
class ActionEntry:
    """What we know about one action."""
    action_id: int
    definition: str = ""             # NL description from exploration LLM
    effects_by_type: dict[str, str] = field(default_factory=dict)  # type_name → effect desc
    preconditions: list[str] = field(default_factory=list)         # known preconditions
    confidence: float = 0.0
    observation_count: int = 0


@dataclass
class RelationshipEntry:
    """A known or hypothesized relationship between two object types."""
    type_a: str
    type_b: str
    relation: str                    # e.g. "blocks", "pushes", "destroys", "bounces_off"
    description: str = ""
    confidence: float = 0.0
    confirmed_by_code: bool = False  # True if synthesis verified this
    observations: list[str] = field(default_factory=list)


@dataclass
class GoalHypothesis:
    """A hypothesis about the game's objective."""
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    subgoals: list[str] = field(default_factory=list)


@dataclass
class SynthesisFeedback:
    """Feedback from the synthesis backend to guide exploration."""
    timestamp: str = ""
    accuracy: float = 0.0
    is_definitive: bool = False      # True if 100% accuracy
    confirmed_facts: list[str] = field(default_factory=list)
    tentative_findings: list[str] = field(default_factory=list)
    questions_for_exploration: list[str] = field(default_factory=list)
    code_summary: str = ""           # what the code does


class SharedWorldModel:
    """
    The shared world model that both exploration and synthesis operate on.

    Persists to disk as world_model.json (structured) and world_model.md
    (human/LLM readable).
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Core state
        self.goal: GoalHypothesis | None = None
        self.objects: dict[int, ObjectEntry] = {}      # type_id → ObjectEntry
        self.actions: dict[int, ActionEntry] = {}       # action_id → ActionEntry
        self.relationships: list[RelationshipEntry] = []
        self.open_questions: list[str] = []
        self.breakthroughs: list[str] = []
        self.environment_description: str = ""
        self.background_color: str = ""
        self.boundary_description: str = ""
        self.synthesis_feedback: list[SynthesisFeedback] = []

        # Game metadata
        self.win_levels: int = 0             # total levels in the game
        self.goal_reached_count: int = 0     # how many times goal was reached on this level
        self.goal_reaching_actions: list[dict] = []  # records of what caused goal completion

        # Load if exists
        if self.json_path.exists():
            self._load()

    @property
    def json_path(self) -> Path:
        return self.run_dir / "world_model.json"

    @property
    def md_path(self) -> Path:
        return self.run_dir / "world_model.md"

    # =========================================================================
    # Exploration LLM writes
    # =========================================================================

    def update_from_exploration(
        self,
        action_knowledge: dict[str, Any],
        environment: Any,
        sprite_registry: Any = None,
        epistemic_state: Any = None,
    ) -> None:
        """
        Called by the exploration engine after action/environment analysis.
        Merges the LLM's understanding into the shared model.
        """
        # Update environment description
        if hasattr(environment, 'domain_description') and environment.domain_description:
            self.environment_description = environment.domain_description
        if hasattr(environment, 'background_color') and environment.background_color:
            self.background_color = str(environment.background_color)
        if hasattr(environment, 'border_description') and environment.border_description:
            self.boundary_description = environment.border_description

        # Update objects from environment analysis
        if hasattr(environment, 'identified_objects'):
            for obj_dict in environment.identified_objects:
                name = obj_dict.get('name', '') if isinstance(obj_dict, dict) else getattr(obj_dict, 'name', '')
                role = obj_dict.get('role_hypothesis', '') if isinstance(obj_dict, dict) else getattr(obj_dict, 'role_hypothesis', '')
                color = obj_dict.get('color', '') if isinstance(obj_dict, dict) else getattr(obj_dict, 'color', '')

                # Try to match to existing entry by name
                matched = False
                for entry in self.objects.values():
                    if entry.name.lower() == name.lower():
                        if role:
                            entry.role = role
                        matched = True
                        break

                if not matched and name:
                    # Create placeholder — will be matched to type_id later
                    pass

        # Update objects from sprite registry (deterministic detection)
        if sprite_registry:
            for type_id, stype in sprite_registry.types.items():
                if type_id not in self.objects:
                    self.objects[type_id] = ObjectEntry(
                        type_id=type_id,
                        name=stype.name,
                        colors=[str(c) for c in stype.colors],
                        size=f"{stype.canonical_width}x{stype.canonical_height}",
                        is_static=stype.is_static or False,
                        is_player=stype.is_player or False,
                    )
                else:
                    entry = self.objects[type_id]
                    entry.name = stype.name
                    entry.is_static = stype.is_static or False
                    entry.is_player = stype.is_player or False

        # Update action knowledge from exploration LLM
        for action_key, knowledge in action_knowledge.items():
            try:
                aid = int(action_key.replace("ACTION", ""))
            except (ValueError, AttributeError):
                continue

            if aid not in self.actions:
                self.actions[aid] = ActionEntry(action_id=aid)

            entry = self.actions[aid]
            if hasattr(knowledge, 'current_definition') and knowledge.current_definition:
                entry.definition = knowledge.current_definition
            if hasattr(knowledge, 'observations'):
                entry.observation_count = len(knowledge.observations)

        # Update from epistemic state (deterministic observations)
        if epistemic_state:
            for (type_id, action_id), effect in epistemic_state.effects.items():
                if type_id in self.objects and action_id in self.actions:
                    type_name = self.objects[type_id].name
                    self.actions[action_id].effects_by_type[type_name] = effect.describe()
                    self.actions[action_id].confidence = max(
                        self.actions[action_id].confidence,
                        effect.confidence,
                    )

        # Update goal hypothesis from NL LLM's environment analysis
        if hasattr(environment, 'goal_hypothesis') and environment.goal_hypothesis:
            self.update_goal_hypothesis(
                description=environment.goal_hypothesis,
                evidence=getattr(environment, 'goal_evidence', []) or [],
                confidence=getattr(environment, 'goal_confidence', 0.0),
                subgoals=getattr(environment, 'goal_conditions', []) or [],
            )
            # Update object roles for goal
            for role_desc in getattr(environment, 'object_roles_for_goal', []) or []:
                for obj in self.objects.values():
                    if obj.name.lower() in role_desc.lower():
                        obj.relationship_to_goal = role_desc

        # Update open questions from environment
        if hasattr(environment, 'open_questions'):
            for q in environment.open_questions:
                if q not in self.open_questions:
                    self.open_questions.append(q)

        # Update breakthroughs
        if hasattr(environment, 'breakthroughs'):
            for b in environment.breakthroughs:
                if b not in self.breakthroughs:
                    self.breakthroughs.append(b)

        self._save()

    def update_goal_hypothesis(
        self,
        description: str,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
        subgoals: list[str] | None = None,
    ) -> None:
        """Update the goal hypothesis (called by exploration LLM)."""
        self.goal = GoalHypothesis(
            description=description,
            evidence=evidence or [],
            confidence=confidence,
            subgoals=subgoals or [],
        )
        self._save()

    def record_goal_reached(
        self,
        level_index: int,
        action_id: int | None = None,
        transition_count: int = 0,
        previous_world: Any = None,
    ) -> None:
        """
        Called when the game signals level completion (goal reached).

        Records what state/action caused the goal, so both the exploration
        LLM and synthesizer can reason about the goal condition.
        """
        self.goal_reached_count += 1

        record = {
            "level_index": level_index,
            "action_id": action_id,
            "transition_count": transition_count,
            "attempt": self.goal_reached_count,
        }

        # Capture what objects were where when goal was reached
        if previous_world is not None:
            record["objects_at_goal"] = []
            for s in previous_world.sprites:
                record["objects_at_goal"].append({
                    "type_name": s.type_name,
                    "position": s.position,
                    "track_id": s.track_id,
                })

        self.goal_reaching_actions.append(record)

        # Add as breakthrough
        action_str = f"ACTION{action_id}" if action_id else "unknown action"
        self.breakthroughs.append(
            f"[GOAL REACHED] Level {level_index} completed via {action_str} "
            f"after {transition_count} transitions (attempt #{self.goal_reached_count})"
        )

        # If we have goal-reaching data, we should update the goal hypothesis
        # based on what objects were in what positions when the goal was reached.
        # This is left for the exploration LLM to analyze — we just record the facts.
        if not self.goal:
            self.goal = GoalHypothesis(
                description="Unknown — level was completed but goal condition not yet analyzed",
                evidence=[f"Level completed via {action_str}"],
                confidence=0.2,
            )
        else:
            self.goal.evidence.append(
                f"Level {level_index} completed via {action_str} "
                f"(attempt #{self.goal_reached_count})"
            )

        self._save()

    def add_relationship(
        self,
        type_a: str,
        type_b: str,
        relation: str,
        description: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Add or update a relationship between object types."""
        # Check if exists
        for rel in self.relationships:
            if rel.type_a == type_a and rel.type_b == type_b and rel.relation == relation:
                rel.description = description or rel.description
                rel.confidence = max(rel.confidence, confidence)
                self._save()
                return

        self.relationships.append(RelationshipEntry(
            type_a=type_a, type_b=type_b,
            relation=relation, description=description,
            confidence=confidence,
        ))
        self._save()

    def add_open_question(self, question: str) -> None:
        if question not in self.open_questions:
            self.open_questions.append(question)
            self._save()

    def resolve_question(self, question: str) -> None:
        self.open_questions = [q for q in self.open_questions if q != question]
        self._save()

    # =========================================================================
    # Synthesis backend writes
    # =========================================================================

    def update_from_synthesis(
        self,
        accuracy: float,
        confirmed_facts: list[str] | None = None,
        tentative_findings: list[str] | None = None,
        questions: list[str] | None = None,
        code_summary: str = "",
    ) -> None:
        """
        Called by the synthesis backend at the end of a reflexion run.

        If accuracy == 1.0: facts are definitive.
        Otherwise: findings are tentative, questions guide exploration.
        """
        is_definitive = accuracy >= 1.0

        feedback = SynthesisFeedback(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            accuracy=accuracy,
            is_definitive=is_definitive,
            confirmed_facts=confirmed_facts or [],
            tentative_findings=tentative_findings or [],
            questions_for_exploration=questions or [],
            code_summary=code_summary,
        )
        self.synthesis_feedback.append(feedback)

        # If definitive, promote facts to the object/relationship entries
        if is_definitive and confirmed_facts:
            for fact in confirmed_facts:
                if fact not in self.breakthroughs:
                    self.breakthroughs.append(f"[CONFIRMED BY CODE] {fact}")

        # If stuck, add questions for exploration
        if not is_definitive and questions:
            for q in questions:
                if q not in self.open_questions:
                    self.open_questions.append(f"[FROM SYNTHESIS] {q}")

        self._save()

    # =========================================================================
    # Readers (for both exploration LLM and synthesis)
    # =========================================================================

    def to_markdown(self) -> str:
        """
        Render the world model as a structured markdown document.
        This is what both LLMs read.
        """
        sections = []

        # Header with caveat
        sections.append("# World Model")
        sections.append(
            "> **NOTE**: This world model is a hypothesis built from limited observations. "
            "It may contain errors — incorrect roles, wrong action effects, or missing "
            "relationships. When writing transition rules, defer to the test data (ground "
            "truth transitions) over the descriptions here. Use this as a starting point "
            "for understanding, then refine through testing."
        )
        sections.append("")

        # Game metadata
        if self.win_levels > 0:
            sections.append("## Game Info")
            sections.append(f"- Total levels: {self.win_levels}")
            sections.append(f"- Goal reached on this level: {self.goal_reached_count} time(s)")
            if self.goal_reaching_actions:
                sections.append("- Goal-reaching events:")
                for gr in self.goal_reaching_actions[-3:]:  # last 3
                    sections.append(
                        f"  - Level {gr['level_index']}: "
                        f"ACTION{gr.get('action_id', '?')} after "
                        f"{gr.get('transition_count', '?')} transitions"
                    )
                    if "objects_at_goal" in gr:
                        for obj in gr["objects_at_goal"][:5]:
                            sections.append(
                                f"    {obj['type_name']} at {obj['position']}"
                            )
            sections.append("")

        # Goal
        sections.append("## Goal Hypothesis")
        if self.goal:
            sections.append(f"**Hypothesis**: {self.goal.description}")
            sections.append(f"**Confidence**: {self.goal.confidence:.0%}")
            if self.goal.evidence:
                sections.append("**Evidence**:")
                for e in self.goal.evidence:
                    sections.append(f"  - {e}")
            if self.goal.subgoals:
                sections.append("**Subgoals**:")
                for sg in self.goal.subgoals:
                    sections.append(f"  - {sg}")
        else:
            sections.append("*No goal hypothesis yet — exploration should prioritize discovering the objective.*")
        sections.append("")

        # Environment
        sections.append("## Environment")
        if self.environment_description:
            sections.append(self.environment_description)
        if self.background_color:
            sections.append(f"- Background: {self.background_color}")
        if self.boundary_description:
            sections.append(f"- Boundaries: {self.boundary_description}")
        sections.append("")

        # Objects
        sections.append("## Objects")
        if self.objects:
            for tid in sorted(self.objects.keys()):
                obj = self.objects[tid]
                role_str = f" [{obj.role}]" if obj.role else ""
                static_str = " [static]" if obj.is_static else ""
                player_str = " [player]" if obj.is_player else ""
                sections.append(
                    f"### {obj.name} (type {tid}){role_str}{static_str}{player_str}"
                )
                sections.append(f"- Colors: {', '.join(obj.colors)}")
                sections.append(f"- Size: {obj.size}")
                if obj.behavior_summary:
                    sections.append(f"- Behavior: {obj.behavior_summary}")
                if obj.relationship_to_goal:
                    sections.append(f"- Relationship to goal: {obj.relationship_to_goal}")
                if obj.observations:
                    sections.append("- Observations:")
                    for obs in obj.observations[-5:]:
                        sections.append(f"  - {obs}")
                sections.append("")
        else:
            sections.append("*No objects detected yet.*\n")

        # Actions
        sections.append("## Actions")
        for aid in sorted(self.actions.keys()):
            act = self.actions[aid]
            sections.append(f"### ACTION{aid}")
            if act.definition:
                sections.append(f"- Definition: {act.definition}")
            sections.append(f"- Confidence: {act.confidence:.0%}")
            sections.append(f"- Observations: {act.observation_count}")
            if act.effects_by_type:
                sections.append("- Effects by object type:")
                for tname, effect in act.effects_by_type.items():
                    sections.append(f"  - {tname}: {effect}")
            if act.preconditions:
                sections.append("- Known preconditions:")
                for p in act.preconditions:
                    sections.append(f"  - {p}")
            sections.append("")

        # Relationships
        if self.relationships:
            sections.append("## Object Relationships")
            for rel in self.relationships:
                conf_str = f" ({rel.confidence:.0%} confidence)"
                confirmed = " ✓" if rel.confirmed_by_code else ""
                sections.append(
                    f"- **{rel.type_a}** {rel.relation} **{rel.type_b}**{conf_str}{confirmed}"
                )
                if rel.description:
                    sections.append(f"  {rel.description}")
            sections.append("")

        # Breakthroughs
        if self.breakthroughs:
            sections.append("## Breakthroughs")
            for b in self.breakthroughs:
                sections.append(f"- {b}")
            sections.append("")

        # Synthesis Feedback
        if self.synthesis_feedback:
            latest = self.synthesis_feedback[-1]
            sections.append("## Latest Synthesis Feedback")
            sections.append(f"- Accuracy: {latest.accuracy:.0%}")
            sections.append(f"- Status: {'DEFINITIVE' if latest.is_definitive else 'TENTATIVE'}")
            if latest.confirmed_facts:
                sections.append("- Confirmed facts:")
                for f in latest.confirmed_facts:
                    sections.append(f"  - {f}")
            if latest.tentative_findings:
                sections.append("- Tentative findings:")
                for f in latest.tentative_findings:
                    sections.append(f"  - {f}")
            if latest.code_summary:
                sections.append(f"- Code summary: {latest.code_summary}")
            sections.append("")

        # Open Questions
        sections.append("## Open Questions")
        if self.open_questions:
            for q in self.open_questions:
                sections.append(f"- {q}")
        else:
            sections.append("*No open questions — model is well understood.*")
        sections.append("")

        return "\n".join(sections)

    def get_synthesis_context(self) -> str:
        """
        Get the world model as context for the synthesis backend.
        Same as to_markdown() but can be customized if needed.
        """
        return self.to_markdown()

    def get_exploration_context(self) -> str:
        """
        Get the world model as context for the exploration LLM.
        Same as to_markdown() but can be customized if needed.
        """
        return self.to_markdown()

    @property
    def goal_is_understood(self) -> bool:
        """Whether we have a goal hypothesis with decent confidence."""
        return self.goal is not None and self.goal.confidence >= 0.5

    @property
    def is_converged(self) -> bool:
        """
        Whether the world model is converged enough to stop exploration.

        Requires:
        1. Goal is understood
        2. All objects have roles
        3. Latest synthesis achieved 100%
        4. No unresolved open questions from synthesis
        """
        if not self.goal_is_understood:
            return False

        # All objects need roles
        for obj in self.objects.values():
            if not obj.role and not obj.is_static:
                return False

        # Latest synthesis must be perfect
        if self.synthesis_feedback:
            if not self.synthesis_feedback[-1].is_definitive:
                return False
        else:
            return False

        # No synthesis-generated questions outstanding
        synth_questions = [
            q for q in self.open_questions if q.startswith("[FROM SYNTHESIS]")
        ]
        if synth_questions:
            return False

        return True

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        """Save to both JSON (machine) and markdown (human/LLM)."""
        # JSON
        data = {
            "goal": _dataclass_to_dict(self.goal) if self.goal else None,
            "objects": {str(k): _dataclass_to_dict(v) for k, v in self.objects.items()},
            "actions": {str(k): _dataclass_to_dict(v) for k, v in self.actions.items()},
            "relationships": [_dataclass_to_dict(r) for r in self.relationships],
            "open_questions": self.open_questions,
            "breakthroughs": self.breakthroughs,
            "environment_description": self.environment_description,
            "background_color": self.background_color,
            "boundary_description": self.boundary_description,
            "synthesis_feedback": [_dataclass_to_dict(f) for f in self.synthesis_feedback],
            "win_levels": self.win_levels,
            "goal_reached_count": self.goal_reached_count,
            "goal_reaching_actions": self.goal_reaching_actions,
        }
        self.json_path.write_text(json.dumps(data, indent=2, default=str))

        # Markdown
        self.md_path.write_text(self.to_markdown())

    def _load(self) -> None:
        """Load from JSON."""
        try:
            data = json.loads(self.json_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return

        if data.get("goal"):
            self.goal = GoalHypothesis(**data["goal"])

        for tid_str, obj_data in data.get("objects", {}).items():
            self.objects[int(tid_str)] = ObjectEntry(**obj_data)

        for aid_str, act_data in data.get("actions", {}).items():
            self.actions[int(aid_str)] = ActionEntry(**act_data)

        self.relationships = [
            RelationshipEntry(**r) for r in data.get("relationships", [])
        ]

        self.open_questions = data.get("open_questions", [])
        self.breakthroughs = data.get("breakthroughs", [])
        self.environment_description = data.get("environment_description", "")
        self.background_color = data.get("background_color", "")
        self.boundary_description = data.get("boundary_description", "")

        self.synthesis_feedback = [
            SynthesisFeedback(**f) for f in data.get("synthesis_feedback", [])
        ]

        self.win_levels = data.get("win_levels", 0)
        self.goal_reached_count = data.get("goal_reached_count", 0)
        self.goal_reaching_actions = data.get("goal_reaching_actions", [])


def _dataclass_to_dict(obj) -> dict:
    """Convert a dataclass to a dict (handles nested dataclasses)."""
    if obj is None:
        return {}
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {"value": str(obj)}
