"""
LLM Agents for the Learning Agent.

This module provides LLM integration for:
1. Action Analysis - Understanding what an action did
2. Next Action Suggestion - Deciding what to do next

Supports both PydanticAI (experimental) and raw OpenAI client (stable).
"""

import base64
import json
import os
from abc import ABC, abstractmethod

from openai import OpenAI

from .knowledge import KnowledgeManager
from .models import (
    ACTION_SEMANTICS,
    ActionAnalysisResult,
    ActionID,
    ActionKnowledge,
    AgentState,
    BoundaryInfo,
    DiffResult,
    EnvironmentAnalysisResult,
    EnvironmentKnowledge,
    IdentifiedObject,
    NextActionSuggestion,
    SuggestedActionUpdate,
)


# ============================================================================
# Helper Functions
# ============================================================================


def format_action_semantics(available_actions: list[str] | None = None) -> str:
    """
    Format action semantics for inclusion in prompts.

    Args:
        available_actions: List of available action names (e.g., ["ACTION1", "ACTION2"]).
                          If None or empty, shows all actions.
    """
    lines = ["ACTION SEMANTIC HINTS (these are hints, verify through observation):"]

    # Filter to available actions if specified
    for action_id, semantic in ACTION_SEMANTICS.items():
        if available_actions and action_id.value not in available_actions:
            continue  # Skip unavailable actions
        lines.append(f"  {action_id.value}: {semantic}")

    # Add note about available actions
    if available_actions:
        lines.append(f"\n⚠️ NOTE: Only {len(available_actions)} actions are available in this game: {', '.join(sorted(available_actions))}")

    return "\n".join(lines)


# ============================================================================
# Prompt Templates
# ============================================================================

ACTION_ANALYSIS_SYSTEM_PROMPT = """You are a game analyst learning how an unknown game works.

You observe state transitions (before/after frames) and analyze what actions do.
You have NO prior knowledge of this game - base all conclusions on observed evidence only.

Your job is to:
1. Describe what happened when the action was taken
2. Build/update a definition of what this action does
3. Note any objects or environment features involved
4. Track whether the action had any visible effect

IMPORTANT - CURRENT UNDERSTANDING MAY BE WRONG:
- The environment understanding provided is TENTATIVE and may be incorrect
- Object identifications and role hypotheses could be wrong
- If what you observe CONTRADICTS the current understanding, note this!
- Your observations help REFINE or CORRECT the environment model

IMPORTANT - UI ELEMENTS AND MOVE COUNTERS:
- There are likely UI elements (dots, squares, lines) that count successful moves
- When an action HAS EFFECT (moves something), the move counter typically decreases
- Changes in UI elements (e.g., red dots disappearing) indicate move consumption, NOT action effects
- Focus on what happened in the PLAY AREA, not just UI changes
- Grid boundaries may result in looping interactions.
- Running out of moves (counter reaches zero) causes game failure and reset

Be precise and EVIDENCE-BASED. If you're uncertain, say so.
Mark any guesses without evidence as uncertain."""


NEXT_ACTION_SYSTEM_PROMPT = """You are a game exploration strategist.

Your goal is to systematically learn what each action does in an unknown game.
You do this by choosing which action to test next based on:

1. Actions that have never been observed (highest priority)
2. Actions that need more validation (not yet verified)
3. Whether the current game state is good for testing the chosen action

An action is "verified" when you have 3 consistent observations of what it does.

CRITICAL: If an action had NO EFFECT in the current state, DO NOT test it again
in the same state! Instead:
1. Use a SETUP SEQUENCE of verified movement actions to reach a DIFFERENT position
2. Then test the action from the new position
3. Actions may only work in specific contexts (e.g., interact only works when near an object)

If the current state is not good for testing an action (e.g., against a wall when
testing movement, or no nearby objects for interaction), you MUST suggest a
SETUP SEQUENCE of verified actions to reach a better state first.

Be strategic and efficient in your exploration. Vary positions frequently to
discover context-dependent behaviors."""


ENVIRONMENT_ANALYSIS_SYSTEM_PROMPT = """You are an environment analyst studying an unknown game world.

Your SOLE FOCUS is understanding the ENVIRONMENT - not the actions.

CRITICAL: YOUR CURRENT UNDERSTANDING MAY BE WRONG!
- Previous object identifications and role hypotheses are TENTATIVE and may need correction
- If action results contradict your current understanding, UPDATE IT
- You can and SHOULD replace incorrect objects/roles with better interpretations
- Every analysis is an opportunity to REFINE or CORRECT the environment model

You must discover and document:
1. BOUNDARIES/WALLS - Where can entities NOT move? What stops movement?
2. BACKGROUND - What is the base color/pattern? What areas are "empty"?
3. OBJECTS - What distinct objects exist? What are their visual properties?
4. OBJECT ROLES - Are some objects "players"? "Goals"? "Obstacles"? "Collectibles"?
5. SPATIAL STRUCTURE - Is there a grid? Rooms? Paths? Corridors?
6. INTERACTABLES - What objects can be interacted with? How do you know?
7. UI ELEMENTS - Look for move counters, score displays, progress indicators!

CRITICAL: Look for WALLS and BOUNDARIES!
- If an object tried to move but couldn't, WHY?
- Are there internal walls or obstacles blocking change?
- What color/pattern indicates a boundary vs empty space?

IMPORTANT: UI ELEMENTS AND MOVE COUNTERS!
- There are likely UI elements (dots, squares, lines) that track the number of SUCCESSFUL moves
- These counters typically decrease or change with each move that has an EFFECT
- When the counter runs out (e.g., all dots disappear), the game will FAIL and force a RESET
- Look for patterns at the top, bottom, or sides of the screen that seem separate from the play area
- Red dots, grey lines, or colored squares often serve as move/turn indicators

EVIDENCE-BASED ANALYSIS:
- Only make claims you have EVIDENCE for from observed state changes
- If you're guessing without evidence, explicitly mark it as "UNCONFIRMED HYPOTHESIS"
- When an action causes a state change, that's EVIDENCE - use it to confirm or refute hypotheses
- No-effect actions also provide evidence about boundaries and constraints

Be a detective. Make BREAKTHROUGHS in understanding. Don't just describe - ANALYZE.
When you notice something new, explain its SIGNIFICANCE for gameplay.
When previous understanding was WRONG, explicitly note the correction."""


# ============================================================================
# Abstract Agent Interface
# ============================================================================


class LLMAgent(ABC):
    """Abstract base class for LLM-powered agents."""

    @abstractmethod
    def analyze_action(
        self,
        before_image_b64: str,
        after_image_b64: str,
        action_id: ActionID,
        diff: DiffResult,
        action_knowledge: ActionKnowledge,
        environment: EnvironmentKnowledge,
        all_action_knowledge: dict[str, ActionKnowledge] | None = None,
        animation_frames_b64: list[str] | None = None,
        sequential_diffs: list[dict] | None = None,
        stage_context: str = "",
    ) -> ActionAnalysisResult:
        """Analyze what an action did."""
        ...

    @abstractmethod
    def suggest_next_action(
        self,
        current_frame_b64: str,
        state: AgentState,
    ) -> NextActionSuggestion:
        """Suggest what action to take next."""
        ...

    @abstractmethod
    def analyze_environment(
        self,
        current_frame_b64: str,
        environment: EnvironmentKnowledge,
        action_context: str,
        diff: DiffResult | None = None,
        action_knowledge: dict[str, ActionKnowledge] | None = None,
        had_state_change: bool = False,
        stage_context: str = "",
        action_analysis: ActionAnalysisResult | None = None,
    ) -> EnvironmentAnalysisResult:
        """Analyze the environment to understand its structure."""
        ...


# ============================================================================
# JSON Schemas for structured output
# ============================================================================

ACTION_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {
            "type": "string",
            "description": "What happened when the action was taken"
        },
        "update_definition": {
            "type": "boolean",
            "description": "Set to true ONLY if the current definition needs to be changed. False to keep existing definition."
        },
        "new_definition": {
            "type": ["string", "null"],
            "description": "New definition ONLY if update_definition is true. Null otherwise."
        },
        "is_consistent_with_previous": {
            "type": ["boolean", "null"],
            "description": "Whether this matches previous observations. Null if first observation."
        },
        "context_that_caused_this_outcome": {
            "type": "string",
            "description": "What specific environment configuration/context caused this particular outcome? Why did the action have this effect (or no effect) in THIS situation?"
        },
        "objects_involved": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objects that were affected"
        },
        "context_description": {
            "type": "string",
            "description": "Description of the game state context"
        },
        "had_effect": {
            "type": "boolean",
            "description": "Whether the action changed anything visible"
        },
        "no_effect_reason": {
            "type": ["string", "null"],
            "description": "If no effect, explain why (wall, edge, etc.)"
        },
        "environment_updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New observations about the environment"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in this analysis (0-1)"
        }
    },
    "required": [
        "interpretation", "update_definition", "new_definition", "is_consistent_with_previous",
        "context_that_caused_this_outcome", "objects_involved", "context_description", "had_effect",
        "environment_updates", "confidence"
    ],
    "additionalProperties": False
}

NEXT_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "target_action": {
            "type": "string",
            "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
            "description": "The action to test next"
        },
        "setup_sequence": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
            },
            "description": "Verified actions to execute first (can be empty)"
        },
        "reasoning": {
            "type": "string",
            "description": "Why this action/sequence was chosen"
        },
        "expected_information_gain": {
            "type": "string",
            "description": "What we hope to learn from this action"
        },
        "current_board_assessment": {
            "type": "string",
            "description": "Assessment of the current game state"
        }
    },
    "required": [
        "target_action", "setup_sequence", "reasoning",
        "expected_information_gain", "current_board_assessment"
    ],
    "additionalProperties": False
}

ENVIRONMENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "background_color": {
            "type": "string",
            "description": "The background/empty space color (e.g., 'Color 0 (black)', 'Color 3 (green)')"
        },
        "boundaries": {
            "type": "object",
            "properties": {
                "has_border": {
                    "type": "boolean",
                    "description": "Is there a visible border/wall around the play area?"
                },
                "border_color": {
                    "type": ["string", "null"],
                    "description": "Color of the border/walls if present"
                },
                "border_description": {
                    "type": "string",
                    "description": "Description of boundary structure (thickness, pattern, gaps)"
                },
                "internal_walls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of internal walls/obstacles observed"
                }
            },
            "required": ["has_border", "border_color", "border_description", "internal_walls"],
            "additionalProperties": False
        },
        "objects_identified": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Descriptive name for this object type"},
                    "color": {"type": "string", "description": "Color of the object"},
                    "shape": {"type": "string", "description": "Shape description"},
                    "role_hypothesis": {"type": "string", "description": "What role might this object play? (player, obstacle, goal, collectible, etc.)"},
                    "evidence_for_role": {"type": "string", "description": "What evidence supports this role hypothesis?"}
                },
                "required": ["name", "color", "shape", "role_hypothesis", "evidence_for_role"],
                "additionalProperties": False
            },
            "description": "List of distinct object types identified"
        },
        "spatial_structure": {
            "type": "string",
            "description": "Description of spatial layout (grid, rooms, corridors, open space, etc.)"
        },
        "movement_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Observed constraints on movement (walls block X direction, can't move through color Y, etc.)"
        },
        "breakthroughs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "NEW significant discoveries about the environment this observation"
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unresolved questions about the environment to investigate"
        },
        "suggested_action_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
                        "description": "The action to update"
                    },
                    "suggested_definition": {
                        "type": "string",
                        "description": "Simple, conceptual definition (e.g., 'moves up, blocked by grey walls')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this update makes sense given environment discoveries"
                    }
                },
                "required": ["action_id", "suggested_definition", "reasoning"],
                "additionalProperties": False
            },
            "description": "Suggested updates to action definitions based on environment breakthroughs"
        },
        "domain_description": {
            "type": "string",
            "description": "High-level conceptual description of the game domain: What type of game is this? What is the goal? How do the mechanics work together? Think like explaining to someone who has never seen this game."
        },
        "unexplored_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Colors, sections, or visual features that have NOT been thoroughly analyzed yet. What parts of the grid need more investigation?"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Overall confidence in environment understanding (0-1)"
        }
    },
    "required": [
        "background_color", "boundaries", "objects_identified", "spatial_structure",
        "movement_constraints", "breakthroughs", "open_questions", "suggested_action_updates",
        "domain_description", "unexplored_elements", "confidence"
    ],
    "additionalProperties": False
}


# ============================================================================
# OpenAI Client Implementation (Stable)
# ============================================================================


class OpenAIClientAgent(LLMAgent):
    """
    LLM Agent using raw OpenAI client with OpenRouter.

    Uses JSON schema structured outputs for reliable parsing.
    This is the stable implementation that works around pydantic-ai bugs.
    """

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        reasoning: bool = False,
    ):
        self.model_name = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.reasoning = reasoning
        self.knowledge_manager = KnowledgeManager()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        # Store last messages sent for logging (without base64 images to save space)
        self.last_messages: list[dict] = []

        if self.reasoning:
            print(f"🧠 Reasoning enabled for model: {self.model_name}")

    def _get_reasoning_params(self) -> dict:
        """Get extra parameters for reasoning/extended thinking if enabled."""
        if not self.reasoning:
            return {}
        # OpenRouter reasoning parameter for models that support extended thinking
        # See: https://openrouter.ai/docs/use-cases/reasoning
        return {
            "extra_body": {
                "reasoning": {
                    "effort": "high"  # Options: low, medium, high
                }
            }
        }

    def _sanitize_messages_for_logging(self, messages: list[dict]) -> list[dict]:
        """Remove base64 image data from messages for logging (keeps structure but saves space)."""
        sanitized = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                # Multi-content message (with images)
                sanitized_content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        # Replace image data with placeholder
                        sanitized_content.append({
                            "type": "image_url",
                            "image_url": {"url": "[BASE64_IMAGE_DATA_OMITTED]"}
                        })
                    else:
                        sanitized_content.append(item)
                sanitized.append({"role": msg["role"], "content": sanitized_content})
            else:
                sanitized.append(msg)
        return sanitized

    def analyze_action(
        self,
        before_image_b64: str,
        after_image_b64: str,
        action_id: ActionID,
        diff: DiffResult,
        action_knowledge: ActionKnowledge,
        environment: EnvironmentKnowledge,
        all_action_knowledge: dict[str, ActionKnowledge] | None = None,
        animation_frames_b64: list[str] | None = None,
        sequential_diffs: list[dict] | None = None,
        stage_context: str = "",
    ) -> ActionAnalysisResult:
        """Analyze what an action did."""
        # Format knowledge for prompt
        context = self.knowledge_manager.format_for_action_analysis(
            action_id, action_knowledge, environment, diff
        )

        is_first = len(action_knowledge.observations) == 0

        if is_first:
            history_section = "This is the FIRST time observing this action."
        else:
            history_section = f"""YOUR CURRENT DEFINITION OF {action_id.value}:
"{action_knowledge.current_definition}"

PREVIOUS OBSERVATIONS FOR THIS ACTION ({len(action_knowledge.observations)} total):
{context['observation_history']}"""

        # Include object-level analysis (Gestalt grouping)
        object_section = ""
        if context.get('object_changes'):
            object_section = f"""
OBJECT-LEVEL ANALYSIS (Before → Final state, grouped by visual proximity/similarity):
{context['object_changes']}

OBJECTS BEFORE ACTION:
{context.get('before_objects', 'Unknown')}
"""

        # Get semantic hint for this action
        action_semantic = ACTION_SEMANTICS.get(action_id, "Unknown action type")

        # Build animation frames description with per-frame object analysis
        num_animation_frames = len(animation_frames_b64) if animation_frames_b64 else 1
        if num_animation_frames > 1 and sequential_diffs:
            animation_section = f"""
ANIMATION SEQUENCE: This action produced {num_animation_frames} animation frames showing the motion.
The images below show the FULL ANIMATION SEQUENCE - watch how objects move frame by frame!

SEQUENTIAL FRAME ANALYSIS (changes between consecutive animation frames):
"""
            for seq_diff in sequential_diffs:
                frame_analysis = f"""
--- Frame {seq_diff['from_frame']} → Frame {seq_diff['to_frame']} ({seq_diff['pixel_count']} pixels changed) ---"""
                # Include object changes if available
                if seq_diff.get('object_changes'):
                    frame_analysis += f"""
Object Changes: {seq_diff['object_changes']}"""
                frame_analysis += f"""
Diff Grid:
{seq_diff['diff_ascii']}
"""
                animation_section += frame_analysis

            images_description = f"""I'm showing you a SEQUENCE of {num_animation_frames + 1} LABELED images:
1. IMAGE 1: Frame BEFORE the action
2-{num_animation_frames + 1}. IMAGEs 2-{num_animation_frames + 1}: Animation frames showing the motion step by step

⚠️ IMPORTANT: Analyze ALL animation frames to understand the FULL motion path!
Objects may move multiple tiles - track their path through each frame."""
        else:
            animation_section = ""
            images_description = """I'm showing you 2 LABELED images:
1. IMAGE 1: The frame BEFORE the action
2. IMAGE 2: The frame AFTER the action (final state)"""

        # Stage context section
        stage_section = ""
        if stage_context:
            stage_section = f"\n{stage_context}\n"

        # ASCII grid section - include the original grids
        ascii_section = ""
        if diff.before_ascii and diff.after_ascii:
            ascii_section = f"""
ASCII GRID (BEFORE ACTION):
{diff.before_ascii}

ASCII GRID (AFTER ACTION - FINAL STATE):
{diff.after_ascii}

DIFF GRID (. = unchanged, value = new color at that position):
{diff.diff_ascii}
"""

        prompt = f"""Analyze the effect of action {action_id.value}.
{stage_section}
SEMANTIC HINT: {action_id.value} is expected to mean "{action_semantic}" (but verify through observation!)

{history_section}

ENVIRONMENT UNDERSTANDING:
{context['environment']}

FRAME CHANGES (before → final state, pixel level):
{context['diff_summary']}
{ascii_section}{object_section}{animation_section}
{images_description}

IMPORTANT INSTRUCTIONS:
1. Think about changes at the OBJECT level, not just pixels!
2. If there are MULTIPLE animation frames, track the FULL MOTION path of objects
3. REASON about WHY this action produced THIS outcome in THIS context
4. If the action had NO EFFECT, explain what context caused that (wall? edge? wrong object?)
5. If the action HAD EFFECT, explain what context enabled that effect
6. Only set update_definition=true if you need to CHANGE the definition. If the current definition is accurate, set update_definition=false.
7. If an object changes position or features, analyze what ENVIRONMENTAL FEATURES limit/define/bound the change. What stopped the object? What caused the change to occur in that specific way? All needed information is visible in the environment.

Analyze what happened and provide your structured analysis."""

        # Build messages starting with system prompt
        messages = [
            {"role": "system", "content": ACTION_ANALYSIS_SYSTEM_PROMPT},
        ]

        # Add conversation history for in-context learning (if available)
        if all_action_knowledge:
            history_messages = self.knowledge_manager.build_conversation_history(
                all_action_knowledge, max_tokens=150000  # Leave room for current prompt
            )
            messages.extend(history_messages)

        # Build image content list with labeled images
        image_content = [{"type": "text", "text": prompt}]

        # Add before image with label
        image_content.append({
            "type": "text",
            "text": "📷 IMAGE 1: BEFORE ACTION (state prior to executing the action)"
        })
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{before_image_b64}"},
        })

        # Add animation frames (or just after image if no animation)
        if animation_frames_b64 and len(animation_frames_b64) > 1:
            # Multiple animation frames - add them all with labels
            for i, frame_b64 in enumerate(animation_frames_b64):
                frame_num = i + 1
                if frame_num == len(animation_frames_b64):
                    label = f"📷 IMAGE {frame_num + 1}: ANIMATION FRAME {frame_num} (FINAL STATE after action)"
                else:
                    label = f"📷 IMAGE {frame_num + 1}: ANIMATION FRAME {frame_num} (intermediate motion)"
                image_content.append({
                    "type": "text",
                    "text": label
                })
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
                })
        else:
            # Single frame - just add the after image with label
            image_content.append({
                "type": "text",
                "text": "📷 IMAGE 2: AFTER ACTION (final state after executing the action)"
            })
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{after_image_b64}"},
            })

        # Add current analysis request with images
        messages.append({
            "role": "user",
            "content": image_content,
        })

        # Store sanitized messages for logging
        self.last_messages = self._sanitize_messages_for_logging(messages)

        # Call the API with structured output
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "action_analysis",
                    "strict": True,
                    "schema": ACTION_ANALYSIS_SCHEMA,
                },
            },
            **self._get_reasoning_params(),
        )

        # Parse response
        content = response.choices[0].message.content
        data = json.loads(content)

        return ActionAnalysisResult(**data)

    def suggest_next_action(
        self,
        current_frame_b64: str,
        state: AgentState,
    ) -> NextActionSuggestion:
        """Suggest what action to take next."""
        # Format knowledge for prompt
        context = self.knowledge_manager.format_for_next_action(
            state.action_knowledge, state.environment
        )

        # Include semantic hints for available actions only
        semantics_section = format_action_semantics(
            state.available_actions if state.available_actions else None
        )

        # Build no-op avoidance section (only shown after 3+ attempts)
        no_op_section = ""
        if context.get('no_op_avoidance_warning'):
            no_op_section = f"\n{context['no_op_avoidance_warning']}\n"

        # Get stage context
        stage_context = state.get_stage_context()

        prompt = f"""Decide what action to test next.

{stage_context}

{semantics_section}

CURRENT ACTION KNOWLEDGE STATUS:
{context['action_status']}

RECENT ACTION HISTORY (last 10 actions taken):
{context['recent_action_history']}

ENVIRONMENT UNDERSTANDING:
{context['environment']}

VERIFIED ACTIONS (can use for setup): {context['verified_actions']}
PENDING ACTIONS (need testing): {context['pending_actions']}

RECENT NO-EFFECT ATTEMPTS: {context['recent_no_effects']}
{no_op_section}
Looking at the current game state (image shown), decide:
1. Which action should we test next?
2. Do we need a setup sequence first to reach a better position?
3. Why this choice?

CRITICAL RULES:
- If an action had NO EFFECT recently, you MUST use a setup sequence to MOVE first
- Do NOT repeat the same action in the same state - it will have no effect again
- Use verified movement actions (ACTION1-4) in setup_sequence to change position
- Only after moving to a NEW position should you retry a no-effect action

Provide your structured suggestion."""

        # Build messages
        messages = [
            {"role": "system", "content": NEXT_ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{current_frame_b64}"},
                    },
                ],
            },
        ]

        # Store sanitized messages for logging
        self.last_messages = self._sanitize_messages_for_logging(messages)

        # Call the API with structured output
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "next_action",
                    "strict": True,
                    "schema": NEXT_ACTION_SCHEMA,
                },
            },
            **self._get_reasoning_params(),
        )

        # Parse response
        content = response.choices[0].message.content
        data = json.loads(content)

        # Convert string action IDs to enum
        data["target_action"] = ActionID(data["target_action"])
        data["setup_sequence"] = [ActionID(a) for a in data["setup_sequence"]]

        return NextActionSuggestion(**data)

    def analyze_environment(
        self,
        current_frame_b64: str,
        environment: EnvironmentKnowledge,
        action_context: str,
        diff: DiffResult | None = None,
        action_knowledge: dict[str, "ActionKnowledge"] | None = None,
        had_state_change: bool = False,
        stage_context: str = "",
        action_analysis: ActionAnalysisResult | None = None,
    ) -> EnvironmentAnalysisResult:
        """Analyze the environment to understand its structure."""
        # Format current environment understanding
        env_summary = self.knowledge_manager.format_environment_for_analysis(environment)

        # Build the prompt
        diff_context = ""
        if diff and diff.has_changes:
            diff_context = f"""
RECENT CHANGES OBSERVED (THIS IS EVIDENCE!):
{diff.change_summary}

Object-level changes: {diff.object_changes if diff.object_changes else 'None'}
"""
        else:
            diff_context = """
NO CHANGES OBSERVED - This action had NO EFFECT.
This is EVIDENCE that something blocked or prevented the action!
"""

        # Format current action definitions for the LLM to suggest updates
        action_defs = ""
        if action_knowledge:
            action_lines = []
            for action_id, knowledge in action_knowledge.items():
                if knowledge.current_definition:
                    action_lines.append(f"  {action_id}: \"{knowledge.current_definition}\"")
                else:
                    action_lines.append(f"  {action_id}: (no definition yet)")
            action_defs = "\n".join(action_lines)
        else:
            action_defs = "  (no action knowledge available)"

        # Evidence status
        evidence_status = "STATE CHANGE OBSERVED - Use this as evidence!" if had_state_change else "NO STATE CHANGE - Action was blocked or had no effect"

        # Stage context section
        stage_section = ""
        if stage_context:
            stage_section = f"\n{stage_context}\n"

        # Action analysis insights section (from the action analysis LLM call)
        action_insights_section = ""
        if action_analysis:
            insights_parts = []

            # Full interpretation (not truncated)
            if action_analysis.interpretation:
                insights_parts.append(f"Full Interpretation: {action_analysis.interpretation}")

            # Context that caused the outcome
            if action_analysis.context_that_caused_this_outcome:
                insights_parts.append(f"Why This Outcome: {action_analysis.context_that_caused_this_outcome}")

            # New/updated definition
            if action_analysis.new_definition:
                insights_parts.append(f"Updated Action Definition: {action_analysis.new_definition}")

            # Environment updates discovered by action analysis
            if action_analysis.environment_updates:
                updates_str = "\n  - ".join(action_analysis.environment_updates)
                insights_parts.append(f"Environment Updates Discovered:\n  - {updates_str}")

            # Objects involved
            if action_analysis.objects_involved:
                insights_parts.append(f"Objects Involved: {', '.join(action_analysis.objects_involved)}")

            # No effect reason
            if action_analysis.no_effect_reason:
                insights_parts.append(f"Why No Effect: {action_analysis.no_effect_reason}")

            if insights_parts:
                action_insights_section = f"""

=== ACTION ANALYSIS INSIGHTS (from previous LLM analysis of this action) ===
{chr(10).join(insights_parts)}

⚠️ IMPORTANT: The above insights were discovered by analyzing the action's effect.
Use these insights to UPDATE your environment understanding!
"""

        prompt = f"""Analyze the environment in this game frame.
{stage_section}

WHAT JUST HAPPENED:
{action_context}
Evidence Status: {evidence_status}
{diff_context}{action_insights_section}

YOUR CURRENT ENVIRONMENT UNDERSTANDING (MAY BE WRONG - UPDATE IF NEEDED):
{env_summary}

⚠️ IMPORTANT: The above understanding is TENTATIVE and based on limited observations.
If the action results CONTRADICT this understanding, you MUST update it!
You can REPLACE objects_identified entirely if the current list is wrong.

PREVIOUS BREAKTHROUGHS:
{chr(10).join(f'- {b}' for b in environment.breakthroughs) if environment.breakthroughs else 'None yet'}

OPEN QUESTIONS TO INVESTIGATE:
{chr(10).join(f'- {q}' for q in environment.open_questions) if environment.open_questions else 'None yet'}

CURRENT ACTION DEFINITIONS (ALSO TENTATIVE):
{action_defs}

CRITICAL ANALYSIS TASKS:
1. LOOK FOR WALLS/BOUNDARIES - If movement failed, identify what blocked it!
2. What color is the BACKGROUND (empty space)?
3. What distinct OBJECTS exist? What might their ROLES be?
4. Are there patterns in the spatial layout?
5. What movement CONSTRAINTS can you infer?
6. LOOK FOR MOVE COUNTERS/UI ELEMENTS - Are there dots, squares, or lines that track moves?
7. UNEXPLORED ELEMENTS: Which colors or sections of the grid have NOT been thoroughly analyzed yet? List specific colors or regions that need more investigation.
8. DOMAIN DESCRIPTION: Provide a high-level conceptual description of the game. What type of game is this (puzzle, navigation, sokoban-like, etc.)? What appears to be the goal? How do the mechanics work together at a conceptual level?

EVIDENCE-BASED REASONING:
- State changes are EVIDENCE. Use them to CONFIRM or REFUTE hypotheses.
- No-effect actions are EVIDENCE of boundaries, walls, or constraints.
- If you have NO EVIDENCE for a claim, mark it as "UNCONFIRMED HYPOTHESIS"
- Don't make uneducated guesses - only claim what you can support with evidence
- If previous understanding was based on guessing and now contradicted by evidence, CORRECT IT

UPDATING THE ENVIRONMENT MODEL:
- Your objects_identified list REPLACES the previous one - include all objects you're confident about
- If you now think an object has a different role, update the role_hypothesis
- Add evidence_for_role explaining WHY you think an object has that role
- If you realize a previous identification was WRONG, simply don't include it (or include corrected version)

Make BREAKTHROUGHS! If you discover something new and significant, list it.
If previous understanding was WRONG, note the CORRECTION as a breakthrough.

ACTION DEFINITION UPDATES (BE VERY CONSERVATIVE):
Only suggest updates to action definitions if you have a DRASTIC new insight that fundamentally
changes the understanding of what the action does. Default to NOT suggesting updates.

ONLY suggest an update if:
1. The current definition is completely WRONG or MISSING critical information
2. You discovered something that CONTRADICTS the current definition
3. You have HIGH CONFIDENCE based on EVIDENCE (not guessing)

Default: Leave suggested_action_updates as an EMPTY array unless truly necessary."""

        # Build messages with environment conversation history
        messages = [
            {"role": "system", "content": ENVIRONMENT_ANALYSIS_SYSTEM_PROMPT},
        ]

        # Add environment analysis history for in-context learning
        history_messages = self.knowledge_manager.build_environment_history(
            environment, max_tokens=100000
        )
        messages.extend(history_messages)

        # Add current analysis request with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{current_frame_b64}"},
                },
            ],
        })

        # Store sanitized messages for logging
        self.last_messages = self._sanitize_messages_for_logging(messages)

        # Call the API with structured output
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "environment_analysis",
                    "strict": True,
                    "schema": ENVIRONMENT_ANALYSIS_SCHEMA,
                },
            },
            **self._get_reasoning_params(),
        )

        # Parse response
        content = response.choices[0].message.content
        data = json.loads(content)

        # Convert nested dicts to models
        boundaries = BoundaryInfo(**data["boundaries"])
        objects = [IdentifiedObject(**obj) for obj in data["objects_identified"]]
        suggested_updates = [
            SuggestedActionUpdate(**update)
            for update in data.get("suggested_action_updates", [])
        ]

        return EnvironmentAnalysisResult(
            background_color=data["background_color"],
            boundaries=boundaries,
            objects_identified=objects,
            spatial_structure=data["spatial_structure"],
            movement_constraints=data["movement_constraints"],
            breakthroughs=data["breakthroughs"],
            open_questions=data["open_questions"],
            suggested_action_updates=suggested_updates,
            domain_description=data.get("domain_description", ""),
            unexplored_elements=data.get("unexplored_elements", []),
            confidence=data["confidence"],
        )


class OpenRouterAgent(OpenAIClientAgent):
    """LLM Agent using OpenRouter."""

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        reasoning: bool = False,
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            reasoning=reasoning,
        )


# Aliases for backwards compatibility
PydanticAIAgent = OpenAIClientAgent
OpenAICompatibleAgent = OpenAIClientAgent


# ============================================================================
# Synchronous Wrapper
# ============================================================================


class SyncLLMAgent:
    """
    Wrapper that provides sync interface for LLM agents.

    The underlying PydanticAI agent uses run_sync, so this is just a passthrough.
    """

    def __init__(self, agent: LLMAgent):
        self.agent = agent

    @property
    def last_messages(self) -> list[dict]:
        """Get the last messages sent to the LLM (for logging)."""
        if hasattr(self.agent, "last_messages"):
            return self.agent.last_messages
        return []

    def analyze_action(
        self,
        before_image_b64: str,
        after_image_b64: str,
        action_id: ActionID,
        diff: DiffResult,
        action_knowledge: ActionKnowledge,
        environment: EnvironmentKnowledge,
        all_action_knowledge: dict[str, ActionKnowledge] | None = None,
        animation_frames_b64: list[str] | None = None,
        sequential_diffs: list[dict] | None = None,
        stage_context: str = "",
    ) -> ActionAnalysisResult:
        """Analyze what an action did."""
        return self.agent.analyze_action(
            before_image_b64,
            after_image_b64,
            action_id,
            diff,
            action_knowledge,
            environment,
            all_action_knowledge,
            animation_frames_b64,
            sequential_diffs,
            stage_context,
        )

    def suggest_next_action(
        self,
        current_frame_b64: str,
        state: AgentState,
    ) -> NextActionSuggestion:
        """Suggest what action to take next."""
        return self.agent.suggest_next_action(current_frame_b64, state)

    def analyze_environment(
        self,
        current_frame_b64: str,
        environment: EnvironmentKnowledge,
        action_context: str,
        diff: DiffResult | None = None,
        action_knowledge: dict[str, ActionKnowledge] | None = None,
        had_state_change: bool = False,
        stage_context: str = "",
        action_analysis: ActionAnalysisResult | None = None,
    ) -> EnvironmentAnalysisResult:
        """Analyze the environment to understand its structure."""
        return self.agent.analyze_environment(
            current_frame_b64,
            environment,
            action_context,
            diff,
            action_knowledge,
            had_state_change,
            stage_context,
            action_analysis,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_agent(
    provider: str = "openrouter",
    model: str | None = None,
    api_key: str | None = None,
    reasoning: bool = False,
) -> LLMAgent:
    """
    Factory function to create an LLM agent.

    Args:
        provider: "openrouter" (default)
        model: Model name (provider-specific)
        api_key: API key (falls back to environment variables)
        reasoning: Enable extended thinking/reasoning for supported models

    Returns:
        Configured LLM agent
    """
    if provider == "openrouter":
        return OpenRouterAgent(
            model=model or "google/gemini-2.5-flash",
            api_key=api_key,
            reasoning=reasoning,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openrouter'.")
