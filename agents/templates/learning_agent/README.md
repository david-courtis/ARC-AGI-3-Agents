# Learning Agent

A sophisticated game mechanics discovery agent that learns how unknown games work through systematic experimentation and LLM-powered analysis. The agent has **no prior game-specific knowledge** - it builds understanding entirely from observed state transitions.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Design](#pipeline-design)
- [Data Structures](#data-structures)
- [Core Components](#core-components)
- [LLM Integration](#llm-integration)
- [Knowledge Management](#knowledge-management)
- [Vision & Frame Processing](#vision--frame-processing)
- [Object Detection](#object-detection)
- [Logging System](#logging-system)
- [Configuration](#configuration)
- [Usage](#usage)

---

## Overview

The Learning Agent discovers game mechanics through a systematic exploration loop:

1. **Takes actions** and observes state transitions (before/after frames)
2. **Analyzes transitions** using vision-language models to understand what each action does
3. **Builds iterative understanding** of actions, objects, and environment rules
4. **Validates hypotheses** through repeated observations until verified
5. **Tracks stages/levels** and adapts to new mechanics as the game progresses

The agent terminates when all actions are either:
- **Verified**: 3 consecutive consistent observations (including expected no-ops like hitting the same wall twice)
- **Exhausted**: 8 consecutive no-effect observations (the action appears to never work)
- **Maxed out**: 8 total observations without achieving verification

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LearningAgent (agent.py)                          │
│                         Main orchestration loop                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  LLM Agents   │         │   Knowledge     │         │     Vision      │
│ (llm_agents.py)│         │   Manager       │         │   (vision.py)   │
│               │         │ (knowledge.py)  │         │                 │
│ - Action      │         │                 │         │ - GridFrameRend │
│   Analysis    │         │ - State Store   │         │ - FrameCapture  │
│ - Next Action │         │ - Formatter     │         │ - Color Palette │
│ - Environment │         │ - Updates       │         │                 │
└───────┬───────┘         └────────┬────────┘         └────────┬────────┘
        │                          │                           │
        │                          │                           │
        ▼                          ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Models     │         │    Run Logger   │         │  Frame Differ   │
│  (models.py)  │         │ (run_logger.py) │         │   (diff.py)     │
│               │         │                 │         │                 │
│ - AgentState  │         │ - LLM Call Logs │         │ - PixelDiffer   │
│ - ActionKnow  │         │ - Frame Storage │         │ - SmartDiffer   │
│ - EnvKnow     │         │ - Final Report  │         │ - ASCII Grids   │
└───────────────┘         └─────────────────┘         └────────┬────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────┐
                                                      │Object Detection │
                                                      │(object_detect.py)│
                                                      │                 │
                                                      │ - Gestalt Group │
                                                      │ - Object Diff   │
                                                      │ - Movement Track│
                                                      └─────────────────┘
```

---

## Pipeline Design

The agent operates in a **multi-phase loop** for each action cycle. Note that Phases 1 and 2 are coupled - environment analysis is triggered at the end of action analysis.

### Phase 1: Analyze Previous Action (if pending)

After taking an action, the agent must analyze what happened before deciding the next move.

```
Input:
  - before_frame: Grid state before action
  - after_frame: Grid state after action (may include animation frames)
  - action_id: Which action was taken (ACTION1-5)
  - diff: Computed pixel-level and object-level differences

Processing:
  1. Render frames to base64 PNG images
  2. Compute sequential diffs for animation frames
  3. Build LLM prompt with:
     - Current action definition (if any)
     - Previous observations history
     - Environment understanding
     - ASCII grid representations
     - Object-level change analysis
  4. Call LLM with structured JSON output

Output: ActionAnalysisResult
  - interpretation: What happened
  - had_effect: Boolean - did anything change?
  - new_definition: Updated understanding of the action
  - is_consistent_with_previous: Validation status
  - environment_updates: New discoveries about the world

→ Triggers Phase 2 (Environment Analysis) at the end
```

### Phase 2: Analyze Environment (called from Phase 1)

A **dedicated separate LLM call** that runs after every action analysis to continuously refine environment understanding. This is NOT part of the action analysis - it's an independent call with its own system prompt focused purely on understanding the game world.

**When it runs:** After every action analysis, but only once at least one state change has been observed (to avoid uneducated guesses without evidence).

```
Input:
  - current_frame: Current game state
  - action_context: What action just happened and its effect
  - diff: The changes that occurred
  - action_analysis: Full insights from Phase 1

Processing:
  1. Build comprehensive environment prompt
  2. Include previous breakthroughs and constraints
  3. Request structured analysis of:
     - Background and boundaries
     - Object identification with role hypotheses
     - Movement constraints
     - UI elements (move counters, etc.)

Output: EnvironmentAnalysisResult
  - background_color: What color is empty space?
  - boundaries: Border info, internal walls
  - objects_identified: List of IdentifiedObject
  - movement_constraints: What blocks movement?
  - breakthroughs: Key discoveries
  - domain_description: High-level game understanding
```

### Phase 3: Continue Setup Sequence (if pending)

If the previous decision included a setup sequence, continue executing it.

```
Setup sequences use VERIFIED actions to:
  1. Move to a different position
  2. Change the game state
  3. Reach a better context for testing a target action

Setup actions are NOT analyzed (we trust verified actions).
```

### Phase 4: Get Next Action Suggestion

Ask the LLM what action to test next.

```
Input:
  - current_frame: Current game state image
  - state: Full AgentState with all knowledge

Processing:
  1. Build status of all actions:
     - VERIFIED: Has 3 consistent observations
     - EXHAUSTED: 8+ consecutive no-effects
     - PENDING: Needs more testing
  2. Include recent action history
  3. Include environment understanding
  4. Apply hard guards for no-effect loops

Output: NextActionSuggestion
  - target_action: Which action to test (ACTION1-5)
  - setup_sequence: Actions to execute first (can be empty)
  - reasoning: Why this choice
  - expected_information_gain: What we hope to learn
```

### Phase 5: Execute Target Action

Execute the chosen action and prepare for analysis.

```
1. Capture before frame
2. Execute action via game API
3. Store pending analysis info
4. Increment action count
5. Return to Phase 1 on next iteration
```

---

## Data Structures

All data models use **Pydantic** for validation, serialization, and structured LLM output parsing.

### ActionID (Enum)

```python
class ActionID(str, Enum):
    ACTION1 = "ACTION1"  # Semantically: Move UP (decrease row/y)
    ACTION2 = "ACTION2"  # Semantically: Move DOWN (increase row/y)
    ACTION3 = "ACTION3"  # Semantically: Move LEFT (decrease column/x)
    ACTION4 = "ACTION4"  # Semantically: Move RIGHT (increase column/x)
    ACTION5 = "ACTION5"  # Semantically: INTERACT/SELECT/ROTATE
```

Semantic hints are provided to the LLM but must be verified through observation.

### ActionObservation

A single observation of what an action did:

```python
class ActionObservation(BaseModel):
    timestamp: str                    # ISO timestamp
    before_frame_path: str            # Path to before image
    after_frame_path: str             # Path to after image
    diff_summary: str                 # Human-readable diff
    llm_interpretation: str           # LLM's analysis
    context_description: str          # Game state context
    had_effect: bool                  # Did anything change?
    was_consistent: Optional[bool]    # Matches previous understanding?
    context_that_caused_outcome: str  # Why this outcome in this context
    object_changes: str               # Object-level diff description
```

### ActionKnowledge

Everything known about a single action:

```python
class ActionKnowledge(BaseModel):
    action_id: ActionID
    current_definition: Optional[str]           # Current understanding
    observations: list[ActionObservation]       # All observations
    verification_attempts: int                  # Total attempts
    is_verified: bool                           # 3+ consistent observations
    consecutive_no_effects: int                 # For exhaustion detection
    is_exhausted: bool                          # 8+ consecutive no-effects
```

**Verification Logic:**
- `verification_attempts` increments for **every observation** (not just those with effect)
- An action becomes **verified** when it has 3 consecutive observations marked as consistent by the LLM
  - This **includes expected no-ops** (e.g., hitting the same wall twice is "consistent" behavior)
  - Only explicit inconsistency (`was_consistent=False`) breaks the streak
- An action becomes **exhausted** after 8 **consecutive** no-effect observations
  - A single observation with effect resets the `consecutive_no_effects` counter
- Termination: `verification_attempts >= 8` OR `is_verified` OR `is_exhausted`

### EnvironmentKnowledge

Everything known about the game environment:

```python
class EnvironmentKnowledge(BaseModel):
    # Structured analysis (from dedicated LLM calls)
    background_color: str                       # e.g., "Color 0 (black)"
    has_border: bool
    border_color: Optional[str]
    border_description: str
    internal_walls: list[str]                   # Obstacle descriptions
    movement_constraints: list[str]             # e.g., "Grey walls block all movement"
    identified_objects: list[dict]              # IdentifiedObject as dicts
    spatial_structure: str                      # Grid layout description
    breakthroughs: list[str]                    # Key discoveries
    open_questions: list[str]                   # Unresolved questions
    domain_description: str                     # High-level game type/goal
    unexplored_elements: list[str]              # Areas needing investigation
    analysis_count: int                         # How many analyses done

    # Legacy fields (from action analysis side effects)
    objects: list[ObjectHypothesis]
    spatial_rules: list[str]
    general_observations: list[str]
    iteration_history: list[str]
```

### StageInfo

Tracks game stages/levels:

```python
class StageInfo(BaseModel):
    stage_number: int              # 0-indexed
    started_at_action: int         # Which action count this stage started
    entry_score: int               # Score when entering this stage
    observations_count: int        # Observations in this stage
    key_discoveries: list[str]     # Notable discoveries
```

Stage transitions are detected when the game score increases, signaling a new level that may have different mechanics.

### AgentState

Complete serializable state of the agent:

```python
class AgentState(BaseModel):
    run_id: str                                 # Unique run identifier
    action_knowledge: dict[str, ActionKnowledge]
    environment: EnvironmentKnowledge
    current_frame_path: Optional[str]
    previous_frame_path: Optional[str]
    action_count: int
    llm_call_count: int
    phase: str                                  # "exploration" | "complete"

    # Stage tracking
    current_score: int                          # 0-254
    previous_score: int
    current_stage: int
    stage_history: list[StageInfo]
    actions_since_stage_change: int
    just_transitioned_stage: bool

    # Available actions (from game API)
    available_actions: list[str]
```

### DiffResult

Result of comparing two frames:

```python
class DiffResult(BaseModel):
    changed_pixels: list[PixelChange]    # Individual pixel changes
    change_summary: str                  # Human-readable summary
    has_changes: bool
    change_regions: list[str]            # e.g., "upper_left_playarea: 15 pixels"

    # ASCII representations
    before_ascii: str                    # Grid as ASCII characters
    after_ascii: str
    diff_ascii: str                      # '.' for unchanged, value for changed

    # Object-level analysis
    before_objects: str                  # Object descriptions before
    after_objects: str                   # Object descriptions after
    object_changes: str                  # Movement, appearance, disappearance
```

### LLM Response Models

#### ActionAnalysisResult

```python
class ActionAnalysisResult(BaseModel):
    interpretation: str                 # What happened
    update_definition: bool             # Should we change the definition?
    new_definition: Optional[str]       # New definition if updating
    is_consistent_with_previous: Optional[bool]  # None for first observation
    context_that_caused_this_outcome: str
    objects_involved: list[str]
    context_description: str
    had_effect: bool
    no_effect_reason: Optional[str]
    environment_updates: list[str]
    confidence: float                   # 0.0-1.0
```

#### NextActionSuggestion

```python
class NextActionSuggestion(BaseModel):
    target_action: ActionID
    setup_sequence: list[ActionID]      # Empty if no setup needed
    reasoning: str
    expected_information_gain: str
    current_board_assessment: str
```

#### EnvironmentAnalysisResult

```python
class EnvironmentAnalysisResult(BaseModel):
    background_color: str
    boundaries: BoundaryInfo
    objects_identified: list[IdentifiedObject]
    spatial_structure: str
    movement_constraints: list[str]
    breakthroughs: list[str]
    open_questions: list[str]
    suggested_action_updates: list[SuggestedActionUpdate]
    domain_description: str
    unexplored_elements: list[str]
    confidence: float
```

---

## Core Components

### LearningAgent (agent.py)

The main agent class that orchestrates the exploration loop.

**Key Responsibilities:**
- Initializes all components (LLM agent, knowledge manager, differ, renderer)
- Manages the 5-phase action loop
- Handles game state transitions (GAME_OVER, NOT_PLAYED, WIN)
- Tracks stage/level transitions via score changes
- Implements hard guards against infinite loops
- Cleans up and generates final report

**Configuration Options:**
```python
LearningAgent(
    llm_provider="openrouter",           # LLM provider
    llm_model="google/gemini-2.5-flash", # Model to use
    reasoning=False,                     # Enable extended thinking
    results_dir="results",               # Where to save logs
    verbose=True,                        # Console output
)
```

**Hard Guards:**
- Tracks consecutive no-effect actions
- Blocks actions that had no effect until state changes
- Forces setup sequences to escape stuck states
- Clears blocked actions after 3 consecutive failures

### Frame Differ (diff.py)

Computes differences between game frames at multiple levels.

**PixelDiffer:**
- Pixel-by-pixel comparison
- Region classification (top_ui, bottom_ui, playarea quadrants)
- Bounding box calculation
- ASCII grid generation

**SmartDiffer:**
- Extends PixelDiffer with pattern detection
- Detects horizontal/vertical movement
- Identifies localized changes vs. widespread changes
- Classifies simple swaps (e.g., player sprite movement)

**ASCII Grid Format:**
```
0|0|0|5|5|5|0|0
0|0|0|5|2|5|0|0    <- Before
0|0|0|5|5|5|0|0

.|.|.|.|.|.|.|.
.|.|.|.|3|.|.|.    <- Diff (3 = new color at that position)
.|.|.|.|.|.|.|.
```

**Sequential Diffs for Animation:**
```python
compute_sequential_diffs(frames: list) -> list[dict]
# Returns diffs between consecutive animation frames
# Useful for tracking full motion paths
```

---

## LLM Integration

### OpenAIClientAgent (llm_agents.py)

Uses the OpenAI-compatible API (via OpenRouter) with structured JSON output.

**Three LLM Call Types:**

#### 1. Action Analysis (`analyze_action`)

```
System Prompt: Game analyst learning from state transitions
                No prior knowledge - evidence-based only
                Watch for UI elements and move counters

User Content:
  - Semantic hint for the action
  - Current definition (if any)
  - Previous observations history
  - Environment understanding
  - ASCII grids (before/after/diff)
  - Object-level analysis
  - Animation frame sequence (if multiple)
  - Before and after images

Output: ActionAnalysisResult (JSON schema enforced)
```

#### 2. Next Action Suggestion (`suggest_next_action`)

```
System Prompt: Exploration strategist
                Prioritize unobserved actions
                Use setup sequences for context changes
                Never repeat no-effect actions in same state

User Content:
  - Stage context
  - Action semantics
  - All action knowledge status
  - Recent action history
  - Environment understanding
  - Verified/pending action lists
  - Current frame image

Output: NextActionSuggestion (JSON schema enforced)
```

#### 3. Environment Analysis (`analyze_environment`)

```
System Prompt: Environment analyst
                Focus on boundaries, objects, spatial structure
                Evidence-based analysis
                Can correct previous wrong understanding

User Content:
  - What action just happened
  - Evidence from action effect (or no-effect)
  - Current environment understanding (tentative)
  - Previous breakthroughs
  - Open questions
  - Current action definitions
  - Action analysis insights
  - Current frame image

Output: EnvironmentAnalysisResult (JSON schema enforced)
```

### JSON Schema Enforcement

All LLM calls use structured output with strict JSON schemas:

```python
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
)
```

This ensures:
- All required fields are present
- Types are correct
- Enums are valid values
- No extra fields

### Conversation History (In-Context Learning)

The agent builds conversation history from previous observations to enable in-context learning:

```python
def build_conversation_history(action_knowledge, max_tokens=200000):
    # Collects all observations across actions
    # Sorts by timestamp (oldest first)
    # Formats as user/assistant message pairs
    # Truncates to fit token limit
```

This helps the LLM learn from the entire session's observations.

---

## Knowledge Management

### KnowledgeManager (knowledge.py)

Central manager for all knowledge operations.

**State Persistence (JSONStateStore):**
```
results/
└── run_YYYYMMDD_HHMMSS/
    └── state.json              # Full AgentState serialized
```

**Knowledge Updates:**
```python
def update_from_analysis(state, action_id, analysis, diff, before_path, after_path):
    # 1. Create ActionObservation record
    # 2. Update action definition (if LLM chose to update)
    # 3. Add observation to action knowledge
    # 4. Update environment from environment_updates
    # 5. Track mentioned objects
    # 6. Return updated state
```

**Prompt Formatting:**

The `PromptKnowledgeFormatter` formats knowledge for LLM prompts:

```python
def format_action_knowledge(knowledge) -> str:
    # Action status, definition, observation count

def format_environment(environment) -> str:
    # Background, borders, objects, constraints, breakthroughs

def format_observation_history(observations, limit=5) -> str:
    # Recent observations with effect status and interpretations
```

**No-Op Avoidance:**
```python
def _format_no_op_avoidance_warning(action_knowledge) -> str:
    # Only shown after 3+ attempts
    # Warns about actions with consecutive no-effects
    # Includes last known reason for no-op
```

---

## Vision & Frame Processing

### GridFrameRenderer (vision.py)

Converts integer grid values to colored PNG images.

**Color Palette:**
```python
DEFAULT_COLOR_PALETTE = {
    0: (0, 0, 0),       # Black
    1: (0, 116, 217),   # Blue
    2: (255, 0, 0),     # Red
    3: (46, 204, 64),   # Green
    4: (0, 255, 0),     # Lime
    5: (255, 220, 0),   # Yellow
    6: (0, 0, 255),     # Blue
    7: (255, 255, 0),   # Yellow
    8: (255, 165, 0),   # Orange
    9: (128, 0, 128),   # Purple
    10: (255, 255, 255),# White
    11: (128, 128, 128),# Gray
    # ... up to 15
}
```

**Rendering Pipeline:**
```
64x64 integer grid → RGB image → Scale 8x → PNG file or base64
```

### FrameCapture

Manages frame capture and storage:

```python
class FrameCapture:
    def capture(frame, label) -> str:  # Save to file, return path
    def capture_pair(before, after, prefix) -> tuple[str, str]
    def to_base64(frame) -> str        # For LLM API
    def frames_to_base64_list(frames) -> list[str]  # Animation frames
```

**Animation Frame Handling:**
- Frame data can be 2D (single frame) or 3D (animation sequence)
- Always takes the LAST frame for state comparison
- Sends ALL frames to LLM for motion tracking

---

## Object Detection

### ObjectDetector (object_detection.py)

Groups pixels into objects using Gestalt principles.

**Detection Algorithm:**
1. Identify unique colors (excluding background)
2. For each color, find connected components (8-connectivity)
3. Filter by minimum object size
4. Calculate properties: bounding box, center, area, shape

**DetectedObject Properties:**
```python
class DetectedObject:
    object_id: int
    color: int
    pixels: list[tuple[int, int]]
    bounding_box: tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    center: tuple[float, float]
    area: int
    width: int
    height: int
    is_rectangular: bool  # Fills 90%+ of bounding box
    aspect_ratio: float
```

**Object Comparison (ObjectDiff):**
```python
class ObjectDiff:
    appeared: list[DetectedObject]      # New objects in after frame
    disappeared: list[DetectedObject]   # Objects gone from after frame
    moved: list[tuple[before, after]]   # Same color, different position
    color_changed: list[tuple[before, after]]  # Same position, different color
    unchanged: list[DetectedObject]
```

**Matching Logic:**
- **Exact match**: Same color + 80% bounding box overlap
- **Moved**: Same color + similar size + 2-20 pixel distance
- **Color changed**: Different color + 60% bounding box overlap

---

## Logging System

### RunLogger (run_logger.py)

Creates comprehensive logs for debugging and analysis.

**Directory Structure:**
```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── state.json                    # Current AgentState
    ├── final_report.md               # Human-readable summary
    ├── frames/
    │   ├── before_0000.png
    │   ├── after_0001.png
    │   ├── current_0002.png
    │   └── ...
    └── calls/
        ├── 001_ACTION1_analysis/
        │   ├── before.png
        │   ├── after.png
        │   ├── prompt.txt            # Actual prompt sent to LLM
        │   ├── diff.json
        │   ├── response.json
        │   ├── state_before.json
        │   ├── state_after.json
        │   ├── metadata.json
        │   └── full_conversation.json
        ├── 002_next_action_suggestion/
        │   ├── current_frame.png
        │   ├── prompt.txt
        │   ├── response.json
        │   └── state.json
        ├── 003_environment_analysis/
        │   ├── context.txt
        │   ├── response.json
        │   ├── breakthroughs.txt
        │   ├── constraints.txt
        │   └── action_updates.txt
        └── ...
```

**LLMCallLog:**
```python
class LLMCallLog(BaseModel):
    call_id: int
    timestamp: str
    call_type: str           # "action_analysis" | "next_action_suggestion" | "environment_analysis"
    action_id: str | None
    prompt: str
    images_sent: list[str]
    context_provided: dict
    full_messages: list[dict]  # Complete conversation history
    response: dict
    duration_ms: float | None
```

### ConsoleLogger

Real-time console output for monitoring:

```
[INFO] 14:30:15 | Starting new run: run_20241203_143015
------------------------------------------------------------
[LLM] next_action_suggestion
[RESULT] Target: ACTION1 | Setup: [] | Testing first action...
[ACTION] ACTION1 | Target action - Testing movement...
[LLM] action_analysis (ACTION1)
[RESULT] HAD EFFECT | first obs | Moved player up by 1 tile...
[LLM] environment_analysis
[INFO] 14:30:18 | BREAKTHROUGH: Grey color (5) represents walls
```

### Final Report

Markdown summary generated at the end:

```markdown
# Learning Agent Exploration Report

**Run ID:** run_20241203_143015
**Total Actions:** 45
**Total LLM Calls:** 90

## Action Knowledge Summary

### ACTION1 (VERIFIED)
**Definition:** Moves the player UP (decreases row), blocked by grey walls
**Observations:** 12
**Effective Attempts:** 8/8

### ACTION5 (EXHAUSTED)
**Definition:** Appears to have no effect in tested contexts
**Consecutive No-Effects:** 8 (exhausted)

## Environment Understanding

**Background Color:** Color 0 (black)
**Border:** Yes (Color 5 - grey)

### Key Breakthroughs
- Grey color (5) represents impassable walls
- Red square (2) is the player - responds to ACTION1-4
- Yellow squares (7) are collectible items

### Movement Constraints
- Player cannot move through grey walls
- Player cannot move outside the border
```

---

## Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### Agent Parameters

```python
LearningAgent(
    # LLM Configuration
    llm_provider="openrouter",           # Provider name
    llm_model="google/gemini-2.5-flash", # Model identifier
    reasoning=False,                     # Extended thinking mode

    # Output Configuration
    results_dir="results",               # Base directory for logs
    verbose=True,                        # Console output
)
```

### Hardcoded Constants

```python
# agent.py
MAX_ACTIONS = 200           # Maximum actions before forced termination

# models.py (inline in ActionKnowledge.add_observation)
# - Verified when: _count_recent_consistent() >= 3
# - Exhausted when: consecutive_no_effects >= 8
# - Needs more when: verification_attempts < 8 AND not verified AND not exhausted

# object_detection.py
min_object_size = 4         # Minimum pixels for object detection (constructor param)
```

---

## Usage

### Command Line

```bash
python main.py -a learningagent
```

### Programmatic

```python
from agents.templates.learning_agent import LearningAgent

agent = LearningAgent(
    game_id="your_game",
    llm_model="google/gemini-2.5-flash",
    results_dir="my_results",
)

# The agent is then used by the game runner
```

### Available Exports

```python
from agents.templates.learning_agent import (
    # Main agent
    LearningAgent,

    # Models
    ActionID,
    ActionObservation,
    ActionKnowledge,
    EnvironmentKnowledge,
    AgentState,
    DiffResult,
    ActionAnalysisResult,
    NextActionSuggestion,
    EnvironmentAnalysisResult,

    # Components
    KnowledgeManager,
    FrameDiffer,
    ObjectDetector,
    GridFrameRenderer,
    FrameCapture,

    # LLM
    LLMAgent,
    OpenRouterAgent,
    create_agent,

    # Logging
    RunLogger,
    ConsoleLogger,
)
```

---

## Design Rationale

### Why Three-Phase LLM Architecture?

The agent makes three distinct LLM calls per action cycle (when applicable):

1. **Action Analysis** - "What did this action do?"
2. **Environment Analysis** - "What do I understand about the world now?"
3. **Next Action Suggestion** - "What should I try next?"

**Benefits:**
1. **Separation of Concerns**: Each call has a focused task with specialized prompts
2. **Better Context**: Each call gets only the context it needs, not a monolithic prompt
3. **Iterative Refinement**: Environment understanding builds separately from action definitions
4. **Debugging**: Easy to trace issues to specific call types in the logs
5. **Independent Evolution**: Environment analysis can correct previous wrong understanding without being tied to action analysis output

### Why Object-Level Analysis?

1. **Human-Like Understanding**: Games are understood as objects, not pixels
2. **Motion Tracking**: Easier to detect "player moved left" than pixel deltas
3. **Role Discovery**: Objects have roles (player, obstacle, goal, collectible)
4. **LLM Alignment**: Vision-language models understand objects naturally

### Why Verification Through Consistency?

1. **Context Sensitivity**: Actions may behave differently in different states
2. **Noise Tolerance**: Single observations may be misleading
3. **Confidence Building**: 3 consistent observations provides high confidence
4. **Expected No-Ops Count**: If the LLM says "this no-op is consistent with the definition" (e.g., hitting a known wall), it counts toward verification
5. **Early Termination**: Avoids over-testing well-understood actions

### Why Track No-Effects Separately?

1. **Different from Inconsistency**: A no-op can be "consistent" (expected wall hit) or indicate an untested context
2. **Exhaustion Detection**: 8 **consecutive** no-effects suggests the action never works - stop testing
3. **Loop Prevention**: Hard guards block recently no-effect actions until state changes
4. **Information Value**: No-effects teach about boundaries and constraints
5. **Counter Reset**: A single effective action resets `consecutive_no_effects`, giving the action another chance

### Why Stage Tracking?

1. **Mechanic Changes**: New levels may introduce new rules
2. **Reset Understanding**: Previous assumptions may become invalid
3. **Adaptation**: Agent can re-explore after stage transitions
4. **Progress Tracking**: Helps understand game progression
