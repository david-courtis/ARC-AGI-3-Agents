# Learning Agent - Implementation Plan

## Overview

A learning agent that discovers game mechanics from scratch by:
1. Taking actions and observing state transitions
2. Building iterative understanding of what actions do
3. Building understanding of environment/objects
4. Validating hypotheses through repeated observations

**No game-specific knowledge baked in.**

---

## Core Data Structures

### 1. ActionKnowledge

Tracks what we know about each action.

```python
from pydantic import BaseModel
from enum import Enum

class ActionID(str, Enum):
    ACTION1 = "ACTION1"  # Semantically: up
    ACTION2 = "ACTION2"  # Semantically: down
    ACTION3 = "ACTION3"  # Semantically: left
    ACTION4 = "ACTION4"  # Semantically: right
    ACTION5 = "ACTION5"  # Semantically: interact/select/rotate

class ActionObservation(BaseModel):
    """A single observation of what an action did."""
    timestamp: str
    before_frame_path: str
    after_frame_path: str
    diff_summary: str
    llm_interpretation: str
    context: str  # e.g., "player was at top-left corner"

class ActionKnowledge(BaseModel):
    """Everything we know about an action."""
    action_id: ActionID
    current_definition: str | None = None
    observations: list[ActionObservation] = []
    verification_attempts: int = 0
    is_verified: bool = False

    # Verification requires:
    # - At least 3 consistent observations in different contexts
    # - OR 8 attempts reached (forced stop)
```

### 2. EnvironmentKnowledge

Tracks understanding of the game world.

```python
class ObjectHypothesis(BaseModel):
    """A hypothesis about an object in the environment."""
    name: str  # e.g., "red square", "door-like structure"
    description: str
    first_seen: str  # timestamp
    observations: list[str]  # Natural language observations
    confidence: float  # 0.0 - 1.0

class EnvironmentKnowledge(BaseModel):
    """Everything we know about the environment."""
    objects: list[ObjectHypothesis] = []
    spatial_relationships: list[str] = []  # e.g., "player can move on orange areas"
    rules: list[str] = []  # e.g., "green areas block movement"
    iteration_history: list[str] = []  # How understanding evolved
```

### 3. AgentState

The complete state of the learning agent.

```python
class AgentState(BaseModel):
    """Complete agent state - serializable for logging."""
    run_id: str
    action_knowledge: dict[ActionID, ActionKnowledge]
    environment: EnvironmentKnowledge
    current_frame: str | None  # Path to current frame image
    previous_frame: str | None
    action_count: int = 0
    llm_call_count: int = 0
    phase: str = "exploration"  # exploration | validation | planning
```

### 4. LLMCallLog

For comprehensive logging.

```python
class LLMCallLog(BaseModel):
    """Complete log of an LLM interaction."""
    call_id: int
    timestamp: str
    purpose: str  # "first_observation" | "validation" | "environment_update"
    action_being_analyzed: ActionID | None

    # Inputs
    images_sent: list[str]  # Paths to images
    text_prompt: str
    context_provided: str  # Previous definitions, environment knowledge

    # Outputs
    raw_response: str
    parsed_response: dict

    # State snapshot
    agent_state_before: dict
    agent_state_after: dict
```

---

## Pipeline Steps

### Step 0: Initialization

```
1. Create run directory: results/run_{timestamp}/
2. Initialize ActionKnowledge for ACTION1-5 (all empty)
3. Initialize empty EnvironmentKnowledge
4. Create subdirectories: calls/, frames/, state/
```

### Step 1: Select Next Action

**Goal:** Choose which action to take next.

```python
def select_next_action(state: AgentState) -> tuple[ActionID, list[ActionID]]:
    """
    Returns:
        - action_to_test: The action we want to learn about
        - setup_sequence: Actions to execute first (if needed)
    """

    # Priority 1: Actions with no observations
    for action_id, knowledge in state.action_knowledge.items():
        if len(knowledge.observations) == 0:
            return action_id, []

    # Priority 2: Actions needing validation (not verified, < 8 attempts)
    for action_id, knowledge in state.action_knowledge.items():
        if not knowledge.is_verified and knowledge.verification_attempts < 8:
            # May need setup sequence - ask LLM
            setup = plan_setup_sequence(state, action_id)
            return action_id, setup

    # All actions verified or maxed out
    return None, []
```

### Step 2: Execute Setup Sequence (if needed)

```python
def execute_setup_sequence(sequence: list[ActionID], game) -> bool:
    """Execute a sequence of verified actions to reach desired state."""
    for action in sequence:
        game.execute(action)
        # Brief pause, capture frame for logging
    return True
```

### Step 3: Capture Before State

```python
def capture_frame(game, run_dir: str, label: str) -> str:
    """Capture current frame, save as image, return path."""
    frame = game.get_current_frame()
    path = f"{run_dir}/frames/{label}_{timestamp}.png"
    save_as_image(frame, path)
    return path
```

### Step 4: Execute Target Action

```python
def execute_action(action: ActionID, game) -> FrameData:
    """Execute action and return new frame."""
    return game.execute(action)
```

### Step 5: Capture After State

Same as Step 3.

### Step 6: Compute Diff

```python
def compute_diff(before_frame: np.ndarray, after_frame: np.ndarray) -> DiffResult:
    """
    Compare frames pixel-by-pixel.

    Returns:
        DiffResult with:
        - changed_pixels: list of (row, col, old_val, new_val)
        - change_summary: str (e.g., "15 pixels changed in region (20,30)-(25,35)")
        - has_changes: bool
    """
    changes = []
    for i in range(64):
        for j in range(64):
            if before_frame[i][j] != after_frame[i][j]:
                changes.append((i, j, before_frame[i][j], after_frame[i][j]))

    return DiffResult(
        changed_pixels=changes,
        change_summary=summarize_changes(changes),
        has_changes=len(changes) > 0
    )
```

### Step 7: LLM Analysis

Two modes:

#### Mode A: First Observation

```python
async def analyze_first_observation(
    agent: PydanticAIAgent,
    before_img: str,
    after_img: str,
    diff: DiffResult,
    action: ActionID,
    environment: EnvironmentKnowledge
) -> ActionAnalysisResult:
    """First time seeing this action - no prior definition."""

    prompt = f"""
    You are learning how a game works by observing state transitions.

    ACTION TAKEN: {action.value}

    CURRENT ENVIRONMENT UNDERSTANDING:
    {format_environment(environment)}

    FRAME CHANGES:
    {diff.change_summary}

    Changed pixels (row, col): old_value -> new_value
    {format_pixel_changes(diff.changed_pixels[:50])}  # Limit for context

    I'm showing you:
    1. The frame BEFORE the action
    2. The frame AFTER the action

    Please analyze:
    1. What did this action appear to do?
    2. What objects or regions were affected?
    3. Provide a concise definition of what {action.value} does.
    4. Note any new objects or environment features you observe.
    """

    result = await agent.run(prompt, images=[before_img, after_img])
    return result.data
```

#### Mode B: Validation / Update

```python
async def validate_action_definition(
    agent: PydanticAIAgent,
    before_img: str,
    after_img: str,
    diff: DiffResult,
    action: ActionID,
    current_knowledge: ActionKnowledge,
    environment: EnvironmentKnowledge
) -> ValidationResult:
    """Validate or update existing definition."""

    prompt = f"""
    You are validating your understanding of a game action.

    ACTION TAKEN: {action.value}

    YOUR CURRENT DEFINITION:
    "{current_knowledge.current_definition}"

    PREVIOUS OBSERVATIONS ({len(current_knowledge.observations)} total):
    {format_observation_history(current_knowledge.observations[-5:])}  # Last 5

    CURRENT ENVIRONMENT UNDERSTANDING:
    {format_environment(environment)}

    NEW OBSERVATION:
    {diff.change_summary}

    Changed pixels: {format_pixel_changes(diff.changed_pixels[:30])}

    I'm showing you the BEFORE and AFTER frames.

    Please analyze:
    1. Is your current definition of {action.value} consistent with this observation?
    2. If NOT consistent, provide an UPDATED definition.
    3. If the action had no effect, explain why (wall? edge? wrong context?).
    4. Note any updates to environment understanding.

    Respond with:
    - is_consistent: true/false
    - updated_definition: (new definition if changed, else null)
    - no_effect_reason: (if action did nothing)
    - environment_updates: (any new observations about the world)
    """

    result = await agent.run(prompt, images=[before_img, after_img])
    return result.data
```

### Step 8: Update Knowledge

```python
def update_knowledge(
    state: AgentState,
    action: ActionID,
    llm_result: ActionAnalysisResult | ValidationResult,
    diff: DiffResult,
    before_path: str,
    after_path: str
) -> AgentState:
    """Update action knowledge and environment based on LLM response."""

    knowledge = state.action_knowledge[action]

    # Add observation to history
    observation = ActionObservation(
        timestamp=now(),
        before_frame_path=before_path,
        after_frame_path=after_path,
        diff_summary=diff.change_summary,
        llm_interpretation=llm_result.interpretation,
        context=llm_result.context_description
    )
    knowledge.observations.append(observation)

    # Update definition
    if isinstance(llm_result, ValidationResult):
        knowledge.verification_attempts += 1
        if llm_result.is_consistent:
            # Check if we have enough consistent observations
            if count_recent_consistent(knowledge) >= 3:
                knowledge.is_verified = True
        else:
            # Reset consistency counter, update definition
            knowledge.current_definition = llm_result.updated_definition
    else:
        # First observation
        knowledge.current_definition = llm_result.action_definition

    # Update environment
    if llm_result.environment_updates:
        update_environment(state.environment, llm_result.environment_updates)

    return state
```

### Step 9: Log Everything

```python
def log_call(
    run_dir: str,
    call_id: int,
    purpose: str,
    action: ActionID,
    before_path: str,
    after_path: str,
    prompt: str,
    response: dict,
    state_before: AgentState,
    state_after: AgentState
):
    """Create comprehensive log of this LLM call."""

    call_dir = f"{run_dir}/calls/{call_id:03d}_{action.value}_{purpose}"
    os.makedirs(call_dir)

    # Copy images
    shutil.copy(before_path, f"{call_dir}/before.png")
    shutil.copy(after_path, f"{call_dir}/after.png")

    # Save prompt
    with open(f"{call_dir}/prompt.txt", "w") as f:
        f.write(prompt)

    # Save response
    with open(f"{call_dir}/response.json", "w") as f:
        json.dump(response, f, indent=2)

    # Save state snapshots
    with open(f"{call_dir}/state_before.json", "w") as f:
        f.write(state_before.model_dump_json(indent=2))

    with open(f"{call_dir}/state_after.json", "w") as f:
        f.write(state_after.model_dump_json(indent=2))
```

### Step 10: Check Termination

```python
def should_terminate(state: AgentState) -> bool:
    """Check if exploration phase is complete."""

    for action_id, knowledge in state.action_knowledge.items():
        if not knowledge.is_verified and knowledge.verification_attempts < 8:
            return False  # Still have work to do

    return True  # All actions verified or maxed out
```

---

## Main Loop

```python
async def run_exploration(game, run_dir: str):
    """Main exploration loop."""

    # Initialize
    state = initialize_state(run_dir)

    while not should_terminate(state):
        # 1. Select action to test
        target_action, setup_sequence = select_next_action(state)

        if target_action is None:
            break

        # 2. Execute setup sequence if needed
        if setup_sequence:
            execute_setup_sequence(setup_sequence, game)

        # 3. Capture before
        before_path = capture_frame(game, run_dir, f"before_{state.action_count}")
        state_before = state.model_copy(deep=True)

        # 4. Execute action
        execute_action(target_action, game)
        state.action_count += 1

        # 5. Capture after
        after_path = capture_frame(game, run_dir, f"after_{state.action_count}")

        # 6. Compute diff
        diff = compute_diff(load_frame(before_path), load_frame(after_path))

        # 7. LLM analysis
        knowledge = state.action_knowledge[target_action]
        if len(knowledge.observations) == 0:
            result = await analyze_first_observation(
                agent, before_path, after_path, diff, target_action, state.environment
            )
            purpose = "first_observation"
        else:
            result = await validate_action_definition(
                agent, before_path, after_path, diff, target_action, knowledge, state.environment
            )
            purpose = "validation"

        state.llm_call_count += 1

        # 8. Update knowledge
        state = update_knowledge(state, target_action, result, diff, before_path, after_path)

        # 9. Log everything
        log_call(run_dir, state.llm_call_count, purpose, target_action,
                 before_path, after_path, prompt, result, state_before, state)

        # 10. Save current state
        save_state(state, run_dir)

    # Exploration complete
    print(f"Exploration complete after {state.action_count} actions")
    print(f"LLM calls: {state.llm_call_count}")
    generate_final_report(state, run_dir)
```

---

## File Structure

```
results/
└── run_2024_12_02_19_30_00/
    ├── state.json                    # Current AgentState
    ├── action_knowledge.json         # All action definitions + history
    ├── environment.json              # Environment understanding
    ├── final_report.md               # Human-readable summary
    ├── frames/
    │   ├── before_001.png
    │   ├── after_001.png
    │   ├── before_002.png
    │   └── ...
    └── calls/
        ├── 001_ACTION1_first_observation/
        │   ├── before.png
        │   ├── after.png
        │   ├── prompt.txt
        │   ├── response.json
        │   ├── state_before.json
        │   └── state_after.json
        ├── 002_ACTION3_first_observation/
        │   └── ...
        ├── 003_ACTION1_validation/
        │   └── ...
        └── ...
```

---

## PydanticAI Agent Setup

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ActionAnalysisResult(BaseModel):
    """Result of analyzing an action for the first time."""
    action_definition: str
    objects_involved: list[str]
    context_description: str  # Where was the player, what was nearby
    environment_updates: list[str]
    confidence: float

class ValidationResult(BaseModel):
    """Result of validating an existing action definition."""
    is_consistent: bool
    interpretation: str  # What happened this time
    updated_definition: str | None  # Only if not consistent
    no_effect_reason: str | None  # If action did nothing
    context_description: str
    environment_updates: list[str]

# Agent for action analysis
action_agent = Agent(
    'openai:gpt-4o',  # Or use OpenRouter
    result_type=ActionAnalysisResult | ValidationResult,
    system_prompt="""
    You are a game analyst learning how an unknown game works.
    You observe state transitions and build understanding of:
    1. What each action does
    2. What objects exist in the environment
    3. How objects interact

    You have NO prior knowledge of this game.
    Base all conclusions on observed evidence only.
    """
)
```

---

## Design Decisions

1. **Setup sequence planning**: LLM suggests action(s) to reach a state where target action provides good information. This is a **separate LLM call** from action analysis.

2. **Game resets**: Only on game over. Preserve exploration progress.

3. **"No effect" handling**:
   - Does NOT count toward 8 validation attempts
   - Still recorded and used as information for LLM
   - Helps build understanding of constraints/walls/edges

4. **Verification criteria**: 3 consistent observations in different contexts.

5. **Image representation**: Rendered PNG with colors (human-readable for logging too).

---

## Two-Phase LLM Calls

Each turn involves **two separate LLM calls**:

### Call 1: Action Analysis

Analyze what just happened (after executing an action).

```python
class ActionAnalysisResult(BaseModel):
    """Result of analyzing what an action did."""
    interpretation: str  # What happened
    action_definition: str  # Current understanding of what this action does
    is_consistent_with_previous: bool | None  # None if first observation
    objects_involved: list[str]
    context_description: str  # Game state context
    had_effect: bool  # Did the action change anything?
    no_effect_reason: str | None  # If no effect, why?
    environment_updates: list[str]  # New observations about world
    confidence: float

async def analyze_action(
    agent: Agent,
    before_img: str,
    after_img: str,
    diff: DiffResult,
    action: ActionID,
    action_knowledge: ActionKnowledge,
    environment: EnvironmentKnowledge
) -> ActionAnalysisResult:
    """Analyze what an action did. First LLM call."""

    is_first = len(action_knowledge.observations) == 0

    prompt = f"""
    You are learning how an unknown game works by observing state transitions.

    ACTION TAKEN: {action.value}

    {"This is the FIRST time observing this action." if is_first else f'''
    YOUR CURRENT DEFINITION OF {action.value}:
    "{action_knowledge.current_definition}"

    PREVIOUS OBSERVATIONS ({len(action_knowledge.observations)} total):
    {format_observation_history(action_knowledge.observations[-3:])}
    '''}

    ENVIRONMENT UNDERSTANDING:
    {format_environment(environment)}

    FRAME CHANGES:
    {diff.change_summary if diff.has_changes else "NO CHANGES DETECTED"}

    {f"Changed pixels: {format_pixel_changes(diff.changed_pixels[:30])}" if diff.has_changes else ""}

    Analyze the BEFORE and AFTER frames shown.

    Respond with:
    1. What did this action do? (interpretation)
    2. {"Define what this action does" if is_first else "Is this consistent with your previous definition?"}
    3. Did the action have any effect on the game state?
    4. If no effect, why? (wall, edge, invalid context?)
    5. What objects or environment features did you observe?
    """

    return await agent.run(prompt, images=[before_img, after_img])
```

### Call 2: Next Action Suggestion

Decide what to do next.

```python
class NextActionSuggestion(BaseModel):
    """LLM's suggestion for what action to take next."""
    target_action: ActionID  # The action we want to learn about
    setup_sequence: list[ActionID]  # Actions to execute first (can be empty)
    reasoning: str  # Why this action/sequence
    expected_information_gain: str  # What we hope to learn
    current_board_assessment: str  # LLM's view of current game state

async def suggest_next_action(
    agent: Agent,
    current_frame: str,
    action_knowledge: dict[ActionID, ActionKnowledge],
    environment: EnvironmentKnowledge
) -> NextActionSuggestion:
    """Suggest what action to take next. Second LLM call."""

    # Build summary of action knowledge status
    action_status = []
    for action_id, knowledge in action_knowledge.items():
        if knowledge.is_verified:
            status = f"VERIFIED ({knowledge.current_definition})"
        elif len(knowledge.observations) == 0:
            status = "NO OBSERVATIONS YET"
        else:
            consistent = count_recent_consistent(knowledge)
            status = f"{consistent}/3 consistent, {knowledge.verification_attempts}/8 attempts"
            status += f" - Current def: {knowledge.current_definition}"
        action_status.append(f"{action_id.value}: {status}")

    prompt = f"""
    You are learning how an unknown game works.

    CURRENT GAME STATE:
    [Image of current frame shown]

    ACTION KNOWLEDGE STATUS:
    {chr(10).join(action_status)}

    ENVIRONMENT UNDERSTANDING:
    {format_environment(environment)}

    Your goal is to learn what each action does through experimentation.
    An action is "verified" when you have 3 consistent observations.

    DECISION CRITERIA:
    1. Priority: Actions with NO observations yet
    2. Then: Actions needing more validation (not verified, < 8 attempts)
    3. Consider: Is the current board state good for testing the target action?

    If the current board state is NOT good for testing an action (e.g., against a wall
    when you want to test movement), suggest a SETUP SEQUENCE of verified actions
    to reach a better state first.

    Respond with:
    1. Which action should we test next? (target_action)
    2. Do we need to execute other actions first? (setup_sequence, can be empty)
    3. Why this choice? (reasoning)
    4. What do we hope to learn? (expected_information_gain)
    5. Describe the current board state (current_board_assessment)
    """

    return await agent.run(prompt, images=[current_frame])
```

---

## Updated Main Loop

```python
async def run_exploration(game, run_dir: str):
    """Main exploration loop with two-phase LLM calls."""

    state = initialize_state(run_dir)

    while not should_terminate(state):
        # ============================================
        # PHASE 1: Decide what action to take
        # ============================================

        current_frame_path = capture_frame(game, run_dir, f"current_{state.action_count}")

        # LLM Call: Suggest next action
        suggestion = await suggest_next_action(
            next_action_agent,
            current_frame_path,
            state.action_knowledge,
            state.environment
        )

        log_call(run_dir, state.llm_call_count, "next_action_suggestion",
                 None, current_frame_path, None, suggestion)
        state.llm_call_count += 1

        # Check termination
        if suggestion.target_action is None:
            break

        # ============================================
        # PHASE 2: Execute setup sequence (if any)
        # ============================================

        for setup_action in suggestion.setup_sequence:
            game.execute(setup_action)
            # Brief capture for logging but don't analyze
            capture_frame(game, run_dir, f"setup_{state.action_count}")
            state.action_count += 1

        # ============================================
        # PHASE 3: Execute target action and analyze
        # ============================================

        before_path = capture_frame(game, run_dir, f"before_{state.action_count}")
        state_before = state.model_copy(deep=True)

        # Execute the target action
        game.execute(suggestion.target_action)
        state.action_count += 1

        after_path = capture_frame(game, run_dir, f"after_{state.action_count}")

        # Compute diff
        diff = compute_diff(load_frame(before_path), load_frame(after_path))

        # ============================================
        # PHASE 4: LLM analyzes what happened
        # ============================================

        analysis = await analyze_action(
            analysis_agent,
            before_path,
            after_path,
            diff,
            suggestion.target_action,
            state.action_knowledge[suggestion.target_action],
            state.environment
        )

        log_call(run_dir, state.llm_call_count, "action_analysis",
                 suggestion.target_action, before_path, after_path, analysis)
        state.llm_call_count += 1

        # ============================================
        # PHASE 5: Update knowledge
        # ============================================

        state = update_knowledge(
            state,
            suggestion.target_action,
            analysis,
            diff,
            before_path,
            after_path
        )

        # Save state after each cycle
        save_state(state, run_dir)

        # Handle game over
        if game.is_game_over():
            game.reset()

    # Exploration complete
    generate_final_report(state, run_dir)
```

---

## Updated Knowledge Update Logic

```python
def update_knowledge(
    state: AgentState,
    action: ActionID,
    analysis: ActionAnalysisResult,
    diff: DiffResult,
    before_path: str,
    after_path: str
) -> AgentState:
    """Update action knowledge based on LLM analysis."""

    knowledge = state.action_knowledge[action]

    # Always record the observation (even if no effect)
    observation = ActionObservation(
        timestamp=now(),
        before_frame_path=before_path,
        after_frame_path=after_path,
        diff_summary=diff.change_summary,
        llm_interpretation=analysis.interpretation,
        context=analysis.context_description,
        had_effect=analysis.had_effect
    )
    knowledge.observations.append(observation)

    # Update definition
    knowledge.current_definition = analysis.action_definition

    # Only count toward verification if action HAD an effect
    if analysis.had_effect:
        knowledge.verification_attempts += 1

        if analysis.is_consistent_with_previous:
            # Count consistent observations (only those with effect)
            consistent_count = count_consistent_with_effect(knowledge)
            if consistent_count >= 3:
                knowledge.is_verified = True

    # Update environment knowledge
    for update in analysis.environment_updates:
        add_environment_observation(state.environment, update)

    return state


def count_consistent_with_effect(knowledge: ActionKnowledge) -> int:
    """Count recent consecutive consistent observations that had effect."""
    count = 0
    for obs in reversed(knowledge.observations):
        if not obs.had_effect:
            continue  # Skip no-effect observations
        # Check if this observation was marked consistent
        # (This would need to be stored in the observation)
        if obs.was_consistent:
            count += 1
        else:
            break  # Reset on inconsistency
    return count
```

---

## Termination Criteria

```python
def should_terminate(state: AgentState) -> bool:
    """Check if exploration phase is complete."""

    for action_id, knowledge in state.action_knowledge.items():
        # Skip if verified
        if knowledge.is_verified:
            continue

        # Count attempts WITH effect only
        attempts_with_effect = sum(1 for o in knowledge.observations if o.had_effect)

        if attempts_with_effect < 8:
            return False  # Still have attempts left

    return True  # All actions verified or maxed out (8 effective attempts)
```

---

## Next Steps

1. Create the file structure
2. Implement data models (Pydantic)
3. Implement diff computation
4. Implement PydanticAI agents (two separate agents)
5. Implement logging
6. Implement main loop
7. Test with actual game

