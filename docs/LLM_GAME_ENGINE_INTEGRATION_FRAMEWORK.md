## 5. Python Game Engine Design

### 5.1 Two-Layer Architecture

The engine has two layers:

1. **Core Library** (domain-agnostic, human-authored, shared across all games)
2. **Per-Game Subgame Modules** (LLM-generated, evolved each iteration)

### 5.2 Core Library

The core library provides base classes and utilities that every subgame module uses. It is written once by a human and does not change per game.

#### 5.2.1 Grid / Board Representation

```python
class Grid:
    """NumPy-backed 2D grid with cell accessors and region queries.

    Wraps the raw ARC-AGI-3 frame data (64x64 integer grid) with
    convenient spatial operations.
    """

    def __init__(self, data: np.ndarray):
        self.data = data.copy()
        self.height, self.width = data.shape

    def get(self, row: int, col: int) -> int:
        """Get cell value, returns -1 for out-of-bounds."""

    def set(self, row: int, col: int, value: int) -> None:
        """Set cell value."""

    def in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""

    def neighbors(self, row: int, col: int, connectivity: int = 4) -> list[tuple[int, int]]:
        """Get valid neighbor positions (4 or 8 connectivity)."""

    def find_all(self, value: int) -> list[tuple[int, int]]:
        """Find all cells with a given value."""

    def region(self, min_row: int, min_col: int, max_row: int, max_col: int) -> np.ndarray:
        """Extract a rectangular region."""

    def diff(self, other: "Grid") -> list[tuple[int, int, int, int]]:
        """Cell-by-cell comparison. Returns [(row, col, self_val, other_val), ...]."""

    def snapshot(self) -> "Grid":
        """Deep copy for state comparison."""

    @classmethod
    def from_frame(cls, frame_data: list | np.ndarray) -> "Grid":
        """Construct from ARC-AGI-3 frame data (handles 2D and 3D)."""
```

#### 5.2.2 GameObject Base Class

```python
@dataclass
class GameObject:
    """Base class for game objects identified in the environment.

    The LLM subclasses this for each game's specific object types
    (e.g., Player, Wall, Box, Target, Key, Door).
    """

    object_id: str              # Unique identifier
    color: int                  # Grid color value (0-15)
    position: tuple[int, int]   # (row, col) of top-left corner
    width: int
    height: int
    properties: dict[str, Any]  # Game-specific properties

    def cells(self) -> list[tuple[int, int]]:
        """All cells this object occupies."""

    def overlaps(self, other: "GameObject") -> bool:
        """Check if this object overlaps with another."""

    def distance_to(self, other: "GameObject") -> float:
        """Euclidean distance between centers."""

    def adjacent_to(self, other: "GameObject", direction: str | None = None) -> bool:
        """Check adjacency (optionally in a specific direction)."""
```

#### 5.2.3 GameState Container

```python
@dataclass
class GameState:
    """Complete state of the game at a point in time.

    Holds the grid, all instantiated objects, and metadata.
    Used as the input/output type for transition functions.
    """

    grid: Grid                          # Current grid state
    objects: dict[str, GameObject]      # All game objects by ID
    step: int                           # Action count
    score: int                          # Current score
    action_history: list[str]           # Actions taken so far
    metadata: dict[str, Any]            # Game-specific metadata

    @classmethod
    def from_grid(cls, grid: Grid, object_registry: "ObjectRegistry") -> "GameState":
        """Parse a grid into a GameState by detecting objects using the registry."""

    def snapshot(self) -> "GameState":
        """Deep copy for prediction comparison."""

    def apply(self, new_grid: Grid) -> "GameState":
        """Create new state with updated grid, re-detecting objects."""
```

#### 5.2.4 Object Registry

```python
class ObjectRegistry:
    """Maps grid patterns to GameObject subclasses.

    Each subgame module registers its object types here.
    The registry is used by GameState.from_grid() to parse
    a raw grid into typed game objects.
    """

    def register(self, color: int, cls: type[GameObject],
                 matcher: Callable[[Grid, DetectedObject], bool] | None = None) -> None:
        """Register an object type. Matcher is optional custom logic."""

    def parse_grid(self, grid: Grid) -> dict[str, GameObject]:
        """Detect and instantiate all objects from a grid."""
```

#### 5.2.5 Action Interface

```python
class Action(str, Enum):
    """ARC-AGI-3 actions, mirroring the existing ActionID enum."""
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
```

#### 5.2.6 SubgameModule Protocol

```python
class SubgameModule(Protocol):
    """Protocol that every LLM-generated subgame module must satisfy.

    The core library's verification harness calls these methods.
    """

    def create_registry(self) -> ObjectRegistry:
        """Define game object types and how to detect them on the grid."""

    def transition(self, state: GameState, action: Action) -> GameState:
        """Apply an action: run the action's transition, then physics, then constraints.
        Returns the predicted next state."""

    def get_transition_log(self) -> list[str]:
        """Return a log of which transitions, physics, and constraints fired
        during the last call to transition(). Used for error diagnosis."""
```

#### 5.2.7 Verification Harness

```python
@dataclass
class PredictionResult:
    """Result of comparing a predicted state against an observed state."""

    step: int
    action: Action
    predicted_grid: Grid
    actual_grid: Grid
    cell_diffs: list[tuple[int, int, int, int]]  # (row, col, predicted, actual)
    is_correct: bool
    transition_log: list[str]  # Which transitions/physics/constraints fired


@dataclass
class VerificationReport:
    """Full report from replaying observation history against the Python model."""

    total_steps: int
    correct_predictions: int
    failed_predictions: int
    failures: list[PredictionResult]
    regression_failures: list[PredictionResult]  # Failures on previously-correct steps
    accuracy: float  # correct / total


class VerificationHarness:
    """Replays observation history against a SubgameModule to verify correctness."""

    def verify_single(self, module: SubgameModule,
                      before_grid: Grid, action: Action,
                      actual_after_grid: Grid) -> PredictionResult:
        """Predict one step and compare against actual."""

    def verify_history(self, module: SubgameModule,
                       observations: list[ObservationRecord]) -> VerificationReport:
        """Replay all observations, producing a full verification report.

        Each observation is independently verified (before → action → predicted
        vs. actual after). This catches both current errors and regressions.
        """

    def format_error_for_llm(self, failure: PredictionResult) -> str:
        """Format a prediction failure into a structured error message
        suitable for inclusion in the LLM's next prompt.

        Example output:
          Step 12: ACTION1 (move_up)
          PREDICTION FAILED at 8 cells:
            (20, 12): predicted 0 (empty), actual 5 (wall) — wall_collision constraint missed
            (16, 12): predicted 2 (player), actual 0 (empty) — player did not move here
          Transition log: move_up fired, gravity NOT applied, wall_collision NOT checked
          Likely cause: wall at (20,12) was not registered in the object model
        """
```

### 5.3 Per-Game Subgame Module Structure

Each game's subgame module is a Python file with four sections, all generated and iteratively refined by the LLM from the NL-JSON state. The structure maps directly onto the existing `EnvironmentKnowledge` and `ActionKnowledge` representations.

#### 5.3.1 Section 1: Object Encoding

Maps directly from `EnvironmentKnowledge.identified_objects`:

```python
# === SECTION 1: OBJECT ENCODING ===
# Source: EnvironmentKnowledge.identified_objects + breakthroughs

class Player(GameObject):
    """The player character.
    NL source: 'red (color 2) 4x4 block, responds to ACTION1-4'
    """
    pass

class Wall(GameObject):
    """Impassable wall blocks.
    NL source: 'grey (color 5) blocks, form border and internal maze'
    """
    pass

class PushableBox(GameObject):
    """Boxes that can be pushed by the player.
    NL source: 'orange (color 8) 4x4 block, moves when player pushes into it'
    """
    is_on_target: bool = False

class Target(GameObject):
    """Target positions for boxes.
    NL source: 'yellow (color 7) 4x4 marker, destination for boxes'
    """
    pass

def create_registry() -> ObjectRegistry:
    registry = ObjectRegistry()
    registry.register(color=2, cls=Player)
    registry.register(color=5, cls=Wall)
    registry.register(color=8, cls=PushableBox)
    registry.register(color=7, cls=Target)
    return registry
```

#### 5.3.2 Section 2: Transition Functions

Maps directly from `ActionKnowledge.current_definition`:

```python
# === SECTION 2: TRANSITIONS ===
# Source: ActionKnowledge[ACTION1-5].current_definition

def transition_action1(state: GameState) -> GameState:
    """ACTION1: Move player up.
    NL source: 'Moves player up by one tile. If wall above, no movement.
                If pushable box above with empty space above it, both move up.'
    """
    player = state.objects.get("player")
    if not player:
        return state

    target_row = player.position[0] - 4  # One tile up (4px grid cells)
    target_col = player.position[1]

    # Check what is at the target position
    target_cell_value = state.grid.get(target_row, target_col)

    if target_cell_value == 5:  # Wall
        _log("move_up blocked by wall")
        return state  # No movement

    if target_cell_value == 8:  # Pushable box
        box_target_row = target_row - 4
        box_target_value = state.grid.get(box_target_row, target_col)
        if box_target_value in (0, 7):  # Empty or target
            _log("move_up: pushing box")
            # Move box up
            _move_object(state, "box", box_target_row, target_col)
            # Move player up
            _move_object(state, "player", target_row, target_col)
            return state
        else:
            _log("move_up blocked: box cannot be pushed (obstruction)")
            return state  # Box is blocked

    if target_cell_value == 0:  # Empty space
        _log("move_up: player moves freely")
        _move_object(state, "player", target_row, target_col)

    return state

# Similar functions for ACTION2-5...
```

#### 5.3.3 Section 3: Physics / Utilities

Maps from `EnvironmentKnowledge.spatial_rules` and `movement_constraints`:

```python
# === SECTION 3: PHYSICS / UTILITIES ===
# Source: EnvironmentKnowledge.movement_constraints + spatial_rules + breakthroughs

def apply_gravity(state: GameState) -> GameState:
    """Apply gravity: unsupported objects fall to lowest open cell.
    NL source: breakthrough 'Objects fall when unsupported'
    Only applies if gravity is a discovered mechanic for this game.
    """
    for obj_id, obj in state.objects.items():
        if isinstance(obj, PushableBox):
            # Find lowest empty row below current position
            lowest_row = obj.position[0]
            for row in range(obj.position[0] + 4, state.grid.height - 4, 4):
                if state.grid.get(row, obj.position[1]) == 0:
                    lowest_row = row
                else:
                    break
            if lowest_row != obj.position[0]:
                _move_object(state, obj_id, lowest_row, obj.position[1])
    return state

def check_win_condition(state: GameState) -> bool:
    """Check if all boxes are on targets.
    NL source: domain_description 'Sokoban-like puzzle'
    """
    boxes = [o for o in state.objects.values() if isinstance(o, PushableBox)]
    targets = [o for o in state.objects.values() if isinstance(o, Target)]
    for target in targets:
        if not any(b.position == target.position for b in boxes):
            return False
    return True
```

#### 5.3.4 Section 4: Constraints

Maps from `EnvironmentKnowledge.movement_constraints` and contextual observations:

```python
# === SECTION 4: CONSTRAINTS ===
# Source: ActionObservation.context_that_caused_outcome patterns
#         EnvironmentKnowledge.movement_constraints

def constraint_move_budget(state: GameState, action: Action) -> GameState | None:
    """Move budget constraint: UI dots count remaining moves.
    NL source: breakthrough 'Red dots at top decrease with each effective move.
               When all dots are gone, game resets.'

    Returns None to allow the transition to proceed normally.
    Returns a modified GameState to override the transition result.
    """
    if state.metadata.get("moves_remaining", float("inf")) <= 0:
        _log("constraint: move budget exhausted, action blocked")
        return state  # No-op: out of moves
    return None  # Allow transition to proceed

def constraint_boundary(state: GameState, action: Action) -> GameState | None:
    """Boundary constraint: nothing can move outside the border.
    NL source: 'Color 5 grey border surrounds the play area'
    """
    # The transition functions already check for walls,
    # but this is a safety constraint that catches edge cases
    return None
```

#### 5.3.5 Top-Level Transition Orchestrator

```python
# === ORCHESTRATOR ===
# Combines all sections into the SubgameModule protocol

TRANSITION_MAP = {
    Action.ACTION1: transition_action1,  # Move up
    Action.ACTION2: transition_action2,  # Move down
    Action.ACTION3: transition_action3,  # Move left
    Action.ACTION4: transition_action4,  # Move right
    Action.ACTION5: transition_action5,  # Interact
}

PHYSICS = [
    apply_gravity,          # Only if gravity is discovered
]

CONSTRAINTS = [
    constraint_move_budget,
    constraint_boundary,
]

_transition_log: list[str] = []

def _log(msg: str):
    _transition_log.append(msg)

def transition(state: GameState, action: Action) -> GameState:
    """Full transition: constraints → action → physics → constraints."""
    _transition_log.clear()

    # Pre-transition constraints (may block the action entirely)
    for constraint in CONSTRAINTS:
        override = constraint(state, action)
        if override is not None:
            return override

    # Apply action-specific transition
    transition_fn = TRANSITION_MAP.get(action)
    if transition_fn:
        state = transition_fn(state)
        _log(f"transition {action.value} applied")
    else:
        _log(f"no transition for {action.value}")

    # Apply physics
    for physics_fn in PHYSICS:
        state = physics_fn(state)

    # Post-transition constraints
    for constraint in CONSTRAINTS:
        override = constraint(state, action)
        if override is not None:
            return override

    return state

def get_transition_log() -> list[str]:
    return list(_transition_log)
```

---

## 6. Integration with the LearningAgent Lifecycle

### 6.1 Where the Game Engine Fits in the Existing Phases

The Python game engine introduces two new sub-phases that slot into the existing pipeline without disrupting it. The existing three LLM calls remain unchanged; two new calls are added conditionally.

```
EXISTING PHASE 1: Action Analysis (LLM Call 1)
  → Returns ActionAnalysisResult
  → Updates ActionKnowledge via KnowledgeManager.update_from_analysis()

EXISTING PHASE 1.5: Environment Analysis (LLM Call 2)
  → Returns EnvironmentAnalysisResult
  → Updates EnvironmentKnowledge via KnowledgeManager.update_environment_from_analysis()

  ╔══════════════════════════════════════════════════════════════════╗
  ║ NEW PHASE 1.7: Python Model Prediction & Verification          ║
  ║                                                                ║
  ║   a) If subgame_module exists:                                 ║
  ║      predicted = subgame_module.transition(before_state, action)║
  ║      result = verification_harness.verify_single(              ║
  ║          subgame_module, before_grid, action, actual_after_grid)║
  ║                                                                ║
  ║   b) If prediction correct:                                    ║
  ║      Increment model_correct_count on ActionKnowledge          ║
  ║      (Stronger verification signal than LLM consistency alone) ║
  ║                                                                ║
  ║   c) If prediction incorrect:                                  ║
  ║      error_report = harness.format_error_for_llm(result)       ║
  ║      Store error_report for use in next compile step           ║
  ╚══════════════════════════════════════════════════════════════════╝

  ╔══════════════════════════════════════════════════════════════════╗
  ║ NEW PHASE 1.9: Recompile Python Model (LLM Call 4, periodic)  ║
  ║                                                                ║
  ║   Triggered when:                                              ║
  ║     - NL-JSON model was updated (new definition or env change) ║
  ║     - AND (no Python model exists yet                          ║
  ║           OR prediction was incorrect                          ║
  ║           OR every N actions as a refresh)                     ║
  ║                                                                ║
  ║   Input to LLM:                                                ║
  ║     - Current NL-JSON state (all ActionKnowledge + Environment)║
  ║     - Core library API reference (base classes + methods)      ║
  ║     - Previous subgame module code (if any)                    ║
  ║     - Error report from verification (if any)                  ║
  ║     - Full observation history for regression context          ║
  ║                                                                ║
  ║   Output from LLM:                                             ║
  ║     - Updated Python subgame module code                       ║
  ║                                                                ║
  ║   Post-compilation:                                            ║
  ║     regression_report = harness.verify_history(                ║
  ║         new_module, all_observations)                          ║
  ║     If regressions introduced: log them, optionally re-compile ║
  ╚══════════════════════════════════════════════════════════════════╝

EXISTING PHASE 3: Next Action Suggestion (LLM Call 3)
  → Now includes Python model accuracy as additional context:
    "Python model accuracy: 85% (17/20 correct predictions)"
    "Last failure: ACTION1 at step 14 — wall constraint missed"

EXISTING PHASES 4/5: Execute Action
  → No changes
```

### 6.2 New Fields on Existing Models

Minimal additions to the existing Pydantic models:

```python
# Addition to ActionKnowledge (models.py)
class ActionKnowledge(BaseModel):
    # ... existing fields ...
    model_correct_predictions: int = 0    # Python model predicted correctly
    model_incorrect_predictions: int = 0  # Python model predicted incorrectly
    last_model_error: str | None = None   # Most recent prediction error detail

# Addition to AgentState (models.py)
class AgentState(BaseModel):
    # ... existing fields ...
    subgame_module_code: str | None = None       # Current Python model source
    subgame_module_version: int = 0              # Increments on each recompile
    model_verification_accuracy: float = 0.0     # Overall prediction accuracy
    last_verification_report: str | None = None  # Summary of last full verification
```

### 6.3 New KnowledgeManager Methods

```python
# Addition to KnowledgeManager (knowledge.py)

def format_for_model_compilation(
    self, state: AgentState, error_report: str | None = None
) -> dict[str, str]:
    """Format all knowledge for the LLM model compilation prompt.

    Returns a dict with:
    - 'nl_state': Full NL-JSON state formatted for the compilation prompt
    - 'core_api': Reference for core library base classes
    - 'previous_code': Previous subgame module code (if any)
    - 'error_report': Structured error from last verification failure
    - 'observation_history': All observations for regression context
    """

def update_from_verification(
    self, state: AgentState, action_id: ActionID,
    prediction_result: PredictionResult
) -> AgentState:
    """Update state based on Python model prediction result.

    Increments model_correct_predictions or model_incorrect_predictions
    on the relevant ActionKnowledge.
    """
```

---

## 7. The Compilation Step: NL-JSON to Python Model

### 7.1 The Compilation Prompt

The compilation step is a new (4th) LLM call that translates the current NL-JSON understanding into executable Python code. This is conceptually similar to asking the LLM to "write a unit test" for its own understanding.

**System prompt** (new, `llm_agents.py`):

```
You are a game model compiler. Your job is to translate a natural language
description of a game's mechanics into an executable Python simulation.

You will receive:
1. A structured description of all game objects, their properties and roles
2. Definitions of what each action does
3. Environment constraints and physics rules
4. A core library API that provides Grid, GameObject, GameState, and Action classes
5. Optionally: the previous version of the model and an error report

Your output must be a valid Python module that implements the SubgameModule protocol:
- create_registry(): Register all object types
- transition(state, action): Apply action with physics and constraints
- get_transition_log(): Return what happened during the last transition

CRITICAL RULES:
- Every claim in the NL description must be translated to executable code
- If the NL description is ambiguous, make the most conservative assumption
- Include _log() calls for every decision point (wall check, push check, etc.)
- The model must be deterministic: same state + same action = same result
- Use the core library's Grid, GameObject, and GameState — do not reimplement them
```

**User prompt structure**:

```
GAME OBJECTS (from EnvironmentKnowledge):
{formatted_identified_objects}

ACTION DEFINITIONS (from ActionKnowledge):
{formatted_action_definitions}

ENVIRONMENT CONSTRAINTS (from EnvironmentKnowledge):
{formatted_movement_constraints}
{formatted_internal_walls}
{formatted_spatial_rules}

BREAKTHROUGHS / PHYSICS RULES:
{formatted_breakthroughs}

DOMAIN DESCRIPTION:
{domain_description}

CORE LIBRARY API:
{core_library_reference}

PREVIOUS MODEL (version {version}):
{previous_subgame_module_code_or_none}

ERROR REPORT FROM LAST VERIFICATION:
{error_report_or_none}

Generate the complete Python subgame module.
```

### 7.2 When to Compile

Not every cycle needs a recompilation. The compilation trigger conditions:

| Condition | Compile? | Rationale |
|-----------|----------|-----------|
| No Python model exists yet AND at least 3 actions observed | Yes | Need minimum observations before first model |
| NL-JSON updated AND previous Python prediction was wrong | Yes | Both signals indicate model needs fixing |
| NL-JSON updated AND it's been >5 actions since last compile | Yes | Periodic refresh to incorporate accumulated changes |
| NL-JSON NOT updated AND prediction was correct | No | Model is working, don't waste an LLM call |
| Stage transition detected | Yes | New stage may change rules entirely |

### 7.3 Compilation Output Handling

The LLM's output is a Python source string. The system:

1. **Receives** the code as structured output (or a code block extraction)
2. **Validates** syntax with `ast.parse()` (no execution yet)
3. **Loads** as a module via `importlib` in a sandboxed namespace
4. **Verifies** the module satisfies the `SubgameModule` protocol (has required methods)
5. **Runs** the verification harness against all observation history
6. **Stores** the source in `AgentState.subgame_module_code` if validation passes
7. **Falls back** to the previous version if the new code has syntax errors or runtime crashes

---

## 8. Verification Harness and Feedback Loop

### 8.1 Single-Step Verification

For each action the agent takes:

```
Input:
  before_grid: Grid from before_frame (stored in pending_analysis)
  action: The action that was taken
  actual_after_grid: Grid from the actual API response

Process:
  1. Parse before_grid into GameState using subgame_module.create_registry()
  2. predicted_state = subgame_module.transition(game_state, action)
  3. cell_diffs = predicted_state.grid.diff(actual_after_grid)
  4. transition_log = subgame_module.get_transition_log()

Output:
  PredictionResult:
    is_correct: len(cell_diffs) == 0
    cell_diffs: [(row, col, predicted_val, actual_val), ...]
    transition_log: ["move_up fired", "wall_collision checked", ...]
```

### 8.2 History Replay (Regression Testing)

After every recompilation, the harness replays the **entire observation history**:

```python
# Every ActionObservation has before_frame_path and after_frame_path
# These are loaded, and the model is tested against each one

observations = collect_all_observations(state.action_knowledge)
report = harness.verify_history(new_module, observations)

# report.accuracy = correct / total
# report.regression_failures = steps that WERE correct before but now FAIL
```

This ensures that fixing one transition doesn't break another. Regression failures are prominently highlighted in the next compilation prompt.

### 8.3 Error Signal Format

The error signal fed back to the LLM is structured for maximum diagnostic value:

```
═══════════════════════════════════════════════════
PYTHON MODEL VERIFICATION FAILURE — Step 14, ACTION1
═══════════════════════════════════════════════════

PREDICTION vs ACTUAL (8 cells differ):
  Cell (20, 12): predicted=0 (empty), actual=5 (wall)
  Cell (16, 12): predicted=2 (player), actual=0 (empty)
  Cell (20, 16): predicted=0 (empty), actual=2 (player)
  ...

TRANSITION LOG:
  1. move_up fired
  2. target cell (16,12) checked: value=0 → classified as EMPTY
  3. player moved to (16,12)
  4. gravity NOT applied (not registered)
  5. wall_collision NOT checked for cell (20,12)

LIKELY ISSUE:
  The model moved the player to (16,12) but the actual player ended up at (20,16).
  Cell (20,12) was predicted empty but is actually a wall (color 5).
  → Possible causes:
    a) A wall at (20,12) was not registered in the object model
    b) The transition checked the wrong cell for collision
    c) A constraint should have redirected the movement

NL-JSON DEFINITION FOR ACTION1:
  "Moves player up by one tile, blocked by grey walls"

RELEVANT ENVIRONMENT KNOWLEDGE:
  Internal walls: ["Grey blocks at rows 20-24 form a horizontal wall"]
  → This wall IS described in the NL model but NOT implemented in the Python model
═══════════════════════════════════════════════════
```
