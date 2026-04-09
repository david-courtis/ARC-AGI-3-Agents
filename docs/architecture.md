# System Architecture: World Model Synthesis for ARC-AGI-3

March 2026

---

## 1. Problem Statement

ARC-AGI-3 presents agents with interactive gridworld games rendered as 64x64 RGB
pixel frames. The agent has access to a finite set of anonymous actions (typically
five directional actions plus a click action) and must discover the game mechanics
purely through exploration. No action semantics, object labels, or game rules are
provided a priori.

Our research addresses two capabilities:

1. **Skill acquisition**: learning what actions do to which objects, under what
   conditions, from raw pixel observations.
2. **Experience-driven adaptation**: refining an internal world model as new
   evidence accumulates, including detecting and resolving conditional effects
   and context-dependent preconditions.

We deliberately exclude planning and policy optimization from our scope. The
agent's sole objective is to build an accurate, executable world model that
correctly predicts the outcome of any action in any observed state. The
synthesized Python code is not a plan; it is a simulator. Game score is a
downstream consequence but not the optimization target.

Formally, the environment is a reward-free deterministic MDP
`M = (S, A, T, s_0)` where `S` is the set of reachable grid states,
`A = {a_1, ..., a_N}` is the action set, `T : S x A -> S` is the unknown
transition function, and `s_0` is the initial state. The agent's objective is
to synthesize `T_hat` such that `T_hat(s, a) = T(s, a)` for all observed
`(s, a)` pairs, then generalize to unobserved pairs.

---

## 2. Three Degrees of Complexity

We organize ARC-AGI-3 games by the depth of exploration they require, from
simplest to hardest.

### Degree 1: Object Interactions

The simplest games require discovering what happens when objects interact.
What does pushing a block do? What happens when the player collides with a wall?
What does clicking on a switch change?

These are single-step cause-effect relationships: one action, applied to one
object, produces one observable effect. The exploration challenge is covering
the full (sprite_type x action) matrix.

### Degree 2: Non-Trivial Exploration

Some games require visiting multiple objects in sequence before the game
mechanics become apparent. For example, a game where clicking objects in a
specific order triggers a cascade, or a game where multi-object interactions
produce effects that are not visible from any single action. This resembles a
TSP tour over interesting objects: the agent must systematically navigate to
and interact with each target.

These games expose the need for a **"move to sprite" macro primitive** that
composes directional actions into a navigation sequence toward a target. Without
this primitive, the agent spends most of its action budget on random walks rather
than targeted investigation.

### Degree 3: Gridworld Subset with Object Selection + One Arbitrary Action

The vast majority of ARC-AGI-3 games fall into a common structural pattern:

- Directional movement (ACTION1-4) to navigate a player/cursor.
- Object selection via click (ACTION6) or proximity.
- One "interesting" action (ACTION5 or the click itself) that triggers
  the game-specific mechanic.

This means the agent needs to handle directional movement (largely solved
by player identification and BFS pathfinding), click targeting (solved by
the perception layer), and one domain-specific transition rule. The
combinatorial complexity concentrates in a single action applied across
different object types and contexts.

This observation has a practical consequence: if the system handles "move to
sprite then act" reliably, approximately 90% of ARC-AGI-3 games reduce to
learning a single action's conditional effects across the sprite type space.

---

## 3. Architecture Overview

The system decomposes into five subsystems: perception, epistemic state,
exploration, actions, and synthesis. Perception and actions are deterministic.
Epistemic state and exploration are algorithmic. Synthesis is the only
component that invokes an LLM.

```
                                  ARC-AGI-3 Environment
                                         |
                                    raw RGB frame
                                         |
                                         v
                           +---------------------------+
                           |    Perception Pipeline    |
                           |   (deterministic, no LLM) |
                           +---------------------------+
                           | FragmentDetector          |
                           | SpriteRegistry            |
                           | ComovementTracker         |
                           | ObjectTracker             |
                           | FrameParser (orchestrator) |
                           +---------------------------+
                                         |
                                    WorldState
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
           +------------------------+       +------------------------+
           |    Epistemic State     |       |    Exploration Engine   |
           | (continuous confidence)|       |  (phased, object-centric)|
           +------------------------+       +------------------------+
           | ObservationContext     |       | Phase 0: DETECT         |
           | TransitionRecord      |       | Phase 1: IDENTIFY_PLAYER|
           | CellKnowledge         |       | Phase 2: CLICK_SURVEY   |
           | EpistemicState        |       | Phase 3: EXPLORE        |
           +------------------------+       | Phase 4: SYNTHESIS_READY|
                        |                   +------------------------+
                        |                            |
                        v                            v
           +------------------------+       +------------------------+
           |    Action System       |       |   Synthesis Pipeline    |
           |   (deterministic)      |       |   (LLM-driven)         |
           +------------------------+       +------------------------+
           | ClickPlanner (ACTION6) |       | SynthesisWorkspace     |
           | MacroPlanner (BFS nav) |       | ReflexionLoop          |
           | MacroAction sequencer  |       | SynthesisBackend (ABC) |
           +------------------------+       | ClaudeCodeBackend      |
                        |                   +------------------------+
                        v                            |
                  GameAction to env            transition_rules.py
```

### 3.1 Perception Pipeline (Deterministic)

The perception pipeline converts raw 64x64 RGB frames into structured
WorldState objects. It is entirely deterministic -- no LLM calls, no learned
parameters. This is the most consequential design decision in the system.

**Fragment Detection** (`FragmentDetector`):
Color-based connected component labeling with 8-connectivity. Each contiguous
region of non-background color becomes a Fragment with a color, pixel mask,
bounding box, and centroid. Background (color 0) is excluded. Implementation
uses `scipy.ndimage.label` with a 3x3 structuring element.

**Co-Movement Tracking** (`ComovementTracker`):
Discovers multi-color sprites by observing which fragments consistently move
together across frames. When fragments of different colors always displace by
the same vector, they are merged into a single composite sprite. This is how
the system discovers, for example, that a character is composed of a red body
and blue eyes.

**Sprite Registry** (`SpriteRegistry`):
Maintains a library of discovered sprite types. Each sprite type is defined
by a canonical set of fragment colors, relative positions, and bounding
dimensions. Supports merge (two types discovered to be parts of the same
composite) and split (a type that was incorrectly merged). New fragments
are matched to existing types by color and shape; unmatched fragments create
new types.

**Object Tracker** (`ObjectTracker`):
Assigns persistent track IDs to sprite instances across frames. Uses a
three-pass matching algorithm: (1) color match, (2) spatial overlap among
same-color candidates, (3) nearest-neighbor by centroid for unmatched
objects. Objects that appear in the new frame but not the previous are
marked as "appeared"; objects present before but absent after are
"disappeared."

**Frame Parser** (`FrameParser`):
Orchestrates the full pipeline. Each call to `parse(frame, frame_index)`
returns a WorldState containing all detected sprite instances with assigned
track IDs and type IDs. The parser also feeds transitions into the
ComovementTracker and triggers SpriteRegistry merges when co-movement
evidence exceeds a reliability threshold.

**Justification for deterministic perception**: This decision was validated
empirically on the ls20 ARC-AGI-3 game. When the LLM was asked to synthesize
a `perceive()` function alongside transition logic, it spent all reflexion
iterations debugging perception bugs and achieved 0% transition accuracy
across all iterations. When perception was made deterministic and the LLM
was asked to write only `transition()` rules, accuracy reached 88% within
4 iterations. The lesson is clear: perception is a solved problem for these
grid environments (connected components are sufficient), and asking the LLM
to re-solve it wastes the entire synthesis budget.

### 3.2 Epistemic State (Continuous Confidence)

The epistemic state replaces the binary Known-True / Known-False / Unknown
(KT/KF/UK) model with a continuous confidence representation. The binary
model was found to have two critical failure modes:

1. **Conflation of "no effect" and "blocked by context"**: If the player
   attempts to move right but a wall is in the way, the binary model records
   "ACTION3 has no effect on the player" (KF). This is wrong -- ACTION3 does
   move the player, but was blocked. The continuous model records the full
   observation with its context, preserving the wall's presence as explanatory
   information.

2. **Premature certainty**: With a threshold count of 3 confirmations, the
   binary model declares an effect "known." But 3 observations in the same
   context provide no evidence about behavior in different contexts. The
   continuous model tracks context diversity explicitly.

**ObservationContext**: A fingerprint of the game state at the time of an
observation. Captures what sprite types are nearby, their relative positions,
whether anything was selected, and the player's relative position. Two
observations have different contexts if the spatial arrangement of objects
differs.

**TransitionRecord**: A single observation: which sprite type was the target,
what action was taken, what effect was observed (position delta, appearance
change, existence change), and the full ObservationContext.

**CellKnowledge**: For each (sprite_type, action) pair, stores all
TransitionRecords, computes context diversity (number of unique contexts
observed), effect consistency (fraction of observations agreeing with the
majority effect), and detects conditional effects (when the same type-action
pair produces different outcomes in different contexts).

**EpistemicState**: The full (sprite_type x action) matrix of CellKnowledge
entries. Provides an `exploration_priority(type_id, action_id)` score derived
from: inverse observation count (prefer under-tested cells), inverse context
diversity (prefer cells tested only in one context), and inconsistency bonus
(prefer cells with conflicting observations, indicating conditional effects).

The key insight is that "Known True" is a fiction. The system never
achieves certainty about an effect; it tracks confidence, which naturally
decays the exploration priority as evidence accumulates.

### 3.3 Exploration (Object-Centric, Phased)

Exploration proceeds through five phases, each with a clear entry condition
and purpose.

| Phase | Name | Entry Condition | Purpose |
|-------|------|-----------------|---------|
| 0 | DETECT | First frame received | Parse frame, bootstrap sprite registry, count types |
| 1 | IDENTIFY_PLAYER | At least one sprite type detected | Test ACTION1-4 in sequence; the sprite that moves is the player |
| 2 | CLICK_SURVEY | Player identified, ACTION6 available | Click each sprite type once to discover selectability and click effects |
| 3 | EXPLORE | Player identified | Priority-driven testing based on epistemic state. For each step, pick the (type, action) cell with highest exploration_priority, navigate to a target of that type, execute the action, observe the result |
| 4 | SYNTHESIS_READY | Mean confidence exceeds threshold | Enough evidence to attempt world model synthesis |

The explorer (`ObjectExplorer`) emits `ActionRequest` objects. Each request
is either an atomic action (execute one action) or a macro step (one step
within a multi-step navigation macro). The agent loop processes these
requests, executes the corresponding game actions, and feeds the results
back to the explorer via `observe_result()`.

Phase transitions are automatic. The explorer checks transition conditions
after each observation and advances when criteria are met.

### 3.4 Actions

The action system provides two capabilities beyond raw game actions:

**ClickPlanner**: Computes (x, y) pixel coordinates for ACTION6 (click) from
the perception layer's SpriteInstances. Targets the center of the sprite's
bounding box, clamped to the [0, 63] range. Supports clicking by sprite type
(targets the nearest instance) or by specific position.

**MacroPlanner**: Composes directional actions (ACTION1-4) into navigation
sequences using BFS pathfinding. Given the player's current position and a
target position, the planner computes the shortest path on the grid, avoiding
known walls and obstacles from the epistemic state. The path is emitted as a
`MacroAction` -- a sequence of atomic action steps that the agent loop
executes one at a time.

The "move to sprite then act" pattern is the composition of MacroPlanner
(navigate to target) followed by an atomic action (interact). This is the
fundamental primitive for Degree 2 and Degree 3 games.

### 3.5 Synthesis (Modular Backend + Reflexion Loop)

Synthesis is the only LLM-dependent component. It converts the epistemic
state and replay buffer into executable Python code that predicts transitions.

**SynthesisWorkspace**: A persistent directory on disk containing all
synthesis artifacts:

```
{run_dir}/synthesis/
  transition_rules.py       # The code being synthesized
  test_runner.py            # Verifies code against replay buffer
  context.txt               # Structured game analysis for the LLM
  replay_buffer.pkl         # Serialized observed transitions
  reflections/              # Accumulated reflections (one per iteration)
  history.json              # Accuracy over time
```

The workspace persists across synthesis calls. When new observations are
collected, the replay buffer is updated and synthesis is re-triggered. The
LLM sees the previous code and can edit it incrementally rather than
rewriting from scratch.

**SynthesisBackend (abstract)**: Any system that can read a workspace,
edit `transition_rules.py`, run `test_runner.py`, and iterate on failures.
The interface is a single method: `run(workspace) -> bool`.

**ClaudeCodeBackend**: The primary backend. Spawns the `claude` CLI as a
subprocess with a prompt pointing at the workspace directory. Claude Code
reads `context.txt`, writes or edits `transition_rules.py`, runs the test
runner, reads the failure output, reflects, edits again, and repeats until
all tests pass or it exhausts its turn budget. This provides full agentic
capabilities: Claude Code can create helper functions, refactor code, add
debug prints, and use any strategy it finds effective.

**What the LLM writes**: The LLM writes ONLY a `transition(world_state,
action_id)` function. It does not write perception, rendering, or test
infrastructure. The function takes a dictionary of typed objects with
positions and mutates their positions/existence to predict the next state.
This constraint is critical: it focuses the LLM's entire synthesis budget
on transition logic, which is the only unknown component.

**ReflexionLoop**: Orchestrates the synthesize-test-reflect cycle:
1. Prepare the workspace (write context, serialize replay buffer, generate
   test runner).
2. Invoke the backend.
3. Read back results (accuracy, per-transition pass/fail).
4. If accuracy < 100%, update the workspace with structured error feedback
   and invoke the backend again.
5. Repeat until 100% or max iterations.

The test runner verifies at the object-state level: for each transition in
the replay buffer, it calls `transition()` on the before-state and compares
predicted object positions against actual object positions in the after-state.
The error signal reports exactly which objects had wrong positions and by
how much, enabling targeted fixes.

### 3.6 Level Gating

The agent enforces a strict policy: it does NOT advance to the next level
until the synthesized world model achieves 100% accuracy on all observed
transitions for the current level. If the agent accidentally completes a
level during exploration (e.g., by pushing a block onto a goal), the level
is reset. The agent is informed of the completion but continues learning.

This forces thorough understanding before progression. Without level gating,
the agent would advance to levels whose mechanics it cannot predict, making
subsequent levels harder (the world model for level N informs exploration
strategy for level N+1).

---

## 4. Accuracy Metrics

We cannot use the standard ARC-AGI-3 accuracy metric (game score) because
our agent does not attempt to solve the games. Instead, we define three
metrics that evaluate world model quality directly.

### 4.1 Object Layout Representation

The percentage of sprites in the ground-truth frame that are correctly
detected and matched by the perception pipeline. This measures perception
fidelity independently of transition prediction.

Formally, given ground-truth frame `s` with objects `O` and perception output
`O_hat`: `layout_accuracy = |O matched in O_hat| / |O|`, where a match
requires correct type, correct position (within a tolerance), and correct
color.

### 4.2 Object Relations Representation

A qualitative metric: given the reconstructed domain (sprite types, their
properties, spatial relations), can a human examine the representation and
replicate a sequence of actions to produce the same outcomes? This measures
whether the structured state captures enough information for downstream use,
even if the transition rules are imperfect.

### 4.3 Transition Accuracy

The primary quantitative metric. The percentage of observed transitions
correctly predicted by the synthesized `transition()` function:

```
transition_accuracy = |{(s, a, s') in D : T_hat(s, a) = s'}| / |D|
```

where `D` is the replay buffer and equality is evaluated at the object-state
level (predicted positions vs. actual positions for all tracked objects).

This metric is computed by the test runner during synthesis and reported in
`history.json`. The synthesis loop targets 100%.

---

## 5. Key Design Decisions and Their Justification

### 5.1 Deterministic Perception vs. LLM Perception

| | LLM Perception | Deterministic Perception |
|--|---------------|-------------------------|
| **ls20 accuracy** | 0% (all iterations spent on `perceive()` bugs) | 88% (LLM focused on `transition()` logic) |
| **Failure mode** | LLM wastes synthesis budget on a solved problem | None observed for grid domains |
| **Domain generality** | Could theoretically handle non-grid domains | Limited to grid environments with color-based objects |
| **Maintenance** | Fragile: perception quality varies per prompt | Robust: deterministic, parameter-free |

The Fragment -> Sprite -> Composite hierarchy handles multi-color objects
through co-movement discovery rather than requiring the LLM to reason about
visual composition. On ls20, the deterministic pipeline detected 17
fragments, resolved them into 13 sprite types, discovered a multi-color
merge, and correctly identified the player as type 11 -- all without LLM
involvement.

### 5.2 Continuous Confidence vs. Binary KT/KF/UK

The binary model partitions the (type x action) matrix into three categories:
Known True (effect observed and confirmed), Known False (no effect observed),
and Unknown (not yet tested). This model has two fundamental problems.

**Problem 1: False negatives from blocked actions.** When the player moves
toward a wall, the action has no visible effect. The binary model records this
as KF: "this action has no effect on the player." But the action DOES move
the player -- it was blocked by context. The continuous model records the
observation with its full context (wall was adjacent), preserving the
distinction between "no effect" and "effect blocked."

**Problem 2: Insufficient diversity detection.** Three confirmations in
identical contexts provide no evidence about behavior in novel contexts.
The continuous model tracks unique context count explicitly. An action
observed 10 times in the same corner of the grid has lower confidence
than an action observed 3 times in 3 different spatial arrangements.

### 5.3 Object-Centric Factoring vs. Action-Centric or Monolithic

The codebase implements three agent architectures for comparison:

- **Monolithic** (`monolithic_agent/`): LLM writes a single `predict(frame,
  action_id) -> frame` function. No structural constraints.
- **Action-centric OOP** (`oop_agent/`): LLM writes Action classes that
  orchestrate object updates. Each Action handles all types; each GameObject
  responds passively.
- **Object-centric OOP** (`object_centric_agent/`): LLM writes only
  GameObject classes. Each type handles all actions via `respond()`. No
  Action classes.

The full analysis is in `docs/action-vs-object-centric.md` and
`docs/oop-vs-monolithic.md`. The object-centric approach is the primary
agent because it:
- Synthesizes fewer programs (no Action classes).
- Localizes regressions (fixing one type cannot break another).
- Transfers more cleanly (a learned `Wall` class is self-contained).
- Aligns naturally with the perception pipeline (which already produces
  typed objects).

The tradeoff is update ordering: when objects interact, the order in which
they respond matters. The `respond_order()` function must be correct. This is
an additional failure mode not present in the action-centric design, where
update ordering is explicit in `Action.apply()`.

### 5.4 Persistent Workspace + Agentic Synthesis

The synthesis workspace lives on disk, tied to the run directory. This
means:

- **Code persists across synthesis calls.** When new observations arrive,
  the LLM sees its previous code and can edit incrementally.
- **History accumulates.** Reflections, accuracy traces, and intermediate
  code versions are preserved for debugging and analysis.
- **Backend is swappable.** The abstract `SynthesisBackend` interface allows
  replacing Claude Code with any system that can edit files and run tests:
  a different LLM, a neural program synthesizer, or a human.
- **The LLM has full agency.** Claude Code can read any file in the workspace,
  create helper modules, add debug output, refactor freely. It is not
  constrained to a single function signature edit.

### 5.5 Level Gating: Perfect Model Before Advancement

The agent does not advance to the next level until the world model is 100%
accurate on observed transitions. This is a deliberate tradeoff:

- **Pro**: Forces thorough understanding. A model that is 80% accurate on
  level 1 will make the wrong predictions on level 2, corrupting the
  exploration strategy.
- **Con**: If the model cannot reach 100% (due to LLM limitations or
  representation inadequacy), the agent is stuck.
- **Mitigation**: After a configurable number of synthesis attempts without
  reaching 100%, the agent can lower the threshold or advance with a
  partial model. This is a policy parameter, not a hard constraint.

---

## 6. Contributions

This system introduces five contributions to world model synthesis for
interactive environments:

1. **Deterministic perception pipeline with co-movement-based sprite
   merging.** The Fragment -> Sprite -> Composite hierarchy handles
   multi-color objects without LLM involvement. Co-movement tracking
   discovers composite sprites from temporal evidence rather than
   spatial heuristics.

2. **Continuous epistemic model replacing binary KT/KF/UK.** Each
   (sprite_type, action) cell tracks observation count, context diversity,
   effect consistency, and conditional effects. Exploration priority
   emerges from these quantities rather than from a hand-coded state
   machine.

3. **Object-centric OOP factoring for LLM synthesis.** The world model
   is structured as a set of typed object classes where each type's
   `respond()` method handles all actions. This reduces the synthesis
   search space from monolithic to factored (Proposition 4.2 in
   `docs/oop-formalism.tex`) and enables per-type regression isolation.

4. **Modular synthesis backend with persistent workspace.** The abstract
   `SynthesisBackend` interface and disk-based workspace allow any
   code-editing agent (Claude Code, API-based reflexion, custom systems)
   to serve as the synthesizer. Artifacts persist across calls.

5. **Level-gating policy enforcing perfect model before advancement.**
   Prevents the agent from progressing to environments it cannot predict,
   ensuring that each level's world model is complete before it is used
   to inform exploration of subsequent levels.

---

## 7. Experimental Results

### 7.1 pb01 (Custom Sokoban)

A hand-crafted Sokoban variant used for development. 5 actions, player +
walls + pushable blocks + goals.

- **API reflexion backend**: 88% peak accuracy in 4 iterations. The
  remaining 12% failure was attributable to a push-chain interaction
  (block pushed into another block) that the LLM did not handle correctly.

### 7.2 ls20 (Official ARC-AGI-3 Game)

A game from the ARC-AGI-3 evaluation set.

**Old approach (LLM writes perceive + respond + render)**:
- 0% transition accuracy across all reflexion iterations.
- The LLM spent every iteration fixing perception bugs. Not a single
  iteration addressed transition logic.
- Conclusion: asking the LLM to write perception is catastrophic.

**API reflexion backend (stateless LLM calls, deterministic perception)**:
- 88% peak transition accuracy in 4 iterations.
- 17 fragments detected, 13 sprite types, multi-color merge discovered.
- Player correctly identified as type 11.
- The remaining 12% involved conditional effects that the LLM did not
  yet capture (context-dependent interactions).

**Claude Code backend (agentic, Opus 4.6, deterministic perception)**:
- **100% transition accuracy (8/8)** in a single synthesis run.
- Claude Code read the context, wrote transition rules, ran the test
  runner, and iterated until all tests passed — with full conversation
  history and file editing capabilities.
- The synthesized code correctly models: player paddle movement,
  multi-directional ball physics (sprite_14_merged), conditional entity
  movement (sprite_6, sprite_10), and trail spawning mechanics.
- This validates the full architecture: deterministic perception +
  agentic synthesis backend + persistent workspace = perfect accuracy
  on a real ARC-3 game.

The progression from 0% (LLM perception) → 88% (deterministic perception +
API reflexion) → 100% (deterministic perception + Claude Code agentic backend)
demonstrates the value of each architectural decision.

---

## 8. Open Questions

### 8.1 Partial Observability

Is the state space always partially observable until the game is solved?
Some games may have hidden state (e.g., a counter that is not rendered)
that makes perfect prediction impossible from pixel observations alone.
How should the system detect and handle hidden state?

### 8.2 Gridworld Classification

What exactly classifies a game as a gridworld? Our perception pipeline
assumes color-based connected components on a discrete grid. Games with
continuous motion, overlapping sprites, or transparency would violate
these assumptions. Can the system detect when its assumptions are violated?

### 8.3 Version Spaces for Transition Hypotheses

Currently, the LLM freely generates transition code. A more structured
approach would maintain a version space of candidate transition rules for
each (type, action) cell, pruned by observations. This would provide
stronger theoretical guarantees (convergence to the correct rule with
bounded samples) but may be impractical for complex conditional effects.

### 8.4 Level Transitions

When the agent advances to a new level, should the world model be
re-initialized or carried forward? If carried forward, how should the
system detect that level N+1 has different mechanics from level N? If
re-initialized, how much transfer is lost? The current implementation
resets per-level state but preserves the sprite registry across levels.

### 8.5 Scaling to Complex Interactions

The current system handles pairwise interactions well (player pushes block,
click toggles switch). Multi-step cascading effects (push block, which
pushes another block, which falls due to gravity) require the transition
function to simulate multiple rounds of updates per action. The reflexion
loop may struggle to discover these chains from observations alone.

---

## 9. Relation to Prior Work

**WorldCoder (Tang, Key & Ellis, NeurIPS 2024)**: The closest prior system.
Synthesizes world models as monolithic Python programs via CEGIS. Our system
extends WorldCoder in three directions: (1) OOP factoring of the synthesized
code, reducing the search space from monolithic to factored; (2) deterministic
perception as a preprocessing step, removing perception from the synthesis
target; (3) continuous epistemic model for exploration, replacing WorldCoder's
known/unknown partition.

**PDDL Action Model Learning (Cresswell et al., Arora et al., Juba et al.)**:
Classical approaches learn STRIPS-style action schemas (preconditions, add
lists, delete lists) from observed transitions. Our system learns equivalent
information but encodes it as executable Python rather than declarative PDDL.
This allows arbitrary computational logic (spatial reasoning, iteration,
complex conditionals) that PDDL cannot express. The `transition()` function
is operationally equivalent to a grounded STRIPS transition but is not
constrained to the STRIPS representation language.

**Reflexion (Shinn et al., NeurIPS 2023)**: Verbal self-reflection as a
learning signal for LLM agents. Our synthesis loop follows the same
generate-test-reflect pattern, but with structured per-object error signals
rather than verbal self-reflection. The error signal reports exactly which
objects had wrong positions and by how much, enabling targeted code fixes.

**Object-Centric Learning (Greff et al., Locatello et al.)**: Neural
approaches to discovering object representations from pixels (Slot Attention,
MONet). Our perception pipeline solves the same problem but deterministically:
connected-component labeling replaces learned slot assignment. This sacrifices
generality (only works for grid environments with discrete colors) but gains
determinism and zero-shot transfer.

**DreamCoder (Ellis et al., PLDI 2021)**: Bootstrapping program synthesis
with wake-sleep library learning. Our OOP type hierarchy serves an
analogous role: learned sprite types form a reusable library that transfers
across games. The OOP formalism (Section 4 of `docs/oop-formalism.tex`)
shows that transfer via inheritance reduces sample complexity by limiting
re-synthesis to overridden methods.

---

## 10. Repository Structure

```
src/
  shared/                          # Infrastructure shared across agents
    agent_base.py                  # Abstract SynthesisAgent base class
    exploration_engine.py          # LLM-driven exploration (legacy)
    frame_utils.py                 # Grid extraction utilities
    models.py                      # ActionID enum, shared types
    run_logger.py                  # Structured logging
    object_detection.py            # Shared detection utilities
    knowledge.py                   # Shared knowledge structures
    diff.py                        # State diff computation
    llm_agents.py                  # LLM calling utilities
    vision.py                      # Vision/rendering utilities

  object_centric_agent/            # PRIMARY AGENT (object-centric OOP)
    agent.py                       # Original agent (shared exploration)
    agent_v2.py                    # v2: perception-driven, level-gating
    world_model.py                 # Domain base classes, diff utilities
    synthesis.py                   # Original synthesis (API reflexion)
    synthesis_v2.py                # v2: workspace-based synthesis

    perception/                    # Deterministic perception pipeline
      fragment_detector.py         # Color-based connected components
      sprite_registry.py           # Sprite type library
      comovement_tracker.py        # Multi-color sprite discovery
      object_tracker.py            # Cross-frame identity persistence
      frame_parser.py              # Pipeline orchestrator

    epistemic/                     # Continuous confidence model
      knowledge_state.py           # EpistemicState, CellKnowledge

    exploration/                   # Phased exploration
      object_explorer.py           # 5-phase explorer, priority scoring

    actions/                       # Action system
      click_action.py              # ACTION6 click targeting
      macro_primitives.py          # BFS pathfinding, move-to-sprite

    synth_loop/                    # Modular synthesis infrastructure
      backend.py                   # Abstract SynthesisBackend
      claude_code_backend.py       # Claude CLI integration
      workspace.py                 # Persistent disk workspace
      loop.py                      # ReflexionLoop orchestrator

    state/                         # State representations
      object_state.py              # Fragment, SpriteInstance, WorldState

  oop_agent/                       # Action-centric OOP agent
    agent.py, world_model.py, synthesis.py

  monolithic_agent/                # Monolithic baseline agent
    agent.py, world_model.py, synthesis.py

  learning_agent/                  # Legacy LLM-exploration agent

docs/
  formalism.tex                    # Mathematical foundations (FOL, STRIPS, PAC-MDP)
  oop-formalism.tex                # OOP-specific formalism (type hierarchy, factored bounds)
  oop-vs-monolithic.md             # Controlled comparison: OOP vs monolithic
  action-vs-object-centric.md      # Controlled comparison: action-centric vs object-centric
  architecture.md                  # This document
  run_locally.md                   # Setup instructions
```
