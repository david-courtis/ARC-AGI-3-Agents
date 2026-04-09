# Action-Centric vs Object-Centric World Model Synthesis

## A Controlled Comparison Within the OOP Approach for ARC-AGI-3

March 2026

---

## 1. What This Document Is

This document describes two OOP agents that differ in where transition logic
lives. Both agents impose polymorphic OOP structure on the synthesized code
(unlike the monolithic baseline, which imposes none). The question is not
"should we use OOP?" but rather "given that we use OOP, who should drive the
transition — the action or the object?"

Both agents share:
- 64x64 RGB pixel frames, 5 anonymous actions
- Tabula rasa (no domain knowledge)
- LLM-synthesized world models in Python
- CEGIS verification against a replay buffer
- Identical exploration, phase management, and exploitation

The only variable is the synthesis prompt and which methods the LLM writes.

---

## 2. The Two Approaches

### 2.1 Action-Centric (`oop_agent/`)

Actions drive transitions. The LLM synthesizes both object classes and action
classes:

```python
class Player(GameObject):
    def respond_to_action(self, action, world):
        # passive: called by the action
        ...

class MoveUp(Action):
    def apply(self, world):
        # ACTIVE: selects objects, orchestrates updates
        player = world.get_objects_of_type(Player)[0]
        target = (player.row - 1, player.col)
        walls = world.get_objects_of_type(Wall)
        if not any(w.row == target[0] and w.col == target[1] for w in walls):
            player.row -= 1

class SynthesizedDomain(Domain):
    def perceive(self, frame) -> World: ...
    def get_action(self, action_id) -> Action: ...
```

The transition is:
```
perceive(frame) -> World
get_action(action_id) -> Action
action.apply(world)        # Action drives
world.render() -> frame
```

Action.apply() is the heavy method. It knows about object types, selects
affected objects, and coordinates multi-object updates.

### 2.2 Object-Centric (`object_centric_agent/`)

Objects drive transitions. The LLM synthesizes only object classes — no Action
classes exist:

```python
class Player(GameObject):
    def respond(self, action_id, world):
        # ACTIVE: handles ALL actions for this type
        if action_id == 1:    # inferred: move up
            target = (self.row - 1, self.col)
            walls = world.get_objects_of_type(Wall)
            if not any(w.row == target[0] ...):
                self.row -= 1
        elif action_id == 2:  # inferred: move down
            ...

class Wall(GameObject):
    def respond(self, action_id, world):
        pass  # walls never change

class SynthesizedDomain(Domain):
    def perceive(self, frame) -> World: ...
    # optionally: respond_order(world, action_id) -> list
```

The transition is:
```
perceive(frame) -> World
for obj in respond_order(world, action_id):
    obj.respond(action_id, world)   # each object drives itself
world.render() -> frame
```

Each object's respond() is the heavy method. It handles all 5 actions via
if/elif on the action_id. Actions are just integer signals, not objects.

---

## 3. The Structural Difference

The two designs are duals of each other. They represent the same information
but organize it along different axes.

### 3.1 Where the Logic Lives

| | Action-centric | Object-centric |
|--|---------------|----------------|
| **Heavy classes** | Action subclasses | GameObject subclasses |
| **Light classes** | GameObject subclasses | (no Action classes) |
| **Each method handles** | One action, all object types | All actions, one object type |
| **Cross-object coordination** | Explicit in Action.apply() | Implicit via world queries |
| **Locus of control** | Action selects and updates objects | Objects check their own state |

### 3.2 The Dispatch Table

For T object types and A actions, the full behavior is a T × A matrix. Each
cell is "what happens to object type T_i when action A_j occurs."

**Action-centric**: The matrix is sliced by columns. Each Action class handles
one column (one action, all object types). The LLM writes A methods that each
reference up to T object types.

**Object-centric**: The matrix is sliced by rows. Each GameObject class handles
one row (one object type, all actions). The LLM writes T methods that each
reference up to A action IDs.

### 3.3 The Extensibility Tradeoff

| Adding a new... | Action-centric cost | Object-centric cost |
|-----------------|--------------------|--------------------|
| Object type | Modify every Action class | Write one new GameObject class |
| Action | Write one new Action class | Modify every GameObject class |

This is the classic expression problem from programming language theory. Neither
approach dominates — the right choice depends on which axis varies more. In
ARC-AGI-3, neither axis is known in advance.

---

## 4. Conjectures

### Conjecture 1: Object-Centric Wins on Entity-Rich Games

**Statement**: For games with 4+ visually distinct entity types that behave
independently, object-centric synthesis produces higher accuracy than
action-centric.

**Rationale**: When entities behave independently, each object's respond()
method is self-contained. The LLM writes T independent methods. In the
action-centric case, each Action.apply() must reference all T types, creating
coupling. If the LLM gets one object's behavior wrong inside an Action, it
may corrupt others during refinement.

More formally: if entity behaviors are independent, the respond() methods are
shorter (each handles only its own state) than the apply() methods (each
handles all entities). Shorter methods mean fewer opportunities for LLM error.

**How to test**: Compare first-attempt accuracy on games with varying entity
counts. Control for game complexity.

### Conjecture 2: Action-Centric Wins on Interaction-Heavy Games

**Statement**: For games where actions trigger coordinated multi-object effects
(push chains, cascading transformations, simultaneous updates), action-centric
synthesis produces higher accuracy.

**Rationale**: Multi-object coordination is explicit and local in Action.apply():
the method sees all affected objects and can order their updates. In the
object-centric case, coordination is implicit: each object queries the world
independently, and the respond_order() must be correct for the updates to be
consistent. If the Player moves before the Block checks for pushes, the Block
sees the Player's new position. If the order is wrong, the behavior is wrong.

This is the classic argument for imperative control flow over autonomous agents:
when coordination matters, a single orchestrator (the Action) is safer than
distributed decision-making (each object independently).

**How to test**: Identify games with push mechanics, chain reactions, or
simultaneous effects. Compare accuracy specifically on transitions involving
multi-object interactions.

### Conjecture 3: Object-Centric Has Lower Regression Rate

**Statement**: When refining a counterexample that involves one object type,
object-centric refinement is less likely to regress behavior for other types.

**Rationale**: In the object-centric case, fixing Player.respond() does not
touch Wall.respond() or Block.respond(). The methods are physically separate.
In the action-centric case, fixing MoveUp.apply() may change behavior for
all object types that MoveUp interacts with, because they are all handled in
one method.

**How to test**: Track correct-to-incorrect transitions per CEGIS step. Measure
whether regressions correlate with "other object types" more in the
action-centric case.

### Conjecture 4: respond_order Is a Hidden Hazard

**Statement**: Object-centric agents are more sensitive to update ordering bugs
than action-centric agents.

**Rationale**: In the action-centric case, update ordering is explicit within
each Action.apply() — the LLM writes the order directly. In the object-centric
case, the global respond_order() determines which objects respond first. If the
LLM does not synthesize a correct respond_order(), objects may see stale state
from other objects. This is a new failure mode that does not exist in the
action-centric design.

Default respond_order (iterate all objects) works only when entities are truly
independent. For any game with interactions, the order matters and must be
correct.

**How to test**: Count transitions where the prediction error is attributable
to ordering (i.e., running the same respond methods in a different order would
produce the correct result).

### Conjecture 5: Object-Centric Transfers Better

**Statement**: Learned object types from the object-centric agent transfer
across games more effectively than learned action+object types from the
action-centric agent.

**Rationale**: An object-centric Wall class encapsulates everything about walls:
how they look, how they respond to every action (namely, they don't). To reuse
this Wall in a new game, just include it. No Action classes need to know about
it.

An action-centric Wall class only encapsulates rendering and a passive
respond_to_action. The actual "wall blocks movement" logic lives in each
Action's apply(). To transfer "walls block movement" to a new game, you need
to transfer not just the Wall class but every Action class that references Wall.
The reusable unit is larger and more entangled.

**How to test**: Train both agents on game 1. Extract learned types. Deploy on
game 2 with the library. Measure initial accuracy before re-synthesis.

---

## 5. Theory

### 5.1 Factoring the Behavior Matrix

Let B be the T × A behavior matrix where B[i,j] is the transition logic for
object type i under action j. The full world model must encode all T × A cells.

**Action-centric factoring**: B is stored as A column-vectors, each of length T.
Each Action class holds one column. Total synthesized programs: A + T + 1
(A actions, T objects with render/respond_to_action, 1 perceive).

**Object-centric factoring**: B is stored as T row-vectors, each of length A.
Each GameObject class holds one row. Total synthesized programs: T + 1
(T objects with render/respond, 1 perceive). No Action classes needed.

The object-centric approach synthesizes fewer programs (no Action classes),
which means fewer interface boundaries and less total code. But each
respond() method is longer (handles A actions) than each respond_to_action()
(handles 1 action).

### 5.2 Method Length Analysis

Let L(t,a) be the code length for the behavior of type t under action a.

**Action-centric**: Action_a.apply() has length ≈ Σ_t L(t,a) (it handles all
types for one action). Object_t.respond_to_action() has length ≈ L(t,a) for the
specific action dispatched to it.

**Object-centric**: Object_t.respond() has length ≈ Σ_a L(t,a) (it handles all
actions for one type).

The longest method in each approach:
- Action-centric: max_a Σ_t L(t,a) — the most complex action
- Object-centric: max_t Σ_a L(t,a) — the most complex object type

Neither is universally shorter. If one object type is much more complex than
others (e.g., a Player that responds to all actions while walls are no-ops),
the object-centric approach concentrates that complexity. If one action is much
more complex than others (e.g., a "push" action that affects many object types),
the action-centric approach concentrates it.

### 5.3 Coordination Complexity

The critical distinction is how cross-object interactions are encoded.

**Action-centric**: Cross-object interaction is local to Action.apply(). The
Action method has all participants visible and can order updates explicitly:

```python
class Push(Action):
    def apply(self, world):
        player = world.get_objects_of_type(Player)[0]
        direction = self.get_direction()
        target = player.position + direction
        block = world.get_at(target)
        if block and isinstance(block, PushableBlock):
            block_target = block.position + direction
            if not world.get_at(block_target):
                block.position = block_target  # move block first
                player.position = target       # then move player
```

**Object-centric**: Cross-object interaction requires querying. Each object
checks the world independently:

```python
class PushableBlock(GameObject):
    def respond(self, action_id, world):
        # Check if something pushed into me
        player = world.get_objects_of_type(Player)[0]
        direction = action_id_to_direction(action_id)
        if player.position + direction == self.position:
            block_target = self.position + direction
            if not world.get_at(block_target):
                self.position = block_target
```

The object-centric version requires that Player.respond() runs before
PushableBlock.respond() (so the block can see whether the player attempted to
move into it). This ordering dependency is implicit and must be captured in
respond_order(). The action-centric version has no such requirement because the
Action explicitly controls the sequence.

### 5.4 Synthesis Complexity for the LLM

What does the LLM need to keep in working memory during synthesis?

**Action-centric**: When writing MoveUp.apply(), the LLM needs to know all
object types and their properties. The method is a mini-orchestrator.

**Object-centric**: When writing Player.respond(), the LLM needs to know all
action semantics and the interfaces of other object types it queries.

In practice, the LLM's working memory is the context window. Both approaches
include the full replay buffer in the prompt. The difference is what the LLM
must reason about simultaneously within each method. Neither approach strictly
requires less reasoning — the complexity is rearranged, not reduced.

### 5.5 The Expression Problem in Synthesis

New ARC-AGI-3 games may stress either axis:
- A game with many entity types but simple actions (e.g., 8 types, each just
  moves in one direction per action) favors object-centric.
- A game with few entity types but complex actions (e.g., 2 types, but actions
  trigger elaborate multi-step effects) favors action-centric.

Neither approach handles both extensions gracefully. This suggests a possible
hybrid: start with one approach, detect poor convergence, and switch to the
other. The shared replay buffer and CEGIS infrastructure make this feasible.

---

## 6. Metrics for Evaluation

Same primary metrics as the OOP-vs-monolithic comparison (see
`docs/oop-vs-monolithic.md` section 6), plus:

| Metric | What it measures |
|--------|-----------------|
| **Ordering sensitivity** | (Object-centric only) Fraction of errors attributable to respond_order |
| **Method length distribution** | Std dev of synthesized method lengths — high variance means unbalanced factoring |
| **Cross-type references** | Number of isinstance/get_objects_of_type calls per method — measures coupling |
| **Refinement locality** | When fixing a counterexample, how many classes were changed? |

### What We Expect

- Games with independent entities: object-centric wins (cleaner factoring)
- Games with coordinated effects: action-centric wins (explicit coordination)
- Overall regression rate: object-centric lower (isolated methods)
- Ordering bugs: object-centric has more (new failure mode)
- Transfer: object-centric transfers more cleanly (self-contained types)
- Token usage: object-centric slightly lower (no Action classes to synthesize)

---

## 7. Implementation Details

### 7.1 Shared Infrastructure

Both agents inherit from `agents.agent.Agent`. The exploration policy, phase
management, CEGIS loop, and exploitation heuristic are identical. Frame
utilities are duplicated in each agent's world_model.py for independence.

### 7.2 Action-Centric Specifics (`oop_agent/`)

The LLM synthesizes T + A + 1 programs:
- T GameObject subclasses with respond_to_action(action, world) and render(frame)
- A Action subclasses with apply(world)
- 1 SynthesizedDomain with perceive(frame) and get_action(action_id)

The Domain.transition() calls: perceive -> get_action -> action.apply -> render.

### 7.3 Object-Centric Specifics (`object_centric_agent/`)

The LLM synthesizes T + 1 programs:
- T GameObject subclasses with respond(action_id, world) and render(frame)
- 1 SynthesizedDomain with perceive(frame) and optionally respond_order()

The Domain.transition() calls: perceive -> respond_order -> broadcast respond
to each object -> render.

Key difference: no get_action() method, no Action base class. The action_id
integer is passed directly to each object.

### 7.4 Running the Agents

```
python main.py -a oopagent              # Action-centric OOP
python main.py -a objectcentricagent    # Object-centric OOP
python main.py -a monolithicagent       # Monolithic baseline
```

All three can run in parallel on the same game. They share no state.

---

## 8. Open Questions

1. **Can we detect which approach to use?** Given a few initial observations,
   can we estimate whether the game is entity-rich (favoring object-centric) or
   interaction-heavy (favoring action-centric) and pick the right agent?

2. **Is respond_order learnable?** Can the LLM reliably synthesize the correct
   respond_order from observations, or does it need explicit guidance? Should we
   expose ordering errors as a distinct counterexample type?

3. **Hybrid approaches**: Could an object's respond() method dispatch to
   action-specific sub-methods? This would give per-type organization with
   per-action modularity within each type. Is this the best of both worlds or
   added complexity for no gain?

4. **Does the LLM prefer one style?** LLMs have training-data priors. If most
   Python code uses action-centric patterns (function handles all types) or
   object-centric patterns (class handles all operations), the LLM may be
   systematically better at one. This is an empirical question about the LLM's
   implicit bias.

5. **Scale effects**: As games get more complex (more types, more actions), does
   one approach degrade faster than the other? At what T × A size does the
   factoring advantage (if any) become decisive?

---

## 9. Relation to Prior Work

**The Expression Problem (Wadler, 1998)**: The action-centric vs object-centric
distinction is a direct instantiation of the expression problem. Action-centric
corresponds to functions over datatypes (easy to add new functions/actions, hard
to add new datatypes/objects). Object-centric corresponds to the OOP solution
(easy to add new classes/objects, hard to add new methods/actions).

**Entity-Component-System (ECS)**: Game engines often use ECS, where components
hold data and systems hold behavior (roughly action-centric). The object-centric
approach is closer to traditional OOP game design where game objects are "smart"
and handle their own behavior.

**Actor Model (Hewitt, 1973)**: The object-centric approach resembles the actor
model: each object is an autonomous agent that receives messages (action IDs)
and updates its own state. The action-centric approach is more like a central
coordinator sending commands.

**WorldCoder (Tang, Key & Ellis, NeurIPS 2024)**: Uses monolithic transition
functions. Neither action-centric nor object-centric. Our comparison tests
whether imposing either OOP factoring improves on the monolithic baseline, and
which factoring direction works better.
