# OOP vs Monolithic World Model Synthesis

## A Controlled Comparison for LLM-Driven World Discovery in ARC-AGI-3

March 2026

---

## 1. What This Document Is

This document describes two agents that solve the same problem with different
constraints on the synthesized code. Both agents:

- Observe 64x64 RGB pixel frames
- Have 5 anonymous actions
- Know nothing about the game domain
- Use an LLM to synthesize a world model (Python code) from observed transitions
- Verify the model against a replay buffer (CEGIS loop)
- Exploit the verified model to play the game

The only variable is what structural constraints the synthesis prompt imposes.

---

## 2. The Two Approaches

### 2.1 Monolithic Agent (`monolithic_agent/`)

The LLM synthesizes a single class with one method:

```python
class SynthesizedModel(WorldModel):
    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        # ... anything goes ...
```

No structure is imposed. The LLM can write if-elif chains, build internal
data structures, do object detection inline, use helper functions. Whatever
it wants. The only contract is: given a frame and an action, produce the
predicted next frame.

### 2.2 OOP Agent (`oop_agent/`)

The LLM synthesizes multiple classes following a polymorphic pattern:

```python
class Player(GameObject):
    def respond_to_action(self, action, world):
        # how Player reacts to this action

class Wall(GameObject):
    def respond_to_action(self, action, world):
        # how Wall reacts (probably no-op)

class MoveUp(Action):
    def apply(self, world):
        # dispatch to affected objects

class SynthesizedDomain(Domain):
    def perceive(self, frame) -> World:
        # parse frame into typed objects
    def get_action(self, action_id) -> Action:
        # map action ID to Action class
```

The structural constraint is: separate the model into object types (each with
a polymorphic respond_to_action method) and action types (each with an apply
method that dispatches to objects). The Domain ties perception, actions, and
rendering together. The transition is: perceive -> get_action -> apply -> render.

The base classes enforce only the pattern: `GameObject` has `obj_id` and
abstract methods `respond_to_action` and `render`. No domain-specific properties
(position, color, movement, blocking) are baked in. The LLM defines whatever
properties each object type needs.

---

## 3. What We Are Comparing

Both agents share identical:
- Exploration policy (round-robin untested actions, then least-observed)
- Replay buffer structure
- CEGIS loop mechanics (synthesize -> verify -> refine with counterexamples)
- Phase management (explore -> synthesize -> exploit)
- Exploitation heuristic (one-step lookahead, pick action with most change)
- Frame analysis utilities (color detection, region finding, diff computation)
- LLM provider configuration

The only difference is the synthesis prompt and the base classes provided to
the exec'd code. This isolates the effect of the OOP structural constraint.

---

## 4. Conjectures

We state these as testable hypotheses. Each can be evaluated by running both
agents on the same game and comparing metrics.

### Conjecture 1: Sub-program Length and Synthesis Accuracy

**Statement**: OOP synthesis produces higher first-attempt accuracy than
monolithic synthesis when the game has 3 or more visually distinct entity types.

**Rationale**: LLM code generation quality degrades with function length.
A monolithic predict() handling T object types and A actions is one function
with up to T x A interacting cases. The OOP decomposition splits this into
~T + A shorter methods, each handling a single concern.

If the LLM's per-token error rate is p, then the probability of a correct
function of length L is approximately (1 - p)^L. For two functions of length
L/2, the joint probability is (1 - p)^(L/2) x (1 - p)^(L/2) = (1 - p)^L.
The overall probability is the same, but each component can be verified and
fixed independently. The monolithic function either works or it doesn't.

**How to test**: Record first-attempt verification accuracy for both agents
across games of varying entity complexity.

### Conjecture 2: Regression Rate During Refinement

**Statement**: OOP refinement has a lower regression rate than monolithic
refinement. That is, when fixing a counterexample, OOP is less likely to break
a previously correct prediction.

**Rationale**: In OOP, fixing Player's response to MoveUp does not touch
Wall's response to anything. In the monolithic case, the entire predict()
function is regenerated, and the LLM may inadvertently change behavior for
unrelated action/entity combinations.

Formally: let R(n) be the probability that refinement step n breaks a
prediction that was correct at step n-1. We conjecture R_OOP(n) < R_mono(n)
for all n.

**How to test**: Track the set of correctly predicted transitions at each
CEGIS iteration. Count how many move from correct to incorrect after each
refinement.

### Conjecture 3: Convergence Speed

**Statement**: For games with interacting objects, OOP converges to a perfect
model in fewer CEGIS iterations than monolithic.

**Rationale**: OOP counterexamples are more informative. When a transition
fails, the OOP structure makes it possible (in principle) to identify which
object type or action type is responsible. The refinement prompt can say "the
Player class's response to MoveUp is wrong" rather than "the predict function
is wrong somewhere." This targeted feedback reduces the LLM's search space
during refinement.

**How to test**: Count CEGIS iterations to perfect accuracy (or to a fixed
accuracy threshold, if perfect is not reached).

### Conjecture 4: Monolithic Wins on Simple Domains

**Statement**: For games with 1 moving entity and no interactions (or games
that are better modeled as pixel transformations rather than entity dynamics),
monolithic achieves equal or higher accuracy with fewer LLM tokens.

**Rationale**: The OOP decomposition has overhead: the perceive() method must
parse the frame into typed objects and recreate them every frame. If the game
does not naturally decompose into objects (e.g., cellular automata, global
pixel shifts), this parsing step introduces errors before any rule learning
even happens. The monolithic approach avoids this by operating directly on
pixels.

**How to test**: Run both agents on a game with a single moving dot and no
interactions. Compare accuracy and LLM token usage.

### Conjecture 5: Transfer Advantage

**Statement**: Learned OOP object types transfer across games more effectively
than learned monolithic predict() functions.

**Rationale**: A learned Wall class (renders as a colored block, does not
respond to any action) is reusable across any game that has walls. A learned
monolithic predict() function is specific to the exact game it was trained on
and contains all behaviors entangled in one function.

Transfer with OOP: load existing class library, attempt to perceive and
transition, identify which classes need modification, synthesize only the
new/changed classes.

Transfer with monolithic: start from scratch every time (or use the previous
predict() as a "starting point" prompt, but the LLM must understand the entire
function to modify any part of it).

**How to test**: Train agent A on game 1, then deploy on game 2. Measure how
many transitions agent A gets correct on game 2 before any re-synthesis. Do
this for both agents.

---

## 5. Theory

### 5.1 The Synthesis Search Space

Consider a game with T object types and A actions. The world model must encode
the behavior of each (object type, action) pair.

**Monolithic search space**: The LLM must produce one program P that
simultaneously handles all T x A behavior patterns. The search space is the
set of all valid Python programs of a given length. Errors anywhere in P
invalidate the entire model.

**OOP search space**: The LLM must produce T + A + 1 programs (T object
classes, A action classes, 1 perceive function), each handling a single
concern. The search space factorizes as:

    W = W_perceive x W_object1 x ... x W_objectT x W_action1 x ... x W_actionA

Each factor is searched independently (modulo interface contracts).

### 5.2 Error Attribution

Given a counterexample (frame, action, expected_next_frame), where the model
prediction differs from the expected:

**Monolithic**: The error is in predict(). The LLM must inspect the entire
function to find and fix the bug. The prompt says "your predict function
produces the wrong output for this input."

**OOP**: The error can be partially attributed. By examining which pixels
differ and which objects occupy those pixels (according to the perceive
function), the agent can hypothesize which object type's respond_to_action
is wrong, or whether the perceive function itself is wrong. The prompt can
say "the Player at position (20,30) was predicted to move to (20,34) but
actually moved to (20,26) in response to ACTION3."

This is not always possible (if perceive itself is wrong, attribution is
unreliable). But when perception is correct, error attribution narrows the
repair to a single class.

### 5.3 Compositionality and Independence

A key assumption behind the OOP advantage is that object behaviors are mostly
independent. Specifically:

**Independence assumption**: Object O1's response to action A depends only on
O1's state and a small number of other objects' states (those it interacts
with), not on all objects in the world.

When this holds, each respond_to_action method is a function of local state,
and fixing one method cannot break another. The OOP factorization is sound.

When this fails (e.g., a global rule like "all objects of color X
simultaneously flip to color Y"), the OOP decomposition is misleading. The
behavior is a property of the action, not of individual objects. A monolithic
approach may handle this more naturally.

Most ARC-AGI-3 games satisfy the independence assumption (they are designed
by humans who think in terms of objects with local interactions). But this is
an empirical claim, not a guarantee.

### 5.4 Perception as the Bottleneck

Both approaches must ultimately parse raw pixels. The monolithic agent does
this implicitly (the predict() function must figure out what pixels to move).
The OOP agent does this explicitly (the perceive() method must segment the
frame into typed objects).

This creates an asymmetry. The OOP agent can fail in ways the monolithic agent
cannot: if perceive() misidentifies an object's type or boundaries, all
downstream behavior is wrong, even if the respond_to_action methods are
correct. The monolithic agent does not have this failure mode because it never
commits to an explicit object decomposition.

Conversely, when perceive() is correct, the OOP agent has a structural
advantage: it operates on a clean, typed representation rather than raw pixels.

This suggests that the OOP advantage grows with perception accuracy, and that
improving perception (via better prompting, specialized vision, or iterative
refinement of perceive) has high leverage in the OOP approach.

### 5.5 The Library Hypothesis

The strongest argument for OOP is not about single-game performance. It is
about the accumulation of reusable components across games.

**Claim**: Over N games, the OOP agent builds a library of K object types.
For game N+1, the expected number of new types needed decreases as K grows,
because games share common entity archetypes (walls, movable objects,
collectibles, hazards, goals).

For the monolithic agent, no such accumulation occurs. Each game requires a
full predict() function synthesized from scratch.

This is not tested by running a single game. It requires a multi-game
evaluation protocol where the agent is given a sequence of games and we
measure per-game synthesis cost over time.

---

## 6. Metrics for Evaluation

We deliberately exclude logical dimensionality (K double perp) as a metric.
It is a property of the hypothesis class, not of the synthesis process.
In practice, the LLM's synthesis is not a systematic search through a
hypothesis space. It is a noisy, heuristic process whose performance depends
on prompt quality, code length, and the LLM's training distribution. The
theoretical bound is an upper bound on a systematic search that nobody
actually performs.

Instead, we use these empirical metrics:

### 6.1 Primary Metrics

| Metric | What it measures |
|--------|-----------------|
| **First-attempt accuracy** | Fraction of replay buffer transitions correctly predicted by the first synthesized model, before any CEGIS refinement |
| **Convergence iterations** | Number of CEGIS iterations to reach perfect accuracy (or timeout) |
| **Final accuracy** | Best accuracy achieved across all synthesis attempts |
| **Regression rate** | Fraction of CEGIS refinements that break a previously correct prediction |
| **Game score** | Final game score (the actual objective) |

### 6.2 Secondary Metrics

| Metric | What it measures |
|--------|-----------------|
| **LLM tokens** | Total input + output tokens consumed during synthesis |
| **Synthesized code length** | Lines of code in the final synthesized model |
| **Code complexity** | Number of distinct classes (OOP) or cyclomatic complexity (monolithic) |
| **Synthesis failures** | Number of attempts where exec() failed (syntax errors, runtime errors) |
| **Perception accuracy** | (OOP only) Fraction of objects correctly identified and typed by perceive() |
| **Transfer score** | (Multi-game) Score on game N+1 before any re-synthesis |

### 6.3 What We Expect

Based on the conjectures above, our predictions:

- For games with 1 entity type: monolithic wins or ties on all metrics
- For games with 2-3 entity types: mixed results, game-dependent
- For games with 4+ entity types: OOP wins on accuracy and convergence
- Across a sequence of games: OOP accumulates advantage via library reuse
- Monolithic always uses fewer LLM tokens (no perceive/render overhead)

---

## 7. Implementation Details

### 7.1 Shared Infrastructure

Both agents inherit from `agents.agent.Agent` and implement `is_done()` and
`choose_action()`. They share the same game API interface, frame format, and
action encoding.

The exploration policy, phase management, and CEGIS loop structure are
identical in both agents (duplicated, not shared, to maintain independence).

Frame utilities (compute_diff, find_unique_colors, find_color_regions,
region_bbox, most_common_color) are duplicated in both agents' world_model.py
files.

### 7.2 OOP-Specific Details

The OOP base classes define only structural contracts:

- `GameObject(obj_id, **properties)`: takes an ID and arbitrary keyword
  arguments as properties. No fixed attributes like position, color, alive.
  The LLM decides what each object type needs.
- `Action(action_id)`: takes the action ID. No preconditions baked in.
- `World(frame, objects)`: holds the frame and a list of objects. Provides
  generic query methods (get_objects_of_type, get_by_id, add/remove).
- `Domain`: abstract with perceive, get_action. Concrete transition method
  calls perceive -> get_action -> apply -> render.

The synthesis prompt explicitly instructs the LLM to create separate classes
for each entity type and each action. It explains the polymorphic dispatch
pattern and shows the available base classes.

### 7.3 Monolithic-Specific Details

The monolithic base class is:

- `WorldModel`: abstract with predict(frame, action_id) -> frame.
- `IdentityModel`: trivial baseline (predicts no change).

The synthesis prompt gives the LLM complete freedom. It explains the frame
format and shows the available utilities, but imposes no structure on how
predict() works internally.

### 7.4 Running the Agents

The agents are independent packages. Each can be run separately:

```
python main.py -a oopagent        # OOP agent
python main.py -a monolithicagent  # Monolithic agent
```

Or in parallel on the same game for comparison. They do not share state.

---

## 8. Open Questions

1. **Is the OOP decomposition always correct?** Some games may have global
   rules that do not decompose into per-object behaviors. How do we detect
   this and fall back?

2. **How sensitive is OOP to perception errors?** If perceive() gets the
   object boundaries wrong by a few pixels, does the entire model collapse
   or degrade gracefully?

3. **Does the LLM actually follow the OOP structure?** If the LLM is prompted
   to write separate classes but puts all logic in one class's respond_to_action,
   we get OOP in name only. How do we verify that the synthesized code actually
   factorizes behavior?

4. **What is the right granularity for object types?** Too few types
   (everything is GenericObject) gives no decomposition benefit. Too many types
   (every pixel group is its own type) creates overhead. How does the agent
   decide?

5. **Can we hybridize?** Start monolithic, and if the CEGIS loop stalls,
   re-prompt with OOP structure as a hint. This treats OOP as a fallback
   inductive bias.

---

## 9. Relation to Prior Work

**WorldCoder (Tang, Key & Ellis, NeurIPS 2024)**: Synthesizes world models as
Python programs. Uses a monolithic transition function with if-elif dispatch.
Our OOP agent extends this by imposing polymorphic structure on the synthesized
code. Our monolithic agent is a closer reproduction of WorldCoder's approach.

**PDDL-based approaches**: Use declarative action schemas with preconditions
and effects. The OOP approach can express everything PDDL can (actions with
preconditions, conditional effects) plus things PDDL cannot (complex spatial
computations, procedural rendering, dynamic object creation). The monolithic
approach can also express everything, but without the factored structure.

**Object-centric learning (Greff et al., Locatello et al.)**: Learn object
representations from pixels via neural networks. Our approach is symbolic: the
LLM writes explicit Python code for perception and transition. The object types
are named classes, not latent vectors.

**Program synthesis (Gulwani, Polozov & Singh)**: Our CEGIS loop follows the
standard counterexample-guided inductive synthesis pattern. The novelty is
using an LLM as the synthesizer (rather than enumerative or constraint-based
search) and structuring the synthesis target as an OOP class hierarchy.
