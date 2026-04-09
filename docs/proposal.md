# Object-Centric World Model Synthesis via Factored Program Induction

**A Proposal Extending WorldCoder with OOP Factoring, Continuous Epistemic Exploration, and Object-Centric Transition Functions**

April 2026

---

## 1. Problem Statement

An agent is placed in an unknown interactive environment with discrete states and a finite action set. No action semantics, object labels, or transition rules are provided. The agent must synthesize an executable world model -- a program that correctly predicts the next state given the current state and an action -- purely from interaction experience.

WorldCoder (Tang, Key & Ellis, NeurIPS 2024) demonstrated that LLMs can synthesize such models as monolithic Python programs via a CEGIS loop, achieving strong sample efficiency (~50 interactions) on Sokoban, MiniGrid, and AlfWorld. However, the monolithic program structure limits scalability: the LLM must reason about all object types and interactions in a single function, the search space grows combinatorially with domain complexity, and learned knowledge cannot transfer across domains.

We propose three extensions that address these limitations:

1. **OOP-factored synthesis**: The world model is structured as a set of typed object classes, each with a `respond(action, context)` method encoding how that object type reacts to actions. This reduces the synthesis search space from monolithic to factored and encapsulates learned behaviors as reusable, transferable units.

2. **Continuous epistemic model for exploration**: WorldCoder partitions knowledge into known/unknown. We replace this binary partition with continuous confidence scores that track observation count, context diversity, and effect consistency per (object_type, action) cell. This drives smarter exploration that actively seeks diverse contexts rather than merely covering the action space.

3. **Object-centric transition functions**: Instead of a single `transition(state, action) -> state` function, each object type owns its transition logic. Objects know how they respond to actions, how they interact with other objects, and under what spatial/relational preconditions effects trigger. The global transition emerges from composing per-object responses.

---

## 2. Background: WorldCoder

WorldCoder synthesizes world models as Python programs with two subroutines:
- `transition_function(state, action) -> next_state`
- `reward_function(context)(state, action, next_state) -> (reward, done)`

States are object-oriented MDPs: sets of objects with typed attributes (name, position, etc.). The synthesis loop (called "REx") operates as follows:

1. Sample transitions from a replay buffer.
2. Prompt GPT-4 to generate a candidate Python program.
3. Verify against two constraints:
   - **Data fitting (phi_1)**: predictions must match all observed transitions.
   - **Optimism under uncertainty (phi_2)**: the model must admit a plan leading to reward (drives exploration before reward is observed).
4. If verification fails, backprompt the LLM with counterexample transitions.
5. A bandit algorithm balances exploiting promising programs vs. exploring new ones.

**Evaluated domains:**
- **Sokoban**: Box-pushing puzzle. Good performance up to 4 boxes, planning bottleneck at 5+.
- **MiniGrid**: Gridworld suite for language-conditioned RL. Tests transfer and zero-shot instruction following.
- **AlfWorld**: Text-based household robotics. Uses MCTS as the planner.

**Key limitations:**
- Monolithic program structure limits scalability.
- Restricted to fully-observable, deterministic, symbolic environments.
- No mechanism for distinguishing "no effect" from "effect blocked by context."
- No transfer of learned transition logic across domains.
- Planning quality degrades on harder instances (5+ box Sokoban).

**Follow-up work:** PoE-World (Ellis group, 2025) replaces the monolithic program with a product of programmatic experts -- multiple smaller programs composed together. Our OOP factoring is a complementary approach that achieves compositionality through object-oriented class hierarchy rather than expert mixture.

---

## 3. Proposed System

### 3.1 Object-Centric OOP Factoring

The synthesized world model is not a single function but a collection of typed classes:

```python
class Wall(GameObject):
    def respond(self, action, context):
        pass  # Walls never move

class Player(GameObject):
    def respond(self, action, context):
        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        new_pos = (self.x + dx, self.y + dy)
        if not context.blocked(new_pos):
            self.x, self.y = new_pos

class Box(GameObject):
    def respond(self, action, context):
        # Only responds when player pushes into it
        if context.player_adjacent(self, action):
            push_pos = self.pushed_position(action)
            if not context.blocked(push_pos):
                self.x, self.y = push_pos
```

**Why this matters:**

- **Factored search space.** A monolithic transition function over K object types and N actions has a joint search space. With OOP factoring, the LLM synthesizes K independent `respond()` methods, each handling N actions for one type. The search space decomposes from `O(C^(K*N))` to `O(K * C^N)` where C is the per-cell code complexity.

- **Regression isolation.** Fixing the `Box.respond()` method cannot break `Wall.respond()`. In a monolithic function, any edit can introduce regressions anywhere.

- **Transfer via inheritance.** A `Wall` class learned in Sokoban can be inherited in a new domain. Only the novel object types need fresh synthesis. WorldCoder has no mechanism for this -- each domain starts from scratch.

- **Natural alignment with object-oriented MDP representation.** WorldCoder already represents states as sets of typed objects. Our factoring extends this to the transition function itself.

### 3.2 Continuous Epistemic Model

WorldCoder uses optimism under uncertainty (phi_2) to drive exploration: the model must admit a plan leading to reward. This is effective but coarse -- it does not distinguish between "never tested" and "tested once in one context."

We replace the binary known/unknown partition with a continuous confidence model over the (object_type, action) matrix:

**Per-cell tracking:**
- **Observation count**: How many times has this (type, action) pair been observed?
- **Context diversity**: How many distinct spatial/relational contexts have been observed? (e.g., player surrounded by walls vs. open space vs. adjacent to box)
- **Effect consistency**: What fraction of observations agree with the majority effect?
- **Conditional effect detection**: When the same (type, action) pair produces different outcomes in different contexts, flag it as conditional and record the distinguishing context features.

**Exploration priority** for cell (type_i, action_j):

```
priority(i, j) = w_1 / (1 + obs_count)           # prefer under-tested
              + w_2 / (1 + context_diversity)      # prefer low-diversity
              + w_3 * inconsistency_score           # prefer cells with conflicting observations
```

**Smart exploration**: When a cell has high inconsistency, the explorer actively seeks game states that would challenge the current understanding. "What spatial configuration might cause action X to behave differently?" This is a form of active hypothesis testing -- the agent doesn't just cover the action space, it designs experiments to discriminate between competing transition hypotheses.

**Convergence signal**: Confidence increases when new observations in novel contexts are consistent with existing observations. When consistency is high across diverse contexts, the cell is well-understood. When we see inconsistency, there's an unidentified precondition -- which is itself valuable information.

### 3.3 Object-Centric Transitions with Relational Preconditions

WorldCoder's monolithic transition function handles all interactions in one place. Our object-centric design distributes transition logic to the objects themselves, but with an important extension: **relational preconditions**.

Many game mechanics are relational:
- A box only moves when the **player pushes** it (proximity + action)
- A switch only activates when an object is **on top of** it
- A key only works when the player is **adjacent to** the door

These preconditions depend on the spatial relationship between two or more objects. They cannot be discovered by testing objects in isolation ("What does ACTION4 do to the box?"). The right question is "What happens to the box when the player executes ACTION4 while adjacent?"

Our `respond()` method receives a `context` parameter that provides relational queries:

```python
def respond(self, action, context):
    # context.player_adjacent(self, action) - is the player pushing toward me?
    # context.objects_at(position) - what's at a given position?
    # context.nearest(type) - where is the nearest object of a type?
    # context.selected - is this object currently selected?
```

The system learns these relational preconditions dynamically through exploration. When a (type, action) cell shows inconsistent effects, the explorer hypothesizes that a relational precondition explains the inconsistency and designs experiments to test it.

---

## 4. Architecture

```
                              Environment (Sokoban / MiniGrid / AlfWorld / ARC-AGI-3)
                                       |
                                  observation
                                       |
                                       v
                         +---------------------------+
                         |    Perception Pipeline    |
                         |   (domain-specific)       |
                         +---------------------------+
                         | Symbolic: direct state     |
                         | Grid: connected components |
                         | Text: NLP parse            |
                         +---------------------------+
                                       |
                                  WorldState (typed objects with attributes)
                                       |
                      +----------------+----------------+
                      |                                 |
                      v                                 v
         +------------------------+       +------------------------+
         |    Epistemic State     |       |    Exploration Engine   |
         | (continuous confidence)|       |  (active hypothesis    |
         +------------------------+       |   testing)             |
         | Per-cell:              |       +------------------------+
         |   obs_count            |       | Priority-driven        |
         |   context_diversity    |       | Context-seeking        |
         |   effect_consistency   |       | Relational probing     |
         |   conditional_flags    |       +------------------------+
         +------------------------+                |
                      |                            v
                      +----------+----------+------+
                                 |
                                 v
                    +------------------------+
                    |   Synthesis Pipeline    |
                    |   (LLM-driven CEGIS)   |
                    +------------------------+
                    | Per-type class synthesis |
                    | Reflexion loop           |
                    | Test against replay buf  |
                    | Persistent workspace     |
                    +------------------------+
                                 |
                                 v
                    +------------------------+
                    |   OOP World Model       |
                    +------------------------+
                    | Wall.respond()          |
                    | Player.respond()        |
                    | Box.respond()           |
                    | ...per domain type...   |
                    +------------------------+
```

### 4.1 Perception Pipeline

Perception is domain-specific and deterministic:

| Domain | Perception | State Representation |
|--------|-----------|---------------------|
| **Sokoban** | Symbolic state provided directly | Objects with (type, position) |
| **MiniGrid** | Symbolic grid provided directly | Objects with (type, position, direction, state) |
| **AlfWorld** | Text observation parsed to object set | Objects with (type, location, attributes) |
| **ARC-AGI-3** | Color-based connected component detection on 64x64 RGB grid | Sprites with (type_id, position, color, bounding_box) |

For symbolic domains (Sokoban, MiniGrid), perception is trivial -- the environment provides structured state. For ARC-AGI-3, we use a deterministic fragment detection + sprite registry + co-movement tracking pipeline (validated at 100% detection accuracy on test games). This decision is empirically justified: when the LLM was asked to synthesize perception alongside transitions, it spent all synthesis iterations debugging perception and achieved 0% transition accuracy.

### 4.2 Synthesis Pipeline

The CEGIS loop synthesizes one class at a time:

1. **Select target type**: Pick the object type with the lowest synthesis confidence (derived from epistemic state).
2. **Prepare context**: Extract all transitions involving this type from the replay buffer. Include the current class code (if any) and the signatures of other types' classes (for relational context).
3. **Prompt LLM**: Generate or edit the `respond()` method for this type.
4. **Verify**: Run the full world model (all types' `respond()` methods composed) against the replay buffer.
5. **If verification fails**: Provide counterexample transitions (predicted vs. actual state for failing cases) and re-prompt.
6. **Iterate**: Repeat until all transitions pass or synthesis budget exhausted.

The key difference from WorldCoder's synthesis: the LLM reasons about one type at a time, with the other types' behaviors held fixed. This dramatically reduces the cognitive load per synthesis step.

**Persistent workspace**: Code, replay buffer, reflections, and accuracy history persist on disk across synthesis calls. When new observations arrive, the LLM sees its previous code and edits incrementally.

### 4.3 Exploration Engine

Exploration is organized in phases:

| Phase | Purpose | Entry Condition |
|-------|---------|-----------------|
| **DISCOVER** | Identify object types present in the domain | First observation |
| **IDENTIFY_AGENT** | Determine which object the agent controls | At least one type detected |
| **SURVEY** | Test each action on each reachable object type once | Agent identified |
| **TARGETED** | Priority-driven testing based on epistemic state | Initial survey complete |
| **RELATIONAL** | Probe interactions between object pairs | Inconsistencies detected |

The TARGETED and RELATIONAL phases are where our system diverges most from WorldCoder. Instead of relying on optimism-driven planning to explore, we explicitly:

1. Compute exploration priority per (type, action) cell.
2. Navigate to a target of the highest-priority type.
3. Execute the action.
4. Observe the result and update the epistemic state.
5. When inconsistency is detected, enter RELATIONAL mode: systematically vary the spatial context (e.g., test the same action with and without an adjacent object) to identify preconditions.

---

## 5. Evaluation

### 5.1 Benchmarks

We evaluate on WorldCoder's three benchmarks plus ARC-AGI-3:

| Benchmark | Domain Type | State | Actions | Key Challenge |
|-----------|------------|-------|---------|---------------|
| **Sokoban** | Gridworld, box-pushing | Symbolic, fully observable | 4 directional | Push chains, deadlocks |
| **MiniGrid** | Gridworld suite | Symbolic, partially observable (agent view) | 7 (move, turn, pickup, drop, toggle, done) | Transfer across task variants, language conditioning |
| **AlfWorld** | Text-based household | Text descriptions | Natural language commands | Large object/action space, compositional tasks |
| **ARC-AGI-3** | Pixel gridworld | 64x64 RGB frames | 5-6 anonymous actions | Raw perception, unknown action semantics, diverse game mechanics |

Sokoban, MiniGrid, and AlfWorld allow direct comparison with WorldCoder. ARC-AGI-3 tests the system's ability to handle raw pixel observations and fully anonymous actions -- a setting WorldCoder has not been evaluated on.

### 5.2 Metrics

**Primary metric: Transition accuracy.** The percentage of observed transitions correctly predicted by the synthesized world model:

```
transition_accuracy = |{(s, a, s') in D : T_hat(s, a) = s'}| / |D|
```

This is evaluated at the object-state level: predicted positions and attributes vs. actual positions and attributes for all tracked objects.

**Secondary metrics:**

- **Sample efficiency**: Number of environment interactions required to reach X% transition accuracy. Direct comparison with WorldCoder's ~50 interaction figure.
- **Synthesis efficiency**: Number of LLM tokens consumed to reach X% accuracy. WorldCoder uses ~400k tokens; our factored approach should require fewer because each synthesis step is smaller.
- **Transfer score**: When trained on domain A, what transition accuracy does the model achieve on domain B without additional synthesis? Measured by inheriting learned classes and testing on novel domains.
- **Perception accuracy** (ARC-AGI-3 only): Percentage of ground-truth objects correctly detected and typed.

We deliberately exclude game score / task completion from our primary metrics. Our system synthesizes world models, not policies. A perfect world model enables perfect planning, but planning quality is a function of the planner, not the model.

### 5.3 Baselines

- **WorldCoder** (monolithic CEGIS): Direct comparison on all four benchmarks.
- **PoE-World** (product of experts): Comparison on scalability and compositionality.
- **Monolithic synthesis** (our system with OOP factoring disabled): Ablation showing the value of factored synthesis.
- **Binary epistemic model** (our system with continuous confidence replaced by KT/KF/UK): Ablation showing the value of continuous confidence.
- **Random exploration** (our system with priority-driven exploration replaced by uniform random): Ablation showing the value of smart exploration.

### 5.4 Hypotheses

1. **OOP factoring reduces synthesis cost.** On domains with 3+ object types, factored synthesis should reach equivalent accuracy with fewer LLM tokens than monolithic synthesis.
2. **Continuous epistemic model improves exploration efficiency.** Context-diversity-aware exploration should reach equivalent accuracy with fewer environment interactions than WorldCoder's optimism-driven exploration.
3. **Object-centric transitions enable transfer.** Classes learned in one Sokoban variant should transfer to new variants, reducing synthesis cost. WorldCoder cannot transfer.
4. **The system handles raw pixel observations.** On ARC-AGI-3, deterministic perception + OOP synthesis should achieve >80% transition accuracy, demonstrating that the approach extends beyond symbolic-state domains.

---

## 6. Contributions

1. **OOP-factored program synthesis for world models.** Transition logic is distributed across typed object classes rather than concentrated in a monolithic function. This reduces synthesis search space, isolates regressions, and enables transfer via inheritance.

2. **Continuous epistemic model replacing binary known/unknown.** Per-cell tracking of observation count, context diversity, and effect consistency drives exploration that actively seeks diverse contexts and identifies conditional preconditions.

3. **Object-centric transition functions with relational preconditions.** Each object type owns its transition logic and receives relational context (adjacency, selection, co-location) enabling the system to learn interaction effects that depend on spatial relationships between objects.

4. **Evaluation on four benchmarks spanning symbolic and pixel-observation domains.** Direct comparison with WorldCoder on Sokoban, MiniGrid, and AlfWorld, plus a novel evaluation on ARC-AGI-3 demonstrating generalization to raw pixel observations with anonymous actions.

---

## 7. Relation to Prior Work

| System | Representation | Synthesis | Exploration | Transfer |
|--------|---------------|-----------|-------------|----------|
| **WorldCoder** | Monolithic Python | CEGIS + bandit | Optimism (phi_2) | None |
| **PoE-World** | Product of experts | Compositional | Optimism | Partial (expert reuse) |
| **PDDL learners** | STRIPS schemas | Constraint-based | Coverage-driven | Schema reuse |
| **Ours** | OOP class hierarchy | Per-type CEGIS + reflexion | Continuous epistemic + active probing | Inheritance |

**WorldCoder**: Our direct predecessor. We preserve the core insight (LLM-synthesized Python as world model) but restructure the output from monolithic to factored, add continuous epistemic tracking, and make transitions object-centric.

**PoE-World** (2025): Addresses WorldCoder's monolithicity via product of experts. Complementary to our approach -- PoE-World composes experts via mixture, we compose via OOP class hierarchy. OOP provides stronger encapsulation and more natural transfer (inheritance vs. expert selection).

**PDDL Action Model Learning** (Cresswell et al., Arora et al., Juba et al.): Learns STRIPS-style action schemas from observed transitions. Our `respond()` methods are operationally equivalent to grounded STRIPS operators but encoded as executable Python, allowing arbitrary computational logic that PDDL cannot express.

**Reflexion** (Shinn et al., NeurIPS 2023): Verbal self-reflection for LLM agents. Our synthesis loop uses the same generate-test-reflect pattern but with structured per-object error signals rather than verbal reflection.

**DreamCoder** (Ellis et al., PLDI 2021): Bootstrapping program synthesis with library learning. Our OOP type hierarchy serves an analogous role: learned classes form a reusable library that transfers across domains.

**Object-Centric Learning** (Slot Attention, MONet, etc.): Neural approaches to discovering object representations from pixels. For grid environments we use deterministic connected-component detection instead, gaining determinism at the cost of generality.

---

## 8. Open Problems

### 8.1 Relational Precondition Discovery

How do we systematically identify which relational features (proximity, color, selection state, motion direction) serve as preconditions for conditional effects? The space of possible relational predicates is large. We propose starting with a small set of domain-general predicates (adjacent, on-top-of, same-color, selected) and expanding via LLM hypothesis generation when inconsistencies cannot be explained by the existing predicate set.

### 8.2 Update Ordering

When multiple objects respond to the same action, the order of execution matters. Player moves, then box is pushed, then goal checks. Getting this ordering wrong produces cascading prediction errors. WorldCoder avoids this by computing the full transition atomically. Our object-centric approach must explicitly model execution order, which is an additional failure mode.

### 8.3 Partial Observability

MiniGrid provides partial observations (the agent sees a limited field of view). AlfWorld provides text descriptions that omit unvisited rooms. ARC-AGI-3 may have hidden state. How should the epistemic model handle observations that are known to be incomplete? The current design assumes full observability of the rendered state.

### 8.4 Non-Deterministic Environments

All four benchmarks are primarily deterministic. Extending to stochastic transitions (e.g., random enemy movement) would require the world model to output distributions over next states rather than point predictions. The OOP structure naturally accommodates this (each `respond()` can return a distribution), but verification becomes harder.

### 8.5 Scaling the Type Hierarchy

Domains with many object types (AlfWorld has dozens of household object categories) may make per-type synthesis expensive. Grouping types into abstract superclasses (e.g., `Container`, `Movable`, `Tool`) could amortize synthesis cost, but the grouping itself must be learned.

---

## 9. Implementation Plan

### Phase 1: Core Infrastructure
- Object-centric world model representation (typed classes with `respond()`)
- Continuous epistemic state tracker
- Replay buffer and test runner
- Synthesis workspace with persistent disk state

### Phase 2: Sokoban Evaluation
- Implement Sokoban perception (trivial -- symbolic state)
- Implement exploration engine with phased exploration
- Run CEGIS synthesis loop on Sokoban 1-5 boxes
- Compare against WorldCoder numbers (sample efficiency, accuracy)

### Phase 3: MiniGrid + AlfWorld
- Extend perception for MiniGrid symbolic state and AlfWorld text
- Test transfer: train on one MiniGrid variant, evaluate on another
- Evaluate on AlfWorld with MCTS planner

### Phase 4: ARC-AGI-3
- Integrate deterministic perception pipeline (fragment detection, sprite registry, co-movement tracking, object tracking)
- Handle anonymous actions and pixel-level observations
- Evaluate on ARC-AGI-3 game suite

### Phase 5: Ablations and Paper
- Monolithic vs. OOP factoring ablation
- Continuous vs. binary epistemic model ablation
- Random vs. priority-driven exploration ablation
- Transfer experiments
- Write-up

---

## 10. References

- Tang, H., Key, D., & Ellis, K. (2024). WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment. *NeurIPS 2024*.
- Ellis, K. et al. (2025). PoE-World: Compositional World Models via Product of Programmatic Experts. *arXiv 2505.10819*.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
- Ellis, K. et al. (2021). DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. *PLDI 2021*.
- Cresswell, S., McCluskey, T., & West, M. (2013). Acquiring Planning Domain Models Using LOCM. *Knowledge Engineering Review*.
- Greff, K. et al. (2019). Multi-Object Representation Learning with Iterative Variational Inference. *ICML 2019*.
