# Speaker Notes: Learning Agent Presentation

Talking points for each slide. Details to expand on verbally.

---

## Slide: Title
- This is my work on co-evolving action and environment understanding
- For ARC-AGI-3, the interactive version coming in 2026
- The key word is "co-evolving" - we'll come back to this

---

## Slide: ARC-3: The Core Challenge
- ARC-3 requires discovering three things: Goal, Environment, Actions
- The diagram shows they're all interconnected with bidirectional arrows
- Key insight: **these cannot be learned in isolation**
- Unlike standard RL, there's no reward signal until you discover the goal
- ARC Prize 2024: state-of-art improved from 33% to 55.5%
- But ARC-3 is interactive, not static puzzles - entirely new challenge

---

## Slide: The Chicken-and-Egg Problem
- This slide explains WHY co-evolution is necessary
- "ACTION1 moves the red thing" - action references an object
- "The red thing is the player" - role inferred from action effect
- You can't learn one without the other
- Standard approaches try to learn sequentially - that fails here

---

## Slide: The Original Vision
- The five boxes show the iterative loop
- Observe → Act → Compute Diff → Ask LLM → Refine
- Dashed arrow shows it loops back
- First goal: understand the world, not solve the game
- Terminate after learning - goal discovery comes later
- This was the initial vision, let me show what didn't work first

---

## Slide: Dead End #1: CNN Binary Predictor
- Left side: the idea (train CNN to predict frame changes)
- Right side: the problem (binary output)
- The red arrow shows: what CNN learns ≠ what we need
- You can learn "ACTION1 usually changes the frame"
- But you learn **nothing** about WHAT it does or HOW
- Can't build a plan from binary signals
- Full implementation in `agents/templates/action_agent.py`

**Elaboration/Justification:**
- **Why train CNN at test-time?** The idea was to learn a lightweight predictor during gameplay that could identify which actions are "interesting" (cause changes). This would let us focus exploration on actions that actually do something.
- **Why is True/False insufficient?** To plan, you need to predict *outcomes*, not just *whether something happens*. Knowing "ACTION1 changes the frame" doesn't tell you if it moves the player, rotates an object, or triggers a trap. You can't chain actions into a plan without knowing their effects.
- **Why can't we extend this to predict more?** A CNN trained from scratch at test-time has no prior knowledge. It would need thousands of examples to learn even basic spatial relationships, and even then it only learns statistical correlations, not causal mechanics.

---

## Slide: Dead End #2: Direct VLM Prompting
- This shows what happens when you just throw a VLM at the problem
- We tested Gemini 3.0 Pro via OpenRouter
- Sent raw 64×64 integer grid as text
- Model made detailed observations - but they were **completely wrong**

**The actual run (show if asked):**
```
ACTION4 (Right): no change → "path to right was blocked"
ACTION2 (Down): no change → "movement blocked, trying different direction"
ACTION3 (Left): "slight change in Row 2" → "successfully navigated"
ACTION3 (Left): "continued successful movement"
```

**What went wrong:**
- **400K tokens in 6 actions**: ~70K tokens per turn for observation + action selection
  - *Why so many?* The model has to re-read the entire 64×64 grid each time, reason about it from scratch, and maintain conversational context. No compression, no learning.
- **Wrong position assumptions**: Model claimed "player is at row 62, col 9" - completely fabricated
  - *Why?* VLMs don't actually "see" pixel grids well. They process them as text/numbers without spatial grounding. The model confabulates plausible-sounding positions.
- **No state change detection**: Model claimed "successful movement" when the frame was identical
  - *Why?* Without explicit before/after comparison, the model can't tell if anything changed. It just generates plausible narratives.
- **Hallucinated game mechanics**: Model invented a "vertical corridor" and navigation strategy
  - *Why?* VLMs are trained to generate coherent text. When uncertain, they produce confident-sounding but unfounded claims.

**The key insight:** VLMs need **structured input** (object detection, before/after diff) and **verification** (check claims against actual changes) to be useful.

Full implementation in `agents/templates/llm_agents.py` (OpenRouterLLM class)

---

## Slide: Dead End #3: RL-Based Policy Learning
- Four boxes show the four problems
- Sample efficiency: RL needs 1000s-millions of interactions
- Compute cost: model updates are expensive at test-time
- Sparse rewards: no signal until goal is discovered
- Time budget: ARC-3 has limited time
- Even sample-efficient methods need too many interactions

**Elaboration/Justification:**
- **Sample efficiency (1000s of interactions)**: Even the most sample-efficient model-based RL methods (like MuZero, Dreamer) need hundreds of interactions to learn basic dynamics. Test-time budget is maybe 50-200 actions total.
- **Compute cost (model updates expensive)**: Each gradient update requires forward/backward passes through a neural network. At test-time, we can't afford the latency or compute for continuous training.
- **Sparse rewards (no signal until goal)**: RL learns from reward gradients. In ARC-3, you don't know the goal until you discover it - so there's no reward signal to learn from. This is a fundamental chicken-and-egg problem.
- **Time budget (ARC-3 is limited)**: Competition constraints give limited time per puzzle. Even if RL could eventually learn, it can't do so fast enough.
- **Why doesn't Value Equivalence help?** Value Equivalence (Grimm et al.) improves sample efficiency by only modeling value-relevant state features. But it still requires training a value function, which needs reward signal we don't have.

---

## Slide: Related Work: The Gap
- Value Equivalence (Grimm 2020, 2021): great for efficient RL, but still needs training
- Downward Refinement: hierarchical planning, but assumes operators are KNOWN
- Angelic Semantics: needs action hierarchy known a priori
- **Gap**: none solve "learn operators from scratch at test-time"
- Connection to spectral methods (Ng et al.): grouping by similarity

**Elaboration/Justification:**
- **Value Equivalence**: The principle says you only need to model state features that matter for value prediction. Great for efficiency, but you still need to *train* a value function, which requires reward signal.
- **Downward Refinement**: Classical hierarchical planning technique where abstract plans are refined into concrete actions. Assumes you already know what your operators (actions) do - exactly what we're trying to discover.
- **Angelic Semantics**: Uses optimistic/pessimistic bounds on abstract actions to prune search space. Requires knowing the action hierarchy ahead of time - again, we don't have this.
- **Why spectral clustering is related**: Both approaches group things by similarity. We group pixels by color adjacency to form objects. Spectral methods group data points by similarity in a graph. The connection is conceptual - grouping by discovered (not predefined) features.

---

## Slide: Key Insight: VLM as Reasoning Engine
- This is the paradigm shift slide
- Left box (red): Neural network approach problems
- Right box (green): VLM advantages
- Arrow shows the transition
- VLM has already learned about objects, space, causality
- We just need to prompt it correctly
- No backpropagation, no gradient updates during test

**Elaboration/Justification:**
- **Why "reasoning engine" not "learning system"?** We're not training the VLM - we're using its pre-trained knowledge. The VLM already understands objects, spatial relationships, and cause-effect. We just provide structured input and ask it to apply this knowledge.
- **Why does this work when Direct VLM Prompting failed?** The difference is *structure*. Direct prompting gives raw grids and asks for actions. Our approach gives:
  1. Explicit before/after comparison (so the VLM can see what changed)
  2. Object-level descriptions (so the VLM can reason about entities, not pixels)
  3. Previous observations (so the VLM has context for consistency checking)
  4. Structured output format (so we can verify and accumulate knowledge)
- **Trade-off**: This requires sufficient model capacity. Smaller models can't do the causal reasoning even with structured input.

---

## Slide: The Prior Knowledge Problem
- Two boxes comparing Gemini 2.5 Flash vs 3.0 Pro
- Surprising finding: bottleneck isn't game knowledge
- Both models have never seen ARC-3
- Difference is **reasoning capacity**
- 2.5 Flash: can describe what it sees, can't reason about cause/effect
- 3.0 Pro: "this moved because I pressed that button"
- Supports VideoGameBench findings: even 2.5 Pro only 0.48% completion

**Elaboration/Justification:**
- **Why isn't game knowledge the bottleneck?** Neither model has seen ARC-3 games before (knowledge cutoff). The difference is in *reasoning ability*, not *memorized facts*.
- **What reasoning does 2.5 Flash lack?**
  - Can describe: "There is a red square at position (10,15)"
  - Can't infer: "The red square moved because I pressed ACTION1, so ACTION1 probably means 'move up'"
  - Fails at: Linking observations across time, forming causal hypotheses, updating beliefs
- **What does 3.0 Pro have?**
  - Temporal reasoning: "Before vs after, the red thing moved"
  - Causal inference: "It moved when I pressed the button, so the button caused it"
  - Belief updating: "My previous hypothesis was wrong, here's a better one"
- **VideoGameBench validation**: Independent benchmark showed even Gemini 2.5 Pro only completed 0.48% of video games. Basic tasks like grid navigation failed. Our findings align with this.

---

## Slide: The Gestalt Gap in VLMs
- Visual shows: pixel grid → VLM sees "numbers" → we need "red square at (2,2)"
- VLMs process images holistically but struggle with pixel grids
- Three Gestalt principles VLMs lack:
  - Proximity: adjacent same-color = one object
  - Similarity: same color = same category
  - Common fate: move together = one object
- This is why we need explicit preprocessing

**Elaboration/Justification:**
- **Why can't VLMs see objects in pixel grids?** VLMs are trained on natural images where objects have texture, edges, semantic context. A 64×64 grid of integers has none of this. The VLM sees "a bunch of numbers" not "a red square."
- **Proximity (adjacent = same object)**: Humans naturally group adjacent same-colored pixels. VLMs don't - they'd need explicit training on this representation.
- **Similarity (same color = same category)**: Humans assume all red pixels might be related. VLMs might, but can't reliably distinguish "player red" from "wall red" without context.
- **Common fate (move together = one object)**: Humans track motion grouping automatically. VLMs need explicit before/after comparison to even detect motion.
- **Open question**: Could we fine-tune a VLM on pixel grids to learn these principles? Or is the representation gap too fundamental?

---

## Slide: Solution: Object Detection Layer
- Four boxes show the pipeline: Pixels → Connected Components → Properties → Output
- Uses scipy.ndimage for connected components
- 8-connectivity: diagonal pixels count as connected
- Properties: bounding box, center, area, shape, color
- is_rectangular: fills 90%+ of bounding box?
- Open question: can this be learned end-to-end?

**Elaboration/Justification:**
- **Why connected components?** Classic computer vision technique that groups adjacent pixels of the same value. Works reliably without training.
- **Why 8-connectivity?** Allows diagonal adjacency. A pixel is connected to its 8 neighbors, not just 4. This matches human perception of objects.
- **What properties matter?**
  - *Bounding box*: Where is the object?
  - *Center*: Reference point for motion tracking
  - *Area*: Size of the object
  - *is_rectangular*: Shape heuristic - most game objects are rectangular
- **Why this works**: Transforms ambiguous pixel data into semantic descriptions the VLM can reason about. "Red 3×3 block at (10,15)" is much easier to track than "pixels at (10,15), (10,16), (11,15)..."

---

## Slide: Three-Phase Architecture: Overview
- Four boxes: Phase 1, Phase 2, Phase 3, Execute
- Dashed arrow shows the loop
- Each has one question: "What happened?", "What is this?", "What to try?"
- Key point: each LLM call has ONE job
- Original vision was single call - didn't work well
- Separation allows independent refinement

**Elaboration/Justification:**
- **Why not a single LLM call?** We tried this first. Problems:
  - Conflicting objectives in one prompt (analyze + decide + update)
  - Context overload (all information dumped at once)
  - No separation of concerns (hard to debug which part failed)
- **Why three separate calls?**
  - *Action Analysis*: Focused on "what did this specific action do?" with before/after context
  - *Environment Analysis*: Focused on "what do I understand about the world?" with cumulative evidence
  - *Next Action*: Focused on "what should I explore?" with strategic priorities
- **Why does separation help?**
  - Each call has tailored context (only relevant information)
  - Each call has specialized prompt (focused instructions)
  - Environment can self-correct without being tied to action analysis
  - Easy to debug: if action definitions are wrong, check Phase 1; if environment model is wrong, check Phase 2

---

## Slide: Phase 1: Action Analysis
- Two columns: Input and Output
- Input: images, ASCII grids, object changes, previous observations
- Output: interpretation, had_effect, new_definition, is_consistent
- Key design: LLM decides whether to update the definition
- Most complex prompt of the three
- Images rendered at 8x scale for visibility

---

## Slide: Phase 2: Environment Analysis
- Left: what it discovers (background, walls, objects, constraints)
- Right: example role hypotheses
- **This is a separate dedicated LLM call**, not part of action analysis
- Can explicitly correct previous mistakes
- Runs after every action analysis (when state changed)
- Evidence-based: needs justification for claims

---

## Slide: Phase 3: Next Action Selection
- Priority order: unobserved > unverified
- The diagram shows setup sequences
- Example: player at wall → move right twice → now test move left
- Critical: don't repeat no-effect in same state
- Setup sequences use only VERIFIED actions
- Hard guards in agent.py prevent infinite loops

---

## Slide: Action Analysis Prompt (Key Points)
- Shortened version of the actual prompt
- Key points: NO prior knowledge, current understanding may be WRONG
- UI elements and move counters distinction is important
- Evidence-based: don't guess without evidence
- Full prompt is ~100 lines with all context

---

## Slide: Environment Analysis Prompt (Key Points)
- Sole focus: environment, not actions
- Critical emphasis: may be WRONG
- Seven specific things to discover
- "Be a detective" - active investigation mindset
- This is what enables self-correction

---

## Slide: Co-Evolution: How It Works
- Four iterations showing progressive understanding
- Iter 1: raw pixel observation, minimal understanding
- Iter 2: object-level thinking, role hypothesis forms
- Iter 3: constraint discovery from no-effect
- Iter 4: verified understanding, both sides reference each other
- **This is the core slide** - bidirectional building is the key innovation

**Elaboration/Justification:**
- **Iteration 1 - Raw observation**: "ACTION1 changed pixels at (10,15)" + "There's a shape there"
  - At this point, we don't know what ACTION1 does or what the shape is
  - Just recording that *something happened* at a *location*
- **Iteration 2 - Object emergence**: "ACTION1 moved that shape UP" + "Shape is probably the PLAYER"
  - Now we're thinking in terms of objects, not pixels
  - Role hypothesis: "things that respond to input are usually players"
  - Action definition starts referencing the object: "moves the shape"
- **Iteration 3 - Constraint discovery**: "ACTION1 tried to move up but hit something" + "Grey is a WALL"
  - No-effect actions are informative! They reveal constraints
  - Environment model grows: "grey things block movement"
  - Action definition becomes conditional: "moves up, blocked by grey"
- **Iteration 4 - Verified understanding**: "Moves player up, blocked by walls" + "Player=red, Walls=grey"
  - Definitions now reference each other bidirectionally
  - Action: "moves **player** up, blocked by **walls**"
  - Environment: "**player** is controlled by movement actions, **walls** block movement"
  - 3 consistent observations = VERIFIED

**Why is this "co-evolution"?**
- You can't define ACTION1 without knowing what a "player" is
- You can't identify the "player" without seeing what ACTION1 affects
- Both must develop *together* through iterative observation

---

## Slide: Verification Logic
- State diagram: Unverified → Verified (3 consistent) or Exhausted (8 no-effects)
- Key insight: expected no-ops count as "consistent"
- Hitting the same wall twice is "consistent with definition"
- Single effective action resets the no-effect counter
- Full logic in models.py, ActionKnowledge class

**Elaboration/Justification:**
- **Why 3 consecutive consistent observations?**
  - 1 observation could be coincidence
  - 2 could be pattern but not verified
  - 3 provides high confidence (especially across different contexts)
  - More than 3 wastes exploration budget
- **Why do expected no-ops count as "consistent"?**
  - Example: Definition says "ACTION1 moves player up, blocked by walls"
  - Player is against a wall, you press ACTION1, nothing happens
  - This is *expected behavior* - the definition predicted it correctly
  - Counting this as "inconsistent" would prevent verification of context-dependent actions
- **Why 8 consecutive no-effects = exhausted?**
  - If an action never does anything after 8 tries, it's probably not useful
  - Could be a broken/disabled action, or we're always in wrong context
  - Stop wasting exploration budget on it
- **Why does one effective action reset the counter?**
  - If ACTION1 has no effect 7 times but then works once, it's not broken
  - It's context-dependent - worth continuing to explore
  - Reset gives it another chance to be verified
- **Termination**: An action is "done" when:
  - `is_verified` (3 consistent) OR
  - `is_exhausted` (8 consecutive no-effects) OR
  - `verification_attempts >= 8` (maxed out attempts without either)

---

## Slide: Results: It Works!
- Left: checklist of demonstrated capabilities
- Right: example outputs from actual runs
- "Grey (5) represents walls" - explains past no-effects retroactively
- Be prepared to show actual logged output if asked
- Ready for goal discovery phase

---

## Slide: Comparison with Related Work
- Table shows we're unique: test-time + from scratch + semantic
- Value Equiv RL: not test-time, not semantic
- Hierarchical Planning: test-time, but not from scratch
- CNN Binary: test-time, but binary only, not semantic
- Semantic output enables goal reasoning later

---

## Slide: Current Scope & Next Steps
- Green boxes: Actions and Environment (done)
- Orange dashed boxes: Goal and Planning (future)
- Foundation is solid: semantic understanding complete
- Goal discovery: likely inferred from object roles
- Planning: use verified definitions as operators
- Goal discovery not yet implemented - next phase

---

## Slide: Open Research Questions
- Q1: Can object detection be learned end-to-end?
  - Current: hand-coded Gestalt grouping
  - Could fine-tuning help?
- Q2: What's the minimum reasoning capacity?
  - 2.5 Flash insufficient, 3.0 Pro sufficient
  - Systematic ablation needed
- Q3: Can we reduce LLM calls?
  - Currently 3 calls per action (~$1-5 per game)
  - Could batch or reduce env analysis frequency

---

## Slide: Key Takeaways
- Six numbered points to remember
- Co-evolution: the core insight
- Binary signals: why CNN failed
- RL: why too expensive for test-time
- VLMs: why this works (needs 3.0 Pro level)
- Gestalt: the preprocessing layer needed
- Works: ready for next phase

---

## Slide: Architecture Summary
- Simple diagram showing component relationships
- LearningAgent (agent.py) at top
- Three branches: LLM Agents, Knowledge, Vision
- Object Detection under Vision
- ~2000 lines of code total
- Clean separation of concerns

---

## Slide: Questions?
- Table of key files and their purposes
- All code in `agents/templates/learning_agent/`
- Be ready to show actual log output
- Discuss Gemini 2.5 vs 3.0 comparison
- Talk about next steps for goal discovery

---

## Backup Discussion Points

**If asked about ARC Prize 2024:**
- State of art improved from 33% to 55.5%
- Top approaches: test-time training, program synthesis
- OpenAI o3 reached 87.5% but at $3460/task
- ARC-3 is different: interactive, not static

**If asked about VideoGameBench:**
- Gemini 2.5 Pro: only 0.48% completion rate
- Models failed at basic tasks: dragging, 2D grid navigation
- Our approach addresses this with explicit object detection

**If asked about the CNN more:**
- Full implementation in `action_agent.py`
- Two-headed output: action logits + coordinate logits
- Trained with binary cross-entropy on `frame_changed`
- Can predict WHICH actions might change frame
- Cannot predict HOW - useless for mechanics

**If asked about test-time compute cost:**
- ~3 LLM calls per action
- Full run: 50-100 actions = 150-300 LLM calls
- Cost estimate: ~$1-5 per game
- Could optimize with batching or less frequent env analysis

---

## Key Sources

1. [ARC Prize 2024 Technical Report](https://arxiv.org/abs/2412.04604)
2. [Value Equivalence Principle](https://arxiv.org/pdf/2011.03506)
3. [Proper Value Equivalence](https://arxiv.org/abs/2106.10316)
4. [Downward Refinement](https://www.sciencedirect.com/science/article/abs/pii/0004370294900620)
5. [Angelic Semantics](https://people.eecs.berkeley.edu/~russell/papers/icaps07-hla.pdf)
6. [Spectral Clustering](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)
7. [VideoGameBench](https://opencv.org/blog/videogamebench/)
