"""
OOP Program Synthesis: LLM generates polymorphic object/action classes.

This is the core of what makes the OOP agent different from the monolithic one.
The synthesis prompt forces the LLM to decompose the world model into:

    1. GameObject subclasses — one per visual entity type you identify.
    2. Action subclasses — one per anonymous action.
    3. A SynthesizedDomain — ties perception, action mapping, and rendering.

Architecture (from formalism.tex):
    The compilation function f_compile receives:
        - K^A_t: Action Knowledge Base (NL definitions from VLM exploration)
        - K^M_t: Environment Knowledge Base (NL description from VLM exploration)
        - Σ*_API: Target API specification (base classes, method signatures)
        - (θ_prev, E_prev): Previous code + verification errors (for refinement)

    The VLM exploration agent produces NL-JSON knowledge (declarative).
    This synthesis "compiler" converts it to executable Python (procedural).
    The structured game description IS the AgentState that the exploration
    engine accumulates — serialized and passed directly to synthesis.

The verification accuracy is the FEEDBACK SIGNAL — Python execution replaces
asking an LLM whether the code is correct.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from .world_model import (
    GameObject,
    Action,
    World,
    Domain,
    PixelDiff,
    compute_diff,
    find_unique_colors,
    find_color_regions,
    region_bbox,
    most_common_color,
)
from src.shared.frame_utils import extract_grid, ARC_PALETTE as _ARC_PALETTE


@dataclass
class Transition:
    """A recorded (frame, action, next_frame) transition."""
    before_frame: np.ndarray
    action_id: int
    after_frame: np.ndarray
    timestep: int


@dataclass
class VerificationResult:
    correct: int = 0
    incorrect: int = 0
    errors: list[str] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.correct + self.incorrect
        return self.correct / total if total > 0 else 0.0

    @property
    def is_perfect(self) -> bool:
        return self.incorrect == 0 and self.correct > 0


def verify_domain(domain: Domain, replay_buffer: list[Transition]) -> VerificationResult:
    """Verify a synthesized domain against every transition in the replay buffer."""
    result = VerificationResult()

    for trans in replay_buffer:
        try:
            predicted = domain.transition(trans.before_frame, trans.action_id)
            predicted = np.asarray(predicted, dtype=np.uint8)

            if np.array_equal(predicted, trans.after_frame):
                result.correct += 1
            else:
                result.incorrect += 1

                if predicted.ndim == 3:
                    diff_mask = np.any(predicted != trans.after_frame, axis=-1)
                else:
                    diff_mask = predicted != trans.after_frame

                diff_pixels = int(np.sum(diff_mask))
                diff_positions = list(zip(*np.where(diff_mask)))[:10]

                result.mismatches.append({
                    "timestep": trans.timestep,
                    "action_id": trans.action_id,
                    "diff_pixel_count": diff_pixels,
                    "sample_diffs": [
                        {
                            "pos": (int(r), int(c)),
                            "predicted": predicted[r, c].tolist() if predicted.ndim == 3 else int(predicted[r, c]),
                            "actual": trans.after_frame[r, c].tolist() if trans.after_frame.ndim == 3 else int(trans.after_frame[r, c]),
                        }
                        for r, c in diff_positions
                    ],
                })
        except Exception as e:
            result.incorrect += 1
            result.errors.append(f"Timestep {trans.timestep}: {type(e).__name__}: {e}")

    return result


# =============================================================================
# NL Knowledge → Structured Game Description (from VLM exploration)
# =============================================================================

def _rgb_to_palette_index(rgb: tuple[int, ...]) -> int:
    """Map an RGB tuple to the nearest ARC palette index (0-15)."""
    for idx, color in enumerate(_ARC_PALETTE):
        if tuple(int(c) for c in color) == rgb[:3]:
            return idx
    return -1


def build_game_description_from_state(
    agent_state: Any,
    replay_buffer: list[Transition],
    frame_shape: tuple[int, ...] | None = None,
) -> str:
    """
    Build the NL game description from the VLM exploration agent's accumulated
    knowledge (AgentState).

    This implements the compilation function's input:
        f_compile(K^A_t, K^M_t, ...) → θ
    where K^A_t = action_knowledge and K^M_t = environment.

    The output is a formatted string (not JSON) that reads like the learning
    agent's PromptKnowledgeFormatter output — because that's exactly what the
    VLM produced and the synthesis LLM should consume.
    """
    parts = []

    # Frame metadata
    if frame_shape:
        parts.append(f"Frame shape: {frame_shape} (dtype uint8, RGB)")

    # ================================================================
    # K^M_t: Environment Knowledge (from VLM environment analysis)
    # ================================================================
    if agent_state and hasattr(agent_state, 'environment'):
        env = agent_state.environment
        parts.append("\n== ENVIRONMENT KNOWLEDGE (from VLM exploration) ==")

        if env.background_color:
            parts.append(f"BACKGROUND: {env.background_color}")

        if env.has_border:
            border_info = f"BORDER: Yes ({env.border_color or 'unknown color'})"
            if env.border_description:
                border_info += f"\n  Description: {env.border_description}"
            parts.append(border_info)

        if env.spatial_structure:
            parts.append(f"SPATIAL STRUCTURE:\n  {env.spatial_structure}")

        if env.domain_description:
            parts.append(f"DOMAIN DESCRIPTION:\n  {env.domain_description}")

        if env.breakthroughs:
            breakthrough_lines = "\n".join(f"  - {b}" for b in env.breakthroughs)
            parts.append(f"KEY BREAKTHROUGHS:\n{breakthrough_lines}")

        if env.movement_constraints:
            constraint_lines = "\n".join(f"  - {c}" for c in env.movement_constraints)
            parts.append(f"MOVEMENT CONSTRAINTS:\n{constraint_lines}")

        if env.internal_walls:
            wall_lines = "\n".join(f"  - {w}" for w in env.internal_walls)
            parts.append(f"INTERNAL WALLS/OBSTACLES:\n{wall_lines}")

        if env.identified_objects:
            obj_lines = []
            for obj in env.identified_objects:
                name = obj.get('name', 'Unknown')
                color = obj.get('color', '?')
                shape = obj.get('shape', '?')
                role = obj.get('role_hypothesis', 'Unknown')
                evidence = obj.get('evidence_for_role', '')
                obj_line = f"  - {name}: {color} {shape} (Role: {role})"
                if evidence:
                    obj_line += f"\n      Evidence: {evidence}"
                obj_lines.append(obj_line)
            parts.append(f"IDENTIFIED OBJECTS:\n" + "\n".join(obj_lines))

    # ================================================================
    # K^A_t: Action Knowledge (from VLM action analysis)
    # ================================================================
    if agent_state and hasattr(agent_state, 'action_knowledge'):
        parts.append("\n== ACTION KNOWLEDGE (from VLM exploration) ==")
        for action_id_str, knowledge in sorted(agent_state.action_knowledge.items()):
            if knowledge.is_verified:
                status = "VERIFIED"
            elif knowledge.is_exhausted:
                status = "EXHAUSTED (no effect)"
            elif len(knowledge.observations) == 0:
                status = "NO OBSERVATIONS"
            else:
                status = f"{knowledge.verification_attempts} observations"

            definition = knowledge.current_definition or "unknown"
            parts.append(f"{action_id_str}: [{status}] {definition}")

            # Include key observation context for actions with effects
            effect_observations = [
                obs for obs in knowledge.observations if obs.had_effect
            ]
            for obs in effect_observations[-3:]:  # Last 3 observations with effects
                parts.append(f"  - Context: {obs.context_description}")
                parts.append(f"    Result: {obs.llm_interpretation}")
                if obs.object_changes:
                    parts.append(f"    Objects: {obs.object_changes}")

    # ================================================================
    # Transition data (pixel-level diffs for grounding)
    # ================================================================
    parts.append("\n== SAMPLE TRANSITIONS (pixel-level verification data) ==")
    for trans in replay_buffer[:8]:
        diff = compute_diff(trans.before_frame, trans.after_frame)
        parts.append(
            f"\nTimestep {trans.timestep}, ACTION{trans.action_id}: "
            f"{diff.count} pixels changed"
        )
        if diff.bbox and diff.count > 0 and diff.count < 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            for pos, bc, ac in zip(diff.positions[:15], diff.before_colors[:15], diff.after_colors[:15]):
                bc_idx = _rgb_to_palette_index(bc) if len(bc) >= 3 else bc
                ac_idx = _rgb_to_palette_index(ac) if len(ac) >= 3 else ac
                parts.append(f"  ({pos[0]},{pos[1]}): color {bc_idx} -> color {ac_idx}")
            if diff.count > 15:
                parts.append(f"  ... and {diff.count - 15} more")
        elif diff.count >= 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            parts.append(f"  (large change, {diff.count} pixels)")

    return "\n".join(parts)


def _frame_to_palette_ascii(frame: np.ndarray) -> str:
    """Convert a frame to compact palette-indexed ASCII."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        # Already palette-indexed
        lines = []
        for row in frame:
            chars = []
            for v in row:
                v = int(v)
                chars.append(str(v) if v < 10 else chr(ord('A') + v - 10) if v < 16 else '?')
            lines.append(''.join(chars))
        return '\n'.join(lines)

    # RGB -> palette
    h, w = frame.shape[:2]
    lines = []
    for y in range(h):
        chars = []
        for x in range(w):
            idx = _rgb_to_palette_index(tuple(int(c) for c in frame[y, x]))
            if idx < 0:
                chars.append('?')
            elif idx < 10:
                chars.append(str(idx))
            else:
                chars.append(chr(ord('A') + idx - 10))
        lines.append(''.join(chars))
    return '\n'.join(lines)


# =============================================================================
# OOP synthesis prompt (adapted from learning_agent style)
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert Python programmer. You are a "compiler" that converts natural language \
game descriptions into executable Python world models.

You receive:
1. ENVIRONMENT KNOWLEDGE — what the game world looks like (background, borders, objects, spatial layout)
2. ACTION KNOWLEDGE — what each action does (verified NL definitions from systematic testing)
3. SAMPLE TRANSITIONS — pixel-level before/after data to ground your implementation

Your job is to faithfully IMPLEMENT the described game mechanics as executable Python.

You MUST produce code that follows this OOP structure:

1. Define **GameObject subclasses** for each visual entity type described.
   Each must implement:
   - __init__(self, obj_id, **properties): store whatever state this object type needs
   - respond_to_action(self, action, world): update this object's state when an action occurs
   - render(self, frame): draw this object onto the frame (a (64,64,3) uint8 numpy array)

2. Define **Action subclasses** for each action described.
   Each must implement:
   - apply(self, world): call respond_to_action on affected objects in causal order

3. Define a **SynthesizedDomain** class (subclassing Domain) with:
   - perceive(self, frame) -> World: parse the raw frame into typed objects
   - get_action(self, action_id) -> Action: map action ID to Action instance

The Domain.transition method is already implemented: it calls perceive -> get_action -> apply -> render.

Available base classes (already imported):
    GameObject(obj_id, **properties)  # base with .obj_id, .type_name, arbitrary attrs
    Action(action_id)                 # base with .action_id
    World(frame, objects)             # .objects, .get_objects_of_type(cls), .get_by_id(id),
                                      # .add_object(obj), .remove_object(obj), .render()
    Domain                            # base with abstract perceive/get_action, concrete transition

Available utilities (already imported):
    compute_diff(before, after) -> PixelDiff  # .count, .positions, .bbox, .before_colors, .after_colors
    find_unique_colors(frame) -> list[tuple[int,...]]
    find_color_regions(frame, color, connectivity=4) -> list[set[tuple[int,int]]]
    region_bbox(pixels) -> (min_row, min_col, max_row, max_col)
    most_common_color(frame) -> tuple[int,...]

Available imports: np (numpy), copy

IMPORTANT GUIDELINES:
- Trust the NL descriptions — they come from systematic VLM exploration with verification.
- Each object type is a SEPARATE class with focused logic.
- Actions call respond_to_action on objects — they don't manipulate pixel arrays directly.
- The perceive method must detect objects by scanning pixel colors in the frame.
- The render method on each object draws it back onto the frame.
- Games use cell-based rendering (each logical cell = NxN pixels). Detect cell size from the data.
- Keep each class focused and small. CORRECTNESS over complexity.
- Frame is (64,64,3) uint8 RGB. Use np.array_equal for color comparisons.

CRITICAL: Your implementation must exactly reproduce the transitions in the sample data.
When you verify mentally, check: does perceive find the right objects? Does the
action move them correctly? Does render put them back in the right place?

Output ONLY valid Python code (no markdown, no explanation). The code will be exec'd.
"""


def build_synthesis_prompt(
    game_description: str,
    replay_buffer: list[Transition],
    previous_code: str | None = None,
    verification_errors: VerificationResult | None = None,
) -> str:
    """
    Build the user prompt for synthesis.

    The game_description is a pre-formatted NL string from the VLM exploration
    agent's accumulated knowledge (environment + action definitions + transitions).
    This is the K^A_t + K^M_t from the formalism.
    """
    parts = [game_description]

    # CEGIS refinement: show previous code and errors
    if previous_code and verification_errors:
        parts.append("\n== PREVIOUS CODE (has errors — fix them) ==")
        parts.append(previous_code)
        parts.append("\n== VERIFICATION ERRORS ==")
        parts.append(f"Accuracy: {verification_errors.accuracy:.1%}")
        parts.append(f"Correct: {verification_errors.correct}, Incorrect: {verification_errors.incorrect}")
        for err in verification_errors.errors[:5]:
            parts.append(f"  ERROR: {err}")
        for mm in verification_errors.mismatches[:5]:
            parts.append(f"  MISMATCH at timestep {mm['timestep']}, ACTION{mm['action_id']}:")
            parts.append(f"    {mm['diff_pixel_count']} pixels differ")
            for sd in mm["sample_diffs"][:5]:
                parts.append(f"    pos {sd['pos']}: predicted {sd['predicted']}, actual {sd['actual']}")
        parts.append("\nFix the errors. Keep the OOP structure. Fix what's wrong, keep what works.")
    else:
        parts.append("\nSynthesize a complete OOP Domain subclass named SynthesizedDomain.")
        parts.append("Implement the game description above as faithfully as possible.")
        parts.append("Remember: separate GameObject subclass for each entity type, separate Action subclass for each action.")

    return "\n".join(parts)


def execute_synthesized_code(code: str) -> Domain | None:
    from . import world_model

    namespace = {
        "GameObject": world_model.GameObject,
        "Action": world_model.Action,
        "World": world_model.World,
        "Domain": world_model.Domain,
        "compute_diff": world_model.compute_diff,
        "find_unique_colors": world_model.find_unique_colors,
        "find_color_regions": world_model.find_color_regions,
        "region_bbox": world_model.region_bbox,
        "most_common_color": world_model.most_common_color,
        "PixelDiff": world_model.PixelDiff,
        "np": np,
        "copy": __import__("copy"),
        "ABC": __import__("abc").ABC,
        "abstractmethod": __import__("abc").abstractmethod,
        "Optional": Optional,
        "Any": Any,
    }

    try:
        exec(code, namespace)
    except Exception as e:
        print(f"[OOPSynthesis] Code execution failed: {e}")
        traceback.print_exc()
        return None

    domain_cls = namespace.get("SynthesizedDomain")
    if domain_cls is None:
        print("[OOPSynthesis] No SynthesizedDomain class found")
        return None

    try:
        return domain_cls()
    except Exception as e:
        print(f"[OOPSynthesis] Instantiation failed: {e}")
        traceback.print_exc()
        return None


class DomainSynthesizer:
    """
    CEGIS loop for OOP domain synthesis.

    Receives a structured game description from the exploration engine's
    accumulated knowledge, and iteratively synthesizes + verifies code
    until it correctly reproduces all observed transitions.
    """

    def __init__(
        self,
        base_url: str = "https://clewdr.wavycats.com/code/v1",
        api_key: str = "LkY56yDjUVYkfL9BPm7VLTpMs7kM6gbXPMp6PV2QysqcAvpr8PAdyjPYYbbgTwgH",
        model: str = "claude-sonnet-4-6-thinking",
        max_refinements: int = 5,
        # Legacy support
        provider: str | None = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_refinements = max_refinements
        self.last_code: str | None = None
        self.synthesis_count = 0

    def synthesize(
        self,
        replay_buffer: list[Transition],
        frame_analysis: str,
        action_observations: dict[int, list[str]],
        frame_shape: tuple[int, ...] | None = None,
        run_logger: Any = None,
        agent_state: Any = None,
    ) -> tuple[Domain | None, str | None, VerificationResult | None]:
        """
        Run the CEGIS synthesis loop.

        Implements the Self-Refine iteration from the formalism:
            θ^(0) = f_compile(K_t, Σ*_API, ⊥)           (initial generation)
            E^(k) = Evaluate(T̂_θ^(k), τ_t)              (feedback)
            θ^(k+1) = f_compile(K_t, Σ*_API, (θ^(k), E^(k)))  (refinement)

        Args:
            agent_state: The AgentState from the exploration engine, containing
                the VLM's accumulated NL knowledge (K^A_t + K^M_t).
        """
        # Build NL game description from VLM's accumulated knowledge
        if agent_state:
            game_description = build_game_description_from_state(
                agent_state=agent_state,
                replay_buffer=replay_buffer,
                frame_shape=frame_shape,
            )
        else:
            # Legacy fallback: use raw frame analysis + observations
            parts = []
            if frame_shape:
                parts.append(f"Frame shape: {frame_shape} (dtype uint8, RGB)")
            parts.append(f"\n== FRAME ANALYSIS ==\n{frame_analysis}")
            parts.append("\n== ACTION OBSERVATIONS ==")
            for action_id, observations in sorted(action_observations.items()):
                parts.append(f"\nACTION{action_id}:")
                for obs in observations[-5:]:
                    parts.append(f"  - {obs}")
            parts.append("\n== SAMPLE TRANSITIONS ==")
            for trans in replay_buffer[:8]:
                diff = compute_diff(trans.before_frame, trans.after_frame)
                parts.append(
                    f"\nTimestep {trans.timestep}, ACTION{trans.action_id}: "
                    f"{diff.count} pixels changed"
                )
            game_description = "\n".join(parts)

        previous_code = None
        previous_errors = None
        best_domain = None
        best_code = None
        best_result = None
        best_accuracy = 0.0

        for attempt in range(1 + self.max_refinements):
            prompt = build_synthesis_prompt(
                game_description=game_description,
                replay_buffer=replay_buffer,
                previous_code=previous_code,
                verification_errors=previous_errors,
            )

            t0 = __import__("time").time()
            code = self._call_llm(prompt)
            duration_ms = (__import__("time").time() - t0) * 1000

            if code is None:
                if run_logger is not None:
                    run_logger.log_synthesis_attempt(
                        attempt_number=self.synthesis_count + 1,
                        prompt=prompt,
                        synthesized_code=None,
                        verification_result=None,
                        llm_messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        duration_ms=duration_ms,
                        refinement_round=attempt,
                    )
                continue

            self.synthesis_count += 1
            self.last_code = code

            domain = execute_synthesized_code(code)
            if domain is None:
                previous_code = code
                previous_errors = VerificationResult(
                    errors=["Code failed to execute or instantiate"]
                )
                if run_logger is not None:
                    run_logger.log_synthesis_attempt(
                        attempt_number=self.synthesis_count,
                        prompt=prompt,
                        synthesized_code=code,
                        verification_result=previous_errors,
                        llm_messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        duration_ms=duration_ms,
                        refinement_round=attempt,
                    )
                continue

            result = verify_domain(domain, replay_buffer)

            if run_logger is not None:
                run_logger.log_synthesis_attempt(
                    attempt_number=self.synthesis_count,
                    prompt=prompt,
                    synthesized_code=code,
                    verification_result=result,
                    llm_messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    duration_ms=duration_ms,
                    refinement_round=attempt,
                )

            # Track best result
            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                best_domain = domain
                best_code = code
                best_result = result

            if result.is_perfect:
                print(f"[OOPSynthesis] Perfect domain after {attempt + 1} attempt(s)")
                return domain, code, result

            print(
                f"[OOPSynthesis] Attempt {attempt + 1}: "
                f"{result.accuracy:.1%} ({result.correct}/{result.correct + result.incorrect})"
            )

            previous_code = code
            previous_errors = result

        # Return best result even if not perfect
        if best_domain:
            return best_domain, best_code, best_result

        return None, None, None

    def _call_llm(self, user_prompt: str) -> str | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=65000,
            )

            content = response.choices[0].message.content
            if not content:
                return None

            code = content.strip()
            if "```python" in code:
                code = code.split("```python", 1)[1]
                code = code.split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1]
                code = code.split("```", 1)[0]

            return code.strip()

        except Exception as e:
            print(f"[OOPSynthesis] LLM call failed: {e}")
            return None


@dataclass
class CounterexampleDiagnosis:
    is_stuck: bool
    failed_transitions: list[Transition]
    synthesis_attempts: int
    description: str


def diagnose_persistent_errors(
    replay_buffer: list[Transition],
    domain: Domain,
    synthesis_attempts: int,
    max_attempts: int = 3,
) -> CounterexampleDiagnosis:
    result = verify_domain(domain, replay_buffer)

    failed = []
    for mm in result.mismatches:
        for trans in replay_buffer:
            if trans.timestep == mm["timestep"]:
                failed.append(trans)
                break

    is_stuck = synthesis_attempts >= max_attempts and len(failed) > 0

    return CounterexampleDiagnosis(
        is_stuck=is_stuck,
        failed_transitions=failed,
        synthesis_attempts=synthesis_attempts,
        description=f"{'Stuck' if is_stuck else 'Improving'}: {len(failed)} failures after {synthesis_attempts} attempts",
    )
