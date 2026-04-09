"""
Object-Centric Program Synthesis: objects contain all transition logic.

The key structural difference from the action-centric OOP agent:

Action-centric (oop_agent/): LLM writes Action.apply() methods that
orchestrate object updates. Action knows about object types.

Object-centric (this agent): LLM writes GameObject.respond() methods
that handle ALL actions. Objects know about action semantics. Actions
are just integer signals, not classes.

This changes what the synthesis prompt asks for:
- No Action subclasses needed
- Each GameObject subclass must handle all action IDs in its respond() method
- The Domain needs perceive() and optionally respond_order()
- The LLM writes T + 1 programs (T object types + perceive), not T + A + 1

This tests the hypothesis that the LLM produces more correct code when
each object type encapsulates all of its own behavior (respond to every
action) vs when each action type encapsulates its effect on all objects.

See docs/action-vs-object-centric.md for the full comparison.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from .world_model import (
    GameObject,
    World,
    Domain,
    PixelDiff,
    compute_diff,
    find_unique_colors,
    find_color_regions,
    region_bbox,
    most_common_color,
)


@dataclass
class Transition:
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
# Object-centric synthesis prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer synthesizing object-centric world models for grid-based games.

You observe a game through 64x64 RGB pixel frames. Actions are anonymous integers (ACTION1 through ACTION5).
You may be provided with pre-detected object types (sprite registry) and confirmed transition effects
(epistemic state with KT=Known True, KF=Known False, UK=Unknown). Use this structured information to
guide your implementation — it tells you which objects exist, what colors they are, and what each action
does to each object type. When this information is provided, build your classes to match the detected types.

You MUST produce code that follows this OBJECT-CENTRIC structure:

1. Define **GameObject subclasses** for each visual entity type you identify.
   Each must implement:
   - __init__(self, obj_id, **properties): store whatever state this object type needs
   - respond(self, action_id, world): handle ALL action IDs for this object type.
     This is the core method. The object decides what each action means for it.
     Different object types respond differently to the same action (polymorphism).
   - render(self, frame): draw this object onto the frame (a (64,64,3) uint8 numpy array)

2. Define a **SynthesizedDomain** class (subclassing Domain) with:
   - perceive(self, frame) -> World: parse the raw frame into typed objects
   - optionally respond_order(self, world, action_id) -> list: control which objects
     respond first (default: all objects in natural order)

There are NO Action classes. Actions are just integer IDs (1-5) passed to each
object's respond() method. Each object independently decides how to react.

The Domain.transition method is already implemented: it calls perceive, then
calls respond(action_id, world) on every object (in respond_order), then renders.

Available base classes (already imported):
    GameObject(obj_id, **properties)  # base with .obj_id, .type_name, arbitrary attrs
    World(frame, objects)             # .objects, .get_objects_of_type(cls), .get_by_id(id),
                                      # .add_object(obj), .remove_object(obj), .render()
    Domain                            # base with abstract perceive, concrete transition

Available utilities (already imported):
    compute_diff(before, after) -> PixelDiff
    find_unique_colors(frame) -> list[tuple[int,...]]
    find_color_regions(frame, color, connectivity=4) -> list[set[tuple[int,int]]]
    region_bbox(pixels) -> (min_row, min_col, max_row, max_col)
    most_common_color(frame) -> tuple[int,...]

Available imports: np (numpy), copy

IMPORTANT OBJECT-CENTRIC GUIDELINES:
- Each object type handles ALL actions in its respond() method. Use if/elif on action_id.
- There are NO Action classes. Do not create Action subclasses.
- Each object updates only itself. To check other objects, query the world.
- Cross-object interaction is handled by querying: "is there a Player adjacent to me?"
  NOT by having the Action move both the Player and the Block.
- Think from each object's perspective: "what happens to ME when action X occurs?"
- If an object type never changes, its respond() is a no-op (pass).
- respond_order matters when objects need to see each other's updated state.
  For example, if the Player moves first and then Blocks check if they were pushed,
  the Player should respond before Blocks.

Output ONLY valid Python code (no markdown). The code will be exec'd.
"""


def build_synthesis_prompt(
    replay_buffer: list[Transition],
    frame_analysis: str,
    action_observations: dict[int, list[str]],
    previous_code: str | None = None,
    verification_errors: VerificationResult | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> str:
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
        if diff.bbox and diff.count > 0 and diff.count < 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            for pos, bc, ac in zip(diff.positions[:15], diff.before_colors[:15], diff.after_colors[:15]):
                parts.append(f"  ({pos[0]},{pos[1]}): {bc} -> {ac}")
            if diff.count > 15:
                parts.append(f"  ... and {diff.count - 15} more")
        elif diff.count >= 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            parts.append(f"  (large change, {diff.count} pixels)")

    if previous_code and verification_errors:
        parts.append("\n== PREVIOUS CODE (has errors) ==")
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
        parts.append("\nFix the errors. Keep the object-centric structure (NO Action classes). Fix what's wrong, keep what works.")
    else:
        parts.append("\nSynthesize a complete object-centric Domain subclass named SynthesizedDomain.")
        parts.append("Remember: NO Action classes. Each GameObject.respond() handles all action IDs.")

    return "\n".join(parts)


def execute_synthesized_code(code: str) -> Domain | None:
    from . import world_model

    namespace = {
        "GameObject": world_model.GameObject,
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
        print(f"[ObjectCentricSynthesis] Code execution failed: {e}")
        traceback.print_exc()
        return None

    domain_cls = namespace.get("SynthesizedDomain")
    if domain_cls is None:
        print("[ObjectCentricSynthesis] No SynthesizedDomain class found")
        return None

    try:
        return domain_cls()
    except Exception as e:
        print(f"[ObjectCentricSynthesis] Instantiation failed: {e}")
        traceback.print_exc()
        return None


class DomainSynthesizer:
    def __init__(
        self,
        base_url: str = "https://clewdr.wavycats.com/code/v1",
        api_key: str = "LkY56yDjUVYkfL9BPm7VLTpMs7kM6gbXPMp6PV2QysqcAvpr8PAdyjPYYbbgTwgH",
        model: str = "claude-sonnet-4-6-thinking",
        max_refinements: int = 3,
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
    ) -> tuple[Domain | None, str | None, VerificationResult | None]:
        previous_code = None
        previous_errors = None

        for attempt in range(1 + self.max_refinements):
            prompt = build_synthesis_prompt(
                replay_buffer=replay_buffer,
                frame_analysis=frame_analysis,
                action_observations=action_observations,
                previous_code=previous_code,
                verification_errors=previous_errors,
                frame_shape=frame_shape,
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

            if result.is_perfect:
                print(f"[ObjectCentricSynthesis] Perfect domain after {attempt + 1} attempt(s)")
                return domain, code, result

            print(
                f"[ObjectCentricSynthesis] Attempt {attempt + 1}: "
                f"{result.accuracy:.1%} ({result.correct}/{result.correct + result.incorrect})"
            )

            previous_code = code
            previous_errors = result

        if previous_code:
            domain = execute_synthesized_code(previous_code)
            if domain:
                result = verify_domain(domain, replay_buffer)
                return domain, previous_code, result

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
            print(f"[ObjectCentricSynthesis] LLM call failed: {e}")
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
