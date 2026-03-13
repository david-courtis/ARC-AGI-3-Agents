"""
Program synthesis: LLM-driven generation and refinement of world models.

This module handles:
1. Prompting the LLM to synthesize WorldModel subclasses
2. Verifying synthesized code against the replay buffer
3. Counterexample-guided refinement (CEGIS loop)

The synthesized model operates on raw RGB frames (64x64x3 uint8 numpy arrays).
No domain assumptions are imposed — the LLM figures out the structure.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from .world_model import (
    WorldModel,
    PixelDiff,
    compute_diff,
    find_unique_colors,
    find_color_regions,
    region_bbox,
    most_common_color,
)


# =============================================================================
# Replay buffer and verification
# =============================================================================


@dataclass
class Transition:
    """A recorded (frame, action, next_frame) transition."""

    before_frame: np.ndarray  # (H, W, 3) uint8
    action_id: int
    after_frame: np.ndarray   # (H, W, 3) uint8
    timestep: int


@dataclass
class VerificationResult:
    """Result of verifying a model against the replay buffer."""

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


def verify_model(model: WorldModel, replay_buffer: list[Transition]) -> VerificationResult:
    """
    Verify a synthesized model against every transition in the replay buffer.
    """
    result = VerificationResult()

    for trans in replay_buffer:
        try:
            predicted = model.predict(trans.before_frame, trans.action_id)
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
# Prompt construction
# =============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer synthesizing world models for grid-based games.

You observe a game through 64x64 RGB pixel frames. Actions are anonymous integers (ACTION1 through ACTION5).
You know NOTHING about the game's rules, objects, or structure — you must infer everything from the transitions.

You must produce a complete Python class named `SynthesizedModel` that subclasses `WorldModel`.
The class must implement:
    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray
        - frame is a (64, 64, 3) uint8 numpy array (RGB pixels)
        - action_id is an integer (1-5)
        - returns a (64, 64, 3) uint8 numpy array (the predicted next frame)

You also have these utility functions available (already imported):
    compute_diff(before, after) -> PixelDiff  # pixel-level diff with .count, .positions, .bbox, .before_colors, .after_colors
    find_unique_colors(frame) -> list[tuple[int,...]]  # all unique RGB colors
    find_color_regions(frame, color, connectivity=4) -> list[set[tuple[int,int]]]  # connected regions of a color
    region_bbox(pixels) -> (min_row, min_col, max_row, max_col)  # bounding box of pixel set
    most_common_color(frame) -> tuple[int,...]  # likely background color

And these imports are available: np (numpy), copy

IMPORTANT GUIDELINES:
- The frames are raw RGB pixels. Look at the transition data carefully to understand the visual structure.
- Games may use cell-based rendering (each logical cell = NxN pixels). Detect this from the data.
- Some pixels may be decorative borders, indicators, or overlays that change with actions.
- Focus on correctly predicting the pixels that actually change. Start simple.
- If most pixels stay the same, model only what changes.
- Keep the code clean and correct. Correctness over complexity.

Your output must be ONLY valid Python code (no markdown, no explanation outside comments).
The code will be exec'd, so it must be syntactically valid and self-contained
(aside from the imports above which are provided).
"""


def build_synthesis_prompt(
    replay_buffer: list[Transition],
    frame_analysis: str,
    action_observations: dict[int, list[str]],
    previous_code: str | None = None,
    verification_errors: VerificationResult | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> str:
    """Build the user prompt for model synthesis."""
    parts = []

    # Frame info
    if frame_shape:
        parts.append(f"Frame shape: {frame_shape} (dtype uint8, RGB)")

    # Frame analysis (colors, regions, structure)
    parts.append(f"\n== FRAME ANALYSIS ==\n{frame_analysis}")

    # Action observations
    parts.append("\n== ACTION OBSERVATIONS ==")
    for action_id, observations in sorted(action_observations.items()):
        parts.append(f"\nACTION{action_id}:")
        for obs in observations[-5:]:
            parts.append(f"  - {obs}")

    # Sample transitions
    parts.append("\n== SAMPLE TRANSITIONS ==")
    for trans in replay_buffer[:8]:
        diff = compute_diff(trans.before_frame, trans.after_frame)
        parts.append(
            f"\nTimestep {trans.timestep}, ACTION{trans.action_id}: "
            f"{diff.count} pixels changed"
        )
        if diff.bbox and diff.count > 0 and diff.count < 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            for i, (pos, bc, ac) in enumerate(
                zip(diff.positions[:15], diff.before_colors[:15], diff.after_colors[:15])
            ):
                parts.append(f"  ({pos[0]},{pos[1]}): {bc} -> {ac}")
            if diff.count > 15:
                parts.append(f"  ... and {diff.count - 15} more")
        elif diff.count >= 200:
            parts.append(f"  Changed region bbox: rows {diff.bbox[0]}-{diff.bbox[2]}, cols {diff.bbox[1]}-{diff.bbox[3]}")
            parts.append(f"  (large change, {diff.count} pixels)")

    # Previous code and errors (for refinement)
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

        parts.append("\nFix the errors. Keep what works, fix what doesn't.")
    else:
        parts.append("\nSynthesize a complete WorldModel subclass named SynthesizedModel.")

    return "\n".join(parts)


# =============================================================================
# Code execution sandbox
# =============================================================================


def execute_synthesized_code(code: str) -> WorldModel | None:
    """
    Execute synthesized Python code and extract the SynthesizedModel class.

    Returns None if execution fails.
    """
    from . import world_model

    namespace = {
        "WorldModel": world_model.WorldModel,
        "compute_diff": world_model.compute_diff,
        "find_unique_colors": world_model.find_unique_colors,
        "find_color_regions": world_model.find_color_regions,
        "region_bbox": world_model.region_bbox,
        "most_common_color": world_model.most_common_color,
        "PixelDiff": world_model.PixelDiff,
        "np": np,
        "copy": __import__("copy"),
        "Optional": Optional,
        "Any": Any,
    }

    try:
        exec(code, namespace)
    except Exception as e:
        print(f"[Synthesis] Code execution failed: {e}")
        traceback.print_exc()
        return None

    model_cls = namespace.get("SynthesizedModel")
    if model_cls is None:
        print("[Synthesis] No SynthesizedModel class found in synthesized code")
        return None

    try:
        model = model_cls()
        return model
    except Exception as e:
        print(f"[Synthesis] Failed to instantiate SynthesizedModel: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# Synthesizer: orchestrates the CEGIS loop
# =============================================================================


class ModelSynthesizer:
    """
    Orchestrates LLM-based synthesis and refinement of world models.

    CEGIS loop:
    1. Synthesize initial model code from observations
    2. Verify against replay buffer
    3. If errors, refine with counterexample feedback
    4. Repeat until correct or max iterations reached
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "google/gemini-2.5-flash",
        max_refinements: int = 3,
    ):
        import os
        if provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

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
    ) -> tuple[WorldModel | None, str | None, VerificationResult | None]:
        """
        Synthesize and verify a world model.

        Returns:
            Tuple of (model, code, verification_result).
            model is None if synthesis failed entirely.
        """
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

            code = self._call_llm(prompt)
            if code is None:
                continue

            self.synthesis_count += 1
            self.last_code = code

            model = execute_synthesized_code(code)
            if model is None:
                previous_code = code
                previous_errors = VerificationResult(
                    errors=["Code failed to execute or instantiate"]
                )
                continue

            result = verify_model(model, replay_buffer)

            if result.is_perfect:
                print(f"[Synthesis] Perfect model after {attempt + 1} attempt(s)")
                return model, code, result

            print(
                f"[Synthesis] Attempt {attempt + 1}: "
                f"{result.accuracy:.1%} accuracy "
                f"({result.correct}/{result.correct + result.incorrect})"
            )

            previous_code = code
            previous_errors = result

        # Return best effort
        if previous_code:
            model = execute_synthesized_code(previous_code)
            if model:
                result = verify_model(model, replay_buffer)
                return model, previous_code, result

        return None, None, None

    def _call_llm(self, user_prompt: str) -> str | None:
        """Call the LLM and extract Python code from the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=8000,
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
            print(f"[Synthesis] LLM call failed: {e}")
            return None


# =============================================================================
# Counterexample diagnosis
# =============================================================================


@dataclass
class CounterexampleDiagnosis:
    """Diagnosis of persistent verification failures."""

    is_stuck: bool
    failed_transitions: list[Transition]
    synthesis_attempts: int
    description: str


def diagnose_persistent_errors(
    replay_buffer: list[Transition],
    model: WorldModel,
    synthesis_attempts: int,
    max_attempts: int = 3,
) -> CounterexampleDiagnosis:
    """
    Determine if persistent errors indicate the model needs more data or
    a fundamentally different approach.

    Heuristic: if synthesis has been attempted max_attempts times and
    the same transitions keep failing, the model is stuck.
    """
    result = verify_model(model, replay_buffer)

    failed = []
    for mm in result.mismatches:
        for trans in replay_buffer:
            if trans.timestep == mm["timestep"]:
                failed.append(trans)
                break

    is_stuck = synthesis_attempts >= max_attempts and len(failed) > 0

    desc = (
        f"{'Stuck' if is_stuck else 'Improving'}: "
        f"{len(failed)} failures after {synthesis_attempts} synthesis attempts"
    )

    return CounterexampleDiagnosis(
        is_stuck=is_stuck,
        failed_transitions=failed,
        synthesis_attempts=synthesis_attempts,
        description=desc,
    )
