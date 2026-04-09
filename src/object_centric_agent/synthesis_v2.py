"""
Reflexion-based program synthesis for object transition rules.

The LLM does NOT write perception or rendering code. Instead:
- Perception is handled by the deterministic FrameParser pipeline
- The LLM writes ONLY transition rules: how each sprite type's state
  changes in response to each action
- Verification compares predicted object states to actual observed states

The synthesized code is a Python module with one function per sprite type:

    def respond_player(obj, action_id, world):
        if action_id == 4:  # right
            obj['col'] += 3
            # check wall collision...

    def respond_wall(obj, action_id, world):
        pass  # static

    def transition(world_state, action_id):
        for obj in world_state['objects']:
            respond_fn = RESPOND_MAP[obj['type_name']]
            respond_fn(obj, action_id, world_state)
        return world_state

This is much simpler than synthesizing perceive()+respond()+render() — the
LLM only reasons about state transitions, not pixel detection.

Verification: for each transition in the replay buffer, we:
1. Parse the before_frame into a WorldState (deterministic)
2. Run the synthesized transition rules
3. Compare predicted object positions to the actual after_frame WorldState
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI

from .state.object_state import WorldState, SpriteInstance


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Transition:
    """A single observed (before, action, after) triple."""
    before_frame: np.ndarray
    action_id: int
    after_frame: np.ndarray
    timestep: int


@dataclass
class ObjectStateDiff:
    """Diff between predicted and actual state for one object."""
    track_id: int
    type_name: str
    predicted_position: tuple[int, int] | None
    actual_position: tuple[int, int] | None
    position_match: bool
    predicted_exists: bool
    actual_exists: bool
    details: str = ""


@dataclass
class TransitionTestResult:
    """Result of testing one transition."""
    timestep: int
    action_id: int
    passed: bool
    object_diffs: list[ObjectStateDiff] = field(default_factory=list)
    error: str | None = None


@dataclass
class TestReport:
    """Full evaluation against replay buffer."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: list[TransitionTestResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def is_perfect(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.total > 0

    def format_failures(self, max_failures: int = 5) -> str:
        failures = [r for r in self.results if not r.passed][:max_failures]
        if not failures:
            return "All tests passed."

        parts = [f"FAILED {self.failed + self.errors}/{self.total} transitions:"]
        for f in failures:
            parts.append(f"\n  Transition {f.timestep} (ACTION{f.action_id}):")
            if f.error:
                parts.append(f"    RUNTIME ERROR: {f.error}")
            else:
                for d in f.object_diffs:
                    if not d.position_match:
                        parts.append(
                            f"    {d.type_name} (track {d.track_id}): "
                            f"predicted pos={d.predicted_position}, "
                            f"actual pos={d.actual_position}"
                        )
                    if d.details:
                        parts.append(f"      {d.details}")

        remaining = (self.failed + self.errors) - len(failures)
        if remaining > 0:
            parts.append(f"\n  ... and {remaining} more failures")

        return "\n".join(parts)


@dataclass
class ReflexionAttempt:
    iteration: int
    code: str | None
    test_report: TestReport | None
    reflection: str | None
    duration_ms: float = 0


# =============================================================================
# World state serialization (for the LLM)
# =============================================================================

def world_state_to_dict(ws: WorldState) -> dict:
    """Convert a WorldState to a simple dict the synthesized code can operate on."""
    objects = []
    for s in ws.sprites:
        objects.append({
            "track_id": s.track_id,
            "type_id": s.type_id,
            "type_name": s.type_name,
            "row": s.position[0],
            "col": s.position[1],
            "center_row": s.center[0],
            "center_col": s.center[1],
            "colors": [list(c) for c in s.colors],
            "width": s.bbox[3] - s.bbox[1] + 1 if s.fragments else 0,
            "height": s.bbox[2] - s.bbox[0] + 1 if s.fragments else 0,
            "exists": True,
        })
    return {
        "objects": objects,
        "background_color": list(ws.background_color),
        "frame_height": ws.frame.shape[0],
        "frame_width": ws.frame.shape[1],
    }


def render_from_state(ws_dict: dict, original_frame: np.ndarray) -> np.ndarray:
    """Render a frame from the world state dict (simple pixel stamping)."""
    h, w = ws_dict["frame_height"], ws_dict["frame_width"]
    bg = ws_dict["background_color"]
    frame = np.full((h, w, 3), bg, dtype=np.uint8)

    # Re-stamp objects from the original frame's fragment data
    # This is a simplified render — we just need positions to match
    # The actual pixel data comes from the original frame
    return frame  # placeholder — actual render handled by comparison


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_rules(
    transition_fn: Any,
    replay_buffer: list[Transition],
    frame_parser: Any,
) -> TestReport:
    """
    Test synthesized transition rules against the replay buffer.

    For each transition:
    1. Parse before_frame → WorldState (deterministic)
    2. Convert to dict, run transition_fn
    3. Parse after_frame → actual WorldState (deterministic)
    4. Compare predicted vs actual object positions
    """
    report = TestReport(total=len(replay_buffer))

    for trans in replay_buffer:
        try:
            # Parse both frames
            before_ws = frame_parser.parse(
                trans.before_frame, frame_index=trans.timestep * 2,
            )
            after_ws = frame_parser.parse(
                trans.after_frame, frame_index=trans.timestep * 2 + 1,
                action_id=trans.action_id,
            )

            # Convert before state to dict and run transition rules
            state_dict = world_state_to_dict(before_ws)
            try:
                transition_fn(state_dict, trans.action_id)
            except Exception as e:
                report.errors += 1
                report.results.append(TransitionTestResult(
                    timestep=trans.timestep, action_id=trans.action_id,
                    passed=False, error=f"transition_fn: {type(e).__name__}: {e}",
                ))
                continue

            # Compare predicted positions to actual
            predicted_objects = {o["track_id"]: o for o in state_dict["objects"]}
            actual_objects = {s.track_id: s for s in after_ws.sprites}

            all_tracks = set(predicted_objects.keys()) | set(actual_objects.keys())
            diffs = []
            all_match = True

            for tid in all_tracks:
                pred = predicted_objects.get(tid)
                actual = actual_objects.get(tid)

                pred_pos = (pred["row"], pred["col"]) if pred and pred.get("exists", True) else None
                actual_pos = actual.position if actual else None

                pos_match = pred_pos == actual_pos

                if not pos_match:
                    all_match = False
                    type_name = (
                        pred["type_name"] if pred
                        else actual.type_name if actual
                        else "unknown"
                    )
                    diffs.append(ObjectStateDiff(
                        track_id=tid,
                        type_name=type_name,
                        predicted_position=pred_pos,
                        actual_position=actual_pos,
                        position_match=False,
                        predicted_exists=pred is not None and pred.get("exists", True),
                        actual_exists=actual is not None,
                    ))

            if all_match:
                report.passed += 1
            else:
                report.failed += 1

            report.results.append(TransitionTestResult(
                timestep=trans.timestep, action_id=trans.action_id,
                passed=all_match, object_diffs=diffs,
            ))

        except Exception as e:
            report.errors += 1
            report.results.append(TransitionTestResult(
                timestep=trans.timestep, action_id=trans.action_id,
                passed=False, error=f"{type(e).__name__}: {e}",
            ))

    return report


# =============================================================================
# Code execution
# =============================================================================

def execute_rules_code(code: str) -> Any:
    """
    Execute synthesized transition rules code.
    Returns the `transition` function, or None on failure.
    """
    namespace = {
        "np": np,
        "copy": __import__("copy"),
    }

    try:
        exec(code, namespace)
    except Exception as e:
        print(f"[ReflexionSynth] Code execution failed: {e}")
        traceback.print_exc()
        return None

    transition_fn = namespace.get("transition")
    if transition_fn is None:
        print("[ReflexionSynth] No `transition` function found in code")
        return None

    return transition_fn


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT_GENERATE = """You are an expert Python programmer writing transition rules for a grid-based game.

You are given:
- A list of detected object types (sprite types) with their colors and sizes
- Observed transition effects: what happened to each object type for each action
- Sample world states showing object positions before and after actions

Your job: write a `transition(world_state, action_id)` function that MUTATES the
world_state dict to predict the next state after an action.

The world_state dict has this structure:
{
    "objects": [
        {
            "track_id": 0,
            "type_id": 1,
            "type_name": "player",
            "row": 30,        # top-left row of bounding box
            "col": 15,        # top-left col of bounding box
            "center_row": 31.5,
            "center_col": 16.5,
            "colors": [[0, 116, 217]],
            "width": 3,
            "height": 3,
            "exists": True
        },
        ...
    ],
    "background_color": [0, 0, 0],
    "frame_height": 64,
    "frame_width": 64
}

Your `transition(world_state, action_id)` function should:
1. Find relevant objects by type_name or type_id
2. Apply movement/state changes based on action_id
3. Handle collisions (check if destination is occupied by blocking objects)
4. Handle conditional effects (action may have different results depending on context)
5. Mutate the objects in-place (update row, col, exists, etc.)

You do NOT need to:
- Parse frames or detect objects (already done)
- Render frames (already done)
- Define classes or complex structures

IMPORTANT:
- Objects are dicts, not class instances. Access with obj["row"], obj["col"], etc.
- The function must be named exactly `transition`
- Mutate `world_state["objects"]` in-place
- Helper functions are fine
- Available imports: np (numpy), copy

Output ONLY valid Python code. No markdown."""


SYSTEM_PROMPT_REFLECT = """You are debugging transition rules for a grid-based game.

You will be given:
1. The transition rules code
2. A test report showing which transitions failed
3. For each failure: which objects had wrong predicted positions vs actual positions

Diagnose WHY the rules are wrong. Be specific:
- Which object type's rules are incorrect?
- What case does the code miss (collision? boundary? conditional behavior)?
- What should the correct behavior be based on the test data?

Do NOT write code. Write a diagnosis and plan.

Format:
DIAGNOSIS: [what is wrong]
PLAN: [specific fixes needed]"""


# =============================================================================
# Synthesizer
# =============================================================================

class ReflexionSynthesizer:
    """
    Reflexion-based synthesizer for transition rules.

    Loops until 100% accuracy on ALL transitions, not an arbitrary cap.
    Carries forward best code between synthesize() calls.
    Only exits when perfect OR truly stuck (no improvement for N iterations).
    """

    # If accuracy doesn't improve for this many consecutive iterations, give up
    STUCK_PATIENCE: int = 4

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_iterations: int = 15,  # hard safety cap, not the normal exit
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.attempts: list[ReflexionAttempt] = []
        self.total_synthesis_count: int = 0

        # Persistent across synthesize() calls: carry forward best code
        self._best_code: str | None = None
        self._best_accuracy: float = 0.0

    def synthesize(
        self,
        replay_buffer: list[Transition],
        structured_analysis: str,
        frame_parser: Any = None,
        frame_shape: tuple[int, ...] | None = None,
        run_logger: Any = None,
    ) -> tuple[Any, str | None, TestReport | None]:
        """
        Reflexion loop: runs until 100% accuracy or stuck.

        If we already have code from a previous call, we start by
        re-evaluating it against the (possibly larger) replay buffer.
        If it's already perfect, return immediately. Otherwise, reflexion
        loop from the existing code.
        """
        self.attempts = []

        # Start from best code from previous call (if any)
        prev_code = self._best_code
        prev_report: TestReport | None = None
        prev_reflection: str | None = None

        best_fn = None
        best_code = self._best_code
        best_report = None
        best_accuracy = 0.0

        # If we have prior code, re-evaluate against current replay buffer
        if prev_code is not None:
            transition_fn = execute_rules_code(prev_code)
            if transition_fn is not None and frame_parser is not None:
                report = evaluate_rules(transition_fn, replay_buffer, frame_parser)
                print(
                    f"[ReflexionSynth] Re-eval prior code: "
                    f"{report.accuracy:.0%} ({report.passed}/{report.total})"
                )
                if report.is_perfect:
                    print("[ReflexionSynth] Prior code still perfect!")
                    self._best_code = prev_code
                    self._best_accuracy = 1.0
                    return transition_fn, prev_code, report

                best_fn = transition_fn
                best_code = prev_code
                best_report = report
                best_accuracy = report.accuracy
                prev_report = report

        # Track stuck detection: if accuracy doesn't improve, count up
        iterations_without_improvement = 0

        for iteration in range(self.max_iterations):
            t0 = time.time()

            # Generate
            if iteration == 0 and prev_code is None:
                # Fresh start: no prior code
                code = self._generate_initial(structured_analysis, frame_shape)
            else:
                # Reflexion: use prior code + report + reflection
                code = self._generate_with_reflection(
                    structured_analysis,
                    prev_code or best_code,
                    prev_report or best_report,
                    prev_reflection,
                    frame_shape,
                )

            duration_ms = (time.time() - t0) * 1000

            if code is None:
                self.attempts.append(ReflexionAttempt(
                    iteration=iteration, code=None,
                    test_report=None, reflection=None,
                    duration_ms=duration_ms,
                ))
                iterations_without_improvement += 1
                if iterations_without_improvement >= self.STUCK_PATIENCE:
                    print(f"[ReflexionSynth] Stuck — {self.STUCK_PATIENCE} iterations with no code generated")
                    break
                continue

            self.total_synthesis_count += 1

            # Execute
            transition_fn = execute_rules_code(code)
            if transition_fn is None:
                report = TestReport(total=len(replay_buffer), errors=len(replay_buffer))
                self.attempts.append(ReflexionAttempt(
                    iteration=iteration, code=code,
                    test_report=report, reflection=None,
                    duration_ms=duration_ms,
                ))
                prev_code = code
                prev_report = report
                prev_reflection = "The code failed to execute. Check for syntax errors."
                iterations_without_improvement += 1
                if iterations_without_improvement >= self.STUCK_PATIENCE:
                    print(f"[ReflexionSynth] Stuck — code keeps failing to execute")
                    break
                continue

            # Evaluate
            if frame_parser is not None:
                report = evaluate_rules(transition_fn, replay_buffer, frame_parser)
            else:
                report = self._evaluate_pixel_level(transition_fn, replay_buffer)

            print(
                f"[ReflexionSynth] Iteration {iteration}: "
                f"{report.accuracy:.0%} ({report.passed}/{report.total})"
            )

            # Track improvement
            if report.accuracy > best_accuracy:
                best_accuracy = report.accuracy
                best_fn = transition_fn
                best_code = code
                best_report = report
                iterations_without_improvement = 0  # reset stuck counter
            else:
                iterations_without_improvement += 1

            # === SUCCESS: 100% accuracy ===
            if report.is_perfect:
                print(f"[ReflexionSynth] PERFECT after {iteration + 1} iteration(s)!")
                self.attempts.append(ReflexionAttempt(
                    iteration=iteration, code=code,
                    test_report=report, reflection=None,
                    duration_ms=duration_ms,
                ))
                self._best_code = code
                self._best_accuracy = 1.0
                return transition_fn, code, report

            # === STUCK: no improvement for N iterations ===
            if iterations_without_improvement >= self.STUCK_PATIENCE:
                print(
                    f"[ReflexionSynth] Stuck at {best_accuracy:.0%} — "
                    f"no improvement for {self.STUCK_PATIENCE} iterations"
                )
                self.attempts.append(ReflexionAttempt(
                    iteration=iteration, code=code,
                    test_report=report, reflection=None,
                    duration_ms=duration_ms,
                ))
                break

            # Self-reflect
            t1 = time.time()
            reflection = self._self_reflect(code, report)
            duration_ms += (time.time() - t1) * 1000

            print(f"[ReflexionSynth] Reflection: {reflection[:150]}...")

            self.attempts.append(ReflexionAttempt(
                iteration=iteration, code=code,
                test_report=report, reflection=reflection,
                duration_ms=duration_ms,
            ))

            prev_code = code
            prev_report = report
            prev_reflection = reflection

        # Save best for next call
        if best_code is not None:
            self._best_code = best_code
            self._best_accuracy = best_accuracy

        return best_fn, best_code, best_report

    def _evaluate_pixel_level(self, transition_fn, replay_buffer):
        """Fallback pixel-level evaluation when no frame_parser is available."""
        report = TestReport(total=len(replay_buffer))
        for trans in replay_buffer:
            report.errors += 1
            report.results.append(TransitionTestResult(
                timestep=trans.timestep, action_id=trans.action_id,
                passed=False, error="No frame_parser for evaluation",
            ))
        return report

    # =========================================================================
    # LLM calls
    # =========================================================================

    def _generate_initial(self, structured_analysis, frame_shape):
        user_prompt = self._build_prompt(structured_analysis, frame_shape)
        return self._call_llm(SYSTEM_PROMPT_GENERATE, user_prompt)

    def _generate_with_reflection(
        self, structured_analysis, prev_code, prev_report, prev_reflection,
        frame_shape,
    ):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
        ]

        initial_prompt = self._build_prompt(structured_analysis, frame_shape)
        messages.append({"role": "user", "content": initial_prompt})

        if prev_code:
            messages.append({"role": "assistant", "content": prev_code})

        feedback_parts = []
        if prev_report:
            feedback_parts.append(
                f"[test results]:\n{prev_report.format_failures()}"
            )
        if prev_reflection:
            feedback_parts.append(
                f"\n[self-reflection]:\n{prev_reflection}"
            )
        feedback_parts.append(
            "\n[improved implementation]:\n"
            "Write the complete corrected `transition(world_state, action_id)` function. "
            "Address the issues in the reflection."
        )
        messages.append({"role": "user", "content": "\n".join(feedback_parts)})

        return self._call_llm_messages(messages)

    def _self_reflect(self, code, report):
        user_prompt = (
            f"[transition rules code]:\n```python\n{code}\n```\n\n"
            f"[test results]:\n{report.format_failures()}\n\n"
            f"[self-reflection]:"
        )
        result = self._call_llm(SYSTEM_PROMPT_REFLECT, user_prompt)
        return result or "Unable to generate reflection."

    def _build_prompt(self, structured_analysis, frame_shape):
        parts = []
        if frame_shape:
            parts.append(f"Frame shape: {frame_shape}")
        parts.append(structured_analysis)
        parts.append(
            "\nWrite a `transition(world_state, action_id)` function that "
            "mutates the world_state dict to predict the next state."
        )
        parts.append(
            "Objects are dicts with keys: track_id, type_id, type_name, "
            "row, col, center_row, center_col, colors, width, height, exists."
        )
        return "\n".join(parts)

    def _call_llm(self, system, user):
        return self._call_llm_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    def _call_llm_messages(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
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
            print(f"[ReflexionSynth] LLM call failed: {e}")
            return None
