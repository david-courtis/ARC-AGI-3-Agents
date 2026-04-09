"""
Persistent synthesis workspace on disk.

All synthesis artifacts live in a directory tied to the run:
    {run_dir}/synthesis/
    ├── transition_rules.py       # The code being synthesized (edited in-place)
    ├── test_runner.py            # Script that tests the code against replay buffer
    ├── context.txt               # Structured analysis for the LLM
    ├── replay_buffer.pkl         # Serialized transitions
    ├── reflections/              # Accumulated reflections (if using API backend)
    │   ├── 001.txt
    │   └── ...
    └── history.json              # Accuracy over time
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Transition:
    """A single observed (before, action, after) triple."""
    before_frame: np.ndarray
    action_id: int
    after_frame: np.ndarray
    timestep: int


@dataclass
class AttemptRecord:
    """Record of one synthesis iteration."""
    iteration: int
    accuracy: float
    passed: int
    total: int
    reflection: str | None = None
    duration_ms: float = 0
    timestamp: str = ""


class SynthesisWorkspace:
    """
    Manages the persistent disk workspace for synthesis.

    The workspace is a directory containing all artifacts needed by the
    synthesis backend. Any backend (API, Claude Code, custom) operates
    on these files.
    """

    def __init__(self, workspace_dir: str | Path):
        self.dir = Path(workspace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "reflections").mkdir(exist_ok=True)

    @property
    def code_path(self) -> Path:
        return self.dir / "game_engine.py"

    @property
    def initial_frame_path(self) -> Path:
        return self.dir / "initial_frame.npy"

    @property
    def test_runner_path(self) -> Path:
        return self.dir / "test_runner.py"

    @property
    def context_path(self) -> Path:
        return self.dir / "context.txt"

    @property
    def replay_buffer_path(self) -> Path:
        return self.dir / "replay_buffer.pkl"

    @property
    def history_path(self) -> Path:
        return self.dir / "history.json"

    @property
    def reflections_dir(self) -> Path:
        return self.dir / "reflections"

    def write_context(self, structured_analysis: str) -> None:
        self.context_path.write_text(structured_analysis)

    def write_replay_buffer(self, replay_buffer: list[Transition]) -> None:
        with open(self.replay_buffer_path, "wb") as f:
            pickle.dump(replay_buffer, f)

    def write_initial_frame(self, frame: np.ndarray) -> None:
        """Save the first frame for perceive() bootstrapping."""
        np.save(self.initial_frame_path, frame)

    def write_code(self, code: str) -> None:
        """Write or overwrite the transition rules code."""
        self.code_path.write_text(code)

    def read_code(self) -> str | None:
        """Read the current transition rules code, or None if not yet written."""
        if self.code_path.exists():
            return self.code_path.read_text()
        return None

    def write_test_runner(self) -> None:
        """
        Write the test runner that verifies game_engine.py via PIXEL-LEVEL comparison.

        The game engine is a STATEFUL SIMULATOR:
        - perceive(frame): initialize internal state from initial frame (called once)
        - transition(action_id): update internal state
        - render(): produce predicted frame from internal state

        Test: predicted pixels vs actual game frame pixels.
        Transitions are tested SEQUENTIALLY (state accumulates).
        """
        script = r'''#!/usr/bin/env python3
"""
Test runner: pixel-level verification of game_engine.py.

The engine must implement three functions:
- perceive(frame): initialize internal state from initial frame (H,W,3 uint8 RGB)
- transition(action_id): update internal state for the given action
- render(): return predicted frame as (H,W,3) uint8 RGB numpy array

Tests run SEQUENTIALLY: perceive once, then transition+render for each step.
Comparison is pure pixel diff: predicted render vs actual game frame.
"""

import sys
import os
import pickle
import importlib.util
import numpy as np

def load_engine(code_path):
    spec = importlib.util.spec_from_file_location("game_engine", code_path)
    module = importlib.util.module_from_spec(spec)
    module.np = np
    module.copy = __import__("copy")
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Code failed to load: {type(e).__name__}: {e}"

    missing = []
    if not hasattr(module, "perceive"): missing.append("perceive")
    if not hasattr(module, "transition"): missing.append("transition")
    if not hasattr(module, "render"): missing.append("render")
    if missing:
        return None, f"Missing required functions: {', '.join(missing)}"
    return module, None


def pixel_diff(predicted, actual):
    if predicted.shape != actual.shape:
        return predicted.shape[0] * predicted.shape[1], \
            f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}", []

    if predicted.ndim == 3:
        diff_mask = np.any(predicted != actual, axis=-1)
    else:
        diff_mask = predicted != actual

    diff_count = int(np.sum(diff_mask))
    samples = []
    if diff_count > 0:
        positions = list(zip(*np.where(diff_mask)))[:10]
        for r, c in positions:
            p = tuple(int(x) for x in predicted[r, c]) if predicted.ndim == 3 else int(predicted[r, c])
            a = tuple(int(x) for x in actual[r, c]) if actual.ndim == 3 else int(actual[r, c])
            samples.append(f"({r},{c}): predicted={p} actual={a}")

    return diff_count, "", samples


def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(workspace, "game_engine.py")
    buffer_path = os.path.join(workspace, "replay_buffer.pkl")
    init_frame_path = os.path.join(workspace, "initial_frame.npy")

    for name, path in [("game_engine.py", code_path), ("replay_buffer.pkl", buffer_path),
                        ("initial_frame.npy", init_frame_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found")
            sys.exit(1)

    engine, err = load_engine(code_path)
    if err:
        print(f"LOAD_ERROR: {err}")
        sys.exit(1)

    initial_frame = np.load(init_frame_path)
    with open(buffer_path, "rb") as f:
        replay_buffer = pickle.load(f)

    # Bootstrap: perceive the initial frame
    try:
        engine.perceive(initial_frame)
    except Exception as e:
        print(f"PERCEIVE_ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Sequential test: transition → render → compare pixels
    total = len(replay_buffer)
    passed = 0
    failures = []

    for i, trans in enumerate(replay_buffer):
        try:
            engine.transition(trans.action_id)
            predicted = engine.render()

            if predicted is None:
                failures.append({"i": i, "t": trans.timestep, "a": trans.action_id,
                                 "error": "render() returned None"})
                continue

            predicted = np.asarray(predicted, dtype=np.uint8)
            actual = np.asarray(trans.after_frame, dtype=np.uint8)

            diff_count, shape_err, samples = pixel_diff(predicted, actual)

            if diff_count == 0:
                passed += 1
            else:
                detail = shape_err if shape_err else "; ".join(samples)
                if diff_count > 10 and not shape_err:
                    detail += f"; ... and {diff_count - 10} more pixels"
                failures.append({"i": i, "t": trans.timestep, "a": trans.action_id,
                                 "diff": diff_count, "detail": detail})

        except Exception as e:
            import traceback
            failures.append({"i": i, "t": trans.timestep, "a": trans.action_id,
                             "error": f"{type(e).__name__}: {e}"})

    # Report
    print(f"RESULT: {passed}/{total} passed ({passed/total*100:.0f}%)")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:15]:
            print(f"  Step {f['i']} (T{f['t']} ACTION{f['a']}):")
            if "error" in f:
                print(f"    ERROR: {f['error']}")
            else:
                print(f"    {f['diff']} pixels differ")
                print(f"    {f['detail']}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
    else:
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
'''
        self.test_runner_path.write_text(script)

    def add_reflection(self, iteration: int, text: str) -> None:
        """Store a reflection on disk."""
        path = self.reflections_dir / f"{iteration:03d}.txt"
        path.write_text(text)

    def get_reflections(self, last_n: int = 3) -> list[str]:
        """Read the last N reflections from disk."""
        files = sorted(self.reflections_dir.glob("*.txt"))
        texts = []
        for f in files[-last_n:]:
            texts.append(f.read_text())
        return texts

    def record_attempt(self, record: AttemptRecord) -> None:
        """Append an attempt to history.json."""
        history = self._load_history()
        history.append({
            "iteration": record.iteration,
            "accuracy": record.accuracy,
            "passed": record.passed,
            "total": record.total,
            "reflection": record.reflection,
            "duration_ms": record.duration_ms,
            "timestamp": record.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.history_path.write_text(json.dumps(history, indent=2))

    def get_history(self) -> list[dict]:
        return self._load_history()

    def get_best_accuracy(self) -> float:
        history = self._load_history()
        if not history:
            return 0.0
        return max(h["accuracy"] for h in history)

    def _load_history(self) -> list[dict]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []
