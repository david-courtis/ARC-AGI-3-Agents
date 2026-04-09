"""
Orchestrates the synthesis loop: workspace setup → backend run → result extraction.

The ReflexionLoop is what the agent calls. It:
1. Sets up the workspace (writes context, replay buffer, test runner)
2. Delegates to the SynthesisBackend (Claude Code, API, or custom)
3. Reads back the results (code, test accuracy)
4. Reports success/failure to the agent
"""

from __future__ import annotations

import logging
import subprocess
import os
from pathlib import Path
from typing import Any

import numpy as np

from .workspace import SynthesisWorkspace, Transition
from .backend import SynthesisBackend

logger = logging.getLogger(__name__)


class ReflexionLoop:
    """
    Top-level synthesis orchestrator.

    Usage:
        loop = ReflexionLoop(
            backend=ClaudeCodeBackend(),
            workspace_dir="results/run_xxx/synthesis",
        )
        success = loop.run(
            replay_buffer=transitions,
            structured_analysis="...",
        )
        if success:
            code = loop.workspace.read_code()
    """

    def __init__(
        self,
        backend: SynthesisBackend,
        workspace_dir: str | Path,
    ):
        self.backend = backend
        self.workspace = SynthesisWorkspace(workspace_dir)
        self._run_count: int = 0

    @property
    def current_code(self) -> str | None:
        return self.workspace.read_code()

    @property
    def best_accuracy(self) -> float:
        return self.workspace.get_best_accuracy()

    @property
    def is_perfect(self) -> bool:
        return self.best_accuracy >= 1.0

    def run(
        self,
        replay_buffer: list[Transition],
        structured_analysis: str,
        initial_frame: Any = None,
    ) -> bool:
        """
        Set up workspace and run the synthesis backend.

        Args:
            replay_buffer: Observed transitions.
            structured_analysis: World model markdown.
            initial_frame: First frame for perceive() bootstrapping.

        Returns True if all tests pass.
        """
        self._run_count += 1

        logger.info(
            f"[SynthLoop] Run #{self._run_count}: "
            f"{len(replay_buffer)} transitions, "
            f"prior code: {'yes' if self.current_code else 'no'}, "
            f"prior best: {self.best_accuracy:.0%}"
        )

        # Write/update workspace files
        self.workspace.write_context(structured_analysis)
        self.workspace.write_replay_buffer(replay_buffer)
        self.workspace.write_test_runner()

        # Write initial frame for perceive() bootstrapping
        if initial_frame is not None:
            self.workspace.write_initial_frame(initial_frame)

        # If no code exists yet, write an OOP stub
        if self.current_code is None:
            self.workspace.write_code(
                '# Game engine — Object-Oriented game simulator\n'
                '# Read context.txt for the world model.\n'
                '# Structure: GameObject subclasses + World + perceive/transition/render\n\n'
                'import numpy as np\nimport copy\n\n'
                '# ========== OBJECT TYPES ==========\n'
                '# Define one class per object type detected in context.txt.\n'
                '# Each class must implement respond(action_id, world) and render(frame).\n\n'
                'class StaticObject:\n'
                '    """Base for objects that never change (walls, borders)."""\n'
                '    def __init__(self, obj_id, row, col, pixels):\n'
                '        self.obj_id = obj_id\n'
                '        self.row = row\n'
                '        self.col = col\n'
                '        self.pixels = pixels  # list of (r, c, color) tuples\n\n'
                '    def respond(self, action_id, world):\n'
                '        pass  # static\n\n'
                '    def render(self, frame):\n'
                '        for r, c, color in self.pixels:\n'
                '            if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1]:\n'
                '                frame[r, c] = color\n\n'
                '# ========== WORLD ==========\n'
                'class World:\n'
                '    def __init__(self, objects, background):\n'
                '        self.objects = objects\n'
                '        self.background = background\n\n'
                '    def get_objects_of_type(self, cls):\n'
                '        return [o for o in self.objects if isinstance(o, cls)]\n\n'
                '# ========== MODULE INTERFACE ==========\n'
                '_world = None\n\n'
                'def perceive(frame):\n'
                '    """Detect objects from initial frame, build World."""\n'
                '    global _world\n'
                '    _world = World([], frame.copy())\n\n'
                'def transition(action_id):\n'
                '    """Broadcast action to all objects in respond order."""\n'
                '    global _world\n'
                '    for obj in _world.objects:\n'
                '        obj.respond(action_id, _world)\n\n'
                'def render():\n'
                '    """Composite render: background + each object."""\n'
                '    global _world\n'
                '    frame = _world.background.copy()\n'
                '    for obj in _world.objects:\n'
                '        obj.render(frame)\n'
                '    return frame\n'
            )

        # Check if current code already passes (e.g., new transitions are
        # already handled by existing rules)
        if self._quick_test():
            logger.info("[SynthLoop] Current code already passes all tests!")
            return True

        # Run the backend
        success = self.backend.run(self.workspace)

        # Record result
        accuracy = self._get_test_accuracy()
        self.workspace.record_attempt(
            __import__('src.object_centric_agent.synth_loop.workspace', fromlist=['AttemptRecord']).AttemptRecord(
                iteration=self._run_count,
                accuracy=accuracy,
                passed=int(accuracy * len(replay_buffer)),
                total=len(replay_buffer),
            )
        )

        if success:
            logger.info(f"[SynthLoop] Backend achieved 100%!")
        else:
            logger.info(f"[SynthLoop] Backend finished at {accuracy:.0%}")

        return success

    def _quick_test(self) -> bool:
        """Run the test runner and check if all tests pass."""
        if not self.workspace.code_path.exists():
            return False

        project_root = str(Path(__file__).resolve().parents[3])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        try:
            result = subprocess.run(
                ["python", str(self.workspace.test_runner_path)],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.workspace.dir),
                env=env,
            )
            if "100%)" in result.stdout or "ALL TESTS PASSED" in result.stdout:
                return True
        except Exception:
            pass
        return False

    def _get_test_accuracy(self) -> float:
        """Run the test runner and parse accuracy from output."""
        project_root = str(Path(__file__).resolve().parents[3])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        try:
            result = subprocess.run(
                ["python", str(self.workspace.test_runner_path)],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.workspace.dir),
                env=env,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("RESULT:"):
                    # Parse "RESULT: 7/10 passed (70%)"
                    parts = line.split()
                    if len(parts) >= 2:
                        frac = parts[1].split("/")
                        if len(frac) == 2:
                            return int(frac[0]) / int(frac[1])
        except Exception:
            pass
        return 0.0
