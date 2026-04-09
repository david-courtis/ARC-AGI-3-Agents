"""
Claude Code backend: uses the `claude` CLI as the reflexion agent.

Claude Code maintains a persistent conversation, can read/edit files,
run tests, see results, and iterate — exactly the reflexion loop but
with full context and agentic capabilities.

The backend:
1. Writes workspace files (context, test runner, replay buffer, initial code)
2. Spawns `claude` with a prompt pointing at the workspace
3. Claude Code reads context, edits transition_rules.py, runs test_runner.py
4. Claude Code iterates until all tests pass (or hits max turns)
5. Backend reads back the final code and test results
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .backend import SynthesisBackend
from .workspace import SynthesisWorkspace

logger = logging.getLogger(__name__)

# Path to claude CLI — discovered at import time
_CLAUDE_PATHS = [
    os.path.expanduser("~/.vscode-server/extensions/anthropic.claude-code-2.1.85-linux-x64/resources/native-binary/claude"),
    os.path.expanduser("~/.vscode-server/extensions/anthropic.claude-code-2.1.81-linux-x64/resources/native-binary/claude"),
    "claude",  # fallback to PATH
]


def _find_claude() -> str | None:
    for p in _CLAUDE_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Try PATH
    from shutil import which
    return which("claude")


SYNTHESIS_PROMPT_TEMPLATE = """You are building a stateful game simulator for an unknown domain. Your workspace is:
{workspace_dir}

FILES:
- context.txt: World Model — detected objects, hypothesized roles, observed effects. READ FIRST.
- game_engine.py: YOUR CODE. Edit this file.
- initial_frame.npy: First game frame (numpy, H×W×3, uint8, RGB).
- test_runner.py: Run with: PYTHONPATH={project_root} python {test_runner_path}
- replay_buffer.pkl: Ground truth transitions (don't modify).

TASK: Write game_engine.py, run tests, fix until "ALL TESTS PASSED".

===== CODE STRUCTURE =====

game_engine.py must be OBJECT-ORIENTED. The domain is unknown — discover
what object types exist and how they behave from context.txt and the tests.

PATTERN (adapt class names and logic to the domain you discover):

    # One class per object type discovered in the game.
    # Each class has respond() (how it reacts to actions) and render() (how it draws).

    class SomeType:
        def __init__(self, obj_id, row, col, ...):
            self.obj_id = obj_id
            self.row = row          # atomic state — mutable
            self.col = col
            # ... any other state this type needs

        def respond(self, action_id, world):
            # How does THIS object change when action_id fires?
            # Query world for other objects (collisions, adjacency, etc.)
            # Mutate only self.
            pass

        def render(self, frame):
            # Paint this object's pixels onto frame at current position.
            pass

    # Container for all objects + queries.
    class World:
        def __init__(self, objects, background):
            self.objects = objects
            self.background = background

        def get_objects_of_type(self, cls):
            return [o for o in self.objects if isinstance(o, cls)]

        def remove_object(self, obj):
            self.objects = [o for o in self.objects if o is not obj]

    # Module interface (called by test runner):
    _world = None

    def perceive(frame):
        # Called ONCE. Detect objects from pixels. Build World.
        global _world
        ...

    def transition(action_id):
        # Called per step. Update all objects in causal order.
        global _world
        for obj in _world.objects:
            obj.respond(action_id, _world)

    def render():
        # Composite: background + each object renders itself.
        global _world
        frame = _world.background.copy()
        for obj in _world.objects:
            obj.render(frame)
        return frame

KEY PRINCIPLES:
- One class per object type. Class names come from what you discover.
- Each object's respond() handles ALL actions for that type (polymorphism).
- Objects query the World for inter-object checks (collision, adjacency, overlap).
- Object order in transition() may matter (causal dependencies).
- Static objects have trivial respond() (no-op).
- Render is per-object: each paints its own pixels at its current position.

THE WORLD MODEL MAY BE WRONG:
context.txt is a hypothesis. The test pixels are ground truth. If they conflict,
trust the tests. Through testing and fixing, refine your understanding.

REQUIREMENTS:
- 100% pixel accuracy. Do not stop until all tests pass.
- Tests are sequential — state accumulates. Fix early failures first.
- Available imports: numpy as np, copy

START: read context.txt, write game_engine.py, run tests, iterate."""


class ClaudeCodeBackend(SynthesisBackend):
    """
    Synthesis backend that spawns Claude Code CLI to do the reflexion loop.

    Claude Code gets full agentic capabilities: read files, edit code,
    run tests, iterate with full conversation history.
    """

    def __init__(
        self,
        max_turns: int = 30,
        timeout_seconds: int = 600,
        model: str | None = None,
        claude_path: str | None = None,
    ):
        """
        Args:
            max_turns: Maximum number of Claude Code turns (tool calls).
            timeout_seconds: Maximum wall-clock time for the synthesis run.
            model: Model to use (e.g., "sonnet"). None = default.
            claude_path: Path to claude CLI. None = auto-detect.
        """
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.claude_path = claude_path or _find_claude()

        if self.claude_path is None:
            raise RuntimeError(
                "Claude Code CLI not found. Install it or provide claude_path."
            )

    def run(self, workspace: SynthesisWorkspace) -> bool:
        """
        Spawn Claude Code to synthesize transition rules.

        Returns True if all tests pass.
        """
        t0 = time.time()

        # Resolve paths first
        test_runner = str(workspace.test_runner_path.resolve())
        workspace_abs = str(workspace.dir.resolve())
        project_root = str(Path(__file__).resolve().parents[3])

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            workspace_dir=workspace_abs,
            test_runner_path=test_runner,
            project_root=project_root,
        )

        # Build command — use -p with prompt as argument (not stdin)
        cmd = [
            self.claude_path,
            "-p", prompt,
            "--max-turns", str(self.max_turns),
            "--output-format", "text",
            "--allowedTools", "Bash,Read,Write,Edit",
        ]

        if self.model:
            cmd.extend(["--model", self.model])

        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        logger.info(
            f"[ClaudeCode] Spawning claude with {self.max_turns} max turns, "
            f"{self.timeout_seconds}s timeout"
        )
        logger.info(f"[ClaudeCode] Workspace: {workspace.dir}")
        logger.info(f"[ClaudeCode] cmd: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(workspace.dir),
                env=env,
            )

            duration_ms = (time.time() - t0) * 1000

            # Log output
            if result.stdout:
                # Truncate for logging but save full output
                output_path = workspace.dir / "claude_output.txt"
                output_path.write_text(result.stdout)
                logger.info(f"[ClaudeCode] Output saved to {output_path}")

                # Check for success in output
                for line in result.stdout.split("\n"):
                    if "ALL TESTS PASSED" in line or "100%)" in line:
                        logger.info(f"[ClaudeCode] SUCCESS — all tests passed in {duration_ms:.0f}ms")
                        return True

            if result.stderr:
                err_path = workspace.dir / "claude_stderr.txt"
                err_path.write_text(result.stderr)
                logger.warning(f"[ClaudeCode] stderr saved to {err_path}")

            if result.returncode != 0:
                logger.warning(
                    f"[ClaudeCode] Exited with code {result.returncode} "
                    f"after {duration_ms:.0f}ms"
                )

            # Check if the final code passes tests (Claude may have fixed it
            # but the success message wasn't in the last output)
            final_code = workspace.read_code()
            if final_code:
                logger.info("[ClaudeCode] Checking final code with test runner...")
                test_result = subprocess.run(
                    ["python", str(workspace.test_runner_path)],
                    capture_output=True, text=True,
                    timeout=30,
                    cwd=str(workspace.dir),
                    env=env,
                )
                if test_result.stdout:
                    logger.info(f"[ClaudeCode] Final test: {test_result.stdout.strip().split(chr(10))[0]}")
                    if "100%)" in test_result.stdout or "ALL TESTS PASSED" in test_result.stdout:
                        return True

            logger.info(f"[ClaudeCode] Did not achieve 100% in {duration_ms:.0f}ms")
            return False

        except subprocess.TimeoutExpired:
            logger.warning(f"[ClaudeCode] Timed out after {self.timeout_seconds}s")
            return False

        except Exception as e:
            logger.error(f"[ClaudeCode] Error: {e}")
            return False
