"""
Abstract synthesis backend interface.

Any system that can edit code based on test feedback implements this.
The workspace is on disk; the backend reads/writes files in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .workspace import SynthesisWorkspace


class SynthesisBackend(ABC):
    """
    Abstract interface for a code synthesis agent.

    The backend operates on a SynthesisWorkspace:
    - Reads context.txt (what we know about the game)
    - Reads/edits transition_rules.py (the code being synthesized)
    - Runs test_runner.py to check correctness
    - Iterates until all tests pass or gives up

    Implementations:
    - ClaudeCodeBackend: spawns `claude` CLI with the workspace
    - APIReflexionBackend: stateless API calls (fallback)
    - Any custom agent that can edit files and run tests
    """

    @abstractmethod
    def run(self, workspace: SynthesisWorkspace) -> bool:
        """
        Run the synthesis loop on the given workspace.

        The backend should:
        1. Read context.txt to understand the game
        2. Write/edit transition_rules.py
        3. Run `python test_runner.py` to check
        4. If tests fail: reflect, edit code, test again
        5. Repeat until all tests pass or stuck

        Returns True if all tests pass, False if stuck/gave up.

        All artifacts (code, reflections, history) should be written
        to the workspace so they persist across calls.
        """
        raise NotImplementedError
