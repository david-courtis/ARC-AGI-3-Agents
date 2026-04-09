"""
Modular synthesis: abstract backend interface + persistent workspace.

The SynthesisBackend interface is swappable:
    - ClaudeCodeBackend: spawns `claude` CLI for agentic code editing
    - (Add your own: any system that can edit files and run tests)
"""

from .workspace import SynthesisWorkspace, Transition, AttemptRecord
from .backend import SynthesisBackend
from .claude_code_backend import ClaudeCodeBackend
from .loop import ReflexionLoop

__all__ = [
    "SynthesisWorkspace",
    "SynthesisBackend",
    "ClaudeCodeBackend",
    "ReflexionLoop",
    "Transition",
    "AttemptRecord",
]
