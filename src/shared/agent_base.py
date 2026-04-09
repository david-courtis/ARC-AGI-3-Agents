"""
Base class for ARC-AGI-3 synthesis agents.

Replaces the old vendor/arc agents.agent.Agent base class with a lightweight
interface decoupled from HTTP transport. The agent loop is now driven by the
runner (run.py) rather than being baked into the base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from arcengine import FrameDataRaw, GameAction, GameState


class SynthesisAgent(ABC):
    """
    Lightweight base class for world model synthesis agents.

    The agent only decides what action to take and when to stop.
    Environment interaction (step, reset, scorecards) is handled by the runner.
    """

    MAX_ACTIONS: int = 100
    action_counter: int = 0

    # ACTION6 coordinate data. Set by choose_action before returning
    # GameAction.ACTION6. The runner reads this after choose_action returns.
    pending_action_data: dict[str, int] | None = None

    @abstractmethod
    def choose_action(self, obs: FrameDataRaw) -> GameAction:
        """Choose which action the agent should take given the current observation.

        Args:
            obs: The current FrameDataRaw from the environment.

        Returns:
            A GameAction to execute.
        """
        raise NotImplementedError

    @abstractmethod
    def is_done(self, obs: FrameDataRaw) -> bool:
        """Decide if the agent is done playing.

        Args:
            obs: The current FrameDataRaw from the environment.

        Returns:
            True if the agent should stop.
        """
        raise NotImplementedError
