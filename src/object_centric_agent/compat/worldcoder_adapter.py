"""
Compatibility adapter for running our object-centric agent on WorldCoder's
MiniGrid environments and evaluating with WorldCoder's metrics.

WorldCoder states are lists of dicts:
    [{"type": "agent", "position": (3,4), "direction": (1,0)},
     {"type": "key",   "position": (1,2), "color": "red"}, ...]

WorldCoder actions are string labels:
    "turn left", "turn right", "move forward", "pick up", "drop", "toggle", "nothing"

Our system expects:
    - Structured state as WorldState (list of SpriteInstance with type/position/track_id)
    - Integer action IDs (1..N)

This module bridges the two.
"""
from __future__ import annotations

import sys
import os
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Add WorldCoder vendor path so we can import its env machinery
# ---------------------------------------------------------------------------
_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "WorldCoder"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))


# ---------------------------------------------------------------------------
# WorldCoder state ←→ our structured state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectSnapshot:
    """Minimal object representation shared between both systems."""
    type_name: str
    position: tuple[int, int]
    attributes: dict = field(default_factory=dict)  # color, direction, state, carrying, ...

    def to_wc_dict(self) -> dict:
        d = {"type": self.type_name, "position": list(self.position)}
        d.update(self.attributes)
        return d

    @classmethod
    def from_wc_dict(cls, d: dict) -> "ObjectSnapshot":
        attrs = {k: v for k, v in d.items() if k not in ("type", "position")}
        return cls(
            type_name=d["type"],
            position=tuple(d["position"]),
            attributes=attrs,
        )


@dataclass
class StructuredState:
    """
    A domain-agnostic structured state used as the common representation
    between WorldCoder environments and our object-centric agent.

    WorldCoder's State.state is a list[dict].  We lift that into a list
    of ObjectSnapshot for type-safe handling, and provide conversion in
    both directions.
    """
    objects: list[ObjectSnapshot]

    # -- conversions --------------------------------------------------------

    @classmethod
    def from_wc_state(cls, wc_state_list: list[dict]) -> "StructuredState":
        return cls(objects=[ObjectSnapshot.from_wc_dict(d) for d in wc_state_list])

    def to_wc_state_list(self) -> list[dict]:
        return [o.to_wc_dict() for o in self.objects]

    # -- queries used by our agent's relational context ---------------------

    def objects_of_type(self, type_name: str) -> list[ObjectSnapshot]:
        return [o for o in self.objects if o.type_name == type_name]

    def object_at(self, pos: tuple[int, int]) -> list[ObjectSnapshot]:
        return [o for o in self.objects if o.position == pos]

    def player(self) -> ObjectSnapshot | None:
        agents = self.objects_of_type("agent")
        return agents[0] if agents else None

    # -- equality (used for transition accuracy) ----------------------------

    def __eq__(self, other):
        if not isinstance(other, StructuredState):
            return False
        return sorted(str(o) for o in self.objects) == sorted(str(o) for o in other.objects)


# ---------------------------------------------------------------------------
# Action mapping: WorldCoder string actions ←→ our integer IDs
# ---------------------------------------------------------------------------

MINIGRID_ACTIONS = [
    "turn left",     # 0
    "turn right",    # 1
    "move forward",  # 2
    "pick up",       # 3
    "drop",          # 4
    "toggle",        # 5
    "nothing",       # 6
]

ACTION_STR_TO_ID = {s: i + 1 for i, s in enumerate(MINIGRID_ACTIONS)}  # 1-indexed
ACTION_ID_TO_STR = {v: k for k, v in ACTION_STR_TO_ID.items()}


def wc_action_to_id(wc_action) -> int:
    """Convert a WorldCoder Action object (or string) to our integer ID."""
    s = wc_action.to_pyrunnable() if hasattr(wc_action, "to_pyrunnable") else str(wc_action)
    return ACTION_STR_TO_ID[s]


def id_to_wc_action_str(action_id: int) -> str:
    return ACTION_ID_TO_STR[action_id]


# ---------------------------------------------------------------------------
# Transition collection: run WorldCoder env, record (s, a, s') tuples
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A single observed transition in our format."""
    before: StructuredState
    action_id: int
    after: StructuredState
    reward: float = 0.0
    done: bool = False
    timestep: int = 0


def collect_transitions_from_wc_env(
    env,
    n_episodes: int = 10,
    max_steps: int = 50,
    seed: int = 0,
    policy: str = "random",
) -> list[Transition]:
    """
    Run a WorldCoder environment and collect transitions in our format.

    Args:
        env: A WorldCoder environment instance (e.g., from worldcoder.envs.get_env).
        n_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        policy: "random" for uniform random actions.

    Returns:
        List of Transition objects.
    """
    rng = np.random.default_rng(seed)
    transitions = []
    timestep = 0

    for ep in range(n_episodes):
        state, mission, info = env.reset(seed=ep + seed)
        done = False

        for t in range(max_steps):
            if done:
                break

            # Select action
            valid_actions = state.get_valid_actions() if hasattr(state, "get_valid_actions") else None
            if valid_actions:
                action = rng.choice(valid_actions)
            else:
                # Fallback: use env action space
                action = rng.choice(list(range(env.action_space.n)))

            before_ss = StructuredState.from_wc_state(state.state)

            new_state, reward, done, new_info = env.step(action)

            after_ss = StructuredState.from_wc_state(new_state.state)
            action_id = wc_action_to_id(action)

            transitions.append(Transition(
                before=before_ss,
                action_id=action_id,
                after=after_ss,
                reward=reward,
                done=done,
                timestep=timestep,
            ))

            state = copy.deepcopy(new_state)
            timestep += 1

    return transitions


# ---------------------------------------------------------------------------
# Evaluation: transition accuracy of our model on WorldCoder data
# ---------------------------------------------------------------------------

def evaluate_transition_accuracy(
    predict_fn,
    transitions: list[Transition],
) -> dict:
    """
    Evaluate a transition prediction function against collected transitions.

    Args:
        predict_fn: Callable(StructuredState, int) -> StructuredState
            Takes a before-state and action_id, returns predicted after-state.
        transitions: List of ground-truth Transition objects.

    Returns:
        Dict with accuracy, correct_count, total_count, and per-transition results.
    """
    correct = 0
    results = []

    for t in transitions:
        try:
            predicted = predict_fn(t.before, t.action_id)
            is_correct = (predicted == t.after)
        except Exception as e:
            predicted = None
            is_correct = False

        correct += int(is_correct)
        results.append({
            "timestep": t.timestep,
            "action_id": t.action_id,
            "correct": is_correct,
            "predicted": predicted,
            "actual": t.after,
        })

    total = len(transitions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct_count": correct,
        "total_count": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Object type mapping: WorldCoder entity types → our type IDs
# ---------------------------------------------------------------------------

# Standard MiniGrid types. Stable integer IDs for the epistemic matrix.
MINIGRID_TYPE_IDS = {
    "agent": 0,
    "wall": 1,
    "door": 2,
    "key": 3,
    "goal": 4,
    "box": 5,
    "ball": 6,
    "lava": 7,
}


def type_name_to_id(type_name: str) -> int:
    return MINIGRID_TYPE_IDS.get(type_name.lower(), -1)


# ---------------------------------------------------------------------------
# Quick integration test
# ---------------------------------------------------------------------------

def _smoke_test():
    """Verify the adapter works with a WorldCoder MiniGrid env."""
    try:
        from worldcoder.envs import get_env
    except ImportError:
        print("WorldCoder not on path or missing dependencies. Skipping smoke test.")
        return

    env_args = {"env_name": "MiniGrid-Empty-5x5-v0", "with_api": False}
    env = get_env(env_args)

    transitions = collect_transitions_from_wc_env(env, n_episodes=2, max_steps=20)
    print(f"Collected {len(transitions)} transitions")

    # Trivial "identity" predictor (always returns the before-state unchanged)
    def identity_predict(before, action_id):
        return before

    result = evaluate_transition_accuracy(identity_predict, transitions)
    print(f"Identity predictor accuracy: {result['accuracy']:.2%} "
          f"({result['correct_count']}/{result['total_count']})")

    # Check state conversion round-trip
    if transitions:
        t = transitions[0]
        wc_list = t.before.to_wc_state_list()
        roundtrip = StructuredState.from_wc_state(wc_list)
        assert roundtrip == t.before, "Round-trip conversion failed"
        print("State round-trip: OK")

    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
