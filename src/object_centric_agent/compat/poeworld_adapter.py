"""
Compatibility adapter for running our object-centric agent on PoE-World's
Atari environments (Pong, Montezuma's Revenge) and evaluating with their metrics.

PoE-World states are ObjList objects containing Obj instances with:
    obj_type, id, x, y, prev_x, prev_y, w, h, velocity_x, velocity_y, deleted, ...

PoE-World actions are string labels:
    Pong: 'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'
    Montezuma: 'NOOP', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'FIRE', ...

Our system expects:
    - StructuredState (list of ObjectSnapshot with type/position/attributes)
    - Integer action IDs (1..N)

This module bridges the two representations.
"""
from __future__ import annotations

import sys
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Add PoE-World vendor path
# ---------------------------------------------------------------------------
_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "poe-world"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from .worldcoder_adapter import ObjectSnapshot, StructuredState


# ---------------------------------------------------------------------------
# PoE-World Obj ←→ our ObjectSnapshot
# ---------------------------------------------------------------------------

def obj_to_snapshot(obj) -> ObjectSnapshot:
    """Convert a PoE-World Obj to our ObjectSnapshot."""
    attrs = {
        "velocity_x": _resolve_value(obj.velocity_x),
        "velocity_y": _resolve_value(obj.velocity_y),
        "w": obj.w,
        "h": obj.h,
        "deleted": _resolve_value(obj.deleted),
        "id": obj.id,
    }
    # Include prev position if available
    if hasattr(obj, "prev_x") and obj.prev_x is not None:
        attrs["prev_x"] = obj.prev_x
        attrs["prev_y"] = obj.prev_y

    return ObjectSnapshot(
        type_name=obj.obj_type,
        position=(obj.x, obj.y),
        attributes=attrs,
    )


def _resolve_value(v) -> int | float:
    """Resolve a PoE-World RandomValues to its max-prob value, or pass through."""
    if hasattr(v, "get_max_prob_value"):
        return v.get_max_prob_value()
    return v


def objlist_to_structured_state(obj_list) -> StructuredState:
    """Convert a PoE-World ObjList to our StructuredState."""
    snapshots = []
    objs = obj_list.objs if hasattr(obj_list, "objs") else obj_list
    for obj in objs:
        # Skip deleted objects
        deleted = _resolve_value(obj.deleted) if hasattr(obj, "deleted") else 0
        if deleted:
            continue
        snapshots.append(obj_to_snapshot(obj))
    return StructuredState(objects=snapshots)


# ---------------------------------------------------------------------------
# Action mapping per game
# ---------------------------------------------------------------------------

PONG_ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
MONTEZUMA_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]

GAME_ACTIONS = {
    "Pong": PONG_ACTIONS,
    "MontezumaRevenge": MONTEZUMA_ACTIONS,
}


def make_action_maps(game: str) -> tuple[dict[str, int], dict[int, str]]:
    """Create bidirectional action maps for a game. IDs are 1-indexed."""
    actions = GAME_ACTIONS[game]
    str_to_id = {s: i + 1 for i, s in enumerate(actions)}
    id_to_str = {v: k for k, v in str_to_id.items()}
    return str_to_id, id_to_str


# ---------------------------------------------------------------------------
# Object type registries per game
# ---------------------------------------------------------------------------

PONG_TYPES = {
    "player": 0,
    "ball": 1,
    "enemy": 2,
    "player_score": 3,
    "enemy_score": 4,
    "wall": 5,
    "zone": 6,
}

MONTEZUMA_TYPES = {
    "player": 0,
    "skull": 1,
    "spider": 2,
    "snake": 3,
    "key": 4,
    "amulet": 5,
    "torch": 6,
    "sword": 7,
    "ruby": 8,
    "barrier": 9,
    "beam": 10,
    "rope": 11,
    "platform": 12,
    "ladder": 13,
    "conveyer_belt": 14,
    "wall": 15,
    "disappearing_platform": 16,
    "score": 17,
    "life": 18,
}

GAME_TYPE_IDS = {
    "Pong": PONG_TYPES,
    "MontezumaRevenge": MONTEZUMA_TYPES,
}


# ---------------------------------------------------------------------------
# Transition collection from PoE-World saved observations
# ---------------------------------------------------------------------------

@dataclass
class PoeTransition:
    """A single observed transition in our format, from PoE-World data."""
    before: StructuredState
    action_id: int
    action_str: str
    after: StructuredState
    game_state_before: str | None = None
    game_state_after: str | None = None
    timestep: int = 0


def load_poeworld_transitions(
    game: str,
    obs_suffix: str = "_basic2",
) -> list[PoeTransition]:
    """
    Load saved PoE-World observations and convert to our transition format.

    Args:
        game: "Pong" or "MontezumaRevenge"
        obs_suffix: observation file suffix (e.g., "_basic2" for Pong)

    Returns:
        List of PoeTransition objects.
    """
    try:
        from data.atari import load_atari_observations
    except ImportError:
        # Try alternate import path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "data.atari",
            str(_VENDOR_ROOT / "data" / "atari.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        load_atari_observations = mod.load_atari_observations

    identifier = f"{game}{obs_suffix}"
    observations, actions, game_states = load_atari_observations(identifier)

    str_to_id, _ = make_action_maps(game)
    transitions = []

    for t in range(len(actions)):
        if t + 1 >= len(observations):
            break

        before_ss = objlist_to_structured_state(observations[t])
        after_ss = objlist_to_structured_state(observations[t + 1])
        action_str = actions[t]
        action_id = str_to_id.get(action_str, 0)

        gs_before = game_states[t].value if hasattr(game_states[t], "value") else str(game_states[t])
        gs_after = game_states[t + 1].value if t + 1 < len(game_states) and hasattr(game_states[t + 1], "value") else None

        transitions.append(PoeTransition(
            before=before_ss,
            action_id=action_id,
            action_str=action_str,
            after=after_ss,
            game_state_before=gs_before,
            game_state_after=gs_after,
            timestep=t,
        ))

    return transitions


# ---------------------------------------------------------------------------
# Collect transitions from a live PoE-World AtariEnv
# ---------------------------------------------------------------------------

def collect_transitions_from_atari_env(
    game: str,
    config: Any,
    action_sequence: list[str] | None = None,
    n_steps: int = 100,
    seed: int = 0,
) -> list[PoeTransition]:
    """
    Run a PoE-World Atari environment and collect transitions.

    Args:
        game: "Pong" or "MontezumaRevenge"
        config: Hydra DictConfig for the game
        action_sequence: Specific actions to execute (if None, random)
        n_steps: Number of steps if action_sequence is None
        seed: Random seed for random actions

    Returns:
        List of PoeTransition objects.
    """
    from classes.envs.env import AtariEnv
    from classes.envs.object_tracker import ObjectTracker

    if game == "Pong":
        from classes.game_utils.pong import PongActions, pong_wh_dict
        from classes.envs.game_state_tracker import PongStateTracker
        from classes.helper import Constants
        Constants.set_constants(
            pong_wh_dict,
            max_abs_velocity=30,
            history_length=100,
            max_abs_size_change=1,
            actions=PongActions.get_all_possible_actions(),
        )
        actions_enum = PongActions
        tracker = PongStateTracker()
    elif game == "MontezumaRevenge":
        from classes.game_utils.montezuma import MontezumaRevengeActions, montezuma_revenge_wh_dict
        from classes.envs.game_state_tracker import MontezumaRevengeStateTracker
        from classes.helper import Constants
        Constants.set_constants(
            montezuma_revenge_wh_dict,
            max_abs_velocity=15,
            history_length=100,
            max_abs_size_change=1,
            actions=MontezumaRevengeActions.get_all_possible_actions(),
        )
        actions_enum = MontezumaRevengeActions
        tracker = MontezumaRevengeStateTracker()
    else:
        raise ValueError(f"Unknown game: {game}")

    env = AtariEnv(
        config=config,
        env_name=game,
        object_tracker=ObjectTracker(),
        game_state_tracker=tracker,
        actions_enum=actions_enum,
    )

    obj_list, game_state = env.reset()
    str_to_id, _ = make_action_maps(game)
    all_actions = actions_enum.get_all_possible_actions()
    rng = np.random.default_rng(seed)

    if action_sequence is None:
        action_sequence = [rng.choice(all_actions) for _ in range(n_steps)]

    transitions = []
    for t, action_str in enumerate(action_sequence):
        before_ss = objlist_to_structured_state(obj_list)
        gs_before = game_state.value if hasattr(game_state, "value") else str(game_state)

        obj_list, game_state = env.step(action_str)

        after_ss = objlist_to_structured_state(obj_list)
        gs_after = game_state.value if hasattr(game_state, "value") else str(game_state)

        transitions.append(PoeTransition(
            before=before_ss,
            action_id=str_to_id.get(action_str, 0),
            action_str=action_str,
            after=after_ss,
            game_state_before=gs_before,
            game_state_after=gs_after,
            timestep=t,
        ))

    return transitions


# ---------------------------------------------------------------------------
# Evaluation: transition accuracy of our model on PoE-World data
# ---------------------------------------------------------------------------

def evaluate_transition_accuracy(
    predict_fn,
    transitions: list[PoeTransition],
    mode: str = "partial",
) -> dict:
    """
    Evaluate a prediction function against PoE-World transitions.

    Args:
        predict_fn: Callable(StructuredState, int) -> StructuredState
        transitions: Ground-truth transitions.
        mode: "exact" (all attributes match) or "partial" (position-only for
              moving objects, matching PoE-World's eval.py metric).

    Returns:
        Dict with accuracy, correct_count, total_count, per-transition results.
    """
    correct = 0
    results = []

    for t in transitions:
        try:
            predicted = predict_fn(t.before, t.action_id)
            if mode == "partial":
                is_correct = _partial_match(predicted, t.after)
            else:
                is_correct = (predicted == t.after)
        except Exception:
            is_correct = False

        correct += int(is_correct)
        results.append({
            "timestep": t.timestep,
            "action": t.action_str,
            "correct": is_correct,
        })

    total = len(transitions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct_count": correct,
        "total_count": total,
        "results": results,
    }


def _partial_match(predicted: StructuredState, actual: StructuredState) -> bool:
    """
    PoE-World's partial match: only compare moving object types (player,
    ball, enemy, skull, spider, snake). Static objects (walls, scores, etc.)
    are ignored.
    """
    moving_types = {"player", "ball", "enemy", "skull", "spider", "snake"}

    pred_moving = sorted(
        [(o.type_name, o.position) for o in predicted.objects if o.type_name in moving_types],
        key=str,
    )
    actual_moving = sorted(
        [(o.type_name, o.position) for o in actual.objects if o.type_name in moving_types],
        key=str,
    )
    return pred_moving == actual_moving


# ---------------------------------------------------------------------------
# Adapter for using PoE-World's saved world models
# ---------------------------------------------------------------------------

def load_poeworld_model(model_path: str | Path):
    """
    Load a pre-trained PoE-World world model from a text file.

    Returns a WorldModel object that can be used with sample_next_scene().
    """
    from learners.models import WorldModel
    import pickle

    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        world_model = pickle.load(f)

    if hasattr(world_model, "prepare_callables"):
        world_model.prepare_callables()

    return world_model
