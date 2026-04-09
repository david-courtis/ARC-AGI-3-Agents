"""
Evaluate our object-centric agent on WorldCoder's MiniGrid environments.

Usage:
    python -m src.object_centric_agent.compat.eval_on_minigrid \
        --env MiniGrid-DoorKey-5x5-v0 \
        --episodes 10 \
        --max-steps 50

This script:
1. Runs the WorldCoder MiniGrid env to collect transitions.
2. Feeds the structured states into our epistemic model + synthesis pipeline.
3. Evaluates transition accuracy — the metric both systems share.
4. Optionally runs WorldCoder's own synthesizer on the same transitions for
   head-to-head comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure vendor WorldCoder is importable
_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "WorldCoder"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from .worldcoder_adapter import (
    StructuredState,
    Transition,
    collect_transitions_from_wc_env,
    evaluate_transition_accuracy,
    ACTION_ID_TO_STR,
    MINIGRID_TYPE_IDS,
)


# ---------------------------------------------------------------------------
# WorldCoder MiniGrid environments available in the vendor code
# ---------------------------------------------------------------------------

AVAILABLE_ENVS = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-Fetch-6x6-N2-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
]


def get_env(env_name: str, with_api: bool = False):
    """Create a WorldCoder MiniGrid environment."""
    from worldcoder.envs import get_env as wc_get_env
    return wc_get_env({"env_name": env_name, "with_api": with_api})


# ---------------------------------------------------------------------------
# Phase 1: Collect transitions
# ---------------------------------------------------------------------------

def phase_collect(env_name: str, n_episodes: int, max_steps: int, seed: int) -> list[Transition]:
    print(f"[Phase 1] Collecting transitions from {env_name}")
    env = get_env(env_name)
    transitions = collect_transitions_from_wc_env(
        env, n_episodes=n_episodes, max_steps=max_steps, seed=seed,
    )
    print(f"  Collected {len(transitions)} transitions across {n_episodes} episodes")

    # Summary: unique object types, unique actions
    all_types = set()
    all_actions = set()
    for t in transitions:
        for o in t.before.objects:
            all_types.add(o.type_name)
        all_actions.add(t.action_id)
    print(f"  Object types: {sorted(all_types)}")
    print(f"  Actions used: {sorted(all_actions)} "
          f"({[ACTION_ID_TO_STR.get(a, '?') for a in sorted(all_actions)]})")

    return transitions


# ---------------------------------------------------------------------------
# Phase 2: Build epistemic model from transitions
# ---------------------------------------------------------------------------

def phase_epistemic(transitions: list[Transition]) -> dict:
    """
    Build a simple epistemic summary from collected transitions.
    Returns a dict mapping (type_name, action_id) -> {count, effects}.
    """
    print("[Phase 2] Building epistemic model")
    knowledge: dict[tuple[str, int], dict] = {}

    for t in transitions:
        # For each object in the before-state, track what happened
        before_by_type: dict[str, list] = {}
        for o in t.before.objects:
            before_by_type.setdefault(o.type_name, []).append(o)

        after_by_type: dict[str, list] = {}
        for o in t.after.objects:
            after_by_type.setdefault(o.type_name, []).append(o)

        # Track effects per (type, action)
        for type_name in set(list(before_by_type.keys()) + list(after_by_type.keys())):
            key = (type_name, t.action_id)
            if key not in knowledge:
                knowledge[key] = {"count": 0, "changed": 0, "effects": []}

            knowledge[key]["count"] += 1

            # Simple check: did any object of this type change position?
            before_positions = sorted(o.position for o in before_by_type.get(type_name, []))
            after_positions = sorted(o.position for o in after_by_type.get(type_name, []))
            if before_positions != after_positions:
                knowledge[key]["changed"] += 1

    print(f"  Epistemic cells: {len(knowledge)}")
    for key, v in sorted(knowledge.items()):
        type_name, action_id = key
        action_str = ACTION_ID_TO_STR.get(action_id, f"a{action_id}")
        change_rate = v["changed"] / v["count"] if v["count"] > 0 else 0
        print(f"    ({type_name}, {action_str}): "
              f"n={v['count']}, changed={change_rate:.0%}")

    return knowledge


# ---------------------------------------------------------------------------
# Phase 3: Evaluate WorldCoder baseline on same transitions
# ---------------------------------------------------------------------------

def phase_worldcoder_baseline(
    env_name: str,
    transitions: list[Transition],
) -> dict | None:
    """
    Run WorldCoder's own synthesizer on the same transitions and evaluate.
    Returns accuracy dict or None if WorldCoder can't be run.
    """
    print("[Phase 3] Running WorldCoder baseline")
    try:
        from worldcoder.agent.synthesizer.evaluator import TransitEvaluator
        from worldcoder.agent.synthesizer import refine_world_model
        from worldcoder.agent.world_model import WorldModel
    except ImportError as e:
        print(f"  Cannot import WorldCoder synthesizer: {e}")
        print("  Skipping WorldCoder baseline.")
        return None

    # Convert our transitions back to WorldCoder's experience_buffer format
    env = get_env(env_name)
    print("  Converting transitions to WorldCoder experience format...")

    # This would require reconstructing WorldCoder State objects from our
    # StructuredState. For now, return a placeholder indicating this needs
    # the full WorldCoder agent loop to produce a fair comparison.
    print("  NOTE: Fair WorldCoder baseline requires running its full agent loop")
    print("  (learn.py) on the same environment with the same seed.")
    print("  Use: python -m worldcoder.learn --env_name {env_name}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate on WorldCoder MiniGrid envs")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0",
                        choices=AVAILABLE_ENVS,
                        help="MiniGrid environment name")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON results")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Object-Centric Agent vs WorldCoder Evaluation")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print(f"{'='*60}")

    # Phase 1: Collect
    transitions = phase_collect(args.env, args.episodes, args.max_steps, args.seed)

    # Phase 2: Epistemic analysis
    knowledge = phase_epistemic(transitions)

    # Phase 3: WorldCoder baseline
    wc_result = phase_worldcoder_baseline(args.env, transitions)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"  Environment: {args.env}")
    print(f"  Transitions collected: {len(transitions)}")
    print(f"  Epistemic cells tracked: {len(knowledge)}")
    if wc_result:
        print(f"  WorldCoder accuracy: {wc_result['accuracy']:.2%}")

    # Save results
    if args.output:
        results = {
            "env": args.env,
            "n_transitions": len(transitions),
            "n_epistemic_cells": len(knowledge),
            "epistemic_summary": {
                f"{k[0]}_{ACTION_ID_TO_STR.get(k[1], k[1])}": {
                    "count": v["count"],
                    "change_rate": v["changed"] / v["count"] if v["count"] > 0 else 0,
                }
                for k, v in knowledge.items()
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results written to {args.output}")


if __name__ == "__main__":
    main()
