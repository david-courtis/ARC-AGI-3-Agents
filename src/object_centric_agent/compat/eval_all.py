"""
Unified evaluation script for all benchmarks:
  - MiniGrid (via WorldCoder adapter)
  - Atari Pong / Montezuma's Revenge (via PoE-World adapter)
  - ARC-AGI-3 (native)

Usage:
    # MiniGrid
    python -m src.object_centric_agent.compat.eval_all \
        --benchmark minigrid --env MiniGrid-DoorKey-5x5-v0

    # Atari (from saved PoE-World observations)
    python -m src.object_centric_agent.compat.eval_all \
        --benchmark pong --obs-suffix _basic2

    python -m src.object_centric_agent.compat.eval_all \
        --benchmark montezuma --obs-suffix _basic17

    # List available benchmarks
    python -m src.object_centric_agent.compat.eval_all --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "minigrid": {
        "source": "WorldCoder",
        "envs": [
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-Unlock-v0",
            "MiniGrid-Fetch-6x6-N2-v0",
            "MiniGrid-UnlockPickup-v0",
            "MiniGrid-BlockedUnlockPickup-v0",
        ],
        "state_type": "symbolic",
        "perception": "trivial (env provides object dicts)",
    },
    "pong": {
        "source": "PoE-World",
        "game": "Pong",
        "obs_suffix": "_basic2",
        "state_type": "symbolic (OCAtari)",
        "perception": "OCAtari RAM parsing (hand-tuned per game)",
    },
    "montezuma": {
        "source": "PoE-World",
        "game": "MontezumaRevenge",
        "obs_suffix": "_basic17",
        "state_type": "symbolic (OCAtari)",
        "perception": "OCAtari RAM parsing (hand-tuned per game)",
    },
    "arc-agi-3": {
        "source": "native",
        "state_type": "64x64 RGB pixels",
        "perception": "deterministic pipeline (fragment detection + sprite registry + co-movement + tracking)",
    },
}


# ---------------------------------------------------------------------------
# Evaluation dispatch
# ---------------------------------------------------------------------------

def eval_minigrid(env_name: str, episodes: int, max_steps: int, seed: int) -> dict:
    from .worldcoder_adapter import (
        collect_transitions_from_wc_env,
        evaluate_transition_accuracy,
        ACTION_ID_TO_STR,
    )
    from .eval_on_minigrid import get_env

    print(f"\n{'='*60}")
    print(f"MiniGrid: {env_name}")
    print(f"{'='*60}")

    env = get_env(env_name)
    transitions = collect_transitions_from_wc_env(
        env, n_episodes=episodes, max_steps=max_steps, seed=seed,
    )
    print(f"Collected {len(transitions)} transitions")

    # Compute type/action coverage
    types = set()
    actions = set()
    for t in transitions:
        for o in t.before.objects:
            types.add(o.type_name)
        actions.add(t.action_id)

    print(f"Object types: {sorted(types)}")
    print(f"Actions: {sorted(actions)}")

    return {
        "benchmark": "minigrid",
        "env": env_name,
        "n_transitions": len(transitions),
        "object_types": sorted(types),
        "n_actions": len(actions),
    }


def eval_atari(game: str, obs_suffix: str) -> dict:
    from .poeworld_adapter import (
        load_poeworld_transitions,
        evaluate_transition_accuracy,
        GAME_TYPE_IDS,
        GAME_ACTIONS,
    )

    print(f"\n{'='*60}")
    print(f"Atari ({game}): loading observations with suffix '{obs_suffix}'")
    print(f"{'='*60}")

    try:
        transitions = load_poeworld_transitions(game, obs_suffix=obs_suffix)
    except FileNotFoundError as e:
        print(f"  ERROR: Could not load observations: {e}")
        print(f"  Run PoE-World's make_observations.py first to generate data.")
        return {"benchmark": game.lower(), "error": str(e)}

    print(f"Loaded {len(transitions)} transitions")

    # Compute type/action coverage
    types = set()
    actions = set()
    game_states = set()
    for t in transitions:
        for o in t.before.objects:
            types.add(o.type_name)
        actions.add(t.action_str)
        if t.game_state_before:
            game_states.add(t.game_state_before)

    print(f"Object types: {sorted(types)}")
    print(f"Actions used: {sorted(actions)}")
    print(f"Game states seen: {sorted(game_states)}")

    return {
        "benchmark": game.lower(),
        "n_transitions": len(transitions),
        "object_types": sorted(types),
        "actions_used": sorted(actions),
        "game_states": sorted(game_states),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object-centric agent across all benchmarks",
    )
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmark to evaluate on")
    parser.add_argument("--env", type=str, default=None,
                        help="MiniGrid env name (for minigrid benchmark)")
    parser.add_argument("--obs-suffix", type=str, default=None,
                        help="PoE-World observation suffix (for atari benchmarks)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:\n")
        for name, info in BENCHMARKS.items():
            print(f"  {name}")
            print(f"    Source: {info['source']}")
            print(f"    State: {info['state_type']}")
            print(f"    Perception: {info['perception']}")
            if "envs" in info:
                print(f"    Environments: {', '.join(info['envs'])}")
            print()
        return

    if args.benchmark is None:
        parser.error("--benchmark is required (or use --list)")

    if args.benchmark == "minigrid":
        env_name = args.env or "MiniGrid-DoorKey-5x5-v0"
        result = eval_minigrid(env_name, args.episodes, args.max_steps, args.seed)

    elif args.benchmark in ("pong", "montezuma"):
        info = BENCHMARKS[args.benchmark]
        obs_suffix = args.obs_suffix or info["obs_suffix"]
        result = eval_atari(info["game"], obs_suffix)

    elif args.benchmark == "arc-agi-3":
        print("ARC-AGI-3 evaluation uses the native agent pipeline.")
        print("Run: python -m src.object_centric_agent.agent_v2")
        result = {"benchmark": "arc-agi-3", "note": "use native pipeline"}

    else:
        parser.error(f"Unknown benchmark: {args.benchmark}")
        return

    # Print summary
    print(f"\n{'='*60}")
    print("Result:")
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
