#!/usr/bin/env python3
"""
Run a research agent against ARC-AGI-3 environments.

Uses the official arc-agi package (pip install arc-agi) for environment management.

Usage:
    # Run against the remote API (default):
    python run.py --agent oop --game ls20

    # Run in offline mode (local game files only, no API key needed):
    python run.py --agent oop --game ls20 --offline

    # Run all available games:
    python run.py --agent oop

    # List available environments:
    python run.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from typing import Type

from arcengine import GameAction, GameState, FrameDataRaw

logger = logging.getLogger()

# Map of friendly names -> agent module paths
AGENT_MAP = {
    "oop": "src.oop_agent.OOPAgent",
    "monolithic": "src.monolithic_agent.MonolithicAgent",
    "object_centric": "src.object_centric_agent.ObjectCentricAgent",
    "object_centric_v2": "src.object_centric_agent.ObjectCentricAgentV2",
}


def import_agent(name: str) -> Type:
    """Import and return an agent class by friendly name."""
    if name not in AGENT_MAP:
        raise ValueError(f"Unknown agent: {name}. Choose from: {list(AGENT_MAP.keys())}")
    module_path, class_name = AGENT_MAP[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run_agent_on_env(agent, env, game_id: str) -> None:
    """Run an agent on a single environment until done or max actions."""
    obs = env.reset()
    if obs is None:
        logger.error(f"[{game_id}] Failed to reset environment")
        return

    while not agent.is_done(obs) and agent.action_counter <= agent.MAX_ACTIONS:
        action = agent.choose_action(obs)

        # Build data dict for complex actions (ACTION6 with x,y)
        data = None
        if hasattr(agent, "pending_action_data") and agent.pending_action_data is not None:
            data = agent.pending_action_data
            agent.pending_action_data = None
        elif action.is_complex() and hasattr(action, "action_data"):
            ad = action.action_data
            if hasattr(ad, "x") and hasattr(ad, "y"):
                data = {"x": ad.x, "y": ad.y}

        # Build reasoning dict
        reasoning = None
        if hasattr(action, "reasoning") and action.reasoning:
            reasoning = {"text": str(action.reasoning)}

        obs = env.step(action, data=data, reasoning=reasoning)
        if obs is None:
            logger.warning(f"[{game_id}] env.step returned None, resetting")
            obs = env.reset()
            if obs is None:
                logger.error(f"[{game_id}] Failed to reset after None step")
                return

        agent.action_counter += 1

        logger.info(
            f"[{game_id}] {action.name}: step {agent.action_counter}, "
            f"levels={obs.levels_completed}, state={obs.state.name}"
        )

    logger.info(
        f"[{game_id}] Agent finished after {agent.action_counter} actions "
        f"(state={obs.state.name}, levels={obs.levels_completed})"
    )


def run_agent(args) -> None:
    """Run an agent against ARC-AGI-3 environments using the arc-agi package."""
    from arc_agi import Arcade, OperationMode

    # Select operation mode
    if args.offline:
        mode = OperationMode.OFFLINE
    else:
        mode = OperationMode.NORMAL

    # Control run logging via environment variable
    if args.no_log:
        os.environ["ARC_DISABLE_RUN_LOGGING"] = "1"
    else:
        os.environ.pop("ARC_DISABLE_RUN_LOGGING", None)

    # Create Arcade instance
    arc = Arcade(
        operation_mode=mode,
        arc_api_key=os.getenv("ARC_API_KEY", ""),
    )

    # Get available environments
    envs = arc.get_environments()
    if not envs:
        logger.error("No environments available. Check your API key or operation mode.")
        sys.exit(1)

    available_ids = [e.game_id for e in envs]
    logger.info(f"Available environments: {available_ids}")

    # Filter games if requested
    if args.game:
        filters = args.game.split(",")
        game_ids = [g for g in available_ids if any(g.startswith(f) for f in filters)]
    else:
        game_ids = available_ids

    if not game_ids:
        logger.error(f"No matching games. Available: {available_ids}")
        sys.exit(1)

    # Import agent class
    AgentClass = import_agent(args.agent)
    agent_key = args.agent.lower()
    logger.info(f"Agent: {AgentClass.__name__} | Games: {game_ids} | Mode: {mode.name}")

    # Build agent kwargs from CLI args
    agent_kwargs = {}
    if args.results_dir:
        agent_kwargs["results_dir"] = args.results_dir

    # Run each game (sequentially for now; can be threaded later)
    for game_id in game_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting game: {game_id}")
        logger.info(f"{'='*60}")

        env = arc.make(
            game_id,
            save_recording=not args.no_recording,
            render_mode=args.render_mode,
        )
        if env is None:
            logger.error(f"Failed to create environment for {game_id}")
            continue

        agent = AgentClass(**agent_kwargs)

        try:
            run_agent_on_env(agent, env, game_id)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.exception(f"Error running agent on {game_id}: {e}")

    # Get and display scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        logger.info("\n--- FINAL SCORECARD ---")
        logger.info(f"Score: {scorecard.score}")
        logger.info(scorecard.model_dump_json(indent=2))

    arc.close_scorecard()


def list_environments(args) -> None:
    """List available environments."""
    from arc_agi import Arcade, OperationMode

    mode = OperationMode.OFFLINE if args.offline else OperationMode.NORMAL
    arc = Arcade(
        operation_mode=mode,
        arc_api_key=os.getenv("ARC_API_KEY", ""),
    )

    envs = arc.get_environments()
    if not envs:
        print("No environments available.")
        return

    print(f"\nAvailable environments ({len(envs)}):")
    for e in envs:
        tags = f" [{', '.join(e.tags)}]" if e.tags else ""
        title = f" - {e.title}" if e.title else ""
        print(f"  {e.game_id}{title}{tags}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="ARC-AGI-3 Research Agent Runner")
    parser.add_argument("-a", "--agent", choices=list(AGENT_MAP.keys()), help="Agent to run")
    parser.add_argument("-g", "--game", help="Game ID filter (comma-separated prefixes)")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (local games only)")
    parser.add_argument("--no-log", action="store_true", help="Disable detailed run logging to results/")
    parser.add_argument("--no-recording", action="store_true", help="Disable game recordings")
    parser.add_argument("--render-mode", choices=["terminal", "terminal-fast", "human"], help="Render mode")
    parser.add_argument("--results-dir", default="results", help="Directory for run results")
    parser.add_argument("--list", action="store_true", help="List available environments")
    args = parser.parse_args()

    if args.list:
        list_environments(args)
    elif args.agent:
        run_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
