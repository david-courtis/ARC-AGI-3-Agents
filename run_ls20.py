#!/usr/bin/env python3
"""
Run the object-centric v2 agent on ls20 with level-gating.

The agent will:
1. Explore level 1, build epistemic model, synthesize world model
2. If it accidentally completes a level before model is perfect → reset
3. Only advance when model is 100% accurate on observed transitions
4. Log every stage of the pipeline for inspection
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

os.environ["ARC_DISABLE_RUN_LOGGING"] = "1"  # avoid disk I/O noise

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState, FrameDataRaw
from src.object_centric_agent import ObjectCentricAgentV2


def main():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='./environment_files')
    os.environ["ARC_DISABLE_RUN_LOGGING"] = "1"

    envs = arc.get_environments()
    ls20 = [e for e in envs if 'ls20' in e.game_id]
    if not ls20:
        logger.error("ls20 not found in available environments")
        sys.exit(1)

    game_id = ls20[0].game_id
    logger.info(f"Target game: {game_id}")

    env = arc.make(game_id)
    if env is None:
        logger.error(f"Failed to create environment for {game_id}")
        sys.exit(1)

    agent = ObjectCentricAgentV2(
        synthesis_model="opus",
        synthesis_max_turns=30,
        verbose=True,
    )

    obs = env.reset()
    if obs is None:
        logger.error("Failed to reset environment")
        sys.exit(1)

    logger.info(f"Game started. Available actions: {obs.available_actions}")
    logger.info(f"Initial state: {obs.state.name}")

    step = 0
    while not agent.is_done(obs) and step < agent.MAX_ACTIONS:
        action = agent.choose_action(obs)

        # Build data for ACTION6
        data = None
        if hasattr(agent, "pending_action_data") and agent.pending_action_data is not None:
            data = agent.pending_action_data
            agent.pending_action_data = None

        obs = env.step(action, data=data)

        if obs is None:
            logger.warning("env.step returned None, resetting")
            obs = env.reset()
            if obs is None:
                logger.error("Failed to reset after None step")
                break

        agent.action_counter += 1
        step += 1

        # Periodic summary
        if step % 25 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP {step} SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"  State: {obs.state.name}, levels_completed: {obs.levels_completed}")
            logger.info(f"  Agent level: {agent._current_level_idx}")
            logger.info(f"  Transitions: {len(agent._level_state.replay_buffer)}")
            logger.info(f"  Synthesis attempts: {agent._level_state.synthesis_attempts}")
            logger.info(f"  Best accuracy: {agent._level_state.best_accuracy:.0%}")
            logger.info(f"  Model perfect: {agent._level_state.model_is_perfect}")
            if agent.explorer:
                logger.info(f"  Phase: {agent.explorer.phase.name}")
                logger.info(f"  Confidence: {agent.explorer.epistemic.mean_confidence():.0%}")
            logger.info(f"  Registry: {agent.frame_parser.get_registry_summary()}")
            logger.info(f"{'='*60}\n")

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("FINAL REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Total steps: {step}")
    logger.info(f"Final state: {obs.state.name}")
    logger.info(f"Levels completed: {obs.levels_completed}")
    logger.info(f"Levels mastered: {agent._total_levels_mastered}")
    logger.info(f"Level resets: {agent._level_state.level_completed_count}")
    logger.info(f"Best model accuracy: {agent._level_state.best_accuracy:.0%}")

    if agent.explorer:
        logger.info(f"\nFinal epistemic state:")
        logger.info(agent.explorer.get_epistemic_summary())

    logger.info(f"\nFinal registry:")
    logger.info(agent.frame_parser.get_registry_summary())

    if agent._synth_loop and agent._synth_loop.current_code:
        logger.info(f"\nSynthesized code ({len(agent._synth_loop.current_code)} chars):")
        logger.info(agent._synth_loop.current_code[:2000])

    # Close scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        logger.info(f"\nScorecard: {scorecard.score}")
    arc.close_scorecard()


if __name__ == "__main__":
    main()
