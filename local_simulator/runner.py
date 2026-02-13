"""Local runner for agents on simulated games."""

import logging
import time
from typing import Type, Any

from agents.structs import FrameData, GameAction, GameState
from local_simulator.core.base_game import BaseGame

logger = logging.getLogger(__name__)


class LocalRunner:
    """Runs an agent against a local game simulation."""

    def __init__(self, game: BaseGame, max_actions: int = 80, verbose: bool = True):
        self.game = game
        self.max_actions = max_actions
        self.verbose = verbose
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.timer = 0.0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run_agent_class(self, agent_class: Type, **agent_kwargs: Any) -> dict:
        """Run an agent class against the local game. Returns results dict."""
        self.timer = time.time()
        self.frames = []
        self.action_counter = 0
        
        initial_frame = FrameData(
            game_id=self.game.game_id,
            frame=[],
            state=GameState.NOT_PLAYED,
            score=0,
            available_actions=[GameAction.RESET],
        )
        self.frames.append(initial_frame)
        
        agent = self._create_agent_instance(agent_class, **agent_kwargs)
        
        self.log(f"\n{'='*50}")
        self.log(f"Running {agent.__class__.__name__} on {self.game.game_id}")
        self.log(f"{'='*50}")
        
        while not agent.is_done(self.frames, self.frames[-1]) and self.action_counter < self.max_actions:
            action = agent.choose_action(self.frames, self.frames[-1])
            frame = self._execute_action(action)
            self.frames.append(frame)
            
            self.log(
                f"  [{self.action_counter:3d}] {action.name:8s} → "
                f"state={frame.state.name}, score={frame.score}, "
                f"level={self.game.current_level_index + 1}/{self.game.get_level_count()}"
            )
            
            self.action_counter += 1
            
            if frame.state == GameState.WIN:
                self.log(f"\n🎉 WIN! Final score: {frame.score}")
                break

        elapsed = time.time() - self.timer
        fps = self.action_counter / max(elapsed, 0.01)
        
        result = {
            "game_id": self.game.game_id,
            "agent": agent.__class__.__name__,
            "final_state": self.frames[-1].state,
            "final_score": self.frames[-1].score,
            "actions_taken": self.action_counter,
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(fps, 2),
            "levels_completed": self.game.get_completed_levels(),
            "total_levels": self.game.get_level_count(),
        }
        
        self.log(f"\n{'='*50}")
        self.log(f"Result: {result['final_state'].name}")
        self.log(f"Score: {result['final_score']}, Actions: {result['actions_taken']}")
        self.log(f"Levels: {result['levels_completed']}/{result['total_levels']}")
        self.log(f"Time: {result['elapsed_seconds']}s ({result['fps']} fps)")
        self.log(f"{'='*50}\n")
        
        return result

    def _create_agent_instance(self, agent_class: Type, **kwargs: Any):
        """Create agent instance without HTTP dependencies."""
        class LocalAgent(agent_class):
            def __init__(self, game_id: str, **kw):
                self.game_id = game_id
                self.frames = [FrameData(score=0)]
                self.action_counter = 0
                self.timer = 0
                self.tags = []
                import random
                seed = int(time.time() * 1000000) + hash(game_id) % 1000000
                random.seed(seed)
        
        return LocalAgent(game_id=self.game.game_id, **kwargs)

    def _execute_action(self, action: GameAction) -> FrameData:
        if action == GameAction.RESET:
            return self.game.reset()
        return self.game.step(action)


def run_random_agent_on_game(game: BaseGame, max_actions: int = 80) -> dict:
    """Run the Random agent on a game."""
    from agents.templates.random_agent import Random
    runner = LocalRunner(game, max_actions=max_actions)
    return runner.run_agent_class(Random)


if __name__ == "__main__":
    import argparse
    from local_simulator.games.simple_maze import SimpleMaze
    from local_simulator.games.ice_sliding import IceSlidingPuzzle
    
    parser = argparse.ArgumentParser(description='Run local simulator')
    parser.add_argument('--game', type=str, choices=['maze', 'ice'], default='maze')
    parser.add_argument('--actions', type=int, default=200)
    args = parser.parse_args()
    
    game = IceSlidingPuzzle() if args.game == 'ice' else SimpleMaze()
    run_random_agent_on_game(game, max_actions=args.actions)
