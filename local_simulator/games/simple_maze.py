"""Simple Maze Game - Navigate through mazes to reach goals."""

from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.structs import GameAction
from ..core.base_game import BaseGame, GameLevel, SuccessCondition, LevelState
from ..core.game_object import GameObject, Position, Player, Wall, Floor, Goal, ObjectType
from ..core.renderer import Renderer


class SimpleMaze(BaseGame):
    """Navigate through mazes to reach the goal. Actions: UP/DOWN/LEFT/RIGHT."""
    
    ACTION_MAP = {
        GameAction.ACTION1: Position(0, -1),
        GameAction.ACTION2: Position(0, 1),
        GameAction.ACTION3: Position(-1, 0),
        GameAction.ACTION4: Position(1, 0),
    }
    
    def __init__(self):
        self.renderer = Renderer(width=64, height=64, background_color=(20, 20, 20))
        self.player: Optional[Player] = None
        super().__init__(game_id="simple_maze")
    
    def setup_levels(self) -> None:
        self.levels = [
            self._create_level("1", "Tutorial", [
                "##########",
                "#P       #",
                "#        #",
                "#        #",
                "#       G#",
                "##########",
            ], 10),
            self._create_level("2", "Simple Maze", [
                "############",
                "#P    #    #",
                "# ### # ## #",
                "# #      # #",
                "# # #### # #",
                "#   #    # #",
                "# ### ## # #",
                "#        #G#",
                "############",
            ], 20),
            self._create_level("3", "Challenge", [
                "################",
                "#P #   #   #   #",
                "# ## # # # # # #",
                "#    # # #   # #",
                "#### # ### ### #",
                "#    #     #   #",
                "# #### ### # ###",
                "#      #   #  G#",
                "################",
            ], 30),
        ]
    
    def _create_level(self, level_id: str, name: str, maze: list[str], max_score: int) -> GameLevel:
        level = GameLevel(level_id=level_id, name=name, max_score=max_score)
        
        for y, row in enumerate(maze):
            for x, char in enumerate(row):
                pos = Position(x, y)
                if char == '#':
                    level.add_object(Wall(pos))
                elif char == 'P':
                    level.add_object(Player(pos))
                    level.add_object(Floor(pos))
                elif char == 'G':
                    level.add_object(Goal(pos))
                elif char == ' ':
                    level.add_object(Floor(pos))
        
        def player_at_goal(lvl: GameLevel) -> bool:
            players = lvl.get_objects_by_type(Player)
            goals = lvl.get_objects_by_type(Goal)
            if not players or not goals:
                return False
            return any(players[0].position == goal.position for goal in goals)
        
        level.success_conditions.append(SuccessCondition(name="reach_goal", check_fn=player_at_goal))
        level.score = max_score
        return level
    
    def get_valid_actions(self) -> list[GameAction]:
        return [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    
    def handle_action(self, action: GameAction) -> None:
        if action not in self.ACTION_MAP:
            return
        
        level = self.current_level
        if not level:
            return
        
        players = level.get_objects_by_type(Player)
        if not players:
            return
        
        player = players[0]
        direction = self.ACTION_MAP[action]
        new_pos = player.position + direction
        
        if not new_pos.in_bounds(level.width, level.height):
            return
        if level.is_blocked(new_pos, exclude=player):
            return
        
        player.position = new_pos
    
    def render_frame(self) -> list[list[list[int]]]:
        frame = self.renderer.create_empty_frame()
        level = self.current_level
        if not level:
            return frame
        
        def sort_key(obj: GameObject) -> int:
            if isinstance(obj, Floor): return 0
            if isinstance(obj, Wall): return 1
            if isinstance(obj, Goal): return 2
            if isinstance(obj, Player): return 3
            return 1
        
        sorted_objects = sorted(level.objects, key=sort_key)
        frame = self.renderer.render_objects(sorted_objects, frame)
        
        for i, char in enumerate(level.level_id):
            self.renderer.draw_text_simple(frame, 58 + i * 4, 2, char, (150, 150, 150))
        
        return frame


__all__ = ["SimpleMaze"]
