"""Ice Sliding Puzzle Game - Player slides on ice until hitting an obstacle."""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from local_simulator.core.base_game import (
    BaseGame, GameLevel, SuccessCondition, LevelState
)
from local_simulator.core.game_object import GameObject, Position
from agents.structs import FrameData, GameAction, GameState as AgentGameState


COLOR_BACKGROUND = [180, 140, 200]
COLOR_OBSTACLE = [40, 40, 40]
COLOR_GOAL = [255, 255, 255]
COLOR_PLAYER = [255, 60, 60]
COLOR_BORDER = [100, 200, 100]
COLOR_BORDER_CORNER = [40, 40, 40]


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class IceSlidingPuzzle(BaseGame):
    """
    Ice sliding puzzle - player slides until hitting obstacle.
    
    Grid format (14x14 playable area):
        . = empty space
        X = obstacle (black)
        P = player start
        G = goal wall (white, blocks movement)
        R = receptacle (win spot)
    """
    
    GRID_SIZE = 16
    CELL_SIZE = 4
    
    def __init__(self):
        self.player_pos: tuple[int, int] = (0, 0)
        self.obstacle_grid: list[list[bool]] = []
        self.goal_grid: list[list[bool]] = []
        self.receptacle_pos: tuple[int, int] = (0, 0)
        self.last_direction: Optional[Direction] = None
        self.pending_frames: List[FrameData] = []
        self._level_grids: list[tuple[str, str, int]] = []  # (id, grid, score)
        super().__init__("ice_sliding")
    
    def setup_levels(self) -> None:
        self.levels = [
            self._create_level("1", "Tutorial", """
                ......XX..X...
                .X.....XXX....
                ..XX..XXPX....
                ..XXX.........
                .XXXXX........
                ..XX.....X....
                .......XXX....
                .......XXX....
                ........X.X...
                ...GRG........
                ...GGG........
                ..............
                ..............
                ..............
            """, 100),
            
            self._create_level("2", "Open Field", """
                ..............
                ..X.......X...
                ..X...P...X...
                ..X.......X...
                ..............
                ......X.......
                ......X.......
                ..............
                ..X.......X...
                ..X..GGG..X...
                ..X..GRG..X...
                ..............
                ..............
                ..............
            """, 150),
            
            self._create_level("3", "Maze", """
                ..............
                .XXXX....XXXX.
                .X........P.X.
                .X.XXXXXXXX.X.
                .X..........X.
                .XXXXXXXXXX.X.
                ............X.
                .XXXXXXXXXXXX.
                ..............
                .GGG..........
                .GRG..........
                ..............
                ..............
                ..............
            """, 200),
        ]
    
    def _create_level(self, level_id: str, name: str, grid: str, score: int) -> GameLevel:
        level = GameLevel(level_id=level_id, name=name, max_score=score)
        level.properties['grid'] = grid
        level.properties['score'] = score
        level.success_conditions.append(
            SuccessCondition(name="reach_goal", check_fn=lambda _: self._check_win())
        )
        level.score = score
        return level
    
    def _load_level(self, level: GameLevel) -> None:
        grid = level.properties.get('grid', '')
        data = self._parse_grid(grid)
        
        self.player_pos = data['player_start']
        self.receptacle_pos = data['receptacle_pos']
        
        self.obstacle_grid = [[False] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        self.goal_grid = [[False] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        
        for i in range(self.GRID_SIZE):
            self.obstacle_grid[0][i] = True
            self.obstacle_grid[15][i] = True
            self.obstacle_grid[i][0] = True
            self.obstacle_grid[i][15] = True
        
        for ox, oy in data['obstacles']:
            if 0 <= ox < self.GRID_SIZE and 0 <= oy < self.GRID_SIZE:
                self.obstacle_grid[oy][ox] = True
        
        for gx, gy in data['goal_walls']:
            if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
                self.goal_grid[gy][gx] = True
        
        self.last_direction = None
        self.pending_frames = []
    
    def _parse_grid(self, grid_string: str) -> dict:
        lines = [line.strip() for line in grid_string.strip().split('\n') if line.strip()]
        while len(lines) < 14:
            lines.append('.' * 14)
        
        player_start = None
        receptacle_pos = None
        goal_walls = []
        obstacles = []
        
        for y, line in enumerate(lines[:14]):
            line = line.ljust(14, '.')[:14]
            for x, char in enumerate(line):
                grid_x, grid_y = x + 1, y + 1
                if char == 'P':
                    player_start = (grid_x, grid_y)
                elif char == 'G':
                    goal_walls.append((grid_x, grid_y))
                elif char == 'R':
                    receptacle_pos = (grid_x, grid_y)
                elif char == 'X':
                    obstacles.append((grid_x, grid_y))
        
        if player_start is None:
            player_start = (7, 7)
        if receptacle_pos is None:
            receptacle_pos = (7, 12)
            if not goal_walls:
                goal_walls = [(6, 11), (7, 11), (8, 11), (6, 12), (8, 12)]
        
        return {
            'player_start': player_start,
            'receptacle_pos': receptacle_pos,
            'goal_walls': goal_walls,
            'obstacles': obstacles,
        }
    
    def reset(self, card_id: str = "") -> FrameData:
        import uuid
        self.guid = str(uuid.uuid4())
        self.current_level_index = 0
        self.total_score = 0
        self.action_count = 0
        self.game_state = AgentGameState.NOT_FINISHED
        self.setup_levels()
        
        if self.current_level:
            self.current_level.state = LevelState.IN_PROGRESS
            self._load_level(self.current_level)
        
        return self._create_frame_data()
    
    def get_valid_actions(self) -> list[GameAction]:
        return [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    
    def handle_action(self, action: GameAction) -> None:
        direction_map = {
            GameAction.ACTION1: Direction.UP,
            GameAction.ACTION2: Direction.DOWN,
            GameAction.ACTION3: Direction.LEFT,
            GameAction.ACTION4: Direction.RIGHT,
        }
        if action in direction_map:
            self.last_direction = direction_map[action]
            self._slide(self.last_direction)
    
    def _slide(self, direction: Direction) -> None:
        dx, dy = direction.value
        x, y = self.player_pos
        start_x, start_y = x, y
        
        self.pending_frames = []
        path = [(x, y)]
        visited = {(x, y)}
        
        while True:
            next_raw_x = x + dx
            next_raw_y = y + dy
            next_x = ((next_raw_x - 1) % 14) + 1
            next_y = ((next_raw_y - 1) % 14) + 1
            
            if self._is_blocked(next_x, next_y):
                break
            
            if (next_x, next_y) in visited:
                x, y = start_x, start_y
                if path[-1] != (x, y):
                    path.append((x, y))
                break
            
            if (next_x, next_y) == self.receptacle_pos:
                x, y = next_x, next_y
                path.append((x, y))
                break
            
            x, y = next_x, next_y
            path.append((x, y))
            visited.add((x, y))
        
        for pos in path[1:]:
            self.player_pos = pos
            self.pending_frames.append(self._create_frame_data())
        
        if path:
            self.player_pos = path[-1]
    
    def step(self, action: GameAction) -> FrameData:
        self.action_count += 1
        self.handle_action(action)
        
        if self.current_level:
            self.current_level.update()
            if self.current_level.check_failure():
                self.current_level.state = LevelState.FAILED
                self.game_state = AgentGameState.GAME_OVER
            elif self.current_level.check_success():
                self._advance_level()
        
        return self._create_frame_data()
    
    def _advance_level(self):
        self.current_level.state = LevelState.COMPLETED
        self.total_score += self.current_level.score
        
        if self.current_level_index + 1 < len(self.levels):
            self.current_level_index += 1
            self.levels[self.current_level_index].state = LevelState.IN_PROGRESS
            self._load_level(self.levels[self.current_level_index])
        else:
            self.game_state = AgentGameState.WIN
    
    def get_pending_frames(self) -> List[FrameData]:
        frames = self.pending_frames.copy()
        self.pending_frames = []
        return frames
    
    def get_all_slide_frames(self, action: GameAction) -> List[FrameData]:
        self.pending_frames = []
        self.action_count += 1
        self.handle_action(action)
        
        intermediate_frames = self.pending_frames.copy()
        self.pending_frames = []
        
        if self.current_level:
            self.current_level.update()
            if self.current_level.check_failure():
                self.current_level.state = LevelState.FAILED
                self.game_state = AgentGameState.GAME_OVER
            elif self.current_level.check_success():
                self._advance_level()
        
        if not intermediate_frames:
            intermediate_frames = [self._create_frame_data()]
        else:
            intermediate_frames[-1] = self._create_frame_data()
        
        return intermediate_frames
    
    def _is_blocked(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.GRID_SIZE or y < 0 or y >= self.GRID_SIZE:
            return True
        return self.obstacle_grid[y][x] or self.goal_grid[y][x]
    
    def _check_win(self) -> bool:
        return self.player_pos == self.receptacle_pos
    
    def render_frame(self) -> list[list[list[int]]]:
        frame = [[list(COLOR_BACKGROUND) for _ in range(64)] for _ in range(64)]
        self._draw_border(frame)
        
        for y in range(1, 15):
            for x in range(1, 15):
                if self.obstacle_grid[y][x]:
                    self._fill_cell(frame, x, y, COLOR_OBSTACLE)
        
        for y in range(1, 15):
            for x in range(1, 15):
                if self.goal_grid[y][x]:
                    self._fill_cell(frame, x, y, COLOR_GOAL)
        
        px, py = self.player_pos
        self._fill_cell(frame, px, py, COLOR_PLAYER)
        return frame
    
    def _draw_border(self, frame: list[list[list[int]]]) -> None:
        for x in range(64):
            for y in range(4):
                frame[y][x] = list(COLOR_BORDER)
        for x in range(64):
            for y in range(60, 64):
                frame[y][x] = list(COLOR_BORDER)
        for y in range(64):
            for x in range(4):
                frame[y][x] = list(COLOR_BORDER)
        for y in range(64):
            for x in range(60, 64):
                frame[y][x] = list(COLOR_BORDER)
        
        for corner_y, corner_x in [(0, 0), (0, 60), (60, 0), (60, 60)]:
            for dy in range(4):
                for dx in range(4):
                    frame[corner_y + dy][corner_x + dx] = list(COLOR_BORDER_CORNER)
        
        if self.last_direction:
            highlight = [100, 255, 100]
            if self.last_direction == Direction.UP:
                for x in range(4, 60):
                    for y in range(4):
                        frame[y][x] = list(highlight)
            elif self.last_direction == Direction.DOWN:
                for x in range(4, 60):
                    for y in range(60, 64):
                        frame[y][x] = list(highlight)
            elif self.last_direction == Direction.LEFT:
                for y in range(4, 60):
                    for x in range(4):
                        frame[y][x] = list(highlight)
            elif self.last_direction == Direction.RIGHT:
                for y in range(4, 60):
                    for x in range(60, 64):
                        frame[y][x] = list(highlight)
    
    def _fill_cell(self, frame: list[list[list[int]]], logical_x: int, logical_y: int, color: list[int]) -> None:
        start_x = logical_x * self.CELL_SIZE
        start_y = logical_y * self.CELL_SIZE
        for dy in range(self.CELL_SIZE):
            for dx in range(self.CELL_SIZE):
                px, py = start_x + dx, start_y + dy
                if 0 <= px < 64 and 0 <= py < 64:
                    frame[py][px] = list(color)


if __name__ == "__main__":
    game = IceSlidingPuzzle()
    frame = game.reset()
    print(f"Levels: {game.get_level_count()}")
    print(f"Player: {game.player_pos}, Receptacle: {game.receptacle_pos}")
    
    frames = game.get_all_slide_frames(GameAction.ACTION2)
    print(f"Slide DOWN: {len(frames)} frames")
