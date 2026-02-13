# Local ARC-AGI-3 Simulator

A framework for implementing local versions of ARC-AGI-3 games, allowing you to develop and test agents without hitting the official API.

## Architecture

```
local_simulator/
├── core/
│   ├── base_game.py     # BaseGame, GameLevel, SuccessCondition
│   ├── game_object.py   # GameObject hierarchy (Player, Wall, Goal, etc.)
│   └── renderer.py      # 64x64 RGB frame rendering
├── games/
│   └── simple_maze.py   # Example game implementation
└── mock_server.py       # Flask API server (drop-in replacement)
```

## Key Concepts

Each game has:
- **Sub-levels**: Stages/puzzles within the game (`GameLevel`)
- **Success conditions**: Per-level win criteria (`SuccessCondition`)
- **Valid actions**: Which `GameAction`s the game supports
- **Game objects**: Entities within each level (`GameObject`)

## Quick Start

### 1. Run the Mock Server

```bash
cd ARC-AGI-3-Agents
python -m local_simulator.mock_server --port 8001
```

### 2. Point Your Agent to Localhost

```bash
export SCHEME=http
export HOST=localhost
export PORT=8001
uv run main.py --agent=random --game=simple_maze
```

## Creating a New Game

1. Create a new file in `local_simulator/games/`:

```python
from ..core.base_game import BaseGame, GameLevel, SuccessCondition
from ..core.game_object import Player, Wall, Goal, Position
from agents.structs import GameAction

class MyGame(BaseGame):
    def __init__(self):
        super().__init__(game_id="my_game")
    
    def setup_levels(self) -> None:
        level = GameLevel(level_id="1", name="Level 1")
        level.add_object(Player(Position(5, 5)))
        level.add_object(Goal(Position(10, 10)))
        level.success_conditions.append(
            SuccessCondition("win", lambda l: check_win(l))
        )
        self.levels.append(level)
    
    def get_valid_actions(self) -> list[GameAction]:
        return [GameAction.RESET, GameAction.ACTION1, ...]
    
    def handle_action(self, action: GameAction) -> None:
        # Your game logic here
        pass
    
    def render_frame(self) -> list[list[list[int]]]:
        return self.renderer.render_objects(self.current_level.objects)
```

2. Register it in `mock_server.py`:

```python
from local_simulator.games.my_game import MyGame

GAME_REGISTRY = {
    "my_game": MyGame,
    # ...
}
```

## Class Reference

### BaseGame
Abstract base for all games. Implement:
- `setup_levels()` - Create GameLevel instances
- `get_valid_actions()` - Return valid GameActions
- `handle_action(action)` - Process actions
- `render_frame()` - Return 64x64 RGB grid

### GameLevel  
Container for a sub-level:
- `objects: list[GameObject]` - Entities in the level
- `success_conditions` - Win conditions
- `failure_conditions` - Lose conditions  
- `score` - Current score

### GameObject
Base for game entities:
- `Player` - Player-controlled entity
- `Wall` - Solid obstacle
- `Floor` - Passable ground
- `Goal` - Win target
- Extend `DynamicObject` for moving entities
