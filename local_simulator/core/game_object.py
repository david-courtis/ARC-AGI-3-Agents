"""Game Objects - entities that exist within a game level."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ObjectType(Enum):
    STATIC = auto()
    DYNAMIC = auto()
    INTERACTIVE = auto()
    PLAYER = auto()


@dataclass
class Position:
    """2D position in the game grid."""
    x: int
    y: int
    
    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def in_bounds(self, width: int, height: int) -> bool:
        return 0 <= self.x < width and 0 <= self.y < height


@dataclass
class GameObject(ABC):
    """Base class for all game objects."""
    position: Position
    object_type: ObjectType
    color: tuple[int, int, int] = (255, 255, 255)
    size: tuple[int, int] = (1, 1)
    solid: bool = True
    visible: bool = True
    name: str = ""
    properties: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__
    
    @abstractmethod
    def update(self, game_state) -> None:
        pass
    
    @abstractmethod
    def on_interact(self, actor: "GameObject", game_state) -> bool:
        pass
    
    def render(self) -> list[tuple[Position, tuple[int, int, int]]]:
        if not self.visible:
            return []
        cells = []
        for dx in range(self.size[0]):
            for dy in range(self.size[1]):
                pos = Position(self.position.x + dx, self.position.y + dy)
                cells.append((pos, self.color))
        return cells
    
    def collides_with(self, other: "GameObject") -> bool:
        for dx in range(self.size[0]):
            for dy in range(self.size[1]):
                my_pos = Position(self.position.x + dx, self.position.y + dy)
                for ox in range(other.size[0]):
                    for oy in range(other.size[1]):
                        other_pos = Position(other.position.x + ox, other.position.y + oy)
                        if my_pos == other_pos:
                            return True
        return False


class StaticObject(GameObject):
    def __init__(self, position: Position, color: tuple[int, int, int], **kwargs):
        super().__init__(position=position, object_type=ObjectType.STATIC, color=color, **kwargs)
    
    def update(self, game_state) -> None:
        pass
    
    def on_interact(self, actor, game_state) -> bool:
        return False


class Wall(StaticObject):
    def __init__(self, position: Position, color: tuple[int, int, int] = (100, 100, 100)):
        super().__init__(position=position, color=color, solid=True, name="Wall")


class Floor(StaticObject):
    def __init__(self, position: Position, color: tuple[int, int, int] = (50, 50, 50)):
        super().__init__(position=position, color=color, solid=False, name="Floor")


class Goal(StaticObject):
    def __init__(self, position: Position, color: tuple[int, int, int] = (0, 255, 0)):
        super().__init__(position=position, color=color, solid=False, name="Goal")
    
    def on_interact(self, actor, game_state) -> bool:
        if actor.object_type == ObjectType.PLAYER:
            game_state.trigger_event("goal_reached", actor=actor, goal=self)
            return True
        return False


class DynamicObject(GameObject):
    velocity: Position = field(default_factory=lambda: Position(0, 0))
    
    def __init__(self, position: Position, color: tuple[int, int, int], **kwargs):
        super().__init__(position=position, object_type=ObjectType.DYNAMIC, color=color, **kwargs)
        self.velocity = Position(0, 0)
    
    def update(self, game_state) -> None:
        new_pos = self.position + self.velocity
        if new_pos.in_bounds(game_state.width, game_state.height):
            if not game_state.is_blocked(new_pos, exclude=self):
                self.position = new_pos
    
    def on_interact(self, actor, game_state) -> bool:
        return False
    
    def move(self, direction: Position, game_state) -> bool:
        new_pos = self.position + direction
        if new_pos.in_bounds(game_state.width, game_state.height):
            if not game_state.is_blocked(new_pos, exclude=self):
                self.position = new_pos
                return True
        return False


class Player(DynamicObject):
    def __init__(self, position: Position, color: tuple[int, int, int] = (0, 100, 255)):
        super().__init__(position=position, color=color, name="Player")
        self.object_type = ObjectType.PLAYER
