"""Base Game - abstract framework for local ARC-AGI-3 games."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.structs import FrameData, GameAction, GameState as AgentGameState
from .game_object import GameObject, Position


class LevelState(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SuccessCondition:
    """Condition that must be met to complete a level."""
    name: str
    check_fn: Callable[["GameLevel"], bool]
    description: str = ""
    required: bool = True
    
    def is_met(self, level: "GameLevel") -> bool:
        return self.check_fn(level)


@dataclass 
class GameLevel:
    """A single sub-level/stage within a game."""
    level_id: str
    name: str
    width: int = 64
    height: int = 64
    objects: list[GameObject] = field(default_factory=list)
    success_conditions: list[SuccessCondition] = field(default_factory=list)
    failure_conditions: list[SuccessCondition] = field(default_factory=list)
    state: LevelState = LevelState.NOT_STARTED
    score: int = 0
    max_score: int = 100
    events: dict[str, list[Callable]] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    
    def add_object(self, obj: GameObject) -> None:
        self.objects.append(obj)
    
    def remove_object(self, obj: GameObject) -> None:
        if obj in self.objects:
            self.objects.remove(obj)
    
    def get_objects_at(self, pos: Position) -> list[GameObject]:
        result = []
        for obj in self.objects:
            for dx in range(obj.size[0]):
                for dy in range(obj.size[1]):
                    if Position(obj.position.x + dx, obj.position.y + dy) == pos:
                        result.append(obj)
                        break
        return result
    
    def get_objects_by_type(self, obj_type: type) -> list[GameObject]:
        return [obj for obj in self.objects if isinstance(obj, obj_type)]
    
    def is_blocked(self, pos: Position, exclude: Optional[GameObject] = None) -> bool:
        for obj in self.objects:
            if obj is exclude or not obj.solid:
                continue
            for dx in range(obj.size[0]):
                for dy in range(obj.size[1]):
                    if Position(obj.position.x + dx, obj.position.y + dy) == pos:
                        return True
        return False
    
    def trigger_event(self, event_name: str, **kwargs) -> None:
        if event_name in self.events:
            for handler in self.events[event_name]:
                handler(**kwargs)
    
    def on_event(self, event_name: str, handler: Callable) -> None:
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(handler)
    
    def update(self) -> None:
        for obj in self.objects:
            obj.update(self)
    
    def check_success(self) -> bool:
        return all(cond.is_met(self) for cond in self.success_conditions if cond.required)
    
    def check_failure(self) -> bool:
        return any(cond.is_met(self) for cond in self.failure_conditions)
    
    def add_score(self, points: int) -> None:
        self.score = min(self.score + points, self.max_score)


class BaseGame(ABC):
    """Abstract base class for local ARC-AGI-3 games."""
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.levels: list[GameLevel] = []
        self.current_level_index: int = 0
        self.game_state: AgentGameState = AgentGameState.NOT_PLAYED
        self.total_score: int = 0
        self.action_count: int = 0
        self.guid: Optional[str] = None
        self.setup_levels()
    
    @property
    def current_level(self) -> Optional[GameLevel]:
        if 0 <= self.current_level_index < len(self.levels):
            return self.levels[self.current_level_index]
        return None
    
    @abstractmethod
    def setup_levels(self) -> None:
        """Initialize all levels for this game."""
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> list[GameAction]:
        """Return list of valid GameActions."""
        pass
    
    @abstractmethod
    def handle_action(self, action: GameAction) -> None:
        """Process action and update game state."""
        pass
    
    @abstractmethod
    def render_frame(self) -> list[list[list[int]]]:
        """Render current state to 64x64 RGB grid."""
        pass
    
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
        return self._create_frame_data()
    
    def step(self, action: GameAction) -> FrameData:
        self.action_count += 1
        self.handle_action(action)
        
        if self.current_level:
            self.current_level.update()
            if self.current_level.check_failure():
                self.current_level.state = LevelState.FAILED
                self.game_state = AgentGameState.GAME_OVER
            elif self.current_level.check_success():
                self.current_level.state = LevelState.COMPLETED
                self.total_score += self.current_level.score
                if self.current_level_index + 1 < len(self.levels):
                    self.current_level_index += 1
                    self.levels[self.current_level_index].state = LevelState.IN_PROGRESS
                else:
                    self.game_state = AgentGameState.WIN
        
        return self._create_frame_data()
    
    def _create_frame_data(self) -> FrameData:
        return FrameData(
            game_id=self.game_id,
            frame=self.render_frame(),
            state=self.game_state,
            score=min(self.total_score + (self.current_level.score if self.current_level else 0), 254),
            guid=self.guid,
            available_actions=self.get_valid_actions(),
        )
    
    def get_level_count(self) -> int:
        return len(self.levels)
    
    def get_completed_levels(self) -> int:
        return sum(1 for level in self.levels if level.state == LevelState.COMPLETED)
