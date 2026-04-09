"""
Object-Centric World Model: objects drive their own transitions.

Architecture
------------
This is a variant of the OOP agent where the locus of control is inverted.
In the action-centric OOP agent, the Action.apply() method is the driver:
it knows about object types, selects affected objects, and orchestrates
their updates. In this object-centric variant, each GameObject is the
driver: after an action is broadcast, every object independently decides
how it responds.

The transition decomposes as:

    1. perceive(frame)              -> World (frame parsed into typed objects)
    2. for each object in world:
         object.respond(action_id, world)  <- OBJECT drives the transition
    3. world.render()               -> predicted_frame

Compare with the action-centric OOP agent:

    1. perceive(frame)              -> World
    2. action = get_action(id)
       action.apply(world)          <- ACTION drives the transition
         (internally calls object.respond_to_action)
    3. world.render()               -> predicted_frame

The structural difference
-------------------------
Where the logic lives changes everything about what the LLM synthesizes:

Action-centric (oop_agent/):
    - Action classes are heavy: MoveUp.apply() knows about Player, Wall, Block
    - Object classes are light: Player.respond_to_action checks action type
    - Cross-object coordination is explicit in Action.apply()
    - Adding a new action = writing a new Action class
    - Adding a new object type = modifying every Action class

Object-centric (this agent):
    - Object classes are heavy: Player.respond() handles all 5 actions
    - Actions are just signals: no apply() method, just an ID
    - Cross-object coordination is implicit: each object queries the world
    - Adding a new object type = writing a new GameObject class
    - Adding a new action = modifying every GameObject class

The tradeoff: action-centric is better when actions have complex multi-object
coordination (push chains, cascading effects). Object-centric is better when
objects have rich individual behavior and actions are simple signals.

See docs/action-vs-object-centric.md for the full comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# =============================================================================
# Core object-centric base classes
# =============================================================================


class GameObject(ABC):
    """
    Base class for all game objects. THE primary locus of transition logic.

    In the object-centric model, each object is responsible for its own
    state transitions. When an action occurs, every object's respond()
    method is called. Each object independently decides:
    - Does this action affect me?
    - If so, how does my state change?
    - What do I need to query from other objects to decide?

    The LLM synthesizes subclasses that encode all per-type behavior:

        class Player(GameObject):
            def respond(self, action_id, world):
                if action_id == 1:    # MoveUp
                    target = (self.row - 1, self.col)
                    if not world.is_occupied(target):
                        self.row -= 1
                elif action_id == 2:  # MoveDown
                    ...

        class Wall(GameObject):
            def respond(self, action_id, world):
                pass  # walls never change

        class PushableBlock(GameObject):
            def respond(self, action_id, world):
                # Check if something pushed into me
                pusher = world.get_adjacent(self, direction=opposite(action_id))
                if pusher and isinstance(pusher, Player):
                    # I was pushed: move in the push direction
                    ...

    Note the inversion: in the action-centric model, the Action decides
    which objects to move. Here, the Block checks whether it was pushed.
    Each object is an autonomous agent that reacts to the action signal.
    """

    def __init__(self, obj_id: str, **properties):
        self.obj_id = obj_id
        for key, value in properties.items():
            setattr(self, key, value)

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def respond(self, action_id: int, world: World) -> None:
        """
        Update this object's state in response to an action.

        This is the core method. Each object type implements its own
        response to every action. The method receives the raw action_id
        (not an Action object) because actions are just signals here.

        The object can query world to inspect other objects, check
        spatial relationships, etc. But it mutates only itself.

        Args:
            action_id: Integer action ID (1-5). The object decides
                       what each ID means for its type.
            world: The current world state (for querying, not mutating
                   other objects).
        """
        pass

    @abstractmethod
    def render(self, frame: np.ndarray) -> None:
        """Write this object's pixels onto the frame (mutated in-place)."""
        pass

    def __repr__(self) -> str:
        props = {k: v for k, v in self.__dict__.items()
                 if k != "obj_id" and not k.startswith("_")}
        return f"{self.type_name}(id={self.obj_id}, {props})"


class World:
    """
    The world state: a collection of typed objects plus the raw frame.

    Same as in the action-centric OOP agent. The World is a passive
    data structure that objects query during their respond() calls.
    """

    def __init__(self, frame: np.ndarray, objects: list[GameObject]):
        self.frame = frame.copy()
        self.objects = list(objects)
        self.height = frame.shape[0]
        self.width = frame.shape[1]

    def get_objects_of_type(self, cls: type) -> list[GameObject]:
        return [o for o in self.objects if isinstance(o, cls)]

    def get_by_id(self, obj_id: str) -> GameObject | None:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None

    def add_object(self, obj: GameObject) -> None:
        self.objects.append(obj)

    def remove_object(self, obj: GameObject) -> None:
        self.objects = [o for o in self.objects if o is not obj]

    def render(self, background: np.ndarray | None = None) -> np.ndarray:
        if background is not None:
            frame = background.copy()
        else:
            frame = self.frame.copy()
        for obj in self.objects:
            obj.render(frame)
        return frame


class Domain(ABC):
    """
    Domain specification for the object-centric model.

    The key difference from the action-centric Domain: there is no
    get_action() method and no Action classes. The transition broadcasts
    the raw action_id to every object and lets each object respond.

    The LLM synthesizes:
    - GameObject subclasses (with respond() handling all action IDs)
    - A SynthesizedDomain with perceive() (frame -> World)
    - Optionally, a custom respond_order() to control update ordering

    The transition is:
        world = perceive(frame)
        for obj in respond_order(world, action_id):
            obj.respond(action_id, world)
        return world.render()
    """

    @abstractmethod
    def perceive(self, frame: np.ndarray) -> World:
        """Parse a raw frame into a World of typed objects."""
        pass

    def respond_order(self, world: World, action_id: int) -> list[GameObject]:
        """
        Determine the order in which objects respond to the action.

        Override this to control causal ordering. For example, the player
        should respond before blocks (so blocks can see the player's new
        position when deciding whether they were pushed).

        Default: all objects in their natural order.
        """
        return list(world.objects)

    def transition(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        """
        Full transition: perceive -> broadcast action -> render.

        Each object independently responds to the action_id signal.
        No Action class, no apply() method. Objects drive the transition.
        """
        world = self.perceive(frame)
        for obj in self.respond_order(world, action_id):
            obj.respond(action_id, world)
        return world.render()


# =============================================================================
# Frame utilities (duplicated for independence from other agents)
# =============================================================================


@dataclass
class PixelDiff:
    count: int
    positions: list[tuple[int, int]]
    bbox: tuple[int, int, int, int] | None
    before_colors: list[tuple[int, int, int]]
    after_colors: list[tuple[int, int, int]]


def compute_diff(before: np.ndarray, after: np.ndarray) -> PixelDiff:
    if before.shape != after.shape:
        return PixelDiff(count=0, positions=[], bbox=None,
                         before_colors=[], after_colors=[])

    if before.ndim == 3:
        diff_mask = np.any(before != after, axis=-1)
    else:
        diff_mask = before != after

    positions = list(zip(*np.where(diff_mask)))
    count = len(positions)

    if count == 0:
        return PixelDiff(count=0, positions=[], bbox=None,
                         before_colors=[], after_colors=[])

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    bbox = (min(rows), min(cols), max(rows), max(cols))

    sample = positions[:100]
    if before.ndim == 3:
        before_colors = [tuple(int(x) for x in before[r, c]) for r, c in sample]
        after_colors = [tuple(int(x) for x in after[r, c]) for r, c in sample]
    else:
        before_colors = [(int(before[r, c]),) for r, c in sample]
        after_colors = [(int(after[r, c]),) for r, c in sample]

    return PixelDiff(
        count=count, positions=positions[:1000], bbox=bbox,
        before_colors=before_colors, after_colors=after_colors,
    )


def find_unique_colors(frame: np.ndarray) -> list[tuple[int, ...]]:
    if frame.ndim == 3:
        flat = frame.reshape(-1, frame.shape[-1])
        unique = np.unique(flat, axis=0)
        return [tuple(int(x) for x in row) for row in unique]
    else:
        return [(int(x),) for x in np.unique(frame)]


def find_color_regions(
    frame: np.ndarray,
    target_color: tuple[int, ...] | np.ndarray,
    connectivity: int = 4,
) -> list[set[tuple[int, int]]]:
    from scipy import ndimage

    target = np.array(target_color)
    if frame.ndim == 3:
        mask = np.all(frame == target, axis=-1)
    else:
        mask = frame == target[0] if len(target) == 1 else frame == target

    if connectivity == 8:
        structure = np.ones((3, 3))
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    labeled, num_features = ndimage.label(mask.astype(np.int32), structure=structure)

    regions = []
    for comp_id in range(1, num_features + 1):
        pixels = set(zip(*np.where(labeled == comp_id)))
        regions.append(pixels)

    return regions


def region_bbox(pixels: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    return (min(rows), min(cols), max(rows), max(cols))


def most_common_color(frame: np.ndarray) -> tuple[int, ...]:
    if frame.ndim == 3:
        flat = frame.reshape(-1, frame.shape[-1])
        hashes = flat[:, 0].astype(np.int64) * 65536 + flat[:, 1].astype(np.int64) * 256 + flat[:, 2].astype(np.int64)
        values, counts = np.unique(hashes, return_counts=True)
        best = values[np.argmax(counts)]
        return (int(best // 65536), int((best % 65536) // 256), int(best % 256))
    else:
        values, counts = np.unique(frame, return_counts=True)
        return (int(values[np.argmax(counts)]),)


# Re-export from shared for backward compatibility
from src.shared.frame_utils import extract_grid, palette_to_rgb, ARC_PALETTE  # noqa: F401
