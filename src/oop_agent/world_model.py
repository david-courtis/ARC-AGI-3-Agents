"""
OOP World Model: Polymorphic, object-centric framework for world discovery.

Architecture
------------
This module defines the structural contract that LLM-synthesized code must
follow. The contract enforces an object-centric decomposition:

    Domain.transition(frame, action_id) -> predicted_frame

which internally decomposes as:

    1. perceive(frame)           -> World (frame parsed into typed objects)
    2. get_action(action_id)     -> Action instance
    3. action.apply(world)       -> mutates World by calling each object's
                                    respond_to_action (polymorphic dispatch)
    4. world.render()            -> predicted_frame (objects drawn back to pixels)

This factorization means the LLM writes T + A + 1 small programs (one per
object type, one per action type, one perceive function) instead of one
monolithic predict(). See docs/oop-vs-monolithic.md for the comparison.

Design Principles
-----------------
1. **No domain-specific properties baked in.** The base classes define the
   *pattern* (objects respond to actions polymorphically), not the *content*
   (what properties objects have, what actions do). The LLM fills in all
   domain-specific details when it synthesizes concrete subclasses.

2. **Arbitrary properties via **kwargs.** GameObject.__init__ accepts obj_id
   plus arbitrary keyword arguments that become instance attributes. The LLM
   decides what each object type needs (position, color, pixels, velocity,
   health, whatever).

3. **No assumptions about frames.** Frames are raw (H, W, 3) uint8 numpy
   arrays. The module provides utility functions for analyzing them (color
   detection, region finding, diffing) but never assumes what the colors or
   regions mean.

Why OOP structure matters (see docs/oop-vs-monolithic.md for full argument):
- Factored synthesis: each object type and action is a separate sub-program
- Error localization: counterexamples can be traced to specific object/action pairs
- Transfer: learned object types can be stored in a library for reuse
- Monotonic refinement: fixing one class does not break others
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# =============================================================================
# Core OOP base classes — structure only, no domain content
# =============================================================================


class GameObject(ABC):
    """
    Base class for all game objects. The root of the type hierarchy.

    This class defines the structural contract that every object type must
    satisfy. It does NOT define any domain-specific properties. The LLM
    creates subclasses that add whatever instance variables (atomic
    predicates) and methods (derived predicates) the domain requires.

    The key method is respond_to_action: each object type implements its
    own transition logic polymorphically. When an Action is applied, it
    calls respond_to_action on each affected object, and each object
    updates itself according to its type. This is the OOP dispatch that
    replaces the monolithic if-elif chain.

    Example of what the LLM might synthesize::

        class Player(GameObject):
            def __init__(self, obj_id, row, col, color):
                super().__init__(obj_id, row=row, col=col, color=color)

            def respond_to_action(self, action, world):
                if isinstance(action, MoveUp):
                    self.row = max(0, self.row - 1)

            def render(self, frame):
                frame[self.row, self.col] = self.color

    Properties are stored as regular instance attributes. The **kwargs
    pattern lets the LLM define whatever properties each type needs
    without modifying the base class.
    """

    def __init__(self, obj_id: str, **properties):
        self.obj_id = obj_id
        # LLM-defined properties are passed as kwargs and set as attributes.
        # This keeps the base class generic: the LLM decides what state
        # each object type carries (position, color, pixels, health, etc.)
        for key, value in properties.items():
            setattr(self, key, value)

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def respond_to_action(self, action: Action, world: World) -> None:
        """
        Update this object's state in response to an action.

        This is the polymorphic dispatch point. Each subclass implements
        its own response logic. The method mutates self in-place.

        Args:
            action: The action being applied.
            world: The current world state (for querying other objects).
        """
        pass

    @abstractmethod
    def render(self, frame: np.ndarray) -> None:
        """
        Write this object's pixels onto the frame.

        The LLM decides how each object type renders itself — single pixels,
        rectangular blocks, irregular shapes, etc.

        Args:
            frame: (H, W, 3) uint8 array to draw on (mutated in-place).
        """
        pass

    def __repr__(self) -> str:
        props = {k: v for k, v in self.__dict__.items()
                 if k != "obj_id" and not k.startswith("_")}
        return f"{self.type_name}(id={self.obj_id}, {props})"


class Action(ABC):
    """
    Base class for all actions.

    The agent observes anonymous actions (ACTION1 through ACTION5). The LLM
    maps each to a concrete Action subclass that encodes the action's
    semantics. The apply method is the action's transition logic: it
    iterates over affected objects and calls their respond_to_action
    methods in causal order.

    Causal ordering matters when objects interact. For example, in a
    push-block game, the action's apply should first check the player,
    then the block the player pushes, then anything that block bumps into.
    The LLM is responsible for getting this ordering right.

    Example of what the LLM might synthesize::

        class MoveRight(Action):
            def apply(self, world):
                for player in world.get_objects_of_type(Player):
                    player.respond_to_action(self, world)

    Note: no preconditions method is imposed. The LLM can build
    precondition checks into apply() if needed. This keeps the base
    class minimal.
    """

    action_id: int

    def __init__(self, action_id: int):
        self.action_id = action_id

    @abstractmethod
    def apply(self, world: World) -> None:
        """
        Apply this action to the world by dispatching to affected objects.

        The LLM decides which objects are affected and in what order.
        Each affected object's respond_to_action is called, which mutates
        the object's state in-place.

        Args:
            world: The world state (mutated in-place via object methods).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.action_id})"


class World:
    """
    The world state: a collection of typed objects plus the raw frame.

    Provides query methods for objects to inspect each other during
    respond_to_action. The specific query methods are generic enough
    to work for any domain.
    """

    def __init__(self, frame: np.ndarray, objects: list[GameObject]):
        self.frame = frame.copy()
        self.objects = list(objects)
        self.height = frame.shape[0]
        self.width = frame.shape[1]

    def get_objects_of_type(self, cls: type) -> list[GameObject]:
        """Get all objects that are instances of cls (including subclasses)."""
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
        """
        Render all objects onto a frame.

        Args:
            background: Base frame to render onto. If None, uses self.frame.

        Returns:
            (H, W, 3) uint8 array with all objects rendered.
        """
        if background is not None:
            frame = background.copy()
        else:
            frame = self.frame.copy()

        for obj in self.objects:
            obj.render(frame)

        return frame


class Domain(ABC):
    """
    A complete domain specification. The LLM synthesizes one of these.

    The Domain is the unit of synthesis in the OOP agent. It ties together:

    1. **Perception** (perceive): raw frame -> World of typed objects.
       This is where the LLM decides how to segment the 64x64 RGB pixels
       into meaningful entities and assign them types.

    2. **Action mapping** (get_action): anonymous action ID -> typed Action.
       This is where the LLM encodes its understanding of what each button does.

    3. **Transition** (transition): perceive -> apply action -> render.
       This is the top-level predict equivalent. It is concrete (not abstract)
       and calls perceive and get_action, so the LLM only needs to implement
       those two plus the object/action classes.

    The CEGIS loop synthesizes and refines the Domain. Counterexamples are
    transitions where domain.transition(frame, action) != actual_next_frame.
    """

    @abstractmethod
    def perceive(self, frame: np.ndarray) -> World:
        """
        Parse a raw frame into a structured World of typed objects.

        The LLM decides how to segment the frame, what object types
        to create, and what properties to assign.

        Args:
            frame: (H, W, 3) uint8 array.

        Returns:
            World containing typed GameObjects.
        """
        pass

    @abstractmethod
    def get_action(self, action_id: int) -> Action:
        """
        Map an anonymous action ID (1-5) to a typed Action instance.
        """
        pass

    def transition(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        """
        Full transition: perceive -> apply action -> render.

        This is the predict() equivalent. Returns the predicted next frame.

        Args:
            frame: Current (H, W, 3) uint8 frame.
            action_id: Which action to apply (1-5).

        Returns:
            Predicted next frame as (H, W, 3) uint8 array.
        """
        world = self.perceive(frame)
        action = self.get_action(action_id)
        action.apply(world)
        return world.render()


# =============================================================================
# Frame utilities (shared with monolithic agent but duplicated for independence)
# =============================================================================


@dataclass
class PixelDiff:
    """Summary of pixel-level differences between two frames."""
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
