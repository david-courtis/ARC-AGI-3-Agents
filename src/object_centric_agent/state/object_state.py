"""
Core data structures for the three-level object hierarchy.

Level 0 — Fragment:
    A single-color connected component. Deterministic, parameter-free extraction
    via color-CC (connected components on per-color binary masks, 8-connectivity).
    This is what you get from a single frame with zero prior knowledge.

Level 1 — Sprite (SpriteType / SpriteInstance):
    One or more Fragments that always co-move as a rigid body. Discovered by
    observing multiple frames: if Fragment A (red) and Fragment B (blue) both
    translate by exactly (0, -1) on every action, they are one Sprite.

    A SpriteType is the *class* (canonical template, like "3-color person").
    A SpriteInstance is a specific occurrence in a frame (with position, track_id).

Level 2 — Composite (CompositeType / CompositeInstance):
    Sprites with persistent spatial relationships that don't necessarily co-move
    rigidly but are semantically grouped. Example: a "player holding an item" where
    the player sprite and item sprite always appear adjacent but the item can be
    dropped. Composites are a stretch goal — the fragment→sprite pipeline handles
    the core multi-color object case.

Discovery flow:
    Frame 0:  color-CC → Fragments (each tentatively its own SpriteType)
    Frames 1+: observe co-movement → merge Fragments into multi-fragment SpriteTypes
    Later:     LLM suggests semantic groupings → CompositeTypes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Level 0: Fragment — atomic single-color connected component
# =============================================================================


@dataclass(frozen=True)
class FragmentID:
    """Unique identifier for a fragment within a single frame."""
    color: tuple[int, int, int]
    component_index: int  # which CC of this color (0-indexed)

    def __repr__(self) -> str:
        return f"Frag(rgb={self.color}, #{self.component_index})"


@dataclass
class Fragment:
    """
    A single-color connected component extracted from one frame.

    This is the atomic unit of perception. Every non-background pixel belongs
    to exactly one Fragment. Fragments are extracted deterministically via
    8-connectivity CC on each color channel.
    """
    frag_id: FragmentID
    color: tuple[int, int, int]
    pixels: frozenset[tuple[int, int]]  # set of (row, col) positions
    bbox: tuple[int, int, int, int]     # (min_row, min_col, max_row, max_col)
    center: tuple[float, float]         # (row, col) centroid
    area: int

    # Which sprite type this fragment belongs to (assigned by registry)
    sprite_type_id: int | None = None

    @staticmethod
    def from_pixels(
        color: tuple[int, int, int],
        component_index: int,
        pixels: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> Fragment:
        """Construct a Fragment from a set of pixel positions."""
        pixels = frozenset(pixels)
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        return Fragment(
            frag_id=FragmentID(color=color, component_index=component_index),
            color=color,
            pixels=pixels,
            bbox=(min(rows), min(cols), max(rows), max(cols)),
            center=(sum(rows) / len(rows), sum(cols) / len(cols)),
            area=len(pixels),
        )

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def top_left(self) -> tuple[int, int]:
        return (self.bbox[0], self.bbox[1])

    def translate(self, dr: int, dc: int) -> Fragment:
        """Return a new Fragment shifted by (dr, dc). Useful for prediction."""
        new_pixels = frozenset((r + dr, c + dc) for r, c in self.pixels)
        return Fragment.from_pixels(self.color, self.frag_id.component_index, new_pixels)

    def pixel_template(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """
        Extract the bounding-box crop as an RGBA array.
        Alpha=255 where this fragment has pixels, 0 elsewhere.
        Returns shape (height, width, 4) uint8.
        """
        h, w = self.height, self.width
        template = np.zeros((h, w, 4), dtype=np.uint8)
        r0, c0 = self.bbox[0], self.bbox[1]
        for r, c in self.pixels:
            lr, lc = r - r0, c - c0
            template[lr, lc, :3] = self.color
            template[lr, lc, 3] = 255
        return template

    def iou(self, other: Fragment) -> float:
        """Intersection over union of pixel sets."""
        intersection = len(self.pixels & other.pixels)
        union = len(self.pixels | other.pixels)
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Level 1: Sprite — one or more co-moving fragments
# =============================================================================


@dataclass
class SpriteType:
    """
    A discovered sprite class: a canonical visual template that may consist of
    multiple fragments (colors) that always co-move.

    SpriteTypes are discovered over time:
    - Initially, each Fragment is its own SpriteType (single-color, single-CC).
    - When co-movement is observed, fragments are merged into multi-fragment SpriteTypes.

    The canonical_fragments store the fragment shapes relative to the sprite's
    anchor point (top-left of the combined bounding box), so we can match
    instances across frames regardless of absolute position.
    """
    type_id: int
    name: str  # human-readable, initially auto-generated like "sprite_0"

    # Canonical shape: list of (relative_pixels, color) per constituent fragment.
    # relative_pixels are offset from the sprite's anchor (0,0) = top-left of combined bbox.
    canonical_fragments: list[tuple[frozenset[tuple[int, int]], tuple[int, int, int]]]

    # Combined bounding box size of the canonical template
    canonical_width: int
    canonical_height: int

    # Discovery metadata
    observation_count: int = 0
    first_seen_frame: int = 0
    colors: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)

    # Behavioral observations (filled in by exploration)
    is_static: bool | None = None      # None = unknown, True = never moves
    is_player: bool | None = None      # None = unknown, True = responds to directional actions
    is_selectable: bool | None = None   # None = unknown, True = responds to ACTION6 click

    def __post_init__(self):
        if not self.colors:
            self.colors = frozenset(color for _, color in self.canonical_fragments)

    @property
    def is_multi_color(self) -> bool:
        return len(self.colors) > 1

    @property
    def num_fragments(self) -> int:
        return len(self.canonical_fragments)

    @property
    def total_pixels(self) -> int:
        return sum(len(px) for px, _ in self.canonical_fragments)

    def canonical_template(self) -> np.ndarray:
        """
        Render the canonical template as an RGBA image.
        Shape: (canonical_height, canonical_width, 4), uint8.
        Alpha=255 where sprite has pixels, 0 elsewhere.
        """
        template = np.zeros((self.canonical_height, self.canonical_width, 4), dtype=np.uint8)
        for rel_pixels, color in self.canonical_fragments:
            for r, c in rel_pixels:
                if 0 <= r < self.canonical_height and 0 <= c < self.canonical_width:
                    template[r, c, :3] = color
                    template[r, c, 3] = 255
        return template

    def match_score(self, fragments: list[Fragment]) -> float:
        """
        How well does a list of fragments match this sprite type?
        Returns 0.0 (no match) to 1.0 (perfect match).

        Compares the combined pixel pattern (relative to combined bbox anchor)
        against the canonical template.
        """
        if not fragments:
            return 0.0

        # Compute combined bbox of the candidate fragments
        all_pixels: set[tuple[int, int, tuple[int, int, int]]] = set()
        for frag in fragments:
            for r, c in frag.pixels:
                all_pixels.add((r, c, frag.color))

        if not all_pixels:
            return 0.0

        rows = [p[0] for p in all_pixels]
        cols = [p[1] for p in all_pixels]
        anchor_r, anchor_c = min(rows), min(cols)

        # Build relative pixel set with colors
        candidate_rel: set[tuple[int, int, tuple[int, int, int]]] = set()
        for r, c, color in all_pixels:
            candidate_rel.add((r - anchor_r, c - anchor_c, color))

        # Build canonical pixel set with colors
        canonical_rel: set[tuple[int, int, tuple[int, int, int]]] = set()
        for rel_pixels, color in self.canonical_fragments:
            for r, c in rel_pixels:
                canonical_rel.add((r, c, color))

        # IoU on the colored pixel sets
        intersection = len(candidate_rel & canonical_rel)
        union = len(candidate_rel | canonical_rel)
        return intersection / union if union > 0 else 0.0


@dataclass
class SpriteInstance:
    """
    A specific occurrence of a SpriteType in a particular frame.

    Each instance has a persistent track_id assigned by the object tracker,
    allowing us to follow the "same" object across frames.
    """
    track_id: int
    sprite_type: SpriteType
    position: tuple[int, int]          # (row, col) of anchor (top-left of combined bbox)
    center: tuple[float, float]        # (row, col) centroid of all pixels
    fragments: list[Fragment]          # the actual fragments in this frame
    frame_index: int                   # which frame this was observed in

    # Mutable state (updated by exploration / transition model)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def type_id(self) -> int:
        return self.sprite_type.type_id

    @property
    def type_name(self) -> str:
        return self.sprite_type.name

    @property
    def all_pixels(self) -> frozenset[tuple[int, int]]:
        """Union of all fragment pixels."""
        pixels: set[tuple[int, int]] = set()
        for frag in self.fragments:
            pixels |= frag.pixels
        return frozenset(pixels)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Combined bounding box of all fragments."""
        all_px = self.all_pixels
        rows = [p[0] for p in all_px]
        cols = [p[1] for p in all_px]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def colors(self) -> set[tuple[int, int, int]]:
        return {frag.color for frag in self.fragments}

    def displacement_from(self, other: SpriteInstance) -> tuple[int, int]:
        """(dr, dc) from other's position to this instance's position."""
        return (self.position[0] - other.position[0],
                self.position[1] - other.position[1])

    def render(self, frame: np.ndarray) -> None:
        """Stamp this sprite's pixels onto a frame (mutates in-place)."""
        for frag in self.fragments:
            for r, c in frag.pixels:
                if 0 <= r < frame.shape[0] and 0 <= c < frame.shape[1]:
                    frame[r, c] = frag.color


# =============================================================================
# Level 2: Composite — sprites with persistent spatial relationships
# =============================================================================


@dataclass
class CompositeType:
    """
    A group of SpriteTypes that have a persistent spatial/semantic relationship.

    Example: "player_with_item" = Player sprite + HeldItem sprite, where the
    item always appears 1 cell to the right of the player.

    Composites are discovered later in the pipeline (by observation or LLM hint)
    and are optional — the core pipeline works with SpriteTypes alone.
    """
    composite_id: int
    name: str
    member_types: list[SpriteType]
    spatial_constraints: list[str]  # e.g., ["item is always 1 cell right of player"]
    observation_count: int = 0


@dataclass
class CompositeInstance:
    """A specific occurrence of a CompositeType in a frame."""
    composite_type: CompositeType
    members: list[SpriteInstance]
    frame_index: int


# =============================================================================
# WorldState — structured snapshot of a single frame
# =============================================================================


@dataclass
class WorldState:
    """
    Complete structured representation of a single game frame.

    Produced by the perception pipeline (frame_parser). This replaces
    raw numpy frames in the replay buffer for structured reasoning.
    """
    frame: np.ndarray                          # raw RGB pixels (H, W, 3)
    frame_index: int                           # monotonically increasing frame counter
    fragments: list[Fragment]                  # Level 0: all detected fragments
    sprites: list[SpriteInstance]              # Level 1: tracked sprite instances
    composites: list[CompositeInstance]         # Level 2: composite groupings (may be empty)
    background_color: tuple[int, int, int]     # detected background
    selected_sprite: SpriteInstance | None = None  # currently selected object (if known)

    @property
    def num_sprites(self) -> int:
        return len(self.sprites)

    @property
    def sprite_types_present(self) -> set[int]:
        return {s.type_id for s in self.sprites}

    def get_sprites_of_type(self, type_id: int) -> list[SpriteInstance]:
        return [s for s in self.sprites if s.type_id == type_id]

    def get_sprite_by_track(self, track_id: int) -> SpriteInstance | None:
        for s in self.sprites:
            if s.track_id == track_id:
                return s
        return None

    def get_sprite_at(self, row: int, col: int) -> SpriteInstance | None:
        """Find the sprite whose pixels contain (row, col), if any."""
        for s in self.sprites:
            if (row, col) in s.all_pixels:
                return s
        return None

    def describe(self) -> str:
        """Human-readable summary for LLM context."""
        lines = [
            f"Frame {self.frame_index}: {self.frame.shape[0]}x{self.frame.shape[1]}, "
            f"bg=RGB{self.background_color}",
            f"  Fragments: {len(self.fragments)}",
            f"  Sprites: {len(self.sprites)} instances of {len(self.sprite_types_present)} types",
        ]
        for s in self.sprites:
            lines.append(
                f"    [{s.track_id}] {s.type_name} at ({s.position[0]},{s.position[1]}) "
                f"colors={s.colors} area={sum(f.area for f in s.fragments)}"
            )
        if self.selected_sprite:
            lines.append(f"  Selected: [{self.selected_sprite.track_id}] {self.selected_sprite.type_name}")
        if self.composites:
            lines.append(f"  Composites: {len(self.composites)}")
        return "\n".join(lines)
