"""
Sprite registry: maintains a library of discovered SpriteTypes.

The registry is the bridge between raw fragment detection and typed object
instances. It manages:

1. **Bootstrap**: On the first frame, every fragment becomes its own SpriteType.
2. **Merge**: When co-movement evidence shows fragments always move together,
   merge them into a single multi-fragment SpriteType.
3. **Match**: Given fragments from a new frame, match them to known SpriteTypes
   to produce SpriteInstances.
4. **Split** (rare): If a previously-merged sprite is observed with its parts
   moving independently, split it back into separate types.
5. **LLM refinement**: Accept hints from the LLM about which fragments should
   be grouped (semantic grouping that goes beyond co-movement).

The registry is stateful and evolves over the course of a game.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from ..state.object_state import Fragment, SpriteType, SpriteInstance
from .comovement_tracker import ComovementTracker

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """A candidate match between a SpriteType and fragments in a frame."""
    sprite_type: SpriteType
    fragments: list[Fragment]
    score: float
    position: tuple[int, int]  # anchor (top-left of combined bbox)
    center: tuple[float, float]


class SpriteRegistry:
    """
    Maintains the library of known SpriteTypes and matches incoming fragments.

    Lifecycle:
        1. registry = SpriteRegistry()
        2. registry.bootstrap(first_frame_fragments)      # initialize types
        3. registry.update_from_comovement(evidence)       # merge co-moving fragments
        4. instances = registry.match(frame_fragments)     # match per frame
        5. registry.refine_from_hint(merge_colors=[...])   # LLM suggestion
    """

    def __init__(self):
        self._types: dict[int, SpriteType] = {}  # type_id → SpriteType
        self._next_type_id: int = 0
        self._bootstrapped: bool = False

        # Color → type_id mapping for fast single-color lookups
        # Only valid for single-fragment types; multi-fragment types need
        # template matching.
        self._color_to_type: dict[tuple[int, int, int], list[int]] = defaultdict(list)

        # Merge history: track which types were merged into which
        self._merge_log: list[tuple[list[int], int]] = []  # (old_ids, new_id)

    @property
    def types(self) -> dict[int, SpriteType]:
        return dict(self._types)

    @property
    def num_types(self) -> int:
        return len(self._types)

    def bootstrap(self, fragments: list[Fragment], frame_index: int = 0) -> list[SpriteType]:
        """
        Initialize the registry from the first frame's fragments.

        Each unique (color, shape) combination becomes its own SpriteType.
        Fragments with the same color and similar shape are grouped under one type.

        Args:
            fragments: Fragments from the first frame.
            frame_index: Frame number (for metadata).

        Returns:
            List of newly created SpriteTypes.
        """
        new_types: list[SpriteType] = []

        # Group fragments by color
        by_color: dict[tuple[int, int, int], list[Fragment]] = defaultdict(list)
        for frag in fragments:
            by_color[frag.color].append(frag)

        for color, color_frags in by_color.items():
            # Within same color, group by similar shape (area + aspect ratio)
            shape_groups = self._group_by_shape(color_frags)

            for group in shape_groups:
                # Create a SpriteType from the first fragment as canonical
                representative = group[0]
                sprite_type = self._create_type_from_fragments(
                    [representative], frame_index
                )
                new_types.append(sprite_type)

                # Assign all fragments in this group to this type
                for frag in group:
                    frag.sprite_type_id = sprite_type.type_id

                self._color_to_type[color].append(sprite_type.type_id)

        self._bootstrapped = True
        logger.info(
            f"[SpriteRegistry] Bootstrapped {len(new_types)} sprite types "
            f"from {len(fragments)} fragments"
        )
        return new_types

    def merge_from_comovement(
        self,
        comovement: ComovementTracker,
    ) -> list[SpriteType]:
        """
        Check co-movement evidence and merge fragment types that always co-move.

        Args:
            comovement: The ComovementTracker with accumulated evidence.

        Returns:
            List of newly created merged SpriteTypes (empty if no merges).
        """
        clusters = comovement.get_reliable_clusters()
        new_types: list[SpriteType] = []

        for color_set in clusters:
            if len(color_set) < 2:
                continue  # single-color, nothing to merge

            # Find existing single-color types for these colors
            types_to_merge: list[SpriteType] = []
            for color in color_set:
                for type_id in self._color_to_type.get(color, []):
                    if type_id in self._types:
                        types_to_merge.append(self._types[type_id])

            if len(types_to_merge) < 2:
                continue  # already merged or not enough types

            # Check if these types are already part of a multi-fragment type
            already_merged = any(t.num_fragments > 1 for t in types_to_merge)
            if already_merged:
                continue

            # Merge: combine canonical fragments from all types
            merged = self._merge_types(types_to_merge)
            if merged is not None:
                new_types.append(merged)
                logger.info(
                    f"[SpriteRegistry] Merged types {[t.type_id for t in types_to_merge]} "
                    f"into {merged.type_id} ({merged.name}) — colors: {color_set}"
                )

        return new_types

    def match(
        self,
        fragments: list[Fragment],
        frame_index: int = 0,
    ) -> list[SpriteInstance]:
        """
        Match fragments from a frame to known SpriteTypes, producing SpriteInstances.

        Strategy:
        1. Try multi-fragment types first (they consume multiple fragments).
        2. Then match remaining fragments to single-fragment types.
        3. Unmatched fragments get new ad-hoc types (discovery).

        Args:
            fragments: All fragments detected in the current frame.
            frame_index: Frame number (for metadata).

        Returns:
            List of SpriteInstances (one per matched sprite in the frame).
            track_id is NOT assigned here — that's the ObjectTracker's job.
        """
        instances: list[SpriteInstance] = []
        used_fragments: set[int] = set()  # indices into fragments list

        # Sort types: multi-fragment first (greedier matches first)
        sorted_types = sorted(
            self._types.values(),
            key=lambda t: t.num_fragments,
            reverse=True,
        )

        for sprite_type in sorted_types:
            if sprite_type.num_fragments > 1:
                # Multi-fragment type: find groups of fragments that match
                new_instances = self._match_multi_fragment(
                    sprite_type, fragments, used_fragments, frame_index
                )
                instances.extend(new_instances)
            else:
                # Single-fragment type: match individual fragments
                new_instances = self._match_single_fragment(
                    sprite_type, fragments, used_fragments, frame_index
                )
                instances.extend(new_instances)

        # Handle unmatched fragments: create new types on-the-fly
        for i, frag in enumerate(fragments):
            if i not in used_fragments:
                # This fragment doesn't match any known type — new discovery
                new_type = self._create_type_from_fragments([frag], frame_index)
                self._color_to_type[frag.color].append(new_type.type_id)

                inst = self._make_instance(new_type, [frag], frame_index)
                instances.append(inst)
                used_fragments.add(i)

                logger.debug(
                    f"[SpriteRegistry] New type {new_type.type_id} from unmatched "
                    f"fragment color={frag.color} area={frag.area}"
                )

        return instances

    def refine_from_hint(
        self,
        merge_colors: list[tuple[int, int, int]] | None = None,
        merge_type_ids: list[int] | None = None,
        split_type_id: int | None = None,
    ) -> SpriteType | list[SpriteType] | None:
        """
        Accept a refinement hint (e.g., from LLM analysis).

        Args:
            merge_colors: Colors to merge into one sprite type.
            merge_type_ids: Type IDs to merge.
            split_type_id: Type ID to split back into per-fragment types.

        Returns:
            New or modified SpriteType(s), or None if no change.
        """
        if merge_type_ids:
            types = [self._types[tid] for tid in merge_type_ids if tid in self._types]
            if len(types) >= 2:
                return self._merge_types(types)

        if merge_colors and len(merge_colors) >= 2:
            types: list[SpriteType] = []
            for color in merge_colors:
                for tid in self._color_to_type.get(color, []):
                    if tid in self._types:
                        types.append(self._types[tid])
            if len(types) >= 2:
                return self._merge_types(types)

        if split_type_id and split_type_id in self._types:
            return self._split_type(split_type_id)

        return None

    def get_type(self, type_id: int) -> SpriteType | None:
        return self._types.get(type_id)

    def describe(self) -> str:
        """Human-readable summary of all known types."""
        lines = [f"SpriteRegistry: {self.num_types} types"]
        for t in self._types.values():
            multi = f" ({t.num_fragments} fragments)" if t.is_multi_color else ""
            static = " [static]" if t.is_static else ""
            player = " [player]" if t.is_player else ""
            lines.append(
                f"  Type {t.type_id} ({t.name}): colors={t.colors}, "
                f"size={t.canonical_width}x{t.canonical_height}, "
                f"obs={t.observation_count}{multi}{static}{player}"
            )
        return "\n".join(lines)

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _create_type_from_fragments(
        self,
        fragments: list[Fragment],
        frame_index: int,
    ) -> SpriteType:
        """Create a new SpriteType from one or more fragments."""
        # Compute combined bbox
        all_pixels: set[tuple[int, int]] = set()
        for f in fragments:
            all_pixels |= f.pixels
        rows = [p[0] for p in all_pixels]
        cols = [p[1] for p in all_pixels]
        anchor_r, anchor_c = min(rows), min(cols)

        # Build canonical fragments (relative to anchor)
        canonical = []
        for f in fragments:
            rel_pixels = frozenset((r - anchor_r, c - anchor_c) for r, c in f.pixels)
            canonical.append((rel_pixels, f.color))

        type_id = self._next_type_id
        self._next_type_id += 1

        sprite_type = SpriteType(
            type_id=type_id,
            name=f"sprite_{type_id}",
            canonical_fragments=canonical,
            canonical_width=max(cols) - anchor_c + 1,
            canonical_height=max(rows) - anchor_r + 1,
            observation_count=1,
            first_seen_frame=frame_index,
        )

        self._types[type_id] = sprite_type
        return sprite_type

    def _merge_types(self, types_to_merge: list[SpriteType]) -> SpriteType | None:
        """Merge multiple single-fragment types into one multi-fragment type."""
        if len(types_to_merge) < 2:
            return None

        # Combine all canonical fragments
        all_canonical = []
        for t in types_to_merge:
            all_canonical.extend(t.canonical_fragments)

        # Recompute anchor and relative positions
        all_rel_pixels: set[tuple[int, int]] = set()
        for rel_px, _ in all_canonical:
            all_rel_pixels |= rel_px

        if not all_rel_pixels:
            return None

        rows = [p[0] for p in all_rel_pixels]
        cols = [p[1] for p in all_rel_pixels]

        type_id = self._next_type_id
        self._next_type_id += 1

        merged = SpriteType(
            type_id=type_id,
            name=f"sprite_{type_id}_merged",
            canonical_fragments=all_canonical,
            canonical_width=max(cols) + 1,
            canonical_height=max(rows) + 1,
            observation_count=sum(t.observation_count for t in types_to_merge),
            first_seen_frame=min(t.first_seen_frame for t in types_to_merge),
        )

        self._types[type_id] = merged

        # Remove old types
        old_ids = []
        for t in types_to_merge:
            old_ids.append(t.type_id)
            self._types.pop(t.type_id, None)
            # Clean color index
            for color in t.colors:
                if color in self._color_to_type:
                    self._color_to_type[color] = [
                        tid for tid in self._color_to_type[color] if tid != t.type_id
                    ]

        # Update color index for merged type
        for color in merged.colors:
            self._color_to_type[color].append(merged.type_id)

        self._merge_log.append((old_ids, type_id))
        return merged

    def _split_type(self, type_id: int) -> list[SpriteType]:
        """Split a multi-fragment type back into per-fragment types."""
        old_type = self._types.get(type_id)
        if old_type is None or old_type.num_fragments < 2:
            return []

        new_types = []
        for rel_pixels, color in old_type.canonical_fragments:
            # Each fragment becomes its own type
            new_tid = self._next_type_id
            self._next_type_id += 1
            rows = [p[0] for p in rel_pixels]
            cols = [p[1] for p in rel_pixels]
            t = SpriteType(
                type_id=new_tid,
                name=f"sprite_{new_tid}",
                canonical_fragments=[(rel_pixels, color)],
                canonical_width=max(cols) + 1 if cols else 1,
                canonical_height=max(rows) + 1 if rows else 1,
                first_seen_frame=old_type.first_seen_frame,
            )
            self._types[new_tid] = t
            self._color_to_type[color].append(new_tid)
            new_types.append(t)

        # Remove old type
        self._types.pop(type_id, None)
        for color in old_type.colors:
            if color in self._color_to_type:
                self._color_to_type[color] = [
                    tid for tid in self._color_to_type[color] if tid != type_id
                ]

        return new_types

    def _match_single_fragment(
        self,
        sprite_type: SpriteType,
        fragments: list[Fragment],
        used: set[int],
        frame_index: int,
    ) -> list[SpriteInstance]:
        """Match single-fragment type to available fragments."""
        instances = []

        # The canonical has one fragment with one color
        if not sprite_type.canonical_fragments:
            return []
        _, canonical_color = sprite_type.canonical_fragments[0]
        canonical_pixels = sprite_type.canonical_fragments[0][0]
        canonical_area = len(canonical_pixels)

        for i, frag in enumerate(fragments):
            if i in used:
                continue
            if frag.color != canonical_color:
                continue

            # Check area similarity (within 50% ratio)
            area_ratio = min(frag.area, canonical_area) / max(frag.area, canonical_area)
            if area_ratio < 0.5:
                continue

            # Accept match
            inst = self._make_instance(sprite_type, [frag], frame_index)
            instances.append(inst)
            used.add(i)
            sprite_type.observation_count += 1

        return instances

    def _match_multi_fragment(
        self,
        sprite_type: SpriteType,
        fragments: list[Fragment],
        used: set[int],
        frame_index: int,
    ) -> list[SpriteInstance]:
        """
        Match multi-fragment type: find groups of fragments that together
        match the canonical template.
        """
        instances = []
        needed_colors = {color for _, color in sprite_type.canonical_fragments}

        # Group available fragments by color
        available_by_color: dict[tuple[int, int, int], list[tuple[int, Fragment]]] = defaultdict(list)
        for i, frag in enumerate(fragments):
            if i not in used and frag.color in needed_colors:
                available_by_color[frag.color].append((i, frag))

        # Check if we have at least one fragment per needed color
        if not all(color in available_by_color for color in needed_colors):
            return []

        # Try to find matching groups using spatial proximity
        # For each fragment of the "anchor" color, try to assemble a full match
        anchor_color = list(needed_colors)[0]
        other_colors = needed_colors - {anchor_color}

        for anchor_idx, anchor_frag in available_by_color.get(anchor_color, []):
            if anchor_idx in used:
                continue

            # Try to find nearby fragments of the other colors
            candidate_group = [(anchor_idx, anchor_frag)]
            group_viable = True

            for other_color in other_colors:
                best_match: tuple[int, Fragment] | None = None
                best_dist = float("inf")

                for oi, of in available_by_color.get(other_color, []):
                    if oi in used or any(oi == ci for ci, _ in candidate_group):
                        continue
                    dist = np.sqrt(
                        (anchor_frag.center[0] - of.center[0]) ** 2
                        + (anchor_frag.center[1] - of.center[1]) ** 2
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_match = (oi, of)

                if best_match is None or best_dist > max(sprite_type.canonical_width, sprite_type.canonical_height) * 2:
                    group_viable = False
                    break
                candidate_group.append(best_match)

            if not group_viable:
                continue

            # Score the match
            group_frags = [f for _, f in candidate_group]
            score = sprite_type.match_score(group_frags)

            if score > 0.3:  # threshold for accepting
                inst = self._make_instance(sprite_type, group_frags, frame_index)
                instances.append(inst)
                for ci, _ in candidate_group:
                    used.add(ci)
                sprite_type.observation_count += 1

        return instances

    def _make_instance(
        self,
        sprite_type: SpriteType,
        fragments: list[Fragment],
        frame_index: int,
    ) -> SpriteInstance:
        """Create a SpriteInstance from a type and matched fragments."""
        all_pixels: set[tuple[int, int]] = set()
        for f in fragments:
            all_pixels |= f.pixels

        rows = [p[0] for p in all_pixels]
        cols = [p[1] for p in all_pixels]

        return SpriteInstance(
            track_id=-1,  # assigned later by ObjectTracker
            sprite_type=sprite_type,
            position=(min(rows), min(cols)),
            center=(sum(rows) / len(rows), sum(cols) / len(cols)),
            fragments=fragments,
            frame_index=frame_index,
        )

    def _group_by_shape(self, fragments: list[Fragment]) -> list[list[Fragment]]:
        """
        Group same-color fragments by similar shape (area + aspect ratio).

        Fragments with similar area (within 2x) and similar aspect ratio
        (within 0.5) are considered the same shape → same SpriteType.
        """
        if not fragments:
            return []

        groups: list[list[Fragment]] = []

        for frag in fragments:
            matched = False
            for group in groups:
                representative = group[0]

                # Area similarity
                area_ratio = min(frag.area, representative.area) / max(frag.area, representative.area)
                if area_ratio < 0.5:
                    continue

                # Aspect ratio similarity
                frag_ar = frag.width / frag.height if frag.height > 0 else 1.0
                rep_ar = representative.width / representative.height if representative.height > 0 else 1.0
                ar_diff = abs(frag_ar - rep_ar)
                if ar_diff > 0.5:
                    continue

                group.append(frag)
                matched = True
                break

            if not matched:
                groups.append([frag])

        return groups
