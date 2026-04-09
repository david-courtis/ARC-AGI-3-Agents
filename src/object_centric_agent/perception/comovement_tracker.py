"""
Co-movement tracker: discovers which fragments move together across frames.

This is the key mechanism for discovering multi-color sprites. Given two
consecutive frames (before/after an action), we:

1. Detect fragments in both frames.
2. Match fragments across frames by (color, shape similarity, proximity).
3. Compute displacement vectors for each matched pair.
4. Group fragments that share the same displacement — they co-moved.

Over multiple observations, co-movement evidence accumulates. Fragments that
*always* co-move (same displacement on every action that produces movement)
are strong candidates for being parts of the same sprite.

The output is a set of CoMovementGroup objects, each containing fragment IDs
that moved together. The SpriteRegistry consumes these to merge fragments
into multi-fragment SpriteTypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from ..state.object_state import Fragment


@dataclass
class FragmentMatch:
    """A matched pair of fragments across two consecutive frames."""
    before: Fragment
    after: Fragment
    displacement: tuple[int, int]  # (dr, dc) from before to after position
    similarity: float              # 0.0 to 1.0 template match score

    @property
    def moved(self) -> bool:
        return self.displacement != (0, 0)


@dataclass
class ComovementGroup:
    """
    A set of fragments that shared the same displacement in one transition.

    If fragments A, B, C all moved by (0, -1) while D, E stayed at (0, 0),
    we get two groups: {A, B, C} with displacement (0, -1) and {D, E} with (0, 0).
    """
    displacement: tuple[int, int]
    fragment_colors: frozenset[tuple[int, int, int]]  # colors of constituent fragments
    fragment_matches: list[FragmentMatch]

    @property
    def size(self) -> int:
        return len(self.fragment_matches)

    @property
    def moved(self) -> bool:
        return self.displacement != (0, 0)


@dataclass
class ComovementEvidence:
    """
    Accumulated evidence that a set of fragment colors co-move.

    Each observation adds to the count. A group is considered a reliable
    co-movement cluster when it has been observed N times with the same
    membership (same set of colors always moving together).
    """
    member_colors: frozenset[tuple[int, int, int]]
    observation_count: int = 0
    displacements_observed: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        """At least 2 observations of co-movement (not just static)."""
        non_zero = [d for d in self.displacements_observed if d != (0, 0)]
        return len(non_zero) >= 2


class ComovementTracker:
    """
    Tracks co-movement patterns across frame transitions to discover
    which fragments belong to the same multi-color sprite.

    Usage:
        tracker = ComovementTracker()

        # After each action:
        groups = tracker.observe_transition(before_fragments, after_fragments)

        # Query accumulated evidence:
        clusters = tracker.get_reliable_clusters()
        # → e.g., [frozenset({(255,0,0), (0,0,255)})]  means red+blue always co-move
    """

    def __init__(self, match_distance_threshold: float = 15.0):
        """
        Args:
            match_distance_threshold: Maximum centroid distance (in pixels) to
                consider two fragments as potential matches across frames.
        """
        self.match_distance_threshold = match_distance_threshold

        # Accumulated evidence: frozenset of colors → ComovementEvidence
        self._evidence: dict[frozenset[tuple[int, int, int]], ComovementEvidence] = {}

        # History of all observed groups (for debugging)
        self._history: list[list[ComovementGroup]] = []

    def observe_transition(
        self,
        before_fragments: list[Fragment],
        after_fragments: list[Fragment],
    ) -> list[ComovementGroup]:
        """
        Observe a single before→after transition and extract co-movement groups.

        Args:
            before_fragments: Fragments from the frame before the action.
            after_fragments: Fragments from the frame after the action.

        Returns:
            List of ComovementGroup objects for this transition.
        """
        # Step 1: Match fragments across frames
        matches = self._match_fragments(before_fragments, after_fragments)

        if not matches:
            return []

        # Step 2: Group by displacement
        by_displacement: dict[tuple[int, int], list[FragmentMatch]] = defaultdict(list)
        for m in matches:
            by_displacement[m.displacement].append(m)

        # Step 3: Build ComovementGroups
        groups: list[ComovementGroup] = []
        for disp, match_list in by_displacement.items():
            colors = frozenset(m.before.color for m in match_list)
            groups.append(ComovementGroup(
                displacement=disp,
                fragment_colors=colors,
                fragment_matches=match_list,
            ))

        # Step 4: Update evidence for groups that actually moved
        # (static groups are less informative — everything static trivially "co-moves")
        moving_groups = [g for g in groups if g.moved]
        for group in moving_groups:
            self._update_evidence(group)

        # Also track that static fragments did NOT move with the moving ones
        # (negative evidence: they're separate sprites)

        self._history.append(groups)
        return groups

    def get_reliable_clusters(self) -> list[frozenset[tuple[int, int, int]]]:
        """
        Return color sets that have reliable co-movement evidence.

        A cluster is reliable if the same set of colors has been observed
        co-moving (with non-zero displacement) at least 2 times.
        """
        return [
            ev.member_colors
            for ev in self._evidence.values()
            if ev.is_reliable
        ]

    def get_all_evidence(self) -> dict[frozenset[tuple[int, int, int]], ComovementEvidence]:
        """Return all accumulated evidence (for inspection/debugging)."""
        return dict(self._evidence)

    def _match_fragments(
        self,
        before: list[Fragment],
        after: list[Fragment],
    ) -> list[FragmentMatch]:
        """
        Match fragments between two frames using color + proximity + shape.

        Strategy: greedy matching within same-color fragments, closest first.
        """
        matches: list[FragmentMatch] = []
        used_after: set[int] = set()  # indices into after list

        # Group by color for efficiency
        before_by_color: dict[tuple[int, int, int], list[tuple[int, Fragment]]] = defaultdict(list)
        after_by_color: dict[tuple[int, int, int], list[tuple[int, Fragment]]] = defaultdict(list)

        for i, f in enumerate(before):
            before_by_color[f.color].append((i, f))
        for i, f in enumerate(after):
            after_by_color[f.color].append((i, f))

        for color in before_by_color:
            if color not in after_by_color:
                continue  # all fragments of this color disappeared

            b_frags = before_by_color[color]
            a_frags = after_by_color[color]

            # Build cost matrix: distance between centroids
            candidates: list[tuple[float, int, int, Fragment, Fragment]] = []
            for bi, bf in b_frags:
                for ai, af in a_frags:
                    if ai in used_after:
                        continue
                    dist = np.sqrt(
                        (bf.center[0] - af.center[0]) ** 2
                        + (bf.center[1] - af.center[1]) ** 2
                    )
                    if dist <= self.match_distance_threshold:
                        # Also check shape similarity (area ratio)
                        area_ratio = min(bf.area, af.area) / max(bf.area, af.area)
                        similarity = area_ratio * (1.0 - dist / self.match_distance_threshold)
                        candidates.append((dist, bi, ai, bf, af))

            # Greedy: closest first
            candidates.sort(key=lambda x: x[0])
            used_before_local: set[int] = set()

            for dist, bi, ai, bf, af in candidates:
                if bi in used_before_local or ai in used_after:
                    continue

                # Compute displacement as integer offset of bounding-box top-left
                dr = af.bbox[0] - bf.bbox[0]
                dc = af.bbox[1] - bf.bbox[1]

                # Compute similarity
                area_ratio = min(bf.area, af.area) / max(bf.area, af.area)
                sim = area_ratio

                matches.append(FragmentMatch(
                    before=bf,
                    after=af,
                    displacement=(dr, dc),
                    similarity=sim,
                ))

                used_before_local.add(bi)
                used_after.add(ai)

        return matches

    def _update_evidence(self, group: ComovementGroup) -> None:
        """Update co-movement evidence with a new observation."""
        key = group.fragment_colors

        if key not in self._evidence:
            self._evidence[key] = ComovementEvidence(member_colors=key)

        ev = self._evidence[key]
        ev.observation_count += 1
        ev.displacements_observed.append(group.displacement)
