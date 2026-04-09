"""
Object tracker: assigns persistent track_ids to SpriteInstances across frames.

When the SpriteRegistry produces SpriteInstances from a frame, they have
track_id = -1. The ObjectTracker matches them to instances from the previous
frame to assign stable identities.

Matching strategy (greedy, prioritized):
1. Same sprite_type + closest position → strongest signal.
2. Same sprite_type + similar area → weaker but still good.
3. Unmatched instances get new track_ids.

We use a cost matrix (type mismatch penalty + Euclidean distance) and
solve greedily (sorted by cost). For the typical ARC-3 game with <20
objects, Hungarian algorithm is overkill — greedy is fine and simpler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..state.object_state import SpriteInstance

logger = logging.getLogger(__name__)


@dataclass
class TrackRecord:
    """History of a single tracked object."""
    track_id: int
    sprite_type_id: int
    positions: list[tuple[int, int]]     # (row, col) per frame observed
    frame_indices: list[int]             # which frames this was seen in
    last_seen_frame: int = 0
    is_active: bool = True               # False if disappeared

    @property
    def age(self) -> int:
        """Number of frames since first observed."""
        if not self.frame_indices:
            return 0
        return self.frame_indices[-1] - self.frame_indices[0] + 1

    @property
    def last_position(self) -> tuple[int, int]:
        return self.positions[-1] if self.positions else (0, 0)


class ObjectTracker:
    """
    Assigns persistent track_ids to SpriteInstances across frames.

    Usage:
        tracker = ObjectTracker()
        # Each frame:
        tracked = tracker.update(sprite_instances, frame_index)
        # tracked is the same list but with track_ids assigned.
    """

    def __init__(
        self,
        max_match_distance: float = 20.0,
        type_mismatch_penalty: float = 100.0,
        max_frames_missing: int = 3,
    ):
        """
        Args:
            max_match_distance: Maximum position distance for a valid match.
            type_mismatch_penalty: Extra cost for matching different sprite types.
            max_frames_missing: After this many consecutive frames without
                seeing a track, mark it as inactive.
        """
        self.max_match_distance = max_match_distance
        self.type_mismatch_penalty = type_mismatch_penalty
        self.max_frames_missing = max_frames_missing

        self._next_track_id: int = 0
        self._tracks: dict[int, TrackRecord] = {}
        self._previous_instances: list[SpriteInstance] = []

    @property
    def active_tracks(self) -> dict[int, TrackRecord]:
        return {tid: t for tid, t in self._tracks.items() if t.is_active}

    @property
    def all_tracks(self) -> dict[int, TrackRecord]:
        return dict(self._tracks)

    def update(
        self,
        instances: list[SpriteInstance],
        frame_index: int,
    ) -> list[SpriteInstance]:
        """
        Match new instances to existing tracks and assign track_ids.

        Args:
            instances: SpriteInstances with track_id = -1 (from SpriteRegistry.match).
            frame_index: Current frame number.

        Returns:
            The same instances, but with track_ids assigned.
        """
        if not self._previous_instances:
            # First frame: every instance gets a new track
            for inst in instances:
                tid = self._new_track(inst, frame_index)
                inst.track_id = tid
            self._previous_instances = instances
            return instances

        # Build cost matrix: previous × current
        prev = self._previous_instances
        costs: list[tuple[float, int, int]] = []  # (cost, prev_idx, curr_idx)

        for pi, prev_inst in enumerate(prev):
            for ci, curr_inst in enumerate(instances):
                cost = self._match_cost(prev_inst, curr_inst)
                if cost < self.max_match_distance + self.type_mismatch_penalty:
                    costs.append((cost, pi, ci))

        # Greedy matching: lowest cost first
        costs.sort(key=lambda x: x[0])
        matched_prev: set[int] = set()
        matched_curr: set[int] = set()

        for cost, pi, ci in costs:
            if pi in matched_prev or ci in matched_curr:
                continue
            if cost > self.max_match_distance + self.type_mismatch_penalty:
                continue

            # Assign previous track_id to current instance
            tid = prev[pi].track_id
            instances[ci].track_id = tid

            # Update track record
            if tid in self._tracks:
                self._tracks[tid].positions.append(instances[ci].position)
                self._tracks[tid].frame_indices.append(frame_index)
                self._tracks[tid].last_seen_frame = frame_index
                self._tracks[tid].sprite_type_id = instances[ci].sprite_type.type_id

            matched_prev.add(pi)
            matched_curr.add(ci)

        # Unmatched current instances: new tracks
        for ci, inst in enumerate(instances):
            if ci not in matched_curr:
                tid = self._new_track(inst, frame_index)
                inst.track_id = tid

        # Unmatched previous tracks: mark as potentially disappeared
        for pi, prev_inst in enumerate(prev):
            if pi not in matched_prev:
                tid = prev_inst.track_id
                if tid in self._tracks:
                    frames_since = frame_index - self._tracks[tid].last_seen_frame
                    if frames_since >= self.max_frames_missing:
                        self._tracks[tid].is_active = False

        self._previous_instances = instances
        return instances

    def get_displacement(self, track_id: int) -> tuple[int, int] | None:
        """Get the most recent displacement for a track (last two positions)."""
        track = self._tracks.get(track_id)
        if track is None or len(track.positions) < 2:
            return None
        p1 = track.positions[-2]
        p2 = track.positions[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])

    def get_track_history(self, track_id: int) -> TrackRecord | None:
        return self._tracks.get(track_id)

    def _match_cost(self, prev: SpriteInstance, curr: SpriteInstance) -> float:
        """Compute matching cost between a previous and current instance."""
        # Position distance
        dist = np.sqrt(
            (prev.center[0] - curr.center[0]) ** 2
            + (prev.center[1] - curr.center[1]) ** 2
        )

        # Type mismatch penalty
        type_penalty = 0.0
        if prev.sprite_type.type_id != curr.sprite_type.type_id:
            # Different types: heavy penalty unless colors overlap
            shared_colors = prev.colors & curr.colors
            if shared_colors:
                type_penalty = self.type_mismatch_penalty * 0.5
            else:
                type_penalty = self.type_mismatch_penalty

        return dist + type_penalty

    def _new_track(self, inst: SpriteInstance, frame_index: int) -> int:
        """Create a new track for an instance."""
        tid = self._next_track_id
        self._next_track_id += 1

        self._tracks[tid] = TrackRecord(
            track_id=tid,
            sprite_type_id=inst.sprite_type.type_id,
            positions=[inst.position],
            frame_indices=[frame_index],
            last_seen_frame=frame_index,
        )

        return tid
