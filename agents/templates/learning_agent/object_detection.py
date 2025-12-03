"""
Object detection for the Learning Agent using Gestalt principles.

This module groups pixels into objects using principles like:
- Proximity: Nearby pixels of same color belong together
- Similarity: Pixels of same color are related
- Common region: Rectangular blocks are likely objects

This helps the LLM understand the game at an object level rather than pixel level.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass
class DetectedObject:
    """A detected object in the frame."""

    object_id: int
    color: int
    pixels: list[tuple[int, int]]  # List of (row, col) positions
    bounding_box: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    center: tuple[float, float]  # (row, col) center position
    area: int  # Number of pixels
    width: int
    height: int
    is_rectangular: bool  # True if object fills its bounding box
    aspect_ratio: float

    def describe(self) -> str:
        """Generate a human-readable description of the object."""
        shape = "rectangular block" if self.is_rectangular else "irregular shape"
        size_desc = f"{self.width}x{self.height}"
        pos_desc = f"at ({self.bounding_box[0]}, {self.bounding_box[1]})"
        return f"Color {self.color} {shape} ({size_desc}) {pos_desc}"


@dataclass
class ObjectDiff:
    """Difference in objects between two frames."""

    appeared: list[DetectedObject] = field(default_factory=list)
    disappeared: list[DetectedObject] = field(default_factory=list)
    moved: list[tuple[DetectedObject, DetectedObject]] = field(default_factory=list)  # (before, after)
    color_changed: list[tuple[DetectedObject, DetectedObject]] = field(default_factory=list)
    unchanged: list[DetectedObject] = field(default_factory=list)

    def describe(self) -> str:
        """Generate a summary of object-level changes."""
        parts = []

        if self.appeared:
            parts.append(f"APPEARED: {len(self.appeared)} objects")
            for obj in self.appeared[:5]:  # Limit to 5
                parts.append(f"  + {obj.describe()}")

        if self.disappeared:
            parts.append(f"DISAPPEARED: {len(self.disappeared)} objects")
            for obj in self.disappeared[:5]:
                parts.append(f"  - {obj.describe()}")

        if self.moved:
            parts.append(f"MOVED: {len(self.moved)} objects")
            for before, after in self.moved[:5]:
                dx = after.center[1] - before.center[1]
                dy = after.center[0] - before.center[0]
                direction = []
                if dy < -1:
                    direction.append("up")
                elif dy > 1:
                    direction.append("down")
                if dx < -1:
                    direction.append("left")
                elif dx > 1:
                    direction.append("right")
                dir_str = "+".join(direction) if direction else "slightly"
                parts.append(f"  ~ {before.describe()} moved {dir_str}")

        if self.color_changed:
            parts.append(f"COLOR CHANGED: {len(self.color_changed)} objects")
            for before, after in self.color_changed[:5]:
                parts.append(
                    f"  * Object at ({before.bounding_box[0]}, {before.bounding_box[1]}): "
                    f"color {before.color} -> {after.color}"
                )

        if not parts:
            parts.append("No object-level changes detected")

        return "\n".join(parts)


class ObjectDetector:
    """
    Detects and tracks objects in game frames.

    Uses connected component analysis with Gestalt-inspired grouping.
    """

    def __init__(
        self,
        min_object_size: int = 4,
        background_colors: set[int] | None = None,
    ):
        """
        Initialize the object detector.

        Args:
            min_object_size: Minimum pixels for an object (filters noise)
            background_colors: Colors to treat as background (default: {0})
        """
        self.min_object_size = min_object_size
        self.background_colors = background_colors or {0}

    def detect_objects(self, frame: np.ndarray | list) -> list[DetectedObject]:
        """
        Detect all objects in a frame.

        Uses connected component analysis - pixels of the same color
        that are adjacent (8-connectivity) are grouped as one object.

        Args:
            frame: 2D array of color values (64x64 grid)

        Returns:
            List of detected objects
        """
        # Convert to numpy if needed
        if isinstance(frame, list):
            frame = np.array(frame)
        if frame.ndim == 3:
            # Take LAST frame (in case of animation)
            frame = frame[-1]

        objects = []
        object_id = 0

        # Get unique colors (excluding background)
        unique_colors = set(np.unique(frame)) - self.background_colors

        for color in unique_colors:
            # Create binary mask for this color
            mask = (frame == color).astype(int)

            # Find connected components (8-connectivity)
            labeled, num_features = ndimage.label(
                mask, structure=np.ones((3, 3))
            )

            for component_id in range(1, num_features + 1):
                # Get pixels for this component
                component_mask = labeled == component_id
                pixels = list(zip(*np.where(component_mask)))

                if len(pixels) < self.min_object_size:
                    continue

                # Calculate properties
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]

                min_row, max_row = min(rows), max(rows)
                min_col, max_col = min(cols), max(cols)
                width = max_col - min_col + 1
                height = max_row - min_row + 1

                # Check if rectangular (fills bounding box)
                expected_area = width * height
                actual_area = len(pixels)
                is_rectangular = actual_area >= expected_area * 0.9

                obj = DetectedObject(
                    object_id=object_id,
                    color=int(color),
                    pixels=pixels,
                    bounding_box=(min_row, min_col, max_row, max_col),
                    center=(np.mean(rows), np.mean(cols)),
                    area=actual_area,
                    width=width,
                    height=height,
                    is_rectangular=is_rectangular,
                    aspect_ratio=width / height if height > 0 else 1.0,
                )
                objects.append(obj)
                object_id += 1

        # Sort by position (top-left to bottom-right)
        objects.sort(key=lambda o: (o.bounding_box[0], o.bounding_box[1]))

        return objects

    def compare_frames(
        self, before: np.ndarray | list, after: np.ndarray | list
    ) -> ObjectDiff:
        """
        Compare objects between two frames.

        Identifies which objects appeared, disappeared, moved, or changed color.

        Args:
            before: Frame before action
            after: Frame after action

        Returns:
            ObjectDiff describing the changes
        """
        before_objects = self.detect_objects(before)
        after_objects = self.detect_objects(after)

        diff = ObjectDiff()

        # Track which objects have been matched
        matched_before = set()
        matched_after = set()

        # First pass: Find exact matches (same position, same color)
        for b_obj in before_objects:
            for a_obj in after_objects:
                if a_obj.object_id in matched_after:
                    continue

                if self._objects_match(b_obj, a_obj):
                    diff.unchanged.append(a_obj)
                    matched_before.add(b_obj.object_id)
                    matched_after.add(a_obj.object_id)
                    break

        # Second pass: Find moved objects (same color, different position)
        for b_obj in before_objects:
            if b_obj.object_id in matched_before:
                continue

            for a_obj in after_objects:
                if a_obj.object_id in matched_after:
                    continue

                if self._objects_moved(b_obj, a_obj):
                    diff.moved.append((b_obj, a_obj))
                    matched_before.add(b_obj.object_id)
                    matched_after.add(a_obj.object_id)
                    break

        # Third pass: Find color changes (same position, different color)
        for b_obj in before_objects:
            if b_obj.object_id in matched_before:
                continue

            for a_obj in after_objects:
                if a_obj.object_id in matched_after:
                    continue

                if self._objects_color_changed(b_obj, a_obj):
                    diff.color_changed.append((b_obj, a_obj))
                    matched_before.add(b_obj.object_id)
                    matched_after.add(a_obj.object_id)
                    break

        # Remaining unmatched objects
        for b_obj in before_objects:
            if b_obj.object_id not in matched_before:
                diff.disappeared.append(b_obj)

        for a_obj in after_objects:
            if a_obj.object_id not in matched_after:
                diff.appeared.append(a_obj)

        return diff

    def _objects_match(self, obj1: DetectedObject, obj2: DetectedObject) -> bool:
        """Check if two objects are the same (same color, same position)."""
        if obj1.color != obj2.color:
            return False

        # Check if bounding boxes overlap significantly
        bb1, bb2 = obj1.bounding_box, obj2.bounding_box
        overlap = self._bbox_overlap(bb1, bb2)
        min_area = min(obj1.area, obj2.area)

        return overlap >= min_area * 0.8

    def _objects_moved(self, obj1: DetectedObject, obj2: DetectedObject) -> bool:
        """Check if obj2 is obj1 that has moved."""
        if obj1.color != obj2.color:
            return False

        # Similar size
        size_ratio = obj1.area / obj2.area if obj2.area > 0 else 0
        if not (0.7 <= size_ratio <= 1.3):
            return False

        # Not too far apart (within reasonable movement distance)
        dist = np.sqrt(
            (obj1.center[0] - obj2.center[0]) ** 2
            + (obj1.center[1] - obj2.center[1]) ** 2
        )

        # Movement should be noticeable but not too far
        return 2 < dist < 20

    def _objects_color_changed(
        self, obj1: DetectedObject, obj2: DetectedObject
    ) -> bool:
        """Check if obj2 is obj1 with changed color."""
        if obj1.color == obj2.color:
            return False

        # Same position (bounding boxes overlap significantly)
        bb1, bb2 = obj1.bounding_box, obj2.bounding_box
        overlap = self._bbox_overlap(bb1, bb2)
        min_area = min(obj1.area, obj2.area)

        return overlap >= min_area * 0.6

    def _bbox_overlap(
        self,
        bb1: tuple[int, int, int, int],
        bb2: tuple[int, int, int, int],
    ) -> int:
        """Calculate overlap area between two bounding boxes."""
        min_row1, min_col1, max_row1, max_col1 = bb1
        min_row2, min_col2, max_row2, max_col2 = bb2

        overlap_min_row = max(min_row1, min_row2)
        overlap_max_row = min(max_row1, max_row2)
        overlap_min_col = max(min_col1, min_col2)
        overlap_max_col = min(max_col1, max_col2)

        if overlap_min_row > overlap_max_row or overlap_min_col > overlap_max_col:
            return 0

        return (overlap_max_row - overlap_min_row + 1) * (
            overlap_max_col - overlap_min_col + 1
        )

    def describe_frame(self, frame: np.ndarray | list) -> str:
        """
        Generate a human-readable description of objects in a frame.

        Args:
            frame: 2D array of color values

        Returns:
            String description of detected objects
        """
        objects = self.detect_objects(frame)

        if not objects:
            return "No significant objects detected (only background)"

        # Group by color
        by_color: dict[int, list[DetectedObject]] = {}
        for obj in objects:
            if obj.color not in by_color:
                by_color[obj.color] = []
            by_color[obj.color].append(obj)

        lines = [f"Detected {len(objects)} objects:"]

        for color in sorted(by_color.keys()):
            color_objects = by_color[color]
            lines.append(f"\nColor {color}: {len(color_objects)} objects")

            for obj in color_objects[:10]:  # Limit per color
                shape = "block" if obj.is_rectangular else "shape"
                lines.append(
                    f"  - {obj.width}x{obj.height} {shape} at "
                    f"row {obj.bounding_box[0]}, col {obj.bounding_box[1]}"
                )

        return "\n".join(lines)


def create_object_detector(
    min_object_size: int = 4,
    background_colors: set[int] | None = None,
) -> ObjectDetector:
    """Factory function to create an object detector."""
    return ObjectDetector(
        min_object_size=min_object_size,
        background_colors=background_colors,
    )
