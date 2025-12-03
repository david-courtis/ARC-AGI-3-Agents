"""
Frame diff computation for the Learning Agent.

This module handles pixel-by-pixel comparison of game frames
and produces structured diff results.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from .models import DiffResult, PixelChange
from .object_detection import ObjectDetector


def frame_to_ascii(frame: np.ndarray, separator: str = "|") -> str:
    """
    Convert a frame to ASCII representation.

    Each cell shows its color value (0-9, A-F for 10-15).
    Cells are separated by the given separator.

    Args:
        frame: 2D numpy array of color values
        separator: Character to separate cells (default: |)

    Returns:
        ASCII string representation of the frame
    """
    # Map values to single characters (0-9, then A-F for 10-15)
    def val_to_char(v: int) -> str:
        if v < 10:
            return str(v)
        elif v < 16:
            return chr(ord('A') + v - 10)
        else:
            return '?'

    lines = []
    for row in frame:
        cells = [val_to_char(int(v)) for v in row]
        lines.append(separator.join(cells))

    return '\n'.join(lines)


def diff_to_ascii(before: np.ndarray, after: np.ndarray, separator: str = "|") -> str:
    """
    Create an ASCII diff showing where changes occurred.

    Uses '.' for unchanged cells and the new value for changed cells.
    Highlights the changes for easy visual inspection.

    Args:
        before: 2D numpy array of before state
        after: 2D numpy array of after state
        separator: Character to separate cells

    Returns:
        ASCII string with '.' for unchanged, new value for changed
    """
    min_shape = (
        min(before.shape[0], after.shape[0]),
        min(before.shape[1], after.shape[1]),
    )

    def val_to_char(v: int) -> str:
        if v < 10:
            return str(v)
        elif v < 16:
            return chr(ord('A') + v - 10)
        else:
            return '?'

    lines = []
    for row in range(min_shape[0]):
        cells = []
        for col in range(min_shape[1]):
            before_val = int(before[row, col])
            after_val = int(after[row, col])
            if before_val == after_val:
                cells.append('.')
            else:
                # Show the new value for changed cells
                cells.append(val_to_char(after_val))
        lines.append(separator.join(cells))

    return '\n'.join(lines)


def compute_sequential_diffs(frames: list, include_object_analysis: bool = True) -> list[dict]:
    """
    Compute ASCII diffs between consecutive animation frames.

    Args:
        frames: List of 2D frames (each frame is a 2D array/list)
        include_object_analysis: Whether to include object-level change analysis

    Returns:
        List of dicts with 'from_frame', 'to_frame', 'diff_ascii', 'pixel_count', 'object_changes'
    """
    if len(frames) <= 1:
        return []

    # Lazily create object detector for animation frame analysis
    object_detector = None
    if include_object_analysis:
        try:
            object_detector = ObjectDetector(min_object_size=4)
        except Exception:
            pass

    diffs = []
    for i in range(len(frames) - 1):
        before = np.array(frames[i])
        after = np.array(frames[i + 1])

        diff_ascii = diff_to_ascii(before, after)
        pixel_count = np.sum(before != after)

        diff_entry = {
            'from_frame': i,
            'to_frame': i + 1,
            'diff_ascii': diff_ascii,
            'pixel_count': int(pixel_count),
            'object_changes': "",
        }

        # Add object-level analysis if available and there are changes
        if object_detector and pixel_count > 0:
            try:
                object_diff = object_detector.compare_frames(before, after)
                diff_entry['object_changes'] = object_diff.describe()
            except Exception:
                pass

        diffs.append(diff_entry)

    return diffs


class FrameDiffer(ABC):
    """Abstract base class for frame differencing strategies."""

    @abstractmethod
    def compute_diff(
        self, before_frame: Any, after_frame: Any
    ) -> DiffResult:
        """Compute the difference between two frames."""
        ...


class PixelDiffer(FrameDiffer):
    """
    Pixel-by-pixel frame differencing.

    Compares frames at the pixel level and produces a structured
    summary of changes including regions affected.
    """

    def __init__(self, grid_size: int = 64, use_object_detection: bool = True):
        self.grid_size = grid_size
        self.use_object_detection = use_object_detection
        if use_object_detection:
            self.object_detector = ObjectDetector(min_object_size=4)

    def compute_diff(
        self, before_frame: list | np.ndarray, after_frame: list | np.ndarray
    ) -> DiffResult:
        """
        Compute pixel-by-pixel diff between frames.

        Args:
            before_frame: The frame before the action (64x64 grid or nested list)
            after_frame: The frame after the action

        Returns:
            DiffResult with all changes catalogued
        """
        # Convert to numpy if needed
        before = self._to_numpy(before_frame)
        after = self._to_numpy(after_frame)

        # Handle dimension mismatches
        min_shape = (
            min(before.shape[0], after.shape[0]),
            min(before.shape[1], after.shape[1]),
        )

        changed_pixels: list[PixelChange] = []
        region_changes: dict[str, int] = defaultdict(int)

        for row in range(min_shape[0]):
            for col in range(min_shape[1]):
                before_val = int(before[row, col])
                after_val = int(after[row, col])

                if before_val != after_val:
                    changed_pixels.append(
                        PixelChange(
                            row=row,
                            col=col,
                            old_value=before_val,
                            new_value=after_val,
                        )
                    )
                    # Track region
                    region = self._get_region_name(row, col)
                    region_changes[region] += 1

        has_changes = len(changed_pixels) > 0
        change_summary = self._build_summary(changed_pixels, region_changes)
        change_regions = [
            f"{region}: {count} pixels" for region, count in region_changes.items()
        ]

        # Generate ASCII representations
        before_ascii = frame_to_ascii(before)
        after_ascii = frame_to_ascii(after)
        diff_ascii_str = diff_to_ascii(before, after)

        # Object-level analysis
        before_objects = ""
        after_objects = ""
        object_changes = ""

        if self.use_object_detection:
            before_objects = self.object_detector.describe_frame(before)
            after_objects = self.object_detector.describe_frame(after)
            object_diff = self.object_detector.compare_frames(before, after)
            object_changes = object_diff.describe()

        return DiffResult(
            changed_pixels=changed_pixels,
            change_summary=change_summary,
            has_changes=has_changes,
            change_regions=change_regions,
            before_ascii=before_ascii,
            after_ascii=after_ascii,
            diff_ascii=diff_ascii_str,
            before_objects=before_objects,
            after_objects=after_objects,
            object_changes=object_changes,
        )

    def _to_numpy(self, frame: list | np.ndarray) -> np.ndarray:
        """Convert frame to numpy array, handling nested lists.

        For multi-frame animations, takes the LAST frame to capture
        the final state after the action completes.
        """
        if isinstance(frame, np.ndarray):
            # Handle 3D arrays (e.g., [N, 64, 64] where N is animation frames)
            if frame.ndim == 3:
                # Take LAST frame (in case of animation)
                return frame[-1]
            return frame

        # Handle nested list structure
        if isinstance(frame, list):
            if len(frame) > 0 and isinstance(frame[0], list):
                if len(frame[0]) > 0 and isinstance(frame[0][0], list):
                    # 3D list: [[[...]]] - multiple animation frames
                    # Take LAST frame to see final state after action
                    return np.array(frame[-1])
                # 2D list: [[...]] - single frame
                return np.array(frame)

        return np.array(frame)

    def _get_region_name(self, row: int, col: int) -> str:
        """Get a human-readable region name for a pixel position."""
        # Define regions based on typical game layout
        if row < 4:
            return "top_ui"  # Likely UI elements (health, energy, etc.)
        elif row >= 60:
            return "bottom_ui"  # Likely UI elements (key display, etc.)
        elif col < 4:
            return "left_edge"
        elif col >= 60:
            return "right_edge"
        else:
            # Divide main play area into quadrants
            v_region = "upper" if row < 32 else "lower"
            h_region = "left" if col < 32 else "right"
            return f"{v_region}_{h_region}_playarea"

    def _build_summary(
        self, changes: list[PixelChange], regions: dict[str, int]
    ) -> str:
        """Build a human-readable summary of changes."""
        if not changes:
            return "No changes detected. The action had no visible effect on the game state."

        total = len(changes)
        summary_parts = [f"Total: {total} pixels changed."]

        # Summarize by region
        if regions:
            region_strs = [f"{r} ({c})" for r, c in sorted(regions.items(), key=lambda x: -x[1])]
            summary_parts.append(f"Regions affected: {', '.join(region_strs)}")

        # Identify potential player movement
        playarea_changes = sum(
            v for k, v in regions.items() if "playarea" in k
        )
        ui_changes = sum(v for k, v in regions.items() if "ui" in k)

        if playarea_changes > 0 and ui_changes == 0:
            summary_parts.append("Changes appear to be movement in the play area.")
        elif ui_changes > 0 and playarea_changes == 0:
            summary_parts.append("Changes appear to be UI updates only.")
        elif playarea_changes > 0 and ui_changes > 0:
            summary_parts.append("Changes in both play area and UI elements.")

        # Calculate bounding box of changes
        if changes:
            min_row = min(c.row for c in changes)
            max_row = max(c.row for c in changes)
            min_col = min(c.col for c in changes)
            max_col = max(c.col for c in changes)
            summary_parts.append(
                f"Change bounding box: rows [{min_row}-{max_row}], cols [{min_col}-{max_col}]"
            )

        return " ".join(summary_parts)


class SmartDiffer(FrameDiffer):
    """
    Intelligent frame differencing with pattern detection.

    Extends basic pixel diffing with:
    - Object movement detection
    - Color pattern analysis
    - Change classification
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.pixel_differ = PixelDiffer(grid_size)

    def compute_diff(
        self, before_frame: list | np.ndarray, after_frame: list | np.ndarray
    ) -> DiffResult:
        """Compute diff with smart analysis."""
        # Start with basic pixel diff
        result = self.pixel_differ.compute_diff(before_frame, after_frame)

        # Enhance summary with pattern detection
        if result.has_changes:
            patterns = self._detect_patterns(result.changed_pixels)
            if patterns:
                result.change_summary += f" Detected patterns: {', '.join(patterns)}"

        return result

    def _detect_patterns(self, changes: list[PixelChange]) -> list[str]:
        """Detect common patterns in changes."""
        patterns = []

        if not changes:
            return patterns

        # Check for horizontal movement pattern
        rows = [c.row for c in changes]
        cols = [c.col for c in changes]

        row_range = max(rows) - min(rows)
        col_range = max(cols) - min(cols)

        if row_range <= 4 and col_range > 4:
            patterns.append("horizontal_movement")
        elif col_range <= 4 and row_range > 4:
            patterns.append("vertical_movement")
        elif row_range <= 4 and col_range <= 4:
            patterns.append("localized_change")

        # Check for consistent value changes (e.g., player sprite)
        old_values = set(c.old_value for c in changes)
        new_values = set(c.new_value for c in changes)

        if len(old_values) <= 2 and len(new_values) <= 2:
            patterns.append("simple_swap")

        return patterns


def create_differ(strategy: str = "smart") -> FrameDiffer:
    """Factory function to create a frame differ."""
    if strategy == "pixel":
        return PixelDiffer()
    elif strategy == "smart":
        return SmartDiffer()
    else:
        raise ValueError(f"Unknown differ strategy: {strategy}")
