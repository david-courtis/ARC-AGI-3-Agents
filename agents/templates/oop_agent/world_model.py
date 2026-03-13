"""
World Model: Domain-agnostic base classes for ARC-AGI-3.

The agent observes raw frames (64x64 RGB pixel grids) and anonymous actions
(ACTION1..ACTION5). It knows nothing else about the domain: no object types,
no action semantics, no spatial structure, no color meanings.

This module provides:
- WorldModel: the minimal interface — given (frame, action_id), predict the next frame
- Frame utilities: diff computation, region detection, color quantization
- No domain assumptions baked in. Everything domain-specific is synthesized by the LLM.

The LLM synthesizes a concrete WorldModel subclass that encodes whatever
internal representation it needs (objects, grids, rules, physics — all learned).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Core interface: the only contract the synthesized model must satisfy
# =============================================================================


class WorldModel(ABC):
    """
    A world model predicts the next frame given the current frame and an action.

    This is the sole interface contract. The LLM synthesizes a subclass that
    implements `predict` however it wants — it may parse the frame into objects,
    track internal state, apply rules, etc. All of that is up to the synthesis.

    The model operates on raw numpy arrays (H, W, C) representing RGB frames.
    """

    @abstractmethod
    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        """
        Predict the next frame after taking action_id in the given frame.

        Args:
            frame: Current frame as (H, W, 3) uint8 numpy array (RGB).
            action_id: Integer action ID (1-5) from the environment API.

        Returns:
            Predicted next frame as (H, W, 3) uint8 numpy array (RGB).
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state the model tracks across frames.

        Called when the game resets. Default: no-op.
        Override if the model maintains cross-frame state.
        """
        pass


class IdentityModel(WorldModel):
    """
    Trivial model that predicts no change. Used as the initial baseline
    before any synthesis has occurred.
    """

    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        return frame.copy()


# =============================================================================
# Frame utilities: domain-agnostic tools available to synthesized models
# =============================================================================


@dataclass
class PixelDiff:
    """Summary of pixel-level differences between two frames."""
    count: int
    positions: list[tuple[int, int]]  # (row, col) of changed pixels
    bbox: tuple[int, int, int, int] | None  # min_row, min_col, max_row, max_col
    before_colors: list[tuple[int, int, int]]
    after_colors: list[tuple[int, int, int]]


def compute_diff(before: np.ndarray, after: np.ndarray) -> PixelDiff:
    """
    Compute pixel-level differences between two frames.

    Works with any array shape — RGB (H,W,3), grayscale (H,W), etc.
    """
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

    # Sample colors (cap to avoid huge lists)
    sample = positions[:100]
    if before.ndim == 3:
        before_colors = [tuple(int(x) for x in before[r, c]) for r, c in sample]
        after_colors = [tuple(int(x) for x in after[r, c]) for r, c in sample]
    else:
        before_colors = [(int(before[r, c]),) for r, c in sample]
        after_colors = [(int(after[r, c]),) for r, c in sample]

    return PixelDiff(
        count=count,
        positions=positions[:1000],  # cap for memory
        bbox=bbox,
        before_colors=before_colors,
        after_colors=after_colors,
    )


def find_unique_colors(frame: np.ndarray) -> list[tuple[int, ...]]:
    """
    Find all unique colors in a frame.

    Returns list of color tuples — RGB triples for 3-channel, single values otherwise.
    """
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
    """
    Find connected regions of a specific color in a frame.

    Args:
        frame: Input frame (H,W,3) or (H,W).
        target_color: The color to look for.
        connectivity: 4 or 8 connectivity.

    Returns:
        List of pixel sets, one per connected region.
    """
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
    """Compute bounding box (min_row, min_col, max_row, max_col) for a set of pixels."""
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    return (min(rows), min(cols), max(rows), max(cols))


def most_common_color(frame: np.ndarray) -> tuple[int, ...]:
    """Find the most common color in a frame (likely the background)."""
    if frame.ndim == 3:
        flat = frame.reshape(-1, frame.shape[-1])
        # Use a hash-based approach for RGB
        hashes = flat[:, 0].astype(np.int64) * 65536 + flat[:, 1].astype(np.int64) * 256 + flat[:, 2].astype(np.int64)
        values, counts = np.unique(hashes, return_counts=True)
        best = values[np.argmax(counts)]
        r = int(best // 65536)
        g = int((best % 65536) // 256)
        b = int(best % 256)
        return (r, g, b)
    else:
        values, counts = np.unique(frame, return_counts=True)
        return (int(values[np.argmax(counts)]),)


def extract_grid(frame_data: list) -> np.ndarray | None:
    """
    Extract a numpy array from raw frame data (list of lists).

    Handles both 2D (H,W,C) and 3D (T,H,W,C) animation frame structures.
    Returns the last animation frame if multiple are present.
    """
    if not frame_data:
        return None
    try:
        arr = np.array(frame_data, dtype=np.uint8)
        if arr.ndim == 4:
            # (T, H, W, C) — multiple animation frames, use last
            return arr[-1]
        elif arr.ndim == 3:
            # (H, W, C) — single frame
            return arr
        elif arr.ndim == 2:
            # (H, W) — grayscale single frame
            return arr
        else:
            return None
    except (ValueError, TypeError):
        return None
