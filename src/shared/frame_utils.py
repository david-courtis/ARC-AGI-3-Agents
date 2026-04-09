"""
Unified frame extraction utilities for ARC-AGI-3.

Handles frame data from both the old vendor API (list[list[list[int]]]) and
the new arc-agi package (List[ndarray]).
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ARC-AGI palette: maps color indices 0-15 to RGB values.
ARC_PALETTE = np.array([
    [0, 0, 0],        # 0: Black
    [0, 116, 217],    # 1: Blue
    [255, 0, 0],      # 2: Red
    [46, 204, 64],    # 3: Green
    [0, 255, 0],      # 4: Lime
    [255, 220, 0],    # 5: Yellow
    [0, 0, 255],      # 6: Blue
    [255, 255, 0],    # 7: Yellow
    [255, 165, 0],    # 8: Orange
    [128, 0, 128],    # 9: Purple
    [255, 255, 255],  # 10: White
    [128, 128, 128],  # 11: Gray
    [0, 255, 255],    # 12: Cyan
    [255, 0, 255],    # 13: Magenta
    [255, 192, 203],  # 14: Pink
    [165, 42, 42],    # 15: Brown
], dtype=np.uint8)


def palette_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert a 2D palette-indexed grid (H, W) to (H, W, 3) RGB."""
    clamped = np.clip(grid, 0, 15)
    return ARC_PALETTE[clamped]


def extract_grid(frame_data: Any) -> np.ndarray | None:
    """Extract a (H, W, 3) RGB numpy array from frame data.

    Handles multiple formats:
    - arcengine FrameDataRaw.frame: List[ndarray] — list of layers (palette-indexed)
    - Old vendor FrameData.frame: list[list[list[int]]] — nested lists
    - numpy arrays of various shapes

    Always returns a (H, W, 3) uint8 RGB array, or None on failure.
    """
    if frame_data is None:
        return None

    # Handle List[ndarray] from arcengine FrameDataRaw.frame
    if isinstance(frame_data, list) and len(frame_data) > 0 and isinstance(frame_data[0], np.ndarray):
        # List of numpy array layers — take the last layer
        layer = frame_data[-1]
        return _ndarray_to_rgb(layer)

    # Handle raw numpy array
    if isinstance(frame_data, np.ndarray):
        return _ndarray_to_rgb(frame_data)

    # Handle nested lists (old vendor format)
    if isinstance(frame_data, list):
        if not frame_data:
            return None
        try:
            arr = np.array(frame_data, dtype=np.uint8)
            return _ndarray_to_rgb(arr)
        except (ValueError, TypeError):
            return None

    return None


def _ndarray_to_rgb(arr: np.ndarray) -> np.ndarray | None:
    """Convert a numpy array of any supported shape to (H, W, 3) RGB uint8."""
    arr = np.asarray(arr, dtype=np.uint8)

    if arr.ndim == 4:
        # [grids][rows][cols][channels] — multi-frame RGB; take last
        return arr[-1]

    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            # [rows][cols][3] — already RGB
            return arr
        # [grids][rows][cols] — palette-indexed layers; take last
        return palette_to_rgb(arr[-1])

    if arr.ndim == 2:
        # [rows][cols] — single palette-indexed grid
        return palette_to_rgb(arr)

    return None
