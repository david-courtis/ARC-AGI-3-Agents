"""
Monolithic World Model: flat predict(frame, action) -> frame interface.

Architecture
------------
The monolithic approach imposes NO structural constraints on the synthesized
code. The LLM produces a single class with one method:

    class SynthesizedModel(WorldModel):
        def predict(self, frame, action_id) -> frame

The LLM can write whatever Python it wants inside predict(): if-elif chains,
helper functions, internal data structures, inline object detection, pixel
manipulation, etc. There is no factorization requirement.

This is the baseline agent for comparing against the OOP agent. It tests
the hypothesis that imposing OOP structure improves synthesis outcomes. If
the monolithic agent achieves equal or better accuracy, the OOP structure
adds overhead without benefit.

See docs/oop-vs-monolithic.md for the full comparison and conjectures.

Design Principles
-----------------
1. **No structure imposed.** The only contract is predict(frame, action_id)
   returns a frame. Everything else is up to the LLM.

2. **Same utilities as OOP agent.** Both agents have access to the same
   frame analysis tools (compute_diff, find_unique_colors, etc.). The
   difference is purely in the synthesis prompt and base classes.

3. **Fully independent.** This module is self-contained and does not import
   from the OOP agent. The two agents can run in parallel without interaction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class WorldModel(ABC):
    """Predict the next frame given the current frame and an action."""

    @abstractmethod
    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        """
        Args:
            frame: (H, W, 3) uint8 numpy array (RGB).
            action_id: Integer action ID (1-5).
        Returns:
            Predicted next frame as (H, W, 3) uint8 numpy array.
        """
        pass

    def reset(self) -> None:
        pass


class IdentityModel(WorldModel):
    """Trivial baseline: predicts no change."""

    def predict(self, frame: np.ndarray, action_id: int) -> np.ndarray:
        return frame.copy()


# =============================================================================
# Frame utilities
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
