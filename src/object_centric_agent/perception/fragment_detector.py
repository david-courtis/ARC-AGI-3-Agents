"""
Fragment detection via color-based connected components.

Given an RGB frame, extracts all single-color connected components as Fragment
objects. This is the Level 0 perception — deterministic, parameter-free (aside
from background detection and minimum size filtering).

Algorithm:
    1. Detect background color (most frequent color in the frame).
    2. For each non-background color present in the frame:
       a. Create a binary mask of pixels matching that color.
       b. Run connected-component labeling with 8-connectivity.
       c. For each component, create a Fragment object.
    3. Filter fragments below minimum size (noise removal).
    4. Return all fragments sorted by position (top-left to bottom-right).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..state.object_state import Fragment, FragmentID


class FragmentDetector:
    """
    Extracts single-color connected components from RGB frames.

    This is the ground-level perception: no learning, no state, purely
    geometric. Each call to detect() is independent.
    """

    def __init__(self, min_fragment_size: int = 1):
        """
        Args:
            min_fragment_size: Minimum pixel count for a fragment. Fragments
                smaller than this are discarded as noise. Default 1 means
                keep everything (even single pixels). For noisy games, set
                to 3-4.
        """
        self.min_fragment_size = min_fragment_size

        # 8-connectivity structuring element
        self._structure_8 = np.ones((3, 3), dtype=np.int32)

    def detect(
        self,
        frame: np.ndarray,
        background_color: tuple[int, int, int] | None = None,
    ) -> tuple[list[Fragment], tuple[int, int, int]]:
        """
        Extract all fragments from an RGB frame.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            background_color: If provided, skip this color. If None, auto-detect
                as the most frequent color in the frame.

        Returns:
            (fragments, background_color) — list of Fragment objects sorted by
            position, and the detected/provided background color.
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, (
            f"Expected (H, W, 3) RGB frame, got shape {frame.shape}"
        )

        if background_color is None:
            background_color = self._detect_background(frame)

        # Find all unique colors (as (R,G,B) tuples)
        flat = frame.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)

        fragments: list[Fragment] = []
        bg = np.array(background_color, dtype=np.uint8)

        for color_arr in unique_colors:
            color = (int(color_arr[0]), int(color_arr[1]), int(color_arr[2]))

            # Skip background
            if np.array_equal(color_arr, bg):
                continue

            # Binary mask for this color
            mask = np.all(frame == color_arr, axis=-1).astype(np.int32)

            # Connected components with 8-connectivity
            labeled, num_components = ndimage.label(mask, structure=self._structure_8)

            for comp_idx in range(1, num_components + 1):
                comp_mask = labeled == comp_idx
                pixels = frozenset(
                    (int(r), int(c)) for r, c in zip(*np.where(comp_mask))
                )

                if len(pixels) < self.min_fragment_size:
                    continue

                frag = Fragment.from_pixels(
                    color=color,
                    component_index=len(fragments),  # global index
                    pixels=pixels,
                )
                fragments.append(frag)

        # Sort by position: top-left to bottom-right
        fragments.sort(key=lambda f: (f.bbox[0], f.bbox[1]))

        return fragments, background_color

    def _detect_background(self, frame: np.ndarray) -> tuple[int, int, int]:
        """Detect background as the most frequent color."""
        flat = frame.reshape(-1, 3)
        # Hash colors for fast counting
        hashes = (
            flat[:, 0].astype(np.int64) * 65536
            + flat[:, 1].astype(np.int64) * 256
            + flat[:, 2].astype(np.int64)
        )
        values, counts = np.unique(hashes, return_counts=True)
        best = values[np.argmax(counts)]
        return (int(best // 65536), int((best % 65536) // 256), int(best % 256))
