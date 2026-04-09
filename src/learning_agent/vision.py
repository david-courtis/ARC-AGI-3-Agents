"""
Vision utilities for the Learning Agent.

Handles frame capture, rendering, and image processing.
"""

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Color palette for rendering game frames
DEFAULT_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # Black
    1: (0, 116, 217),  # Blue
    2: (255, 0, 0),  # Red
    3: (46, 204, 64),  # Green
    4: (0, 255, 0),  # Lime
    5: (255, 220, 0),  # Yellow
    6: (0, 0, 255),  # Blue
    7: (255, 255, 0),  # Yellow
    8: (255, 165, 0),  # Orange
    9: (128, 0, 128),  # Purple
    10: (255, 255, 255),  # White
    11: (128, 128, 128),  # Gray
    12: (0, 255, 255),  # Cyan
    13: (255, 0, 255),  # Magenta
    14: (255, 192, 203),  # Pink
    15: (165, 42, 42),  # Brown
}


class FrameRenderer(ABC):
    """Abstract base class for rendering game frames."""

    @abstractmethod
    def render(self, frame: Any) -> Image.Image:
        """Render a frame to a PIL Image."""
        ...

    @abstractmethod
    def render_to_file(self, frame: Any, path: str | Path) -> str:
        """Render a frame and save to file. Returns the path."""
        ...

    @abstractmethod
    def render_to_base64(self, frame: Any) -> str:
        """Render a frame to base64-encoded PNG."""
        ...


class GridFrameRenderer(FrameRenderer):
    """
    Renders grid-based game frames as images.

    Converts integer grid values to colored pixels using a palette.
    """

    def __init__(
        self,
        color_palette: dict[int, tuple[int, int, int]] | None = None,
        scale: int = 8,
        grid_size: int = 64,
    ):
        self.color_palette = color_palette or DEFAULT_COLOR_PALETTE
        self.scale = scale
        self.grid_size = grid_size

    def render(self, frame: Any) -> Image.Image:
        """
        Render a game frame to a PIL Image.

        Args:
            frame: A 2D grid of integer color values, or nested list structure

        Returns:
            PIL Image with scaled pixels
        """
        # Convert to 2D numpy array
        grid = self._normalize_frame(frame)

        # Create RGB image
        height, width = grid.shape
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color_idx = int(grid[y, x])
                color = self.color_palette.get(color_idx, (128, 128, 128))
                img_array[y, x] = color

        # Create image and scale up
        img = Image.fromarray(img_array, mode="RGB")
        if self.scale > 1:
            img = img.resize(
                (width * self.scale, height * self.scale),
                resample=Image.Resampling.NEAREST,
            )

        return img

    def render_to_file(self, frame: Any, path: str | Path) -> str:
        """Render frame and save to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        img = self.render(frame)
        img.save(str(path), "PNG")

        return str(path)

    def render_to_base64(self, frame: Any) -> str:
        """Render frame to base64-encoded PNG string."""
        img = self.render(frame)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def _normalize_frame(self, frame: Any) -> np.ndarray:
        """Convert various frame formats to 2D numpy array.

        For multi-frame animations, takes the LAST frame to capture
        the final state after the action completes.
        """
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3:
                # Take LAST frame (in case of animation)
                return frame[-1]
            return frame

        # Handle nested list structure from game
        if isinstance(frame, list):
            if len(frame) > 0 and isinstance(frame[0], list):
                if len(frame[0]) > 0 and isinstance(frame[0][0], list):
                    # 3D list: [[[...]]] - multiple animation frames
                    # Take LAST frame to see final state after action
                    return np.array(frame[-1])
                # 2D list: [[...]] - single frame
                return np.array(frame)

        return np.array(frame)


class FrameCapture:
    """
    Manages frame capture and storage.

    Provides a clean interface for capturing game frames
    and storing them as images.
    """

    def __init__(
        self,
        renderer: FrameRenderer | None = None,
        output_dir: str | Path = "frames",
    ):
        self.renderer = renderer or GridFrameRenderer()
        self.output_dir = Path(output_dir)
        self.capture_count = 0

    def capture(
        self,
        frame: Any,
        label: str | None = None,
    ) -> str:
        """
        Capture a frame and save to file.

        Args:
            frame: The game frame to capture
            label: Optional label for the filename

        Returns:
            Path to the saved image file
        """
        self.capture_count += 1

        if label:
            filename = f"{label}.png"
        else:
            filename = f"frame_{self.capture_count:04d}.png"

        path = self.output_dir / filename
        return self.renderer.render_to_file(frame, path)

    def capture_pair(
        self,
        before_frame: Any,
        after_frame: Any,
        prefix: str,
    ) -> tuple[str, str]:
        """
        Capture a before/after frame pair.

        Args:
            before_frame: Frame before action
            after_frame: Frame after action
            prefix: Prefix for filenames

        Returns:
            Tuple of (before_path, after_path)
        """
        before_path = self.capture(before_frame, f"{prefix}_before")
        after_path = self.capture(after_frame, f"{prefix}_after")
        return before_path, after_path

    def to_base64(self, frame: Any) -> str:
        """Convert frame to base64 for LLM APIs."""
        return self.renderer.render_to_base64(frame)

    def frames_to_base64_list(self, frames: Any) -> list[str]:
        """
        Convert all animation frames to a list of base64 images.

        Args:
            frames: 3D frame data [num_frames, height, width] or 2D single frame

        Returns:
            List of base64-encoded PNG strings, one per frame
        """
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                # Multiple frames
                return [self.renderer.render_to_base64(frames[i]) for i in range(len(frames))]
            else:
                # Single 2D frame
                return [self.renderer.render_to_base64(frames)]

        if isinstance(frames, list):
            if len(frames) > 0 and isinstance(frames[0], list):
                if len(frames[0]) > 0 and isinstance(frames[0][0], list):
                    # 3D list: multiple animation frames
                    return [self.renderer.render_to_base64(frame) for frame in frames]
                # 2D list: single frame
                return [self.renderer.render_to_base64(frames)]

        return [self.renderer.render_to_base64(frames)]

    def set_output_dir(self, output_dir: str | Path) -> None:
        """Change the output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_frame_from_file(path: str | Path) -> np.ndarray:
    """Load a frame from an image file back to numpy array."""
    img = Image.open(path)
    return np.array(img)


def create_comparison_image(
    before_frame: Any,
    after_frame: Any,
    renderer: FrameRenderer | None = None,
) -> Image.Image:
    """
    Create a side-by-side comparison image.

    Useful for visual debugging and logging.
    """
    renderer = renderer or GridFrameRenderer()

    before_img = renderer.render(before_frame)
    after_img = renderer.render(after_frame)

    # Create side-by-side image
    total_width = before_img.width + after_img.width + 10  # 10px gap
    max_height = max(before_img.height, after_img.height)

    combined = Image.new("RGB", (total_width, max_height), (50, 50, 50))
    combined.paste(before_img, (0, 0))
    combined.paste(after_img, (before_img.width + 10, 0))

    return combined
