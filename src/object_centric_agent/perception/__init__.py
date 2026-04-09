"""
Perception pipeline: raw RGB frames → structured object representations.

The pipeline has three stages:
1. fragment_detector: color-CC → Fragment objects (deterministic, single-frame)
2. sprite_registry: maintains SpriteType library, merges co-moving fragments
3. object_tracker: persistent identity across frames (track_id assignment)
4. frame_parser: orchestrates the full pipeline
"""

from .fragment_detector import FragmentDetector
from .sprite_registry import SpriteRegistry
from .comovement_tracker import ComovementTracker
from .object_tracker import ObjectTracker
from .frame_parser import FrameParser

__all__ = [
    "FragmentDetector",
    "SpriteRegistry",
    "ComovementTracker",
    "ObjectTracker",
    "FrameParser",
]
