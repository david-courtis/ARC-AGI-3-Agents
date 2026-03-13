"""
OOP World Model Agent - Learns game mechanics via program synthesis.

This agent assumes NOTHING about the domain. It observes raw 64x64 RGB
pixel frames and anonymous actions (ACTION1-5). It:
1. Explores games by taking actions and recording frame transitions
2. Analyzes frames for structure (colors, regions, diffs)
3. Synthesizes world models (WorldModel subclasses) via LLM (CEGIS loop)
4. Verifies models against a replay buffer
5. Exploits verified models for score-maximizing play

Usage:
    python main.py -a oopagent
"""

from .agent import OOPAgent
from .synthesis import (
    CounterexampleDiagnosis,
    ModelSynthesizer,
    Transition,
    VerificationResult,
    diagnose_persistent_errors,
    verify_model,
)
from .world_model import (
    WorldModel,
    IdentityModel,
    PixelDiff,
    compute_diff,
    find_unique_colors,
    find_color_regions,
    region_bbox,
    most_common_color,
    extract_grid,
)

__all__ = [
    # Main agent
    "OOPAgent",
    # World model
    "WorldModel",
    "IdentityModel",
    "PixelDiff",
    "compute_diff",
    "find_unique_colors",
    "find_color_regions",
    "region_bbox",
    "most_common_color",
    "extract_grid",
    # Synthesis
    "ModelSynthesizer",
    "Transition",
    "VerificationResult",
    "CounterexampleDiagnosis",
    "verify_model",
    "diagnose_persistent_errors",
]
