"""
Shared exploration infrastructure used by all agents.

Extracted from the learning_agent to provide a common foundation:
- Pydantic models for state, knowledge, and LLM responses
- Knowledge management and prompt formatting
- Frame differencing with ASCII grids and object detection
- Vision pipeline (rendering, capture, base64)
- LLM client for action analysis, environment analysis, next action suggestion
- Run logger for detailed per-run logging
"""

from .models import (
    ACTION_SEMANTICS,
    ActionAnalysisResult,
    ActionID,
    ActionKnowledge,
    ActionObservation,
    AgentState,
    BoundaryInfo,
    DiffResult,
    EnvironmentAnalysisResult,
    EnvironmentKnowledge,
    EnvironmentObservation,
    IdentifiedObject,
    NextActionSuggestion,
    ObjectHypothesis,
    PixelChange,
    SuggestedActionUpdate,
)
from .knowledge import KnowledgeManager, PromptKnowledgeFormatter
from .diff import FrameDiffer, PixelDiffer, SmartDiffer, create_differ
from .object_detection import ObjectDetector, DetectedObject, ObjectDiff, create_object_detector
from .vision import FrameCapture, GridFrameRenderer
from .llm_agents import LLMAgent, OpenAIClientAgent, create_agent
from .run_logger import RunLogger, ConsoleLogger
from .exploration_engine import ExplorationEngine
from .agent_base import SynthesisAgent
from .frame_utils import extract_grid, palette_to_rgb, ARC_PALETTE

__all__ = [
    # Models
    "ACTION_SEMANTICS",
    "ActionID",
    "ActionObservation",
    "ActionKnowledge",
    "EnvironmentKnowledge",
    "EnvironmentAnalysisResult",
    "EnvironmentObservation",
    "BoundaryInfo",
    "IdentifiedObject",
    "ObjectHypothesis",
    "SuggestedActionUpdate",
    "DiffResult",
    "PixelChange",
    "ActionAnalysisResult",
    "NextActionSuggestion",
    "AgentState",
    # Knowledge
    "KnowledgeManager",
    "PromptKnowledgeFormatter",
    # Diff
    "FrameDiffer",
    "PixelDiffer",
    "SmartDiffer",
    "create_differ",
    # Object detection
    "ObjectDetector",
    "DetectedObject",
    "ObjectDiff",
    "create_object_detector",
    # Vision
    "FrameCapture",
    "GridFrameRenderer",
    # LLM
    "LLMAgent",
    "OpenAIClientAgent",
    "create_agent",
    # Logging
    "RunLogger",
    "ConsoleLogger",
    # Exploration engine
    "ExplorationEngine",
    # Agent base
    "SynthesisAgent",
    # Frame utils
    "extract_grid",
    "palette_to_rgb",
    "ARC_PALETTE",
]
