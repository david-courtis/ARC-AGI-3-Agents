"""
Learning Agent - A game mechanics discovery agent.

This agent learns game mechanics from scratch by:
1. Taking actions and observing state transitions
2. Building iterative understanding of what actions do
3. Building understanding of the environment
4. Validating hypotheses through repeated observations

No game-specific knowledge is baked in.

Usage:
    python main.py -a learningagent

Or programmatically:
    from agents.templates.learning_agent import LearningAgent
    agent = LearningAgent(game_id="your_game")
"""

from .agent import LearningAgent
from .diff import FrameDiffer, PixelDiffer, SmartDiffer, create_differ
from .object_detection import ObjectDetector, DetectedObject, ObjectDiff, create_object_detector
from .knowledge import KnowledgeManager, PromptKnowledgeFormatter
from .llm_agents import LLMAgent, PydanticAIAgent, OpenRouterAgent, create_agent
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
from .run_logger import ConsoleLogger, RunLogger
from .vision import FrameCapture, GridFrameRenderer

__all__ = [
    # Main agent
    "LearningAgent",
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
    # Knowledge management
    "KnowledgeManager",
    "PromptKnowledgeFormatter",
    # Diff computation
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
    # LLM Agents
    "LLMAgent",
    "PydanticAIAgent",
    "OpenRouterAgent",
    "create_agent",
    # Logging
    "RunLogger",
    "ConsoleLogger",
]
