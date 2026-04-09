"""
Research agents for ARC-AGI-3 world model synthesis.
"""

from .oop_agent import OOPAgent
from .monolithic_agent import MonolithicAgent
from .object_centric_agent import ObjectCentricAgent

__all__ = ["OOPAgent", "MonolithicAgent", "ObjectCentricAgent"]
