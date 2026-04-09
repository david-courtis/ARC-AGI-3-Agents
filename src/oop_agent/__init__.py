"""
OOP World Model Agent — primary research agent.

Synthesizes polymorphic, object-centric world models where:
- Each object type is a separate class with respond_to_action
- Each action is a separate class that dispatches to objects
- A Domain ties perception, actions, and rendering together

Usage:
    python main.py -a oopagent
"""

from .agent import OOPAgent

__all__ = ["OOPAgent"]
