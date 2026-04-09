"""
Monolithic Agent — baseline for OOP vs monolithic comparison.

Synthesizes flat predict(frame, action) -> frame functions with no
imposed structure. Same explore/synthesize/exploit pipeline as OOPAgent.

Usage:
    python main.py -a monolithicagent
"""

from .agent import MonolithicAgent

__all__ = ["MonolithicAgent"]
