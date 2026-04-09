"""
Root conftest.py — wires vendor/arc/ into sys.path so that
`from agents.agent import Agent` and `from agents.structs import ...`
resolve to the submodule.
"""

import os
import sys

# Add vendor/arc to the front of sys.path so `agents.*` resolves there.
_vendor_arc = os.path.join(os.path.dirname(__file__), "vendor", "arc")
if _vendor_arc not in sys.path:
    sys.path.insert(0, _vendor_arc)

# Disable run logging during tests to avoid polluting results/
os.environ["ARC_DISABLE_RUN_LOGGING"] = "1"
