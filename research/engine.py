"""Compatibility shim — the engine moved to vrp/engine.py.

The one-off study scripts in research/improve/ were written when the engine
lived at research/engine.py and import it as `from engine import ...` after
inserting this directory on sys.path. This shim keeps every recorded study
runnable without editing them. New code must import `vrp.engine` directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrp.engine import *          # noqa: F401,F403,E402
from vrp.engine import DATA, N    # noqa: F401,E402  (non-star extras some studies touch)
