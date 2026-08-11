"""vrp — variance-risk-premium options-selling strategy for Delta Exchange India.

The engine (vrp.engine) is the single source of truth for pricing, costs,
margin, and the backtest loop. Studies in research/ and runners in scripts/
all import from here.
"""
from vrp.engine import Backtest, Config  # noqa: F401
