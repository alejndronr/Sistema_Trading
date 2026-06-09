"""Paquete de estrategias de trading."""
from strategies.base import BaseStrategy, Signal
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
]
