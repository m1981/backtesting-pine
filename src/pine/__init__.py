"""
Pine - A Python backtesting framework inspired by Pine Script.

Provides an intuitive, declarative API for building multi-timeframe
trading strategies with built-in state management and risk controls.
"""

from .strategy import StrategyBase, strategy
from .backtest import Backtest, BacktestResult
from .indicators import MACD, ATR, ema, sma, crossover, crossunder
from .core.types import (
    Mode, Bias, State, SetupSignal, ExitReason,
    Timeframe, StrategyConfig
)
from .core.series import Series
from .data import Bar, TimeframeData, DataLoader

__version__ = "0.1.0"

__all__ = [
    # Main classes
    'StrategyBase',
    'strategy',
    'Backtest',
    'BacktestResult',

    # Indicators
    'MACD',
    'ATR',
    'ema',
    'sma',
    'crossover',
    'crossunder',

    # Types
    'Mode',
    'Bias',
    'State',
    'SetupSignal',
    'ExitReason',
    'Timeframe',
    'StrategyConfig',

    # Data
    'Series',
    'Bar',
    'TimeframeData',
    'DataLoader',
]
