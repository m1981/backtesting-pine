"""
Core type definitions for the Pine backtesting framework.

This module contains all enums and type definitions used throughout the framework.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any


class Mode(Enum):
    """Trading mode for the strategy."""
    LONG_ONLY = auto()
    SHORT_ONLY = auto()
    LONG_SHORT = auto()


class Bias(Enum):
    """Market bias determined by higher timeframe analysis."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


class State(Enum):
    """
    Strategy state machine states.
    
    The strategy progresses through these states:
    IDLE -> WAITING_CONFIRMATION -> WAITING_TRIGGER -> IN_POSITION -> IDLE
    """
    IDLE = auto()
    WAITING_CONFIRMATION = auto()
    WAITING_TRIGGER = auto()
    IN_POSITION = auto()


class SetupSignal(Enum):
    """Signal types from setup detection systems."""
    NONE = auto()      # No setup detected
    VALID = auto()     # Valid setup detected
    REJECTED = auto()  # Setup detected but rejected (e.g., in chop zone)


class ExitReason(Enum):
    """Reasons for exiting a position."""
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_SIGNAL = auto()
    BIAS_CHANGED = auto()
    MANUAL = auto()
    TIMEOUT = auto()


class Timeframe(Enum):
    """
    Supported timeframes for data and analysis.
    
    Values represent the number of minutes in each timeframe.
    """
    MINUTE_1 = 1
    MINUTE_5 = 5
    MINUTE_15 = 15
    MINUTE_30 = 30
    HOUR_1 = 60
    HOUR_2 = 120
    HOUR_4 = 240
    DAILY = 1440
    WEEKLY = 10080
    MONTHLY = 43200
    
    @classmethod
    def from_string(cls, s: str) -> "Timeframe":
        """
        Parse a timeframe string like '1H', '4H', '1D', etc.
        
        Args:
            s: Timeframe string (e.g., '1M', '5M', '15M', '1H', '4H', '1D', '1W')
            
        Returns:
            Corresponding Timeframe enum value
            
        Raises:
            ValueError: If the string cannot be parsed
        """
        mapping = {
            "1M": cls.MINUTE_1,
            "1m": cls.MINUTE_1,
            "5M": cls.MINUTE_5,
            "5m": cls.MINUTE_5,
            "15M": cls.MINUTE_15,
            "15m": cls.MINUTE_15,
            "30M": cls.MINUTE_30,
            "30m": cls.MINUTE_30,
            "1H": cls.HOUR_1,
            "1h": cls.HOUR_1,
            "60M": cls.HOUR_1,
            "60m": cls.HOUR_1,
            "2H": cls.HOUR_2,
            "2h": cls.HOUR_2,
            "4H": cls.HOUR_4,
            "4h": cls.HOUR_4,
            "1D": cls.DAILY,
            "1d": cls.DAILY,
            "D": cls.DAILY,
            "d": cls.DAILY,
            "1W": cls.WEEKLY,
            "1w": cls.WEEKLY,
            "W": cls.WEEKLY,
            "w": cls.WEEKLY,
            "1MO": cls.MONTHLY,
            "1mo": cls.MONTHLY,
            "MO": cls.MONTHLY,
            "mo": cls.MONTHLY,
        }
        if s not in mapping:
            raise ValueError(f"Unknown timeframe: {s}. Valid values: {list(mapping.keys())}")
        return mapping[s]
    
    def to_pandas_freq(self) -> str:
        """Convert to pandas frequency string for resampling."""
        mapping = {
            Timeframe.MINUTE_1: "1min",
            Timeframe.MINUTE_5: "5min",
            Timeframe.MINUTE_15: "15min",
            Timeframe.MINUTE_30: "30min",
            Timeframe.HOUR_1: "1h",
            Timeframe.HOUR_2: "2h",
            Timeframe.HOUR_4: "4h",
            Timeframe.DAILY: "1D",
            Timeframe.WEEKLY: "1W",
            Timeframe.MONTHLY: "1ME",
        }
        return mapping[self]
    
    def to_yfinance_interval(self) -> str:
        """Convert to yfinance interval string."""
        mapping = {
            Timeframe.MINUTE_1: "1m",
            Timeframe.MINUTE_5: "5m",
            Timeframe.MINUTE_15: "15m",
            Timeframe.MINUTE_30: "30m",
            Timeframe.HOUR_1: "1h",
            Timeframe.HOUR_2: "1h",  # yfinance doesn't support 2h, we'll resample
            Timeframe.HOUR_4: "1h",  # yfinance doesn't support 4h, we'll resample
            Timeframe.DAILY: "1d",
            Timeframe.WEEKLY: "1wk",
            Timeframe.MONTHLY: "1mo",
        }
        return mapping[self]


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    description: str = ""
    mode: Mode = Mode.LONG_ONLY
    initial_capital: float = 10000.0
    commission: float = 0.002  # 0.2%
    slippage: float = 0.001   # 0.1%
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    
@dataclass
class TradeMetadata:
    """Metadata attached to a trade for analysis."""
    entry_reason: str = ""
    exit_reason: str = ""
    setup_type: str = ""
    daily_macd: float = 0.0
    setup_distance: float = 0.0
    atr_at_entry: float = 0.0
    extra: dict = field(default_factory=dict)

