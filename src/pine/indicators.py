"""
Technical indicators for the Pine backtesting framework.

This module provides implementations of common technical indicators
used in trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from dataclasses import dataclass

from .core.series import Series


def ema(data: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price data (typically close prices)
        period: EMA period

    Returns:
        Array of EMA values
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean().values
    else:
        # Manual EMA calculation for numpy arrays
        alpha = 2 / (period + 1)
        ema_values = np.empty_like(data, dtype=np.float64)
        ema_values[0] = data[0]

        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]

        return ema_values


def sma(data: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price data
        period: SMA period

    Returns:
        Array of SMA values
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=period).mean().values
    else:
        result = np.empty_like(data, dtype=np.float64)
        result[:period-1] = np.nan
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result


@dataclass
class MACDResult:
    """Result from MACD calculation."""
    macd_line: Series
    signal_line: Series
    histogram: Series

    def __repr__(self) -> str:
        return (f"MACDResult(macd={self.macd_line.current:.4f}, "
                f"signal={self.signal_line.current:.4f}, "
                f"histogram={self.histogram.current:.4f})")


class MACD:
    """
    Moving Average Convergence Divergence (MACD) indicator.

    MACD is calculated as:
    - MACD Line = EMA(fast) - EMA(slow)
    - Signal Line = EMA(MACD Line, signal_period)
    - Histogram = MACD Line - Signal Line

    Usage:
        macd = MACD(close_prices, fast=12, slow=26, signal=9)
        result = macd.calculate()

        # Access components as Series
        result.macd_line.current
        result.signal_line.current
        result.histogram.current
    """

    def __init__(
        self,
        source: Union[pd.Series, np.ndarray, Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ):
        """
        Initialize MACD indicator.

        Args:
            source: Price data (typically close prices)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
        """
        if isinstance(source, Series):
            # Series data is stored with most recent first (index 0 = newest)
            # We need to reverse it for chronological calculation
            self.source = source.to_array()[::-1]  # Now oldest first
        elif isinstance(source, pd.Series):
            self.source = source.values
        else:
            self.source = np.array(source, dtype=np.float64)

        self.fast = fast
        self.slow = slow
        self.signal = signal

        # Pre-calculate all values
        self._macd_line = None
        self._signal_line = None
        self._histogram = None
        self._calculate()

    def _calculate(self) -> None:
        """Calculate MACD components."""
        # Calculate EMAs (source is in chronological order: oldest first)
        fast_ema = ema(self.source, self.fast)
        slow_ema = ema(self.source, self.slow)

        # MACD line = fast EMA - slow EMA
        macd_line = fast_ema - slow_ema

        # Signal line = EMA of MACD line
        signal_line = ema(macd_line, self.signal)

        # Histogram = MACD line - Signal line
        histogram = macd_line - signal_line

        # Reverse back so index 0 = most recent (matching Series convention)
        self._macd_line_data = macd_line[::-1]
        self._signal_line_data = signal_line[::-1]
        self._histogram_data = histogram[::-1]

        # Cache Series objects so set_index() works correctly
        self._macd_line_series = Series(self._macd_line_data, name="MACD")
        self._signal_line_series = Series(self._signal_line_data, name="Signal")
        self._histogram_series = Series(self._histogram_data, name="Histogram")

    def calculate(self) -> MACDResult:
        """
        Get MACD calculation results as Series objects.

        Returns:
            MACDResult with macd_line, signal_line, and histogram as Series
        """
        return MACDResult(
            macd_line=self._macd_line_series,
            signal_line=self._signal_line_series,
            histogram=self._histogram_series
        )

    @property
    def macd_line(self) -> Series:
        """Get MACD line as Series."""
        return self._macd_line_series

    @property
    def signal_line(self) -> Series:
        """Get signal line as Series."""
        return self._signal_line_series

    @property
    def histogram(self) -> Series:
        """Get histogram as Series."""
        return self._histogram_series


class ATR:
    """
    Average True Range (ATR) indicator.

    ATR measures market volatility by calculating the average of true ranges
    over a specified period.

    True Range is the maximum of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|

    Usage:
        atr = ATR(high, low, close, period=14)
        atr_series = atr.calculate()
        atr_series.current  # Current ATR value
    """

    def __init__(
        self,
        high: Union[pd.Series, np.ndarray, Series],
        low: Union[pd.Series, np.ndarray, Series],
        close: Union[pd.Series, np.ndarray, Series],
        period: int = 14
    ):
        """
        Initialize ATR indicator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 14)
        """
        # Convert to numpy arrays
        # Series data is stored with most recent first (index 0 = newest)
        # We need to reverse it for chronological calculation
        if isinstance(high, Series):
            self.high = high.to_array()[::-1]  # Now oldest first
        elif isinstance(high, pd.Series):
            self.high = high.values
        else:
            self.high = np.array(high, dtype=np.float64)

        if isinstance(low, Series):
            self.low = low.to_array()[::-1]  # Now oldest first
        elif isinstance(low, pd.Series):
            self.low = low.values
        else:
            self.low = np.array(low, dtype=np.float64)

        if isinstance(close, Series):
            self.close = close.to_array()[::-1]  # Now oldest first
        elif isinstance(close, pd.Series):
            self.close = close.values
        else:
            self.close = np.array(close, dtype=np.float64)

        self.period = period

        # Pre-calculate
        self._atr = None
        self._calculate()

    def _calculate(self) -> None:
        """Calculate ATR values."""
        # Calculate True Range (data is in chronological order: oldest first)
        tr = np.zeros(len(self.high))

        # First TR is just high - low
        tr[0] = self.high[0] - self.low[0]

        # Subsequent TRs use the formula
        for i in range(1, len(self.high)):
            hl = self.high[i] - self.low[i]
            hc = abs(self.high[i] - self.close[i - 1])
            lc = abs(self.low[i] - self.close[i - 1])
            tr[i] = max(hl, hc, lc)

        # ATR is EMA of True Range (Wilder's smoothing = EMA with alpha = 1/period)
        # Using RMA (Running Moving Average) which is Wilder's original method
        atr = np.zeros_like(tr)
        atr[0] = tr[0]

        alpha = 1 / self.period
        for i in range(1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        # Reverse back so index 0 = most recent (matching Series convention)
        self._atr_data = atr[::-1]

        # Cache Series object so set_index() works correctly
        self._atr_series = Series(self._atr_data, name="ATR")

    def calculate(self) -> Series:
        """
        Get ATR calculation results as a Series.

        Returns:
            Series containing ATR values
        """
        return self._atr_series

    @property
    def values(self) -> Series:
        """Get ATR values as Series."""
        return self._atr_series


def crossover(series_a: Union[Series, np.ndarray],
              series_b: Union[Series, np.ndarray, float]) -> bool:
    """
    Check if series_a crosses above series_b.

    Args:
        series_a: First series
        series_b: Second series or constant value

    Returns:
        True if series_a crossed above series_b on the current bar
    """
    if isinstance(series_a, Series):
        return series_a.crossed_above(series_b)

    # Handle numpy arrays
    if isinstance(series_b, (int, float)):
        return series_a[-2] <= series_b < series_a[-1]
    else:
        return series_a[-2] <= series_b[-2] and series_a[-1] > series_b[-1]


def crossunder(series_a: Union[Series, np.ndarray],
               series_b: Union[Series, np.ndarray, float]) -> bool:
    """
    Check if series_a crosses below series_b.

    Args:
        series_a: First series
        series_b: Second series or constant value

    Returns:
        True if series_a crossed below series_b on the current bar
    """
    if isinstance(series_a, Series):
        return series_a.crossed_below(series_b)

    # Handle numpy arrays
    if isinstance(series_b, (int, float)):
        return series_a[-2] >= series_b > series_a[-1]
    else:
        return series_a[-2] >= series_b[-2] and series_a[-1] < series_b[-1]


def crossed(series_a: Union[Series, np.ndarray],
            series_b: Union[Series, np.ndarray, float]) -> bool:
    """
    Check if series_a crosses series_b in either direction.

    Args:
        series_a: First series
        series_b: Second series or constant value

    Returns:
        True if series_a crossed series_b in either direction
    """
    return crossover(series_a, series_b) or crossunder(series_a, series_b)
