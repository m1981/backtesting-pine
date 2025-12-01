"""
Series class for time-series data access.

The Series class provides an intuitive interface for accessing historical data
with properties like .current, .previous, and methods like .highest(), .lowest().
"""

from __future__ import annotations
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass


class Series:
    """
    A time-series data container with intuitive accessors.
    
    The Series class wraps a numpy array and provides Pine Script-like
    access patterns:
    
    - series.current     -> Current bar value (index 0)
    - series.previous    -> Previous bar value (index 1)
    - series[n]          -> N bars ago
    - series.highest(n)  -> Highest value in last N bars
    - series.lowest(n)   -> Lowest value in last N bars
    - series.average(n)  -> Average of last N bars
    
    The internal data is stored with the most recent value at index 0.
    """
    
    def __init__(self, data: Union[np.ndarray, list], name: str = ""):
        """
        Initialize a Series with data.
        
        Args:
            data: Array of values, most recent first (index 0 = current)
            name: Optional name for the series (for debugging)
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        self._data = data
        self._name = name
        self._index = 0  # Current position in the series
        
    @property
    def current(self) -> float:
        """Get the current bar's value."""
        if len(self._data) == 0:
            return np.nan
        return float(self._data[self._index])
    
    @property
    def previous(self) -> float:
        """Get the previous bar's value."""
        return self[1]
    
    def __getitem__(self, n: int) -> float:
        """
        Get the value N bars ago.
        
        Args:
            n: Number of bars ago (0 = current, 1 = previous, etc.)
            
        Returns:
            The value N bars ago, or NaN if out of range
        """
        idx = self._index + n
        if idx < 0 or idx >= len(self._data):
            return np.nan
        return float(self._data[idx])
    
    def __len__(self) -> int:
        """Return the total length of the series."""
        return len(self._data)
    
    def highest(self, n: int) -> float:
        """
        Get the highest value in the last N bars.
        
        Args:
            n: Number of bars to look back (including current)
            
        Returns:
            The highest value, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if start >= end:
            return np.nan
        return float(np.nanmax(self._data[start:end]))
    
    def lowest(self, n: int) -> float:
        """
        Get the lowest value in the last N bars.
        
        Args:
            n: Number of bars to look back (including current)
            
        Returns:
            The lowest value, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if start >= end:
            return np.nan
        return float(np.nanmin(self._data[start:end]))
    
    def average(self, n: int) -> float:
        """
        Get the average of the last N bars.
        
        Args:
            n: Number of bars to average (including current)
            
        Returns:
            The average value, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if start >= end:
            return np.nan
        return float(np.nanmean(self._data[start:end]))
    
    def sum(self, n: int) -> float:
        """
        Get the sum of the last N bars.
        
        Args:
            n: Number of bars to sum (including current)
            
        Returns:
            The sum, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if start >= end:
            return np.nan
        return float(np.nansum(self._data[start:end]))
    
    def std(self, n: int) -> float:
        """
        Get the standard deviation of the last N bars.
        
        Args:
            n: Number of bars (including current)
            
        Returns:
            The standard deviation, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if start >= end:
            return np.nan
        return float(np.nanstd(self._data[start:end]))
    
    def slope(self, n: int) -> float:
        """
        Get the slope (rate of change) over the last N bars.
        
        Uses linear regression to calculate the slope.
        
        Args:
            n: Number of bars (including current)
            
        Returns:
            The slope, or NaN if insufficient data
        """
        start = self._index
        end = min(start + n, len(self._data))
        if end - start < 2:
            return np.nan
        
        y = self._data[start:end]
        x = np.arange(len(y))
        
        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.nanmean(y)
        
        numerator = np.nansum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def change(self, n: int = 1) -> float:
        """
        Get the change from N bars ago to current.
        
        Args:
            n: Number of bars ago to compare with
            
        Returns:
            current - value[n], or NaN if insufficient data
        """
        return self.current - self[n]
    
    def pct_change(self, n: int = 1) -> float:
        """
        Get the percentage change from N bars ago to current.
        
        Args:
            n: Number of bars ago to compare with
            
        Returns:
            Percentage change, or NaN if insufficient data
        """
        prev = self[n]
        if prev == 0 or np.isnan(prev):
            return np.nan
        return (self.current - prev) / prev
    
    def crossed_above(self, other: Union["Series", float], lookback: int = 1) -> bool:
        """
        Check if this series crossed above another series/value.
        
        Args:
            other: Series or constant value to compare against
            lookback: How many bars ago to check the cross
            
        Returns:
            True if crossed above within the lookback period
        """
        if isinstance(other, Series):
            other_current = other.current
            other_prev = other[lookback]
        else:
            other_current = other_prev = float(other)
            
        return self[lookback] <= other_prev and self.current > other_current
    
    def crossed_below(self, other: Union["Series", float], lookback: int = 1) -> bool:
        """
        Check if this series crossed below another series/value.
        
        Args:
            other: Series or constant value to compare against
            lookback: How many bars ago to check the cross
            
        Returns:
            True if crossed below within the lookback period
        """
        if isinstance(other, Series):
            other_current = other.current
            other_prev = other[lookback]
        else:
            other_current = other_prev = float(other)
            
        return self[lookback] >= other_prev and self.current < other_current
    
    def set_index(self, index: int) -> None:
        """
        Set the current index position in the series.
        
        This is used by the backtesting engine to advance through time.
        
        Args:
            index: The new index position (0 = most recent data)
        """
        self._index = index
        
    def to_array(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get the underlying data as a numpy array.
        
        Args:
            n: Optional number of bars to return (from current)
            
        Returns:
            Numpy array of values
        """
        if n is None:
            return self._data[self._index:].copy()
        end = min(self._index + n, len(self._data))
        return self._data[self._index:end].copy()
    
    def __repr__(self) -> str:
        name_str = f"'{self._name}'" if self._name else ""
        return f"Series({name_str}, current={self.current:.4f}, len={len(self)})"
    
    def __float__(self) -> float:
        """Allow using Series directly as a float (returns current value)."""
        return self.current
    
    # Arithmetic operations
    def __add__(self, other: Union["Series", float]) -> "Series":
        if isinstance(other, Series):
            return Series(self._data + other._data)
        return Series(self._data + other)
    
    def __sub__(self, other: Union["Series", float]) -> "Series":
        if isinstance(other, Series):
            return Series(self._data - other._data)
        return Series(self._data - other)
    
    def __mul__(self, other: Union["Series", float]) -> "Series":
        if isinstance(other, Series):
            return Series(self._data * other._data)
        return Series(self._data * other)
    
    def __truediv__(self, other: Union["Series", float]) -> "Series":
        if isinstance(other, Series):
            return Series(self._data / other._data)
        return Series(self._data / other)
    
    def __neg__(self) -> "Series":
        return Series(-self._data)
    
    def __abs__(self) -> "Series":
        return Series(np.abs(self._data))
    
    # Comparison operations (return current value comparison)
    def __gt__(self, other: Union["Series", float]) -> bool:
        if isinstance(other, Series):
            return self.current > other.current
        return self.current > other
    
    def __lt__(self, other: Union["Series", float]) -> bool:
        if isinstance(other, Series):
            return self.current < other.current
        return self.current < other
    
    def __ge__(self, other: Union["Series", float]) -> bool:
        if isinstance(other, Series):
            return self.current >= other.current
        return self.current >= other
    
    def __le__(self, other: Union["Series", float]) -> bool:
        if isinstance(other, Series):
            return self.current <= other.current
        return self.current <= other

