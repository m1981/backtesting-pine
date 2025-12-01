"""
Multi-timeframe data loading and management.

This module handles fetching market data and resampling it to multiple
timeframes while maintaining proper alignment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from .core.types import Timeframe
from .core.series import Series


@dataclass
class Bar:
    """Represents a single price bar."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    index: int  # Bar index in the series

    def __repr__(self) -> str:
        return (f"Bar({self.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                f"O:{self.open:.2f}, H:{self.high:.2f}, "
                f"L:{self.low:.2f}, C:{self.close:.2f})")


@dataclass
class TimeframeData:
    """
    Container for OHLCV data at a specific timeframe.

    Provides access to price series and metadata.
    """
    timeframe: Timeframe
    data: pd.DataFrame  # OHLCV data with datetime index

    @property
    def open(self) -> Series:
        """Get open prices as Series."""
        return Series(self.data['open'].values[::-1], name="open")

    @property
    def high(self) -> Series:
        """Get high prices as Series."""
        return Series(self.data['high'].values[::-1], name="high")

    @property
    def low(self) -> Series:
        """Get low prices as Series."""
        return Series(self.data['low'].values[::-1], name="low")

    @property
    def close(self) -> Series:
        """Get close prices as Series."""
        return Series(self.data['close'].values[::-1], name="close")

    @property
    def volume(self) -> Series:
        """Get volume as Series."""
        return Series(self.data['volume'].values[::-1], name="volume")

    def __len__(self) -> int:
        """Return number of bars."""
        return len(self.data)

    def __repr__(self) -> str:
        return f"TimeframeData({self.timeframe.name}, {len(self)} bars)"


class DataLoader:
    """
    Loads and manages multi-timeframe market data.

    Handles fetching data from yfinance and resampling to multiple
    timeframes while maintaining proper alignment.
    """

    def __init__(self, symbol: str, base_timeframe: Timeframe = Timeframe.HOUR_1):
        """
        Initialize data loader.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            base_timeframe: Base timeframe for data fetching
        """
        self.symbol = symbol
        self.base_timeframe = base_timeframe
        self._data: Dict[Timeframe, pd.DataFrame] = {}

    def fetch(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance.

        Args:
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            period: Period string (e.g., '1y', '2y', 'max') instead of dates

        Returns:
            DataFrame with OHLCV data

        Note:
            yfinance limits:
            - Hourly data: last 730 days only
            - For older data, must use daily or weekly
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for data fetching. "
                "Install it with: pip install yfinance"
            )

        # Convert period to date range if not specified
        if period and not start_date:
            # Calculate start_date from period
            period_map = {
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                'max': 730  # Max for hourly data
            }
            days = period_map.get(period, 365)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        # Fetch data
        ticker = yf.Ticker(self.symbol)
        interval = self.base_timeframe.to_yfinance_interval()

        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True  # Adjust for splits and dividends
        )

        if df.empty:
            raise ValueError(f"No data fetched for {self.symbol}")

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]

        # Store base timeframe data
        self._data[self.base_timeframe] = df

        return df

    def resample(self, target_timeframe: Timeframe) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            target_timeframe: Target timeframe to resample to

        Returns:
            Resampled DataFrame

        Raises:
            ValueError: If base data hasn't been fetched yet
        """
        if self.base_timeframe not in self._data:
            raise ValueError("Must fetch base data before resampling")

        base_df = self._data[self.base_timeframe]

        # If target is same as base, return base
        if target_timeframe == self.base_timeframe:
            return base_df

        # Resample using pandas
        freq = target_timeframe.to_pandas_freq()

        resampled = base_df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Store resampled data
        self._data[target_timeframe] = resampled

        return resampled

    def get_timeframe_data(self, timeframe: Timeframe) -> TimeframeData:
        """
        Get data for a specific timeframe.

        Args:
            timeframe: Timeframe to get data for

        Returns:
            TimeframeData object

        Raises:
            ValueError: If data for timeframe hasn't been loaded
        """
        if timeframe not in self._data:
            # Try to resample if base data exists
            if self.base_timeframe in self._data:
                self.resample(timeframe)
            else:
                raise ValueError(
                    f"No data for timeframe {timeframe.name}. "
                    f"Call fetch() first."
                )

        return TimeframeData(timeframe=timeframe, data=self._data[timeframe])

    def load_multiple_timeframes(
        self,
        timeframes: List[Timeframe],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None
    ) -> Dict[Timeframe, TimeframeData]:
        """
        Load data for multiple timeframes at once.

        Args:
            timeframes: List of timeframes to load
            start_date: Start date for data
            end_date: End date for data
            period: Period string (e.g., '1y', '2y')

        Returns:
            Dictionary mapping timeframes to TimeframeData objects
        """
        # Fetch base data
        self.fetch(start_date=start_date, end_date=end_date, period=period)

        # Resample to all requested timeframes
        result = {}
        for tf in timeframes:
            if tf != self.base_timeframe:
                self.resample(tf)
            result[tf] = self.get_timeframe_data(tf)

        return result


class MultiTimeframeData:
    """
    Container for accessing multiple timeframes in a strategy.

    Provides a clean API for accessing data across different timeframes:

    Example:
        data = MultiTimeframeData(...)
        data.tf_1h.close.current    # Current 1H close
        data.tf_4h.high.highest(20)  # Highest high in last 20 4H bars
        data.daily.low.lowest(50)    # Lowest low in last 50 daily bars
    """

    def __init__(self, timeframe_data: Dict[str, TimeframeData]):
        """
        Initialize with timeframe data.

        Args:
            timeframe_data: Dict mapping names to TimeframeData objects
                           e.g., {'tf_1h': data1, 'tf_4h': data4, 'daily': dataD}
        """
        self._data = timeframe_data

        # Make timeframes accessible as attributes
        for name, data in timeframe_data.items():
            setattr(self, name, data)

    def align(self, base_timeframe_name: str):
        """
        Aligns all higher timeframe data to the base timeframe's index.
        This prevents lookahead bias by ensuring each bar only sees the
        most recent *completed* higher timeframe data.
        """
        base_df = self._data[base_timeframe_name].data

        for name, tf_data in self._data.items():
            if name == base_timeframe_name:
                continue

            # Reindex the higher timeframe data to the base index, forward-filling values
            aligned_df = tf_data.data.reindex(base_df.index, method='ffill').fillna(method='bfill')
            self._data[name].data = aligned_df

    def set_index(self, index: int):
        """
        Sets the current bar index for all Series in all timeframes.
        This is called by the backtesting engine on each bar.
        """
        for tf_data in self._data.values():
            # This assumes TimeframeData has Series-like properties (open, high, etc.)
            # which it does. We need to update the underlying Series objects.
            # The current implementation creates new Series on every property access.
            # This needs to be optimized. Let's cache them.

            if not hasattr(tf_data, '_series_cache'):
                tf_data._series_cache = {}

            for series_name in ['open', 'high', 'low', 'close', 'volume']:
                if series_name not in tf_data._series_cache:
                    tf_data._series_cache[series_name] = getattr(tf_data, series_name)

                tf_data._series_cache[series_name].set_index(index)

    def __getitem__(self, name: str) -> TimeframeData:
        """Get timeframe data by name."""
        return self._data[name]

    def __repr__(self) -> str:
        tfs = ', '.join(self._data.keys())
        return f"MultiTimeframeData({tfs})"


def align_timeframes(
    base_data: pd.DataFrame,
    higher_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Align higher timeframe data to base timeframe.

    For each bar in base timeframe, finds the corresponding bar
    in higher timeframe (the most recent completed bar).

    Args:
        base_data: Base timeframe DataFrame (e.g., 1H)
        higher_data: Higher timeframe DataFrame (e.g., 4H, Daily)

    Returns:
        Higher timeframe data aligned to base timeframe index

    Example:
        If base is 1H and higher is 4H:
        - 1H bars at 9:00, 10:00, 11:00, 12:00 all align to 4H bar ending at 12:00
        - 1H bars at 13:00, 14:00, 15:00, 16:00 all align to 4H bar ending at 16:00
    """
    # Use forward fill to propagate higher timeframe values
    # This ensures we only see "completed" higher TF bars
    aligned = higher_data.reindex(
        base_data.index,
        method='ffill'
    )

    return aligned


def create_bars_iterator(data: pd.DataFrame) -> List[Bar]:
    """
    Create a list of Bar objects from DataFrame.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        List of Bar objects in chronological order
    """
    bars = []
    for i, (timestamp, row) in enumerate(data.iterrows()):
        bar = Bar(
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            index=i
        )
        bars.append(bar)

    return bars
