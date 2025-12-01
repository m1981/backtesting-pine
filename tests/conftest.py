"""Pytest fixtures for backtesting tests."""

import sys
sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pine import (
    StrategyBase, strategy, Backtest,
    MACD, ATR, crossover, crossunder,
    Mode, State, ExitReason, Timeframe
)
from pine.data import TimeframeData, MultiTimeframeData, Bar
from pine.position import PositionManager


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV price data."""
    np.random.seed(42)

    # Generate 500 bars of data
    n_bars = 500
    dates = pd.date_range(start='2022-01-01', periods=n_bars, freq='1H')

    # Generate realistic price movement
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n_bars)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.001, n_bars)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def trending_price_data():
    """Generate trending price data (strong uptrend)."""
    np.random.seed(42)

    n_bars = 300
    dates = pd.date_range(start='2022-01-01', periods=n_bars, freq='1H')

    # Generate uptrend with noise
    base_price = 100.0
    trend = np.linspace(0, 0.3, n_bars)  # 30% uptrend
    noise = np.random.normal(0, 0.01, n_bars)
    prices = base_price * np.exp(trend + np.cumsum(noise))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.015, n_bars)),
        'low': prices * (1 + np.random.uniform(-0.015, -0.001, n_bars)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def choppy_price_data():
    """Generate choppy/sideways price data."""
    np.random.seed(42)

    n_bars = 300
    dates = pd.date_range(start='2022-01-01', periods=n_bars, freq='1H')

    # Generate sideways movement with high noise
    base_price = 100.0
    noise = np.random.normal(0, 0.015, n_bars)
    prices = base_price * (1 + noise)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n_bars)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.001, n_bars)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def position_manager():
    """Create a position manager for testing."""
    return PositionManager(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )


@pytest.fixture
def multi_timeframe_data(sample_price_data):
    """Create multi-timeframe data from sample data."""
    # Create 1H data
    tf_1h_data = TimeframeData(
        timeframe=Timeframe.HOUR_1,
        data=sample_price_data
    )

    # Resample to 4H
    resampled_4h = sample_price_data.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    tf_4h_data = TimeframeData(
        timeframe=Timeframe.HOUR_4,
        data=resampled_4h
    )

    # Resample to Daily
    resampled_daily = sample_price_data.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    tf_daily_data = TimeframeData(
        timeframe=Timeframe.DAILY,
        data=resampled_daily
    )

    # Create MultiTimeframeData
    mtf_data = MultiTimeframeData({
        'tf_1h': tf_1h_data,
        'tf_4h': tf_4h_data,
        'daily': tf_daily_data
    })

    return mtf_data


@pytest.fixture
def simple_strategy_class():
    """Create a simple test strategy class."""

    @strategy(
        name="Simple Test Strategy",
        mode=Mode.LONG_ONLY,
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )
    class SimpleStrategy(StrategyBase):
        """Simple strategy for testing."""

        # Parameters
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9

        def setup_indicators(self):
            return {
                'macd': MACD(
                    self.data.daily.close,
                    fast=self.macd_fast,
                    slow=self.macd_slow,
                    signal=self.macd_signal
                ),
                'atr': ATR(
                    self.data.daily.high,
                    self.data.daily.low,
                    self.data.daily.close,
                    period=14
                )
            }

        def on_bar(self, bar):
            if self.state == State.IDLE:
                macd = self.indicators['macd']

                if crossover(macd.macd_line, macd.signal_line):
                    atr = self.indicators['atr'].values.current
                    stop = bar.close - (2 * atr)
                    target = bar.close + (4 * atr)

                    self.buy(stop_loss=stop, take_profit=target)
                    self.goto(State.IN_POSITION)

            elif self.state == State.IN_POSITION:
                macd = self.indicators['macd']

                if crossunder(macd.macd_line, macd.signal_line):
                    self.close_position(reason=ExitReason.TRAILING_SIGNAL)
                    self.goto(State.IDLE)

    return SimpleStrategy
