# tests/test_strategy_macd_money_map.py

import pytest
import numpy as np
import pandas as pd

from pine import Backtest, State, ExitReason
from examples.macd_money_map import MACDMoneyMap


# Helper function to generate synthetic data
def generate_synthetic_data(price_points, start_date='2023-01-01', freq='1H'):
    """Generates a DataFrame from a list of closing prices."""
    dates = pd.date_range(start=start_date, periods=len(price_points), freq=freq)
    data = pd.DataFrame({
        'open': np.array(price_points) * 0.99,
        'high': np.array(price_points) * 1.01,
        'low': np.array(price_points) * 0.98,
        'close': price_points,
        'volume': 1000
    }, index=dates)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    return data


@pytest.fixture
def ideal_trade_scenario_data():
    """
    Generates data designed to trigger one successful MACD Money Map trade.
    - Bars 0-50: Establishes a strong daily uptrend (Daily MACD > 0).
    - Bars 50-70: 4H crossover occurs far from zero.
    - Bars 73-75: 1H histogram flips after 'wait_bars' period.
    - Bars 75+: Price rises to hit the take profit.
    """
    # This price action is crafted to manipulate the indicators as needed
    prices = ([100 + i * 0.1 for i in range(50)] +  # Initial uptrend for Daily bias
              [105, 104, 106, 105, 107, 106] +  # Cause 4H crossover
              [106.1, 106.2, 106.0] +  # 'wait_bars' period + histo flip
              [107, 108, 109, 110, 115])  # Rise to take profit
    return generate_synthetic_data(prices)


@pytest.fixture
def bearish_bias_scenario_data():
    """Data where 1H/4H signals are perfect, but Daily MACD is bearish."""
    prices = ([150 - i * 0.1 for i in range(50)] +  # Initial downtrend for Daily bias
              [145, 144, 146, 145, 147, 146] +  # 4H Bullish Crossover
              [146.1, 146.2, 146.0])  # 1H Histo flip
    return generate_synthetic_data(prices)


@pytest.fixture
def chop_zone_scenario_data():
    """Data where 4H crossover happens too close to zero."""
    # Sideways market ensures MACD hovers near zero
    prices = ([100.1, 100.2, 100.1, 100.3, 100.2] * 10 +  # Establish near-zero MACD
              [100.0, 99.8, 100.5, 100.6, 100.7])  # Crossover event
    return generate_synthetic_data(prices)


@pytest.mark.integration
class TestMACDMoneyMapStrategy:

    def test_ideal_trade_scenario(self, ideal_trade_scenario_data):
        """
        Ensures the strategy executes one trade correctly on a perfect setup.
        """
        strategy_instance = MACDMoneyMap()
        # Override wait_bars for deterministic testing
        strategy_instance.wait_bars = 2

        backtest = Backtest(
            strategy=strategy_instance,
            data=ideal_trade_scenario_data,
            timeframes=['1h', '4h', '1d'],
            base_timeframe='1h'
        )
        result = backtest.run(verbose=False)

        assert len(result.trades) == 1, "Should have executed exactly one trade"
        trade = result.trades[0]

        # We expect entry after the setup and wait period
        assert trade.entry_bar_index > 55
        assert trade.pnl > 0, "Trade in an ideal scenario should be profitable"
        assert trade.exit_reason == ExitReason.TAKE_PROFIT

    def test_bearish_bias_filter_prevents_trade(self, bearish_bias_scenario_data):
        """
        Ensures the Daily MACD bearish bias correctly filters out long signals.
        """
        strategy_instance = MACDMoneyMap()
        backtest = Backtest(
            strategy=strategy_instance,
            data=bearish_bias_scenario_data,
            timeframes=['1h', '4h', '1d'],
            base_timeframe='1h'
        )
        result = backtest.run(verbose=False)

        assert len(result.trades) == 0, "No trades should be opened with a bearish daily bias"

    def test_chop_zone_filter_prevents_trade(self, chop_zone_scenario_data):
        """
        Ensures the 4H 'distance rule' filters out signals near the zero line.
        """
        strategy_instance = MACDMoneyMap()
        backtest = Backtest(
            strategy=strategy_instance,
            data=chop_zone_scenario_data,
            timeframes=['1h', '4h', '1d'],
            base_timeframe='1h'
        )
        result = backtest.run(verbose=False)

        assert len(result.trades) == 0, "No trades should be opened for crossovers in the chop zone"