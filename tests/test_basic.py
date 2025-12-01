"""Basic tests for the Pine backtesting framework."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pine.core.series import Series
from pine.core.types import Mode, State, Timeframe
from pine.indicators import MACD, ATR, ema, crossover
from pine.state_machine import StateMachine


def test_series_basic():
    """Test Series basic operations."""
    data = [10, 20, 30, 40, 50]
    s = Series(data)

    assert s.current == 10
    assert s.previous == 20
    assert s[2] == 30
    assert s.highest(3) == 30
    assert s.lowest(3) == 10
    assert s.average(3) == 20
    print("✓ Series basic operations work")


def test_series_crossover():
    """Test Series crossover detection."""
    data1 = [5, 10, 15]
    data2 = [12, 12, 12]

    s1 = Series(data1)
    s2 = Series(data2)

    # s1 crossed above s2 (was 10 < 12, now 5 < 12, looking backwards)
    # Actually with current=5, previous=10, s1 is going down
    # Let me fix the test

    # For crossover: previous <= threshold and current > threshold
    # data = [current, previous, ...]
    # So data1 = [15, 10, 5] means current=15, was below, now above
    data1_corrected = [15, 10, 5]
    s1 = Series(data1_corrected)

    assert s1.crossed_above(12, lookback=1)
    print("✓ Series crossover detection works")


def test_ema_calculation():
    """Test EMA indicator."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    result = ema(data, period=3)

    assert len(result) == len(data)
    assert result[-1] > result[0]  # EMA should be trending up
    print("✓ EMA calculation works")


def test_macd_indicator():
    """Test MACD indicator."""
    # Create sample price data
    prices = np.linspace(100, 110, 50)  # Trending up

    macd_calc = MACD(prices, fast=12, slow=26, signal=9)
    result = macd_calc.calculate()

    assert result.macd_line is not None
    assert result.signal_line is not None
    assert result.histogram is not None
    assert len(result.macd_line) == len(prices)
    print("✓ MACD indicator works")


def test_atr_indicator():
    """Test ATR indicator."""
    # Create sample OHLC data
    high = np.array([102, 104, 103, 105, 106])
    low = np.array([98, 99, 100, 101, 102])
    close = np.array([100, 102, 101, 104, 105])

    atr_calc = ATR(high, low, close, period=3)
    result = atr_calc.calculate()

    assert len(result) == len(high)
    assert result.current > 0  # ATR should be positive
    print("✓ ATR indicator works")


def test_state_machine():
    """Test state machine."""
    sm = StateMachine(initial_state=State.IDLE)

    assert sm.current_state == State.IDLE
    assert sm.bars_in_state() == 0

    sm.transition_to(State.WAITING_CONFIRMATION, trigger="test")
    assert sm.current_state == State.WAITING_CONFIRMATION
    assert sm.previous_state == State.IDLE

    sm.advance_bar()
    assert sm.bars_in_state() == 1

    print("✓ State machine works")


def test_timeframe_parsing():
    """Test timeframe string parsing."""
    tf = Timeframe.from_string('1h')
    assert tf == Timeframe.HOUR_1

    tf = Timeframe.from_string('4H')
    assert tf == Timeframe.HOUR_4

    tf = Timeframe.from_string('1d')
    assert tf == Timeframe.DAILY

    print("✓ Timeframe parsing works")


def test_mode_enum():
    """Test Mode enum."""
    assert Mode.LONG_ONLY != Mode.SHORT_ONLY
    assert Mode.LONG_SHORT != Mode.LONG_ONLY
    print("✓ Mode enum works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running basic tests...")
    print("=" * 60)
    print()

    try:
        test_series_basic()
        test_series_crossover()
        test_ema_calculation()
        test_macd_indicator()
        test_atr_indicator()
        test_state_machine()
        test_timeframe_parsing()
        test_mode_enum()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
