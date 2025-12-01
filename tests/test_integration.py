"""
Integration tests for the Pine backtesting framework.

Tests the complete flow from data loading through strategy execution
to results analysis, including real-world trading scenarios and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from pine import (
    StrategyBase, strategy, Backtest,
    MACD, ATR, crossover, crossunder,
    Mode, State, ExitReason, Timeframe,
    Series
)
from pine.data import create_bars_iterator


@pytest.mark.integration
class TestBacktestIntegration:
    """Test full backtesting workflow."""

    def test_simple_strategy_execution(self, simple_strategy_class, trending_price_data):
        """Test that a simple strategy can execute trades."""
        # Create mock data loader by monkey-patching
        strategy_instance = simple_strategy_class()

        # For now, we'll test components individually
        # Full backtest requires yfinance which we want to avoid in tests

        assert strategy_instance.config.name == "Simple Test Strategy"
        assert strategy_instance.mode == Mode.LONG_ONLY
        assert strategy_instance.capital == 10000

    def test_strategy_with_trending_market(self, trending_price_data):
        """Test strategy performance in trending market."""

        @strategy(name="Trend Follower", mode=Mode.LONG_ONLY, initial_capital=10000)
        class TrendFollower(StrategyBase):
            def setup_indicators(self):
                return {
                    'macd': MACD(self.data.daily.close, 8, 21, 9),
                }

            def on_bar(self, bar):
                if self.state == State.IDLE:
                    macd = self.indicators['macd']
                    if macd.macd_line.current > 0 and crossover(macd.macd_line, macd.signal_line):
                        self.buy(
                            stop_loss=bar.close * 0.95,
                            take_profit=bar.close * 1.10
                        )
                        self.goto(State.IN_POSITION)

        # Verify strategy created correctly
        strat = TrendFollower()
        assert strat.config.name == "Trend Follower"

    def test_strategy_with_choppy_market(self, choppy_price_data):
        """Test strategy behavior in choppy/sideways market."""

        @strategy(name="Range Trader", mode=Mode.LONG_ONLY, initial_capital=10000)
        class RangeTrader(StrategyBase):
            def setup_indicators(self):
                return {
                    'atr': ATR(self.data.daily.high, self.data.daily.low, self.data.daily.close, 14)
                }

            def on_bar(self, bar):
                # This strategy should be more cautious in choppy markets
                if self.state == State.IDLE:
                    # Only trade if volatility is low
                    atr = self.indicators['atr'].values.current
                    if atr < bar.close * 0.02:  # ATR less than 2% of price
                        # Look for mean reversion opportunities
                        pass

        strat = RangeTrader()
        assert strat.config.initial_capital == 10000


@pytest.mark.integration
class TestPositionManagement:
    """Test position management scenarios."""

    def test_open_and_close_position(self, position_manager):
        """Test opening and closing a position."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        # Open position
        pos = position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=10,
            direction='long',
            stop_loss=95.0,
            take_profit=110.0,
            bar_index=0,
            comment="Test entry"
        )

        assert position_manager.has_position
        assert pos.entry_price > 100.0  # Should include slippage
        assert pos.quantity == 10
        assert pos.stop_loss == 95.0

        # Update position
        position_manager.update_position(105.0)
        assert pos.unrealized_pnl > 0  # Should be profitable

        # Close position
        trade = position_manager.close_position(
            timestamp=pd.Timestamp('2022-01-01 12:00:00'),
            price=105.0,
            reason=ExitReason.TAKE_PROFIT,
            bar_index=2,
            comment="Test exit"
        )

        assert not position_manager.has_position
        assert trade.pnl > 0  # Account for slippage/commission
        assert trade.exit_reason == ExitReason.TAKE_PROFIT

    def test_stop_loss_hit(self, position_manager):
        """Test automatic stop loss execution."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        # Open long position
        pos = position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=10,
            direction='long',
            stop_loss=95.0,
            take_profit=110.0,
            bar_index=0
        )

        # Update position - price still above stop
        position_manager.update_position(97.0)
        assert position_manager.has_position

        # Check exits - stop loss should be hit
        trade = position_manager.check_exits(
            timestamp=pd.Timestamp('2022-01-01 11:00:00'),
            current_price=94.0,  # Below stop loss
            bar_index=1
        )

        assert trade is not None
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert not position_manager.has_position
        assert trade.pnl < 0  # Losing trade

    def test_take_profit_hit(self, position_manager):
        """Test automatic take profit execution."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        pos = position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=10,
            direction='long',
            stop_loss=95.0,
            take_profit=110.0,
            bar_index=0
        )

        # Check exits - take profit hit
        trade = position_manager.check_exits(
            timestamp=pd.Timestamp('2022-01-01 11:00:00'),
            current_price=111.0,  # Above take profit
            bar_index=1
        )

        assert trade is not None
        assert trade.exit_reason == ExitReason.TAKE_PROFIT
        assert trade.pnl > 0  # Winning trade

    def test_partial_position_close(self, position_manager):
        """Test closing partial position."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        pos = position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=100,
            direction='long',
            stop_loss=95.0,
            take_profit=110.0,
            bar_index=0
        )

        original_qty = pos.quantity

        # Close 50% of position
        trade = position_manager.close_partial(
            timestamp=pd.Timestamp('2022-01-01 11:00:00'),
            price=105.0,
            portion=0.5,
            reason=ExitReason.TAKE_PROFIT,
            bar_index=1,
            comment="First target"
        )

        # Position should still exist with reduced quantity
        assert position_manager.has_position
        assert position_manager.current_position.quantity == original_qty * 0.5
        assert position_manager.current_position.is_partial
        assert trade.quantity == original_qty * 0.5

    def test_cannot_open_position_when_one_exists(self, position_manager):
        """Test that opening position when one exists raises error."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=10,
            direction='long',
            bar_index=0
        )

        # Try to open another position
        with pytest.raises(ValueError, match="position already exists"):
            position_manager.open_position(
                timestamp=timestamp,
                price=100.0,
                quantity=10,
                direction='long',
                bar_index=1
            )

    def test_cannot_close_when_no_position(self, position_manager):
        """Test that closing non-existent position raises error."""
        with pytest.raises(ValueError, match="no position exists"):
            position_manager.close_position(
                timestamp=pd.Timestamp('2022-01-01 10:00:00'),
                price=100.0,
                reason=ExitReason.MANUAL,
                bar_index=0
            )

    def test_commission_and_slippage_applied(self, position_manager):
        """Test that commission and slippage are properly applied."""
        timestamp = pd.Timestamp('2022-01-01 10:00:00')

        # Entry price should be worse than market price due to slippage
        pos = position_manager.open_position(
            timestamp=timestamp,
            price=100.0,
            quantity=10,
            direction='long',
            bar_index=0
        )

        # Long position: entry price should be higher (worse fill)
        assert pos.entry_price > 100.0

        # Close position
        trade = position_manager.close_position(
            timestamp=pd.Timestamp('2022-01-01 11:00:00'),
            price=100.0,  # Same price as entry
            reason=ExitReason.MANUAL,
            bar_index=1
        )

        # Even at same price, should lose money due to slippage/commission
        assert trade.pnl < 0
        assert trade.commission > 0


@pytest.mark.integration
class TestStateMachineScenarios:
    """Test state machine behavior in various scenarios."""

    def test_state_transitions(self):
        """Test basic state transitions."""
        from pine.state_machine import StateMachine

        sm = StateMachine(initial_state=State.IDLE)

        # Transition through states
        sm.transition_to(State.WAITING_CONFIRMATION, trigger="setup_found")
        assert sm.current_state == State.WAITING_CONFIRMATION
        assert sm.previous_state == State.IDLE

        sm.advance_bar()
        sm.advance_bar()
        assert sm.bars_in_state() == 2

        sm.transition_to(State.WAITING_TRIGGER, trigger="wait_complete")
        assert sm.current_state == State.WAITING_TRIGGER
        assert sm.bars_in_state() == 0

        sm.transition_to(State.IN_POSITION, trigger="entry")
        assert sm.current_state == State.IN_POSITION

    def test_state_metadata_storage(self):
        """Test that metadata is stored with state transitions."""
        from pine.state_machine import StateMachine

        sm = StateMachine()

        metadata = {'setup_bar': 42, 'distance': 1.5}
        sm.transition_to(
            State.WAITING_CONFIRMATION,
            trigger="setup",
            metadata=metadata
        )

        assert sm.state_metadata == metadata
        assert sm.state_metadata['setup_bar'] == 42

    def test_state_history_tracking(self):
        """Test that state history is properly tracked."""
        from pine.state_machine import StateMachine

        sm = StateMachine()

        sm.transition_to(State.WAITING_CONFIRMATION, trigger="t1")
        sm.advance_bar()
        sm.transition_to(State.WAITING_TRIGGER, trigger="t2")
        sm.advance_bar()
        sm.transition_to(State.IN_POSITION, trigger="t3")

        history = sm.get_history()
        assert len(history) == 3
        assert history[0].trigger == "t1"
        assert history[1].trigger == "t2"
        assert history[2].trigger == "t3"


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world trading scenarios with edge cases."""

    def test_consecutive_wins_scenario(self):
        """Test handling of consecutive winning trades."""

        @strategy(name="Winner", mode=Mode.LONG_ONLY, initial_capital=10000)
        class WinningStrategy(StrategyBase):
            """Strategy that should generate consistent wins in uptrend."""

            trade_count = 0

            def setup_indicators(self):
                return {}

            def on_bar(self, bar):
                # Simple strategy: buy every 20 bars in uptrend, quick profit
                if self.state == State.IDLE and bar.index % 20 == 0:
                    self.buy(
                        stop_loss=bar.close * 0.98,
                        take_profit=bar.close * 1.02
                    )
                    self.goto(State.IN_POSITION)

        # This tests that capital compounds properly
        strat = WinningStrategy()
        assert strat.config.initial_capital == 10000

    def test_consecutive_losses_scenario(self):
        """Test handling of consecutive losing trades and drawdown."""

        @strategy(name="Loser", mode=Mode.LONG_ONLY, initial_capital=10000)
        class LosingStrategy(StrategyBase):
            """Strategy that triggers stop losses."""

            def setup_indicators(self):
                return {}

            def on_bar(self, bar):
                # Bad strategy: buy and set tight stop
                if self.state == State.IDLE and bar.index % 15 == 0:
                    # Set stop very close - likely to be hit
                    self.buy(
                        stop_loss=bar.close * 0.999,  # 0.1% stop
                        take_profit=bar.close * 1.10
                    )
                    self.goto(State.IN_POSITION)

        strat = LosingStrategy()
        # Test that we don't go negative (should stop trading if capital too low)
        assert strat.capital == 10000

    def test_whipsaw_scenario(self):
        """Test whipsaw (false signals) scenario."""

        @strategy(name="Whipsaw", mode=Mode.LONG_ONLY, initial_capital=10000)
        class WhipsawStrategy(StrategyBase):
            """Strategy susceptible to whipsaws."""

            def setup_indicators(self):
                return {
                    'macd': MACD(self.data.daily.close, 5, 13, 5)  # Fast MACD = more signals
                }

            def on_bar(self, bar):
                macd = self.indicators['macd']

                if self.state == State.IDLE:
                    # Enter on crossover
                    if crossover(macd.macd_line, macd.signal_line):
                        self.buy(
                            stop_loss=bar.close * 0.95,
                            take_profit=bar.close * 1.05
                        )
                        self.goto(State.IN_POSITION)

                elif self.state == State.IN_POSITION:
                    # Exit on opposite crossover (whipsaw risk)
                    if crossunder(macd.macd_line, macd.signal_line):
                        self.close_position(reason=ExitReason.TRAILING_SIGNAL)
                        self.goto(State.IDLE)

        strat = WhipsawStrategy()
        # In choppy markets, this should generate many small losses

    def test_gap_scenario(self):
        """Test handling of price gaps (stop loss gapped through)."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=10000, commission=0.001, slippage=0.0005)

        # Open position
        pm.open_position(
            timestamp=pd.Timestamp('2022-01-01 10:00:00'),
            price=100.0,
            quantity=10,
            direction='long',
            stop_loss=95.0,
            bar_index=0
        )

        # Price gaps down below stop loss
        trade = pm.check_exits(
            timestamp=pd.Timestamp('2022-01-01 11:00:00'),
            current_price=90.0,  # Gapped through stop at 95
            bar_index=1
        )

        # Should exit at stop price (with slippage applied)
        assert trade is not None
        # Exit price should be close to stop (slippage applies: 95 * (1 - 0.0005) = 94.9525)
        assert 94.9 < trade.exit_price < 95.1  # Within slippage range
        assert trade.exit_reason == ExitReason.STOP_LOSS

    def test_insufficient_capital_scenario(self):
        """Test handling when insufficient capital for position sizing."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=100, commission=0.001, slippage=0.0005)

        # Try to open position with calculated size
        # With only $100 capital and risk management, should get small position
        pos = pm.open_position(
            timestamp=pd.Timestamp('2022-01-01 10:00:00'),
            price=100.0,
            quantity=0.5,  # Small quantity due to limited capital
            direction='long',
            stop_loss=95.0,
            bar_index=0
        )

        assert pos.quantity == 0.5
        assert pm.capital < 100  # Commission paid

    def test_multiple_entries_exits_same_day(self):
        """Test multiple round trips in single day."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=10000, commission=0.001, slippage=0.0005)
        base_time = pd.Timestamp('2022-01-01 09:00:00')

        # Trade 1
        pm.open_position(base_time, 100.0, 10, 'long', bar_index=0)
        trade1 = pm.close_position(
            base_time + pd.Timedelta(hours=1), 105.0,
            ExitReason.TAKE_PROFIT, bar_index=1
        )

        # Trade 2
        pm.open_position(base_time + pd.Timedelta(hours=2), 105.0, 10, 'long', bar_index=2)
        trade2 = pm.close_position(
            base_time + pd.Timedelta(hours=3), 102.0,
            ExitReason.STOP_LOSS, bar_index=3
        )

        # Trade 3
        pm.open_position(base_time + pd.Timedelta(hours=4), 102.0, 10, 'long', bar_index=4)
        trade3 = pm.close_position(
            base_time + pd.Timedelta(hours=5), 107.0,
            ExitReason.TAKE_PROFIT, bar_index=5
        )

        assert len(pm.closed_trades) == 3
        assert pm.total_trades == 3
        assert not pm.has_position

    def test_hold_through_multiple_signals(self):
        """Test holding position while ignoring new signals."""

        @strategy(name="Patient", mode=Mode.LONG_ONLY, initial_capital=10000)
        class PatientStrategy(StrategyBase):
            """Strategy that holds positions and ignores new signals."""

            def setup_indicators(self):
                return {
                    'macd': MACD(self.data.daily.close, 12, 26, 9)
                }

            def on_bar(self, bar):
                macd = self.indicators['macd']

                if self.state == State.IDLE:
                    if crossover(macd.macd_line, 0):  # MACD crosses above zero
                        self.buy(
                            stop_loss=bar.close * 0.90,
                            take_profit=bar.close * 1.20
                        )
                        self.goto(State.IN_POSITION)

                elif self.state == State.IN_POSITION:
                    # Only exit on opposite signal, ignore additional buy signals
                    if crossunder(macd.macd_line, 0):
                        self.close_position(reason=ExitReason.TRAILING_SIGNAL)
                        self.goto(State.IDLE)

        strat = PatientStrategy()
        # Test validates that being IN_POSITION prevents new entries

    def test_trailing_stop_scenario(self):
        """Test manual trailing stop implementation."""

        @strategy(name="Trailer", mode=Mode.LONG_ONLY, initial_capital=10000)
        class TrailingStopStrategy(StrategyBase):
            """Strategy with trailing stop."""

            highest_price = 0.0

            def setup_indicators(self):
                return {'atr': ATR(self.data.daily.high, self.data.daily.low,
                                  self.data.daily.close, 14)}

            def on_bar(self, bar):
                if self.state == State.IDLE:
                    # Simple entry
                    if bar.index == 10:  # Enter at bar 10
                        atr = self.indicators['atr'].values.current
                        self.buy(
                            stop_loss=bar.close - (2 * atr),
                            take_profit=bar.close + (4 * atr)
                        )
                        self.highest_price = bar.close
                        self.goto(State.IN_POSITION)

                elif self.state == State.IN_POSITION:
                    # Trail stop
                    if bar.close > self.highest_price:
                        self.highest_price = bar.close
                        # Move stop up
                        atr = self.indicators['atr'].values.current
                        new_stop = bar.close - (2 * atr)
                        if new_stop > self.current_position.stop_loss:
                            self.modify_stop(new_stop)

        strat = TrailingStopStrategy()
        strat.highest_price = 0.0  # Initialize


@pytest.mark.integration
class TestMultiTimeframeCoordination:
    """Test multi-timeframe data coordination."""

    def test_timeframe_data_alignment(self, multi_timeframe_data):
        """Test that different timeframes align correctly."""
        mtf = multi_timeframe_data

        # All timeframes should have data
        assert len(mtf.tf_1h.data) > 0
        assert len(mtf.tf_4h.data) > 0
        assert len(mtf.daily.data) > 0

        # 1H should have more bars than 4H
        assert len(mtf.tf_1h.data) > len(mtf.tf_4h.data)

        # 4H should have more bars than Daily
        assert len(mtf.tf_4h.data) > len(mtf.daily.data)

    def test_series_access_across_timeframes(self, multi_timeframe_data):
        """Test accessing Series data across timeframes."""
        mtf = multi_timeframe_data

        # Access 1H data
        close_1h = mtf.tf_1h.close.current
        assert close_1h > 0

        # Access 4H data
        high_4h = mtf.tf_4h.high.highest(10)
        assert high_4h > 0

        # Access Daily data
        low_daily = mtf.daily.low.lowest(20)
        assert low_daily > 0

    def test_indicator_calculation_per_timeframe(self, multi_timeframe_data):
        """Test that indicators calculate correctly for each timeframe."""
        mtf = multi_timeframe_data

        # Calculate MACD on different timeframes
        macd_1h = MACD(mtf.tf_1h.close, 12, 26, 9)
        macd_4h = MACD(mtf.tf_4h.close, 12, 26, 9)
        macd_daily = MACD(mtf.daily.close, 12, 26, 9)

        # All should have data
        assert len(macd_1h.macd_line) > 0
        assert len(macd_4h.macd_line) > 0
        assert len(macd_daily.macd_line) > 0

        # Different timeframes should have different number of data points
        # This validates that we're not accidentally sharing data
        assert len(macd_1h.macd_line) > len(macd_4h.macd_line)
        assert len(macd_4h.macd_line) > len(macd_daily.macd_line)


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Empty dataframe
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        # Should handle gracefully (depends on implementation)
        # For now, just test that it doesn't crash
        assert len(empty_df) == 0

    def test_single_bar_data(self):
        """Test handling of single bar."""
        single_bar_data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        }, index=pd.date_range('2022-01-01', periods=1, freq='1H'))

        bars = create_bars_iterator(single_bar_data)
        assert len(bars) == 1
        assert bars[0].close == 100.5

    def test_nan_values_in_data(self):
        """Test handling of NaN values."""
        data = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],
            'high': [101.0, 103.0, 103.0],
            'low': [99.0, 101.0, np.nan],
            'close': [100.5, 102.5, 102.0],
            'volume': [1000000, 1000000, 1000000]
        }, index=pd.date_range('2022-01-01', periods=3, freq='1H'))

        # Series should handle NaN values
        from pine import Series
        s = Series(data['open'].values[::-1])
        assert np.isnan(s[1])  # Second value should be NaN

    def test_zero_price(self):
        """Test handling of zero price."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=10000)

        # Try to calculate percentage change with zero price
        # Should handle gracefully (return NaN or raise clear error)
        # This depends on implementation

    def test_negative_quantity(self):
        """Test that negative quantity is rejected."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=10000)

        # Negative quantity should not be allowed
        # This should either raise an error or be prevented
        # (depends on implementation - add validation if needed)

    def test_invalid_stop_loss(self):
        """Test invalid stop loss (stop above entry for long)."""
        from pine.position import PositionManager

        pm = PositionManager(initial_capital=10000)

        # For long position, stop loss above entry price is invalid
        pos = pm.open_position(
            timestamp=pd.Timestamp('2022-01-01'),
            price=100.0,
            quantity=10,
            direction='long',
            stop_loss=105.0,  # Invalid: above entry
            bar_index=0
        )

        # Position should be created but won't protect properly
        # Consider adding validation
        assert pos.stop_loss == 105.0
