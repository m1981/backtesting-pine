"""
MACD Money Map Strategy - Multi-Timeframe MACD Strategy

This strategy implements the "Three Systems" approach:
- System 1: Trend detection on 4H (crossovers far from zero)
- System 2: Reversal detection (not implemented yet)
- System 3: Multi-timeframe confirmation (Daily bias + 1H trigger)

Usage:
    python examples/macd_money_map.py
"""

import sys
sys.path.insert(0, 'src')

from pine import (
    StrategyBase, strategy, Backtest,
    MACD, ATR, crossover, crossunder,
    Mode, Bias, State, SetupSignal, ExitReason
)


@strategy(
    name="MACD Money Map",
    description="Multi-timeframe MACD strategy with trend and confirmation systems",
    mode=Mode.LONG_ONLY,
    initial_capital=10000,
    commission=0.002,
    slippage=0.001,
    risk_per_trade=0.02
)
class MACDMoneyMap(StrategyBase):
    """
    MACD Money Map Strategy Implementation.

    Implements the three-system approach from the specification:
    - Daily MACD determines bias
    - 4H MACD detects trend setups
    - 1H MACD histogram confirms entries
    """

    # Strategy parameters
    fast_ema = 12
    slow_ema = 26
    signal_period = 9
    atr_period = 14
    atr_threshold = 0.5  # Multiplier for distance from zero
    wait_bars = 2        # Bars to wait after crossover
    risk_reward = 2.0    # Risk:reward ratio

    def setup_indicators(self):
        """Setup indicators for all timeframes."""
        return {
            # Daily MACD (for bias)
            'macd_daily': MACD(
                self.data.daily.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),

            # 4H MACD (for setup detection)
            'macd_4h': MACD(
                self.data.tf_4h.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),

            # 1H MACD (for entry trigger)
            'macd_1h': MACD(
                self.data.tf_1h.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),

            # ATR on 1H (for thresholds and stops)
            'atr': ATR(
                self.data.tf_1h.high,
                self.data.tf_1h.low,
                self.data.tf_1h.close,
                period=self.atr_period
            )
        }

    # ═══════════════════════════════════════════════════════════════
    # SYSTEM 3A: Daily Bias Filter
    # ═══════════════════════════════════════════════════════════════

    def is_bullish_bias(self) -> bool:
        """
        Check if daily MACD shows bullish bias.

        The Zero Line is absolute law:
        - Above zero = ONLY longs
        - Below zero = NO trading (in long-only mode)
        """
        macd_daily = self.indicators['macd_daily']
        return macd_daily.macd_line.current > 0

    # ═══════════════════════════════════════════════════════════════
    # SYSTEM 1: Trend Setup Detection (4H)
    # ═══════════════════════════════════════════════════════════════

    def has_valid_setup(self) -> bool:
        """
        Check for valid trend setup on 4H.

        Valid setup requires:
        1. Bullish crossover (MACD crosses above Signal)
        2. Crossover happens FAR from zero (> atr_threshold × ATR)

        Returns:
            True if valid setup detected
        """
        macd_4h = self.indicators['macd_4h']
        atr = self.indicators['atr'].values.current

        # Check for bullish crossover
        has_crossover = crossover(macd_4h.macd_line, macd_4h.signal_line)

        if not has_crossover:
            return False

        # The Distance Rule: must be far from zero to avoid chop zone
        distance_from_zero = abs(macd_4h.macd_line.current)
        threshold = self.atr_threshold * atr

        if distance_from_zero < threshold:
            # Reject: too close to zero (chop zone)
            return False

        # Store distance for later analysis
        self.state_machine._state_metadata['setup_distance'] = distance_from_zero

        return True

    def setup_invalidated(self) -> bool:
        """
        Check if current setup has been invalidated.

        Setup is invalidated if 4H MACD crosses back down.

        Returns:
            True if setup is no longer valid
        """
        macd_4h = self.indicators['macd_4h']

        # Check for opposite crossover
        return crossunder(macd_4h.macd_line, macd_4h.signal_line)

    # ═══════════════════════════════════════════════════════════════
    # SYSTEM 3B: Entry Trigger (1H)
    # ═══════════════════════════════════════════════════════════════

    def has_entry_trigger(self) -> bool:
        """
        Check for entry trigger on 1H.

        The Flip: Histogram flips from red to green.
        - Previous bar: histogram < 0
        - Current bar: histogram > 0

        Returns:
            True if histogram flip detected
        """
        histogram = self.indicators['macd_1h'].histogram

        prev_red = histogram.previous < 0
        curr_green = histogram.current > 0

        return prev_red and curr_green

    # ═══════════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ═══════════════════════════════════════════════════════════════

    def execute_entry(self) -> None:
        """Execute long entry with calculated stops and targets."""
        entry_price = self.current_bar.close
        atr = self.indicators['atr'].values.current

        # Calculate stop loss (tighter of swing low or ATR-based)
        swing_low = self.data.tf_1h.low.lowest(20)
        atr_stop = entry_price - (2 * atr)
        stop_loss = max(swing_low, atr_stop)  # Use tighter stop

        # Calculate risk
        risk = entry_price - stop_loss

        # Take profit: 2R target
        take_profit = entry_price + (self.risk_reward * risk)

        # Execute entry
        self.buy(
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="MACD Money Map Entry",
            metadata={
                'daily_macd': self.indicators['macd_daily'].macd_line.current,
                'setup_distance': self.state_machine.state_metadata.get('setup_distance', 0),
                'atr_at_entry': atr
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # EXIT CONDITIONS
    # ═══════════════════════════════════════════════════════════════

    def should_exit(self) -> bool:
        """
        Check additional exit conditions beyond stop/target.

        Returns:
            True if position should be closed
        """
        # Exit 1: Trailing signal (4H bearish crossover)
        macd_4h = self.indicators['macd_4h']
        if crossunder(macd_4h.macd_line, macd_4h.signal_line):
            return True

        # Exit 2: Bias change (Daily MACD crosses below zero)
        macd_daily = self.indicators['macd_daily']
        if crossunder(macd_daily.macd_line, 0):
            return True

        return False

    # ═══════════════════════════════════════════════════════════════
    # MAIN STRATEGY LOGIC
    # ═══════════════════════════════════════════════════════════════

    def on_bar(self, bar):
        """Main strategy logic called on each 1H bar."""

        # ─────────────────────────────────────────────────────
        # STATE: IDLE - Looking for setups
        # ─────────────────────────────────────────────────────

        if self.state == State.IDLE:

            # System 3A: Check daily bias first
            if not self.is_bullish_bias():
                return  # No trading allowed today

            # System 1: Check for trend setup on 4H
            if self.has_valid_setup():
                self.transition_to(
                    State.WAITING_CONFIRMATION,
                    trigger="setup_detected",
                    metadata={'crossover_bar': bar.index}
                )

        # ─────────────────────────────────────────────────────
        # STATE: WAITING_CONFIRMATION - Patience filter
        # ─────────────────────────────────────────────────────

        elif self.state == State.WAITING_CONFIRMATION:

            bars_waited = self.bars_in_state()

            # Check if setup is still valid
            if self.setup_invalidated():
                self.transition_to(State.IDLE, trigger="setup_invalidated")
                return

            # Wait 2-3 bars before looking for trigger
            if bars_waited >= self.wait_bars:
                self.transition_to(
                    State.WAITING_TRIGGER,
                    trigger="wait_complete"
                )

        # ─────────────────────────────────────────────────────
        # STATE: WAITING_TRIGGER - Looking for histogram flip
        # ─────────────────────────────────────────────────────

        elif self.state == State.WAITING_TRIGGER:

            # Timeout after 5 bars
            if self.bars_in_state() > 5:
                self.transition_to(State.IDLE, trigger="trigger_timeout")
                return

            # System 3B: Check for entry trigger
            if self.has_entry_trigger():
                self.execute_entry()
                self.transition_to(State.IN_POSITION, trigger="entry_triggered")

        # ─────────────────────────────────────────────────────
        # STATE: IN_POSITION - Managing the trade
        # ─────────────────────────────────────────────────────

        elif self.state == State.IN_POSITION:

            # Check additional exit conditions
            if self.should_exit():
                self.close_position(
                    reason=ExitReason.TRAILING_SIGNAL,
                    comment="Signal exit"
                )
                self.transition_to(State.IDLE, trigger="position_closed")

    def on_trade_closed(self, trade):
        """Called when a trade closes."""
        # Could add custom logging or analysis here
        pass


def main():
    """Run the MACD Money Map strategy backtest."""

    print("=" * 70)
    print("MACD MONEY MAP STRATEGY BACKTEST")
    print("=" * 70)
    print()

    # Create strategy instance
    strategy_instance = MACDMoneyMap()

    # Create backtest
    backtest = Backtest(
        strategy=strategy_instance,
        symbol='SPY',  # S&P 500 ETF
        timeframes=['1h', '4h', '1d'],  # 1H, 4H, Daily
        base_timeframe='1h'  # Execute on 1H bars
    )

    # Run backtest
    print("Running backtest on SPY...")
    print()

    result = backtest.run(
        period='2y',  # Last 2 years
        verbose=True
    )

    # Print results
    print()
    result.print_summary()

    # Show equity curve info
    print()
    print(f"Equity Curve: {len(result.equity_curve)} data points")
    print(f"Starting Equity: ${result.equity_curve['equity'].iloc[0]:,.2f}")
    print(f"Ending Equity:   ${result.equity_curve['equity'].iloc[-1]:,.2f}")

    # Show recent trades
    if result.trades:
        print()
        print("=" * 70)
        print("TRADE DETAILS")
        print("=" * 70)

        for i, trade in enumerate(result.trades[-10:], 1):  # Last 10 trades
            print(f"\nTrade #{len(result.trades) - 10 + i}:")
            print(f"  Entry:  {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M')} @ ${trade.entry_price:.2f}")
            print(f"  Exit:   {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M')} @ ${trade.exit_price:.2f}")
            print(f"  Reason: {trade.exit_reason.name}")
            print(f"  P&L:    ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
            print(f"  Bars:   {trade.duration_bars}")
            if trade.r_multiple:
                print(f"  R:      {trade.r_multiple:.2f}R")


if __name__ == '__main__':
    main()
