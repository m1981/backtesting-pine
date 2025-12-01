"""
Strategy base class and decorator for building trading strategies.

This module provides the foundation for building declarative trading strategies
with multi-timeframe support, state management, and automatic position handling.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import pandas as pd

from .core.types import (
    Mode, Bias, State, SetupSignal, ExitReason,
    StrategyConfig, Timeframe
)
from .data import MultiTimeframeData, Bar
from .state_machine import StateMachine
from .position import PositionManager, Position, Trade
from .core.series import Series


class StrategyBase:
    """
    Base class for all trading strategies.

    Provides core functionality for:
    - Multi-timeframe data access
    - State machine management
    - Position management
    - Order execution

    Subclasses should implement:
    - setup_indicators(): Define indicators
    - on_bar(bar): Main strategy logic
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.mode = config.mode

        # Core components
        self.state_machine = StateMachine(initial_state=State.IDLE)
        self.position_mgr = PositionManager(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage
        )

        # Data (set by backtesting engine)
        self.data: Optional[MultiTimeframeData] = None
        self.current_bar: Optional[Bar] = None
        self.current_bar_index: int = 0

        # Indicators (populated by setup_indicators)
        self.indicators: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════
    # Properties for easy access
    # ═══════════════════════════════════════════════════════════════

    @property
    def state(self) -> State:
        """Get current strategy state."""
        return self.state_machine.current_state

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.position_mgr.has_position

    @property
    def current_position(self) -> Optional[Position]:
        """Get current open position."""
        return self.position_mgr.current_position

    @property
    def capital(self) -> float:
        """Get current capital."""
        return self.position_mgr.capital

    @property
    def equity(self) -> float:
        """Get current equity (capital + unrealized P&L)."""
        return self.position_mgr.equity

    @property
    def trades(self) -> List[Trade]:
        """Get list of closed trades."""
        return self.position_mgr.closed_trades

    # ═══════════════════════════════════════════════════════════════
    # State machine helpers
    # ═══════════════════════════════════════════════════════════════

    def transition_to(
        self,
        new_state: State,
        trigger: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            trigger: Description of what triggered the transition
            metadata: Optional metadata
        """
        self.state_machine.transition_to(new_state, trigger, metadata)

    def goto(self, new_state: State) -> None:
        """
        Alias for transition_to with no trigger name.

        Args:
            new_state: Target state
        """
        self.transition_to(new_state)

    def bars_in_state(self) -> int:
        """Get number of bars in current state."""
        return self.state_machine.bars_in_state()

    def bars_since_entry(self) -> int:
        """
        Get number of bars since position entry.

        Returns:
            Number of bars, or 0 if no position
        """
        if not self.has_position:
            return 0
        return self.current_bar_index - self.current_position.entry_bar_index

    # ═══════════════════════════════════════════════════════════════
    # Order execution methods
    # ═══════════════════════════════════════════════════════════════

    def buy(
        self,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Open a long position.

        Args:
            quantity: Position size (if None, calculated from risk)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Entry comment
            metadata: Additional metadata

        Returns:
            The opened Position
        """
        if self.has_position:
            raise ValueError("Cannot buy: position already exists")

        # Calculate quantity based on risk if not provided
        if quantity is None:
            quantity = self._calculate_position_size(
                entry_price=self.current_bar.close,
                stop_loss=stop_loss,
                direction='long'
            )

        return self.position_mgr.open_position(
            timestamp=self.current_bar.timestamp,
            price=self.current_bar.close,
            quantity=quantity,
            direction='long',
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=self.current_bar_index,
            comment=comment,
            metadata=metadata
        )

    def sell(
        self,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Open a short position.

        Args:
            quantity: Position size (if None, calculated from risk)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Entry comment
            metadata: Additional metadata

        Returns:
            The opened Position
        """
        if self.mode == Mode.LONG_ONLY:
            raise ValueError("Cannot sell: strategy is LONG_ONLY mode")

        if self.has_position:
            raise ValueError("Cannot sell: position already exists")

        # Calculate quantity based on risk if not provided
        if quantity is None:
            quantity = self._calculate_position_size(
                entry_price=self.current_bar.close,
                stop_loss=stop_loss,
                direction='short'
            )

        return self.position_mgr.open_position(
            timestamp=self.current_bar.timestamp,
            price=self.current_bar.close,
            quantity=quantity,
            direction='short',
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=self.current_bar_index,
            comment=comment,
            metadata=metadata
        )

    def close_position(
        self,
        reason: ExitReason = ExitReason.MANUAL,
        comment: str = ""
    ) -> Trade:
        """
        Close the current position.

        Args:
            reason: Exit reason
            comment: Exit comment

        Returns:
            The completed Trade
        """
        if not self.has_position:
            raise ValueError("Cannot close: no position exists")

        return self.position_mgr.close_position(
            timestamp=self.current_bar.timestamp,
            price=self.current_bar.close,
            reason=reason,
            bar_index=self.current_bar_index,
            comment=comment
        )

    def close_partial(
        self,
        portion: float,
        reason: ExitReason = ExitReason.TAKE_PROFIT,
        comment: str = ""
    ) -> Trade:
        """
        Close a portion of the current position.

        Args:
            portion: Portion to close (0.0-1.0, e.g., 0.5 = 50%)
            reason: Exit reason
            comment: Exit comment

        Returns:
            Trade for the closed portion
        """
        if not self.has_position:
            raise ValueError("Cannot close: no position exists")

        return self.position_mgr.close_partial(
            timestamp=self.current_bar.timestamp,
            price=self.current_bar.close,
            portion=portion,
            reason=reason,
            bar_index=self.current_bar_index,
            comment=comment
        )

    def modify_stop(self, new_stop: float) -> None:
        """
        Modify stop loss price.

        Args:
            new_stop: New stop loss price
        """
        if not self.has_position:
            raise ValueError("Cannot modify stop: no position exists")

        self.current_position.modify_stop(new_stop)

    def modify_target(self, new_target: Optional[float]) -> None:
        """
        Modify take profit price.

        Args:
            new_target: New take profit price (None to remove)
        """
        if not self.has_position:
            raise ValueError("Cannot modify target: no position exists")

        self.current_position.modify_target(new_target)

    # ═══════════════════════════════════════════════════════════════
    # Risk management
    # ═══════════════════════════════════════════════════════════════

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        direction: str
    ) -> float:
        """
        Calculate position size based on risk.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'long' or 'short'

        Returns:
            Position size in shares/contracts
        """
        if stop_loss is None:
            # Default to 10% of capital if no stop defined
            return (self.capital * 0.1) / entry_price

        # Calculate risk per share
        if direction == 'long':
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price

        if risk_per_share <= 0:
            raise ValueError("Invalid stop loss: would not limit risk")

        # Calculate position size based on risk percentage
        risk_amount = self.capital * self.config.risk_per_trade
        quantity = risk_amount / risk_per_share

        # Ensure we can afford the position
        max_affordable = self.capital / entry_price
        quantity = min(quantity, max_affordable)

        return quantity

    # ═══════════════════════════════════════════════════════════════
    # Methods to override in subclasses
    # ═══════════════════════════════════════════════════════════════

    def setup_indicators(self) -> Dict[str, Any]:
        """
        Define and calculate indicators.

        Should return a dictionary of indicator names to indicator objects.

        Example:
            return {
                'macd_1h': MACD(self.data.tf_1h.close, 12, 26, 9),
                'atr': ATR(self.data.tf_1h.high, self.data.tf_1h.low,
                          self.data.tf_1h.close, 14)
            }

        Returns:
            Dictionary of indicators
        """
        return {}

    def on_bar(self, bar: Bar) -> None:
        """
        Main strategy logic called on each bar.

        Override this method to implement strategy logic.

        Args:
            bar: Current bar
        """
        pass

    def on_trade_closed(self, trade: Trade) -> None:
        """
        Called when a trade is closed.

        Override to implement custom trade analysis or logging.

        Args:
            trade: The completed trade
        """
        pass

    # ═══════════════════════════════════════════════════════════════
    # Internal methods
    # ═══════════════════════════════════════════════════════════════

    def _initialize(self, data: MultiTimeframeData) -> None:
        """
        Initialize strategy with data.

        Called by backtesting engine before running.

        Args:
            data: Multi-timeframe data
        """
        self.data = data
        self.indicators = self.setup_indicators()

    def _process_bar(self, bar: Bar, bar_index: int) -> None:
        """
        Process a single bar.

        Called by backtesting engine.
        """
        self.current_bar = bar
        self.current_bar_index = bar_index

        # --- NEW: SYNCHRONIZE INDICATOR SERIES INDICES ---
        series_index = len(self.data.tf_1h.data) - 1 - bar_index
        for indicator in self.indicators.values():
            if hasattr(indicator, 'macd_line'): # Handle MACDResult
                indicator.macd_line.set_index(series_index)
                indicator.signal_line.set_index(series_index)
                indicator.histogram.set_index(series_index)
            elif hasattr(indicator, 'values'): # Handle ATR
                indicator.values.set_index(series_index)
        # -----------------------------------------------

        # Update position with current price
        if self.has_position:
            self.position_mgr.update_position(bar.close)

        # Check automatic exits (stop loss, take profit)
        trade = self.position_mgr.check_exits(
            timestamp=bar.timestamp,
            current_price=bar.close,
            bar_index=bar_index
        )

        if trade:
            self.on_trade_closed(trade)
            # If position was closed, transition to IDLE
            if not self.has_position:
                self.transition_to(State.IDLE, trigger="position_closed")

        # Call strategy logic
        self.on_bar(bar)

        # Advance state machine
        self.state_machine.advance_bar()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.

        Returns:
            Dictionary with performance metrics
        """
        return self.position_mgr.get_summary()


def strategy(
    name: str,
    description: str = "",
    mode: Mode = Mode.LONG_ONLY,
    initial_capital: float = 10000.0,
    commission: float = 0.002,
    slippage: float = 0.001,
    risk_per_trade: float = 0.02
):
    """
    Decorator for strategy classes.

    Example:
        @strategy(
            name="My Strategy",
            mode=Mode.LONG_ONLY,
            initial_capital=10000
        )
        class MyStrategy(StrategyBase):
            def on_bar(self, bar):
                # Strategy logic here
                pass

    Args:
        name: Strategy name
        description: Strategy description
        mode: Trading mode
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate
        risk_per_trade: Risk per trade as fraction of capital

    Returns:
        Decorated class
    """
    def decorator(cls):
        # Store config as class attribute
        cls._strategy_config = StrategyConfig(
            name=name,
            description=description,
            mode=mode,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            risk_per_trade=risk_per_trade
        )

        # Modify __init__ to use config
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Initialize base class with config
            StrategyBase.__init__(self, cls._strategy_config)
            # Call original init if it exists and isn't just pass
            if original_init != StrategyBase.__init__:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        return cls

    return decorator
