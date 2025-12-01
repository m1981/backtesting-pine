"""
Position and order management for backtesting.

This module handles position tracking, order execution, and trade management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from .core.types import ExitReason


@dataclass
class Position:
    """
    Represents an open position.

    Tracks entry details, current P&L, and position management.
    """
    # Entry details
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: float
    direction: str  # 'long' or 'short'

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Position tracking
    entry_bar_index: int = 0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    current_price: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_comment: str = ""

    # Management
    is_partial: bool = False  # True if position has been partially closed
    original_quantity: float = 0.0  # Original quantity before partial closes

    def __post_init__(self):
        """Initialize computed fields."""
        if self.original_quantity == 0.0:
            self.original_quantity = self.quantity

    def update_pnl(self, current_price: float) -> None:
        """
        Update unrealized P&L based on current price.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price

        if self.direction == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

        # Calculate percentage P&L
        if self.entry_price != 0:
            if self.direction == 'long':
                self.unrealized_pnl_percent = (
                    (current_price - self.entry_price) / self.entry_price * 100
                )
            else:
                self.unrealized_pnl_percent = (
                    (self.entry_price - current_price) / self.entry_price * 100
                )

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss has been hit.

        Args:
            current_price: Current market price

        Returns:
            True if stop loss hit
        """
        if self.stop_loss is None:
            return False

        if self.direction == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit has been hit.

        Args:
            current_price: Current market price

        Returns:
            True if take profit hit
        """
        if self.take_profit is None:
            return False

        if self.direction == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit

    def modify_stop(self, new_stop: float) -> None:
        """
        Modify stop loss price.

        Args:
            new_stop: New stop loss price
        """
        self.stop_loss = new_stop

    def modify_target(self, new_target: Optional[float]) -> None:
        """
        Modify take profit price.

        Args:
            new_target: New take profit price (None to remove)
        """
        self.take_profit = new_target

    def __repr__(self) -> str:
        return (
            f"Position({self.direction.upper()}, qty={self.quantity:.2f}, "
            f"entry={self.entry_price:.2f}, "
            f"pnl={self.unrealized_pnl:.2f} ({self.unrealized_pnl_percent:.2f}%))"
        )


@dataclass
class Trade:
    """
    Represents a completed trade (closed position).

    Used for performance analysis and reporting.
    """
    # Entry details
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_bar_index: int

    # Exit details
    exit_timestamp: pd.Timestamp
    exit_price: float
    exit_bar_index: int
    exit_reason: ExitReason

    # Trade details
    direction: str  # 'long' or 'short'
    quantity: float
    pnl: float
    pnl_percent: float

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Costs
    commission: float = 0.0
    slippage: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_comment: str = ""
    exit_comment: str = ""

    @property
    def duration_bars(self) -> int:
        """Number of bars the trade was held."""
        return self.exit_bar_index - self.entry_bar_index

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.pnl > 0

    @property
    def r_multiple(self) -> Optional[float]:
        """
        Calculate R-multiple (P&L as multiple of initial risk).

        Returns:
            R-multiple if risk was defined, else None
        """
        if self.stop_loss is None:
            return None

        if self.direction == 'long':
            risk = self.entry_price - self.stop_loss
        else:
            risk = self.stop_loss - self.entry_price

        if risk <= 0:
            return None

        return self.pnl / (risk * self.quantity)

    def __repr__(self) -> str:
        return (
            f"Trade({self.direction.upper()}, "
            f"{self.entry_timestamp.strftime('%Y-%m-%d')} -> "
            f"{self.exit_timestamp.strftime('%Y-%m-%d')}, "
            f"P&L: {self.pnl:.2f} ({self.pnl_percent:.2f}%))"
        )


class PositionManager:
    """
    Manages positions and order execution during backtesting.

    Handles:
    - Position opening/closing
    - Stop loss and take profit management
    - Partial position closes
    - Commission and slippage calculation
    """

    def __init__(
        self,
        initial_capital: float,
        commission: float = 0.002,
        slippage: float = 0.001
    ):
        """
        Initialize position manager.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.002 = 0.2%)
            slippage: Slippage rate (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage

        # Position tracking
        self.current_position: Optional[Position] = None
        self.closed_trades: list[Trade] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.current_position is not None

    @property
    def equity(self) -> float:
        """Calculate current equity (capital + unrealized P&L)."""
        if self.current_position:
            return self.capital + self.current_position.unrealized_pnl
        return self.capital

    def open_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        quantity: float,
        direction: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        bar_index: int = 0,
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            timestamp: Entry timestamp
            price: Entry price
            quantity: Position size
            direction: 'long' or 'short'
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            bar_index: Bar index of entry
            comment: Entry comment
            metadata: Additional metadata

        Returns:
            The opened Position

        Raises:
            ValueError: If position already exists
        """
        if self.has_position:
            raise ValueError("Cannot open position: position already exists")

        # Apply slippage
        if direction == 'long':
            entry_price = price * (1 + self.slippage_rate)
        else:
            entry_price = price * (1 - self.slippage_rate)

        # Calculate commission
        commission = entry_price * quantity * self.commission_rate

        # Create position
        self.current_position = Position(
            entry_timestamp=timestamp,
            entry_price=entry_price,
            quantity=quantity,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar_index=bar_index,
            metadata=metadata or {},
            entry_comment=comment
        )

        # Deduct commission from capital
        self.capital -= commission

        return self.current_position

    def close_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        reason: ExitReason,
        bar_index: int = 0,
        comment: str = ""
    ) -> Trade:
        """
        Close the current position.

        Args:
            timestamp: Exit timestamp
            price: Exit price
            reason: Exit reason
            bar_index: Bar index of exit
            comment: Exit comment

        Returns:
            The completed Trade

        Raises:
            ValueError: If no position exists
        """
        if not self.has_position:
            raise ValueError("Cannot close position: no position exists")

        pos = self.current_position

        # Apply slippage
        if pos.direction == 'long':
            exit_price = price * (1 - self.slippage_rate)
        else:
            exit_price = price * (1 + self.slippage_rate)

        # Calculate P&L
        if pos.direction == 'long':
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Calculate commission
        commission = exit_price * pos.quantity * self.commission_rate

        # Deduct commission from P&L
        pnl -= commission

        # Calculate percentage P&L
        if pos.direction == 'long':
            pnl_percent = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_percent = (pos.entry_price - exit_price) / pos.entry_price * 100

        # Create trade record
        trade = Trade(
            entry_timestamp=pos.entry_timestamp,
            entry_price=pos.entry_price,
            entry_bar_index=pos.entry_bar_index,
            exit_timestamp=timestamp,
            exit_price=exit_price,
            exit_bar_index=bar_index,
            exit_reason=reason,
            direction=pos.direction,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            commission=commission * 2,  # Entry + exit
            metadata=pos.metadata,
            entry_comment=pos.entry_comment,
            exit_comment=comment
        )

        # Update capital
        self.capital += pnl

        # Update statistics
        self.total_trades += 1
        if trade.is_winner:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Store trade
        self.closed_trades.append(trade)

        # Clear position
        self.current_position = None

        return trade

    def close_partial(
        self,
        timestamp: pd.Timestamp,
        price: float,
        portion: float,
        reason: ExitReason,
        bar_index: int = 0,
        comment: str = ""
    ) -> Trade:
        """
        Close a portion of the current position.

        Args:
            timestamp: Exit timestamp
            price: Exit price
            portion: Portion to close (0.0 to 1.0, e.g., 0.5 = 50%)
            reason: Exit reason
            bar_index: Bar index of exit
            comment: Exit comment

        Returns:
            The completed Trade for the closed portion

        Raises:
            ValueError: If no position exists or portion invalid
        """
        if not self.has_position:
            raise ValueError("Cannot close position: no position exists")

        if not 0 < portion < 1:
            raise ValueError("Portion must be between 0 and 1")

        pos = self.current_position

        # Calculate quantity to close
        close_quantity = pos.quantity * portion

        # Temporarily set quantity for close_position
        original_quantity = pos.quantity
        pos.quantity = close_quantity

        # Close the partial position
        trade = self.close_position(timestamp, price, reason, bar_index, comment)

        # Restore position with remaining quantity
        pos.quantity = original_quantity - close_quantity
        pos.is_partial = True
        self.current_position = pos

        return trade

    def update_position(self, current_price: float) -> None:
        """
        Update current position with latest price.

        Args:
            current_price: Current market price
        """
        if self.has_position:
            self.current_position.update_pnl(current_price)

    def check_exits(
        self,
        timestamp: pd.Timestamp,
        current_price: float,
        bar_index: int
    ) -> Optional[Trade]:
        """
        Check if any exit conditions are met.

        Args:
            timestamp: Current timestamp
            current_price: Current market price
            bar_index: Current bar index

        Returns:
            Trade if position was closed, else None
        """
        if not self.has_position:
            return None

        pos = self.current_position

        # Check stop loss first (highest priority)
        if pos.check_stop_loss(current_price):
            return self.close_position(
                timestamp=timestamp,
                price=pos.stop_loss,  # Exit at stop price
                reason=ExitReason.STOP_LOSS,
                bar_index=bar_index,
                comment="Stop loss hit"
            )

        # Check take profit
        if pos.check_take_profit(current_price):
            return self.close_position(
                timestamp=timestamp,
                price=pos.take_profit,  # Exit at target price
                reason=ExitReason.TAKE_PROFIT,
                bar_index=bar_index,
                comment="Take profit hit"
            )

        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Dictionary with performance metrics, always with a consistent structure.
        """
        total_pnl = sum(trade.pnl for trade in self.closed_trades)
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': (self.equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital != 0 else 0.0,
            'final_capital': self.capital,
            'current_equity': self.equity
        }
