"""
Backtesting engine for running strategies.

This module provides the main engine that orchestrates strategy execution,
data management, and performance tracking.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

from .strategy import StrategyBase
from .data import DataLoader, MultiTimeframeData, create_bars_iterator
from .core.types import Timeframe
from .position import Trade


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime

    # Performance metrics
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Trade details
    trades: List[Trade]

    # Equity curve
    equity_curve: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  Strategy: {self.strategy_name}\n"
            f"  Symbol: {self.symbol}\n"
            f"  Period: {self.start_date.date()} to {self.end_date.date()}\n"
            f"  Return: {self.total_return_pct:.2f}%\n"
            f"  Trades: {self.total_trades} (Win Rate: {self.win_rate:.1f}%)\n"
            f"  Final Capital: ${self.final_capital:,.2f}\n"
            f")"
        )

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate
        }

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print("=" * 60)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("=" * 60)
        print(f"Symbol:           {self.symbol}")
        print(f"Period:           {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${self.final_capital:,.2f}")
        print(f"Total Return:     ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)")
        print()
        print(f"Total Trades:     {self.total_trades}")
        print(f"Winning Trades:   {self.winning_trades}")
        print(f"Losing Trades:    {self.losing_trades}")
        print(f"Win Rate:         {self.win_rate:.1f}%")
        print("=" * 60)

        if self.trades:
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:  # Show last 5 trades
                print(f"  {trade}")


class Backtest:
    """
    Backtesting engine.

    Runs a strategy on historical data and produces performance results.

    Example:
        # Create strategy
        strategy = MACDMoneyMap()

        # Run backtest
        bt = Backtest(
            strategy=strategy,
            symbol='AAPL',
            timeframes=['1h', '4h', '1d']
        )

        result = bt.run(period='2y')
        result.print_summary()
    """

    def __init__(
        self,
        strategy: StrategyBase,
        symbol: str,
        timeframes: List[Union[str, Timeframe]],
        base_timeframe: Optional[Union[str, Timeframe]] = None
    ):
        """
        Initialize backtesting engine.

        Args:
            strategy: Strategy instance to test
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            timeframes: List of timeframes to load (e.g., ['1h', '4h', '1d'])
            base_timeframe: Base timeframe for bar-by-bar execution (default: first in list)
        """
        self.strategy = strategy
        self.symbol = symbol

        # Convert timeframe strings to Timeframe enums
        self.timeframes = []
        for tf in timeframes:
            if isinstance(tf, str):
                self.timeframes.append(Timeframe.from_string(tf))
            else:
                self.timeframes.append(tf)

        # Set base timeframe (execution timeframe)
        if base_timeframe:
            if isinstance(base_timeframe, str):
                self.base_timeframe = Timeframe.from_string(base_timeframe)
            else:
                self.base_timeframe = base_timeframe
        else:
            # Use first timeframe as base
            self.base_timeframe = self.timeframes[0]

        # Data loader
        self.data_loader = DataLoader(symbol=symbol, base_timeframe=self.base_timeframe)

        # Tracking
        self.equity_curve_data: List[Dict] = []

    def run(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            period: Period string (e.g., '1y', '2y') instead of dates
            verbose: Print progress messages

        Returns:
            BacktestResult with performance metrics and trade history
        """
        if verbose:
            print(f"Loading data for {self.symbol}...")

        # Load multi-timeframe data
        mtf_data_dict = self.data_loader.load_multiple_timeframes(
            timeframes=self.timeframes,
            start_date=start_date,
            end_date=end_date,
            period=period
        )

        # Create named timeframe dictionary for strategy access
        # Map timeframes to friendly names
        timeframe_names = self._create_timeframe_names(self.timeframes)
        named_data = {
            name: mtf_data_dict[tf]
            for name, tf in zip(timeframe_names, self.timeframes)
        }

        # Create MultiTimeframeData object
        mtf_data = MultiTimeframeData(named_data)

        # Initialize strategy with data
        if verbose:
            print(f"Initializing strategy: {self.strategy.config.name}")

        self.strategy._initialize(mtf_data)

        # Get bars for base timeframe (execution timeframe)
        base_tf_data = mtf_data_dict[self.base_timeframe]
        bars = create_bars_iterator(base_tf_data.data)

        if verbose:
            print(f"Running backtest on {len(bars)} bars...")

        # Reset tracking
        self.equity_curve_data = []

        # Run bar-by-bar
        for i, bar in enumerate(bars):
            # Process bar in strategy
            self.strategy._process_bar(bar, i)

            # Record equity
            self.equity_curve_data.append({
                'timestamp': bar.timestamp,
                'equity': self.strategy.equity,
                'capital': self.strategy.capital,
                'in_position': self.strategy.has_position
            })

            # Progress indicator
            if verbose and (i + 1) % 100 == 0:
                progress = (i + 1) / len(bars) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(bars)} bars)", end='\r')

        if verbose:
            print()  # New line after progress
            print("Backtest complete!")

        # Create result
        result = self._create_result(bars)

        return result

    def _create_timeframe_names(self, timeframes: List[Timeframe]) -> List[str]:
        """
        Create friendly names for timeframes.

        Args:
            timeframes: List of Timeframe enums

        Returns:
            List of friendly names
        """
        names = []
        for tf in timeframes:
            if tf == Timeframe.HOUR_1:
                names.append('tf_1h')
            elif tf == Timeframe.HOUR_4:
                names.append('tf_4h')
            elif tf == Timeframe.DAILY:
                names.append('daily')
            elif tf == Timeframe.MINUTE_15:
                names.append('tf_15m')
            elif tf == Timeframe.MINUTE_30:
                names.append('tf_30m')
            elif tf == Timeframe.WEEKLY:
                names.append('weekly')
            else:
                # Generic name
                names.append(f'tf_{tf.name.lower()}')

        return names

    def _create_result(self, bars: List) -> BacktestResult:
        """
        Create BacktestResult from strategy state.

        Args:
            bars: List of bars processed

        Returns:
            BacktestResult object
        """
        summary = self.strategy.get_summary()

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve_data)
        equity_df.set_index('timestamp', inplace=True)

        # Calculate metrics
        initial_capital = self.strategy.config.initial_capital
        final_capital = self.strategy.capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        result = BacktestResult(
            strategy_name=self.strategy.config.name,
            symbol=self.symbol,
            start_date=bars[0].timestamp.to_pydatetime(),
            end_date=bars[-1].timestamp.to_pydatetime(),
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=summary['total_trades'],
            winning_trades=summary['winning_trades'],
            losing_trades=summary['losing_trades'],
            win_rate=summary['win_rate'],
            trades=self.strategy.trades,
            equity_curve=equity_df
        )

        return result

    def optimize(
        self,
        param_grid: Dict[str, List],
        metric: str = 'total_return_pct',
        **run_kwargs
    ) -> pd.DataFrame:
        """
        Run parameter optimization.

        Args:
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize (default: 'total_return_pct')
            **run_kwargs: Arguments passed to run()

        Returns:
            DataFrame with results for each parameter combination

        Example:
            results = bt.optimize(
                param_grid={
                    'fast_ema': [8, 12, 16],
                    'slow_ema': [21, 26, 30]
                },
                period='2y'
            )
        """
        # TODO: Implement grid search optimization
        # This would iterate through parameter combinations,
        # run backtests, and return results sorted by metric
        raise NotImplementedError("Optimization not yet implemented")
