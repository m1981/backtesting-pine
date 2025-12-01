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
from .data import DataLoader, MultiTimeframeData, create_bars_iterator, TimeframeData
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
    strategy: Optional[StrategyBase] = None

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

    def plot(self, filename: str = "backtest_result.html", show: bool = True):
        """Visualize the backtest results."""
        from .plotting import Plotter
        plotter = Plotter(self)
        plotter.plot(filename=filename, show=show)

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
            for trade in self.trades[-5:]:
                print(f"  {trade}")


class Backtest:
    """Backtesting engine."""

    def __init__(
        self,
        strategy: StrategyBase,
        timeframes: List[Union[str, Timeframe]],
        symbol: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        base_timeframe: Optional[Union[str, Timeframe]] = None
    ):
        if data is None and symbol is None:
            raise ValueError("Either `symbol` or `data` must be provided.")

        self.strategy = strategy
        self.symbol = symbol or "SYNTHETIC"
        self.data = data
        self.timeframes = [Timeframe.from_string(tf) if isinstance(tf, str) else tf for tf in timeframes]
        if base_timeframe:
            self.base_timeframe = Timeframe.from_string(base_timeframe) if isinstance(base_timeframe, str) else base_timeframe
        else:
            self.base_timeframe = self.timeframes[0]
        self.equity_curve_data = []

    def _prepare_data(self, start_date=None, end_date=None, period=None) -> MultiTimeframeData:
        """Loads or uses provided data and prepares it for the backtest."""
        if self.data is not None:
            # Use pre-loaded data
            base_df = self.data
        else:
            # Fetch data using DataLoader
            loader = DataLoader(symbol=self.symbol, base_timeframe=self.base_timeframe)
            base_df = loader.fetch(start_date=start_date, end_date=end_date, period=period)

        # Create the dictionary of TimeframeData objects by resampling the base_df
        mtf_data_dict = {}
        resampling_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }

        for tf in self.timeframes:
            if tf == self.base_timeframe:
                mtf_data_dict[tf] = TimeframeData(tf, base_df)
            else:
                resampled_df = base_df.resample(tf.to_pandas_freq()).agg(resampling_rules).dropna()
                mtf_data_dict[tf] = TimeframeData(tf, resampled_df)

        # Create named timeframe dictionary for strategy access
        timeframe_names = self._create_timeframe_names(self.timeframes)
        named_data = {
            name: mtf_data_dict[tf]
            for name, tf in zip(timeframe_names, self.timeframes)
        }

        return MultiTimeframeData(named_data)

    def run(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run the backtest.
        """
        if verbose:
            print(f"Preparing data for {self.symbol}...")

            # This call correctly prepares and returns the final MultiTimeframeData object.
        mtf_data = self._prepare_data(start_date, end_date, period)

        # Initialize strategy with data
        if verbose:
            print(f"Initializing strategy: {self.strategy.config.name}")

        self.strategy._initialize(mtf_data)

        base_tf_data = mtf_data._data[self._create_timeframe_names([self.base_timeframe])[0]]
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
                print(f"Progress: {(i + 1) / len(bars) * 100:.1f}% ({i + 1}/{len(bars)} bars)", end='\r')

        if verbose:
            print("\nBacktest complete!")
        return self._create_result(bars)

    def _create_timeframe_names(self, timeframes: List[Timeframe]) -> List[str]:
        names = []
        for tf in timeframes:
            if tf == Timeframe.HOUR_1: names.append('tf_1h')
            elif tf == Timeframe.HOUR_4: names.append('tf_4h')
            elif tf == Timeframe.DAILY: names.append('daily')
            elif tf == Timeframe.MINUTE_15: names.append('tf_15m')
            elif tf == Timeframe.MINUTE_30: names.append('tf_30m')
            elif tf == Timeframe.WEEKLY: names.append('weekly')
            else: names.append(f'tf_{tf.name.lower()}')
        return names

    def _create_result(self, bars: List) -> BacktestResult:
        summary = self.strategy.get_summary()
        equity_df = pd.DataFrame(self.equity_curve_data).set_index('timestamp')
        initial_capital = self.strategy.config.initial_capital
        final_capital = self.strategy.capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100 if initial_capital != 0 else 0

        return BacktestResult(
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
            win_rate=summary.get('win_rate', 0.0),
            trades=self.strategy.trades,
            equity_curve=equity_df,
            strategy=self.strategy
        )

    def optimize(self, param_grid: Dict[str, List], metric: str = 'total_return_pct', **run_kwargs) -> pd.DataFrame:
        raise NotImplementedError("Optimization not yet implemented")
