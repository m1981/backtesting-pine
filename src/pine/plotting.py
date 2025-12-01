"""
Visualization module for backtest results using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional

from .core.types import ExitReason
from .position import Trade


class Plotter:
    """
    Handles the creation of interactive charts for backtest analysis.
    """

    def __init__(self, result):
        """
        Initialize with a BacktestResult object.
        """
        self.result = result
        self.strategy = result.strategy
        self.symbol = result.symbol

        # We need the base timeframe data that was used for execution
        # Assuming the strategy stores the data in self.strategy.data
        # We'll grab the 1H data (or whatever the base was)
        self.df = self._get_execution_dataframe()

    def _get_execution_dataframe(self) -> pd.DataFrame:
        """Extract the dataframe used for execution from the strategy."""
        # In the current architecture, strategy.data is MultiTimeframeData
        # We want the base timeframe data.
        # We can infer it from the equity curve index or strategy config.
        # For now, let's assume tf_1h is the execution timeframe as per the example.
        if hasattr(self.strategy.data, 'tf_1h'):
            return self.strategy.data.tf_1h.data

        # Fallback: grab the first available timeframe
        return list(self.strategy.data._data.values())[0].data

    def plot(self, filename: str = "backtest_result.html", show: bool = True):
        """
        Generate and show the interactive plot.
        """
        # Create subplots: Row 1 (Price), Row 2 (MACD), Row 3 (Equity)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{self.symbol} Price & Trades", "MACD (1H)", "Equity Curve")
        )

        # 1. Candlestick Chart
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='Price'
        ), row=1, col=1)

        # 2. Add Indicators (EMAs) if they exist in strategy
        # We look into strategy.indicators dictionary
        if 'macd_1h' in self.strategy.indicators:
            # We can't easily plot EMAs unless we saved them or recalculate them.
            # For this example, let's plot the MACD components which we know exist.
            pass

        # 3. Add Trades (Markers)
        self._add_trade_markers(fig)

        # 4. Add MACD (Row 2)
        self._add_macd_pane(fig)

        # 5. Add Equity Curve (Row 3)
        fig.add_trace(go.Scatter(
            x=self.result.equity_curve.index,
            y=self.result.equity_curve['equity'],
            line=dict(color='blue', width=2),
            name='Equity'
        ), row=3, col=1)

        # Layout styling
        fig.update_layout(
            title=f"Backtest: {self.result.strategy_name} ({self.result.total_return_pct:.2f}%)",
            xaxis_rangeslider_visible=False,
            height=1000,
            template="plotly_dark",
            hovermode="x unified"
        )

        # Remove gaps for non-trading days (optional, makes chart cleaner)
        # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        if filename:
            fig.write_html(filename)
            print(f"Chart saved to {filename}")

        if show:
            fig.show()

    def _add_trade_markers(self, fig):
        """Add entry and exit markers to the chart."""

        for trade in self.result.trades:
            # Entry Marker (Triangle Up for Long)
            fig.add_trace(go.Scatter(
                x=[trade.entry_timestamp],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='yellow'),
                name='Entry',
                showlegend=False,
                hovertext=f"Entry: {trade.entry_price:.2f}<br>{trade.entry_comment}"
            ), row=1, col=1)

            # Exit Marker (Triangle Down)
            color = 'green' if trade.pnl > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[trade.exit_timestamp],
                y=[trade.exit_price],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color=color),
                name='Exit',
                showlegend=False,
                hovertext=f"Exit: {trade.exit_price:.2f}<br>PnL: {trade.pnl:.2f}<br>{trade.exit_reason.name}"
            ), row=1, col=1)

            # Connect Entry and Exit with a line
            fig.add_trace(go.Scatter(
                x=[trade.entry_timestamp, trade.exit_timestamp],
                y=[trade.entry_price, trade.exit_price],
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

    def _add_macd_pane(self, fig):
        """Add MACD lines and histogram to the second pane."""
        # Access the indicator from the strategy
        # Note: This assumes the indicator object stores the full history array
        # The current implementation of MACD in indicators.py calculates everything in __init__
        # so ._macd_line, ._signal_line, ._histogram are numpy arrays matching the data length.

        if 'macd_1h' not in self.strategy.indicators:
            return

        macd = self.strategy.indicators['macd_1h']

        # We need to slice the arrays to match the dataframe index if necessary
        # But usually they should match 1:1 if calculated on the same data

        # MACD Line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=macd._macd_line,
            line=dict(color='#2962FF', width=1.5),
            name='MACD'
        ), row=2, col=1)

        # Signal Line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=macd._signal_line,
            line=dict(color='#FF6D00', width=1.5),
            name='Signal'
        ), row=2, col=1)

        # Histogram
        # Color based on value (Green > 0, Red < 0)
        colors = ['#26A69A' if v >= 0 else '#EF5350' for v in macd._histogram]

        fig.add_trace(go.Bar(
            x=self.df.index,
            y=macd._histogram,
            marker_color=colors,
            name='Hist'
        ), row=2, col=1)