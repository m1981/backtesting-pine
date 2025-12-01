"""
MACD Money Map Backtesting App

A Streamlit-based GUI for running and visualizing backtests of the MACD Money Map strategy.
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from pine import (
    StrategyBase, strategy, Backtest,
    MACD, ATR, crossover, crossunder,
    Mode, Bias, State, SetupSignal, ExitReason
)
from pine.data import DataLoader
from pine.core.types import Timeframe

# Import the strategy
from examples.macd_money_map import MACDMoneyMap


# Page configuration
st.set_page_config(
    page_title="MACD Money Map Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .positive { color: #00cc00; }
    .negative { color: #ff4444; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_trending_data(n_bars: int = 1000, base_price: float = 100.0,
                           trend_strength: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Generate trending price data (uptrend)."""
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')

    # Generate uptrend with noise
    trend = np.linspace(0, trend_strength, n_bars)
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


def generate_choppy_data(n_bars: int = 1000, base_price: float = 100.0,
                         volatility: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """Generate choppy/sideways price data."""
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')

    # Generate sideways movement with high noise
    noise = np.random.normal(0, volatility, n_bars)
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


def generate_volatile_trending_data(n_bars: int = 1000, base_price: float = 100.0,
                                    trend_strength: float = 0.4, volatility: float = 0.025,
                                    seed: int = 42) -> pd.DataFrame:
    """Generate volatile trending data with pullbacks - ideal for MACD strategy."""
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')

    # Generate trend with pullbacks
    trend = np.linspace(0, trend_strength, n_bars)

    # Add cyclical component for pullbacks
    cycles = 0.05 * np.sin(np.linspace(0, 8 * np.pi, n_bars))

    # Add noise
    noise = np.random.normal(0, volatility, n_bars)

    prices = base_price * np.exp(trend + cycles + np.cumsum(noise * 0.3))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.008, 0.008, n_bars)),
        'high': prices * (1 + np.random.uniform(0.002, 0.02, n_bars)),
        'low': prices * (1 + np.random.uniform(-0.02, -0.002, n_bars)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def main():
    st.markdown('<p class="main-header">📈 MACD Money Map Backtester</p>', unsafe_allow_html=True)
    st.markdown("*Multi-timeframe MACD strategy with trend and confirmation systems*")

    # Sidebar - Strategy Parameters
    with st.sidebar:
        st.header("📊 Data Source")

        data_source = st.radio(
            "Select Data Source",
            options=["Live Data (yfinance)", "Synthetic Data"],
            index=1,
            help="Use live market data or synthetic data for testing"
        )

        if data_source == "Live Data (yfinance)":
            symbol = st.text_input("Symbol", value="SPY", help="Stock ticker symbol")
            period = st.selectbox(
                "Data Period",
                options=["6mo", "1y", "2y"],
                index=1,
                help="Historical data period (yfinance limitation: max 730 days for hourly data)"
            )
            synthetic_type = None
            n_bars = None
        else:
            symbol = "SYNTHETIC"
            period = None
            synthetic_type = st.selectbox(
                "Market Type",
                options=["Trending (Uptrend)", "Choppy (Sideways)", "Volatile Trending (Best for MACD)"],
                index=2,
                help="Type of synthetic market data to generate"
            )
            n_bars = st.slider(
                "Number of Bars",
                min_value=500, max_value=3000, value=1500, step=100,
                help="Number of 1-hour bars to generate"
            )
            seed = st.number_input(
                "Random Seed",
                min_value=1, max_value=9999, value=42,
                help="Seed for reproducible results"
            )

        st.header("⚙️ Strategy Parameters")

        st.subheader("MACD Settings")
        fast_ema = st.slider("Fast EMA", min_value=5, max_value=20, value=12)
        slow_ema = st.slider("Slow EMA", min_value=15, max_value=50, value=26)
        signal_period = st.slider("Signal Period", min_value=5, max_value=15, value=9)

        st.subheader("Risk Management")
        atr_threshold = st.slider(
            "ATR Threshold",
            min_value=0.1, max_value=1.0, value=0.5, step=0.1,
            help="Multiplier for distance from zero line"
        )
        wait_bars = st.slider(
            "Wait Bars",
            min_value=1, max_value=5, value=2,
            help="Bars to wait after crossover before entry"
        )
        risk_reward = st.slider(
            "Risk:Reward Ratio",
            min_value=1.0, max_value=4.0, value=2.0, step=0.5
        )

        st.subheader("Capital Settings")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000, max_value=1000000, value=10000, step=1000
        )
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5
        ) / 100

        # Run button
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Main content area
    if run_backtest:
        # Prepare synthetic data parameters
        synthetic_params = None
        if data_source == "Synthetic Data":
            synthetic_params = {
                'type': synthetic_type,
                'n_bars': n_bars,
                'seed': seed
            }

        run_backtest_and_display(
            symbol=symbol,
            period=period,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            signal_period=signal_period,
            atr_threshold=atr_threshold,
            wait_bars=wait_bars,
            risk_reward=risk_reward,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            synthetic_params=synthetic_params
        )
    else:
        # Show instructions
        st.info("👈 Configure parameters in the sidebar and click **Run Backtest** to start.")

        # Show strategy explanation
        with st.expander("📖 Strategy Overview", expanded=True):
            st.markdown("""
            ### MACD Money Map Strategy

            This strategy implements a **three-system approach**:

            #### System 1: Trend Detection (4H)
            - Detects bullish crossovers (MACD crosses above Signal line)
            - Only valid if crossover happens **far from zero** (> ATR threshold × ATR)
            - Filters out "chop zone" signals that have 80% failure rate

            #### System 2: Reversal Detection (Not implemented)
            - Divergence detection between price and MACD

            #### System 3: Multi-Timeframe Confirmation
            - **Part A (Daily):** MACD > 0 = Bullish bias (only longs allowed)
            - **Part B (1H):** Histogram flip from red to green = Entry trigger

            #### Entry Flow
            1. Daily MACD must be above zero (bullish bias)
            2. 4H MACD crossover detected far from zero
            3. Wait 2-3 bars for confirmation
            4. 1H histogram flips green → Enter long

            #### Exit Conditions
            - Stop loss hit
            - Take profit (2R target)
            - 4H bearish crossover (trailing signal)
            - Daily bias changes to bearish
            """)


def run_backtest_and_display(**params):
    """Run backtest with given parameters and display results."""

    symbol = params['symbol']
    synthetic_params = params.get('synthetic_params')

    # Generate synthetic data if requested
    synthetic_data = None
    if synthetic_params:
        with st.spinner("Generating synthetic data..."):
            syn_type = synthetic_params['type']
            n_bars = synthetic_params['n_bars']
            seed = synthetic_params['seed']

            if "Trending (Uptrend)" in syn_type:
                synthetic_data = generate_trending_data(n_bars=n_bars, seed=seed)
            elif "Choppy" in syn_type:
                synthetic_data = generate_choppy_data(n_bars=n_bars, seed=seed)
            else:  # Volatile Trending
                synthetic_data = generate_volatile_trending_data(n_bars=n_bars, seed=seed)

            st.success(f"Generated {len(synthetic_data)} bars of {syn_type} data")

    with st.spinner(f"Running backtest on {symbol}..."):
        try:
            # Create a custom strategy class with the parameters
            @strategy(
                name="MACD Money Map",
                description="Multi-timeframe MACD strategy",
                mode=Mode.LONG_ONLY,
                initial_capital=params['initial_capital'],
                commission=0.002,
                slippage=0.001,
                risk_per_trade=params['risk_per_trade']
            )
            class CustomMACDMoneyMap(MACDMoneyMap):
                fast_ema = params['fast_ema']
                slow_ema = params['slow_ema']
                signal_period = params['signal_period']
                atr_threshold = params['atr_threshold']
                wait_bars = params['wait_bars']
                risk_reward = params['risk_reward']

            # Instantiate the strategy
            strategy_instance = CustomMACDMoneyMap()

            # Create backtest with either synthetic or live data
            if synthetic_data is not None:
                backtest = Backtest(
                    strategy=strategy_instance,
                    data=synthetic_data,
                    timeframes=[Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAILY]
                )
                # Run backtest without period (data already provided)
                result = backtest.run(verbose=False)
            else:
                backtest = Backtest(
                    strategy=strategy_instance,
                    symbol=symbol,
                    timeframes=[Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAILY]
                )
                # Run backtest with period
                result = backtest.run(
                    period=params['period'],
                    verbose=False
                )

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Display results
    display_results(result, params)


def display_results(result, params):
    """Display backtest results with charts and metrics."""

    # Summary metrics
    st.header("📊 Backtest Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Capital",
            f"${result.final_capital:,.2f}",
            f"{result.total_return_pct:+.2f}%"
        )

    with col2:
        st.metric("Total Trades", result.total_trades)

    with col3:
        st.metric("Win Rate", f"{result.win_rate:.1f}%")

    with col4:
        profit_factor = result.profit_factor if hasattr(result, 'profit_factor') else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor else "N/A")

    # Period info
    st.caption(f"Period: {result.start_date} to {result.end_date} | Symbol: {result.symbol}")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "📋 Trade List", "📉 Price Chart"])

    with tab1:
        display_equity_curve(result)

    with tab2:
        display_trade_list(result)

    with tab3:
        display_price_chart(result, params)


def display_equity_curve(result):
    """Display equity curve chart."""
    if result.equity_curve is None or len(result.equity_curve) == 0:
        st.warning("No equity curve data available.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add initial capital line
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital"
    )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_trade_list(result):
    """Display list of trades."""
    if not result.trades:
        st.info("No trades executed during this period.")
        return

    # Convert trades to DataFrame
    trades_data = []
    for trade in result.trades:
        trades_data.append({
            'Entry Date': trade.entry_timestamp,
            'Exit Date': trade.exit_timestamp,
            'Direction': trade.direction.upper(),
            'Entry Price': f"${trade.entry_price:.2f}",
            'Exit Price': f"${trade.exit_price:.2f}",
            'Quantity': f"{trade.quantity:.2f}",
            'P&L': f"${trade.pnl:.2f}",
            'P&L %': f"{trade.pnl_percent:.2f}%",
            'Exit Reason': trade.exit_reason.name if trade.exit_reason else 'N/A'
        })

    df = pd.DataFrame(trades_data)
    st.dataframe(df, use_container_width=True)

    # Summary stats
    if len(result.trades) > 0:
        total_pnl = sum(t.pnl for t in result.trades)
        avg_pnl = total_pnl / len(result.trades)
        winners = [t for t in result.trades if t.pnl > 0]
        losers = [t for t in result.trades if t.pnl < 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L", f"${total_pnl:.2f}")
        with col2:
            st.metric("Average P&L", f"${avg_pnl:.2f}")
        with col3:
            st.metric("Winners/Losers", f"{len(winners)}/{len(losers)}")


def display_price_chart(result, params):
    """Display price chart with MACD indicators and trade markers."""

    # Get the strategy and data
    strat = result.strategy
    if not strat or not hasattr(strat, 'data'):
        st.warning("Strategy data not available for charting.")
        return

    # Get the 1H data (execution timeframe)
    try:
        df = strat.data.tf_1h.data
    except:
        st.warning("Could not access price data.")
        return

    # Create subplots: Price (top), MACD (middle), Equity (bottom)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{result.symbol} Price & Trades", "MACD (1H)", "Equity Curve")
    )

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ), row=1, col=1)

    # 2. Add Trade Markers
    for trade in result.trades:
        # Entry marker (green triangle up)
        fig.add_trace(go.Scatter(
            x=[trade.entry_timestamp],
            y=[trade.entry_price],
            mode='markers',
            marker=dict(symbol='triangle-up', size=14, color='#00FF00', line=dict(width=1, color='white')),
            name='Entry',
            showlegend=False,
            hovertext=f"ENTRY<br>Price: ${trade.entry_price:.2f}<br>Qty: {trade.quantity:.2f}"
        ), row=1, col=1)

        # Exit marker (color based on P&L)
        exit_color = '#00FF00' if trade.pnl > 0 else '#FF0000'
        fig.add_trace(go.Scatter(
            x=[trade.exit_timestamp],
            y=[trade.exit_price],
            mode='markers',
            marker=dict(symbol='triangle-down', size=14, color=exit_color, line=dict(width=1, color='white')),
            name='Exit',
            showlegend=False,
            hovertext=f"EXIT ({trade.exit_reason.name if trade.exit_reason else 'N/A'})<br>Price: ${trade.exit_price:.2f}<br>P&L: ${trade.pnl:.2f}"
        ), row=1, col=1)

        # Connect entry and exit with a line
        fig.add_trace(go.Scatter(
            x=[trade.entry_timestamp, trade.exit_timestamp],
            y=[trade.entry_price, trade.exit_price],
            mode='lines',
            line=dict(color=exit_color, width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    # 3. Add MACD indicator
    if 'macd_1h' in strat.indicators:
        macd = strat.indicators['macd_1h']

        # MACD Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=macd._macd_line_data,
            line=dict(color='#2962FF', width=1.5),
            name='MACD'
        ), row=2, col=1)

        # Signal Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=macd._signal_line_data,
            line=dict(color='#FF6D00', width=1.5),
            name='Signal'
        ), row=2, col=1)

        # Histogram (colored bars)
        colors = ['#26A69A' if v >= 0 else '#EF5350' for v in macd._histogram_data]
        fig.add_trace(go.Bar(
            x=df.index,
            y=macd._histogram_data,
            marker_color=colors,
            name='Histogram',
            showlegend=False
        ), row=2, col=1)

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 4. Equity Curve
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve['equity'],
        line=dict(color='#1f77b4', width=2),
        name='Equity',
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ), row=3, col=1)

    # Add initial capital line
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        row=3, col=1
    )

    # Layout styling
    fig.update_layout(
        title=f"Backtest: {result.strategy_name} | Return: {result.total_return_pct:+.2f}%",
        xaxis_rangeslider_visible=False,
        height=900,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Show trade summary below chart
    if result.trades:
        st.subheader("📊 Trade Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(result.trades))
        with col2:
            winners = len([t for t in result.trades if t.pnl > 0])
            st.metric("Winners", winners)
        with col3:
            losers = len([t for t in result.trades if t.pnl <= 0])
            st.metric("Losers", losers)
        with col4:
            avg_bars = sum(getattr(t, 'bars_held', 0) for t in result.trades) / len(result.trades) if result.trades else 0
            st.metric("Avg Bars Held", f"{avg_bars:.1f}")


if __name__ == "__main__":
    main()

