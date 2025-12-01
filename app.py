"""
MACD Money Map Backtesting App

A Streamlit-based GUI for running and visualizing backtests of the MACD Money Map strategy.
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
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


def main():
    st.markdown('<p class="main-header">📈 MACD Money Map Backtester</p>', unsafe_allow_html=True)
    st.markdown("*Multi-timeframe MACD strategy with trend and confirmation systems*")

    # Sidebar - Strategy Parameters
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")

        # Symbol selection
        symbol = st.text_input("Symbol", value="SPY", help="Stock ticker symbol")

        # Data period
        period = st.selectbox(
            "Data Period",
            options=["6mo", "1y", "2y"],
            index=2,
            help="Historical data period (yfinance limitation: max 730 days for hourly data)"
        )

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
            risk_per_trade=risk_per_trade
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

    with st.spinner(f"Loading data for {params['symbol']}..."):
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

            # Create backtest
            backtest = Backtest(
                strategy=strategy_instance,
                symbol=params['symbol'],
                timeframes=[Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAILY]
            )

            # Run backtest
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
            'Entry Date': trade.entry_time,
            'Exit Date': trade.exit_time,
            'Direction': trade.direction.upper(),
            'Entry Price': f"${trade.entry_price:.2f}",
            'Exit Price': f"${trade.exit_price:.2f}",
            'Quantity': f"{trade.quantity:.2f}",
            'P&L': f"${trade.pnl:.2f}",
            'P&L %': f"{trade.pnl_pct:.2f}%",
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
    """Display price chart with trade markers."""
    st.info("Price chart with trade markers - Coming soon!")
    # TODO: Implement price chart with MACD indicators and trade markers


if __name__ == "__main__":
    main()

