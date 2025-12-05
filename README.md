# backtesting-pine

### Chat 
https://aistudio.google.com/prompts/1OwUbZU1fhhhsogJ_qyEo_hurzh51pjj9

A Python backtesting framework inspired by Pine Script for building multi-timeframe trading strategies with an intuitive, declarative API.

## Features

- **Multi-timeframe support**: Seamlessly work with multiple timeframes (1H, 4H, Daily, etc.)
- **Declarative API**: Write strategies that read like trading rules
- **Built-in state machine**: Track complex strategy states automatically
- **Risk management**: Automatic position sizing based on risk parameters
- **Pine Script-like syntax**: Familiar `.current`, `.previous`, `.highest()`, `.lowest()` accessors
- **Comprehensive indicators**: MACD, ATR, EMA, SMA with more coming
- **Professional backtesting**: Handles commission, slippage, and realistic order execution

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd backtesting-pine

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from pine import (
    StrategyBase, strategy, Backtest,
    MACD, ATR, crossover,
    Mode, State
)

@strategy(
    name="My Strategy",
    mode=Mode.LONG_ONLY,
    initial_capital=10000
)
class MyStrategy(StrategyBase):

    def setup_indicators(self):
        return {
            'macd': MACD(self.data.daily.close, 12, 26, 9),
            'atr': ATR(self.data.daily.high, self.data.daily.low,
                      self.data.daily.close, 14)
        }

    def on_bar(self, bar):
        if self.state == State.IDLE:
            macd = self.indicators['macd']

            # Check for bullish crossover
            if crossover(macd.macd_line, macd.signal_line):
                atr = self.indicators['atr'].values.current
                stop_loss = bar.close - (2 * atr)
                take_profit = bar.close + (4 * atr)

                self.buy(
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment="MACD Crossover"
                )
                self.goto(State.IN_POSITION)

# Run backtest
strategy_instance = MyStrategy()
backtest = Backtest(
    strategy=strategy_instance,
    symbol='AAPL',
    timeframes=['1d']
)

result = backtest.run(period='2y')
result.print_summary()
```

## Example Strategy

The repository includes a complete implementation of the "MACD Money Map" strategy - a sophisticated multi-timeframe system combining trend following and confirmation signals:

```bash
python examples/macd_money_map.py
```

This strategy demonstrates:
- Three-system architecture (Trend + Confirmation)
- Multi-timeframe analysis (Daily/4H/1H)
- State machine for complex logic
- Multiple exit conditions
- Risk-based position sizing

## Project Structure

```
src/pine/
├── core/
│   ├── types.py       # Core enums and dataclasses
│   └── series.py      # Series class for time-series data
├── indicators.py      # Technical indicators (MACD, ATR, etc.)
├── data.py           # Multi-timeframe data loading
├── position.py       # Position and trade management
├── state_machine.py  # State machine infrastructure
├── strategy.py       # Strategy base class and decorator
└── backtest.py       # Backtesting engine

examples/
└── macd_money_map.py # Complete strategy example
```

## Documentation

- **CLAUDE.md** - Architecture overview and development guide
- **spec.md** - MACD Money Map strategy specification
- **api.md** - Detailed API design documentation
- **system-architecture.mermaid** - Visual system architecture

## Key Concepts

### Series Class

Access historical data with intuitive properties:

```python
# Current and historical values
close.current          # Current bar close
close.previous         # Previous bar close
close[5]              # 5 bars ago

# Aggregations
high.highest(20)      # Highest high in 20 bars
low.lowest(50)        # Lowest low in 50 bars
close.average(10)     # 10-bar average

# Crossovers
macd.crossed_above(signal)   # Bullish crossover
macd.crossed_below(signal)   # Bearish crossover
```

### Multi-Timeframe Data

Access different timeframes seamlessly:

```python
# Daily data
self.data.daily.close.current

# 4-Hour data
self.data.tf_4h.high.highest(20)

# 1-Hour data
self.data.tf_1h.low.lowest(10)
```

### State Machine

Track strategy states automatically:

```python
# Define states
self.state == State.IDLE
self.state == State.WAITING_CONFIRMATION
self.state == State.IN_POSITION

# Transition between states
self.transition_to(State.WAITING_TRIGGER, trigger="setup_detected")

# Query state info
self.bars_in_state()      # Bars in current state
self.bars_since_entry()   # Bars since position opened
```

## Requirements

- Python 3.8+
- numpy >= 1.20.0
- pandas >= 1.3.0
- yfinance >= 0.2.0

## Development Status

This is an early-stage project. Core functionality is implemented:

- ✅ Core data structures (Series, Bar, TimeframeData)
- ✅ Multi-timeframe data loading and resampling
- ✅ Technical indicators (MACD, ATR, EMA, SMA)
- ✅ Position and trade management
- ✅ State machine infrastructure
- ✅ Strategy base class and decorator
- ✅ Backtesting engine
- ✅ Complete MACD Money Map example

Coming soon:
- More technical indicators
- Advanced order types
- Parameter optimization
- Performance analytics and reporting
- Visualization tools
- More example strategies

## Contributing

This project is in active development. Contributions are welcome!

## License

MIT License - See LICENSE file for details
