# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**backtesting-pine** is a Python backtesting framework inspired by Pine Script, designed to implement multi-timeframe trading strategies with an intuitive, declarative API. The initial strategy is the "MACD Money Map" - a three-system approach combining trend following, reversal detection, and multi-timeframe confirmation.

## Project Structure

```
src/pine/
├── core/
│   ├── types.py       # Core type definitions (enums, dataclasses)
│   └── series.py      # Series class for time-series data access
└── __init__.py

tests/                 # Test suite (to be implemented)
examples/              # Example strategies (to be implemented)

# Documentation files
spec.md                # MACD Money Map strategy specification
api.md                 # Imaginary API design document
system-architecture.mermaid  # System architecture diagram
sequence-diagram.mermaid     # Execution sequence diagram
```

## Architecture Philosophy

The framework follows these core design principles:

1. **Declarative over Imperative**: Strategies describe *what* to compute, not *how*
2. **Timeframe-native**: Multi-timeframe operations are first-class citizens
3. **State-aware**: Built-in state machine support for complex strategies
4. **Readable**: Code should read like the trading strategy description
5. **Safe**: Prevents common backtesting mistakes (lookahead bias, etc.)

## Core Components

### Series Class (`src/pine/core/series.py`)

The `Series` class provides Pine Script-like access to time-series data:

- **Data access**: `.current`, `.previous`, `[n]` for accessing historical values
- **Aggregations**: `.highest(n)`, `.lowest(n)`, `.average(n)`, `.sum(n)`, `.std(n)`
- **Analysis**: `.slope(n)`, `.change(n)`, `.pct_change(n)`
- **Crossover detection**: `.crossed_above()`, `.crossed_below()`
- **Arithmetic**: Supports `+`, `-`, `*`, `/` operations
- **Comparisons**: Supports `>`, `<`, `>=`, `<=` operations

**Important**: Series data is stored with the most recent value at index 0. The internal `_index` tracks the current position during backtesting.

### Type System (`src/pine/core/types.py`)

Key enums and types:

- **Mode**: `LONG_ONLY`, `SHORT_ONLY`, `LONG_SHORT`
- **Bias**: `BULLISH`, `BEARISH`, `NEUTRAL`
- **State**: `IDLE`, `WAITING_CONFIRMATION`, `WAITING_TRIGGER`, `IN_POSITION`
- **SetupSignal**: `NONE`, `VALID`, `REJECTED`
- **ExitReason**: `STOP_LOSS`, `TAKE_PROFIT`, `TRAILING_SIGNAL`, `BIAS_CHANGED`, `MANUAL`, `TIMEOUT`
- **Timeframe**: Enum with values like `HOUR_1`, `HOUR_4`, `DAILY` (stores minutes as value)
  - Includes converters: `.from_string()`, `.to_pandas_freq()`, `.to_yfinance_interval()`

## The Three-System Strategy

The MACD Money Map strategy consists of three independent systems (see `spec.md` for full details):

### System 1: Trend System (4H timeframe)
- Detects crossovers far from zero line (> 0.5 × ATR)
- Filters out "chop zone" signals near zero
- Requires 2-3 bar confirmation wait period

### System 2: Reversal System (4H timeframe)
- Detects divergence between price and MACD
- Requires histogram flip confirmation
- *Note: Not yet implemented - complex divergence detection*

### System 3: Confirmation System (Multi-timeframe)
- **Part A**: Daily MACD determines bias (above zero = longs only, below zero = no trading)
- **Part B**: 1H histogram flip provides entry trigger

## Multi-Timeframe Design

The strategy operates on three timeframes:
- **Daily**: Establishes market bias (System 3A)
- **4-Hour**: Detects setups (System 1 & 2)
- **1-Hour**: Triggers entries (System 3B)

All processing happens bar-by-bar on the execution timeframe (1H), with higher timeframes accessed as needed. The framework handles timeframe alignment automatically.

## State Machine Flow

```
IDLE
  ↓ (setup detected on 4H + daily bias bullish)
WAITING_CONFIRMATION
  ↓ (2-3 bars elapsed, setup still valid)
WAITING_TRIGGER
  ↓ (1H histogram flips green)
IN_POSITION
  ↓ (exit condition met)
IDLE
```

## Development Status

**Current**: Early stage - core types and Series class implemented
**Next steps**: See `api.md` for the planned API design

Components to be built:
1. Strategy decorator and base class
2. Multi-timeframe data loader with auto-resampling
3. State machine infrastructure
4. Indicator calculation layer (MACD, ATR)
5. Backtesting engine with order management
6. Position and risk management
7. Performance analytics and reporting

## Key Design Decisions

### Data Limitations
- Using yfinance for data: hourly data limited to last 730 days
- 2H and 4H timeframes require resampling from 1H data (yfinance doesn't support them natively)

### Threshold Calculation
The "+0.5" threshold in the strategy will be ATR-based rather than fixed:
- Calculated as `threshold = atr_multiplier × ATR(14)`
- Default multiplier: 0.5 (configurable)
- Makes the strategy adaptable across different price ranges

### Long-Only Mode
When daily MACD < 0 in long-only mode: **no trading** (not shorting)

### Wait Mechanism
After detecting a crossover, the strategy waits 2-3 bars before looking for entry triggers. This filters ~50% of false signals.

## Code Style Guidelines

- Follow the declarative API design shown in `api.md`
- Use type hints for all function signatures
- Use dataclasses for configuration objects
- Use enums for state and signal types
- Keep systems independent and testable
- Avoid premature optimization - clarity over performance initially

## Testing Strategy

When implementing tests:
- Test Series operations independently
- Test each strategy system in isolation
- Test state machine transitions
- Test multi-timeframe alignment
- Use synthetic data for unit tests
- Use historical data for integration tests

## Common Pitfalls to Avoid

1. **Lookahead bias**: Never access future data during backtesting
2. **Repainting**: Ensure indicators are calculated on bar close
3. **Timeframe misalignment**: Respect bar boundaries across timeframes
4. **State leakage**: Ensure state machine resets properly between tests
5. **Overfitting**: Keep the strategy simple and parameter count low

## References

- **Strategy specification**: `spec.md` - Full MACD Money Map strategy details
- **API design**: `api.md` - Planned declarative API interface
- **Architecture diagram**: `system-architecture.mermaid` - Visual system overview
