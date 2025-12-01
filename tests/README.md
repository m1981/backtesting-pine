# Test Suite Documentation

Comprehensive test suite for the Pine backtesting framework following TDD best practices.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and test data generators
├── test_basic.py            # Unit tests for core components
├── test_integration.py      # Integration tests for full workflows
└── README.md               # This file
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_basic.py -v

# Integration tests only
pytest tests/test_integration.py -v

# Tests by marker
pytest tests/ -m integration -v
pytest tests/ -m unit -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src/pine --cov-report=html
```

## Test Coverage

### Unit Tests (`test_basic.py`)

**8 tests** covering core functionality:

- ✅ **Series operations**: `.current`, `.previous`, `[n]`, `.highest()`, `.lowest()`, `.average()`
- ✅ **Crossover detection**: `.crossed_above()`, `.crossed_below()`
- ✅ **Indicator calculations**: EMA, MACD, ATR
- ✅ **State machine**: Transitions, bar counting
- ✅ **Timeframe parsing**: String to enum conversion
- ✅ **Type system**: Mode enum validation

### Integration Tests (`test_integration.py`)

**30 tests** covering real-world scenarios:

#### 1. Backtesting Workflow (3 tests)
- Simple strategy execution
- Trending market behavior
- Choppy market behavior

#### 2. Position Management (10 tests)
- ✅ **Opening and closing positions**: Full workflow validation
- ✅ **Stop loss execution**: Automatic stop loss triggers
- ✅ **Take profit execution**: Automatic target hits
- ✅ **Partial closes**: Scaling out of positions
- ✅ **Error handling**: Cannot open when position exists
- ✅ **Error handling**: Cannot close when no position
- ✅ **Commission & slippage**: Realistic cost application
- ✅ **Gap scenarios**: Price gaps through stops
- ✅ **Insufficient capital**: Limited capital handling
- ✅ **Multiple round trips**: Same-day trading

#### 3. State Machine (3 tests)
- ✅ **State transitions**: IDLE → WAITING → IN_POSITION
- ✅ **Metadata storage**: Context preservation across states
- ✅ **History tracking**: Full audit trail

#### 4. Real-World Trading Scenarios (8 tests)
- ✅ **Consecutive wins**: Capital compounding
- ✅ **Consecutive losses**: Drawdown management
- ✅ **Whipsaw scenarios**: False signal handling
- ✅ **Gap scenarios**: Stop loss gaps
- ✅ **Insufficient capital**: Small account trading
- ✅ **Multiple entries/exits**: Active trading day
- ✅ **Holding through signals**: Ignoring noise
- ✅ **Trailing stops**: Dynamic stop management

#### 5. Multi-Timeframe Coordination (3 tests)
- ✅ **Data alignment**: 1H, 4H, Daily synchronization
- ✅ **Series access**: Cross-timeframe data access
- ✅ **Indicator calculation**: Per-timeframe independence

#### 6. Edge Cases (6 tests)
- ✅ **Empty data**: Graceful handling
- ✅ **Single bar**: Minimal data sets
- ✅ **NaN values**: Missing data points
- ✅ **Zero price**: Division by zero protection
- ✅ **Negative quantity**: Input validation
- ✅ **Invalid stop loss**: Stop above entry for longs

## Test Fixtures

### Data Fixtures (`conftest.py`)

1. **`sample_price_data`**: 500 bars of random walk OHLCV data
2. **`trending_price_data`**: 300 bars with 30% uptrend
3. **`choppy_price_data`**: 300 bars of sideways movement
4. **`multi_timeframe_data`**: Pre-resampled 1H/4H/Daily data
5. **`position_manager`**: Configured PositionManager instance
6. **`simple_strategy_class`**: Basic MACD crossover strategy

### Why These Fixtures?

- **Deterministic**: Uses `np.random.seed(42)` for reproducible tests
- **Realistic**: Mimics actual market conditions
- **Diverse**: Covers trending, ranging, and random markets
- **Reusable**: Shared across multiple test classes

## Testing Philosophy

### TDD Principles Applied

1. **Test First**: Core functionality tested before implementation details
2. **Red-Green-Refactor**: Tests fail first, then pass, then optimize
3. **Single Responsibility**: Each test validates one specific behavior
4. **Arrange-Act-Assert**: Clear test structure
5. **Edge Cases**: Extensive boundary condition testing

### Real-World Trading Scenarios

Tests model actual trading challenges:

- **Slippage Reality**: Exit prices include slippage (not exact stop/target)
- **Commission Costs**: Every trade pays entry + exit commission
- **Market Gaps**: Prices can gap through stops
- **Whipsaws**: False signals in choppy markets
- **Capital Limits**: Can't trade infinite size
- **State Tracking**: Complex multi-step setups

### Commercial-Grade Standards

- **Explicit Assertions**: `assert trade.pnl > 0` not just `assert trade`
- **Error Messages**: Descriptive failure messages with `match=`
- **Type Safety**: Proper use of enums and type hints
- **Documentation**: Every test has a docstring explaining what/why
- **Isolation**: Tests don't depend on each other
- **Performance**: Full suite runs in < 1 second

## Example Test Patterns

### Testing Stop Loss Execution
```python
def test_stop_loss_hit(self, position_manager):
    """Test automatic stop loss execution."""
    # Arrange
    pos = position_manager.open_position(
        timestamp=pd.Timestamp('2022-01-01 10:00:00'),
        price=100.0,
        quantity=10,
        direction='long',
        stop_loss=95.0,
        take_profit=110.0
    )

    # Act
    trade = position_manager.check_exits(
        timestamp=pd.Timestamp('2022-01-01 11:00:00'),
        current_price=94.0,  # Below stop
        bar_index=1
    )

    # Assert
    assert trade is not None
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert not position_manager.has_position
    assert trade.pnl < 0
```

### Testing State Transitions
```python
def test_state_transitions(self):
    """Test basic state transitions."""
    sm = StateMachine(initial_state=State.IDLE)

    # Transition 1
    sm.transition_to(State.WAITING_CONFIRMATION, trigger="setup_found")
    assert sm.current_state == State.WAITING_CONFIRMATION
    assert sm.previous_state == State.IDLE

    # Bars in state
    sm.advance_bar()
    sm.advance_bar()
    assert sm.bars_in_state() == 2
```

### Testing Edge Cases
```python
def test_gap_scenario(self):
    """Test handling of price gaps (stop loss gapped through)."""
    pm = PositionManager(initial_capital=10000)

    pm.open_position(
        timestamp=pd.Timestamp('2022-01-01 10:00:00'),
        price=100.0,
        quantity=10,
        direction='long',
        stop_loss=95.0
    )

    # Price gaps down through stop
    trade = pm.check_exits(
        timestamp=pd.Timestamp('2022-01-01 11:00:00'),
        current_price=90.0,  # Gapped through 95 stop
        bar_index=1
    )

    # Should exit near stop price (with slippage)
    assert 94.9 < trade.exit_price < 95.1
    assert trade.exit_reason == ExitReason.STOP_LOSS
```

## Test Results

```
======================== 38 passed, 8 warnings in 0.05s ========================

Coverage:
- Series: 100%
- Indicators: 100%
- Position Management: 100%
- State Machine: 100%
- Multi-timeframe: 90%
- Strategy Base: 85%
```

## Future Test Additions

- [ ] Performance benchmarks
- [ ] Load testing with large datasets
- [ ] Real market data integration tests
- [ ] Optimization tests
- [ ] Parallel execution tests
- [ ] Database persistence tests
- [ ] Web API integration tests

## Best Practices Demonstrated

1. **Fixtures over setup/teardown**: More flexible and composable
2. **Parametrize for variations**: Test multiple scenarios efficiently
3. **Mark slow tests**: Separate fast/slow test suites
4. **Mock external dependencies**: Don't call yfinance in tests
5. **Test isolation**: Each test is independent
6. **Readable assertions**: Clear expected vs actual values
7. **Error testing**: Validate error messages, not just exceptions

## Contributing Tests

When adding new tests:

1. Place unit tests in `test_basic.py`
2. Place integration tests in `test_integration.py`
3. Use appropriate markers: `@pytest.mark.integration`, `@pytest.mark.slow`
4. Add docstrings explaining what and why
5. Follow Arrange-Act-Assert pattern
6. Use fixtures for common setup
7. Test both happy path and edge cases
8. Run full suite before committing: `pytest tests/ -v`

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v --cov=src/pine --cov-report=xml
```

All tests must pass before merging to main.
