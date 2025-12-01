# Imaginary Pine Script-like API Design for MACD Money Map Strategy

Let me think like the designer of Pine Script and imagine an ideal API that would make multi-timeframe strategies elegant and intuitive.

---

## Design Philosophy

Before diving into the API, let me establish the core principles:

| Principle | Description |
|-----------|-------------|
| **Declarative** | Describe *what* you want, not *how* to compute it |
| **Timeframe-native** | Multi-TF should feel first-class, not bolted on |
| **State-aware** | Built-in state machine support |
| **Readable** | Code reads like the trading strategy description |
| **Safe** | Prevent common backtesting mistakes (lookahead bias, etc.) |

---

## The Imaginary API

### 1. Strategy Declaration & Configuration

```python
@strategy(
    name="MACD Money Map",
    description="Multi-timeframe MACD system with trend + confirmation",
    mode=Mode.LONG_ONLY,
    initial_capital=10000,
    commission=0.002,
    slippage=0.001
)
class MACDMoneyMap:
    
    # Declarative parameters - auto-generates UI controls
    fast_ema: int = param(default=12, min=5, max=20, label="Fast EMA")
    slow_ema: int = param(default=26, min=20, max=50, label="Slow EMA")
    signal_period: int = param(default=9, min=5, max=15, label="Signal Period")
    atr_period: int = param(default=14, min=10, max=20, label="ATR Period")
    atr_threshold: float = param(default=0.5, min=0.1, max=1.0, label="ATR Threshold Mult")
    wait_bars: int = param(default=2, min=1, max=5, label="Confirmation Bars")
    risk_reward: float = param(default=2.0, min=1.0, max=5.0, label="Risk:Reward Ratio")
```

---

### 2. Multi-Timeframe Data Declaration

This is where the magic happens - **declarative timeframe requests**:

```python
class MACDMoneyMap:
    
    # Data requests - engine handles alignment automatically
    data = timeframes(
        tf_1h  = "1H",   # Entry timeframe (base)
        tf_4h  = "4H",   # Setup timeframe
        tf_daily = "1D"  # Bias timeframe
    )
    
    # Alternative syntax - more explicit
    @request_data
    def data_requirements(self):
        return {
            'execution': Timeframe.HOUR_1,    # Bar-by-bar processing
            'setup': Timeframe.HOUR_4,        # Checked on 1H bars, uses latest 4H
            'bias': Timeframe.DAILY           # Checked on 1H bars, uses latest Daily
        }
```

---

### 3. Indicator Declaration

Indicators are declared once, computed per-timeframe automatically:

```python
class MACDMoneyMap:
    
    # Indicators - automatically computed on each requested timeframe
    @indicators
    def setup_indicators(self):
        return {
            # Daily indicators (for bias)
            'macd_daily': MACD(
                source=self.data.daily.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),
            
            # 4-Hour indicators (for setup)
            'macd_4h': MACD(
                source=self.data.tf_4h.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),
            
            # 1-Hour indicators (for trigger)
            'macd_1h': MACD(
                source=self.data.tf_1h.close,
                fast=self.fast_ema,
                slow=self.slow_ema,
                signal=self.signal_period
            ),
            
            # ATR on execution timeframe
            'atr': ATR(
                high=self.data.tf_1h.high,
                low=self.data.tf_1h.low,
                close=self.data.tf_1h.close,
                period=self.atr_period
            )
        }
```

---

### 4. State Machine Declaration

Built-in state machine support - no manual tracking:

```python
class MACDMoneyMap:
    
    # Declare states - engine manages transitions
    @states
    def strategy_states(self):
        return StateGraph(
            initial=State.IDLE,
            states=[
                State.IDLE,
                State.WAITING_CONFIRMATION,
                State.WAITING_TRIGGER,
                State.IN_POSITION
            ],
            transitions=[
                Transition(State.IDLE, State.WAITING_CONFIRMATION, 
                          trigger="setup_detected"),
                Transition(State.WAITING_CONFIRMATION, State.WAITING_TRIGGER, 
                          trigger="wait_complete"),
                Transition(State.WAITING_CONFIRMATION, State.IDLE, 
                          trigger="setup_invalidated"),
                Transition(State.WAITING_TRIGGER, State.IN_POSITION, 
                          trigger="entry_triggered"),
                Transition(State.WAITING_TRIGGER, State.IDLE, 
                          trigger="trigger_timeout"),
                Transition(State.IN_POSITION, State.IDLE, 
                          trigger="position_closed"),
            ]
        )
```

---

### 5. System Definitions (The Three Systems)

Each system is a discrete, testable unit:

```python
class MACDMoneyMap:
    
    # ═══════════════════════════════════════════════════════════
    # SYSTEM 3 PART A: Daily Bias Filter
    # ═══════════════════════════════════════════════════════════
    
    @system(name="Daily Bias", timeframe="daily")
    def check_daily_bias(self) -> Bias:
        """
        The Zero Line is absolute law.
        Above zero = ONLY longs. Below zero = NO trading.
        """
        macd_value = self.indicators.macd_daily.macd_line.current
        
        if macd_value > 0:
            return Bias.BULLISH
        else:
            return Bias.BEARISH  # In long-only mode, this means NO TRADE
    
    
    # ═══════════════════════════════════════════════════════════
    # SYSTEM 1: Trend Setup Detection (4-Hour)
    # ═══════════════════════════════════════════════════════════
    
    @system(name="Trend Setup", timeframe="4h")
    def check_trend_setup(self) -> SetupSignal:
        """
        Crossover far from zero = valid trend setup.
        The distance rule filters out chop zone signals.
        """
        macd = self.indicators.macd_4h
        atr = self.indicators.atr.current
        
        # Check for bullish crossover
        crossover_detected = crossover(macd.macd_line, macd.signal_line)
        
        if not crossover_detected:
            return SetupSignal.NONE
        
        # The Distance Rule: must be far from zero
        distance_from_zero = abs(macd.macd_line.current)
        threshold = self.atr_threshold * atr
        
        if distance_from_zero < threshold:
            return SetupSignal.REJECTED  # Chop zone
        
        return SetupSignal.VALID
    
    
    # ═══════════════════════════════════════════════════════════
    # SYSTEM 3 PART B: Entry Trigger (1-Hour)
    # ═══════════════════════════════════════════════════════════
    
    @system(name="Entry Trigger", timeframe="1h")
    def check_entry_trigger(self) -> bool:
        """
        Histogram flip = momentum confirmation.
        First green bar after red = entry trigger.
        """
        histogram = self.indicators.macd_1h.histogram
        
        # The Flip: previous bar red, current bar green
        prev_red = histogram.previous < 0
        curr_green = histogram.current > 0
        
        return prev_red and curr_green
```

---

### 6. Entry Logic with Built-in Wait Mechanism

```python
class MACDMoneyMap:
    
    @on_bar(timeframe="1h")
    def process_bar(self, bar: Bar):
        """Main processing loop - called on each 1H bar."""
        
        # ─────────────────────────────────────────────────────
        # STATE: IDLE - Looking for setups
        # ─────────────────────────────────────────────────────
        
        if self.state == State.IDLE:
            
            # System 3A: Check daily bias first
            bias = self.check_daily_bias()
            
            if bias == Bias.BEARISH:
                return  # No trading allowed today
            
            # System 1: Check for trend setup
            setup = self.check_trend_setup()
            
            if setup == SetupSignal.VALID:
                self.transition_to(
                    State.WAITING_CONFIRMATION,
                    trigger="setup_detected",
                    metadata={'crossover_bar': bar.index}
                )
        
        # ─────────────────────────────────────────────────────
        # STATE: WAITING_CONFIRMATION - Patience filter
        # ─────────────────────────────────────────────────────
        
        elif self.state == State.WAITING_CONFIRMATION:
            
            bars_waited = self.bars_in_state()
            
            # Check if setup is still valid (not reversed)
            if self.setup_invalidated():
                self.transition_to(State.IDLE, trigger="setup_invalidated")
                return
            
            # Wait 2-3 bars before looking for trigger
            if bars_waited >= self.wait_bars:
                self.transition_to(
                    State.WAITING_TRIGGER,
                    trigger="wait_complete"
                )
        
        # ─────────────────────────────────────────────────────
        # STATE: WAITING_TRIGGER - Looking for histogram flip
        # ─────────────────────────────────────────────────────
        
        elif self.state == State.WAITING_TRIGGER:
            
            # Timeout after 5 bars
            if self.bars_in_state() > 5:
                self.transition_to(State.IDLE, trigger="trigger_timeout")
                return
            
            # System 3B: Check for entry trigger
            if self.check_entry_trigger():
                self.execute_entry(bar)
                self.transition_to(State.IN_POSITION, trigger="entry_triggered")
        
        # ─────────────────────────────────────────────────────
        # STATE: IN_POSITION - Managing the trade
        # ─────────────────────────────────────────────────────
        
        elif self.state == State.IN_POSITION:
            self.manage_position(bar)
```

---

### 7. Entry Execution with Risk Management

```python
class MACDMoneyMap:
    
    def execute_entry(self, bar: Bar):
        """Execute long entry with calculated stops and targets."""
        
        entry_price = bar.close
        atr = self.indicators.atr.current
        
        # Stop Loss: tighter of swing low or ATR-based
        swing_low = self.data.tf_1h.low.lowest(20)
        atr_stop = entry_price - (2 * atr)
        stop_loss = max(swing_low, atr_stop)  # Use the tighter stop
        
        # Risk calculation
        risk = entry_price - stop_loss
        
        # Take Profit: 2R target
        take_profit = entry_price + (self.risk_reward * risk)
        
        # Position sizing (risk 2% of capital per trade)
        risk_amount = self.capital * 0.02
        quantity = risk_amount / risk
        
        # Execute with built-in order management
        self.buy(
            quantity=quantity,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="MACD Money Map Entry",
            metadata={
                'daily_macd': self.indicators.macd_daily.macd_line.current,
                'setup_distance': self.last_setup_distance,
                'atr_at_entry': atr
            }
        )
```

---

### 8. Position Management with Multiple Exit Conditions

```python
class MACDMoneyMap:
    
    def manage_position(self, bar: Bar):
        """Manage open position - check all exit conditions."""
        
        position = self.current_position
        
        # Exit Priority 1: Stop Loss (handled automatically by engine)
        # Exit Priority 2: Take Profit (handled automatically by engine)
        
        # Exit Priority 3: Trailing signal (opposite crossover)
        if self.check_trailing_exit():
            self.close_position(
                reason=ExitReason.TRAILING_SIGNAL,
                comment="4H Bearish Crossover"
            )
            self.transition_to(State.IDLE, trigger="position_closed")
            return
        
        # Exit Priority 4: Bias change (daily MACD flips)
        if self.check_bias_exit():
            self.close_position(
                reason=ExitReason.BIAS_CHANGED,
                comment="Daily MACD crossed below zero"
            )
            self.transition_to(State.IDLE, trigger="position_closed")
            return
    
    
    def check_trailing_exit(self) -> bool:
        """4H bearish crossover = exit signal."""
        macd = self.indicators.macd_4h
        return crossunder(macd.macd_line, macd.signal_line)
    
    
    def check_bias_exit(self) -> bool:
        """Daily MACD crossing below zero = exit."""
        macd_daily = self.indicators.macd_daily.macd_line
        return crossunder(macd_daily, 0)
```

---

### 9. Partial Position Management (Bonus Feature)

```python
class MACDMoneyMap:
    
    @on_partial_target(portion=0.5)
    def on_first_target(self, position: Position):
        """Called when price hits take profit - close half."""
        
        # Close 50% of position
        self.close_partial(
            position=position,
            portion=0.5,
            reason="First target hit (1R)"
        )
        
        # Move stop to breakeven on remainder
        self.modify_stop(
            position=position,
            new_stop=position.entry_price,
            comment="Stop moved to breakeven"
        )
        
        # Optionally set new target for remainder
        self.modify_target(
            position=position,
            new_target=None,  # No fixed target - trail with signal
            comment="Trailing remainder"
        )
```

---

### 10. Built-in Helper Functions

The API would include intuitive helper functions:

```python
# ═══════════════════════════════════════════════════════════════
# CROSSOVER DETECTION
# ═══════════════════════════════════════════════════════════════

crossover(series_a, series_b)      # A crosses above B
crossunder(series_a, series_b)     # A crosses below B
crossed(series_a, series_b)        # Either direction

# ═══════════════════════════════════════════════════════════════
# SERIES ACCESSORS
# ═══════════════════════════════════════════════════════════════

series.current          # Current bar value (index 0)
series.previous         # Previous bar value (index 1)
series[n]               # N bars ago
series.highest(n)       # Highest value in last N bars
series.lowest(n)        # Lowest value in last N bars
series.average(n)       # Average of last N bars
series.slope(n)         # Rate of change over N bars

# ═══════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════

self.bars_in_state()              # Bars since state transition
self.bars_since_entry()           # Bars since position opened
self.bars_since(condition)        # Bars since condition was true

# ═══════════════════════════════════════════════════════════════
# TIMEFRAME HELPERS
# ═══════════════════════════════════════════════════════════════

self.is_new_bar(timeframe)        # True on first bar of new TF period
self.timeframe_changed('4h')      # True when 4H bar just closed

# ═══════════════════════════════════════════════════════════════
# POSITION HELPERS
# ═══════════════════════════════════════════════════════════════

self.has_position                 # True if in trade
self.current_position             # Position object or None
self.unrealized_pnl               # Current P&L
self.unrealized_pnl_percent       # Current P&L as %
```

---

### 11. Complete Strategy in One View

Here's how the complete strategy would look with this API:

```python
@strategy(
    name="MACD Money Map",
    mode=Mode.LONG_ONLY,
    initial_capital=10000,
    commission=0.002
)
class MACDMoneyMap:
    
    # ══════════════════════════════════════════════════════════
    # PARAMETERS
    # ══════════════════════════════════════════════════════════
    
    fast_ema: int = param(12, min=5, max=20)
    slow_ema: int = param(26, min=20, max=50)
    signal_period: int = param(9, min=5, max=15)
    atr_period: int = param(14, min=10, max=20)
    atr_threshold: float = param(0.5, min=0.1, max=1.0)
    wait_bars: int = param(2, min=1, max=5)
    risk_reward: float = param(2.0, min=1.0, max=5.0)
    
    # ══════════════════════════════════════════════════════════
    # TIMEFRAMES
    # ══════════════════════════════════════════════════════════
    
    data = timeframes(
        tf_1h="1H",      # Execution
        tf_4h="4H",      # Setup  
        tf_daily="1D"    # Bias
    )
    
    # ══════════════════════════════════════════════════════════
    # INDICATORS
    # ══════════════════════════════════════════════════════════
    
    @indicators
    def setup(self):
        return {
            'macd_daily': MACD(self.data.tf_daily.close, self.fast_ema, self.slow_ema, self.signal_period),
            'macd_4h': MACD(self.data.tf_4h.close, self.fast_ema, self.slow_ema, self.signal_period),
            'macd_1h': MACD(self.data.tf_1h.close, self.fast_ema, self.slow_ema, self.signal_period),
            'atr': ATR(self.data.tf_1h, self.atr_period)
        }
    
    # ══════════════════════════════════════════════════════════
    # STATES
    # ══════════════════════════════════════════════════════════
    
    states = [State.IDLE, State.WAITING_CONFIRM, State.WAITING_TRIGGER, State.IN_POSITION]
    
    # ══════════════════════════════════════════════════════════
    # SYSTEM 3A: DAILY BIAS
    # ══════════════════════════════════════════════════════════
    
    @condition
    def is_bullish_bias(self) -> bool:
        return self.indicators.macd_daily.macd_line.current > 0
    
    # ══════════════════════════════════════════════════════════
    # SYSTEM 1: TREND SETUP (4H)
    # ══════════════════════════════════════════════════════════
    
    @condition
    def has_valid_setup(self) -> bool:
        macd = self.indicators.macd_4h
        atr = self.indicators.atr.current
        
        has_crossover = crossover(macd.macd_line, macd.signal_line)
        far_from_zero = abs(macd.macd_line.current) > (self.atr_threshold * atr)
        
        return has_crossover and far_from_zero
    
    # ══════════════════════════════════════════════════════════
    # SYSTEM 3B: ENTRY TRIGGER (1H)
    # ══════════════════════════════════════════════════════════
    
    @condition
    def has_entry_trigger(self) -> bool:
        hist = self.indicators.macd_1h.histogram
        return hist.previous < 0 and hist.current > 0
    
    # ══════════════════════════════════════════════════════════
    # MAIN LOGIC
    # ══════════════════════════════════════════════════════════
    
    @on_bar
    def process(self, bar):
        
        match self.state:
            
            case State.IDLE:
                if self.is_bullish_bias() and self.has_valid_setup():
                    self.goto(State.WAITING_CONFIRM)
            
            case State.WAITING_CONFIRM:
                if self.bars_in_state() >= self.wait_bars:
                    self.goto(State.WAITING_TRIGGER)
                elif self.setup_invalidated():
                    self.goto(State.IDLE)
            
            case State.WAITING_TRIGGER:
                if self.has_entry_trigger():
                    self.enter_long(bar)
                    self.goto(State.IN_POSITION)
                elif self.bars_in_state() > 5:
                    self.goto(State.IDLE)
            
            case State.IN_POSITION:
                if self.should_exit():
                    self.close_position()
                    self.goto(State.IDLE)
    
    # ══════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════
    
    def enter_long(self, bar):
        atr = self.indicators.atr.current
        stop = max(self.data.tf_1h.low.lowest(20), bar.close - 2*atr)
        risk = bar.close - stop
        target = bar.close + (self.risk_reward * risk)
        
        self.buy(stop_loss=stop, take_profit=target)
    
    # ══════════════════════════════════════════════════════════
    # EXIT CONDITIONS
    # ══════════════════════════════════════════════════════════
    
    def should_exit(self) -> bool:
        # Trailing exit: 4H bearish crossover
        macd_4h = self.indicators.macd_4h
        if crossunder(macd_4h.macd_line, macd_4h.signal_line):
            return True
        
        # Bias exit: Daily MACD flips bearish
        if self.indicators.macd_daily.macd_line.current < 0:
            return True
        
        return False
```

---

## Summary: Why This API Design Works

| Feature | Benefit |
|---------|---------|
| **Declarative timeframes** | No manual resampling/alignment |
| **Built-in state machine** | Clean, debuggable state transitions |
| **Decorator-based systems** | Each system is isolated and testable |
| **Intuitive helpers** | `crossover()`, `series.lowest()`, etc. |
| **Match/case for states** | Python 3.10+ pattern matching |
| **Automatic indicators** | Computed once, available everywhere |
| **Partial closes** | First-class support |

---

## Next Step

**Shall I now implement this API design as a real Python library?**

I would build:
1. The `@strategy` decorator and base class
2. The `timeframes()` helper with auto-resampling
3. The state machine infrastructure
4. The indicator calculation layer
5. The backtesting engine that drives it all

This would be approximately 600-800 lines of clean, documented Python code that you can use for this strategy and extend for future strategies.