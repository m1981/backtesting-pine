"""
Synthetic Data Generators for Backtesting
"""

import numpy as np
import pandas as pd
from typing import Optional, List

# ═══════════════════════════════════════════════════════════════
# HELPER: REALISTIC OHLC CONSTRUCTOR
# ═══════════════════════════════════════════════════════════════

def generate_realistic_ohlc(
    closes: List[float],
    volatility_factor: float = 0.2, # Controls size of wicks/bodies
    start_date: str = '2023-01-01',
    freq: str = '1h',
    seed: int = 42
) -> pd.DataFrame:
    """
    Creates realistic OHLC candles from a backbone of Close prices.

    Instead of fixed percentages, this uses random noise to generate:
    1. Opens (near previous close)
    2. Highs (above max(open, close))
    3. Lows (below min(open, close))
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=len(closes), freq=freq)
    closes = np.array(closes)
    n = len(closes)

    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)

    # Initialize first open
    opens[0] = closes[0] * (1 + np.random.normal(0, 0.001))

    for i in range(n):
        # 1. DETERMINE OPEN
        if i > 0:
            # Open is usually the previous close, plus some noise (micro-gaps)
            gap_noise = np.random.normal(0, volatility_factor * 0.05)
            opens[i] = closes[i-1] + gap_noise

        # 2. DETERMINE BODY BOUNDARIES
        body_top = max(opens[i], closes[i])
        body_bottom = min(opens[i], closes[i])

        # 3. GENERATE WICKS (Shadows)
        # Wicks are absolute values added to top/bottom
        # We use absolute value of normal dist to ensure positive wick length
        upper_wick = abs(np.random.normal(0, volatility_factor * 0.15))
        lower_wick = abs(np.random.normal(0, volatility_factor * 0.15))

        # Occasional "Spikes" (Fat tails) - 5% chance of a larger wick
        if np.random.random() < 0.05:
            upper_wick *= 3
        if np.random.random() < 0.05:
            lower_wick *= 3

        highs[i] = body_top + upper_wick
        lows[i] = body_bottom - lower_wick

    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(1000, 5000, n)
    }, index=dates)

    return data


# ═══════════════════════════════════════════════════════════════
# SCENARIO: BULLISH DIVERGENCE (Long Setup)
# ═══════════════════════════════════════════════════════════════

def generate_bullish_divergence_scenario(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Recreates the Bullish Divergence pattern (The "W" bottom):
    - Price makes a Lower Low (Panic selling).
    - MACD makes a Higher Low (Momentum slowing).
    """
    # 1. The Crash (Strong Downtrend)
    # Fast drop creates a deep MACD Low
    phase1 = np.linspace(100, 90, 40)

    # 2. The Bounce (Relief Rally)
    # MACD resets towards zero
    phase2 = np.linspace(90, 94, 20)

    # 3. The Divergence Leg (Grinding Lower)
    # Price drops to a new low (88), but SLOWER than phase 1.
    # Slower velocity = MACD stays higher than the first low.
    phase3 = np.linspace(94, 88, 50)

    # 4. The Reversal (Rally)
    # Price explodes up
    phase4 = np.linspace(88, 105, 40)

    all_closes = np.concatenate([phase1, phase2, phase3, phase4])

    # Add slight noise to make lines organic
    np.random.seed(seed)
    noise = np.random.normal(0, 0.05, len(all_closes))
    all_closes += noise

    # Use realistic OHLC generator
    return generate_realistic_ohlc(all_closes, volatility_factor=0.25, seed=seed)

# ═══════════════════════════════════════════════════════════════
# SCENARIO: BEARISH DIVERGENCE (From your Image)
# ═══════════════════════════════════════════════════════════════

def generate_bearish_divergence_scenario(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Recreates the Bearish Divergence pattern:
    - Price makes a Higher High.
    - MACD makes a Lower High.
    """
    # 1. Setup (Uptrend)
    phase1 = np.linspace(100, 105, 40) # Fast move up (High MACD)

    # 2. Pullback
    phase2 = np.linspace(105, 103, 15)

    # 3. The Divergence Leg (Grinding Higher)
    # We move price higher (to 106), but SLOWER than phase 1.
    # Slower velocity = Lower MACD peak.
    phase3 = np.linspace(103, 106.5, 50)

    # 4. The Breakdown
    phase4 = np.linspace(106.5, 100, 30)

    all_closes = np.concatenate([phase1, phase2, phase3, phase4])

    # Add some noise to the closes so the line isn't perfectly straight
    np.random.seed(seed)
    noise = np.random.normal(0, 0.05, len(all_closes))
    all_closes += noise

    return generate_realistic_ohlc(all_closes, volatility_factor=0.3, seed=seed)

# ═══════════════════════════════════════════════════════════════
# SCENARIO: MACD MONEY MAP (Original Request)
# ═══════════════════════════════════════════════════════════════

def generate_macd_money_map_scenario(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Generates the specific MACD Money Map scenario (Bullish).
    """
    # 1. Phase 1: Establish Daily Bullish Bias
    phase1_len = 800
    phase1_prices = [100 + (i * 0.1) for i in range(phase1_len)]

    # 2. Phase 2: The 4H Setup
    phase2a_len = 30
    last_p = phase1_prices[-1]
    phase2a_prices = [last_p - (i * 0.25) for i in range(phase2a_len)]

    phase2b_len = 20
    last_p = phase2a_prices[-1]
    phase2b_prices = [last_p + (0.015 * i**2) for i in range(phase2b_len)]

    # 3. Phase 3: The 1H Trigger
    last_p = phase2b_prices[-1]
    phase3_prices = [last_p - 0.05] * 5 # Consolidation
    phase3_prices += [last_p + 0.2, last_p + 0.4, last_p + 0.6, last_p + 0.8, last_p + 1.0] # Pop

    # 4. Phase 4: Profit Run
    phase4_len = 50
    last_p = phase3_prices[-1]
    phase4_prices = [last_p + (i * 0.2) for i in range(phase4_len)]

    all_closes = phase1_prices + phase2a_prices + phase2b_prices + phase3_prices + phase4_prices

    # Use the new realistic generator
    # volatility_factor=0.2 keeps ATR reasonable but candles looking real
    return generate_realistic_ohlc(all_closes, volatility_factor=0.2, seed=seed)

# ═══════════════════════════════════════════════════════════════
# OTHER GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_trending_data(n_bars=1000, base_price=100.0, trend_strength=0.3, seed=42):
    np.random.seed(seed)
    trend = np.linspace(0, trend_strength, n_bars)
    noise = np.random.normal(0, 0.01, n_bars)
    closes = base_price * np.exp(trend + np.cumsum(noise))
    return generate_realistic_ohlc(closes, volatility_factor=0.3, seed=seed)

def generate_choppy_data(n_bars=1000, base_price=100.0, volatility=0.015, seed=42):
    np.random.seed(seed)
    noise = np.random.normal(0, volatility, n_bars)
    closes = base_price * (1 + noise)
    return generate_realistic_ohlc(closes, volatility_factor=0.4, seed=seed)

def generate_volatile_trending_data(n_bars=1000, base_price=100.0, trend_strength=0.4, volatility=0.025, seed=42):
    np.random.seed(seed)
    trend = np.linspace(0, trend_strength, n_bars)
    cycles = 0.05 * np.sin(np.linspace(0, 8 * np.pi, n_bars))
    noise = np.random.normal(0, volatility, n_bars)
    closes = base_price * np.exp(trend + cycles + np.cumsum(noise * 0.3))
    return generate_realistic_ohlc(closes, volatility_factor=0.5, seed=seed)

# Backward compatibility
def generate_from_prices(price_points, start_date='2023-01-01', freq='1h'):
    return generate_realistic_ohlc(price_points, volatility_factor=0.2, start_date=start_date, freq=freq)