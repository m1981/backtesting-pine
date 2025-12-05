//@version=5
indicator("3TF MACD: Advanced Patterns [Long Only]", overlay=true, shorttitle="3TF Advanced")

// ==========================================
// ─── INPUTS ───
// ==========================================
grp_macd = "MACD Settings"
fast_len = input.int(12, "Fast Length", group=grp_macd)
slow_len = input.int(26, "Slow Length", group=grp_macd)
sig_len  = input.int(9, "Signal Length", group=grp_macd)

grp_tf   = "Timeframe Settings"
tf_daily = input.timeframe("D", "Daily Bias Timeframe", group=grp_tf)
tf_4h    = input.timeframe("240", "4H Setup Timeframe", group=grp_tf)
tf_1h    = input.timeframe("60", "1H Trigger Timeframe", group=grp_tf)

grp_pat  = "Trigger Patterns (1H)"
use_flip   = input.bool(true, "Pattern 1: The Flip (Zero Cross)", group=grp_pat)
use_tower  = input.bool(false, "Pattern 2: Shrinking Tower (Aggressive)", group=grp_pat)
use_bounce = input.bool(false, "Pattern 3: Zero Bounce (Continuation)", group=grp_pat)

// ==========================================
// ─── CALCULATIONS ───
// ==========================================

// Function to get MACD data tuple
get_macd() =>
    [line, sig, hist] = ta.macd(close, fast_len, slow_len, sig_len)
    [line, sig, hist]

// 1. DAILY BIAS (System 3A)

[d_line_stream, d_sig_stream, d_hist_stream] = request.security(syminfo.tickerid, tf_daily, get_macd(), barmerge.gaps_off, barmerge.lookahead_on)


d_line = barstate.isrealtime ? d_line_stream[1] : d_line_stream
d_sig  = barstate.isrealtime ? d_sig_stream[1]  : d_sig_stream
d_hist = barstate.isrealtime ? d_hist_stream[1] : d_hist_stream

daily_bullish_bias = d_line > 0

// 2. 4H SETUP (System 1)
[h4_line_stream, h4_sig_stream, h4_hist_stream] = request.security(syminfo.tickerid, tf_4h, get_macd(), barmerge.gaps_off, barmerge.lookahead_on)

h4_line = barstate.isrealtime ? h4_line_stream[1] : h4_line_stream
h4_sig  = barstate.isrealtime ? h4_sig_stream[1]  : h4_sig_stream
h4_hist = barstate.isrealtime ? h4_hist_stream[1] : h4_hist_stream

h4_bullish_setup = h4_line > h4_sig + 0.5

// 3. 1H TRIGGER NUANCES (System 3B)
[h1_line, h1_sig, h1_hist] = request.security(syminfo.tickerid, tf_1h, get_macd(), barmerge.gaps_off, barmerge.lookahead_off)

// --- Pattern Logic ---

// Pattern 1: The Flip
pat_flip = ta.crossover(h1_hist, 0)

// Pattern 2: Shrinking Tower
pat_tower = (h1_hist < 0) and (h1_hist > h1_hist[1]) and (h1_hist[1] < h1_hist[2])

// Pattern 3: Zero Bounce
pat_bounce = (h1_hist > 0) and (h1_hist > h1_hist[1]) and (h1_hist[1] < h1_hist[2])

// ==========================================
// ─── STRATEGY EXECUTION ───
// ==========================================

signal_flip   = daily_bullish_bias and h4_bullish_setup and use_flip and pat_flip
signal_tower  = daily_bullish_bias and h4_bullish_setup and use_tower and pat_tower
signal_bounce = daily_bullish_bias and h4_bullish_setup and use_bounce and pat_bounce
any_signal    = signal_flip or signal_tower or signal_bounce

// ==========================================
// ─── VISUALS ───
// ==========================================

plotshape(signal_flip, title="Entry Flip", style=shape.arrowup, location=location.belowbar, color=color.green, text="FLIP", textcolor=color.rgb(80, 218, 0), size=size.normal)
plotshape(signal_tower, title="Entry Tower", style=shape.labelup, location=location.belowbar, color=color.lime, text="TWR", textcolor=color.black, size=size.small)
plotshape(signal_bounce, title="Entry Bounce", style=shape.labelup, location=location.belowbar, color=color.teal, text="BNC", textcolor=color.rgb(241, 56, 56), size=size.small)

// ==========================================
// ─── DEBUG DASHBOARD ───
// ==========================================

var table dashboard = table.new(position.top_right, 5, 5, border_width=1)

// Standard TradingView Colors
col_macd = #2962FF // TV Blue
col_sig  = #FF6D00 // TV Orange
col_pos  = #26A69A // TV Green
col_neg  = #EF5350 // TV Red

if barstate.islast
    // Header
    table.cell(dashboard, 0, 0, "3TF Monitor", bgcolor=color.rgb(247, 247, 247), text_color=color.rgb(0, 0, 0))
    table.cell(dashboard, 1, 0, "Status", bgcolor=color.rgb(247, 247, 247), text_color=color.rgb(0, 0, 0))
    table.cell(dashboard, 2, 0, "MACD", bgcolor=color.rgb(247, 247, 247), text_color=col_macd)
    table.cell(dashboard, 3, 0, "Signal", bgcolor=color.rgb(247, 247, 247), text_color=col_sig)
    table.cell(dashboard, 4, 0, "Hist", bgcolor=color.rgb(247, 247, 247), text_color=color.rgb(0, 0, 0))

    // --- ROW 1: DAILY ---
    table.cell(dashboard, 0, 1, "Daily", bgcolor=color.new(color.gray, 90), text_size=size.small)
    table.cell(dashboard, 1, 1, daily_bullish_bias ? "BULLISH" : "BEARISH", bgcolor=daily_bullish_bias ? color.green : color.red, text_color=color.white, text_size=size.small)

    // Raw Data Columns
    table.cell(dashboard, 2, 1, str.tostring(math.round(d_line, 4)), bgcolor=color.rgb(247, 247, 247), text_color=col_macd, text_size=size.small)
    table.cell(dashboard, 3, 1, str.tostring(math.round(d_sig, 4)), bgcolor=color.rgb(247, 247, 247), text_color=col_sig, text_size=size.small)
    table.cell(dashboard, 4, 1, str.tostring(math.round(d_hist, 4)), bgcolor=color.rgb(247, 247, 247), text_color=d_hist >= 0 ? col_pos : col_neg, text_size=size.small)

    // --- ROW 2: 4H ---
    table.cell(dashboard, 0, 2, "4H Setup", bgcolor=color.new(color.gray, 90), text_size=size.small)
    table.cell(dashboard, 1, 2, h4_bullish_setup ? "READY" : "WAITING", bgcolor=h4_bullish_setup ? color.green : color.orange, text_color=color.white, text_size=size.small)

    table.cell(dashboard, 2, 2, str.tostring(math.round(h4_line, 4)), bgcolor=color.rgb(247, 247, 247), text_color=col_macd, text_size=size.small)
    table.cell(dashboard, 3, 2, str.tostring(math.round(h4_sig, 4)), bgcolor=color.rgb(247, 247, 247), text_color=col_sig, text_size=size.small)
    table.cell(dashboard, 4, 2, str.tostring(math.round(h4_hist, 4)), bgcolor=color.rgb(247, 247, 247), text_color=h4_hist >= 0 ? col_pos : col_neg, text_size=size.small)

    // --- ROW 3: 1H ---
    table.cell(dashboard, 0, 3, "1H Trigger", bgcolor=color.new(color.gray, 90), text_size=size.small)

    color hist_col_bg = h1_hist >= 0 ? (h1_hist > h1_hist[1] ? color.lime : color.green) : (h1_hist < h1_hist[1] ? color.red : color.maroon)
    string hist_txt = h1_hist >= 0 ? "POS" : "NEG"
    table.cell(dashboard, 1, 3, hist_txt, bgcolor=hist_col_bg, text_color=color.white, text_size=size.small)

    table.cell(dashboard, 2, 3, str.tostring(math.round(h1_line, 4)), bgcolor=color.rgb(255, 255, 255), text_color=col_macd, text_size=size.small)
    table.cell(dashboard, 3, 3, str.tostring(math.round(h1_sig, 4)), bgcolor=color.rgb(255, 255, 255), text_color=col_sig, text_size=size.small)
    table.cell(dashboard, 4, 3, str.tostring(math.round(h1_hist, 4)), bgcolor=color.rgb(255, 255, 255), text_color=h1_hist >= 0 ? col_pos : col_neg, text_size=size.small)

// ==========================================
// ─── ALERTS ───
// ==========================================
alertcondition(any_signal, title="3TF Long Entry", message="3TF System: Long Entry Triggered on {{ticker}}")