import numpy as np
from scipy.stats import linregress
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]

# Momentum windows
MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 10
SHORT_WINDOW = 8
LONG_WINDOW = 20

# EMA parameters
EMA_FAST = 3
EMA_SLOW = 21
EMA_SLOPE_PERIOD = 22
EMA_SLOPE_LOOKBACK = 3

# MACD parameters
MACD_FAST = 8
MACD_SLOW = 15
MACD_SIGNAL = 4

# Linear regression
LINREG_PERIOD = 16

# Volatility parameters
VOL_LOOKBACK = 24
VOL_SHORT_LOOKBACK = 12
VOL_LONG_LOOKBACK = 36
TARGET_VOL = 0.015

# Entry threshold
BASE_THRESHOLD = 0.005
DYN_THRESHOLD_FLOOR = 0.0045
DYN_THRESHOLD_CEIL = 0.012
TREND_THRESHOLD_SCALE = 0.32       # max threshold reduction in trends
TREND_THRESHOLD_DECAY = 0.14       # abs(ret_long) at which reduction saturates

# RSI voter
RSI_TREND_BIAS = 2.0
RSI_TREND_BIAS_DECAY = 0.10

# RSI exit parameters
RSI_OVERBOUGHT = 73
RSI_OVERSOLD = 27
RSI_OB_TIGHT = 65
RSI_OS_TIGHT = 35
RSI_EXIT_VOL_LOW = 0.7
RSI_EXIT_VOL_HIGH = 1.8
RSI_EXIT_TREND_DECAY = 0.08
RSI_EXIT_PROFIT_THRESHOLD = 0.007
RSI_EXIT_PROFIT_TIGHTEN = 0.15
RSI_EXIT_PROFIT_SCALE = 20.0
RSI_YOUNG_GRACE_BARS = 5
RSI_YOUNG_OB_WIDEN = 4.0
RSI_YOUNG_OS_WIDEN = 4.0

# Peak-profit trailing exit
PEAK_PROFIT_MIN_BASE = 0.025
PEAK_PROFIT_GIVEBACK = 0.25

# Sizing multipliers
BASE_POSITION_SIZE = 0.115
CALM_BOOST_MAX = 0.8
SIDEWAYS_BOOST_MAX = 0.70
CROSS_ASSET_FIXED_BOOST = 0.15
HIGH_VOTE_BOOST_MULT = 1.20
VOL_CONFIRM_LOOKBACK = 12
VOL_CONFIRM_BASE = 24
VOL_CONFIRM_FLOOR = 0.98
VOL_CONFIRM_CAP = 1.10
STRENGTH_FLOOR_SIDEWAYS = 2.6
STRENGTH_FLOOR_DECAY = 0.12

# Combined mult cap
MAX_COMBINED_MULT_HIGH_VOL = 2.5
MAX_COMBINED_MULT_LOW_VOL = 6.5
MAX_COMBINED_VOL_LOW = 0.6
MAX_COMBINED_VOL_HIGH = 1.2
MAX_COMBINED_TREND_BOOST = 1.0

# Trend gate
TREND_GATE_MED_WEIGHT_SIDEWAYS = 0.85
TREND_GATE_MED_WEIGHT_BASE = 0.70
TREND_GATE_DEADZONE = 0.010
MEANREV_TREND_THRESHOLD = 0.05
MEANREV_RSI_OVERSOLD = 49
MEANREV_RSI_OVERBOUGHT = 51

# Vote / cooldown
VOL_BREAKOUT_SHORT = 3
DONCHIAN_PERIOD = 12
MIN_VOTES = 3
FLIP_MIN_VOTES = 4
COOLDOWN_BARS = 3
COOLDOWN_TREND_DECAY = 0.06


def ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

def calc_rsi(closes, period):
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / max(avg_loss, 1e-10)
    return 100 - 100 / (1 + rs)


class Strategy:
    def __init__(self):
        self.entry_prices = {}
        self.exit_bar = {}
        self.bar_count = 0
        self.peak_pnl = {}
        self.entry_bar = {}

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close

            realized_vol = max(np.std(np.diff(np.log(closes[-VOL_LOOKBACK:]))), 1e-6)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.10 + vol_ratio * 0.90) ** 0.85
            dyn_threshold = max(DYN_THRESHOLD_FLOOR, min(DYN_THRESHOLD_CEIL, dyn_threshold))

            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            dyn_threshold *= 1.0 - TREND_THRESHOLD_SCALE * (1.0 - min(abs(ret_long) / TREND_THRESHOLD_DECAY, 1.0) ** 0.85)

            short_vol = max(np.std(np.diff(np.log(closes[-VOL_SHORT_LOOKBACK:]))), 1e-6)
            long_vol = max(np.std(np.diff(np.log(closes[-VOL_LONG_LOOKBACK:]))), 1e-6)
            sl_ratio_raw = short_vol / max(long_vol, 1e-10)

            _lr = linregress(np.arange(LINREG_PERIOD), np.log(closes[-LINREG_PERIOD:]))


            adaptive_med = int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))
            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, adaptive_med))

            ret_vshort = (closes[-1] - closes[-SHORT_WINDOW]) / closes[-SHORT_WINDOW]
            ret_short = (closes[-1] - closes[-adaptive_med]) / closes[-adaptive_med]
            ret_med = (closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]

            mom_bull = ret_short > dyn_threshold
            mom_bear = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.5
            vshort_bear = ret_vshort < -dyn_threshold * 0.5

            ema_fast_arr = ema(closes[-(EMA_SLOW+10):], EMA_FAST)
            ema_slow_arr = ema(closes[-(EMA_SLOW+10):], EMA_SLOW)
            ema_bull = ema_fast_arr[-1] > ema_slow_arr[-1]
            ema_bear = ema_fast_arr[-1] < ema_slow_arr[-1]

            rsi_trend_str = min(abs(ret_long) / RSI_TREND_BIAS_DECAY, 1.0)
            adaptive_rsi_period = int(round(6 + 2 * rsi_trend_str))
            rsi = calc_rsi(closes, adaptive_rsi_period)
            rsi_bias = RSI_TREND_BIAS * rsi_trend_str
            rsi_thresh = 50 + (-rsi_bias if ret_long > 0 else rsi_bias)
            rsi_bull = rsi > rsi_thresh
            rsi_bear = rsi < rsi_thresh

            _macd_fast = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST)
            _macd_slow = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
            _macd_line = _macd_fast - _macd_slow
            _signal_line = ema(_macd_line, MACD_SIGNAL)
            macd_rel = (_macd_line[-1] - _signal_line[-1]) / mid
            macd_bull = macd_rel > 0.0003
            macd_bear = macd_rel < -0.0003

            ema_slope_arr = ema(closes[-(EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5):], EMA_SLOPE_PERIOD)
            ema_slope = (ema_slope_arr[-1] - ema_slope_arr[-EMA_SLOPE_LOOKBACK]) / ema_slope_arr[-EMA_SLOPE_LOOKBACK]
            slope_bull = ema_slope > 0.0005
            slope_bear = ema_slope < -0.0005

            linreg_bull = _lr.slope > 0.0001
            linreg_bear = _lr.slope < -0.0001

            vb_short = max(np.std(np.diff(np.log(closes[-VOL_BREAKOUT_SHORT:]))), 1e-6)
            vol_breakout_bull = vb_short > realized_vol and ret_vshort > dyn_threshold * 0.20
            vol_breakout_bear = vb_short > realized_vol and ret_vshort < -dyn_threshold * 0.20

            donchian_high = np.max(closes[-(DONCHIAN_PERIOD+1):-1])
            donchian_low = np.min(closes[-(DONCHIAN_PERIOD+1):-1])
            donchian_bull = mid > donchian_high * 1.003
            donchian_bear = mid < donchian_low * 0.997

            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, vol_breakout_bull, linreg_bull, donchian_bull, slope_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, vol_breakout_bear, linreg_bear, donchian_bear, slope_bear])

            cooldown_trend_strength = min(abs(ret_long) / COOLDOWN_TREND_DECAY, 1.0)
            trend_avg = (TREND_GATE_MED_WEIGHT_SIDEWAYS - (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength ** 0.85) * ret_med + ((1.0 - TREND_GATE_MED_WEIGHT_SIDEWAYS) + (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength ** 0.85) * ret_long
            trend_bull = trend_avg > 0
            trend_bear = trend_avg < 0

            trend_gate_bypassed = abs(trend_avg) < TREND_GATE_DEADZONE
            bullish = bull_votes >= MIN_VOTES and (trend_bull or (trend_gate_bypassed and bull_votes > bear_votes))
            bearish = bear_votes >= MIN_VOTES and (trend_bear or (trend_gate_bypassed and bear_votes > bull_votes))

            effective_cooldown = COOLDOWN_BARS * cooldown_trend_strength
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < effective_cooldown

            vol_scale = (TARGET_VOL / realized_vol) ** 0.85
            vol_scale = max(0.3, min(2.5, vol_scale))

            vol_ratio_sl = max(0.5, min(2.0, sl_ratio_raw))
            calm_vol_gate = min(1.0, max(0.0, (1.7 - vol_ratio) / 0.4))
            calm_boost = 1.0 + CALM_BOOST_MAX * max(0.0, 1.0 - vol_ratio_sl) ** 0.85 * calm_vol_gate

            sideways_boost = 1.0 + SIDEWAYS_BOOST_MAX * (1.0 - rsi_trend_str ** 1.45)

            vote_boost = HIGH_VOTE_BOOST_MULT

            volumes = bd.history["volume"].values
            recent_vol = np.mean(volumes[-VOL_CONFIRM_LOOKBACK:])
            base_vol = np.mean(volumes[-VOL_CONFIRM_BASE:])
            vol_confirm_mult = max(VOL_CONFIRM_FLOOR, min(VOL_CONFIRM_CAP, recent_vol / base_vol))

            mom_strength = (abs(ret_short) / dyn_threshold) ** 0.85
            sideways_strength = min(abs(ret_long) / STRENGTH_FLOOR_DECAY, 1.0)
            strength_floor = 0.6 + (STRENGTH_FLOOR_SIDEWAYS - 0.6) * (1.0 - sideways_strength)
            strength_scale = max(strength_floor, min(2.0, mom_strength))
            dampened_cross_agree = 1.0 + CROSS_ASSET_FIXED_BOOST * (1.0 - cooldown_trend_strength)
            combined_mult = vol_scale * strength_scale * calm_boost * sideways_boost * dampened_cross_agree * vote_boost * vol_confirm_mult
            adaptive_cap = MAX_COMBINED_MULT_HIGH_VOL if vol_ratio > MAX_COMBINED_VOL_HIGH else MAX_COMBINED_MULT_LOW_VOL - 3.0 * max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_LOW) / (MAX_COMBINED_VOL_HIGH - MAX_COMBINED_VOL_LOW)))
            adaptive_cap += MAX_COMBINED_TREND_BOOST * (1.0 - rsi_trend_str ** 0.85)
            combined_mult = min(combined_mult, adaptive_cap)
            size = equity * BASE_POSITION_SIZE * combined_mult

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                if not in_cooldown:
                    if bullish:
                        target = size
                    elif bearish:
                        target = -size
                    elif abs(ret_long) < MEANREV_TREND_THRESHOLD:
                        if rsi < MEANREV_RSI_OVERSOLD:
                            target = size
                        elif rsi > MEANREV_RSI_OVERBOUGHT:
                            target = -size
            else:
                vol_exit_blend = max(0.0, min(1.0, (vol_ratio - RSI_EXIT_VOL_LOW) / (RSI_EXIT_VOL_HIGH - RSI_EXIT_VOL_LOW)))
                sideways_exit_widen = max(0.0, 1.0 - abs(ret_long) / RSI_EXIT_TREND_DECAY)
                base_ob = RSI_OVERBOUGHT + sideways_exit_widen
                base_os = RSI_OVERSOLD + sideways_exit_widen
                effective_ob = base_ob - (base_ob - RSI_OB_TIGHT) * vol_exit_blend
                effective_os = base_os + (RSI_OS_TIGHT - base_os) * vol_exit_blend
                entry = self.entry_prices[symbol]
                pos_pnl = (mid - entry) / entry
                if current_pos < 0:
                    pos_pnl = -pos_pnl
                adaptive_profit_thresh = RSI_EXIT_PROFIT_THRESHOLD * max(0.7, min(1.4, vol_ratio ** 0.5))
                if pos_pnl > adaptive_profit_thresh:
                    profit_excess = pos_pnl - adaptive_profit_thresh
                    adaptive_profit_scale = RSI_EXIT_PROFIT_SCALE / max(0.6, min(1.8, vol_ratio))
                    calm_tighten_boost = max(0.0, (0.70 - vol_ratio) / 0.15) if vol_ratio < 0.70 else 0.0
                    adaptive_profit_tighten = RSI_EXIT_PROFIT_TIGHTEN * (1.0 + 0.40 * min(1.0, calm_tighten_boost))
                    profit_blend = min(adaptive_profit_tighten, profit_excess * adaptive_profit_scale)
                    effective_ob = effective_ob - (effective_ob - 50.0) * profit_blend
                    effective_os = effective_os + (50.0 - effective_os) * profit_blend
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)
                if bars_held < RSI_YOUNG_GRACE_BARS:
                    grace_blend = 1.0 - bars_held / RSI_YOUNG_GRACE_BARS
                    effective_ob += RSI_YOUNG_OB_WIDEN * grace_blend
                    effective_os -= RSI_YOUNG_OS_WIDEN * grace_blend
                if current_pos > 0 and rsi > effective_ob:
                    target = 0.0
                elif current_pos < 0 and rsi < effective_os:
                    target = 0.0

                if target != 0 and bars_held >= 1:
                    prev_peak = self.peak_pnl.get(symbol, 0.0)
                    self.peak_pnl[symbol] = max(prev_peak, pos_pnl)
                    adaptive_peak_min = PEAK_PROFIT_MIN_BASE * max(0.6, min(2.0, vol_ratio ** 0.5))
                    if self.peak_pnl[symbol] > adaptive_peak_min:
                        giveback = self.peak_pnl[symbol] - pos_pnl
                        if giveback > self.peak_pnl[symbol] * PEAK_PROFIT_GIVEBACK:
                            target = 0.0

                flip_bearish = bear_votes >= FLIP_MIN_VOTES and trend_bear
                flip_bullish = bull_votes >= FLIP_MIN_VOTES and trend_bull
                if current_pos > 0 and flip_bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and flip_bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    self.entry_prices.pop(symbol, None)
                    self.peak_pnl.pop(symbol, None)
                    self.entry_bar.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_pnl[symbol] = 0.0
                    self.entry_bar[symbol] = self.bar_count

        return signals
