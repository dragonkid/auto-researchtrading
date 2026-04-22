import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]

MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 10
LONG_WINDOW = 20
EMA_SLOW = 21
RSI_PERIOD = 8
RSI_OVERBOUGHT = 73
RSI_OVERSOLD = 27

MACD_SLOW = 16
MACD_SIGNAL = 5

EMA_SLOPE_PERIOD = 22
EMA_SLOPE_LOOKBACK = 3
LINREG_PERIOD = 16

VOL_LONG_LOOKBACK = 36
TARGET_VOL = 0.015
VOL_COMPRESS_THRESHOLD = 0.75
MEANREV_TREND_THRESHOLD = 0.05
RSI_EXIT_PROFIT_THRESHOLD = 0.01
RSI_YOUNG_GRACE_BARS = 4
PEAK_PROFIT_MIN = 0.025
PEAK_PROFIT_GIVEBACK = 0.30
PEAK_PROFIT_AGE_BARS = 6
VOL_BREAKOUT_LONG = 20
DONCHIAN_PERIOD = 12
MIN_VOTES = 3
MIN_VOTES_CALM = 3
FLIP_MIN_VOTES = 4
MAX_COMBINED_MULT = 3.5
MAX_COMBINED_MULT_LOW_VOL = 6.5
MAX_COMBINED_VOL_THRESHOLD = 1.2
MAX_COMBINED_LOW_VOL_THRESHOLD = 0.6
VOL_CONFIRM_BASE = 24
LINREG_R2_THRESH_REDUCE = 0.28

def ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

def calc_rsi(closes, period):
    if len(closes) < period + 1:
        return 50.0
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

    def _calc_vol(self, closes, lookback):
        if len(closes) < lookback:
            return TARGET_VOL
        log_rets = np.diff(np.log(closes[-lookback:]))
        return max(np.std(log_rets), 1e-6)

    def _calc_macd(self, closes):
        if len(closes) < MACD_SLOW + MACD_SIGNAL + 5:
            return 0.0
        fast_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], 6)
        slow_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, MACD_SIGNAL)
        return macd_line[-1] - signal_line[-1]

    def _calc_ema_slope(self, closes):
        if len(closes) < EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5:
            return 0.0
        ema_arr = ema(closes[-(EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5):], EMA_SLOPE_PERIOD)
        slope = (ema_arr[-1] - ema_arr[-EMA_SLOPE_LOOKBACK]) / ema_arr[-EMA_SLOPE_LOOKBACK]
        return slope

    def _calc_linreg(self, closes):
        if len(closes) < LINREG_PERIOD:
            return 0.0, 0.0
        y = np.log(closes[-LINREG_PERIOD:])
        n = LINREG_PERIOD
        x = np.arange(n, dtype=float)
        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        slope = ss_xy / ss_xx
        ss_yy = np.sum((y - y_mean) ** 2)
        r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-20 else 0.0
        return slope, r_sq

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        # Cross-asset momentum agreement
        cross_asset_rets = []
        for s in ACTIVE_SYMBOLS:
            if s in bar_data and len(bar_data[s].history) >= MED2_WINDOW + 1:
                c = bar_data[s].history["close"].values
                cross_asset_rets.append((c[-1] - c[-MED2_WINDOW]) / c[-MED2_WINDOW])
        if len(cross_asset_rets) >= 2:
            n_pos = sum(1 for r in cross_asset_rets if r > 0)
            n_neg = sum(1 for r in cross_asset_rets if r < 0)
            n_total = len(cross_asset_rets)
            agree_frac = max(n_pos, n_neg) / n_total
            cross_asset_agree = 1.0 + 0.20 * agree_frac if agree_frac > 0.5 else 1.0
        else:
            cross_asset_agree = 1.0

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close

            realized_vol = self._calc_vol(closes, 24)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = 0.005 * (0.10 + vol_ratio * 0.90) ** 0.85
            dyn_threshold = max(0.004, min(0.015, dyn_threshold))

            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            trend_strength = min(abs(ret_long) / 0.13, 1.0) ** 0.85
            trend_reduction = 0.32 * (1.0 - trend_strength)
            dyn_threshold *= (1.0 - trend_reduction)

            vol_compressed = False
            short_vol = long_vol = None
            sl_ratio_raw = 1.0
            if len(closes) >= VOL_LONG_LOOKBACK + 1:
                short_vol = self._calc_vol(closes, 12)
                long_vol = self._calc_vol(closes, VOL_LONG_LOOKBACK)
                sl_ratio_raw = short_vol / max(long_vol, 1e-10)
                if sl_ratio_raw < VOL_COMPRESS_THRESHOLD:
                    vol_compressed = True
                    compress_str = (VOL_COMPRESS_THRESHOLD - max(0.3, min(1.5, sl_ratio_raw))) / VOL_COMPRESS_THRESHOLD
                    dyn_threshold *= (1.0 - 0.25 * compress_str)

            linreg_slope, linreg_r2 = self._calc_linreg(closes)
            dyn_threshold *= (1.0 - LINREG_R2_THRESH_REDUCE * linreg_r2)

            adaptive_med = int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))
            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, adaptive_med))

            ret_vshort = (closes[-1] - closes[-8]) / closes[-8]
            ret_short = (closes[-1] - closes[-adaptive_med]) / closes[-adaptive_med]
            ret_med = (closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]

            mom_bull = ret_short > dyn_threshold
            mom_bear = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.5
            vshort_bear = ret_vshort < -dyn_threshold * 0.5

            ema_fast_arr = ema(closes[-(EMA_SLOW+10):], 3)
            ema_slow_arr = ema(closes[-(EMA_SLOW+10):], EMA_SLOW)
            ema_bull = ema_fast_arr[-1] > ema_slow_arr[-1]
            ema_bear = ema_fast_arr[-1] < ema_slow_arr[-1]

            rsi_trend_str = min(abs(ret_long) / 0.10, 1.0)
            adaptive_rsi_period = int(round(6 + (RSI_PERIOD - 6) * rsi_trend_str))
            rsi = calc_rsi(closes, adaptive_rsi_period)
            rsi_trend_blend = min(abs(ret_long) / 0.10, 1.0)
            rsi_bias = 1.5 * rsi_trend_blend
            rsi_thresh = 50 + (-rsi_bias if ret_long > 0 else rsi_bias)
            rsi_bull = rsi > rsi_thresh
            rsi_bear = rsi < rsi_thresh

            macd_hist = self._calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            ema_slope = self._calc_ema_slope(closes)
            slope_bull = ema_slope > 0.0005
            slope_bear = ema_slope < -0.0005

            linreg_bull = linreg_slope > 0.0001
            linreg_bear = linreg_slope < -0.0001

            vol_breakout_bull = False
            vol_breakout_bear = False
            if len(closes) >= VOL_BREAKOUT_LONG + 1:
                vb_short = self._calc_vol(closes, 3)
                vb_long = self._calc_vol(closes, VOL_BREAKOUT_LONG)
                if vb_short > vb_long:
                    if ret_vshort > 0:
                        vol_breakout_bull = True
                    elif ret_vshort < 0:
                        vol_breakout_bear = True

            donchian_bull = False
            donchian_bear = False
            if len(closes) >= DONCHIAN_PERIOD + 1:
                donchian_high = np.max(closes[-(DONCHIAN_PERIOD+1):-1])
                donchian_low = np.min(closes[-(DONCHIAN_PERIOD+1):-1])
                if mid >= donchian_high:
                    donchian_bull = True
                elif mid <= donchian_low:
                    donchian_bear = True

            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, vol_breakout_bull, linreg_bull, donchian_bull, slope_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, vol_breakout_bear, linreg_bear, donchian_bear, slope_bear])

            trend_adapt_strength = min(abs(ret_long) / 0.06, 1.0) ** 0.85
            trend_med_weight = 0.90 + (0.70 - 0.90) * trend_adapt_strength
            trend_avg = trend_med_weight * ret_med + (1.0 - trend_med_weight) * ret_long
            trend_bull = trend_avg > 0
            trend_bear = trend_avg < 0

            in_sideways = abs(ret_long) < MEANREV_TREND_THRESHOLD
            effective_min_votes = MIN_VOTES_CALM if (vol_ratio < 0.9 or vol_compressed or in_sideways) else MIN_VOTES
            trend_gate_bypassed = in_sideways and abs(trend_avg) < 0.006
            bullish = bull_votes >= effective_min_votes and (trend_bull or trend_gate_bypassed)
            bearish = bear_votes >= effective_min_votes and (trend_bear or trend_gate_bypassed)

            cooldown_trend_strength = min(abs(ret_long) / 0.06, 1.0)
            effective_cooldown = 3 * cooldown_trend_strength
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < effective_cooldown

            vol_scale = (TARGET_VOL / realized_vol) ** 0.85
            vol_scale = max(0.3, min(2.5, vol_scale))

            calm_boost = 1.0
            vol_compress_boost = 1.0
            if short_vol is not None:
                vol_ratio_sl = max(0.5, min(2.0, sl_ratio_raw))
                calm_boost = 1.0 + 0.8 * max(0.0, 1.0 - vol_ratio_sl) ** 0.85
                if vol_ratio_sl < VOL_COMPRESS_THRESHOLD:
                    compress_strength = (VOL_COMPRESS_THRESHOLD - vol_ratio_sl) / VOL_COMPRESS_THRESHOLD
                    vol_compress_boost = 1.0 + 0.50 * compress_strength ** 0.85

            sideways_trend_ratio = min(abs(ret_long) / 0.10, 1.0)
            sideways_trend_strength = sideways_trend_ratio ** 1.7
            sideways_boost = 1.0 + 0.70 * (1.0 - sideways_trend_strength)

            winning_votes = max(bull_votes, bear_votes)
            vote_boost = 1.20 if winning_votes >= 3 else 1.0

            vol_confirm_mult = 1.0
            volumes = bd.history["volume"].values
            if len(volumes) >= VOL_CONFIRM_BASE:
                recent_vol = np.mean(volumes[-12:])
                base_vol = np.mean(volumes[-VOL_CONFIRM_BASE:])
                if base_vol > 0:
                    vol_confirm_mult = max(0.98, min(1.20, recent_vol / base_vol))

            weight = 0.33
            mom_strength = (abs(ret_short) / dyn_threshold) ** 0.85
            sideways_strength = min(abs(ret_long) / 0.12, 1.0)
            strength_floor = 0.6 + (2.6 - 0.6) * (1.0 - sideways_strength)
            strength_scale = max(strength_floor, min(2.0, mom_strength))
            cross_trend_strength = min(abs(ret_long) / 0.06, 1.0)
            dampened_cross_agree = 1.0 + (cross_asset_agree - 1.0) * (1.0 - cross_trend_strength)
            combined_mult = vol_scale * strength_scale * calm_boost * sideways_boost * dampened_cross_agree * vote_boost * vol_compress_boost * vol_confirm_mult
            if vol_ratio < MAX_COMBINED_LOW_VOL_THRESHOLD:
                adaptive_cap = MAX_COMBINED_MULT_LOW_VOL
            elif vol_ratio > MAX_COMBINED_VOL_THRESHOLD:
                adaptive_cap = 2.5
            else:
                blend = (vol_ratio - MAX_COMBINED_LOW_VOL_THRESHOLD) / (MAX_COMBINED_VOL_THRESHOLD - MAX_COMBINED_LOW_VOL_THRESHOLD)
                adaptive_cap = MAX_COMBINED_MULT_LOW_VOL + (MAX_COMBINED_MULT - MAX_COMBINED_MULT_LOW_VOL) * blend
            trend_cap_strength = min(abs(ret_long) / 0.10, 1.0) ** 0.85
            trend_cap_boost = 1.5 * (1.0 - trend_cap_strength)
            adaptive_cap += trend_cap_boost
            combined_mult = min(combined_mult, adaptive_cap)
            size = equity * 0.30 * weight * combined_mult

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                if not in_cooldown:
                    if bullish:
                        target = size
                    elif bearish:
                        target = -size
                    elif abs(ret_long) < MEANREV_TREND_THRESHOLD:
                        if rsi < 49:
                            target = size
                        elif rsi > 51:
                            target = -size
            else:
                vol_exit_blend = max(0.0, min(1.0, (vol_ratio - 0.7) / (1.8 - 0.7)))
                trend_exit_strength = min(abs(ret_long) / 0.08, 1.0)
                sideways_ob_widen = (74 - RSI_OVERBOUGHT) * (1.0 - trend_exit_strength)
                sideways_os_widen = (RSI_OVERSOLD - 26) * (1.0 - trend_exit_strength)
                base_ob = RSI_OVERBOUGHT + sideways_ob_widen
                base_os = RSI_OVERSOLD + sideways_os_widen
                effective_ob = base_ob - (base_ob - 65) * vol_exit_blend
                effective_os = base_os + (35 - base_os) * vol_exit_blend
                if symbol in self.entry_prices:
                    entry = self.entry_prices[symbol]
                    pos_pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pos_pnl = -pos_pnl
                    if pos_pnl > RSI_EXIT_PROFIT_THRESHOLD:
                        profit_excess = pos_pnl - RSI_EXIT_PROFIT_THRESHOLD
                        profit_blend = min(0.15, profit_excess * 20.0)
                        effective_ob = effective_ob - (effective_ob - 50.0) * profit_blend
                        effective_os = effective_os + (50.0 - effective_os) * profit_blend
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)
                if bars_held < RSI_YOUNG_GRACE_BARS:
                    grace_blend = 1.0 - bars_held / RSI_YOUNG_GRACE_BARS
                    effective_ob += 4.0 * grace_blend
                    effective_os -= 4.0 * grace_blend
                if current_pos > 0 and rsi > effective_ob:
                    target = 0.0
                elif current_pos < 0 and rsi < effective_os:
                    target = 0.0

                if target != 0 and symbol in self.entry_prices and bars_held >= 1:
                    entry = self.entry_prices[symbol]
                    pos_pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pos_pnl = -pos_pnl
                    prev_peak = self.peak_pnl.get(symbol, 0.0)
                    self.peak_pnl[symbol] = max(prev_peak, pos_pnl)
                    if self.peak_pnl[symbol] > PEAK_PROFIT_MIN:
                        profit_blend = min(1.0, (self.peak_pnl[symbol] - PEAK_PROFIT_MIN) / (0.03 - PEAK_PROFIT_MIN))
                        regime_giveback = PEAK_PROFIT_GIVEBACK - 0.08 * (1.0 - sideways_trend_ratio)
                        effective_giveback = regime_giveback + (0.25 - regime_giveback) * profit_blend
                        if bars_held > PEAK_PROFIT_AGE_BARS:
                            age_excess = min(1.0, (bars_held - PEAK_PROFIT_AGE_BARS) / PEAK_PROFIT_AGE_BARS)
                            effective_giveback -= 0.10 * age_excess
                            effective_giveback = max(0.10, effective_giveback)
                        giveback = self.peak_pnl[symbol] - pos_pnl
                        if giveback > self.peak_pnl[symbol] * effective_giveback:
                            target = 0.0

                flip_bearish = bear_votes >= FLIP_MIN_VOTES and trend_bear
                flip_bullish = bull_votes >= FLIP_MIN_VOTES and trend_bull
                if current_pos > 0 and flip_bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and flip_bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target != 0 and current_pos == 0:
                    self.entry_prices[symbol] = mid
                    self.peak_pnl[symbol] = 0.0
                    self.entry_bar[symbol] = self.bar_count
                elif target == 0:
                    self.entry_prices.pop(symbol, None)
                    self.peak_pnl.pop(symbol, None)
                    self.entry_bar.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_pnl[symbol] = 0.0
                    self.entry_bar[symbol] = self.bar_count

        return signals
