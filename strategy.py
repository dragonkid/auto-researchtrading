"""
Exp119: Reduce BASE_THRESHOLD from 0.010 to 0.008.

The momentum entry threshold is the base for dyn_threshold. Lowering it
allows more entries in low-momentum environments (especially sideways),
where signals are currently filtered too aggressively. The vote system
(MIN_VOTES=3) and trend gate still provide quality control. This should
particularly help the sideways regime (weakest at 18.13) by generating
more trades without significantly increasing DD in trending regimes.
"""

import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]
SYMBOL_WEIGHTS = {"BTC": 0.33, "ETH": 0.33, "SOL": 0.33}

SHORT_WINDOW = 8
MED_WINDOW = 12
MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 12
LONG_WINDOW = 20
EMA_FAST = 3
EMA_SLOW = 21
RSI_PERIOD = 8
RSI_BULL = 50
RSI_BEAR = 50
RSI_OVERBOUGHT = 73
RSI_OVERSOLD = 27

MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 7

EMA_SLOPE_PERIOD = 28
EMA_SLOPE_LOOKBACK = 4

FUNDING_LOOKBACK = 24
FUNDING_BOOST = 0.0
BASE_POSITION_PCT = 0.38
VOL_LOOKBACK = 24
VOL_SHORT_LOOKBACK = 12
VOL_LONG_LOOKBACK = 48
VOL_SPIKE_THRESHOLD = 1.5
VOL_SPIKE_SCALE = 0.6
TARGET_VOL = 0.015
ATR_LOOKBACK = 16
ATR_STOP_MULT_BASE = 4.5
ATR_STOP_MULT_MIN = 3.0
ATR_STOP_MULT_MAX = 6.0
TAKE_PROFIT_PCT = 99.0
BASE_THRESHOLD = 0.008
BTC_OPPOSE_THRESHOLD = -99.0

PYRAMID_THRESHOLD = 0.015
PYRAMID_SIZE = 0.0
CORR_LOOKBACK = 72
HIGH_CORR_THRESHOLD = 99.0

DD_REDUCE_THRESHOLD = 99.0
DD_REDUCE_SCALE = 0.5

CALM_BOOST_MAX = 0.8  # max position size boost in calm regimes
SIDEWAYS_BOOST_MAX = 0.70  # max position size boost in weak-trend (sideways) regimes
SIDEWAYS_BOOST_DECAY = 0.08  # abs(ret_long) at which sideways boost fully decays

STOP_WITH_TREND_MULT = 1.25     # wider stop when position aligns with long-term trend
STOP_AGAINST_TREND_MULT = 0.75  # tighter stop when position opposes long-term trend

STOP_FLAT_TREND_BOOST = 0.35    # max stop widening when trend is near zero
STOP_FLAT_TREND_DECAY = 0.08    # abs(ret_long) at which flat-trend boost fully decays

TREND_THRESHOLD_SCALE = 0.38  # max threshold reduction when trend is flat
TREND_THRESHOLD_DECAY = 0.13  # abs(ret_long) at which reduction fully decays

TREND_GATE_MED_WEIGHT_BASE = 0.50   # ret_med weight in trending markets
TREND_GATE_MED_WEIGHT_SIDEWAYS = 0.85  # ret_med weight in trendless markets
TREND_GATE_ADAPT_DECAY = 0.08       # abs(ret_long) at which adaptation fully decays

STRENGTH_FLOOR_SIDEWAYS = 1.6  # strength_scale floor in fully trendless markets
STRENGTH_FLOOR_DECAY = 0.08    # abs(ret_long) at which floor decays back to 0.6

CROSS_ASSET_BOOST = 0.20  # max size boost when all assets agree on direction
COOLDOWN_BARS = 2
MIN_VOTES = 3  # out of 6 — simple majority for more entries in sideways
HIGH_VOTE_THRESHOLD = 4  # votes at or above this count get a sizing bonus
HIGH_VOTE_BOOST = 0.15   # max position size boost for high-conviction entries
FLIP_MIN_VOTES = 4       # votes required to flip an existing position (vs MIN_VOTES for new entry)
MAX_COMBINED_MULT = 5.5  # cap on product of all sizing multipliers to prevent extreme stacking

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
        self.peak_prices = {}
        self.atr_at_entry = {}
        self.btc_momentum = 0.0
        self.pyramided = {}
        self.peak_equity = 100000.0
        self.exit_bar = {}
        self.bar_count = 0

    def _calc_atr(self, history, lookback):
        if len(history) < lookback + 1:
            return None
        highs = history["high"].values[-lookback:]
        lows = history["low"].values[-lookback:]
        closes = history["close"].values[-(lookback+1):-1]
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
        return np.mean(tr)

    def _calc_vol(self, closes, lookback):
        if len(closes) < lookback:
            return TARGET_VOL
        log_rets = np.diff(np.log(closes[-lookback:]))
        return max(np.std(log_rets), 1e-6)

    def _calc_correlation(self, bar_data):
        if "BTC" not in bar_data or "ETH" not in bar_data:
            return 0.5
        btc_h = bar_data["BTC"].history
        eth_h = bar_data["ETH"].history
        if len(btc_h) < CORR_LOOKBACK or len(eth_h) < CORR_LOOKBACK:
            return 0.5
        btc_rets = np.diff(np.log(btc_h["close"].values[-CORR_LOOKBACK:]))
        eth_rets = np.diff(np.log(eth_h["close"].values[-CORR_LOOKBACK:]))
        if len(btc_rets) < 10:
            return 0.5
        corr = np.corrcoef(btc_rets, eth_rets)[0, 1]
        return corr if not np.isnan(corr) else 0.5

    def _calc_macd(self, closes):
        if len(closes) < MACD_SLOW + MACD_SIGNAL + 5:
            return 0.0
        fast_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST)
        slow_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, MACD_SIGNAL)
        return macd_line[-1] - signal_line[-1]

    def _calc_ema_slope(self, closes):
        """Calculate slope of a long EMA over recent bars. Positive = uptrend."""
        if len(closes) < EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5:
            return 0.0
        ema_arr = ema(closes[-(EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5):], EMA_SLOPE_PERIOD)
        # Slope = change in EMA over lookback period, normalized by price
        slope = (ema_arr[-1] - ema_arr[-EMA_SLOPE_LOOKBACK]) / ema_arr[-EMA_SLOPE_LOOKBACK]
        return slope

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        self.peak_equity = max(self.peak_equity, equity)
        current_dd = (self.peak_equity - equity) / self.peak_equity
        dd_scale = 1.0
        if current_dd > DD_REDUCE_THRESHOLD:
            dd_scale = max(DD_REDUCE_SCALE, 1.0 - (current_dd - DD_REDUCE_THRESHOLD) * 5)

        if "BTC" in bar_data and len(bar_data["BTC"].history) >= LONG_WINDOW + 1:
            btc_closes = bar_data["BTC"].history["close"].values
            self.btc_momentum = (btc_closes[-1] - btc_closes[-MED2_WINDOW]) / btc_closes[-MED2_WINDOW]

        btc_eth_corr = self._calc_correlation(bar_data)
        high_corr = btc_eth_corr > HIGH_CORR_THRESHOLD

        # Cross-asset momentum agreement: check if all symbols trend in same direction
        cross_asset_rets = []
        for s in ACTIVE_SYMBOLS:
            if s in bar_data and len(bar_data[s].history) >= MED2_WINDOW + 1:
                c = bar_data[s].history["close"].values
                cross_asset_rets.append((c[-1] - c[-MED2_WINDOW]) / c[-MED2_WINDOW])
        if len(cross_asset_rets) >= 2:
            all_positive = all(r > 0 for r in cross_asset_rets)
            all_negative = all(r < 0 for r in cross_asset_rets)
            cross_asset_agree = 1.0 + CROSS_ASSET_BOOST if (all_positive or all_negative) else 1.0
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

            realized_vol = self._calc_vol(closes, VOL_LOOKBACK)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.3 + vol_ratio * 0.7)
            dyn_threshold = max(0.003, min(0.020, dyn_threshold))

            # Reduce threshold in trendless markets (sideways)
            # When abs(ret_long) is near zero, trend is weak → lower the bar for entries
            ret_long_raw = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            trend_strength = min(abs(ret_long_raw) / TREND_THRESHOLD_DECAY, 1.0)
            trend_reduction = TREND_THRESHOLD_SCALE * (1.0 - trend_strength)
            dyn_threshold *= (1.0 - trend_reduction)

            # Adaptive momentum lookback: shorter in high vol, longer in low vol
            adaptive_med = int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))
            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, adaptive_med))

            ret_vshort = (closes[-1] - closes[-SHORT_WINDOW]) / closes[-SHORT_WINDOW]
            ret_short = (closes[-1] - closes[-adaptive_med]) / closes[-adaptive_med]
            ret_med = (closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]
            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]

            mom_bull = ret_short > dyn_threshold
            mom_bear = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.5
            vshort_bear = ret_vshort < -dyn_threshold * 0.5

            ema_fast_arr = ema(closes[-(EMA_SLOW+10):], EMA_FAST)
            ema_slow_arr = ema(closes[-(EMA_SLOW+10):], EMA_SLOW)
            ema_bull = ema_fast_arr[-1] > ema_slow_arr[-1]
            ema_bear = ema_fast_arr[-1] < ema_slow_arr[-1]

            rsi = calc_rsi(closes, RSI_PERIOD)
            rsi_bull = rsi > RSI_BULL
            rsi_bear = rsi < RSI_BEAR

            macd_hist = self._calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            # EMA slope: rising long EMA = bullish, falling = bearish
            ema_slope = self._calc_ema_slope(closes)
            slope_bull = ema_slope > 0.0005
            slope_bear = ema_slope < -0.0005

            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, slope_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, slope_bear])

            btc_confirm = True
            if symbol != "BTC":
                if bull_votes >= MIN_VOTES and self.btc_momentum < BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False
                if bear_votes >= MIN_VOTES and self.btc_momentum > -BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False

            # Trend gate: weighted average of med and long returns must confirm direction
            # In sideways markets, shift weight toward faster ret_med for responsiveness
            trend_adapt_strength = min(abs(ret_long) / TREND_GATE_ADAPT_DECAY, 1.0)
            trend_med_weight = TREND_GATE_MED_WEIGHT_SIDEWAYS + (TREND_GATE_MED_WEIGHT_BASE - TREND_GATE_MED_WEIGHT_SIDEWAYS) * trend_adapt_strength
            trend_avg = trend_med_weight * ret_med + (1.0 - trend_med_weight) * ret_long
            trend_bull = trend_avg > 0
            trend_bear = trend_avg < 0

            bullish = bull_votes >= MIN_VOTES and btc_confirm and trend_bull
            bearish = bear_votes >= MIN_VOTES and btc_confirm and trend_bear

            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS

            vol_scale = TARGET_VOL / realized_vol
            vol_scale = max(0.4, min(2.0, vol_scale))

            # Vol-spike scaling: reduce size when short-term vol spikes above medium-term
            vol_spike_scale = 1.0
            calm_boost = 1.0
            if len(closes) >= VOL_LONG_LOOKBACK + 1:
                short_vol = self._calc_vol(closes, VOL_SHORT_LOOKBACK)
                long_vol = self._calc_vol(closes, VOL_LONG_LOOKBACK)
                if short_vol > long_vol * VOL_SPIKE_THRESHOLD:
                    vol_spike_scale = VOL_SPIKE_SCALE
                # Calm regime boost: when short vol is close to or below long vol, boost size
                vol_ratio_sl = max(0.5, min(2.0, short_vol / max(long_vol, 1e-10)))
                calm_boost = 1.0 + CALM_BOOST_MAX * max(0.0, 1.0 - vol_ratio_sl)

            # Sideways regime boost: when long-term trend is weak, boost size
            # to capture more return in range-bound markets where risk is low
            sideways_trend_strength = min(abs(ret_long) / SIDEWAYS_BOOST_DECAY, 1.0)
            sideways_boost = 1.0 + SIDEWAYS_BOOST_MAX * (1.0 - sideways_trend_strength)

            # High-conviction vote bonus: boost sizing when 5+ out of 6 signals agree
            winning_votes = max(bull_votes, bear_votes)
            vote_boost = 1.0 + HIGH_VOTE_BOOST if winning_votes >= HIGH_VOTE_THRESHOLD else 1.0

            weight = SYMBOL_WEIGHTS.get(symbol, 0.33)
            if high_corr and symbol == "SOL":
                weight *= 0.5
            mom_strength = abs(ret_short) / dyn_threshold
            # In sideways markets, raise the floor so weak momentum isn't double-penalized
            sideways_strength = min(abs(ret_long) / STRENGTH_FLOOR_DECAY, 1.0)
            strength_floor = 0.6 + (STRENGTH_FLOOR_SIDEWAYS - 0.6) * (1.0 - sideways_strength)
            strength_scale = max(strength_floor, min(2.0, mom_strength))
            combined_mult = vol_scale * vol_spike_scale * strength_scale * calm_boost * sideways_boost * cross_asset_agree * vote_boost
            combined_mult = min(combined_mult, MAX_COMBINED_MULT)
            size = equity * BASE_POSITION_PCT * weight * combined_mult * dd_scale

            funding_rates = bd.history["funding_rate"].values[-FUNDING_LOOKBACK:]
            avg_funding = np.mean(funding_rates) if len(funding_rates) >= FUNDING_LOOKBACK else 0.0

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                if not in_cooldown:
                    funding_mult = 1.0
                    if bullish:
                        if avg_funding < 0:
                            funding_mult = 1.0 + FUNDING_BOOST
                        target = size * funding_mult
                        self.pyramided[symbol] = False
                    elif bearish:
                        if avg_funding > 0:
                            funding_mult = 1.0 + FUNDING_BOOST
                        target = -size * funding_mult
                        self.pyramided[symbol] = False
            else:
                if symbol in self.entry_prices and not self.pyramided.get(symbol, True):
                    entry = self.entry_prices[symbol]
                    pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > PYRAMID_THRESHOLD:
                        if current_pos > 0 and bullish:
                            target = current_pos + size * PYRAMID_SIZE
                            self.pyramided[symbol] = True
                        elif current_pos < 0 and bearish:
                            target = current_pos - size * PYRAMID_SIZE
                            self.pyramided[symbol] = True

                atr = self._calc_atr(bd.history, ATR_LOOKBACK)
                if atr is None:
                    atr = self.atr_at_entry.get(symbol, mid * 0.02)

                # Adaptive stop: tighter in high vol, wider in low vol
                atr_stop_mult = ATR_STOP_MULT_BASE / max(vol_ratio, 0.5)
                atr_stop_mult = max(ATR_STOP_MULT_MIN, min(ATR_STOP_MULT_MAX, atr_stop_mult))

                # Asymmetric stop: wider when position aligns with long-term trend
                if current_pos > 0 and ret_long > 0:
                    atr_stop_mult *= STOP_WITH_TREND_MULT
                elif current_pos > 0 and ret_long < 0:
                    atr_stop_mult *= STOP_AGAINST_TREND_MULT
                elif current_pos < 0 and ret_long < 0:
                    atr_stop_mult *= STOP_WITH_TREND_MULT
                elif current_pos < 0 and ret_long > 0:
                    atr_stop_mult *= STOP_AGAINST_TREND_MULT

                # Flat-trend stop widening: in sideways markets, widen stops
                # to avoid premature exits from mean-reverting price action
                flat_trend_strength = min(abs(ret_long) / STOP_FLAT_TREND_DECAY, 1.0)
                flat_trend_boost = 1.0 + STOP_FLAT_TREND_BOOST * (1.0 - flat_trend_strength)
                atr_stop_mult *= flat_trend_boost

                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] - atr_stop_mult * atr
                    if mid < stop:
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] + atr_stop_mult * atr
                    if mid > stop:
                        target = 0.0

                if symbol in self.entry_prices:
                    entry = self.entry_prices[symbol]
                    pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > TAKE_PROFIT_PCT:
                        target = 0.0

                if current_pos > 0 and rsi > RSI_OVERBOUGHT:
                    target = 0.0
                elif current_pos < 0 and rsi < RSI_OVERSOLD:
                    target = 0.0

                # Require higher conviction to flip (more expensive than new entry)
                flip_bearish = bear_votes >= FLIP_MIN_VOTES and btc_confirm and trend_bear
                flip_bullish = bull_votes >= FLIP_MIN_VOTES and btc_confirm and trend_bull
                if current_pos > 0 and flip_bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and flip_bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target != 0 and current_pos == 0:
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = self._calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
                elif target == 0:
                    self.entry_prices.pop(symbol, None)
                    self.peak_prices.pop(symbol, None)
                    self.atr_at_entry.pop(symbol, None)
                    self.pyramided.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = self._calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
                    self.pyramided[symbol] = False

        return signals
