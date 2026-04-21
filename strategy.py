"""
Exp212: Reduce MAX_COMBINED_MULT_HIGH_VOL 3.0->2.5 to tighten sizing cap
in high-vol regimes. The std across regimes is high (5.08) — the main
drag on composite score. Tighter high-vol cap should reduce DD in crash/
volatile periods while preserving returns in calm regimes, compressing std.
"""

import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]
SYMBOL_WEIGHTS = {"BTC": 0.33, "ETH": 0.33, "SOL": 0.33}

SHORT_WINDOW = 8
MED_WINDOW = 12
MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 10
LONG_WINDOW = 20
EMA_FAST = 3
EMA_SLOW = 21
RSI_PERIOD = 8
RSI_PERIOD_SIDEWAYS = 6
RSI_BULL = 50
RSI_BEAR = 50
RSI_OVERBOUGHT = 73
RSI_OVERSOLD = 27
RSI_OB_TIGHT = 65     # tightest OB exit in extreme high-vol
RSI_OS_TIGHT = 35     # tightest OS exit in extreme high-vol
RSI_OB_WIDE = 76      # widest OB exit in sideways/trendless markets
RSI_OS_WIDE = 24      # widest OS exit in sideways/trendless markets
RSI_EXIT_VOL_LOW = 0.7   # vol_ratio below this: use standard thresholds
RSI_EXIT_VOL_HIGH = 1.8  # vol_ratio above this: use tightest thresholds
RSI_EXIT_TREND_DECAY = 0.10  # abs(ret_long) at which sideways widening fully decays

MACD_FAST = 6
MACD_SLOW = 21
MACD_SIGNAL = 7

EMA_SLOPE_PERIOD = 28
EMA_SLOPE_LOOKBACK = 4
LINREG_PERIOD = 16  # rolling linear regression window for slope voter

FUNDING_LOOKBACK = 24
FUNDING_BOOST = 0.0
FUNDING_EXTREME_PERCENTILE = 0.80  # funding above this percentile = crowded
FUNDING_EXTREME_DECEL_MULT = 0.5   # tighten decel by this factor when crowded
BASE_POSITION_PCT = 0.30
VOL_LOOKBACK = 24
VOL_SHORT_LOOKBACK = 12
VOL_LONG_LOOKBACK = 48
VOL_SPIKE_THRESHOLD = 1.7
VOL_SPIKE_SCALE = 0.7
TARGET_VOL = 0.015
ATR_LOOKBACK = 16
ATR_STOP_MULT_BASE = 4.5
ATR_STOP_MULT_MIN = 3.0
ATR_STOP_MULT_MAX = 6.0
TAKE_PROFIT_PCT = 99.0
BASE_THRESHOLD = 0.005
BTC_OPPOSE_THRESHOLD = -99.0

PYRAMID_THRESHOLD = 0.015
PYRAMID_SIZE = 0.0
CORR_LOOKBACK = 72
HIGH_CORR_THRESHOLD = 99.0

DD_REDUCE_THRESHOLD = 99.0
DD_REDUCE_SCALE = 0.5

CALM_BOOST_MAX = 0.8  # max position size boost in calm regimes
SIDEWAYS_BOOST_MAX = 0.70  # max position size boost in weak-trend (sideways) regimes
SIDEWAYS_BOOST_DECAY = 0.10  # abs(ret_long) at which sideways boost fully decays

STOP_WITH_TREND_MULT = 1.25     # wider stop when position aligns with long-term trend
STOP_AGAINST_TREND_MULT = 0.75  # tighter stop when position opposes long-term trend

STOP_FLAT_TREND_BOOST = 0.35    # max stop widening when trend is near zero
STOP_FLAT_TREND_DECAY = 0.08    # abs(ret_long) at which flat-trend boost fully decays

DECEL_MULT_BASE = 999.0         # DISABLED: decel exit hurts score (+0.430 when removed)
DECEL_MULT_TREND = 999.0        # DISABLED: RSI exit + flip is better exit mechanism
DECEL_TREND_DECAY = 0.10        # abs(ret_long) at which multiplier fully reaches trend value

TREND_THRESHOLD_SCALE = 0.38  # max threshold reduction when trend is flat
TREND_THRESHOLD_DECAY = 0.13  # abs(ret_long) at which reduction fully decays

TREND_GATE_MED_WEIGHT_BASE = 0.50   # ret_med weight in trending markets
TREND_GATE_MED_WEIGHT_SIDEWAYS = 0.90  # ret_med weight in trendless markets
TREND_GATE_ADAPT_DECAY = 0.08       # abs(ret_long) at which adaptation fully decays

STRENGTH_FLOOR_SIDEWAYS = 2.4  # strength_scale floor in fully trendless markets
STRENGTH_FLOOR_DECAY = 0.10    # abs(ret_long) at which floor decays back to 0.6

VOL_COMPRESS_THRESHOLD = 0.70  # short_vol / long_vol below this = compression
VOL_COMPRESS_BOOST = 0.40     # max position size boost during vol compression
VOL_COMPRESS_THRESH_REDUCE = 0.25  # max entry threshold reduction during vol compression
CROSS_ASSET_BOOST = 0.30  # max size boost when all assets agree on direction
CROSS_ASSET_TREND_DECAY = 0.14  # abs(ret_long) at which cross-asset boost fully dampens
VOL_CONFIRM_LOOKBACK = 12     # short-term volume average window
VOL_CONFIRM_BASE = 48         # longer-term volume average window
VOL_CONFIRM_BOOST = 0.35      # max sizing boost when volume is above average
VOL_CONFIRM_FLOOR = 0.92      # min sizing factor when volume is below average
VOL_DIVERGENCE_THRESHOLD = 0.70  # vol ratio below this triggers tighter exit
VOL_DIVERGENCE_DECEL_MULT = 0.5  # decel multiplier when vol divergence detected
MEANREV_TREND_THRESHOLD = 0.03  # abs(ret_long) below this activates mean-reversion entries
MEANREV_SIZE_SCALE = 1.0        # mean-reversion entries use full normal size
MEANREV_RSI_OVERSOLD = 49       # less extreme RSI threshold for mean-reversion entries
MEANREV_RSI_OVERBOUGHT = 51     # less extreme RSI threshold for mean-reversion entries
PROFIT_DECEL_THRESHOLD = 0.02   # profit pct above which decel exit tightens
PROFIT_DECEL_SCALE = 10.0       # how fast decel tightens with excess profit
PROFIT_SMALL_THRESHOLD = 0.01   # profit below this gets wider decel (hold small winners)
PROFIT_SMALL_DECEL_WIDEN = 1.5  # decel multiplier widening for small winners
VOL_BREAKOUT_SHORT = 4   # short window for vol breakout detection
VOL_BREAKOUT_LONG = 20   # long window for vol breakout baseline
VOL_BREAKOUT_MULT = 1.0  # short vol must exceed long vol * this to trigger
DONCHIAN_PERIOD = 12  # lookback for Donchian channel breakout voter
COOLDOWN_BARS = 3
COOLDOWN_SIDEWAYS_BARS = 0  # faster re-entry in trendless markets
COOLDOWN_SIDEWAYS_DECAY = 0.06  # abs(ret_long) below which cooldown is reduced
MIN_VOTES = 3  # out of 6 — simple majority for more entries in sideways
MIN_VOTES_CALM = 2  # reduced vote requirement when vol_ratio < calm threshold
MIN_VOTES_CALM_VOL = 0.9  # vol_ratio below which reduced votes apply
HIGH_VOTE_THRESHOLD = 4  # votes at or above this count get a sizing bonus
HIGH_VOTE_BOOST = 0.20   # max position size boost for high-conviction entries
FLIP_MIN_VOTES = 4       # votes required to flip an existing position (vs MIN_VOTES for new entry)
MAX_COMBINED_MULT = 4.0  # base cap on product of all sizing multipliers
MAX_COMBINED_MULT_LOW_VOL = 6.5  # higher cap in low-vol regimes (more DD headroom)
MAX_COMBINED_MULT_HIGH_VOL = 2.5  # tighter cap in high-vol regimes (protect DD)
MAX_COMBINED_VOL_THRESHOLD = 1.0  # vol_ratio above this triggers tighter cap
MAX_COMBINED_LOW_VOL_THRESHOLD = 0.6  # vol_ratio below this gets the full low-vol cap
MTF_AGREE_BOOST = 0.0  # DISABLED: redundant with trend gate + high-vote boost
MTF_AGREE_TREND_DECAY = 0.10

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

    def _calc_linreg_slope(self, closes):
        """Rolling OLS linear regression slope of log prices. Per-bar change rate."""
        if len(closes) < LINREG_PERIOD:
            return 0.0
        y = np.log(closes[-LINREG_PERIOD:])
        n = LINREG_PERIOD
        x = np.arange(n, dtype=float)
        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        # OLS slope = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
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
            dyn_threshold = BASE_THRESHOLD * (0.15 + vol_ratio * 0.85)
            dyn_threshold = max(0.003, min(0.020, dyn_threshold))

            # Reduce threshold in trendless markets (sideways)
            # When abs(ret_long) is near zero, trend is weak → lower the bar for entries
            ret_long_raw = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            trend_strength = min(abs(ret_long_raw) / TREND_THRESHOLD_DECAY, 1.0)
            trend_reduction = TREND_THRESHOLD_SCALE * (1.0 - trend_strength)
            dyn_threshold *= (1.0 - trend_reduction)

            # Vol-compression threshold reduction: when short vol << long vol,
            # a breakout is brewing — lower entry threshold to catch it early
            if len(closes) >= VOL_LONG_LOOKBACK + 1:
                vc_short = self._calc_vol(closes, VOL_SHORT_LOOKBACK)
                vc_long = self._calc_vol(closes, VOL_LONG_LOOKBACK)
                vc_ratio = max(0.3, min(1.5, vc_short / max(vc_long, 1e-10)))
                if vc_ratio < VOL_COMPRESS_THRESHOLD:
                    compress_str = (VOL_COMPRESS_THRESHOLD - vc_ratio) / VOL_COMPRESS_THRESHOLD
                    dyn_threshold *= (1.0 - VOL_COMPRESS_THRESH_REDUCE * compress_str)

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

            # Adaptive RSI: shorter period in sideways for faster signals
            rsi_trend_str = min(abs(ret_long_raw) / 0.10, 1.0)
            adaptive_rsi_period = int(round(RSI_PERIOD_SIDEWAYS + (RSI_PERIOD - RSI_PERIOD_SIDEWAYS) * rsi_trend_str))
            rsi = calc_rsi(closes, adaptive_rsi_period)
            rsi_bull = rsi > RSI_BULL
            rsi_bear = rsi < RSI_BEAR

            macd_hist = self._calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            # EMA slope: rising long EMA = bullish, falling = bearish
            ema_slope = self._calc_ema_slope(closes)
            slope_bull = ema_slope > 0.0005
            slope_bear = ema_slope < -0.0005

            # Linear regression slope voter: more robust trend detection
            linreg_slope = self._calc_linreg_slope(closes)
            linreg_bull = linreg_slope > 0.0001
            linreg_bear = linreg_slope < -0.0001

            # Volatility breakout voter: vol expanding signals directional move
            vol_breakout_bull = False
            vol_breakout_bear = False
            if len(closes) >= VOL_BREAKOUT_LONG + 1:
                vb_short = self._calc_vol(closes, VOL_BREAKOUT_SHORT)
                vb_long = self._calc_vol(closes, VOL_BREAKOUT_LONG)
                if vb_short > vb_long * VOL_BREAKOUT_MULT:
                    # Vol is expanding — vote with the short-term direction
                    if ret_vshort > 0:
                        vol_breakout_bull = True
                    elif ret_vshort < 0:
                        vol_breakout_bear = True

            # Donchian channel breakout voter: price at N-bar high = bullish, N-bar low = bearish
            donchian_bull = False
            donchian_bear = False
            if len(closes) >= DONCHIAN_PERIOD + 1:
                donchian_high = np.max(closes[-(DONCHIAN_PERIOD+1):-1])
                donchian_low = np.min(closes[-(DONCHIAN_PERIOD+1):-1])
                if mid >= donchian_high:
                    donchian_bull = True
                elif mid <= donchian_low:
                    donchian_bear = True

            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, vol_breakout_bull, linreg_bull, donchian_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, vol_breakout_bear, linreg_bear, donchian_bear])

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

            effective_min_votes = MIN_VOTES_CALM if vol_ratio < MIN_VOTES_CALM_VOL else MIN_VOTES
            bullish = bull_votes >= effective_min_votes and btc_confirm and trend_bull
            bearish = bear_votes >= effective_min_votes and btc_confirm and trend_bear

            # Adaptive cooldown: shorter in sideways markets for faster re-entry
            cooldown_trend_strength = min(abs(ret_long) / COOLDOWN_SIDEWAYS_DECAY, 1.0)
            effective_cooldown = COOLDOWN_SIDEWAYS_BARS + (COOLDOWN_BARS - COOLDOWN_SIDEWAYS_BARS) * cooldown_trend_strength
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < effective_cooldown

            vol_scale = TARGET_VOL / realized_vol
            vol_scale = max(0.3, min(2.5, vol_scale))

            # Vol-spike scaling: reduce size when short-term vol spikes above medium-term
            vol_spike_scale = 1.0
            calm_boost = 1.0
            vol_compress_boost = 1.0
            if len(closes) >= VOL_LONG_LOOKBACK + 1:
                short_vol = self._calc_vol(closes, VOL_SHORT_LOOKBACK)
                long_vol = self._calc_vol(closes, VOL_LONG_LOOKBACK)
                if short_vol > long_vol * VOL_SPIKE_THRESHOLD:
                    vol_spike_scale = VOL_SPIKE_SCALE
                # Calm regime boost: when short vol is close to or below long vol, boost size
                vol_ratio_sl = max(0.5, min(2.0, short_vol / max(long_vol, 1e-10)))
                calm_boost = 1.0 + CALM_BOOST_MAX * max(0.0, 1.0 - vol_ratio_sl)
                # Vol compression boost: when short vol drops well below long vol,
                # a breakout is likely — boost size to capture the move
                if vol_ratio_sl < VOL_COMPRESS_THRESHOLD:
                    compress_strength = (VOL_COMPRESS_THRESHOLD - vol_ratio_sl) / VOL_COMPRESS_THRESHOLD
                    vol_compress_boost = 1.0 + VOL_COMPRESS_BOOST * compress_strength

            # Sideways regime boost: when long-term trend is weak, boost size
            # to capture more return in range-bound markets where risk is low
            sideways_trend_ratio = min(abs(ret_long) / SIDEWAYS_BOOST_DECAY, 1.0)
            sideways_trend_strength = sideways_trend_ratio ** 2  # squared for slower decay
            sideways_boost = 1.0 + SIDEWAYS_BOOST_MAX * (1.0 - sideways_trend_strength)

            # High-conviction vote bonus: boost sizing when 5+ out of 6 signals agree
            winning_votes = max(bull_votes, bear_votes)
            vote_boost = 1.0 + HIGH_VOTE_BOOST if winning_votes >= HIGH_VOTE_THRESHOLD else 1.0

            # Volume confirmation: boost size when recent volume is above longer-term average
            vol_confirm_mult = 1.0
            vol_ratio_raw = 1.0  # raw volume ratio for divergence detection
            volumes = bd.history["volume"].values
            if len(volumes) >= VOL_CONFIRM_BASE:
                recent_vol = np.mean(volumes[-VOL_CONFIRM_LOOKBACK:])
                base_vol = np.mean(volumes[-VOL_CONFIRM_BASE:])
                if base_vol > 0:
                    vol_ratio_raw = recent_vol / base_vol
                    # Above average: boost up to VOL_CONFIRM_BOOST
                    # Below average: reduce down to VOL_CONFIRM_FLOOR
                    vol_confirm_mult = max(VOL_CONFIRM_FLOOR, min(1.0 + VOL_CONFIRM_BOOST, vol_ratio_raw))

            # Multi-timeframe agreement: boost when vshort, short, and long all agree
            # Dampened by trend strength so it only helps in sideways (DD-safe) regimes
            mtf_agree_mult = 1.0
            if (ret_vshort > 0 and ret_short > 0 and ret_long > 0) or \
               (ret_vshort < 0 and ret_short < 0 and ret_long < 0):
                mtf_trend_str = min(abs(ret_long) / MTF_AGREE_TREND_DECAY, 1.0)
                mtf_agree_mult = 1.0 + MTF_AGREE_BOOST * (1.0 - mtf_trend_str)

            weight = SYMBOL_WEIGHTS.get(symbol, 0.33)
            if high_corr and symbol == "SOL":
                weight *= 0.5
            mom_strength = abs(ret_short) / dyn_threshold
            # In sideways markets, raise the floor so weak momentum isn't double-penalized
            sideways_strength = min(abs(ret_long) / STRENGTH_FLOOR_DECAY, 1.0)
            strength_floor = 0.6 + (STRENGTH_FLOOR_SIDEWAYS - 0.6) * (1.0 - sideways_strength)
            strength_scale = max(strength_floor, min(2.0, mom_strength))
            # Dampen cross-asset boost in strong trends (where DD is already near limit)
            cross_trend_strength = min(abs(ret_long) / CROSS_ASSET_TREND_DECAY, 1.0)
            dampened_cross_agree = 1.0 + (cross_asset_agree - 1.0) * (1.0 - cross_trend_strength)
            combined_mult = vol_scale * vol_spike_scale * strength_scale * calm_boost * sideways_boost * dampened_cross_agree * vote_boost * vol_compress_boost * vol_confirm_mult * mtf_agree_mult
            # Adaptive cap: allow more stacking in low-vol (sideways) regimes
            # where DD headroom exists, tighter in high-vol regimes to protect DD
            if vol_ratio < MAX_COMBINED_LOW_VOL_THRESHOLD:
                adaptive_cap = MAX_COMBINED_MULT_LOW_VOL
            elif vol_ratio > MAX_COMBINED_VOL_THRESHOLD:
                adaptive_cap = MAX_COMBINED_MULT_HIGH_VOL
            else:
                # Linear interpolation between low-vol and base caps
                blend = (vol_ratio - MAX_COMBINED_LOW_VOL_THRESHOLD) / (MAX_COMBINED_VOL_THRESHOLD - MAX_COMBINED_LOW_VOL_THRESHOLD)
                adaptive_cap = MAX_COMBINED_MULT_LOW_VOL + (MAX_COMBINED_MULT - MAX_COMBINED_MULT_LOW_VOL) * blend
            combined_mult = min(combined_mult, adaptive_cap)
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
                    # Mean-reversion entries in sideways markets
                    elif abs(ret_long) < MEANREV_TREND_THRESHOLD:
                        mr_size = size * MEANREV_SIZE_SCALE
                        if rsi < MEANREV_RSI_OVERSOLD:
                            target = mr_size
                            self.pyramided[symbol] = False
                        elif rsi > MEANREV_RSI_OVERBOUGHT:
                            target = -mr_size
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
                    # Momentum deceleration exit: if in profit and very-short-term
                    # momentum reverses, exit before trailing stop catches up
                    # Adaptive: wider threshold in trends (hold winners), tighter in sideways
                    elif pnl > 0:
                        decel_trend_str = min(abs(ret_long) / DECEL_TREND_DECAY, 1.0)
                        decel_mult = DECEL_MULT_BASE + (DECEL_MULT_TREND - DECEL_MULT_BASE) * decel_trend_str
                        # Volume-price divergence: tighten decel when volume is fading
                        if vol_ratio_raw < VOL_DIVERGENCE_THRESHOLD:
                            decel_mult *= VOL_DIVERGENCE_DECEL_MULT
                        # Small-winner widening: give marginal winners room to develop
                        if pnl < PROFIT_SMALL_THRESHOLD:
                            decel_mult *= PROFIT_SMALL_DECEL_WIDEN
                        # Profit-scaled tightening: lock in larger profits earlier
                        elif pnl > PROFIT_DECEL_THRESHOLD:
                            profit_excess = pnl - PROFIT_DECEL_THRESHOLD
                            decel_mult /= (1.0 + profit_excess * PROFIT_DECEL_SCALE)
                        # Funding crowding: tighten decel when position is on crowded side
                        if len(funding_rates) >= FUNDING_LOOKBACK:
                            current_funding = funding_rates[-1]
                            funding_pctile = np.mean(funding_rates <= current_funding)
                            if current_pos > 0 and funding_pctile > FUNDING_EXTREME_PERCENTILE:
                                decel_mult *= FUNDING_EXTREME_DECEL_MULT
                            elif current_pos < 0 and funding_pctile < (1.0 - FUNDING_EXTREME_PERCENTILE):
                                decel_mult *= FUNDING_EXTREME_DECEL_MULT
                        decel_threshold = dyn_threshold * decel_mult
                        if current_pos > 0 and ret_vshort < -decel_threshold:
                            target = 0.0
                        elif current_pos < 0 and ret_vshort > decel_threshold:
                            target = 0.0

                # Continuous vol-adaptive RSI exit: tighter in high vol, wider in sideways
                vol_exit_blend = max(0.0, min(1.0, (vol_ratio - RSI_EXIT_VOL_LOW) / (RSI_EXIT_VOL_HIGH - RSI_EXIT_VOL_LOW)))
                # Trend-adaptive widening: in sideways markets, widen OB/OS to hold winners longer
                trend_exit_strength = min(abs(ret_long) / RSI_EXIT_TREND_DECAY, 1.0)
                sideways_ob_widen = (RSI_OB_WIDE - RSI_OVERBOUGHT) * (1.0 - trend_exit_strength)
                sideways_os_widen = (RSI_OVERSOLD - RSI_OS_WIDE) * (1.0 - trend_exit_strength)
                base_ob = RSI_OVERBOUGHT + sideways_ob_widen
                base_os = RSI_OVERSOLD + sideways_os_widen
                effective_ob = base_ob - (base_ob - RSI_OB_TIGHT) * vol_exit_blend
                effective_os = base_os + (RSI_OS_TIGHT - base_os) * vol_exit_blend
                if current_pos > 0 and rsi > effective_ob:
                    target = 0.0
                elif current_pos < 0 and rsi < effective_os:
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
