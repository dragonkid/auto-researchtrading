import numpy as np
from prepare import Signal, PortfolioState, BarData


def _fast_slope(y):
    """OLS slope for y against 0..len(y)-1. ~50x faster than scipy linregress."""
    n = len(y)
    x_mean = (n - 1) / 2.0
    x_var = (n * n - 1) / 12.0  # var of 0..n-1
    y_mean = y.mean()
    slope = ((np.arange(n) - x_mean) * (y - y_mean)).sum() / (n * x_var)
    return slope


def _fast_r2(y):
    """Coefficient of determination (R^2) of the OLS linear fit of y vs 0..len(y)-1.
    Equals squared Pearson correlation between y and a straight line — measures how
    LINEAR (clean-trend) the series is, in [0, 1], independent of trend DIRECTION and
    SLOPE magnitude. Pure shape statistic; no zero-crossing (always >= 0)."""
    n = len(y)
    x = np.arange(n)
    x_mean = (n - 1) / 2.0
    y_mean = y.mean()
    xd = x - x_mean
    yd = y - y_mean
    cov = (xd * yd).sum()
    vx = (xd * xd).sum()
    vy = (yd * yd).sum()
    return (cov * cov) / max(vx * vy, 1e-20)

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]

# Momentum windows
MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 10
SHORT_WINDOW = 8
LONG_WINDOW = 20
VLONG_WINDOW = 96  # multi-day (~4d) trend-context horizon for counter-trend sizing

# EMA parameters
EMA_FAST = 3
EMA_SLOW = 21
EMA_SLOPE_PERIOD = 22
EMA_SLOPE_LOOKBACK = 3

# MACD parameters
MACD_FAST = 8
MACD_SLOW = 16
MACD_SIGNAL = 8  # widened from 4->6->7->8 to smooth MACD histogram further

# Linear regression
LINREG_PERIOD = 16

# Volatility parameters
VOL_LOOKBACK = 24
VOL_SHORT_LOOKBACK = 12
VOL_LONG_LOOKBACK = 36
TARGET_VOL = 0.015

# Entry threshold
BASE_THRESHOLD = 0.005
DYN_THRESHOLD_FLOOR = 0.00475
DYN_THRESHOLD_CEIL = 0.012
TREND_THRESHOLD_SCALE = 0.25       # max threshold reduction in trends (reduced from 0.32 for wider buffer in rally)
TREND_THRESHOLD_DECAY = 0.14       # abs(ret_long) at which reduction saturates

# RSI voter
RSI_TREND_BIAS = 2.0
RSI_TREND_BIAS_DECAY = 0.10

# Exit parameters (momentum-decay + slope + peak-profit + stop-loss)
HOLD_DECAY_START = 6   # bars after which exit pressure begins
HOLD_DECAY_RATE = 0.25  # exit pressure per bar beyond start (0.25 = exit at bar 10 with no momentum)
MOMENTUM_HOLD_BONUS = 2  # max extra bars when slope strongly agrees (conservative cap)
STOP_LOSS_PCT = -0.024
PEAK_PROFIT_MIN_BASE = 0.025
PEAK_PROFIT_GIVEBACK = 0.22

# Sizing multipliers
BASE_POSITION_SIZE = 0.065
CALM_BOOST_MAX = 0.8
SIDEWAYS_BOOST_MAX = 0.50
CROSS_ASSET_FIXED_BOOST = 0.15
HIGH_VOTE_BOOST_MULT = 1.20
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
TREND_GATE_DEADZONE = 0.018
# Vote / cooldown (6 voters, soft tanh contributions)
# Strong-consensus weighted sum: replaces hard count of voters above STRONG_CONF
# with sum of (conf-0.5)*2 for conf>0.5, weighted by margin. Removes noise boundary at 0.65.
STRONG_WEIGHT_MIN = 1.75  # required sum of margin-above-0.5 voter contributions (scaled for 7 voters)
MIN_VOTES = 2.92  # scaled for 7 voters
FLIP_MIN_VOTES = 2.80  # scaled for 7 voters
COOLDOWN_BARS = 1
COOLDOWN_TREND_DECAY = 0.06


def ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

# Position accumulation (build position over bars)
# Architectural: vol-conditioned initial commit. Base 0.43, modulated by vol_ratio
# via tanh: low vol -> larger initial frac (~0.50, capture rally/sideways momentum
# with less first-bar lag); high vol -> smaller initial frac (~0.36, less DD on
# wrong-side first bar in crash). Continuous, bounded ~[0.36, 0.50].
ENTRY_INITIAL_FRAC_BASE = 0.43
ENTRY_INITIAL_FRAC_VOL_AMP = 0.07
ENTRY_INITIAL_FRAC = 0.43  # retained for scale-in start anchor + flip-fraction path
ENTRY_FULL_BARS = 3  # bars to reach full position (linear scale-in over 3 bars)


class Strategy:
    def __init__(self):
        self.entry_prices, self.exit_bar, self.peak_pnl, self.entry_bar = {}, {}, {}, {}
        # Maximum adverse excursion (MAE): per-symbol low-water mark of pos_pnl since entry.
        # Used by adverse-recovery exit pressure (architectural): when current pos_pnl
        # has recovered from MAE but still in modest loss, position is "barely surviving"
        # — lock the recovery before another adverse leg. Distinct from peak_pnl (high-water).
        self._mae = {}
        # Per-symbol last-exit PnL outcome (loss-only cooldown stretch).
        self._last_exit_pnl = {}
        self.bar_count = 0
        self.smoothed_trend = {}
        # Two prior pnl bars for confirmed-peak gate (need 2 rising bars to update).
        self._smoothed_pnl = {}
        # Architectural: per-symbol recent voter strong-sum history (3-bar rolling).
        # Used by entry-persistence gate to require 2 bars of sustained conviction.
        self._recent_strongs = {}
        # Architectural: per-symbol per-voter directional history (8-bar rolling).
        # Used to compute per-voter directional persistence (fraction of last
        # K bars where voter signal sign matched). High-persistence voters
        # are weighted higher in strong-sum aggregation; flip-prone voters
        # are downweighted. New time-varying voter weighting based on each
        # voter's own track record.
        self._voter_sign_history = {}
        # Architectural: per-symbol entry-bar history (rolling list of bar_count
        # values at which entries opened). Used by trade-frequency self-regulator
        # to raise the strong-sum admission threshold when recent entry rate is
        # high — addresses turnover as a cost driver via direct feedback on the
        # entry decision boundary.
        self._entry_bar_history = {}
        # Branch step 4: per-symbol CUMULATIVE-max churn (int) for the persistent
        # churn gate on the low-churn coarse grid — once a symbol bursts (len(_eh)>=3),
        # the grid turns off permanently for it.
        self._churn_hist = {}
        self._peak_equity = 0.0  # portfolio-DD circuit-breaker on entry size

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1
        self._peak_equity = max(self._peak_equity, equity)
        _port_dd_atten = 1.0 - 1.0 * max(0.0, np.tanh(max(0.0, 1.0 - equity / max(self._peak_equity, 1e-10)) / 0.008))

        # Architectural: cross-asset consensus (NEW data source, orthogonal to all 7
        # single-symbol voters which each read only their own symbol). Under the noise
        # test perturbation is ~85% correlated across symbols + ~15% idiosyncratic per
        # symbol; the idiosyncratic component is what flips a single symbol's marginal
        # entries. The cross-asset directional agreement (mean tanh(ret_long/0.04) over
        # the active symbols) is dominated by the CORRELATED market-wide component ->
        # the idiosyncratic per-symbol noise cancels in the mean, making it a noise-
        # robust market-state signal. Used below as a direction-aware first-bar entry-
        # size attenuator: entries ALIGNED with the market-wide trend keep full size;
        # entries against / orthogonal to a weak-consensus (mixed) market are
        # idiosyncratic and noise-fragile, shrunk up to 30%. Computed once per bar
        # (pre-pass). Distinct from the own-symbol counter-trend attenuators (those use
        # THIS symbol's ret_long; this uses the agreement of the OTHER symbols).
        _xa_dirs = []
        for _xs in ACTIVE_SYMBOLS:
            if _xs in bar_data:
                _xh = bar_data[_xs].history["close"].values
                if len(_xh) >= LONG_WINDOW + 1:
                    _xa_dirs.append(np.tanh(((_xh[-1] - _xh[-LONG_WINDOW]) / _xh[-LONG_WINDOW]) / 0.04))
        _xa_mean = float(np.mean(_xa_dirs)) if _xa_dirs else 0.0

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close
            realized_vol = max(np.std(np.diff(np.log(closes[-VOL_LOOKBACK - 1:-1]))), 1e-6)
            # Architectural: per-symbol adaptive vol baseline. Replace constant TARGET_VOL
            # with long-window (200-bar) realized vol blended with TARGET_VOL anchor at
            # 0.5 weight. Long-window vol is each symbol's structural baseline (BTC ~0.012,
            # SOL ~0.022); blending with anchor preserves global scaling but reduces SOL's
            # bias toward "always-elevated" vol_ratio. New cross-bar data dependency on
            # 200-bar log-return std per symbol; smooth (no boundary), continuous.
            _long_n = min(200, len(closes) - 1)
            _baseline_vol = max(np.std(np.diff(np.log(closes[-_long_n - 1:-1]))), 1e-6)
            _target_vol_dyn = 0.7 * TARGET_VOL + 0.3 * _baseline_vol
            vol_ratio = realized_vol / _target_vol_dyn

            # Vol-adaptive smoothing: more in calm (span~3), less in choppy (span~2)
            # vol_ratio < 0.7 (calm): alpha=0.5 (span=3); vol_ratio > 1.2 (choppy): alpha=0.67 (span=2)
            _smooth_alpha = 0.5 + 0.17 * max(0.0, min(1.0, (vol_ratio - 0.7) / 0.5))
            smoothed_closes = np.empty_like(closes, dtype=float)
            smoothed_closes[0] = closes[0]
            for _si in range(1, len(closes)):
                smoothed_closes[_si] = _smooth_alpha * closes[_si] + (1 - _smooth_alpha) * smoothed_closes[_si - 1]
            # Architectural: ATR-anchored base threshold. Replaces global BASE_THRESHOLD
            # (constant 0.005) with per-symbol ATR-derived threshold. SOL (higher ATR)
            # needs larger move to trigger entry, BTC (lower ATR) smaller — structural
            # noise-floor scaling. ATR(14) computed on history high/low/prev-close.
            _atr_high_e = bd.history["high"].values[-14:]
            _atr_low_e = bd.history["low"].values[-14:]
            _atr_close_e = closes[-15:-1]
            _tr_e = np.maximum(_atr_high_e - _atr_low_e, np.maximum(np.abs(_atr_high_e - _atr_close_e), np.abs(_atr_low_e - _atr_close_e)))
            _atr_pct_e = np.mean(_tr_e) / mid
            # Anchor: 0.42 * ATR_pct, clamped to [0.0035, 0.008] keeps within original range
            _base_thresh_dyn = max(0.0040, min(0.0080, 0.45 * _atr_pct_e))
            dyn_threshold = _base_thresh_dyn * (0.10 + vol_ratio * 0.90) ** 0.85
            dyn_threshold = max(DYN_THRESHOLD_FLOOR, min(DYN_THRESHOLD_CEIL, dyn_threshold))

            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            # Architectural: multi-day (~96-bar) trend context, a SLOWER timescale than
            # the 20-bar ret_long. In a grinding rally the 20-bar window frequently
            # shows local negative returns (multi-hour pullbacks) while the multi-day
            # trend is strongly up, so a pullback short looks only mildly counter-trend
            # at 20 bars but is catastrophically counter-trend at the multi-day scale.
            # Exp1 this session proved this insight preserves rally raw (counter-trend
            # shorts ARE noise) — but routing it through ADMISSION collapsed stability
            # (pass/fail boundary at ret_vlong=0). Here it feeds a continuous SIZE
            # attenuator instead (no decision boundary). New cross-timescale data dep.
            # Branch step 2: compute the multi-day trend as an OLS slope over the full
            # 96-bar window (not a 2-endpoint return). _fast_slope on log(HL2) uses ALL
            # 96 points, so each bar's AR(1) noise carries weight ~1/96 instead of the
            # full weight the two endpoints had in step 1 — the position-size wobble that
            # collapsed rally stability is averaged out. Convert the per-bar log slope to
            # an equivalent net window return (slope * n) so the existing tanh(.../0.06)
            # scale is preserved.
            _vlong_n = min(VLONG_WINDOW, len(closes) - 1)
            _hl2_vl = (bd.history["high"].values[-_vlong_n:] + bd.history["low"].values[-_vlong_n:]) / 2.0
            ret_vlong = _fast_slope(np.log(_hl2_vl)) * _vlong_n
            dyn_threshold *= 1.0 - TREND_THRESHOLD_SCALE * (1.0 - min(abs(ret_long) / TREND_THRESHOLD_DECAY, 1.0) ** 0.85)

            _lr_slope = _fast_slope(np.log((bd.history["high"].values[-LINREG_PERIOD:] + bd.history["low"].values[-LINREG_PERIOD:]) / 2.0))

            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))))

            # 5-bar median signal (maximum noise immunity, returns sacrificed for stability)
            _med_ref_med = np.median(smoothed_closes[-adaptive_med - 2: -adaptive_med + 3])
            ret_short = (smoothed_closes[-1] - _med_ref_med) / _med_ref_med

            _ef, _es = ema(closes[-(EMA_SLOW+10):], EMA_FAST)[-1], ema(closes[-(EMA_SLOW+10):], EMA_SLOW)[-1]
            _ret_long_lagged = (closes[-2] - closes[-LONG_WINDOW - 1]) / closes[-LONG_WINDOW - 1]
            rsi_trend_str = min(abs(_ret_long_lagged) / RSI_TREND_BIAS_DECAY, 1.0)
            _rd = np.diff(closes[-(int(round(6 + 2 * rsi_trend_str)) + 1):])
            rsi = 100 - 100 / (1 + np.mean(np.maximum(_rd, 0)) / max(np.mean(np.maximum(-_rd, 0)), 1e-10))
            _ml = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST) - ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
            _ea = ema(closes[-(EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5):], EMA_SLOPE_PERIOD)

            # 7 voters with smooth tanh contribution: hard binary except at threshold boundary.
            # Each voter contribution = 0.5 * (1 + tanh((signal - thresh) * sharpness)) so it behaves like a binary
            # 0/1 except in a narrow band around the threshold where it transitions smoothly. Keeps original
            # vote-count semantics while reducing flip-rate near boundaries.
            _rsi_thresh = 50 + RSI_TREND_BIAS * rsi_trend_str * (-1.0 if ret_long > 0 else 1.0)
            _macd_diff = (_ml[-1] - ema(_ml, MACD_SIGNAL)[-1]) / mid
            _ea_slope = (_ea[-1] - _ea[-EMA_SLOPE_LOOKBACK]) / _ea[-EMA_SLOPE_LOOKBACK]
            # Architectural: 7th voter — volume-weighted price deviation.
            # Volume data is orthogonal to all 6 existing voters (which use price-derived
            # series only). Compute 12-bar VWAP using (high+low+close)/3 typical price
            # weighted by volume; voter signals when current close deviates upward (bull)
            # from VWAP. Captures genuine volume-confirmed directional pressure independent
            # of moving averages, RSI, MACD, slope. New data dependency on volume * price.
            _vwap_n = 12
            _vol_arr = bd.history["volume"].values[-_vwap_n:]
            _tp_arr = (bd.history["high"].values[-_vwap_n:] + bd.history["low"].values[-_vwap_n:] + closes[-_vwap_n:]) / 3.0
            _vwap = (_tp_arr * _vol_arr).sum() / max(_vol_arr.sum(), 1e-10)
            _vwap_dev = (mid - _vwap) / mid  # positive = above VWAP, bull bias
            _voter_signals_bull = [
                (ret_short - dyn_threshold) / max(dyn_threshold * 0.20, 1e-6),
                (_ef - _es) / (mid * 0.0008),
                (rsi - _rsi_thresh) / 4.0,
                (_macd_diff - 0.0003) / 0.00012,
                (_lr_slope - 0.00015) / 0.00010,
                (_ea_slope - 0.0005) / 0.00025,
                _vwap_dev / 0.0030,  # 7th voter: VWAP deviation, halved sharpness (was 0.0015) for softer tanh, less noise in chop
            ]
            # Voter contribution clipping: each conf bounded to [0.1, 0.9] instead of (0,1).
            # Prevents any single voter from dominating the strong-sum under noise saturation.
            # A noise-flipped voter shifts _bull_strong by at most ~0.8 (was ~2.0).
            _bull_confs = [0.1 + 0.8 * 0.5 * (1.0 + np.tanh(s)) for s in _voter_signals_bull]
            _bear_confs = [0.1 + 0.8 * 0.5 * (1.0 + np.tanh(-s)) for s in _voter_signals_bull]
            bull_votes = sum(_bull_confs)
            bear_votes = sum(_bear_confs)
            # Quintic-ramp strong-sum with per-voter noise-sensitivity weights.
            # Voter ordering: [ret_short, EMA_cross, RSI, MACD, slope_16, EMA_slope].
            # Weights inverse to estimated noise sensitivity (sum=6.0, preserves scale).
            # Architectural: trend-strength weight redistribution. In strong trends
            # (high abs(ret_long)), shift weight from mean-reverting voters
            # (RSI=idx2, MACD=idx3) to trend-confirming voters (EMA_cross=idx1,
            # EMA_slope=idx5). In chop, weights stay near base. Continuous via
            # tanh on abs(ret_long)/0.04. New cross-timescale data dependency:
            # voter aggregation function depends on long-window return.
            _trend_strength_w = max(0.0, np.tanh(abs(ret_long) / 0.04))  # in [0, ~1]
            _wt_shift = 0.20 * _trend_strength_w
            # VWAP voter chop-dampener: in low-trend (chop), volume-weighted price
            # is dominated by recent action which oscillates with chop noise; in
            # trends, VWAP captures genuine directional pressure. Scale VWAP voter
            # weight from 0.55 (deep chop) up to 1.05 (strong trend). Continuous
            # via _trend_strength_w. Preserves the rally/crash gain while reducing
            # the sideways regression introduced by full VWAP weight.
            _vwap_wt = 0.55 + 0.50 * _trend_strength_w  # in [0.55, ~1.05]
            _base_weights = (0.7, 1.25 + _wt_shift, 1.10 - _wt_shift, 1.00 - _wt_shift, 0.85, 1.10 + _wt_shift, _vwap_wt)
            # Architectural: per-voter directional persistence weighting.
            # Track each voter's signal sign over last 8 bars. Persistence =
            # |sum(signs)| / count → 1.0 if voter held one direction continuously,
            # 0.0 if voter flipped maximally. Multiply base weight by
            # (0.7 + 0.6 * persistence) so consistent voters get up to 1.3x weight,
            # flip-prone voters get down to 0.7x. Smooth (continuous over time as
            # history rolls forward). New per-symbol per-voter state dependency:
            # voter aggregation weight depends on each voter's recent flip-rate.
            # Architectural: magnitude-weighted persistence replacing binary-sign aggregation.
            # Old: binary sign per bar → persistence = |sum(signs)|/N counts a near-zero
            # signal flip identically to a far-from-zero flip. New: store raw signal values
            # → persistence = |sum(signal)| / sum(|signal|) gives weight to magnitude. A voter
            # that hovers near zero contributes equally to numerator and denominator (low
            # persistence influence); a voter with strong directional history dominates.
            # Reduces noise from near-zero voter flips downweighting active voters.
            _sig_hist = self._voter_sign_history.get(symbol, [])
            _sig_hist.append(tuple(_voter_signals_bull))
            if len(_sig_hist) > 8:
                _sig_hist = _sig_hist[-8:]
            self._voter_sign_history[symbol] = _sig_hist
            if len(_sig_hist) >= 4:
                _arr = np.array(_sig_hist)  # (K, 7)
                _num = np.abs(_arr.sum(axis=0))
                _den = np.maximum(np.abs(_arr).sum(axis=0), 1e-10)
                _persistence = _num / _den  # in [0, 1]
                _persistence_mult = 0.7 + 0.6 * _persistence  # in [0.7, 1.3]
            else:
                _persistence_mult = np.ones(7)
            _voter_weights = tuple(bw * pm for bw, pm in zip(_base_weights, _persistence_mult))
            # Architectural simplification: removed volume-weighted voter aggregation
            # amplifier (_vol_amp_raw, _bull_amp, _bear_amp). Trend-aligned one-sided
            # amplifier composed three multiplicative gates (chop neutralization
            # _trend_strength_w × directional tanh × volume-deviation tanh), max ±15%.
            # With three gates each requiring near-saturated input, the amplifier is
            # dead-code-adjacent: chop zeros the trend gate, weak trend halves it, and
            # counter-trend side zeros the directional gate. The remaining sliver of
            # activation overlaps with _persistence_mult (per-voter sustained-conviction
            # tracking) and _wt_shift trend-confirming voter weight redistribution.
            # Code-structure removal: 14 lines + 3 cross-bar volume reads.
            _bull_strong = sum(max(0.0, (c - 0.5) ** 5 * 97.66) * w for c, w in zip(_bull_confs, _voter_weights))
            _bear_strong = sum(max(0.0, (c - 0.5) ** 5 * 97.66) * w for c, w in zip(_bear_confs, _voter_weights))
            # Architectural: VWAP post-admission SIZE multiplier. VWAP semantically
            # Architectural: maintain rolling 3-bar history of strong-sums per symbol.
            # Used to gate flips on sustained conviction (filters single-bar noise spikes).
            _hist = self._recent_strongs.get(symbol, [])
            _hist.append((_bull_strong, _bear_strong))
            if len(_hist) > 3:
                _hist = _hist[-3:]
            self._recent_strongs[symbol] = _hist
            # Sideways-aware strong-sum threshold: tighten in low-trend regimes to filter
            # noisy entries; relax in trends. Uses continuous rsi_trend_str interpolation.
            _strong_min = STRONG_WEIGHT_MIN + 0.20 * (1.0 - rsi_trend_str)

            # Architectural: trade-frequency self-regulator. Per-symbol rolling
            # entry-bar history over a 30-bar window. When recent entry density
            # exceeds a threshold (>=2 in 30 bars), raise admission proportionally.
            # Smooth via tanh; max factor 1.20.
            _eh = self._entry_bar_history.setdefault(symbol, [])
            while _eh and self.bar_count - _eh[0] > 30:
                _eh.pop(0)
            _freq_factor = 1.0 + 0.20 * max(0.0, np.tanh((len(_eh) - 1.5) / 2.0))
            # Architectural simplification: removed _portfolio_freq_factor (cross-symbol
            # entry frequency regulator). Per-symbol _freq_factor already captures
            # local churn at each symbol — the portfolio-level addition at >=5 entries/30bars
            # composes multiplicatively, double-counting since correlated regimes (crash)
            # naturally produce multi-symbol entries in tandem. Removing eliminates the
            # +15% admission cost during legitimate correlated entry pile-ups (e.g. multi-
            # symbol crash legs). Per-symbol regulator alone provides sufficient churn
            # protection. Code-structure removal: 7 lines + state tracking eliminated.
            # Architectural: bull-only trend-aligned admission relaxation. Bear
            # admission unchanged (crash bear admission is sensitive to dead-cat
            # bounce low-quality entries regardless of magnitude). 20-bar ret_long
            # for trend, smooth tanh, -0.10 max relaxation on bull strong_min in
            # uptrend only. New cross-timescale data dep at admission boundary,
            # one-sided multi-variable structural change.
            # Architectural: counter-trend admission tightening. Bull-only relaxation
            # (afa6281) already reduces bull strong_min in uptrend. This adds the
            # SYMMETRIC COUNTERPART: tighten bull admission in downtrends (crash dead-cat
            # bounce noise), tighten bear admission in uptrends (rally pullback bear noise).
            # Continuous tanh on long-window trend direction, max 15% threshold increase.
            # New cross-component data dep: admission threshold depends on trend direction
            # for counter-trend side. Multi-variable: both bull and bear strong_min modified.
            _bull_strong_min = _strong_min * _freq_factor * (1.0 - 0.10 * max(0.0, np.tanh(ret_long / 0.04))) * (1.0 + 0.15 * max(0.0, np.tanh(-ret_long / 0.04)))
            _bear_strong_min = _strong_min * _freq_factor * (1.0 + 0.15 * max(0.0, np.tanh(ret_long / 0.04)))
            # Conviction margins (relative excess of strong-sum over its admission threshold).
            # Computed at top-level so they are available to both entry and flip paths.
            _bull_margin = (_bull_strong - _bull_strong_min) / max(_bull_strong_min, 1e-6)
            _bear_margin = (_bear_strong - _bear_strong_min) / max(_bear_strong_min, 1e-6)

            cooldown_trend_strength = min(abs(ret_long) / COOLDOWN_TREND_DECAY, 1.0)
            trend_avg = (TREND_GATE_MED_WEIGHT_SIDEWAYS - (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength) * ((closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]) + ((1.0 - TREND_GATE_MED_WEIGHT_SIDEWAYS) + (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength) * ret_long
            # Use trend_avg directly (stateless) — EMA smoothing amplifies noise via state propagation
            self.smoothed_trend[symbol] = trend_avg

            # Smooth cooldown_factor (tanh decay over trend-scaled window) +
            # loss-only outcome-conditioned stretch & first-bar size attenuator.
            _bars_since_exit = self.bar_count - self.exit_bar.get(symbol, -999)
            _loss_only = max(0.0, -np.tanh(self._last_exit_pnl.get(symbol, 0.0) / abs(STOP_LOSS_PCT)))
            _cd_window = max(0.6, 1.5 - 0.9 * cooldown_trend_strength) * (1.0 + 0.6 * _loss_only)
            _cooldown_factor = max(0.0, min(1.0, np.tanh(_bars_since_exit / _cd_window)))
            _outcome_size_mult = 1.0 - 0.45 * max(0.0, 1.0 - _bars_since_exit / 8.0) * _loss_only
            in_cooldown = False

            calm_boost = 1.0 + CALM_BOOST_MAX * max(0.0, 1.0 - max(0.5, max(np.std(np.diff(np.log(closes[-VOL_SHORT_LOOKBACK - 1:-1]))), 1e-6) / max(np.std(np.diff(np.log(closes[-VOL_LONG_LOOKBACK - 1:-1]))), 1e-6))) ** 0.85 * min(1.0, max(0.0, (1.7 - vol_ratio) / 0.4))

            sideways_boost = 1.0 + SIDEWAYS_BOOST_MAX * (1.0 - rsi_trend_str ** 1.45)

            strength_scale = max(0.6 + (STRENGTH_FLOOR_SIDEWAYS - 0.6) * (1.0 - min(abs(ret_long) / STRENGTH_FLOOR_DECAY, 1.0)), min(2.0, (abs(ret_short) / dyn_threshold) ** 0.85))
            # Architectural simplification: removed HIGH_VOTE_BOOST_MULT (constant 1.20).
            # Always-on positive size bias is redundant: strong-sum entry gate already
            # filters by voter conviction, and the conviction-margin first-bar adjuster
            # (_entry_conv_adj) provides conviction-aware sizing. The fixed 1.20x
            # multiplier was load-bearing only as raw size scale, not as a conviction signal.
            # Architectural simplification: removed vol_confirm_mult (volume ratio bounded to
            # [0.98, 1.10]). Effect was structurally tiny (max 12% range, floor near 1.0) —
            # the mechanism could only boost size, never meaningfully cut it. The combined_mult
            # already has multiple vol-conditioning channels (vol_ratio direct, calm_boost,
            # sideways_boost) — adding a near-constant volume multiplier added LOC without
            # orthogonal signal. Removing eliminates redundant near-constant size scaling.
            combined_mult = max(0.3, min(2.5, (TARGET_VOL / realized_vol) ** 0.85)) * strength_scale * calm_boost * sideways_boost * (1.0 + CROSS_ASSET_FIXED_BOOST * (1.0 - cooldown_trend_strength))
            # Architectural simplification: removed _xa_boost (post-cap chop-only +8% boost).
            # Redundant with sideways_boost (max +50% in chop) and CROSS_ASSET_FIXED_BOOST
            # (already in combined_mult, max +15% in chop). Three chop-amplifying multipliers
            # double-count the same regime signal; removing the smallest eliminates the
            # most marginal contributor. Continuous removal (factor was always >= 1.0).
            # Architectural: smooth ONLY the upper hard ternary at vol_ratio=1.2.
            # Keep the original linear 0.6->1.2 interpolation (load-bearing for rally
            # dwell point). Replace discontinuity at vol_ratio=1.2 with smooth blend
            # to MAX_COMBINED_MULT_HIGH_VOL=2.5 over [1.2, 1.5].
            _cap_low_t = max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_LOW) / (MAX_COMBINED_VOL_HIGH - MAX_COMBINED_VOL_LOW)))
            _cap_base = MAX_COMBINED_MULT_LOW_VOL - 3.0 * _cap_low_t  # original linear interp
            # Smooth blend at upper boundary (vol_ratio > 1.2 was hard switch to 2.5)
            _cap_high_t = max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_HIGH) / 0.3))
            _cap_high_smooth = _cap_high_t * _cap_high_t * (3.0 - 2.0 * _cap_high_t)
            _cap_base = _cap_base * (1.0 - _cap_high_smooth) + MAX_COMBINED_MULT_HIGH_VOL * _cap_high_smooth
            combined_mult = min(combined_mult, _cap_base + MAX_COMBINED_TREND_BOOST * (1.0 - rsi_trend_str ** 0.85))
            size = equity * BASE_POSITION_SIZE * combined_mult

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            # Architectural: vol-conditioned initial commit fraction. Continuous tanh
            # mapping vol_ratio (band ~0.5..1.5 -> ~0.50..0.36). Decouples first-bar
            # exposure from a constant in regimes where initial-bar noise risk varies.
            _entry_frac_dyn = ENTRY_INITIAL_FRAC_BASE - ENTRY_INITIAL_FRAC_VOL_AMP * np.tanh((vol_ratio - 1.0) / 0.4)
            # Architectural: Kaufman efficiency ratio gate on initial commitment.
            # ER = |close[-1] - close[-N]| / sum(|close[i] - close[i-1]|), range [0,1].
            # High ER (>0.4) = price moved efficiently in one direction (signal-rich bars).
            # Low ER (<0.2) = path was choppy relative to net move (noise-rich, even if
            # net direction matches voters). Orthogonal to all current primitives:
            # vol_ratio measures magnitude, trend_avg measures net direction, slope measures
            # linear trajectory — ER measures path efficiency. Continuous tanh modulation
            # of _entry_frac_dyn: low ER attenuates, high ER amplifies (range -0.04..+0.04).
            # 12-bar window over smoothed_closes (already noise-attenuated for input parity).
            _er_window = 12
            _er_path = np.sum(np.abs(np.diff(smoothed_closes[-_er_window - 1:])))
            _er_net = abs(smoothed_closes[-1] - smoothed_closes[-_er_window - 1])
            _er = _er_net / max(_er_path, 1e-10)
            # One-sided deep-chop suppression: only fire on very low ER (<0.15),
            # smaller magnitude to avoid uniform size-attenuation across regimes.
            # tanh activates as ER drops below 0.15 toward 0; max attenuation -0.025.
            _er_adj = -0.025 * max(0.0, np.tanh((0.15 - _er) / 0.10))
            _entry_frac_dyn = min(0.55, _entry_frac_dyn + _er_adj)

            if current_pos == 0 and not in_cooldown:
                # Architectural simplification: removed Donchian range-position entry adj.
                # The 20-bar high/low range-position adjustment in [-0.04, +0.04] was a
                # small entry-side bias correlated with trend direction (price near 20-bar
                # high == uptrend, near low == downtrend) — overlapping with ret_long and
                # trend_avg which already gate entry. Removing eliminates a redundant
                # trend-correlated first-bar size modifier; saves cross-bar high/low
                # dependency and 5 lines.
                _range_bull_adj = 0.0
                _range_bear_adj = 0.0
                # Architectural: entry-persistence gate. Reuses the rolling _hist
                # (3-bar strong-sum history maintained for flip sustenance) to
                # require ENTRY-side conviction to be sustained over 2 bars before
                # admission. Filters single-bar noise spikes that currently drive
                # high turnover (~9k+ trades). Continuous: persistence factor uses
                # min over last 2 bars; gate fires when min >= sustain_factor *
                # _strong_min.
                # Architectural: TREND-aware persistence gate (new dependency).
                # In strong trends (high abs(ret_long)), a sudden conviction spike
                # is itself signal — relax persistence to admit fast entries.
                # In chop (low abs(ret_long)), keep strict persistence to filter
                # single-bar noise. Replaces vol-conditioning (which couples to
                # market vol regardless of direction) with trend-magnitude. The
                # vol gate kept as additive (high-vol crash gets some relaxation
                # via trend magnitude already, but vol-relaxation preserved as
                # protective in fast crashes). Continuous tanh on abs(ret_long).
                # New cross-timescale data dependency: entry gate strictness on
                # long-window trend strength.
                _trend_str_persist = max(0.0, np.tanh(abs(ret_long) / 0.05))  # [0,~1]
                _entry_persist_factor = 0.95 - 0.30 * _trend_str_persist  # 0.95 in chop, 0.65 in strong trend
                if len(_hist) >= 2:
                    _min_bull_2 = min(_hist[-2][0], _hist[-1][0])
                    _min_bear_2 = min(_hist[-2][1], _hist[-1][1])
                else:
                    _min_bull_2 = _bull_strong
                    _min_bear_2 = _bear_strong
                _bull_persist_ok = _min_bull_2 >= _entry_persist_factor * _bull_strong_min
                _bear_persist_ok = _min_bear_2 >= _entry_persist_factor * _bear_strong_min
                # Architectural simplification: removed _avg_signal bias from trend gate.
                # _avg_signal is the mean of the same 6 voter signals that drive _bull_strong/
                # _bear_strong (via _bull_confs/_bear_confs). Adding _avg_signal bias to the
                # trend gate created CORRELATED noise amplification at entry — both gates fire
                # on the same underlying noise. Using raw smoothed_trend decouples the trend
                # gate from the voter-signal subsystem; trend_avg derives from price-window
                # returns (orthogonal to per-bar voter signals).
                _trend_biased = self.smoothed_trend[symbol]
                # Architectural: replaced binary deadzone vote-tiebreak with continuous
                # strong-conviction admission. When _bull_strong significantly exceeds
                # _strong_min (margin = (strong - min) / min), the trend-sign requirement
                # softens proportionally: very strong conviction can override small-magnitude
                # wrong-sign trend. Smooth replacement for the binary deadzone clause —
                # gates on conviction magnitude rather than on absolute |_trend_biased|.
                # Architectural multi-variable: invert conviction-margin softening direction.
                # Old logic was dead-code: min(1.0, margin/0.3) capped softening to no-op for margin>=0.3,
                # AND a redundant safety floor clause matched the unsoftened deadzone — net effect
                # zero conviction-margin influence. New: high-conviction entries (margin>0.3) relax
                # the trend deadzone proportional to margin excess (up to 1.5x deadzone wider for
                # margin=0.6). Low/marginal conviction (margin<0.3) keeps strict floor. Multi-variable:
                # changes admission gate logic, removes redundant safety clause, makes conviction-
                # softening actually functional.
                _bull_relax = 1.0 + 0.50 * max(0.0, min(1.0, (_bull_margin - 0.3) / 0.3))
                _bear_relax = 1.0 + 0.50 * max(0.0, min(1.0, (_bear_margin - 0.3) / 0.3))
                _bull_admit = _trend_biased > -TREND_GATE_DEADZONE * _bull_relax
                _bear_admit = _trend_biased < TREND_GATE_DEADZONE * _bear_relax
                # Architectural simplification: removed redundant bull_votes>=MIN_VOTES count gate.
                # The strong-sum gate (_bull_strong >= _bull_strong_min) is highly correlated with the
                # count gate since both derive from the same _bull_confs values. Removing the count
                # gate eliminates correlated-noise amplification at the entry decision boundary
                # (one less hard gate on the same underlying signal). Strong-sum is the primary
                # discriminator (uses voter weights and quintic ramp); count is a coarser version.
                # Architectural: conviction-margin SIZE modulation on cold entry path.
                # Symmetric to flip-path _flip_conv_adj. When the strong-sum is well above
                # its admission threshold (high conviction entry), first-bar commitment
                # is larger; marginal entries (low or negative margin) get standard size.
                # One-sided positive: only positive margin amplifies, negative is treated
                # as zero (avoids cutting size on legitimate but marginal entries near
                # the gate boundary). New data dependency: first-bar size depends on
                # conviction margin for cold entries (was independent before).
                # Architectural: counter-trend SCALE-IN attenuator on cold entry path.
                # When a fresh entry is taken AGAINST the long-window trend (bear entry
                # in uptrend, bull entry in downtrend), attenuate first-bar position
                # size via a smooth tanh on the trend disagreement magnitude. Unlike
                # admission asymmetry (discarded — blocked legitimate pullback signals),
                # this admits the entry but with reduced commitment. Gated above
                # |ret_long|>0.03 to avoid firing in chop. Max attenuation 0.30 (counter-
                # trend entries take 0.70x size in strong trend). New cross-timescale
                # data dependency: cold-entry first-bar size depends on trend disagreement.
                _ct_gate = max(0.0, np.tanh((abs(ret_long) - 0.03) / 0.04))  # 0..1
                _bull_ct_atten = 1.0 - 0.30 * _ct_gate * max(0.0, np.tanh(-ret_long / 0.05))  # bull entry in downtrend
                _bear_ct_atten = 1.0 - 0.30 * _ct_gate * max(0.0, np.tanh(ret_long / 0.05))   # bear entry in uptrend
                # Architectural: multi-day counter-trend SIZE attenuator (layered on the
                # 20-bar term above). The 20-bar _ct_gate is ~0 during a rally pullback
                # (the local 20-bar return is flat/negative at the moment a pullback short
                # opens), so the existing term does NOT shrink rally's pullback shorts —
                # they are exactly the entries that lose (6/12 rally opens are shorts, WR
                # 66%). ret_vlong stays positive through pullbacks, so this term fires:
                # bear entries in a sustained multi-day uptrend, and bull entries in a
                # sustained multi-day downtrend (crash dead-cat-bounce longs), get up to
                # 0.40x size reduction. Continuous (no decision boundary), multiplies into
                # first-bar size alongside _bull_ct_atten / _bear_ct_atten.
                # Branch step 5: restore step-3's per-bar tanh churn gate (beat step-4's
                # cumulative hard gate: rally benefits from the ct-shrink during its QUIET
                # stretches between bursts — when len(_eh) drops back to <=1 — but is hurt
                # DURING bursts; the per-bar gate re-enables the shrink in quiet stretches,
                # the permanent-off cumulative gate does not). The 5e4c5d5c baseline keep
                # is a churn-gated order-emission grid that LIFTS rally stability; changing
                # first-bar entry SIZE shifts which lattice cell each rally resize lands in,
                # so gate on the COMPLEMENT of the noise-immune integer churn count:
                # _calm_ct ~1 at len(_eh)<=1 (bull/crash/sideways + rally quiet stretches),
                # fading to ~0 at len(_eh)>=3 (rally bursts).
                # Branch step 9: step-3's smooth per-bar churn gate + FAST-SATURATION
                # ret_vlong scale (0.06 -> 0.01), no deadzone (step-8's deadzone failed:
                # it put the STEEP tanh region INSIDE rally's operating range). Direct
                # sensitivity analysis: rally's counter-trend shorts operate at multi-day
                # uptrend ret_vlong ~ 0.02-0.04. The DAMAGE to rally stability is the
                # SENSITIVITY d(shrink)/d(ret_vlong) — how much the entry SIZE wobbles per
                # unit of AR(1)-perturbed trend. At scale 0.06 (step 3) that sensitivity is
                # ~5 across rally's range (size wobbles -> stability damage). At scale 0.01
                # the SAME range sits in the FLAT saturated tail of tanh: shrink is a
                # near-CONSTANT ~0.40x (sensitivity ~0.4, an order of magnitude lower) —
                # a large but NOISE-FREE size reduction. This shrinks counter-trend shorts
                # just as much as step 3 but stops their size from tracking noise, so rally
                # stability is no longer damaged while the bull gain (its shorts also past
                # the saturation knee) is preserved. New mechanism: near-binary saturated
                # ct-shrink profile (vs step-3's mid-slope linear region).
                _calm_ct = 1.0 - max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))  # per-bar: ~1 low churn, ~0 bursting
                _bull_ct_vlong = 1.0 - 0.40 * _calm_ct * max(0.0, np.tanh(-ret_vlong / 0.01))  # bull entry in multi-day downtrend
                _bear_ct_vlong = 1.0 - 0.40 * _calm_ct * max(0.0, np.tanh(ret_vlong / 0.01))   # bear entry in multi-day uptrend
                # Architectural: multi-window slope CONSENSUS GATE on first-bar SIZE.
                # Decision-architecture change: replace discrete 4-step map ((0.40,0.60,
                # 0.85,1.0) indexed by sign-agreement count) with continuous magnitude-
                # weighted alignment. Old map ignored slope MAGNITUDE — a barely-positive
                # 8-bar slope counted identically to a strongly-positive one, creating
                # boundary noise when small slopes near zero flip sign. New: alignment_w =
                # tanh(slope_w * pos_dir / scale_w) ∈ [-1, +1] per window, then average
                # the three. Attenuation = 0.40 + 0.60 * (avg_align + 1)/2, continuous in
                # [0.40, 1.00]. Boundary at slope=0 is smooth (tanh), not stepped. Scales
                # picked per-window to give similar saturation thresholds.
                _hl2_e = (bd.history["high"].values + bd.history["low"].values) / 2.0
                _slps = [_fast_slope(np.log(_hl2_e[-_w_e:])) for _w_e in (8, 16, 32)]
                _cons_scales = (0.0010, 0.0007, 0.0005)  # window-specific saturation
                _bull_align = sum(np.tanh(s / sc) for s, sc in zip(_slps, _cons_scales)) / 3.0
                _bear_align = sum(np.tanh(-s / sc) for s, sc in zip(_slps, _cons_scales)) / 3.0
                _bull_consensus_atten = 0.40 + 0.60 * (_bull_align + 1.0) / 2.0
                _bear_consensus_atten = 0.40 + 0.60 * (_bear_align + 1.0) / 2.0
                # Architectural: bilateral-conviction-quality entry size attenuator.
                # New cross-component data dep: own-side first-bar size depends on the
                # OPPOSITE side's strong-sum. When opp_strong is small relative to side_strong,
                # voters are decisively one-sided (high quality entry). When opp_strong is
                # close to side_strong (bilateral noise — voters split), entry quality is
                # low and size should be cut. Compute opp/own ratio (0..1+); attenuate via
                # tanh: 1.0x at ratio<=0.3, ramps down to 0.7x at ratio>=0.9. Independent
                # from entry-pass gate (entry still admitted at marginal quality, but at
                # smaller commitment). Mechanism: filters splitting-vote false entries that
                # the strong-sum gate alone passes when own-side just barely exceeds floor
                # while opp-side is also nearly at floor — exactly the noise-amplified
                # entries that lose most.
                _bull_opp_ratio = _bear_strong / max(_bull_strong, 1e-6)
                _bear_opp_ratio = _bull_strong / max(_bear_strong, 1e-6)
                _bull_quality_atten = 1.0 - 0.30 * max(0.0, min(1.0, np.tanh((_bull_opp_ratio - 0.3) / 0.4)))
                _bear_quality_atten = 1.0 - 0.30 * max(0.0, min(1.0, np.tanh((_bear_opp_ratio - 0.3) / 0.4)))
                # Architectural simplification: removed _vol_entry_atten (low-volume entry
                # size attenuator). The mechanism cut first-bar size by up to 30% on
                # low-volume bars. Redundant with: (a) persistence gate filtering weak
                # conviction entries, (b) consensus attenuator reducing size on
                # multi-window slope disagreement, (c) quality attenuator on bilateral
                # voter splits. Low-volume bars that pass all three gates are already
                # quality-filtered; additional volume-based attenuation is redundant.
                # Code-structure removal: -16 lines + -3 cross-bar volume reads.
                _vol_entry_atten = 1.0
                # Architectural: time-of-day session-quality entry size modulator.
                # Continuous cyclical feature: cos-cycle peaking at UTC 16 (US session
                # peak overlap), trough at UTC 04 (low Asia hour). _activity in [0, 1].
                # Maps to size multiplier in [0.85, 1.15] — entries during high-volume
                # hours get up to 15% larger commitment, low-volume hours up to 15% smaller.
                # NEW DATA SOURCE: bd.timestamp (UTC hour-of-day), orthogonal to all
                # within-bar/within-window primitives currently saturated. Smooth via
                # cos (no boundary). Applied only to first-bar entry size (does not
                # touch voters, exits, or scale-in).
                # NEW DATA SOURCE: timestamp-derived TOD+DOW+MOM compound activity.
                # TOD: cos cycle 24h peaking UTC 16. DOW: cos cycle 7d peaking Wed/Thu.
                # MOM: cos cycle ~30d peaking mid-month (day 15), trough around month-end/start.
                # Mid-month captures monthly options expiry + futures rollover concentration;
                # month-end carries window-dressing rebalance noise. All three compound
                # multiplicatively into _activity. Single _tod_atten var absorbs all three.
                # 5th cycle: semi-annual (~180d cos peak day 90 mid-half-year) — distinct
                # frequency from MOM (30d) and QUARTER (91d), captures half-year market
                # rhythm. Scaled 0.85-1.0 (smallest amplitude) to bound contribution since
                # contributions decrease with each cycle stacked (TOD +0.0001 -> DOW +0.0006
                # -> MOM +0.0003 -> QUARTER +0.0001). +0 LOC fused into single expression.
                _ts_h = bd.timestamp // 3600000
                _activity = 0.5 * (1.0 + np.cos(2.0 * np.pi * (_ts_h % 24 - 16.0) / 24.0)) * (0.6 + 0.4 * 0.5 * (1.0 + np.cos(2.0 * np.pi * ((_ts_h // 24 + 4) % 7 - 3.0) / 7.0))) * (0.7 + 0.3 * 0.5 * (1.0 + np.cos(2.0 * np.pi * ((_ts_h // 24) % 30 - 15.0) / 30.0))) * (0.8 + 0.2 * 0.5 * (1.0 + np.cos(2.0 * np.pi * ((_ts_h // 24) % 91 - 45.0) / 91.0))) * (0.85 + 0.15 * 0.5 * (1.0 + np.cos(2.0 * np.pi * ((_ts_h // 24) % 180 - 90.0) / 180.0))) * (0.9 + 0.1 * 0.5 * (1.0 + np.cos(2.0 * np.pi * ((_ts_h // 24) % 365 - 182.0) / 365.0)))
                _tod_atten = 0.85 + 0.30 * _activity
                # Architectural: conviction-margin first-bar SIZE attenuator (shrink,
                # don't block). Exp2 this session proved marginal-conviction entries
                # drive rally instability — but blocking them (raising admission)
                # collapsed raw (their alpha is real). Instead shrink them: scale
                # first-bar size continuously by how far the strong-sum sits ABOVE its
                # admission floor. margin~0 (just-admitted, noise-sensitive) -> 0.70x;
                # margin>=0.40 (decisive) -> 1.0x. When a marginal entry's timing shifts
                # +-1 bar under noise, its position delta is now 0.70x -> smaller
                # equity-return tracking error -> stability up; Sharpe is scale-invariant
                # so down-weighting low-quality trades vs high-conviction ones is raw-
                # neutral-to-positive. Continuous (no decision-boundary flip — unlike a
                # gated weight); same safe family as _bull_quality_atten. New data dep:
                # first-bar size depends on conviction margin above floor.
                _bull_conv_atten = 0.70 + 0.30 * max(0.0, min(1.0, _bull_margin / 0.40))
                _bear_conv_atten = 0.70 + 0.30 * max(0.0, min(1.0, _bear_margin / 0.40))
                # Architectural: churn-gated first-bar entry SIZE attenuator (shrink,
                # don't block). The diagnostic (c265424d) proved fast re-entries are the
                # entire rally instability; BLOCKING them (branch) gave PERFECT rally
                # stability 1.000 but collapsed raw. The proven winning axis (10bfd268)
                # is SHRINK marginal entries first-bar-only, not block/delay. Here the
                # "marginal" axis is the symbol's own recent entry DENSITY — the integer
                # churn count len(_eh) over the pruned 30-bar window, the SAME noise-immune
                # gate the grid-quantize keep (09f09ab) uses to fire in rally and stay ~0
                # in crash/sideways (spared by construction). len(_eh) currently feeds only
                # the admission THRESHOLD (_freq_factor, a BLOCK mechanism) and the
                # execution-layer grid/deadband — never the entry SIZE. This is the SHRINK
                # counterpart: as churn rises, later-in-burst entries (the most noise-driven
                # re-entries) get smaller first-bar size; the first entry of a sequence
                # (len<=1) stays full. Non-uniform within rally (targets the churny ones),
                # so it cuts their clean/pert tracking error while leaving stable entries at
                # full weight (Sharpe scale-invariant -> raw preserved). Integer-gated (not
                # a continuous price-derived quantity like Exp1 trend-gate proximity which
                # is noisy near boundary). New data dep: first-bar entry size depends on
                # integer churn count.
                _churn_size_atten = 1.0 - 0.25 * max(0.0, np.tanh((len(_eh) - 1.5) / 1.5))
                # Architectural: trend-QUALITY (regression R^2) first-bar entry-size
                # attenuator. NEW orthogonal signal: none of the existing attenuators
                # (conv-margin, voter-quality, multi-window consensus, churn) measure
                # the LINEARITY of the recent price path. R^2 of the OLS fit of log(HL2)
                # over LINREG_PERIOD is a continuous [0,1] cleanliness statistic, distinct
                # from slope DIRECTION (boundary-walled) and slope MAGNITUDE (_consensus
                # uses tanh(slope)): a clean one-directional move has R^2~1 regardless of
                # whether it is up or down; a choppy whipsaw path has low R^2 even if its
                # net slope is large. Mechanism: choppy (low-R^2) entries are whipsaw-prone
                # losers; shrink their first-bar commitment so their clean/perturbed
                # tracking error and Sharpe drag are down-weighted, while clean-trend
                # entries (high R^2, e.g. grinding bull/crash legs) keep full size and stay
                # ~inert. Direction-agnostic (same scalar both sides). Continuous tanh on
                # R^2 (no zero-crossing -> not the walled admission-boundary family); shrink
                # only (caps at 1.0). New data dep: first-bar size depends on path linearity.
                _tq_r2 = _fast_r2(np.log((bd.history["high"].values[-LINREG_PERIOD:] + bd.history["low"].values[-LINREG_PERIOD:]) / 2.0))
                _tq_atten = 0.25 + 0.75 * max(0.0, min(1.0, np.tanh(_tq_r2 / 0.30)))
                # Branch step 6: CHURN-GATE the R^2 shrink (active in low-churn, OFF in
                # high-churn bursts). The R^2 atten's raw gains live in sparse-entry regimes
                # (low len(_eh)) but its noise COST falls on the bursty-entry regime whose
                # stability is the binding constraint — R^2-dependent sizing on bursty
                # entries adds a noise-sensitive quantity to the choppy regime's positions,
                # dropping its stability below baseline. Gate the shrink by the SAME
                # noise-immune integer churn count the baseline grids use: _tq_calm ~1 at
                # len(_eh)<=1 (sparse-entry regimes get the full R^2 shrink -> raw gains kept)
                # fading to ~0 at len(_eh)>=3 (bursty entries get NO shrink -> positions
                # revert to un-attenuated size, sparing the choppy regime's stability).
                # Self-measured behavioral gate (NOT a regime label) — same family as the
                # baseline's churn-gated grids/deadband; regime effects fall out of realized
                # per-symbol entry density. Blend toward 1.0 (no shrink) as churn rises.
                _tq_calm = 1.0 - max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))
                _tq_atten = 1.0 - (1.0 - _tq_atten) * _tq_calm
                # Cross-asset consensus entry-size attenuator (direction-aware, size-down
                # only). _xa_mean>0 = market-wide bullish; bull entries aligned with it
                # keep full size, bull entries into a bearish/mixed market (idiosyncratic,
                # noise-fragile) shrink up to 30%. Symmetric for bear.
                _xa_atten_bull = 0.70 + 0.30 * max(0.0, min(1.0, _xa_mean))
                _xa_atten_bear = 0.70 + 0.30 * max(0.0, min(1.0, -_xa_mean))
                if _bull_strong >= _bull_strong_min and _bull_admit and _bull_persist_ok:
                    target = size * min(0.55, _entry_frac_dyn + _range_bull_adj) * _cooldown_factor * _bull_ct_atten * _bull_ct_vlong * _bull_consensus_atten * _bull_quality_atten * _vol_entry_atten * _outcome_size_mult * _port_dd_atten * _tod_atten * _bull_conv_atten * _churn_size_atten * _tq_atten * _xa_atten_bull
                elif _bear_strong >= _bear_strong_min and _bear_admit and _bear_persist_ok:
                    target = -size * min(0.55, _entry_frac_dyn + _range_bear_adj) * _cooldown_factor * _bear_ct_atten * _bear_ct_vlong * _bear_consensus_atten * _bear_quality_atten * _vol_entry_atten * _outcome_size_mult * _port_dd_atten * _tod_atten * _bear_conv_atten * _churn_size_atten * _tq_atten * _xa_atten_bear
            elif current_pos != 0:
                pos_pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                if current_pos < 0:
                    pos_pnl = -pos_pnl
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)

                # Architectural simplification: removed _trend_agree scale-in override.
                # Trend agreement was already filtered at entry time by _bull_admit/_bear_admit
                # gates (TREND_GATE_DEADZONE). Re-checking trend during scale-in duplicates
                # the entry-time trend gate. If trend deteriorates post-entry, pnl-attn alone
                # captures it (price follows trend in losses). Removing trend_agree blend
                # eliminates correlated double-counting of trend signal across entry+scale-in.
                # Architectural simplification: removed LIVE-CONVICTION scale-in
                # accelerator (_conv_accel / _conv_trend_mute / _live_side_margin) — it
                # made scale-in PACE depend on the per-bar conviction MARGIN (noisy,
                # non-monotonic), adding bull noise. Branch: the accelerator was ALSO a
                # rally-pace source (removing it slowed rally: raw 0.393->0.371). Step 2
                # tried a 48-bar ER gate to restore chop pace but ER is low in EARLY bull
                # too, so it re-added bull noise (bull 0.806->0.772). Step 3: the real issue
                # is that the base pace formula MIS-classifies rally as "chop" (low
                # rsi_trend_str) and assigns it SLOW scale-in (4 bars), but rally is a
                # grinding uptrend whose pullbacks reward FAST scale-in. Fix deterministically
                # by halving the chop-slowdown coefficient (2.0 -> 1.0): chop pace 4.0->3.0,
                # trend pace unchanged at 2.0. No per-bar / ER term — pace depends only on
                # the smooth long-window rsi_trend_str (bull untouched at high rsi_trend_str,
                # so bull's 0.806 noise-removal gain is preserved). Single structural change
                # to the scale-in TIMING, the one lever prior sessions found able to move
                # stability.
                _entry_full_bars_dyn = max(1.5, 2.0 + 1.0 * (1.0 - rsi_trend_str))  # [2.0, 3.0]
                if bars_held <= _entry_full_bars_dyn:
                    _eff_progress = bars_held / max(_entry_full_bars_dyn, 1e-6)
                    _eff_progress = max(0.0, min(1.0, _eff_progress))
                    scale_frac = min(1.0, ENTRY_INITIAL_FRAC + (1.0 - ENTRY_INITIAL_FRAC) * _eff_progress)
                    # Architectural: pnl-conditioned scale-in adverse-move freeze with
                    # COUNTER-TREND gating. Adverse moves during scale-in fall into two
                    # categories: (1) real reversal (counter-trend entries facing the
                    # actual trend, e.g. bear short during uptrend → losses real), and
                    # (2) pullback noise (trend-aligned entries facing temporary noise
                    # in rally). Branch step 2: gate the freeze by counter-trend strength.
                    # Trend-aligned scale-in (pos_dir matches ret_long sign) bypasses
                    # the freeze entirely (rally pullback noise recovers). Counter-trend
                    # scale-in still freezes (real reversals shouldn't grow into losses).
                    # New cross-component dep: scale-in freeze depends on (pos_dir, ret_long).
                    _pos_dir_si = 1.0 if current_pos > 0 else -1.0
                    _ct_si_gate = max(0.0, np.tanh(-ret_long * _pos_dir_si / 0.04))  # [0,~1] counter-trend
                    _adv_freeze = 0.75 * max(0.0, np.tanh(-pos_pnl / (0.4 * abs(STOP_LOSS_PCT)))) * _ct_si_gate
                    scale_frac = scale_frac * (1.0 - _adv_freeze)
                    full_target = size if current_pos > 0 else -size
                    target = full_target * scale_frac
                    # Don't shrink below current position - this is scale-in, not exit
                    if (current_pos > 0 and target < current_pos) or (current_pos < 0 and target > current_pos):
                        target = current_pos

                # Unified soft exit-pressure architecture (slope + peak_profit + time only).
                # Stop-loss kept as hard gate (entry-anchored, already noise-immune).
                # Architectural: confirmed-peak update — peak shifts only when pos_pnl
                # exceeds previous peak AND is rising (pos_pnl > prev_pos_pnl). Single-bar
                # noise spikes don't anchor the peak. Sideways sharpness preserved (peaks
                # confirmed within 1 extra bar). Different from EMA smoothing: this is a
                # gating rule on the high-water mark, not a low-pass filter.
                _prev_pnl = self._smoothed_pnl.get(symbol, pos_pnl)
                self._smoothed_pnl[symbol] = pos_pnl
                _curr_peak = self.peak_pnl.get(symbol, 0.0)
                # Confirmed-peak update: peak shifts only when pos_pnl > prev_peak AND
                # pos_pnl >= prev_pos_pnl (rising bar).
                if pos_pnl > _curr_peak and pos_pnl >= _prev_pnl:
                    self.peak_pnl[symbol] = pos_pnl
                else:
                    self.peak_pnl[symbol] = _curr_peak
                # Architectural: MAE (maximum adverse excursion) low-water mark.
                # Tracks lowest pos_pnl observed since entry; only updates downward.
                _curr_mae = self._mae.get(symbol, 0.0)
                self._mae[symbol] = min(_curr_mae, pos_pnl)

                # Architectural: ATR-based dynamic stop-loss.
                # Replace fixed STOP_LOSS_PCT (-0.024) with ATR-derived per-symbol stop.
                # ATR(14) on smoothed_closes captures each symbol's structural volatility
                # (BTC tighter, SOL wider). Stop = K * ATR_pct, clamped to historical range.
                # Smooth (no boundary): same vol-adaptive band still applies.
                _atr_n = 14
                _atr_high = bd.history["high"].values[-_atr_n:]
                _atr_low = bd.history["low"].values[-_atr_n:]
                _atr_close = closes[-_atr_n - 1:-1]  # prev closes
                _tr = np.maximum(_atr_high - _atr_low, np.maximum(np.abs(_atr_high - _atr_close), np.abs(_atr_low - _atr_close)))
                _atr_pct = np.mean(_tr) / mid
                # Stop scales as 2.5x ATR_pct, clamped to [0.018, 0.035]: keeps in
                # similar range to original 0.024 but adapts per-symbol/per-regime.
                _stop_abs = max(0.018, min(0.035, 2.5 * _atr_pct))
                _loss = -pos_pnl
                _band_half = (0.06 + 0.20 * min(1.0, vol_ratio)) * _stop_abs
                _sl_pressure = max(0.0, min(1.0, (_loss - (_stop_abs - _band_half)) / (2.0 * _band_half)))

                # Slope-against pressure: use MEDIAN of 3 slopes at different windows for
                # robustness. Single _lr_slope (16-bar) is shared with entry voter — coupling
                # entry & exit noise. Computing slopes at 12/16/22 and taking median decouples
                # exit-noise from entry-noise AND robust-aggregates against single-window outliers.
                # Multi-window slope MEAN (not median): mean averages out window-specific noise
                # better than median in low-vol where all 3 slopes are small and noise-dominated.
                # Median can flip on a single window; mean spreads the contribution.
                _hl2 = (bd.history["high"].values + bd.history["low"].values) / 2.0
                _slopes = []
                for _w in (12, 16, 22):
                    _ll = _fast_slope(np.log(_hl2[-_w:]))
                    _slopes.append(_ll)
                _exit_slope = float(np.mean(_slopes))
                _slope_against = -_exit_slope if current_pos > 0 else _exit_slope
                _slope_thresh = 0.0003 + 0.0003 * max(0.0, min(1.0, (0.7 - vol_ratio) / 0.3))
                _slope_band = 0.20 + 0.30 * max(0.0, min(1.0, (0.9 - vol_ratio) / 0.4))
                _sl_slope_pressure = max(0.0, min(1.0, (_slope_against - (1.0 - _slope_band/2) * _slope_thresh) / (_slope_band * _slope_thresh)))
                # Architectural simplification: removed trend-aligned slope-pressure attenuation.
                # Parallel reasoning to _scale_in_w removal (a44612e keep): slope-against IS
                # signal not noise. Trend-aligned positions facing slope-against during
                # pullbacks could be in the first bars of a trend reversal, not just noise.
                # The 35%-cap attenuation structurally delayed exit on the FIRST slope-against
                # signal in trend-aligned holds. Removing the attenuation lets trend-aligned
                # positions exit faster on the earliest slope-reversal indication; subsequent
                # _pp_pressure giveback ratio still protects post-peak losses, and _w_slope
                # is heavier in loss anyway, so winning trend positions retain their soft
                # protection via the heavier-in-loss weight inversion not the trend-align
                # multiplier. Code-structure removal: 11 lines + cross-timescale data dep.

                # Peak-profit soft pressure: vol-adaptive band (same architectural pattern as SL).
                # Low vol -> narrower band (closer to binary, less near-giveback oscillation).
                # High vol -> wider band (absorbs giveback-ratio noise from price chop).
                _pp_min = PEAK_PROFIT_MIN_BASE * max(0.6, min(2.0, vol_ratio ** 0.5))
                _giveback = max(0.0, self.peak_pnl[symbol] - pos_pnl)
                _giveback_ratio = _giveback / max(self.peak_pnl[symbol], _pp_min)
                # Architectural: profit-magnitude-aware giveback amplification
                # with trend-strength attenuation. In strong long-window trends
                # (|ret_long| > 0.06), amplification attenuates toward 0 to let
                # winning trend positions run longer (prevents premature trailing
                # in rally/crash). In chop/moderate trend, full amplification
                # preserves sideways/bull tight-trailing benefit. New cross-
                # timescale data dependency: pp amplification depends on
                # long-window trend magnitude. Continuous via tanh.
                _profit_magnitude = max(0.0, self.peak_pnl[symbol] / max(_pp_min, 1e-6) - 1.0)
                _pm_trend_atten = 1.0 - 0.7 * max(0.0, np.tanh((abs(ret_long) - 0.04) / 0.08))  # in [0.3, 1], gated above 0.04
                _giveback_ratio = _giveback_ratio * (1.0 + 0.18 * _pm_trend_atten * np.tanh(_profit_magnitude / 0.7))
                _pp_band = 0.10 + 0.20 * min(1.0, vol_ratio)
                _pp_lower = PEAK_PROFIT_GIVEBACK * (1.0 - _pp_band)
                # Architectural: smooth pp-activation ramp replacing hard binary gate.
                # Original: pp_pressure = 0 below peak == _pp_min, full ramp above. Hard
                # boundary at peak == _pp_min creates noise discontinuity in stab tests.
                # Replace with smooth tanh activation (peak_pnl/_pp_min - 1.0) scaled by 0.5,
                # giving 0.5 at peak == _pp_min and saturating to 1.0 at peak == 1.5*_pp_min.
                # This is a primitive change to pp_pressure activation: was binary gate,
                # now continuous mixture between unconditional pp_pressure and zero.
                # Trend-gated smooth activation: smooth ramp only in trending regimes
                # (rsi_trend_str high, where bull/crash benefits manifest), near-binary
                # in chop where the smoothing destabilizes peak-protection. cooldown_trend_strength
                # is bounded [0,1] and equals min(|ret_long|/0.06, 1) — well-aligned for this.
                # Narrow boundary smoothing only: linear ramp in [0.95, 1.04]*_pp_min.
                # Slightly narrower upper bound — restores baseline pp_pressure faster
                # at peak ratios above 1.04, recovering raw revenue while keeping the
                # bull-boosting smoothing in the [0.95, 1.04] band.
                # Architectural simplification: removed smooth pp-activation ramp.
                # The 9%-wide boundary smoothing [0.95, 1.04] interpolated _pp_activation
                # to bridge a binary on/off at peak == _pp_min. Since peak_pnl is itself a
                # high-water mark (only updates upward, confirmed by 2 rising bars), it is
                # already smooth — additional boundary smoothing is redundant. Replace with
                # binary activation at _pp_ratio >= 1.0. Code-structure removal: 6 lines
                # → 1 line; eliminates the interpolation table that duplicates smoothing
                # already provided by peak_pnl's high-water-mark mechanic.
                _pp_ratio = self.peak_pnl[symbol] / max(_pp_min, 1e-6)
                _pp_activation = 1.0 if _pp_ratio >= 1.0 else 0.0
                _pp_raw = max(0.0, min(1.0, (_giveback_ratio - _pp_lower) / (PEAK_PROFIT_GIVEBACK * _pp_band)))
                _pp_pressure = _pp_raw * _pp_activation

                # Time pressure: wider smooth ramp (4 bars) to reduce noise sensitivity
                # Uses same robust median exit-slope for consistency within exit subsystem.
                _slope_agrees = (_exit_slope > 0 and current_pos > 0) or (_exit_slope < 0 and current_pos < 0)
                _slope_strength = min(1.0, abs(_exit_slope) / 0.0006)
                # Architectural: vol-conditioned symmetric momentum hold bonus.
                # Slope-against shortens max_hold but only at full strength when
                # slope is signal-dominated (high vol). In low-vol (rally chop) the
                # shortening is attenuated by min(1, vol_ratio) — slope noise in
                # rally would otherwise create noise-driven early time exits.
                # Extension (slope-agrees) remains unchanged (bull/crash extended hold).
                _short_atten = min(1.0, vol_ratio)
                _hold_adj = MOMENTUM_HOLD_BONUS * _slope_strength * (1.0 if _slope_agrees else -_short_atten)
                _max_hold = HOLD_DECAY_START + (1.0 / HOLD_DECAY_RATE) + _hold_adj
                _time_pressure = max(0.0, min(1.0, (bars_held - _max_hold + 3.0) / 4.0))

                # PnL-conditioned exit-pressure weighting (architectural change to fusion):
                # In profit (pos_pnl > 0), peak-profit dominates — preserve gains via giveback.
                # In loss (pos_pnl < 0), slope-against dominates — cut losers via momentum reversal.
                # Stop-loss and time pressure stay at unit weight (protective + structural).
                # Smooth transition via tanh of pos_pnl scaled by stop magnitude.
                _pnl_scale = np.tanh(pos_pnl / abs(STOP_LOSS_PCT))   # in [-1, 1]
                # Architectural simplification: removed _scale_in_w slope-pressure
                # attenuator. The 0.5..1.0 ramp dampened slope-against pressure during
                # scale-in to "let positions reach full size." But early scale-in slope
                # reversals are signal — counter-trend-emerging entries should exit
                # FAST, not survive until full commit. The attenuator structurally
                # opposes the slope-against early-warning function. Code-structure
                # removal: 1 cross-bar dependency on bars_held removed from _w_slope
                # and _w_pp; both revert to single-factor weights.
                _scale_in_w = 1.0
                _w_slope = 1.0 + 0.15 * max(0.0, -_pnl_scale)  # heavier in loss
                # Architectural: vol-conditioned profit-side _w_pp.
                # Low vol (sideways/rally): _w_pp simplified to _scale_in_w (no extra boost).
                #   Peak-profit pressure already amplifies via _profit_magnitude + _pp_activation.
                # High vol (crash): restore profit-side amplification — crash recovery profits
                #   are short-lived and need fast giveback locking.
                # Continuous tanh on (vol_ratio - 1.0)/0.4 — smooth transition around vol_ratio=1.
                _vol_w_pp_gate = max(0.0, np.tanh((vol_ratio - 1.0) / 0.4))  # in [0, ~1]
                _w_pp    = (1.0 + 0.20 * max(0.0, _pnl_scale) * _vol_w_pp_gate) * _scale_in_w
                # Architectural: trend-magnitude-attenuated time-pressure weight.
                # In strong trends (high |ret_long|), trend-aligned winning
                # positions should hold longer — time pressure is noise in trend
                # following. In chop (low |ret_long|), time pressure is critical
                # anti-overstay for sideways/rally stability.
                # Mute the profit-side amp by _trend_strength_w:
                # chop: _w_time = 1.0 + 0.20*max(0,_pnl_scale) (full, anti-overstay)
                # trend: _w_time = 1.0 (base, trend provides directional persistence)
                # New cross-timescale data dep: time weight depends on long-window trend.
                # Following the regime-asymmetric insight from 5648b3a8: time pressure
                # removal helped bull/crash but destroyed sideways/rally.
                _w_time  = 1.0 + 0.20 * max(0.0, _pnl_scale) * (1.0 - _trend_strength_w)
                # Architectural multi-variable restructure: replaced voter-attn
                # multiplicative cross-coupling with bilateral additive voter_bias.
                # Reasoning: _voter_attn applied a 0..0.30 dampening factor to four
                # heterogeneous pressure terms (slope/pp/time/ve), creating cross-
                # subsystem correlated coupling — a single voter-state value scaled
                # all four terms simultaneously. Replace with:
                #   1) voter_bias subtracts from exit when own-side conviction strong
                #      (lets winners run when voters still validate position).
                #   2) voter_bias ADDS to exit when opposite-side conviction strong
                #      (bilateral — explicit reversal evidence raises exit).
                # Bilateral-additive fusion decouples voter influence from individual
                # pressure terms while preserving net effect on exit decision.
                _side_margin = _bull_margin if current_pos > 0 else _bear_margin
                _opp_margin = _bear_margin if current_pos > 0 else _bull_margin
                # Chop-amplified own-side subtraction with divergence taper: in pure sideways
                # non-counter-trend holds, taper _chop_amp toward 1.0 by strong-sum divergence.
                _div_taper = max(0.0, np.tanh(abs(_bull_strong - _bear_strong) / max(_bull_strong + _bear_strong, 1e-6) / 0.30)) * max(0.0, np.tanh((0.015 - abs(ret_long)) / 0.010)) * max(0.0, np.tanh(((1.0 if current_pos > 0 else -1.0) * ret_long + 0.005) / 0.010))
                _chop_amp = (1.0 + 0.7 * max(0.0, min(1.0, (0.03 - abs(ret_long)) / 0.025))) * (1.0 - _div_taper) + _div_taper
                # Architectural: trend-aligned opp-bias attenuator (new cross-component dep).
                # In strong long-window trends WHERE position is trend-aligned, attenuate
                # the opposite-side voter_bias ADDITION. Mechanism: when winning trend
                # positions (bull in uptrend, bear in downtrend) face opposite-side voter
                # spikes (rally pullback bull voters firing on bear positions; crash dead-
                # cat bounce bull voters firing on bear shorts), the additive opp bias
                # currently fires the same as in chop. In confirmed trends, opp-voter
                # signals during pullbacks are more often noise than reversal. Attenuate
                # opp_bias by tanh(ret_long * pos_dir / 0.05) so trend-aligned positions
                # see softer opp-bias contribution to _exit_pressure. Counter-trend
                # positions and chop: unchanged. New cross-timescale data dep: opp-side
                # voter_bias depends on (ret_long, position direction).
                _pos_dir_vb = 1.0 if current_pos > 0 else -1.0
                _trend_align_vb = max(0.0, np.tanh(ret_long * _pos_dir_vb / 0.05))  # [0, ~1]
                _opp_atten = 1.0 - 0.50 * _trend_align_vb  # max 50% attenuation in strong trend-aligned
                # Architectural: trend-magnitude amp on opp_bias (NEW data dep at fusion).
                # In chop (low abs(ret_long)), opp-voter spikes are themselves noise (no
                # directional backing) — mute opp_bias contribution. In trends, opp-voter
                # spikes carry reversal signal — full activation. Continuous tanh on
                # abs(ret_long)/0.04. Symmetric counterpart to _chop_amp on own-side
                # subtraction (chop amplifies own-side hold; chop also mutes opp-side
                # exit-spike). Multi-variable: adds new factor to opp-side fusion.
                _opp_trend_amp = 0.5 + 0.5 * max(0.0, np.tanh(abs(ret_long) / 0.04))  # [0.5, ~1]
                _voter_bias = -0.20 * _chop_amp * max(0.0, np.tanh(_side_margin / 0.30)) + 0.20 * _opp_atten * _opp_trend_amp * max(0.0, np.tanh(_opp_margin / 0.30))
                # Architectural: volatility-expansion exit pressure (5th source).
                # When recent 6-bar realized vol substantially exceeds 18-bar
                # realized vol (vol-of-vol expansion), the price regime has
                # shifted — earlier slope/peak/time signals may be stale. Compute
                # vol_expansion = vol_6 / vol_18, smooth via tanh, contribute
                # smooth pressure [0, 0.6]. Acts as a regime-shift detector
                # orthogonal to slope (direction) and pp (magnitude). New
                # data-dependent exit pressure term in the fusion sum.
                _vol_6 = max(np.std(np.diff(np.log(closes[-7:-1]))), 1e-6)
                _vol_18 = max(np.std(np.diff(np.log(closes[-19:-1]))), 1e-6)
                _vol_expansion = _vol_6 / _vol_18
                # Activate above 1.3x, saturate near 2.0x. Smooth via tanh.
                _ve_pressure = 0.6 * max(0.0, np.tanh((_vol_expansion - 1.3) / 0.4))
                # Profit-side weight: only fire when in profit (lock gains on
                # regime shift); don't punish losing positions for vol expansion
                # since slope-against already handles adverse moves.
                _w_ve = max(0.0, _pnl_scale)  # in [0, 1], only positive pos_pnl
                # Architectural simplification: removed early-profit-lock exit pressure.
                # _ep_pressure fired on small-peak giveback below _pp_min activation.
                # In rally (low-vol grind-up), positions frequently have small peaks
                # from drift, and _ep_pressure fires on pullback giveback — kills
                # trend-following positions on structural pullbacks, not reversals.
                # Large-peak protection already provided by _pp_pressure (fires above
                # _pp_min). Small-peak exits handled by slope/time pressures.
                # Code-structure removal: -16 LOC, -1 exit term from MAX fusion (6→5),
                # eliminates sub-peak giveback data dependency.
                _ep_pressure = 0.0
                _w_ep = 0.0
                # Architectural: adverse-recovery exit pressure (5th soft source).
                # New per-symbol state (MAE low-water mark) drives a new control flow:
                # when current pos_pnl has substantially recovered from MAE but is still
                # in modest loss, the position is "barely surviving" — recovery is at
                # risk of reversing into another adverse leg. Exit pressure rises
                # proportional to recovery fraction. Activates only when:
                #   - MAE is meaningful (< -0.5 * |STOP_LOSS_PCT|)
                #   - current pos_pnl is in modest loss territory (between MAE and 0)
                # Distinct from pp_pressure (which measures peak giveback for winners)
                # and from sl_pressure (binary stop). Targets the "recovered to small
                # loss after dip" pattern — historically a dangerous holding zone.
                _ar_pressure = 0.0
                _curr_mae_e = self._mae.get(symbol, 0.0)
                _mae_floor = -0.5 * abs(STOP_LOSS_PCT)
                if _curr_mae_e < _mae_floor and pos_pnl < 0:
                    # recovery_frac: 0 at MAE, 1 at pos_pnl=0 (full recovery to breakeven)
                    _recovery_frac = max(0.0, min(1.0, (pos_pnl - _curr_mae_e) / max(-_curr_mae_e, 1e-6)))
                    # Activate above 0.5 recovery (mild dip recoveries don't trigger);
                    # ramp smoothly to 0.40 cap at full breakeven recovery.
                    _ar_pressure = 0.40 * max(0.0, min(1.0, (_recovery_frac - 0.5) / 0.4))
                # Weight: only fire on currently-losing positions (definitionally — gated above);
                # full weight (this pressure measures recovery quality on losers, not profit lock-in).
                _w_ar = 1.0
                # Architectural fusion change: element-wise MAX replaces weighted sum.
                # Old: weighted sum of 6 soft terms (slope+pp+time+ve+ep+ar) with pnl-scaled
                # weights. All 6 terms share vol_ratio, HL2/closes, and pnl_scale as input —
                # noise in any shared input propagates to all 6, which then SUMS. Take
                # only the most-pressing term (MAX with weights): eliminates correlated
                # noise addition. Weights preserved so profit-side terms dominate when
                # profitable, loss-side when losing. voter_bias + sl max-blend unchanged.
                _soft_terms = (
                    _w_slope * _sl_slope_pressure,
                    _w_pp * _pp_pressure,
                    _w_time * _time_pressure,
                    _w_ve * _ve_pressure,
                    _w_ep * _ep_pressure,
                    _w_ar * _ar_pressure
                )
                _soft_max = max(_soft_terms)
                # Architectural: multi-source agreement attenuator on soft_max.
                # When only ONE source contributes meaningfully (top-2 ratio low,
                # i.e. dominant single source), attenuate up to 25% — single-source
                # spikes are more often noise than real reversal. When TWO+ sources
                # agree (top-2 ratio high), no attenuation. Continuous via tanh on
                # second-highest/highest ratio. Multi-source CONFIRMATION as a
                # noise filter is structurally different from EMA smoothing (time-
                # axis) and threshold raising (boundary-axis) — operates on the
                # pressure-source dimension. Top exit decision: 2nd-highest term
                # ratio gates the strength of the MAX. New cross-source data dep.
                _sorted_terms = sorted(_soft_terms, reverse=True)
                _ratio_2nd = _sorted_terms[1] / max(_sorted_terms[0], 1e-6) if _sorted_terms[0] > 1e-6 else 0.0
                # Confirmation strength: 0 at single-source, 1 at full agreement
                _agree_gate = max(0.0, min(1.0, np.tanh(_ratio_2nd / 0.30)))
                # Branch step 2: chop-only gating. In trends (high abs(ret_long)),
                # single-source pressure signals are real (slope reversal alone =
                # genuine trend break). Mute the attenuator's effect by trend strength
                # so it operates only in chop where single-source spikes ARE noise.
                _chop_atten_w = 1.0 - max(0.0, np.tanh(abs(ret_long) / 0.04))  # 1 in chop, 0 in trend
                # Attenuator: scaled by chop weight — in chop 0.75x at single, 1.0x at agree;
                # in trend approaches 1.0x always (no attenuation)
                _soft_atten = 1.0 - 0.25 * (1.0 - _agree_gate) * _chop_atten_w
                _soft_max = _soft_max * _soft_atten
                _exit_pressure = max(_sl_pressure, _soft_max) + _voter_bias
                # Architectural: pos_pnl-gated scale-in exit threshold ramp.
                # During scale-in (bars_held <= ENTRY_FULL_BARS) AND winning (pos_pnl > 0),
                # raise the exit threshold from 1.0 to 1.2 along a smooth linear ramp
                # (1.2 at bar 0, 1.0 at bar ENTRY_FULL_BARS). Protects winning scale-in
                # from noise-driven premature exits while letting losing scale-in exit
                # normally (no protection — losing positions are noise-vulnerable too).
                # Stop-loss is exempt (full _sl_pressure forces exit regardless).
                _scale_in_winning = bars_held <= ENTRY_FULL_BARS and pos_pnl > 0
                # Architectural simplification: removed _vt_factor 2D vol-time exit_thresh
                # modulator. The factor activated only in narrow 2D band (low-vol AND mid-life),
                # contributing at most +10%. With the underlying soft-pressure stack already
                # vol-conditioned (de_floor, _w_pp gate, slope band, pp band), the additional
                # ad-hoc band-pass on _exit_thresh is redundant. Keeping scale-in-winning bonus
                # unchanged (load-bearing for early winning protection).
                _exit_thresh = 1.0 + 0.20 * max(0.0, 1.0 - bars_held / ENTRY_FULL_BARS) if _scale_in_winning else 1.0
                # Stop-loss exemption: when _sl_pressure is near saturation, force standard threshold.
                if _sl_pressure >= 0.95:
                    _exit_thresh = 1.0
                # Architectural: graduated partial-exit instead of binary exit.
                # When _exit_pressure crosses below _exit_thresh but above a soft floor
                # (0.65 * _exit_thresh), shrink position size proportionally toward 0
                # rather than waiting for full-pressure exit. This lets positions
                # de-risk gradually under mounting exit pressure (slope weakening,
                # giveback rising) rather than binary-flip on threshold cross.
                # Stop-loss path retains binary exit (full saturation already triggers).
                # New control flow: exit decision is a continuous mapping from
                # exit_pressure ratio to target multiplier, not a single threshold.
                # Architectural: profit-target partial harvest (decision-architecture
                # change to exit subsystem). When peak_pnl crosses into profit-target
                # territory (peak >= 1.6*_pp_min), apply a smooth size scale-down of
                # up to 30% — independent from giveback-based _pp_pressure trailing.
                # This harvests realized peak gains proactively when profit target is
                # hit, even if pos_pnl is currently still near peak (no giveback yet).
                # Ramps smoothly over [1.6*_pp_min, 2.2*_pp_min] via tanh. Subtractive
                # from current target, applied BEFORE exit-threshold logic so it
                # composes correctly with both binary and de-risk exit paths. Skipped
                # if _sl_pressure dominant (full exit will follow). New control flow:
                # exit subsystem now has THREE size-decision paths: full exit, de-risk
                # ramp, and take-profit scale-down — orthogonal to giveback trailing.
                if target != 0 and self.peak_pnl[symbol] > 1.6 * _pp_min and _sl_pressure < 0.5:
                    _tp_ratio = self.peak_pnl[symbol] / max(_pp_min, 1e-6)
                    # Trend-gated activation: in chop (low |ret_long|), peaks are
                    # rare AND likely mean-reverting — disable harvest to let small
                    # sideways wins run. In trending regimes (high |ret_long|), peaks
                    # are real and worth locking. Continuous tanh on |ret_long|/0.04.
                    _tp_trend_gate = max(0.0, np.tanh(abs(ret_long) / 0.04))  # in [0, ~1]
                    # MAE-cleanliness × trend-align × deep-peak gate suppresses harvest
                    # when peak is a confirmed trend extension. Counter-trend or rally
                    # pullback peaks get full harvest (mean-reverting by structure).
                    _ts_supp = (1.0 - max(0.0, min(1.0, np.tanh(-self._mae.get(symbol, 0.0) / abs(STOP_LOSS_PCT) / 0.2)))) * max(0.0, np.tanh(ret_long * (1.0 if current_pos > 0 else -1.0) / 0.04)) * max(0.0, min(1.0, np.tanh((_tp_ratio - 2.8) / 0.5)))
                    _tp_scale = 0.30 * max(0.0, min(1.0, np.tanh((_tp_ratio - 1.6) / 0.6))) * _tp_trend_gate * max(0.0, 1.0 - 1.5 * _ts_supp)
                    target = target * (1.0 - _tp_scale)

                # Architectural: removed binary soft-exit clause (-3 LOC).
                # Old: 2 control-flow branches both fired at pressure=thresh — binary
                # full-exit path AND de-risk ramp (which produces target=0 at boundary).
                # The binary path fired on bars 0-1 (where de-risk is gated off),
                # exposing fresh entries to single-bar soft-pressure noise spikes.
                # Removing it: fresh entries (bars 0-1) become protected from soft-
                # pressure noise (only SL or opp_gate can close them); bars>=2 keep
                # identical exit behavior via de-risk ramp (de_risk=0 at pressure=thresh).
                if _sl_pressure >= 0.95 and _exit_pressure >= 1.0 and target != 0:
                    target = 0.0
                elif target != 0 and bars_held >= 2:
                    # Architectural: PnL-conditioned partial-exit floor (replaces
                    # vol-conditioning). New cross-subsystem data dep at exit
                    # graduation: floor depends on whether position is currently
                    # winning vs losing. Profit (pos_pnl > 0): wider ramp (floor=0.55,
                    # gradual de-risk to lock partial gains while letting upside run
                    # if pressure dissipates). Loss (pos_pnl < 0): narrower ramp
                    # (floor=0.85, near-binary fast exit — losers should not linger
                    # at half-size while soft pressures continue to mount). Smooth
                    # transition via tanh of pos_pnl / abs(STOP_LOSS_PCT) (same
                    # _pnl_scale used in pressure weights). Continuous, no boundary.
                    # Mechanism rationale: existing fast-exit semantics in losers
                    # are achieved by full _exit_pressure crossing _exit_thresh
                    # (binary path); the de-risk path gradient is currently MOST
                    # active in mid-pressure, profit-side mid-life situations where
                    # graduation makes most sense. Tightening loser graduation
                    # routes more loser exits through the _exit_thresh binary path.
                    _de_floor = 0.55 + 0.30 * max(0.0, -_pnl_scale)
                    # Architectural: one-sided trend-aligned de-risk floor relaxation.
                    # When position is trend-aligned (pos_dir matches ret_long sign) AND
                    # profitable, lower the de-risk floor to widen the graduated-exit
                    # ramp — trend-aligned winners de-risk more gradually through pullback
                    # noise. Counter-trend and losing positions keep original floor.
                    # Continuous tanh product, no boundary. Pattern: asymmetric exit-side
                    # relaxation, following afa6281 admission-side bull-only pattern.
                    # Only applies in profit (_pnl_scale > 0), trend-aligned via
                    # tanh(ret_long * pos_dir / 0.04), max relaxation 0.10.
                    _ta_de_align = max(0.0, np.tanh(ret_long * (1.0 if current_pos > 0 else -1.0) / 0.04))
                    _ta_de_profit = max(0.0, _pnl_scale)
                    _de_floor -= 0.10 * _ta_de_align * _ta_de_profit
                    # Architectural: fresh-entry exemption from de-risk path. Bars 0-1
                    # of an entry get binary-exit-only behavior (exit on full pressure
                    # or no exit). Partial exits during scale-in conflict with the
                    # scale-in pace itself — de-risk shrinks position while scale-in
                    # tries to grow it. Defer de-risk consideration until bars_held>=2
                    # so the position has cleared the initial commit-noise window.
                    # New control flow: bars_held condition gates the de-risk branch.
                    if _exit_pressure >= _de_floor * _exit_thresh:
                        _de_risk = 1.0 - (_exit_pressure - _de_floor * _exit_thresh) / ((1.0 - _de_floor) * _exit_thresh)
                        _de_risk = max(0.0, min(1.0, _de_risk))
                        target = target * _de_risk

                # Architectural simplification: removed in-place flip mechanism.
                # Flip win rate is ~5% across all regimes vs ~85% entry WR — flips are
                # the dominant cost driver (flip_pnl -560 to -960 per regime).
                # Replace single-bar reversal with exit-then-cooldown: when opposite-side
                # conviction passes the flip gate, set target=0. The standard cold-entry
                # path (with its 2-bar persistence gate) will re-enter in the opposite
                # direction on a subsequent bar IF conviction sustains. This decouples
                # reversal from a single-bar decision and routes it through the same
                # noise-filtering gate that protects fresh entries.
                # Architectural: graduated opp-gate replacing binary exit-on-reversal.
                # Old: when opp gate fires (bear votes pass + strong sum + trend),
                # set target=0 (full exit). New: scale exit by opp-side conviction
                # margin. Weak reversal evidence partially de-risks; strong reversal
                # fully exits. Smooth tanh on opposite-side margin maps to
                # exit-fraction in [0.4, 1.0]. Mechanism: avoids whipsaw full-exits
                # in crash where bull-side voter spikes are common during dead-cat
                # bounces but trend genuinely down. New decision-boundary mechanism:
                # opp-side reversal triggers partial position scaling, not binary.
                _opp_gate = (current_pos > 0 and bear_votes >= FLIP_MIN_VOTES and _bear_strong >= _bear_strong_min and trend_avg < 0) or \
                            (current_pos < 0 and bull_votes >= FLIP_MIN_VOTES and _bull_strong >= _bull_strong_min and trend_avg > 0)
                if not in_cooldown and _opp_gate:
                    # Graduated opp-gate gated on TREND-ALIGNED + IN-PROFIT.
                    # Counter-trend (rally bear) OR losing positions: binary full
                    # exit (cut risk fast). Trend-aligned + in-profit (crash short
                    # winning): graduated partial exit (preserves winning trend
                    # position through noise spikes). Both gates must hold for
                    # graduated behavior to engage. Continuous via tanh blend.
                    _pos_dir_og = 1.0 if current_pos > 0 else -1.0
                    _trend_align_og = max(0.0, np.tanh(ret_long * _pos_dir_og / 0.04))  # [0, ~1]
                    _profit_gate_og = max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # [0, ~1] only profit
                    _grad_gate = _trend_align_og * _profit_gate_og  # both required
                    _opp_exit_frac_grad = 0.4 + 0.6 * max(0.0, min(1.0, np.tanh(_opp_margin / 0.30)))
                    # Blend: full exit (1.0) by default, graduated only when both gates hold.
                    _opp_exit_frac = 1.0 + (_opp_exit_frac_grad - 1.0) * _grad_gate
                    target = current_pos * (1.0 - _opp_exit_frac)

            # Architectural subsystem redesign (execution/order-emission layer):
            # churn-gated proportional trade-admission deadband. The order-emission
            # gate previously fired on any move > 1.0 unit. Small same-sign resizes
            # (scale-in steps, partial de-risks, mult-driven size wobble) are the
            # dominant churn source — each is a fresh decision boundary that flips
            # under noise. A SYMMETRIC proportional hold-zone on resizes was the only
            # mechanism in many prior sessions to lift the worst regime's stability,
            # but a UNIFORM hold cost the low-churn regimes raw score. Gate the
            # deadband on the symbol's OWN recent entry density (len(_eh), the same
            # pruned-30-bar churn signal _freq_factor reads): high local churn opens a
            # wider hold-zone (suppress micro-resizes → fewer noise-flip boundaries),
            # low churn keeps the gate ~off (preserves raw). Self-measured behavioral
            # feedback, NOT a market-regime classifier — the regime effects fall out
            # of each symbol's realized trade rate. Entries (current_pos==0), full
            # exits (target==0), and flips (sign change) are ALWAYS exempt: risk
            # transitions must never be held. Only same-sign resizes pass through the
            # deadband. New control flow + new cross-component data dep at the gate.
            # Branch step 4: CONSTANT-WIDTH fast-saturating churn-gated snap deadband.
            # Step 3's chop gate read the efficiency-ratio _er, which is itself noise-
            # sensitive — gating the deadband WIDTH on a noisy quantity made the snap
            # boundary doubly-noisy and crushed rally stability (churn-only step1
            # rally=0.385 → churn×chop step3 rally=0.218). The prior-session UNIFORM
            # (constant-width) deadband, by contrast, LIFTED the worst regime's
            # stability to 0.578. Lesson: the deadband helps only when its width is
            # CONSTANT where it fires. So: drop the _er chop gate; gate width ON/OFF
            # by churn alone, with a FAST-saturating sigmoid so the width is
            # effectively constant in high-churn regimes (len>=3 → gate≈1.0) and ~0 in
            # low-churn regimes (len<=1 → gate≈0). Constant width where active = no
            # boundary-noise amplification → reproduces the uniform rally-stab lift
            # while sparing crash/sideways (their churn keeps the gate near 0). Snap-
            # to-hold (Zeno-free). Symmetric growth/shrink. Entries/exits/flips exempt.
            _churn_dz = max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))  # FAST saturation: ~0 at len<=1, ~1 at len>=3
            _deadband_frac = 0.13 * _churn_dz
            _is_resize = current_pos != 0 and target != 0 and (current_pos > 0) == (target > 0)
            if _is_resize and abs(target - current_pos) < _deadband_frac * abs(current_pos):
                target = current_pos  # snap-to-hold: suppress micro-resize, no residual gap
            # Architectural: churn-gated ABSOLUTE-target grid quantization (rally-stab
            # lever, generalizes ef027049 snap-to-hold from the resize DELTA to the resize
            # LEVEL). ef027049 snaps target->current_pos only when the change is tiny; once
            # rally genuinely resizes (change > deadband) the NEW absolute target is a
            # continuous price-derived value that AR(1) noise still perturbs bar-to-bar ->
            # the surviving position-value cascade that caps rally stab. Fix at the root:
            # in high churn, round the absolute resize target onto a coarse grid (step =
            # 0.10 * the symbol's natural position scale `size`). Noise-induced sub-grid
            # wobble in the continuous target then collapses onto the SAME grid level ->
            # fewer DISTINCT position values across the noise ensemble = the exact axis the
            # two rally-stab keeps (snap-to-hold, scale-in pace) moved. Deterministic given
            # target (pure rounding, no new price-derived term, no decision boundary that
            # can flip direction). Gated on the noise-IMMUNE integer churn count (fires in
            # rally, ~0 in crash/sideways = SPARED by construction). Resizes ONLY: entries
            # (current_pos==0), full exits (target==0), flips (sign change) ALWAYS exempt —
            # risk transitions must hit exact targets. Snap direction is toward current_pos
            # so a quantized resize never crosses zero or overshoots past the raw target's
            # side. New control flow at the order-emission layer.
            # Branch step 2: STABLE-LATTICE grid (decouple from bar-varying combined_mult).
            # Step 1 used grid = 0.10*size where size = equity*BASE*combined_mult — but
            # combined_mult (strength_scale*calm_boost*sideways_boost*vol terms) varies
            # every bar, so the lattice lines themselves MOVED bar-to-bar -> AR(1) noise
            # shifted the grid -> the noisy-quantity-gating mistake (ef027049 step3) ->
            # rally collapse. Fix: tie the grid to equity*BASE_POSITION_SIZE only — equity
            # is slow-moving (gradual) and BASE_POSITION_SIZE is constant, so the lattice
            # is stable across the noise ensemble (a perturbed bar barely moves equity).
            # Also finer (0.06) so rally's bidirectional fine resizes are less coarsened.
            if _is_resize and _churn_dz > 0.0:
                _grid = 0.06 * equity * BASE_POSITION_SIZE * _churn_dz
                if _grid > 0:
                    _qt = round(target / _grid) * _grid
                    if (_qt > 0) == (target > 0) and _qt != 0:
                        target = _qt
            # Architectural: COMPLEMENTARY low-churn-gated coarse grid (inverse-churn
            # partition of the order-emission layer). The existing grid above fires
            # ONLY in high churn (_churn_dz>0 at len>=2 = rally bursts); low-churn
            # regimes (crash/sideways/bull, whose entries are rare so len(_eh)<=1)
            # currently get ZERO grid quantization. The always-on grid branch
            # (4a40af0) proved a 0.06 lattice cuts turnover/fee cost in exactly these
            # stability-factor-1.0 regimes (crash raw +0.081, sideways raw +0.148) —
            # but uniform application killed rally (its fine bidirectional resizes
            # need the un-quantized continuum). This adds that same proven coarse
            # grid but gated on the COMPLEMENT: _calm_dz fires at len<=1 (the inverse
            # of _churn_dz), so it quantizes low-churn resizes (crash/sideways/bull)
            # while staying OFF in rally — where the existing fine grid keeps
            # operating. Same stable lattice (0.06*equity*BASE_POSITION_SIZE, equity
            # slow + BASE const = noise-stable lines), same noise-IMMUNE integer-churn
            # gate (just the other side), same resize-only exemptions, same
            # snap-toward-current_pos direction. The two grids are DISJOINT (churn_dz
            # and calm_dz are never both >0). New control flow: a second quantization
            # branch on the complementary churn partition.
            # Branch step 3: restore step-1's strong round()-grid (snap-to-hold in
            # step 2 was a weaker mechanism — it SUPPRESSED profitable resizes
            # wholesale, dropping crash/sideways BELOW baseline; the round-grid
            # preserves resize DIRECTION while cutting distinct-value count, which is
            # what produced the validated crash +0.047 / sideways +0.156 raw gains).
            # The step-1 round-grid's only failure was a rally leak: it fired at the
            # instantaneous len(_eh)<=1, but rally's LONE-entry scale-in resizes ALSO
            # sit at len=1 (diagnostic: crash max len=1, sideways max=2, but rally
            # reaches 5 in bursts). Fix the gate with a PERSISTENT churn signal: the
            # trailing MAX of len(_eh) over a 100-bar window per symbol. crash/sideways
            # NEVER burst (trailing-max stays <=2 the whole regime) so the grid fires
            # for them; rally/bull DO burst (trailing-max reaches >=3) and stay excluded
            # for the full 100-bar window AFTER each burst — sparing rally's quiet
            # post-burst bars that the instantaneous gate leaked. Noise-immune: max of
            # integer counts has no boundary that flips under AR(1) noise (same safety
            # property as len(_eh) itself). New per-symbol state (_churn_hist) + new
            # control flow: grid application gated on trailing-max churn.
            # Branch step 4: CUMULATIVE-max churn gate (replaces step-3's 100-bar
            # window). The window leaked rally because rally's entry CLUSTERS are
            # >100 bars apart — in the quiet stretches between clusters the trailing
            # max dropped <=2 and the coarse grid fired on rally's noise-sensitive
            # resizes (rally 0.000). Cumulative max never windows-out: once a symbol
            # demonstrates ANY entry burst (len(_eh)>=3 even once), the grid turns OFF
            # permanently for that symbol. crash (max len=1) and sideways (max len=2)
            # NEVER burst -> grid stays ON the whole regime (their validated raw gains
            # kept); rally/bull burst early -> grid OFF for the rest, including their
            # quiet stretches that the window leaked. Monotonic integer max =
            # noise-immune (no boundary that flips under AR(1)). Behavioral self-
            # measurement ("has this symbol ever churned"), NOT a date/market-state
            # classifier — generalizes to any persistently-calm vs bursty symbol.
            _cm = self._churn_hist.get(symbol, 0)
            _cm = max(_cm, len(_eh))
            self._churn_hist[symbol] = _cm
            _calm_gate = 1.0 if _cm <= 2 else 0.0  # fire only for never-bursting symbols
            if _is_resize and _calm_gate > 0.0:
                _grid_c = 0.06 * equity * BASE_POSITION_SIZE
                if _grid_c > 0:
                    _qt_c = round(target / _grid_c) * _grid_c
                    if (_qt_c > 0) == (target > 0) and _qt_c != 0:
                        target = _qt_c
            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    if current_pos != 0:
                        _ep = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                        self._last_exit_pnl[symbol] = -_ep if current_pos < 0 else _ep
                    for _d in (self.entry_prices, self.peak_pnl, self.entry_bar, self._smoothed_pnl, self._mae):
                        _d.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol], self.peak_pnl[symbol], self.entry_bar[symbol] = mid, 0.0, self.bar_count
                    self._mae[symbol] = 0.0
                    _h = self._entry_bar_history.setdefault(symbol, [])
                    _h.append(self.bar_count)

        return signals
