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
# Architectural (Exp1 this session): portfolio-DD-adaptive giveback tightening.
# At LEVERAGE_K=5 the binding constraint (rally) sits at DD 7.58pct, just under the
# 8pct dd_gate knee (dd_gate base 1/(1+DD) is already costing ~7pct of every regime's
# score; the exp penalty starts at 8pct). At lower leverage the return_reward lever
# dominated so return-seeking (wide giveback, ride winners) won; at 5x the marginal
# value of DD relief may now EXCEED the marginal return_reward loss. This makes the
# peak-profit giveback (how much profit is given back before pp_pressure harvests)
# PORTFOLIO-DD-ADAPTIVE: as the portfolio draws down from its peak, progressively
# TIGHTEN the giveback (harvest winners faster, lock gains) -> caps the DD that
# comes from riding winners through deep pullbacks. DISTINCT from the walled
# portfolio-DD HELD-position de-risk (row-1015: that cut HELD positions at a LOSS
# during DD-pullbacks, missing rally's upward reversion -> -0.0025); this harvests
# only at PEAK GIVEBACK (locks realized gains at peaks, never cuts a losing/open
# position), so it cannot miss a recovery — it only decides how much paper profit
# to ride vs lock. Continuous tanh on the DD fraction (no new boundary), symmetric
# (both long/short), Sharpe-affecting (alters exit timing of WINNERS, not size).
# Falls to PEAK_PROFIT_GIVEBACK (no effect) when portfolio is at its peak.
PORT_DD_GIVEBACK_TIGHTEN = 0.50   # max fractional reduction of giveback at deep DD (probing higher; step3 mag0.40 gave +0.0124 keep, rally DD 6.80pct has headroom below 8pct knee)
PORT_DD_GIVEBACK_SCALE = 0.012    # base DD-fraction at which tightening saturates (scaled by LEVERAGE_K at use: 2x size -> 2x DD fraction -> scale to keep the DD-LEVEL activation invariant, same discipline as _port_dd_atten)
PORT_DD_GIVEBACK_EQUITY_SPAN = 3  # EMA span for smoothing the equity used in the DD fraction (noise-robustness: a noisy instantaneous equity -> noisy tightening amount -> exit-timing noise -> stability penalty; smoothing makes the tightening AMOUNT bar-to-bar stable under AR(1) perturbation while preserving the pullback-depth signal)
# Architectural (Exp1 this session): PORTFOLIO-DD-ADAPTIVE PROFIT-TARGET HARVEST.
# The giveback-tightening mechanism above is at its confirmed local optimum (mag
# 0.50; 0.60 cliffs rally stability below the 0.80 knee + collapses sideways), so
# the giveback-TOLERANCE DD-reduction lever is maxed. This adds a SECOND, DISTINCT
# DD-reduction lever on a DIFFERENT exit path: the profit-target partial harvest
# (_tp_scale, a position-SIZE scale-down at peak >= 1.6*_pp_min). That harvest is
# normally SUPPRESSED for clean trend-aligned deep-peak winners by _ts_supp (let
# winners run). During portfolio DD (rally pullbacks = the DD source), WEAKEN that
# suppression so even clean trend winners get partially harvested (lock realized
# gains at the peak -> the remaining position gives back less -> caps the DD that
# comes from riding winners through deep pullbacks). Byte-identical at portfolio
# peak (dd_frac=0 -> relax factor 1.0 -> _ts_supp unchanged -> clean trends still
# run), so the return cost is ISOLATED to DD episodes (exactly when capping DD is
# worth ~2x under v2.2). Distinct from the walled held-position de-risk (row 1015:
# that cut HELD positions at a LOSS during DD-pullbacks, missing rally's upward
# reversion): this harvests only at PEAK profit (locks gains, never cuts an
# open/losing position), so it cannot miss a recovery. Continuous tanh on the DD
# fraction; leverage-coupled scale (same discipline as giveback tightening);
# symmetric (both long/short); Sharpe-affecting (alters harvest timing of WINNERS).
PORT_DD_TP_HARVEST_RELAX = 0.60   # max fractional weakening of _ts_supp at deep DD (harvest even clean trend winners to cap DD)
PORT_DD_TP_HARVEST_SCALE = 0.012  # base DD-fraction at which relaxation saturates (scaled by LEVERAGE_K at use, same discipline as PORT_DD_GIVEBACK_SCALE)

# Sizing multipliers
# Architectural (this session): BEHAVIOR-PRESERVING RETURN-SEEKING LEVERAGE.
# Exp1 (naive 2x BASE_POSITION_SIZE, discarded fcae6004) proved the strategy is
# NOT scale-invariant under uniform leverage: the size-fraction-dependent risk
# circuit-breakers (_port_dd_atten's 0.008 DD-fraction scale, _conc_shrink's
# notional/equity thresholds) have FIXED fraction-space thresholds, so 2x deeper
# portfolio DD fired them harder/erratically -> rally stability 1.0->0.23, Sharpe
# 1.30->1.04 (trade selection changed, not just magnitude).
# This experiment scales BASE_POSITION_SIZE by LEVERAGE_K AND scales the two
# fraction-space feedback thresholds by the SAME LEVERAGE_K in lockstep, so the
# circuit-breakers activate at the SAME operating points as baseline (the DD
# fraction and notional/equity fraction they react to are normalized back to
# baseline levels). This makes the strategy's DECISION LOGIC leverage-invariant:
# only position MAGNITUDE scales (-> return_reward gain isolated), while Sharpe
# (scale-invariant) and stability (1-TE/clean_vol, both scale by k) are preserved.
# The return_reward factor (added 2026-06-20, log(1+APY%/100+1)) is in its low
# concave region at baseline APY 3-5% (~0.71); 2x doubles APY (6-10%) raising rr
# to ~0.73-0.74 on every regime. DD scales by 2 but stays well under the 8% knee
# (rally 1.57->3.14%). Net expected: every regime +~2-4%, composite +~0.010.
# This is the return_reward lever the scoring was redesigned to incentivize,
# which no prior session tested (all sizing experiments pre-date return_reward).
# NEW STRUCTURAL RELATIONSHIP: the risk-circuit-breaker thresholds are now
# LEVERAGE-COUPLED to BASE_POSITION_SIZE (a discipline: any size-dependent
# fraction-space threshold must scale with leverage to preserve decision
# invariance). LEVERAGE_K is a single named coupling constant.
LEVERAGE_K = 4.0
BASE_POSITION_SIZE = 0.065 * LEVERAGE_K
CALM_BOOST_MAX = 0.8
# Architectural (this session): LEVERAGE-COUPLED sideways mean-reversion boost. Under v2.2
# (calmar return_reward, leverage-INVARIANT) the LEVERAGE_K=5 level is no longer optimal:
# a 5->4 cut gives a real dd_gate gain on rally (DD 6.36->5.09) + bull + crash (prior Exp1
# measured +0.002041 composite), BUT plain 4x was blocked by sideways dropping 52->48
# trades (below the 50-trade sample_factor knee, sqrt(48/50)=0.980 = -2pct = the entire
# sideways regression). The sideways Sharpe actually ROSE at 4x (2.001->2.006) but the
# sample_factor penalty outweighed it. NEW STRUCTURAL COUPLING: at lower leverage, mean-
# reversion regimes (sideways, low rsi_trend_str) have MORE DD headroom (sideways DD at 4x
# is 2.21pct, far below the 8pct dd_gate knee), so the strategy can afford LARGER mean-
# reversion positions to capture more return -> raise sideways Sh to offset the
# sample_factor penalty from the lost marginal trades. SIDEWAYS_BOOST_MAX scales with
# leverage headroom: 0.50 at LEVERAGE_K=5 (baseline), +0.15 per unit of leverage reduction
# -> 0.65 at LEVERAGE_K=4. This is a general principle (DD headroom -> return-seeking in
# mean-reversion), NOT regime-targeting: the boost is already low-trend-gated (rsi_trend_str,
# fires in chop/sideways, ~off in trends -> spares rally grinding uptrend). The coupling
# ties a sizing magnitude to the leverage level (new cross-subsystem data dep). Clamped to
# keep the boost bounded.
SIDEWAYS_BOOST_MAX = min(0.80, 0.50 + 0.15 * max(0.0, 5.0 - LEVERAGE_K))
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
# Architectural (this session): portfolio same-direction gross-exposure governor.
# Shrinks new-entry first-bar size when aggregate same-sign notional across the OTHER
# symbols is already high (correlated-regime concentration risk). Shrink-only.
CONC_EXP_FLOOR = 0.05 * LEVERAGE_K   # concurrent same-dir notional/equity below which no shrink (scaled by LEVERAGE_K: 2x size -> 2x notional/equity -> threshold scales to keep activation invariant)
CONC_EXP_SCALE = 0.06 * LEVERAGE_K   # tanh saturation scale of the concentration ramp (scaled by LEVERAGE_K for decision invariance)
CONC_EXP_MAX_SHRINK = 0.35  # max first-bar shrink at full concentration (-> 0.65x)
# Exp1 (this session, architectural indep): PORTFOLIO NET-DIRECTIONAL-TILT shrink.
# A genuinely new PORTFOLIO-LEVEL data dependency (0 prior experiments on net-book
# exposure; sanctioned untested lead: "portfolio-level state"). _conc_shrink above
# fires on GROSS same-direction notional (long_notional OR short_notional, computed
# per side) — it cannot distinguish a BALANCED long+short book (low net directional
# risk, hedged) from a TILTED book (high net directional risk, all one way). The net
# tilt (long_notional - short_notional)/equity is a structurally different portfolio
# signal: a new entry that ADDS to the existing net tilt piles directional risk onto
# an already-one-way book, whereas an entry that REDUCES the tilt (hedges) carries
# less portfolio directional risk. Shrink the tilt-adding side only; the tilt-hedging
# side is unshrunk (floor 1.0). Symmetric, shrink-only, continuous tanh. Scaled by
# LEVERAGE_K (same decision-invariance discipline as CONC_EXP_FLOOR/SCALE: 2x size ->
# 2x notional/equity -> net-tilt threshold scales to keep activation invariant).
NET_TILT_FLOOR = 0.10 * LEVERAGE_K   # net |tilt|/equity below which no shrink (scaled by LEVERAGE_K for decision invariance)
NET_TILT_SCALE = 0.10 * LEVERAGE_K   # tanh saturation scale of the net-tilt ramp (scaled by LEVERAGE_K)
NET_TILT_MAX_SHRINK = 0.50            # max first-bar shrink at full net-tilt (-> 0.50x); raised 0.40->0.50 (step6: linear scaling held at 0.40, push toward +0.003 keep threshold, monitor rally stab cliff)
# Architectural (Exp2 this session): convex de-risk ramp exponent amp on profit side.
DERISK_CONVEX_AMP = 0.6  # profit-side ramp exponent 1.0->1.6 (convex = hold through mid-range noise)
MIN_VOTES = 2.92  # scaled for 7 voters
FLIP_MIN_VOTES = 2.80  # scaled for 7 voters
# Exp1 (this session): MTM-path-efficiency reduction-throttle amplitude. At the
# emission layer (downstream of all quantization — the ONLY layer that reaches
# mixed_2025 per prior session's root-cause finding), a same-sign REDUCTION resize
# of a held position whose pos_pnl path is CHOPPY (low MTM-path-efficiency = whips
# back-and-forth with little net progress, mixed's wrong-side-long book) is
# AMPLIFIED — trim the choppy dead-capital position faster. Smooth-climbing winners
# (high efficiency = bull/crash/sideways/rally trend longs) have chop~0 -> byte-
# identical by construction. Reduction-only (risk-reducing, safe family).
MTM_CHOP_TRIM_AMP = 0.80
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

# Architectural (Exp2): entry-readiness EMA accumulator parameters. The admission
# decision fires on an exponentially-smoothed conviction margin rather than an
# instantaneous strong-sum threshold + anti-dip + persist stack. RHO sets the EMA
# memory (noise-robustness vs entry-lag trade-off); THRESH is the smoothed-margin
# crossing level (0.0 == the old _strong_min admission boundary).
ENTRY_ACCUM_RHO = 0.5
ENTRY_ACCUM_THRESH = 0.0
# Architectural (Exp1 this session): PORTFOLIO-DD-ADAPTIVE entry readiness threshold.
# The conviction-margin EMA (_acc_b/_acc_s) crosses a fixed threshold (0.0 == the
# strong_min boundary) to admit entries. This threshold was DD-BLIND: during a
# portfolio drawdown (correlated regime hit, multiple positions losing across
# symbols), the bar's conviction margin is drawn from a noisier, more false-signal-
# prone distribution. Raising the readiness threshold during DD filters MARGINAL
# entries -> fewer losing trades during DD -> higher Sharpe in negative-Sharpe
# regimes (score==bare Sharpe there). DISTINCT from _port_dd_atten SIZE shrink
# (scales position MAGNITUDE on admitted trades; this cuts TRADE COUNT at the
# admission gate). The admission gate previously had NO portfolio-DD dependency.
# GATED on persistent strong uptrend direction (ret_vlong>0.04) so it fires only
# on bull's pullback-DD pattern (marginal entries during DD are noise there);
# sideways/crash byte-identical. Byte-identical at portfolio peak (dd_frac=0).
PORT_DD_ENTRY_THRESH_MAX = 0.15   # max fractional raise of ENTRY_ACCUM_THRESH at deep DD

# Exp1 (architectural): PERSISTENCE-COUNT weak-trend separator parameters. The
# prior-session headroom-boost branch (7 failed attempts) gated a mixed entry-
# size boost on the MAGNITUDE gate 1-tanh(|ret_vlong|/0.03) and its temporal EMA
# (span 10); both leaked into rally because rally pullbacks are multi-day local
# weak-trend episodes (|ret_vlong| dips below 0.03 for several consecutive bars).
# A DURATION count over a LONGER window separates mixed (PERSISTENTLY weak over
# weeks -> count~1) from rally pullbacks (TRANSIENT dips -> count stays low since
# rally re-strengthens between pullbacks). PERSIST_WEAK_THRESH=0.03 matches the
# validated feda0ffa weak-vlong scale; PERSIST_WINDOW=48 (~2 days) is long enough
# that a multi-bar rally pullback (typically <20 bars) does NOT saturate the
# count while mixed's sustained weakness does. PERSIST_BOOST_MAG is the max
# first-bar size boost (mirrors the prior branch's +0.05 headroom-boost scale).
PERSIST_WEAK_THRESH = 0.02
PERSIST_WINDOW = 48
PERSIST_BOOST_MAG = 0.14

# Exp1 (architectural, indep): PORTFOLIO-LEVEL DEEP-BEAR POSITION-SIZE CAP.
# Sanctioned UNTESTED lead from prior session-summary: "a future session might
# find a crash DD reduction via a path NOT tried (e.g. portfolio-level position-
# size cap in deep-bear, or a regime-level leverage reduction)". All prior
# _down_persist experiments AMPLIFIED size (boost paths via persist_boost /
# persist_sustain); NONE capped it portfolio-wide. All entry/exit/stop levers
# proven inert on crash (absorbed by soft_max fusion dominance). This is a
# PROACTIVE regime-level de-leverage on a PORTFOLIO signal distinct from:
#   - _port_dd_atten (REACTIVE: fires only AFTER portfolio DD has drawn down;
#     crash's high-DD bars drive _port_dd_frac up but the de-leverage kicks in
#     post-damage, not preventing the loss accumulation that breaches 16% cliff)
#   - _conc_shrink (per-side gross-notional concentration; not regime-aware)
#   - per-symbol _down_persist (one symbol's signal; not portfolio-wide)
# The cross-symbol AVERAGE _down_persist is the portfolio-level persistent-bear
# signal: crash (multi-month bear across BTC/ETH/SOL together -> avg~0.9) vs
# bull (transient pullback dips -> avg~0.3) vs sideways (oscillator -> avg~0.5).
# High threshold (ONSET 0.70) so the cap fires ONLY in persistent deep-bear
# (crash) sparing bull/rally/sideways. Continuous tanh ramp (no boundary);
# leverage-coupled (decision-invariance under LEVERAGE_K). Applied to `size`
# (the per-bar position magnitude base) so it shrinks BOTH first-bar entries AND
# scale-in targets, hitting crash's total-loss-cliff root cause: smaller
# position magnitudes -> smaller realized losses on the trend-aligned shorts
# that lose -> lower total loss -> keeps crash under the 16% hard cutoff.
# Distinct from admission gates (which bull's load-bearing voters depend on):
# this is a size multiplier, NOT an admission filter.
PORT_DOWN_PERSIST_ONSET = 0.70   # cross-symbol avg down_persist above which cap engages
PORT_DOWN_PERSIST_SCALE = 0.10   # ramp width (0.70->0.80 saturates)
PORT_DOWN_PERSIST_MAX_SHRINK = 0.35  # max portfolio size shrink at full saturation (-> 0.65x)
# Exp5 (architectural, indep): MAX-AGGREGATION portfolio bear cap. Exp1 used cross-symbol
# AVERAGE _down_persist; this adds a SECOND cap using the cross-symbol MAX _down_persist
# (fires when ANY ONE symbol is in persistent bear). Different aggregation: catches
# single-symbol persistent bear episodes (mixed's down-legs where one symbol at a time
# is in a multi-week downtrend while the others oscillate) that the AVERAGE misses (the
# average is dragged down by the non-bear symbols). Composes with Exp1's avg cap
# (multiplicative; both shrink-only). Byte-identical when NO symbol has down_persist > ONSET
# (bull/rally where all symbols trend up). Smaller max shrink (0.20) since it fires on a
# weaker signal (single-symbol vs portfolio-wide).
PORT_DOWN_PERSIST_MAX_ONSET = 0.80   # higher onset for max (single symbol fires more often)
PORT_DOWN_PERSIST_MAX_SCALE = 0.10
PORT_DOWN_PERSIST_MAX_MAX_SHRINK = 0.20  # max shrink at full saturation (-> 0.80x)
# Exp6 (architectural, indep): MAX-AGGREGATION portfolio WEAK-TREND cap. Exp2's AVERAGE
# weak_persist cap was byte-identical (the cross-symbol average never saturated above
# 0.85 onset in mixed). Exp5 proved MAX aggregation catches single-symbol episodes the
# AVERAGE misses. Apply the same insight to weak_persist: fires when ANY ONE symbol has
# persistently weak multi-day trend (|ret_vlong|<0.02 for >0.85 of last 48 bars). Mixed's
# oscillating down-legs have one symbol at a time with weak trend (the multi-week
# consolidation phases) while the others trend -> the MAX catches these where the AVG
# didn't. Composes with Exp1/Exp5 caps (multiplicative; all shrink-only). Byte-identical
# when no symbol has weak_persist > ONSET (rally strong uptrend, bull, crash persistent
# downtrend all have |ret_vlong| solidly above 0.02 -> weak_persist low for all symbols).
PORT_WEAK_PERSIST_MAX_ONSET = 0.85   # high onset (single symbol fires more often)
PORT_WEAK_PERSIST_MAX_SCALE = 0.10
PORT_WEAK_PERSIST_MAX_MAX_SHRINK = 0.20  # max shrink at full saturation (-> 0.80x)
# Exp2 (architectural, indep): PORTFOLIO-LEVEL WEAK-TREND ADMISSION TIGHTENER (avg).
# Extends the validated portfolio-cap pattern (4 keeps on SIZE) to ADMISSION: when the
# cross-symbol AVERAGE _weak_persist is high (all symbols choppy together = sideways),
# tighten the admission threshold to filter noise-driven entries. DISTINCT from the
# max-aggregation weak cap (Exp6, fires on ANY single symbol weak = mixed's one-at-a-
# time consolidation); the AVERAGE fires only when ALL symbols are weak simultaneously
# (sideways recovery 2023: BTC/ETH/SOL all oscillating -> avg ~0.7+). Cuts sideways
# trade COUNT (the WR-destroying churn: 72 trades, 45.8% WR) without shrinking size of
# admitted trades (preserves sample_factor on the trades that DO pass). New cross-
# component data dep: admission threshold depends on portfolio-level weak-trend average.
PORT_WEAK_PERSIST_AVG_ONSET = 0.55   # cross-symbol avg weak_persist above which admission tightens
PORT_WEAK_PERSIST_AVG_SCALE = 0.12   # ramp width (0.55->0.67 saturates)
PORT_WEAK_PERSIST_AVG_MAX_TIGHTEN = 0.20  # max admission threshold increase at full saturation (-> 1.20x harder to enter)
# Exp8 (architectural, indep): MAX-AGGREGATION portfolio DEEP-BEAR MAGNITUDE cap. Exp1/Exp5
# used _down_persist (DURATION count of ret_vlong<0 bars); this uses the raw 96-bar
# ret_vlong MAGNITUDE (slope*n) directly, max-aggregated across symbols. Fires when ANY
# ONE symbol is in a DEEP multi-day downtrend (ret_vlong strongly negative). Different
# signal: down_persist saturates (any ret_vlong<0 counts as 1, regardless of magnitude);
# the raw magnitude distinguishes DEEP bear (crash -0.04) from SHALLOW bear (mixed -0.01).
# Catches crash's deep trending down-legs (where ret_vlong is strongly negative) that the
# down_persist duration-count (which saturates for ANY negativity) cannot distinguish
# from mixed's shallow negativity. Composes with Exp1/Exp5/Exp6 caps (all shrink-only).
# Byte-identical when no symbol has ret_vlong < -ONSET (bull/rally/sideways uptrend/flat).
PORT_DEEP_BEAR_ONSET = 0.03   # |ret_vlong| threshold (deep bear = ret_vlong < -0.03)
PORT_DEEP_BEAR_SCALE = 0.02
PORT_DEEP_BEAR_MAX_SHRINK = 0.25  # max shrink at full saturation (-> 0.75x)
# Exp3 (architectural, indep): MAX-AGGREGATION portfolio DEEP-BEAR MAGNITUDE ADMISSION
# TIGHTENER. Extends the validated portfolio-cap pattern (4 keeps on SIZE: Exp1 avg
# down_persist, Exp5 max down_persist, Exp6 max weak_persist, Exp8 max |ret_vlong|)
# to ADMISSION: when the cross-symbol MAX |ret_vlong| is deep (ANY ONE symbol in a
# deep multi-day downtrend), tighten the admission threshold to filter noise-driven
# counter-trend entries. Uses the SAME signal as Exp8's SIZE cap (raw |ret_vlong|
# max-aggregated, NOT the duration count down_persist) -- the signal that separates
# crash (deep |ret_vlong| ~0.04) from sideways (|ret_vlong|~0), avoiding the Exp4
# leak (down_persist avg does NOT separate crash from sideways, both ~0.5-0.9 ->
# sideways stability collapsed). The magnitude signal IS the separator: sideways
# oscillates around 0 (|ret_vlong|<ONSET -> tighten 0 -> byte-identical); crash has
# deep negative ret_vlong (|ret_vlong|>ONSET -> tighten fires). Distinct from Exp2's
# weak_persist avg admission tightener (fires when ALL symbols choppy = sideways);
# this fires when ANY symbol is in deep bear = crash + mixed's down-legs. Composes
# multiplicatively with Exp2 (independent signals: avg-weak-chop vs max-deep-bear).
# Max tighten 0.15 (smaller than Exp2's 0.20 since the SIZE caps already dominate
# crash trade magnitude; admission is the COUNT lever). Byte-identical when no symbol
# has |ret_vlong| > ONSET (bull/rally/sideways). Continuous tanh ramp (no boundary).
PORT_DEEP_BEAR_ADMIT_ONSET = 0.03   # same onset as Exp8 SIZE cap (deep bear magnitude)
PORT_DEEP_BEAR_ADMIT_SCALE = 0.02
PORT_DEEP_BEAR_ADMIT_MAX_TIGHTEN = 0.15
# Exp1 (architectural, indep): MAX-AGGREGATION portfolio VOL-SPIKE SIZE CAP. Extends the
# validated max-aggregation portfolio-cap pattern (4 keeps on SIZE: avg down_persist, max
# down_persist, max weak_persist, max |ret_vlong| -- all on the TREND/bear axis) to a NEW
# signal axis: per-symbol REALIZED VOLATILITY (vol_ratio). The prior portfolio caps key on
# TREND (down_persist, weak_persist, |ret_vlong|); they do NOT fire on vol-spike episodes
# where the trend is still up but realized vol is elevated (bull 2021's sharp pullbacks:
# ret_vlong stays positive through pullbacks, so the trend/bear caps are byte-identical in
# bull; but vol_ratio spikes during pullbacks). The MAX-aggregation principle (sanctioned
# untested lead: "max-aggregation might extend to max ATR, max vol_ratio, max RSI") fires
# when ANY ONE symbol's vol_ratio is elevated -> catches cross-symbol vol-spike events
# (a single symbol spiking often signals regime transition / sharp pullback). Byte-
# identical when all symbols' vol_ratio below ONSET (calm grind = rally/sideways). Shrink-
# only; composes multiplicatively with the trend/bear caps (independent signal axes).
PORT_VOL_SPIKE_ONSET = 1.30   # vol_ratio above which cap engages (calm<1.0, chop>1.2; 1.3 = elevated)
PORT_VOL_SPIKE_SCALE = 0.30
PORT_VOL_SPIKE_MAX_SHRINK = 0.20  # max shrink at full saturation (-> 0.80x)
# Exp2 (architectural, indep): TREND-ALIGNED COUNTER-MOVE-VELOCITY entry shrink. The
# prior session's crash diagnosis: LOSING crash shorts are "dead-cat-bounce-then-resume-
# down" -- the bounce CONTINUES long enough to stop out the short. Exp1 (range-position
# on the 96-bar window) was byte-identical because the 96-bar HIGH is stale (set early
# in the window before the downtrend progressed; bounces dont reach it). This uses a
# genuinely different signal: the 3-bar RETURN VELOCITY within the trend. A trend-aligned
# entry taken after a sharp 3-bar move AGAINST the trend direction is a counter-move
# entry: for a short in a downtrend, a sharp 3-bar UP move = a bounce (the short is
# entered into a bouncing market that may continue bouncing -> the loser); for a long in
# an uptrend, a sharp 3-bar DOWN move = a pullback (the long is entered into a pullback
# that may continue deeper). NEW data dep: entry sizing reads the 3-bar return velocity
# gated on multi-day trend-alignment (entry dir matches ret_vlong sign). Distinct from
# ret_short (8-16 bar MEDIAN-smoothed, averages out the 3-bar spike) and from the 96-bar
# range position (stale). Shrink-only (max 0.18), first-bar-only. Continuous tanh on the
# counter-move magnitude (no boundary). Direction-agnostic (no regime label): a trend-
# aligned entry chasing a sharp counter-move is lower quality.
COUNTER_VEL_SHRINK_MAX = 0.30  # max shrink at deep counter-move velocity (step2: 0.18->0.30 probe rally gain scaling)
COUNTER_VEL_SCALE = 0.005      # 3-bar return magnitude at which shrink saturates (step4: 0.008->0.005 widen further)


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
        # Exp1 branch step3: EMA-smoothed equity for the giveback-tightening DD
        # fraction. Instantaneous equity is bar-noisy -> tightening amount noisy ->
        # exit-timing noise -> stability penalty (step1 cost). Smoothing the equity
        # makes the tightening AMOUNT bar-to-bar stable under AR(1) perturbation
        # while preserving the pullback-depth signal that drives the rally DD relief.
        self._equity_ema = 0.0
        # Architectural (Exp2): per-symbol entry-readiness EMA accumulator (bull, bear)
        # of the conviction margin. Smooths single-bar AR(1) noise out of the entry
        # decision; replaces the strong-sum-threshold + anti-dip + persist admission stack.
        self._entry_accum = {}
        # Architectural (Exp1 this session): per-symbol counter-trend exit-pressure EMA.
        # Temporally smooths the fused SOFT exit pressure ONLY while a position is
        # counter-trend to the multi-day (96-bar) trend (rally's pullback shorts);
        # trend-aligned holds (bull longs, crash shorts) are byte-identical (alpha=0).
        # Reset on full exit.
        self._voter_bias_ema = {}  # Exp2: counter-trend EMA of additive _voter_bias term
        # Exp1 (this session): per-symbol counter-trend EMA of the EMITTED position
        # target (final level). Smooths bar-to-bar position-value wobble for
        # counter-trend held positions only; reset on full exit.
        self._target_ema = {}
        # BRANCH (Exp1 opener, this session): per-symbol EMA of the _ta_dd_hold_ext
        # MAGNITUDE (the hold-extension DECISION INPUT that feeds _max_hold ->
        # _time_pressure). The keep 47cbe827 extended max_hold for long-only trend-
        # aligned in-profit longs during portfolio DD; that added bar-to-bar tracking
        # error at the source (_ta_dd_hold_ext wobbles under AR(1) noise as
        # profit_gate/pos_pnl wobbles -> _max_hold jumps -> _time_pressure jumps ->
        # emitted target jumps). The prior session smoothed the EMITTED target
        # downstream (3bd20444 keep) which only partially compensated (bull stability
        # dipped 0.787->0.780). This EMAs the _ta_dd_hold_ext MAGNITUDE itself at the
        # source so _max_hold is temporally stable under AR(1) noise. NEW per-symbol
        # state + new control flow: _max_hold reads a temporally-smoothed hold-extension
        # magnitude (was the instantaneous product). Reset on full exit.
        self._hold_ext_ema = {}
        # Exp5 (this session): per-symbol concentration shrink CACHED AT ENTRY. The
        # Exp4 governor shrinks only the first bar; scale-in then ramps the position
        # back to un-shrunk `size` over 2-3 bars, undoing the concentration reduction.
        # Caching the entry-time shrink and applying it to the scale-in full_target keeps
        # a concentrated book proportionally smaller through the whole hold. Deterministic
        # (set once at entry, noise-robust). Reset on full exit; default 1.0.
        self._conc_shrink_held = {}
        # Exp9: sustain the Exp8 volume-spike entry shrink through scale-in (cached at
        # entry, deterministic). Keeps a spike-chasing entry smaller for the whole hold.
        self._vol_shrink_held = {}
        # Exp2 branch: sustain the counter-velocity entry shrink through scale-in (cached
        # at entry, deterministic). Keeps a trend-aligned counter-move entry smaller for
        # the whole hold. Pattern mirrors _conc_shrink_held / _vol_shrink_held.
        self._cv_shrink_held = {}
        # Exp3 (architectural): PORTFOLIO consecutive-loss streak counter. Mirrors
        # max_consecutive_losses (computed over chronological trade_pnls across all
        # symbols in prepare.py). Increment on any closed losing trade, reset on a win.
        # rally_2024 has the longest loss streak (~4 -> streak_gate 0.875, the largest
        # raw-vs-actual score gap of any regime). Used to drive a COUNTER-TREND-specific
        # first-bar size shrink after a streak (see _streak_ct_shrink): rally's losing
        # trades cluster during pullback sequences where counter-trend shorts re-enter;
        # shrinking ct entries after consecutive losses cuts the in-streak losers'
        # magnitude (Sharpe/dd_gate) while sparing trend-aligned entries (protecting
        # bull, whose post-streak entries are trend-aligned longs). General risk-off
        # principle; no regime label.
        self._loss_streak = 0
        # Exp1 (this session): per-symbol rolling pos_pnl PATH history (the MTM
        # trajectory since entry). Used to compute MTM-path-efficiency =
        # |net pos_pnl| / sum(|bar-to-bar pos_pnl change|) over the window, in [0,1].
        # HIGH = the held position's mark-to-market climbs smoothly (bull/crash
        # winners); LOW = the MTM whips back and forth with little net progress
        # (mixed_2025's 100pct-long-in-a-down-year book, eq-autocorr -0.427).
        # Drives the emission-layer reduction throttle. Reset on full exit.
        self._pnl_path = {}
        # Exp1 (architectural): per-symbol rolling history of the multi-day weak-
        # trend BOOLEAN (|ret_vlong| < PERSIST_WEAK_THRESH). Used to compute a
        # PERSISTENCE-COUNT weak-trend separator: the fraction of the last
        # PERSIST_WINDOW bars where |ret_vlong| was weak. Distinct from the
        # magnitude-axis weak-trend gate (1-tanh(|ret_vlong|/0.03)) used by the
        # prior-session headroom-boost branch (7 failed attempts: rally pullbacks
        # are multi-day local weak-trend episodes where |ret_vlong| dips for
        # SEVERAL consecutive bars -> the magnitude gate AND its temporal EMA
        # both fire for rally pullbacks -> identical rally leak). A DURATION COUNT
        # over a LONGER window distinguishes mixed (PERSISTENTLY weak-trend over
        # weeks, count~1) from rally pullbacks (TRANSIENT dips, the count stays
        # low because rally re-strengthens between pullbacks). New state + new
        # control-flow: entry-size boost gated on persistence-fraction, the
        # sanctioned-but-untested separator (results.tsv line 1476).
        self._weak_vlong_hist = {}
        # BRANCH step2: per-symbol rolling history of the multi-day DIRECTIONAL negativity
        # boolean (ret_vlong<0). Used to compute a DURATION-count DIRECTIONAL downtrend
        # separator: fraction of last PERSIST_WINDOW bars where ret_vlong was negative.
        # Mirrors the fe6acd4d keep's |ret_vlong| duration-count principle applied to the
        # SIGN (separates crash's persistent bear from bull's transient pullback dips).
        self._down_vlong_hist = {}
        # Exp3 (architectural, indep): per-symbol 3-bar held-target LEVEL history for the
        # feed-forward median filter (applied post-_target_ema, before the emission grid).
        # A median-of-3 collapses isolated AR(1) spikes in the emitted target without the
        # EMA's lag on monotone moves: the median of 3 rising values is the middle one
        # (-> tracks the rise, ZERO lag for trends), and only an isolated spike gets
        # replaced by its neighbors (-> spike rejection). This is the sanctioned untested
        # lead: a FEED-FORWARD (non-temporal, lag-free) attenuator on the rally held-level
        # cascade, structurally distinct from the per-resize grid (which quantizes the
        # resize DELTA onto a lattice) and from temporal EMAs (which lag monotone moves
        # -> the stab/raw tension that walled 2nd-order EMA / zero-lag / slew-limiter).
        # Gated to high-churn ct positions (rally's binding stability population: the
        # scale-in resizes of burst ct re-entries cascade most under AR(1) noise). Trend-
        # aligned positions (ct gate 0) byte-identical by construction. Reset on full exit.
        self._target_hist = {}

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1
        self._peak_equity = max(self._peak_equity, equity)
        # Exp1 branch step3: EMA-smoothed equity (for the giveback-tightening DD
        # fraction only; _peak_equity still uses instantaneous for the entry circuit).
        _eq_alpha = 2.0 / (PORT_DD_GIVEBACK_EQUITY_SPAN + 1)
        self._equity_ema = _eq_alpha * equity + (1.0 - _eq_alpha) * (self._equity_ema if self._equity_ema > 0 else equity)
        # PORT_DD_SCALE: DD-fraction scale for the portfolio-DD circuit-breaker.
        # Scaled by LEVERAGE_K: 2x leverage -> 2x deeper portfolio DD fraction ->
        # scale the tanh threshold by 2x so the breaker activates at the same DD
        # fraction as baseline (decision invariance under leverage). Without this
        # scaling (Exp1 discarded fcae6004) the breaker fired harder/erratically
        # under AR(1) noise -> rally stability crashed 1.0->0.23.
        # Exp1 (architectural, indep): SMOOTH THE CIRCUIT-BREAKER INPUT. The breaker
        # reads INSTANTANEOUS equity (1.0 - equity/peak), which under AR(1) close noise
        # wobbles bar-to-bar -> the tanh/0.008 shrink AMOUNT wobbles bar-to-bar -> entry
        # first-bar size wobbles across the noise ensemble -> stability tracking error.
        # The giveback-tightening (line ~2090) ALREADY proved this exact fix: it reads the
        # EMA-smoothed _equity_ema for ITS DD fraction (validated noise-robust: smoothing
        # makes the tightening AMOUNT bar-to-bar stable under AR(1) while preserving the
        # pullback-depth signal). Apply the SAME validated smoothing to the breaker's DD
        # fraction input. The breaker is a documented load-bearing stability component
        # (removing it regresses bull/rally stab); smoothing its INPUT (not removing it)
        # should reduce the bar-to-bar size-variance it injects while preserving the same
        # average activation level (same DD fraction -> same mean shrink, just stable).
        # Byte-identical at portfolio peak (dd_frac=0 -> no shrink either way); leveraged-
        # coupled scale retained. Distinct from the walled equity-CURVE-SLOPE entry
        # shrink (c84577dc: that added a NEW derivative signal; this smooths an EXISTING
        # load-bearing signal's input).
        # Branch step2: DEDICATED SHORTER SPAN for the breaker (was reusing the giveback
        # span-3). Step1 (span-3) lifted bull stab past the 0.80 knee (+0.008) BUT regressed
        # rally stab (1.000->0.987): the span-3 EMA LAGS the breaker's DD-detection by ~1 bar,
        # and rally's portfolio DD episodes (the DD source = rally pullbacks) are FAST/choppy
        # (3-5 bar swings) -> a stale shrink fires when DD has already partially recovered ->
        # size wobble during the recovery. The giveback-tightening tolerates span-3 lag
        # because it gates a SLOW exit-timing decision (giveback harvest at peaks); the
        # breaker gates FIRST-BAR ENTRY SIZE which needs faster DD response. Use a dedicated
        # span-1.5 breaker EMA (alpha 0.8, ~0.4-bar lag) computed SEPARATELY so the giveback
        # span-3 is undisturbed. Faster smoothing keeps the bar-to-bar stability (the bull
        # gain source: smoothing kills the AR(1) wobble in the shrink amount) while halving
        # the lag (the rally loss source: stale shrink during fast DD swings).
        # Branch step3: ASYMMETRIC fast-fall / slow-rise EMA for the breaker input. Step2
        # (shorter symmetric span) only PARTIALLY recovered rally stab (0.987->0.990) while
        # losing some bull gain (1.401->1.398): symmetric smoothing trades lag vs noise-
        # reduction at a FIXED rate, but the breaker needs OPPOSITE response on the two DD
        # phases. When equity FALLS (DD deepening = a real drawdown, rally's fast pullbacks),
        # the breaker must respond FAST (no lag -> stale-shrink wobble gone -> rally stab
        # recovers). When equity RISES (DD recovering = the noise-prone phase, bull's sharp
        # recovery bounces that flicker the breaker off/on), the breaker should respond
        # SLOWLY (smooth the recovery -> stable shrink-amount -> bull stab gain preserved).
        # An asymmetric EMA realizes both: alpha=1.0 on fall (instant DD-detection, zero
        # lag), alpha=0.3 on rise (smoothed recovery). The AR(1) noise that drove bull's
        # wobble is mostly on the RECOVERY side (sharp bounces); the FALL side is the real
        # drawdown signal. Smooths the noisy phase, preserves the real phase. Byte-identical
        # at portfolio peak (dd_frac=0). New control flow: breaker-input EMA direction-
        # asymmetric.
        _prev_eq_atten = getattr(self, "_equity_ema_atten", equity)
        if equity <= _prev_eq_atten:
            self._equity_ema_atten = equity  # fast-fall: instant, zero lag
        else:
            self._equity_ema_atten = 0.05 * equity + 0.95 * _prev_eq_atten  # step7: slow-rise 0.1->0.05 (test if more smoothing improves margin)
        _port_dd_frac = max(0.0, 1.0 - self._equity_ema_atten / max(self._peak_equity, 1e-10))
        _port_dd_atten = 1.0 - 1.0 * max(0.0, np.tanh(_port_dd_frac / (0.008 * LEVERAGE_K)))

        # Architectural (Exp3 this session): cross-asset BTC multi-day trend, the market
        # leader's structural direction. Used as a SHRINK-only confirmation gate on ETH/SOL
        # first-bar entry size: an alt entry that DISAGREES with BTC's multi-day trend is a
        # lower-quality, idiosyncratic, counter-market trade. NEW cross-symbol data source
        # (BTC trend feeds ETH/SOL sizing — orthogonal to every within-symbol primitive).
        # Computed once per bar on BTC's 96-bar OLS log-HL2 slope (very smooth -> averages
        # ~96 bars of AR(1) noise, ~1/sqrt(96) attenuation, so it adds negligible noise to
        # alt position values). Converted to net-window-return scale (slope*n) to match the
        # within-symbol ret_vlong tanh scales. Falls to 0 (no effect) if BTC absent/short.
        _btc_trend = 0.0
        if "BTC" in bar_data and len(bar_data["BTC"].history) > 9:
            _btc_closes = bar_data["BTC"].history["close"].values
            _btc_n = min(VLONG_WINDOW, len(_btc_closes) - 1)
            _btc_hl2 = (bar_data["BTC"].history["high"].values[-_btc_n:] + bar_data["BTC"].history["low"].values[-_btc_n:]) / 2.0
            _btc_trend = _fast_slope(np.log(_btc_hl2)) * _btc_n

        # Exp2 (architectural, indep): PORTFOLIO CROSS-SYMBOL TREND CONSENSUS. NEW
        # portfolio-level data dep distinct from every existing cross-symbol primitive
        # (BTC 96-bar price -> alt sizing; BTC/alt volume -> alt; partner-alt price -> alt):
        # ALL THREE symbols' 20-bar ret_long SIGN-AGREEMENT. _btc_trend above is the 96-bar
        # SLOW trend; _alt_lead is the partner 20-bar slope. NEITHER measures whether BTC,
        # ETH, AND SOL 20-bar returns ALL point the same direction simultaneously -- a broad-
        # market trend where all three legs agree is a higher-conviction trend entry (the
        # move is broad-based, not idiosyncratic to one symbol). Compute the signed sum of
        # the three symbols' 20-bar net-return SIGNS (each in {-1,0,+1}); |sum|=3 = full
        # consensus. Converted to a smooth [0,1] consensus strength via tanh on (|sum|-2)/1
        # (saturates at 3-agreement, near-0 at 2-or-fewer). Computed once per bar from the
        # SAME 20-bar window already used by ret_long; falls to 0 if fewer than 2 symbols
        # present. Used as a SMALL first-bar boost (max +0.06) for entries that AGREE with
        # the consensus direction (entry sign == consensus sign): a trend-aligned entry
        # confirmed by all three legs moving together is a high-quality broad-market trend
        # entry -> larger first-bar commitment captures more of the confirmed broad move ->
        # higher Sharpe in the trend regimes (crash return-limited 0.976; all regimes benefit
        # from better broad-trend entries). The sign-of-return uses close (noise-perturbed ->
        # legitimately noise-sensitive, NOT the open-price artifact); the 20-bar window
        # averages ~20 bars of AR(1) noise (~1/sqrt(20) attenuation) so the SIGN is robust.
        # Deep-saturated (the |sum|-2 gate -> near-constant 1.0 where it fires, noise-free
        # per the validated safe-family lesson). Direction-agnostic general principle (no
        # regime label): broad-market confirmation raises entry quality. New portfolio-level
        # cross-symbol data dep at entry sizing (was per-symbol/BTC-only).
        _consensus_syms = [s for s in ACTIVE_SYMBOLS if s in bar_data and len(bar_data[s].history) > LONG_WINDOW]
        _consensus_sign = 0.0
        if len(_consensus_syms) >= 2:
            for _csym in _consensus_syms:
                _cc = bar_data[_csym].history["close"].values
                _csign = 1.0 if _cc[-1] > _cc[-LONG_WINDOW] else (-1.0 if _cc[-1] < _cc[-LONG_WINDOW] else 0.0)
                _consensus_sign += _csign
            _consensus_strength = max(0.0, min(1.0, np.tanh((abs(_consensus_sign) - 2.0) / 1.0)))
            _consensus_dir = 1.0 if _consensus_sign > 0 else (-1.0 if _consensus_sign < 0 else 0.0)
        else:
            _consensus_strength = 0.0
            _consensus_dir = 0.0

        # Exp1 (architectural, indep): PORTFOLIO CROSS-SYMBOL DEEP-BEAR down_persist.
        # Compute each symbol's _down_persist (fraction of last PERSIST_WINDOW bars where
        # 96-bar ret_vlong<0) at the top level and average across symbols present. This is
        # a PORTFOLIO-LEVEL persistent-bear signal: crash (multi-month bear across all 3
        # symbols together -> avg~0.9) vs bull (transient pullback dips -> avg~0.3) vs
        # sideways (oscillator -> avg~0.5). Used to gate a proactive portfolio-level size
        # cap (applied to `size` below) that shrinks ALL position magnitudes in persistent
        # deep-bear, the sanctioned UNTESTED lead: "portfolio-level position-size cap in
        # deep-bear, or a regime-level leverage reduction". Distinct from per-symbol
        # _down_persist (already in baseline for persist_boost gating). Computed once per
        # bar from the SAME 96-bar OLS ret_vlong already used; falls to 0 if no symbols.
        # IMPORTANT: does NOT mutate self._down_vlong_hist (the per-symbol loop below owns
        # the append); the portfolio signal includes the current bar's boolean via the
        # _p_rv < 0 check appended to the read-only history copy.
        _port_down_persist_syms = [s for s in ACTIVE_SYMBOLS if s in bar_data and len(bar_data[s].history) > VLONG_WINDOW + 1]
        _port_down_persist_vals = []
        _port_weak_persist_vals = []  # Exp6: per-symbol weak_persist for max-aggregation
        _port_rv_vals = []  # Exp8: per-symbol raw ret_vlong for max-aggregation deep-bear cap
        _port_vol_ratio_vals = []  # Exp1: per-symbol vol_ratio for max-aggregation vol-spike cap
        for _psym in _port_down_persist_syms:
            _pc = bar_data[_psym].history["close"].values
            _pn = min(VLONG_WINDOW, len(_pc) - 1)
            _phl2 = (bar_data[_psym].history["high"].values[-_pn:] + bar_data[_psym].history["low"].values[-_pn:]) / 2.0
            _p_rv = _fast_slope(np.log(_phl2)) * _pn
            _port_rv_vals.append(_p_rv)
            _pdvh = list(self._down_vlong_hist.get(_psym, []))
            _pdvh.append(1 if _p_rv < 0.0 else 0)
            if len(_pdvh) > PERSIST_WINDOW:
                _pdvh = _pdvh[-PERSIST_WINDOW:]
            _port_down_persist_vals.append((float(sum(_pdvh)) / len(_pdvh)) if _pdvh else 0.0)
            # Exp6: weak_persist for the same symbol (|ret_vlong| < PERSIST_WEAK_THRESH).
            # Read-only copy (the per-symbol loop owns the append); include current bar.
            _pwvh = list(self._weak_vlong_hist.get(_psym, []))
            _pwvh.append(1 if abs(_p_rv) < PERSIST_WEAK_THRESH else 0)
            if len(_pwvh) > PERSIST_WINDOW:
                _pwvh = _pwvh[-PERSIST_WINDOW:]
            _port_weak_persist_vals.append((float(sum(_pwvh)) / len(_pwvh)) if _pwvh else 0.0)
            # Exp1: per-symbol vol_ratio for MAX-aggregation vol-spike cap. Uses the SAME
            # VOL_LOOKBACK realized_vol / _target_vol_dyn computation as the per-symbol loop
            # (line ~795), lifted to the portfolio section so MAX across symbols can fire.
            if len(_pc) > VOL_LOOKBACK + 1:
                _p_rv_short = max(np.std(np.diff(np.log(_pc[-VOL_LOOKBACK - 1:-1]))), 1e-6)
                _p_long_n = min(200, len(_pc) - 1)
                _p_baseline_vol = max(np.std(np.diff(np.log(_pc[-_p_long_n - 1:-1]))), 1e-6)
                _p_target_vol_dyn = 0.7 * TARGET_VOL + 0.3 * _p_baseline_vol
                _port_vol_ratio_vals.append(_p_rv_short / max(_p_target_vol_dyn, 1e-6))
        _port_down_persist = (sum(_port_down_persist_vals) / len(_port_down_persist_vals)) if _port_down_persist_vals else 0.0
        # Exp1: portfolio-level deep-bear size cap (AVERAGE aggregation). Continuous tanh
        # ramp above ONSET; shrink-only (caps at 1.0; never amplifies size).
        _port_bear_cap = 1.0 - PORT_DOWN_PERSIST_MAX_SHRINK * max(0.0, min(1.0, np.tanh((_port_down_persist - PORT_DOWN_PERSIST_ONSET) / PORT_DOWN_PERSIST_SCALE)))
        # Exp5: MAX-aggregation portfolio bear cap (composes with Exp1's avg cap). Fires
        # when ANY ONE symbol has down_persist > MAX_ONSET (single-symbol persistent bear).
        # Catches mixed's down-legs where one symbol at a time is in persistent bear while
        # the average is dragged down by non-bear symbols. Byte-identical when no symbol
        # exceeds MAX_ONSET (bull/rally all symbols trend up).
        _port_down_persist_max = max(_port_down_persist_vals) if _port_down_persist_vals else 0.0
        _port_bear_cap_max = 1.0 - PORT_DOWN_PERSIST_MAX_MAX_SHRINK * max(0.0, min(1.0, np.tanh((_port_down_persist_max - PORT_DOWN_PERSIST_MAX_ONSET) / PORT_DOWN_PERSIST_MAX_SCALE)))
        _port_bear_cap = _port_bear_cap * _port_bear_cap_max
        # Exp6: MAX-aggregation portfolio weak-trend cap (composes with Exp1/Exp5). Fires
        # when ANY ONE symbol has weak_persist > MAX_ONSET (single-symbol persistent weak
        # multi-day trend). Catches mixed's consolidation phases where one symbol at a
        # time has weak |ret_vlong| while the others trend. Byte-identical when no symbol
        # exceeds MAX_ONSET (rally/bull/crash all symbols have |ret_vlong| solidly > 0.02).
        _port_weak_persist_max = max(_port_weak_persist_vals) if _port_weak_persist_vals else 0.0
        _port_weak_cap = 1.0 - PORT_WEAK_PERSIST_MAX_MAX_SHRINK * max(0.0, min(1.0, np.tanh((_port_weak_persist_max - PORT_WEAK_PERSIST_MAX_ONSET) / PORT_WEAK_PERSIST_MAX_SCALE)))
        # Exp2: portfolio-level weak-trend AVERAGE admission tightener. The AVERAGE fires
        # when ALL symbols are choppy (sideways recovery), distinct from the MAX (Exp6, any
        # single symbol = mixed's one-at-a-time consolidation). Continuous tanh ramp.
        _port_weak_persist_avg = (sum(_port_weak_persist_vals) / len(_port_weak_persist_vals)) if _port_weak_persist_vals else 0.0
        _port_weak_admit_tighten = 1.0 + PORT_WEAK_PERSIST_AVG_MAX_TIGHTEN * max(0.0, min(1.0, np.tanh((_port_weak_persist_avg - PORT_WEAK_PERSIST_AVG_ONSET) / PORT_WEAK_PERSIST_AVG_SCALE)))
        # Exp8: MAX-aggregation deep-bear MAGNITUDE cap (composes with Exp1/Exp5/Exp6). Fires
        # when ANY ONE symbol has ret_vlong < -ONSET (deep multi-day downtrend magnitude).
        # Different signal from down_persist (duration count): uses raw ret_vlong magnitude,
        # distinguishing DEEP bear (crash -0.04) from SHALLOW bear (mixed -0.01). Catches
        # crash's deep trending down-legs. Byte-identical when no symbol has ret_vlong < -ONSET.
        _port_rv_min = min(_port_rv_vals) if _port_rv_vals else 0.0  # most-negative ret_vlong
        _port_deep_bear_mag = max(0.0, -_port_rv_min)  # magnitude of the deepest bear symbol
        _port_deep_bear_cap = 1.0 - PORT_DEEP_BEAR_MAX_SHRINK * max(0.0, min(1.0, np.tanh((_port_deep_bear_mag - PORT_DEEP_BEAR_ONSET) / PORT_DEEP_BEAR_SCALE)))
        _port_weak_cap = _port_weak_cap * _port_deep_bear_cap
        # Exp1: MAX-aggregation vol-spike SIZE cap (composes with the trend/bear caps above).
        # Fires when ANY ONE symbol's vol_ratio exceeds ONSET (vol-spike regime). Distinct
        # axis from the trend/bear caps: catches vol-spike episodes where the trend is still
        # up but realized vol is elevated (bull 2021 pullbacks where ret_vlong stays
        # positive but vol_ratio spikes). Byte-identical when all vol_ratios < ONSET.
        _port_vol_ratio_max = max(_port_vol_ratio_vals) if _port_vol_ratio_vals else 0.0
        _port_vol_spike_cap = 1.0 - PORT_VOL_SPIKE_MAX_SHRINK * max(0.0, min(1.0, np.tanh((_port_vol_ratio_max - PORT_VOL_SPIKE_ONSET) / PORT_VOL_SPIKE_SCALE)))
        _port_weak_cap = _port_weak_cap * _port_vol_spike_cap
        # Exp3: MAX-aggregation deep-bear MAGNITUDE admission tightener (composes with
        # Exp2's weak_persist avg admit tightener). Same signal as Exp8 SIZE cap (raw
        # |ret_vlong| max-aggregated), applied to ADMISSION threshold. Byte-identical
        # when no symbol has |ret_vlong| > ONSET (bull/rally/sideways).
        _port_deep_bear_admit_tighten = 1.0 + PORT_DEEP_BEAR_ADMIT_MAX_TIGHTEN * max(0.0, min(1.0, np.tanh((_port_deep_bear_mag - PORT_DEEP_BEAR_ADMIT_ONSET) / PORT_DEEP_BEAR_ADMIT_SCALE)))

        # Exp1 (architectural, indep): BTC (market leader) VOLUME-participation trend. NEW
        # cross-symbol x cross-data-type data dep: prior cross-symbol deps used BTC PRICE
        # (96-bar trend) feeding alt sizing; this uses BTC VOLUME (6/18-bar mean ratio), a
        # genuinely orthogonal leader signal (leader participation, not leader direction).
        # Rising BTC volume = building broad-market participation. Used as a conjunction
        # confirmation on alt entries that AGREE with BTC's price trend: an alt trend entry
        # confirmed by BOTH leader-direction-agreement AND leader-volume-participation is a
        # higher-quality broad-market-trend entry -> larger first-bar commitment. Distinct
        # from own-symbol vol-rise (Exp5, own 6/18-bar volume) and BTC-price-trend boost
        # (Exp3 9cdb2a9a, price only): this is the cross-symbol price-agreement x cross-
        # symbol leader-volume conjunction. Computed once per bar; falls to 0 if BTC
        # absent/short. Deep-saturated (/0.30 volume ratio, /0.03 BTC trend -> near-constant,
        # noise-free per the validated safe-family lesson), first-bar-only, small (+0.05 max).
        _btc_vol_rise = 0.0
        if "BTC" in bar_data and len(bar_data["BTC"].history) > 18:
            _bv = bar_data["BTC"].history["volume"].values
            _btc_vol_recent = float(np.mean(_bv[-6:]))
            _btc_vol_long = max(float(np.mean(_bv[-18:])), 1e-10)
            _btc_vol_rise = max(0.0, min(1.0, np.tanh(((_btc_vol_recent - _btc_vol_long) / _btc_vol_long) / 0.30)))

        # Exp2 (architectural, indep): BTC (market leader) DIRECTIONAL VOLUME PRESSURE
        # (normalized OBV). NEW cross-symbol x cross-data-type dep: the validated volume
        # grid {own,BTC,partner}x{vol,price} has a vol-RISE column (total volume magnitude
        # trend) and a price column, but NO directional-volume column. _btc_dvp = sum(
        # vol[i]*sign(close[i]-close[i-1]))/sum(vol[i]) over 12 bars is the leader's
        # volume-DIRECTION balance (buy-side vs sell-side pressure), distinct from _btc_trend
        # (leader price direction) and _btc_vol_rise (leader volume magnitude). Used as a
        # conjunction confirmation on ALT trend-aligned entries: when the market leader's
        # volume is on the same side as the alt's trend entry (BTC buy-side volume confirming
        # an alt long), the broad market leader is participating directionally -> higher-
        # quality broad-market entry -> larger first-bar commitment. Distinct from own-DVP
        # (Exp1, this alt's own volume direction) and BTC-vol-rise (leader volume magnitude):
        # this is leader volume-DIRECTION x alt price-agreement. Computed once per bar; falls
        # to 0 if BTC absent/short. Deep-saturated (/0.15 DVP, /0.03 BTC trend -> near-
        # constant, noise-free per the validated safe-family lesson), first-bar-only, +0.05 max.
        _btc_dvp = 0.0
        if "BTC" in bar_data and len(bar_data["BTC"].history) > 13:
            _bdvp_c = bar_data["BTC"].history["close"].values[-13:]
            _bdvp_v = bar_data["BTC"].history["volume"].values[-12:]
            _bdvp_rets = np.sign(np.diff(_bdvp_c))
            _btc_dvp = float(np.sum(_bdvp_v * _bdvp_rets) / max(np.sum(_bdvp_v), 1e-10))

        # Exp2 (architectural, indep): cross-alt lead-lag short-term momentum. ETH and SOL
        # are correlated alts where ETH frequently LEADS SOL on intraday-to-daily moves. A
        # NEW cross-symbol data dep distinct from the BTC 96-bar trend (different leader,
        # SHORTER 20-bar timescale): the other alt's 20-bar OLS log-HL2 slope, used as a
        # small trend-confirmation boost on an alt entry when the partner alt's near-term
        # momentum agrees with the entry direction. ETH confirming SOL (and vice-versa) is
        # broad-alt-trend confirmation independent of BTC. Computed once per bar for the alt
        # pair; falls to 0 (no effect) if the partner alt is absent/short.
        _alt_lead = {}  # partner-alt 20-bar slope*n (net window return scale)
        _alt_vol_rise = {}  # partner-alt 6/18-bar volume-trend rise (deep-saturated), Exp3
        _alt_dvp = {}  # partner-alt 12-bar directional volume pressure (normalized OBV), Exp3
        _alt_pair = [s for s in ("ETH", "SOL") if s in bar_data and len(bar_data[s].history) > LONG_WINDOW + 1]
        for _asym in _alt_pair:
            _ac = bar_data[_asym].history["close"].values
            _an = min(LONG_WINDOW, len(_ac) - 1)
            _ahl2 = (bar_data[_asym].history["high"].values[-_an:] + bar_data[_asym].history["low"].values[-_an:]) / 2.0
            _alt_lead[_asym] = _fast_slope(np.log(_ahl2)) * _an
            # Exp3 (architectural, indep): partner-alt VOLUME-participation trend (6/18-bar mean
            # ratio, deep-saturated /0.30). NEW cross-symbol x cross-data-type dep: prior
            # cross-alt dep used the partner alt PRICE (20-bar momentum, _alt_lead); this uses
            # the partner alt VOLUME. Rising partner-alt volume = broad alt-market participation
            # building (both alts accumulating volume together) -> an alt trend entry confirmed
            # by partner-alt-volume-building is a higher-quality broad-alt-trend entry -> larger
            # first-bar commitment. Distinct from Exp1 (BTC leader volume) and Exp5 (own volume):
            # this is the cross-alt PARTNER volume. Computed once per bar; falls to 0 if absent.
            if len(bar_data[_asym].history) > 18:
                _pv = bar_data[_asym].history["volume"].values
                _pv_recent = float(np.mean(_pv[-6:]))
                _pv_long = max(float(np.mean(_pv[-18:])), 1e-10)
                _alt_vol_rise[_asym] = max(0.0, min(1.0, np.tanh(((_pv_recent - _pv_long) / _pv_long) / 0.30)))
            else:
                _alt_vol_rise[_asym] = 0.0
            # Exp3 (architectural, indep): partner-alt DIRECTIONAL VOLUME PRESSURE (normalized
            # OBV). NEW cross-symbol x cross-data-type dep: completes the DVP column of the
            # {own,BTC,partner}x{vol,price,DVP} volume grid (own-DVP=Exp1 keep, BTC-DVP=Exp2
            # keep; this is the partner cell). _alt_dvp[asym] = sum(vol[i]*sign(close[i]-
            # close[i-1]))/sum(vol[i]) over 12 bars on the partner alt = partner volume-
            # DIRECTION balance, distinct from _alt_lead (partner price momentum) and
            # _alt_vol_rise (partner volume magnitude). Used as a conjunction confirmation on
            # an alt trend entry: when the partner alt's volume is on the same side as the
            # entry (partner buy-side volume confirming an alt long), broad alt-market
            # participation is directional -> larger first-bar commitment. Deep-saturated
            # (/0.15 DVP, /0.02 partner-price-agreement gate -> near-constant, noise-free,
            # validated safe family), first-bar-only, +0.05 max. Computed once per bar.
            if len(bar_data[_asym].history) > 13:
                _adv_c = bar_data[_asym].history["close"].values[-13:]
                _adv_v = bar_data[_asym].history["volume"].values[-12:]
                _adv_rets = np.sign(np.diff(_adv_c))
                _alt_dvp[_asym] = float(np.sum(_adv_v * _adv_rets) / max(np.sum(_adv_v), 1e-10))
            else:
                _alt_dvp[_asym] = 0.0

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

            # Exp1 (architectural): PERSISTENCE-COUNT weak-trend separator. Track a
            # rolling boolean (|ret_vlong| < PERSIST_WEAK_THRESH) over PERSIST_WINDOW
            # bars; the FRACTION of weak bars is a DURATION measure of how long the
            # multi-day trend has been weak. Distinct from the magnitude gate
            # 1-tanh(|ret_vlong|/0.03) (and its temporal EMA): a multi-bar rally
            # pullback dips |ret_vlong| below 0.03 for a few bars but rally re-
            # strengthens between pullbacks so the 48-bar FRACTION stays low; mixed's
            # multi-day downtrend keeps |ret_vlong| persistently small -> fraction ~1.
            # The boolean uses the SAME ret_vlong so it is a duration aggregation of
            # an already-96-bar-averaged quantity (each input bar carries ~1/96 of a
            # bar's AR(1) noise -> the boolean is noise-robust; flipping it requires a
            # perturbation large enough to move ret_vlong across 0.03, which is deep
            # in the flat tail of the 96-bar slope distribution). New per-symbol state.
            _wvh = self._weak_vlong_hist.get(symbol, [])
            _wvh.append(1 if abs(ret_vlong) < PERSIST_WEAK_THRESH else 0)
            if len(_wvh) > PERSIST_WINDOW:
                _wvh = _wvh[-PERSIST_WINDOW:]
            self._weak_vlong_hist[symbol] = _wvh
            _weak_persist = (float(sum(_wvh)) / len(_wvh)) if _wvh else 0.0  # [0,1]
            # BRANCH step2 (crash sustained-boost refinement): DIRECTIONAL-DOWNTREND
            # DURATION gate. Step1's deep-downtrend threshold (-0.015) failed: bull's
            # pullback rv dips go as DEEP as crash's persistent rv, so depth cannot
            # separate them. The separator is PERSISTENCE (duration of negativity),
            # mirroring the fe6acd4d keep's validated duration-count principle applied
            # to the SIGN of ret_vlong (not |ret_vlong|). Compute _down_persist =
            # fraction of the last PERSIST_WINDOW bars where ret_vlong<0. Crash (multi-
            # month bear -> ~0.9) vs bull (transient pullback dips -> ~0.3): the 48-bar
            # FRACTION saturates for crash's persistent negativity but stays low for
            # bull's transient dips (rally re-strengthens between pullbacks -> fraction
            # stays low, same structural property that made the magnitude duration-count
            # work for the keep). New per-symbol state parallel to _weak_vlong_hist.
            _dvh = self._down_vlong_hist.get(symbol, [])
            _dvh.append(1 if ret_vlong < 0.0 else 0)
            if len(_dvh) > PERSIST_WINDOW:
                _dvh = _dvh[-PERSIST_WINDOW:]
            self._down_vlong_hist[symbol] = _dvh
            _down_persist = (float(sum(_dvh)) / len(_dvh)) if _dvh else 0.0  # [0,1]

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
            # Exp4 (architectural, indep): 8th voter -- RANGE/CLOSE efficiency-continuation.
            # Prior session CROSS-EXPERIMENT CONCLUSION: the ONLY un-disproven axis for
            # moving a regime raw is "a fundamentally new orthogonal DATA-SOURCE voter
            # added WITHOUT touching existing voter weights." This adds an 8th voter on a
            # signal no existing voter reads: the RATIO of interbar close-movement to
            # intrabar range over 12 bars (distinct from ER/Kaufman which uses |net move|/
            # sum|bar moves|; this uses |close-to-close|/|intrabar range|). High ratio =
            # closes are traveling further than the bar ranges = efficient interbar trend
            # (continuation); low ratio = range dominates closes = chop/mean-reversion.
            # Directionless efficiency SIGNED by the 12-bar close direction (sign of
            # closes[-1]-closes[-12]) so it contributes bull in an uptrend, bear in a
            # downtrend. The sign uses a 12-bar net (smooth, not a 1-bar zero-crossing).
            # Added to _voter_signals_bull with a SMALL fixed weight (0.55, below the 0.7
            # base floor of existing voters) -- appended WITHOUT modifying any of the 7
            # existing _base_weights (the trend-strength redistribution only shifts indices
            # 1-3, leaving this 8th weight untouched). New orthogonal-ish data-source voter.
            _rc_n = 12
            _rc_high = bd.history["high"].values[-_rc_n:]
            _rc_low = bd.history["low"].values[-_rc_n:]
            _rc_intrabar = float(np.mean(_rc_high - _rc_low))
            _rc_interbar = float(np.mean(np.abs(np.diff(closes[-_rc_n - 1:]))))
            _rc_eff = _rc_interbar / max(_rc_intrabar, 1e-10)  # ~1 chop, >1 trending
            _rc_dir = 1.0 if closes[-1] >= closes[-_rc_n] else -1.0
            _rc_signal = (_rc_eff - 1.0) / 0.5 * _rc_dir  # >0 trend-continuation in dir
            _voter_signals_bull = [
                (ret_short - dyn_threshold) / max(dyn_threshold * 0.20, 1e-6),
                (_ef - _es) / (mid * 0.0008),
                (rsi - _rsi_thresh) / 4.0,
                (_macd_diff - 0.0003) / 0.00012,
                (_lr_slope - 0.00015) / 0.00010,
                (_ea_slope - 0.0005) / 0.00025,
                _vwap_dev / 0.0030,  # 7th voter: VWAP deviation, halved sharpness (was 0.0015) for softer tanh, less noise in chop
                _rc_signal / 1.0,  # 8th voter: range/close efficiency-continuation (sharpness 1.0)
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
            _base_weights = (0.7, 1.25 + _wt_shift, 1.10 - _wt_shift, 1.00 - _wt_shift, 0.85, 1.10 + _wt_shift, _vwap_wt, 0.55)  # 8th: range/close efficiency voter (small fixed weight, untouched by _wt_shift)
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
                _arr = np.array(_sig_hist)  # (K, 8)
                _num = np.abs(_arr.sum(axis=0))
                _den = np.maximum(np.abs(_arr).sum(axis=0), 1e-10)
                _persistence = _num / _den  # in [0, 1]
                _persistence_mult = 0.7 + 0.6 * _persistence  # in [0.7, 1.3]
            else:
                _persistence_mult = np.ones(8)
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
            # Sideways-aware strong-sum threshold: tighten in low-trend regimes to filter
            # noisy entries; relax in trends. Uses continuous rsi_trend_str interpolation.
            _strong_min = STRONG_WEIGHT_MIN + 0.20 * (1.0 - rsi_trend_str)
            # Exp2: apply portfolio-level weak-trend AVERAGE admission tightener (computed
            # at top level). Tightens admission when ALL symbols choppy together (sideways).
            # Byte-identical when _port_weak_admit_tighten=1.0 (bull/rally/crash where cross-
            # symbol avg weak_persist < ONSET).
            _strong_min = _strong_min * _port_weak_admit_tighten
            # Exp3: apply MAX-aggregation deep-bear MAGNITUDE admission tightener (computed
            # at top level). Tightens admission when ANY symbol is in deep multi-day downtrend
            # (crash). Composes with Exp2's weak avg tighten (independent signals). Byte-
            # identical when _port_deep_bear_admit_tighten=1.0 (bull/rally/sideways).
            _strong_min = _strong_min * _port_deep_bear_admit_tighten

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
            # Exp5 (architectural, indep): COUNTER-TREND-specific loss-streak admission
            # tightening (admission counterpart to Exp3's ct size shrink). After a
            # portfolio loss streak, tighten the admission bar for COUNTER-TREND entries
            # only (bull entry while multi-day downtrend / bear entry while multi-day
            # uptrend = rally pullback shorts, the clustered losing re-entries), so the
            # WEAK ct re-entries that extend the streak get filtered while strong-conviction
            # ct entries (some are winners) still pass. Trend-aligned entries (ct indicator
            # 0) are NOT tightened -> bull/crash/rally trend longs protected (the lesson
            # from Exp1's blanket tightening which hurt bull). Distinct from Exp3 (size
            # shrink): this filters at the ADMISSION gate (cuts weak ct re-entry COUNT ->
            # directly targets the streak_gate 0.875 gap, the largest raw-vs-actual score
            # lever), Exp3 cuts magnitude. Same fast-saturating /0.01 ret_vlong ct
            # indicator (near-constant, noise-free) x streak ramp. Max 10% tighten.
            _streak_ct_admit = max(0.0, np.tanh((self._loss_streak - 1) / 2.0))
            _bull_strong_min *= 1.0 + 0.10 * _streak_ct_admit * max(0.0, np.tanh(-ret_vlong / 0.01))
            _bear_strong_min *= 1.0 + 0.10 * _streak_ct_admit * max(0.0, np.tanh(ret_vlong / 0.01))
            # Conviction margins (relative excess of strong-sum over its admission threshold).
            # Computed at top-level so they are available to both entry and flip paths.
            _bull_margin = (_bull_strong - _bull_strong_min) / max(_bull_strong_min, 1e-6)
            _bear_margin = (_bear_strong - _bear_strong_min) / max(_bear_strong_min, 1e-6)
            # Architectural subsystem redesign (Exp2, entry-admission gate): entry-readiness
            # EMA accumulator. Replaces three coupled instantaneous mechanisms — the
            # strong-sum threshold crossing, the a5c60e3a max(curr,prev) anti-dip stickiness,
            # and the min-over-2 persist co-gate — with ONE exponentially-smoothed
            # conviction-margin readiness signal. The margin is already threshold-normalized
            # (0.0 == old strong_min boundary); EMA-smoothing it integrates single-bar AR(1)
            # noise OUT of the entry DECISION (crossing of a smooth EMA is far less
            # noise-sensitive than crossing of the instantaneous margin), directly targeting
            # rally entry-TIMING divergence — the binding constraint's root cause and the one
            # axis the persist catch-22 (Exp1: loosening persist admits noise-sensitive
            # trades) could not reach. Distinct from the dead-end size temporal-smoothing
            # (combined_mult is a MINOR input): the conviction margin drives the entry timing
            # that IS the dominant rally tracking-error source. Sustained-conviction filtering
            # (the persist gate's purpose) is preserved — the EMA crosses the threshold only
            # after margin has been positive ~2 bars. New per-symbol state.
            _acc_b, _acc_s = self._entry_accum.get(symbol, (0.0, 0.0))
            _acc_b = ENTRY_ACCUM_RHO * _acc_b + (1.0 - ENTRY_ACCUM_RHO) * _bull_margin
            _acc_s = ENTRY_ACCUM_RHO * _acc_s + (1.0 - ENTRY_ACCUM_RHO) * _bear_margin
            self._entry_accum[symbol] = (_acc_b, _acc_s)
            # Exp1 (architectural, KEEP): PORTFOLIO-DD-adaptive entry readiness
            # threshold, GATED on persistent strong uptrend direction. Raise the
            # conviction-margin crossing threshold during portfolio DD so marginal
            # entries (barely crossing 0.0) are filtered during adverse correlated-
            # regime-hit periods -> fewer losing trades during DD -> higher Sharpe
            # in negative-Sharpe regimes (score==bare Sharpe there). Byte-identical
            # at portfolio peak (dd_frac=0 -> _port_dd_atten=1.0 -> raise 0).
            #
            # GATE: the raise fires only when ret_vlong > 0.04 (persistent STRONG
            # uptrend, bull's regime). This isolates bull's pullback-DD pattern
            # (bull's DD comes from pullbacks DURING a persistent uptrend, where
            # marginal entries are noise). sideways (ret_vlong oscillates around 0,
            # rarely >0.04) is byte-identical -- sideways's marginal-conviction
            # entries are profitable mean-reverters that must NOT be filtered.
            # crash (ret_vlong<0) exempt; rally/mixed at portfolio peak (dd_frac=0)
            # exempt. The gate is a one-sided linear ramp (0 below 0.04, saturating
            # to 1 at 0.08) -- continuous, no decision boundary. DISTINCT from the
            # existing _port_dd_atten SIZE shrink (scales position MAGNITUDE on
            # admitted trades; this cuts TRADE COUNT at the admission gate via the
            # sample_factor / trade-selection axis). The admission gate previously
            # had NO portfolio-DD dependency. New cross-component data dep: entry
            # readiness threshold depends on (portfolio-DD state, multi-day trend
            # direction). MEASURED vs baseline 0441725e: composite -0.0571->-0.0462
            # (+0.0108 KEEP), bull -0.0508->+0.0005 (Sh -0.051->+0.050, flipped
            # positive), all other regimes byte-identical.
            _dd_thresh_dir_gate = max(0.0, min(1.0, (ret_vlong - 0.04) / 0.04))
            _entry_thresh_dd = ENTRY_ACCUM_THRESH + PORT_DD_ENTRY_THRESH_MAX * (1.0 - _port_dd_atten) * _dd_thresh_dir_gate
            _bull_ready = _acc_b >= _entry_thresh_dd
            _bear_ready = _acc_s >= _entry_thresh_dd

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
            # Exp1: apply portfolio deep-bear size cap (computed at top level). Shrinks
            # `size` -> both first-bar entry target AND scale-in targets smaller in
            # persistent deep-bear (crash). Byte-identical when _port_bear_cap=1.0 (bull,
            # rally, sideways where cross-symbol avg down_persist < ONSET).
            # Exp6: apply max-aggregation weak-trend cap (composes with Exp1/Exp5). Fires
            # when any symbol has persistent weak multi-day trend (mixed's consolidation).
            size = equity * BASE_POSITION_SIZE * combined_mult * _port_bear_cap * _port_weak_cap

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
                # Exp5 (architectural simplification, indep): REMOVED dead
                # _range_bull_adj/_range_bear_adj = 0.0 stubs. The prior Donchian
                # range-position entry adjustment (in [-0.04, +0.04]) was removed in a
                # prior simplification but the two zero stubs and their additive uses
                # (_entry_frac_dyn + _range_bull_adj) remained. Adding 0.0 is a pure
                # no-op. Removing follows the 12b6bc63/Exp4 byte-identical simplification
                # precedent (score holds, simpler = better OOS generalization).
                # Exp2 redesign: the entry-persistence (min-over-2) gate AND the
                # max(curr,prev) anti-dip admission are both SUBSUMED by the EMA-of-margin
                # readiness gate (_bull_ready/_bear_ready, computed near the margins above).
                # The EMA crosses its threshold only after conviction has been sustained
                # ~2 bars (preserving the persist gate's spike-filtering purpose) while
                # smoothing single-bar dips/spikes out of the decision (the anti-dip role),
                # so the two former gates collapse into one continuous noise-robust signal.
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
                # Exp3 (architectural): COUNTER-TREND-specific loss-streak size shrink.
                # Distinct from Exp1's blanket escalation (which hurt bull by shrinking
                # trend-aligned post-streak entries): this shrinks ONLY counter-trend
                # entries (bull in multi-day downtrend / bear in multi-day uptrend = rally
                # pullback shorts, the clustered losing re-entries) AFTER a portfolio loss
                # streak, leaving trend-aligned entries at full size (protects bull/crash/
                # rally trend longs). Uses the SAME fast-saturating /0.01 ret_vlong ct
                # indicator as _ct_vlong (near-constant, noise-free) so the shrink is a
                # near-constant magnitude, not a noise-tracking wobble. Shrink-only (safe
                # family), max 0.25, gated on streak>=2. Does NOT cut the streak COUNT
                # (structural) but cuts in-streak ct-loser MAGNITUDE -> smaller realized
                # losses -> higher rally Sharpe + lower DD. Trend-aligned (ct indicator 0)
                # -> 1.0 byte-identical. New cross-component data dep: first-bar ct size
                # depends on portfolio loss-streak x multi-day-ct interaction.
                _streak_ct = max(0.0, np.tanh((self._loss_streak - 1) / 2.0))  # 0 streak<=1, ~1 streak>=3
                _bull_ctmd_streak = max(0.0, np.tanh(-ret_vlong / 0.01))  # bull ct in multi-day downtrend
                _bear_ctmd_streak = max(0.0, np.tanh(ret_vlong / 0.01))   # bear ct in multi-day uptrend (rally pullback shorts)
                _streak_ct_shrink_bull = 1.0 - 0.25 * _streak_ct * _bull_ctmd_streak
                _streak_ct_shrink_bear = 1.0 - 0.25 * _streak_ct * _bear_ctmd_streak
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
                # Exp4 (architectural simplification, indep): REMOVED dead _vol_entry_atten
                # stub (was =1.0 constant no-op multiplied into the target chain at lines
                # ~1761/1765). The prior _vol_entry_atten (low-volume entry size attenuator,
                # up to -30%) was removed in a prior simplification but a stub assignment
                # =1.0 and its two multiplicative uses remained. Multiplying by 1.0 is a
                # pure no-op; removing it follows the 12b6bc63 byte-identical-simplification
                # precedent (score holds, simpler = better OOS generalization).
                # Exp3 (architectural simplification): REMOVED the _tod_atten calendar-cycle
                # compound (6 stacked cos sinusoids: TOD/DOW/MOM/QUARTER/SEMI/ANNUAL). Calendar
                # sinusoids fitted to the 4 in-sample regimes' calendar positions are a prime
                # meta-overfit suspect (the AR(1) stability test cannot detect calendar-cycle
                # overfitting — a cos cycle passes stability while being perfectly overfit to
                # the known regime dates). Removing eliminates the timestamp data dependency
                # and the [0.85,1.15] first-bar size wobble it induced. Test: if score-neutral
                # or positive, the cycle stack was dead/overfit code (keep the simpler version,
                # better OOS generalization); if negative, it was load-bearing in-sample.
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
                # Exp1 (architectural, indep): churn x multi-day-counter-trend first-bar
                # SIZE shrink. The existing _ct_vlong shrink (line ~667) deliberately turns
                # OFF during entry bursts (its _calm_ct gate = 1-churn_dz) because prior
                # sessions found the continuous ret_vlong-modulated sizing hurt rally
                # STABILITY during bursts. So during a rally's choppy-pullback burst (high
                # entry density), the counter-trend SHORT re-entries (bear entry while the
                # 96-bar uptrend ret_vlong stays solidly positive) are NOT shrunk by
                # _ct_vlong -- only by the direction-UNIFORM _churn_size_atten (0.25) and
                # the scale-in _adv_freeze. Those ct re-entries are exactly rally's losing
                # trades ("streak is REAL SEPARATE losing trades = burst re-entries during
                # choppy pullbacks", results.tsv row 792). This adds a DIRECTION-AWARE
                # complement: an extra shrink (max 0.20) on counter-trend-at-multi-day
                # entries specifically when churn is HIGH (the burst partition _ct_vlong
                # leaves alone). Smaller first-bar commitment on the losing ct re-entries
                # -> smaller realized losses -> higher rally Sharpe (the binding raw
                # constraint, all stability factors 1.0 so raw IS the score).
                # Noise-robustness (the lesson that killed continuous burst-time ct sizing):
                # both gates are near-CONSTANT during a burst -- _churn_ct uses the
                # noise-IMMUNE integer len(_eh) (fast-saturating /0.6), and _ctmd uses the
                # validated FAST-saturating /0.01 ret_vlong scale (rally's solidly-positive
                # ret_vlong sits in the flat tail -> ct indicator is a near-binary 1, not a
                # noise-tracking quantity). So the shrink is a near-constant 0.20 for burst
                # ct entries, not a noise-sensitive size wobble -> stability preserved.
                # Sparing: bull/crash/sideways have low entry density (len(_eh)<=1 the whole
                # regime) -> _churn_ct ~0 -> byte-identical (same noise-immune integer-churn
                # gating as the proven grid/deadband/churn_size_atten keeps). First-bar only
                # (respects the proven winning axis: first-bar-only size changes help rally,
                # sustained-through-pullback sizing hurts). Shrink-only (safe family). New
                # cross-component data dep: first-bar size depends on churn x multi-day-ct
                # interaction (ct_vlong's calm-partition complement, operating on the burst
                # partition ct_vlong deliberately leaves alone).
                _churn_ct = max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))  # ~0 calm, ~1 bursting (noise-immune integer gate)
                _bull_ctmd = max(0.0, np.tanh(-ret_vlong / 0.01))  # bull long counter to multi-day downtrend
                _bear_ctmd = max(0.0, np.tanh(ret_vlong / 0.01))   # bear short counter to multi-day uptrend (rally pullback shorts)
                _churn_ct_atten_bull = 1.0 - 0.20 * _churn_ct * _bull_ctmd
                _churn_ct_atten_bear = 1.0 - 0.20 * _churn_ct * _bear_ctmd
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
                # Architectural: anti-noise-dip admission stickiness (avg5 RE-TEST).
                # Re-tests commit 45942a93 (results.tsv row 689) which was RAW BYTE-IDENTICAL
                # on all 4 regimes (zero clean-trade delta: prev-bar crossings are already
                # entered or co-gated by persist/admit) but was DISCARDED only because the
                # OLD per-strategy AST-hash stability seed "re-seeded" rally 0.8008->0.7063
                # vs the baseline's LUCKY single 0.80 draw. The avg5 re-baseline (12d6c5f9)
                # replaced that single-seed lottery with 5 FIXED shared seeds, so a raw-inert
                # edit no longer re-rolls a different seed — baseline and candidate are scored
                # on the SAME 100 noise realizations, making the old re-seed objection
                # obsolete. Mechanism: the primary entry gate fires on max(current, previous-
                # bar) strong-sum so a single-bar AR(1) dip below the admission floor does NOT
                # cancel an otherwise-sustained entry (the min-over-2 persist gate still co-
                # requires sustained conviction -> clean trades unchanged). Hypothesis: anti-
                # dip admission reduces the entry-TIMING divergence that drives rally tracking
                # error (the binding constraint), raising rally stability at byte-identical raw.
                # Exp2 subsystem redesign: readiness gate (EMA-of-margin) replaces the
                # max(curr,prev) anti-dip admission + min-over-2 persist co-gate. Trend
                # admit gate (_bull_admit/_bear_admit) retained (orthogonal, not a rally
                # noise source per 8b7df8fa). Both ready + admit required to open.
                # Architectural (Exp3): cross-asset BTC-trend confirmation SHRINK (ETH/SOL
                # only; BTC self-referential -> 1.0 byte-identical). Smooth tanh on BTC's
                # multi-day trend, shrink-only (caps at 1.0): an alt entry whose direction
                # opposes BTC's multi-day trend gets up to 0.25x first-bar size. Falls out
                # naturally per regime: bull/crash alts trade WITH BTC (agree -> ~no shrink),
                # rally alt counter-trend SHORTS oppose BTC's uptrend (-> shrunk), sideways
                # BTC~flat (-> ~no shrink). Direction-agnostic GENERAL principle (no regime
                # label). Shrinking low-quality counter-market entries is Sharpe-neutral-to-
                # positive (proven size-shrink axis) + cuts alt idiosyncratic tracking error.
                if symbol == "BTC":
                    # Exp3 (architectural, indep): BTC self-trend bilateral boost (mirrors
                    # the alt xasset boost to BTC itself). BTC entries currently get
                    # _xasset=1.0 (no trend boost) while alts get a +0.12 boost when
                    # strongly aligned with BTC's multi-day trend. BTC IS the market
                    # leader; a BTC entry aligned with its OWN strong multi-day trend
                    # (BTC long in rally uptrend, BTC short in crash downtrend) is a
                    # high-quality trend entry -> more upfront commitment captures more of
                    # the trend move -> higher Sharpe in the two binding regimes (rally
                    # 0.673 Sh1.24, crash 0.812 Sh1.26 return-limited). For BTC, _btc_trend
                    # == its own 96-bar ret_vlong, so this is a self-referential trend-
                    # aligned first-bar BOOST (the ct_vlong SHRINKS ct entries; this BOOSTS
                    # trend-aligned ones, the symmetric opposite). Same +0.12 max and /0.03
                    # strong-agreement gate as the validated alt xasset boost (9cdb2a9a
                    # keep) -> weak-trend bull-2021 pullback stretches spared, sideways
                    # (BTC~flat) -> ~no boost. Shrink-side unchanged (_xasset >= 1.0 for
                    # BTC; BTC has no cross-asset disagreement to shrink). New self-
                    # referential data dep at BTC entry sizing (was constant 1.0).
                    _btc_self_boost = 0.12 * max(0.0, np.tanh(abs(_btc_trend) / 0.03))
                    _xasset_bull = 1.0 + _btc_self_boost * max(0.0, np.tanh(_btc_trend / 0.03))
                    _xasset_bear = 1.0 + _btc_self_boost * max(0.0, np.tanh(-_btc_trend / 0.03))
                    # Exp5 (architectural, indep): alt-pair VOLUME-participation confirmation
                    # boost on BTC entries. NEW cross-symbol x cross-data-type data dep: BTC
                    # entry sizing previously read NO alt data (only own self-trend boost above);
                    # this reads the alt pair (ETH+SOL) average 6/18-bar VOLUME rise. When the
                    # followers (alts) are participating (volume building) in the SAME direction
                    # as a BTC trend-aligned entry, the broad market is confirming the leader's
                    # move -> higher-quality broad-market-trend entry -> larger first-bar
                    # commitment. Distinct from Exp1 (BTC volume -> alt) and Exp3 (partner
                    # volume -> alt): this is alt-pair volume -> BTC (mirror direction). Volume
                    # (participation) is the differentiated signal (Exp1/Exp3 proved cross-
                    # symbol VOLUME is non-redundant, unlike cross-symbol price-agreement which
                    # is saturated). Deep-saturated (/0.30 alt-pair vol, /0.03 BTC trend ->
                    # near-constant, noise-free, validated safe family). First-bar-only, +0.05
                    # max. Targets rally (BTC longs confirmed by alts participating) + crash
                    # (BTC shorts confirmed by alts participating); bull weak-trend + sideways
                    # spared by /0.03 BTC-trend gate.
                    _alt_pair_vol_rise = 0.5 * (_alt_vol_rise.get("ETH", 0.0) + _alt_vol_rise.get("SOL", 0.0))
                    _xasset_bull *= 1.0 + 0.05 * _alt_pair_vol_rise * max(0.0, np.tanh(_btc_trend / 0.03))
                    _xasset_bear *= 1.0 + 0.05 * _alt_pair_vol_rise * max(0.0, np.tanh(-_btc_trend / 0.03))
                else:
                    _xasset_bull = 1.0 - 0.25 * max(0.0, np.tanh(-_btc_trend / 0.06))  # BTC downtrend shrinks alt long
                    _xasset_bear = 1.0 - 0.25 * max(0.0, np.tanh(_btc_trend / 0.06))    # BTC uptrend shrinks alt short (rally)
                    # Exp3 (architectural, indep): bilateral expansion of the xasset gate.
                    # The existing term is SHRINK-only (caps at 1.0): alt entries disagreeing
                    # with BTC's multi-day trend get smaller. Add the symmetric BOOST: alt
                    # entries that strongly AGREE with BTC's multi-day trend get larger first-
                    # bar size. Mechanism: when the market leader (BTC) is in a strong
                    # multi-day trend and an alt entry trades the SAME direction, that is a
                    # high-quality correlated-trend entry (rally: BTC/ETH/SOL grind up together
                    # -> alt longs agree; crash: alts follow BTC down -> alt shorts agree) ->
                    # more upfront commitment captures more of the correlated trend move ->
                    # higher Sharpe in the trend regimes (rally 1.215 binding; crash 1.274
                    # with 0.649% MaxDD = large DD headroom). Strong-agreement gate (tanh
                    # saturation /0.03 so only DEEP BTC trend fires, ~off in choppy/weak-trend
                    # bull-2021 pullback stretches) + small max boost 0.12 (size changes are
                    # delicate; bull Sharpe prize is size-sensitive). BTC self-referential ->
                    # 1.0 (byte-identical). New cross-symbol bilateral data dep (was shrink-
                    # only). Continuous tanh, no boundary.
                    _xasset_boost = 0.12 * max(0.0, np.tanh(abs(_btc_trend) / 0.03))
                    _xasset_bull *= 1.0 + _xasset_boost * max(0.0, np.tanh(_btc_trend / 0.03))      # boost alt long when BTC uptrend
                    _xasset_bear *= 1.0 + _xasset_boost * max(0.0, np.tanh(-_btc_trend / 0.03))     # boost alt short when BTC downtrend
                    # Exp4 (architectural, combination): alt OWN-multi-day-trend boost gated
                    # on BTC agreement. Combines the Exp3 BTC self-trend boost (own ret_vlong
                    # as a trend-aligned entry boost) with the xasset BTC-agreement boost above.
                    # The xasset boost fires when the alt agrees with BTC trend; this adds a
                    # SMALL complementary boost when the alt's OWN ret_vlong ALSO strongly agrees
                    # with the entry direction (conjunction: own AND btc both confirm). Only
                    # fires in BROAD-MARKET trends (rally: alt up + BTC up; crash: alt down +
                    # BTC down) -> ~0 in idiosyncratic alt moves (own trend up but BTC flat/down)
                    # and sideways (both ~flat). Targets the binding regimes (rally 0.697, crash
                    # 0.812) via the alt legs (BTC already covered by Exp3). Small +0.06 max
                    # (bounds the combined alt size-up: up to +0.12 xasset + +0.06 own = +0.18 in
                    # rally; crash/rally DD has large headroom at 0.65%/1.54%). Same /0.03 strong-
                    # agreement gate as the validated boosts (bull-2021 weak-trend stretches
                    # spared). New cross-component data dep: alt entry size depends on own
                    # ret_vlong x BTC-trend conjunction (was BTC-trend only).
                    _alt_own_bull = max(0.0, np.tanh(ret_vlong / 0.03))       # alt own multi-day uptrend
                    _alt_own_bear = max(0.0, np.tanh(-ret_vlong / 0.03))      # alt own multi-day downtrend
                    _alt_btc_agree_bull = max(0.0, np.tanh(_btc_trend / 0.03))   # BTC confirms uptrend
                    _alt_btc_agree_bear = max(0.0, np.tanh(-_btc_trend / 0.03))  # BTC confirms downtrend
                    _xasset_bull *= 1.0 + 0.06 * _alt_own_bull * _alt_btc_agree_bull
                    _xasset_bear *= 1.0 + 0.06 * _alt_own_bear * _alt_btc_agree_bear
                    # Exp2: cross-alt lead-lag confirmation boost (partner alt = the OTHER
                    # of ETH/SOL). ETH often leads SOL; a partner alt whose 20-bar momentum
                    # agrees with this entry direction is broad-alt-trend confirmation at a
                    # shorter timescale than BTC's 96-bar trend. Small +0.05 max, strong-
                    # agreement gate (/0.02 so only DEEP partner momentum fires -> off in
                    # idiosyncratic/single-alt moves + sideways). BTC self-referential ->
                    # not reached (this is the alt branch). New cross-symbol pair data dep.
                    _partner = "SOL" if symbol == "ETH" else "ETH"
                    _partner_lead = _alt_lead.get(_partner, 0.0)
                    _xasset_bull *= 1.0 + 0.05 * max(0.0, np.tanh(_partner_lead / 0.02))
                    _xasset_bear *= 1.0 + 0.05 * max(0.0, np.tanh(-_partner_lead / 0.02))
                    # Exp3 (architectural, indep): symmetric partner-DISAGREEMENT entry shrink.
                    # The Exp2 keep validated the partner-alt lead-lag as a confirmation BOOST
                    # (agreement -> bigger). This adds the symmetric SHRINK counterpart: an alt
                    # entry whose direction OPPOSES the partner alt's 20-bar momentum is an
                    # idiosyncratic/counter-alt-trend move (one alt diverging while the other
                    # trends) -> lower quality -> smaller first-bar commitment. Shrink-only
                    # (caps at 1.0, safe family). Same /0.02 deep-disagreement gate so only
                    # STRONG partner opposition fires (mild divergence spared). Distinct from
                    # the BTC-trend shrink (different leader, shorter timescale). New cross-
                    # symbol-pair shrink data dep (Exp2 was boost-only).
                    _xasset_bull *= 1.0 - 0.05 * max(0.0, np.tanh(-_partner_lead / 0.02))
                    _xasset_bear *= 1.0 - 0.05 * max(0.0, np.tanh(_partner_lead / 0.02))
                    # Exp3 (architectural, indep): partner-alt VOLUME-rise x partner-alt-price-
                    # momentum-agreement conjunction boost. _partner_vol_rise (deep-saturated
                    # partner 6/18-bar volume ratio) confirms the partner alt's participation is
                    # BUILDING; the /0.02 partner-price-agreement gate (same as the validated
                    # Exp2 partner lead-lag boost) confirms this alt trades WITH the partner's
                    # near-term direction. The CONJUNCTION (both ~1) fires only when both alts
                    # are participating in the same direction -> broad alt-market trend entry ->
                    # larger first-bar commitment. Small +0.05 max, deep-saturated both gates
                    # (near-constant -> noise-free, validated safe family). First-bar-only.
                    # Distinct from Exp1 (BTC leader volume) and Exp5 (own volume): cross-alt
                    # PARTNER volume x partner-price conjunction.
                    _partner_vol_rise = _alt_vol_rise.get(_partner, 0.0)
                    _xasset_bull *= 1.0 + 0.05 * _partner_vol_rise * max(0.0, np.tanh(_partner_lead / 0.02))
                    _xasset_bear *= 1.0 + 0.05 * _partner_vol_rise * max(0.0, np.tanh(-_partner_lead / 0.02))
                    # Exp1 (architectural, indep): BTC leader-volume-participation x BTC-price-
                    # trend-agreement conjunction boost on alt entries. _btc_vol_rise (deep-
                    # saturated BTC 6/18-bar volume ratio) confirms leader participation is
                    # BUILDING; the /0.03 BTC-trend agreement gate (same as the validated Exp3
                    # xasset boost) confirms the alt trades WITH the leader's multi-day direction.
                    # The CONJUNCTION (both ~1) fires only on broad-market trend entries where
                    # the leader is participating in the same direction -> larger first-bar
                    # commitment captures more of the confirmed broad trend. Small +0.05 max,
                    # deep-saturated both gates (near-constant -> noise-free, the validated safe
                    # family). First-bar-only. BTC self-referential -> not reached (alt branch).
                    # Distinct from own-vol-rise (Exp5: own symbol volume) and BTC-price boost
                    # (Exp3: price only) — this is cross-symbol price-agreement x cross-symbol
                    # leader-volume conjunction.
                    _btc_agree_bull = max(0.0, np.tanh(_btc_trend / 0.03))   # BTC confirms uptrend
                    _btc_agree_bear = max(0.0, np.tanh(-_btc_trend / 0.03))  # BTC confirms downtrend
                    _xasset_bull *= 1.0 + 0.05 * _btc_vol_rise * _btc_agree_bull
                    _xasset_bear *= 1.0 + 0.05 * _btc_vol_rise * _btc_agree_bear
                # Architectural (this session): portfolio same-direction GROSS-EXPOSURE
                # governor. NEW cross-symbol data dependency the strategy entirely lacks:
                # first-bar entry size reads the AGGREGATE already-open same-sign notional
                # across the OTHER active symbols, as a fraction of equity. In correlated
                # regimes (a rally: BTC/ETH/SOL all grind up together) the per-symbol logic
                # independently builds 3 same-direction longs whose COMBINED drawdown is the
                # portfolio-level risk no within-symbol primitive can see — and rally's DD
                # (1.83%, dd_gate~0.35) is the binding low-score driver. When concurrent
                # same-direction exposure is already high, shrink the marginal new entry
                # (shrink-only, floor 1-CONC_EXP_MAX_SHRINK); when it's the first leg
                # (~0 concurrent), no effect. Distinct from Exp2 (equity-vol, temporally
                # DISJOINT from entries -> inert): concurrent position notional is high
                # EXACTLY when correlated entries fire, so this is live at the decision.
                # Direction-aware, shrink-only (respects Exp1 exposure-optimum lesson),
                # smooth tanh (no boundary). Falls out per regime: rally/correlated-bull
                # pile-ups shrink; uncorrelated/single-leg entries unaffected.
                _long_notional = 0.0
                _short_notional = 0.0
                for _osym, _opos in portfolio.positions.items():
                    if _osym != symbol:
                        if _opos > 0:
                            _long_notional += _opos
                        elif _opos < 0:
                            _short_notional += -_opos
                _conc_frac_bull = _long_notional / max(equity, 1e-10)
                _conc_frac_bear = _short_notional / max(equity, 1e-10)
                _conc_shrink_bull = 1.0 - CONC_EXP_MAX_SHRINK * max(0.0, np.tanh((_conc_frac_bull - CONC_EXP_FLOOR) / CONC_EXP_SCALE))
                _conc_shrink_bear = 1.0 - CONC_EXP_MAX_SHRINK * max(0.0, np.tanh((_conc_frac_bear - CONC_EXP_FLOOR) / CONC_EXP_SCALE))
                # Exp1 (architectural, indep): PORTFOLIO NET-DIRECTIONAL-TILT shrink.
                # Reuses the cross-symbol notional aggregated above (the per-side
                # _long_notional/_short_notional loop). Net tilt = (long-short)/equity
                # is the SIGNED book directional exposure — orthogonal to the per-side
                # gross concentration (_conc_shrink). A new BULL entry adds to net-long
                # tilt: shrink by how net-long the book already is. A new BEAR entry
                # REDUCES net-long tilt (hedges): shrink by how net-SHORT the book
                # already is. So only the tilt-ADDING side is shrunk; the hedge side is
                # unshrunk (floor 1.0). Symmetric, shrink-only, continuous tanh. Falls
                # out per regime: single-leg/spike regimes rarely build net tilt -> ~1.0;
                # correlated one-way pile-ups (already rare via _conc_shrink) get an
                # extra directional-risk trim. Composed multiplicatively with _conc_shrink
                # (independent signals: gross-concentration vs net-direction). Sparing
                # magnitude (max 0.20) since it rides on top of _conc_shrink.
                _net_tilt = (_long_notional - _short_notional) / max(equity, 1e-10)
                _net_tilt_shrink_bull = 1.0 - NET_TILT_MAX_SHRINK * max(0.0, np.tanh((_net_tilt - NET_TILT_FLOOR) / NET_TILT_SCALE))
                _net_tilt_shrink_bear = 1.0 - NET_TILT_MAX_SHRINK * max(0.0, np.tanh((-_net_tilt - NET_TILT_FLOOR) / NET_TILT_SCALE))
                # Exp8 (architectural, indep): volume-spike ENTRY shrink. The Exp4 keep
                # validated volume as an exit-side exhaustion signal (bull +0.021). Mirror
                # it to the ENTRY side: a fresh entry taken DURING a volume spike (z>2) is
                # likely chasing a capitulation/exhaustion move (rally FOMO longs at tops,
                # crash capitulation shorts at bottoms) -> higher immediate giveback risk.
                # Shrink first-bar size (shrink-only, no admission boundary -> no bull
                # collapse). New entry-side data dep on volume z-score (distinct from the
                # exit harvest - this prevents over-committing to the spike that the exit
                # harvest would then have to trim). Continuous tanh, fires only at extreme.
                _vol_arr_en = bd.history["volume"].values[-21:-1]
                _vol_mean_en = float(np.mean(_vol_arr_en))
                _vol_std_en = max(float(np.std(_vol_arr_en)), 1e-10)
                _vol_z_en = (float(bd.history["volume"].values[-1]) - _vol_mean_en) / _vol_std_en
                _vol_entry_spike = 1.0 - 0.25 * max(0.0, min(1.0, np.tanh((_vol_z_en - 2.0) / 1.5)))  # max 25% shrink at deep spike
                # Exp3 (architectural, indep): volume-DECLINE entry shrink (volume-price
                # divergence). Complementary to _vol_entry_spike (shrinks HIGH-volume spike
                # entries = capitulation/exhaustion chases): this shrinks LOW-volume entries
                # where the move lacks participation. A price move on DECLINING volume is a
                # low-participation move (rally late-stage weak longs on fading volume, crash
                # dead-cat bounces on thin volume) -> lower quality -> smaller first-bar
                # commitment. NEW data dep: volume TREND (6-bar vs 18-bar mean ratio), signed
                # by decline — distinct from vol_z (level spike) and VWAP voter (level deviation).
                # Trend-strength-gated (_vd_trend_w via |ret_long|/0.04) so it fires only in
                # trending moves where participation confirmation matters; sideways (low
                # |ret_long| AND structurally low volume) is spared by the gate. Shrink-only
                # (caps at 1.0, safe family), max 0.15. Continuous tanh, no boundary. Targets
                # rally/crash raw via smaller low-participation entry losses.
                _vol_recent_m = float(np.mean(bd.history["volume"].values[-6:]))
                _vol_long_m = max(float(np.mean(bd.history["volume"].values[-18:])), 1e-10)
                _vol_trend_r = (_vol_recent_m - _vol_long_m) / _vol_long_m  # + rising, - declining
                _vd_trend_w = max(0.0, np.tanh(abs(ret_long) / 0.04))  # 0 chop, ~1 trend
                _vol_decline_shrink = 1.0 - 0.15 * _vd_trend_w * max(0.0, min(1.0, np.tanh(-_vol_trend_r / 0.30)))
                # Exp6 (architectural, indep): counter-trend volume-decline ADDITIONAL shrink,
                # gated on the MULTI-DAY (ret_vlong) trend. Exp3's symmetric decline-shrink is
                # gated by the 20-bar _vd_trend_w, which goes to ~0 during a rally pullback
                # (ret_long small/negative) — exactly when the losing counter-trend shorts
                # fire. ret_vlong (96-bar) stays strongly positive through rally pullbacks
                # (uptrend intact), so gating an ADDITIONAL decline-shrink on ret_vlong
                # strength keeps it active during pullbacks. Direction-AWARE: only shrink the
                # COUNTER-TREND side (bear entry when multi-day uptrend, bull entry when multi-
                # day downtrend) on declining volume — the losing ct entries. Trend-aligned
                # entries on declining volume are NOT shrunk by this term (a pullback long is a
                # winning trade; only the symmetric Exp3 term touches it). Targets rally's
                # residual ct-short drag without touching the trend-aligned longs Exp5 boosted.
                # Fast-saturating /0.01 ret_vlong scale (rally's solidly-positive ret_vlong
                # sits in the flat tail -> near-constant, noise-free per branch-step-9 lesson).
                # Shrink-only (safe family), max 0.12, first-bar-only. New cross-component data
                # dep: volume-decline x multi-day-trend-direction x entry-direction conjunction.
                _vd_vl_w = max(0.0, np.tanh(abs(ret_vlong) / 0.01))  # ~0 flat multi-day, ~1 strong (noise-free)
                _vd_decline = max(0.0, min(1.0, np.tanh(-_vol_trend_r / 0.30)))
                _vd_ct_shrink_bull = 1.0 - 0.12 * _vd_vl_w * _vd_decline * max(0.0, np.tanh(-ret_vlong / 0.01))  # bull ct in multi-day downtrend
                _vd_ct_shrink_bear = 1.0 - 0.12 * _vd_vl_w * _vd_decline * max(0.0, np.tanh(ret_vlong / 0.01))   # bear ct in multi-day uptrend (rally)
                # Exp2 (architectural, indep): TREND-ALIGNED COUNTER-MOVE-VELOCITY entry shrink.
                # See COUNTER_VEL_SHRINK_MAX header for full rationale. Compute the 3-bar
                # return velocity and shrink trend-aligned entries taken after a sharp counter-
                # move (short after a 3-bar UP bounce in a downtrend; long after a 3-bar DOWN
                # pullback in an uptrend). The 3-bar window is short enough that bounces/pullbacks
                # register distinctly (unlike ret_short's 8-16 bar median which averages them out).
                # Gated on multi-day trend-alignment (entry dir matches ret_vlong sign). Continuous
                # tanh on the counter-move magnitude; shrink-only (max 0.18); first-bar-only.
                _cv_n = 4  # 3-bar return = closes[-1] vs closes[-4]
                if len(closes) > _cv_n:
                    _cv_ret3 = (closes[-1] - closes[-_cv_n]) / closes[-_cv_n]
                else:
                    _cv_ret3 = 0.0
                _cv_ta_bull = max(0.0, np.tanh(ret_vlong / 0.01))      # bull aligned with multi-day uptrend
                _cv_ta_bear = max(0.0, np.tanh(-ret_vlong / 0.01))   # bear aligned with multi-day downtrend
                # counter-move: short after UP bounce (ret3>0), long after DOWN pullback (ret3<0)
                _cv_counter_bull = max(0.0, -_cv_ret3)  # long entered after a down-move
                _cv_counter_bear = max(0.0, _cv_ret3)   # short entered after an up-move
                _cv_shrink_bull = 1.0 - COUNTER_VEL_SHRINK_MAX * _cv_ta_bull * max(0.0, min(1.0, np.tanh(_cv_counter_bull / COUNTER_VEL_SCALE)))
                _cv_shrink_bear = 1.0 - COUNTER_VEL_SHRINK_MAX * _cv_ta_bear * max(0.0, min(1.0, np.tanh(_cv_counter_bear / COUNTER_VEL_SCALE)))
                # Exp5 (architectural, indep): volume-RISING trend-ALIGNED entry boost —
                # bilateral counterpart to the Exp3 decline shrink. A trend-aligned entry on
                # RISING volume has strong participation confirming the trend (rally longs on
                # building volume, crash shorts on capitulation volume) -> high quality ->
                # larger first-bar commitment captures more of the confirmed trend move ->
                # higher Sharpe in the trend regimes. Distinct from vol_entry_spike (HIGH-
                # LEVEL spike = exhaustion shrink), VWAP voter (level deviation), and Exp3
                # (decline = low-participation shrink): this is the rising-SLOPE confirmation
                # BOOST. Safety: trend-ALIGNMENT gated (entry dir matches ret_long sign) so it
                # only boosts genuine trend entries, NOT counter-trend shorts (avoids over-
                # committing losers, the Exp4 lesson). Strong gate (/0.30 deep rising volume),
                # small max (+0.08), trend-strength required. First-bar-only (sustaining
                # shrinks hurt per Exp4; a boost sustained would over-commit worse). New data
                # dep: volume-trend slope x trend-alignment conjunction at entry sizing.
                _vol_rise = max(0.0, min(1.0, np.tanh(_vol_trend_r / 0.30)))  # 0 flat/decline, 1 deep rising
                _vol_rise_align_bull = _vol_rise * max(0.0, np.tanh(ret_long / 0.04))      # bull aligned with uptrend
                _vol_rise_align_bear = _vol_rise * max(0.0, np.tanh(-ret_long / 0.04))     # bear aligned with downtrend
                _vol_rise_boost_bull = 1.0 + 0.08 * _vol_rise_align_bull
                _vol_rise_boost_bear = 1.0 + 0.08 * _vol_rise_align_bear
                # Exp6 (architectural, indep): OWN-volume-rise x PARTNER-alt-price-agreement
                # conjunction boost on ALT entries. Completes the {own,BTC,partner}x{vol,price}
                # agreement grid: Exp1 = BTC-vol x BTC-price -> alt; Exp3 = partner-vol x
                # partner-price -> alt; Exp5 = alt-pair-vol x BTC-price -> BTC; existing vol-rise
                # (Exp5-prior) = own-vol x OWN-price -> all. THIS is own-vol x PARTNER-price ->
                # alt: an alt trend entry confirmed by BOTH own volume building AND the partner
                # alt's 20-bar momentum agreeing is a high-quality broad-alt-trend entry where
                # THIS alt is itself participating (own volume) while the partner confirms ->
                # larger first-bar commitment. Distinct from Exp3 (PARTNER volume, not own) and
                # from existing vol-rise (OWN trend, not partner). Deep-saturated both gates
                # (/0.30 own vol, /0.02 partner price -> near-constant, noise-free, validated
                # safe family). First-bar-only, +0.05 max. BTC self-referential has no partner
                # -> 1.0 byte-identical (guarded by _partner existence).
                if symbol != "BTC":
                    _vol_partner_boost_bull = 1.0 + 0.05 * _vol_rise * max(0.0, np.tanh(_partner_lead / 0.02))
                    _vol_partner_boost_bear = 1.0 + 0.05 * _vol_rise * max(0.0, np.tanh(-_partner_lead / 0.02))
                    # Exp7 (architectural, indep): OWN-volume-rise x BTC-price-trend-agreement
                    # conjunction boost on ALT entries. Last clean cell of the {own,BTC,partner}
                    # x{vol,price} agreement grid: own-vol x BTC-price -> alt. An alt trend entry
                    # where the alt itself is participating (own volume rising) AND the leader
                    # (BTC) confirms the direction is a high-quality broad-market entry (own
                    # participation + leader confirmation) -> larger first-bar commitment.
                    # Distinct from Exp1 (BTC VOLUME, not own) and Exp6 (PARTNER price, not BTC).
                    # Deep-saturated both gates (/0.30 own vol, /0.03 BTC trend -> near-constant,
                    # noise-free, validated safe family). First-bar-only, +0.05 max.
                    _vol_btc_boost_bull = 1.0 + 0.05 * _vol_rise * max(0.0, np.tanh(_btc_trend / 0.03))
                    _vol_btc_boost_bear = 1.0 + 0.05 * _vol_rise * max(0.0, np.tanh(-_btc_trend / 0.03))
                    # Exp (architectural simplification -> BRANCH step1): the two MIXED-cell
                    # volume-boost grid terms (Exp8 BTC-vol x PARTNER-price, Exp9 PARTNER-vol x
                    # BTC-price) were REMOVED in the branch base. MEASURED: removal gained rally
                    # +0.015 (terms over-committed rally trend entries) BUT lost mixed -0.0076
                    # (terms confirmed mixed's bounce longs). The terms are NOT redundant (mixed
                    # regressed) but over-commit STRONG-trend regimes (rally). Step1 RE-ADDS both
                    # terms GATED ON WEAK MULTI-DAY TREND: fire only when ret_vlong is weak/choppy
                    # (mixed's local bounces in a down/chop year) NOT when ret_vlong is strongly
                    # positive (rally's grinding uptrend = the over-commitment source). Gate =
                    # (1 - tanh(|ret_vlong|/0.03)): ~1 weak trend (mixed bounces), ~0 strong trend
                    # (rally). Keeps mixed's bounce-long confirmation while removing rally's over-
                    # commitment. Fast-saturating /0.03 (near-constant, noise-free). Same +0.05
                    # max, deep-saturated vol gates. BTC self-referential -> not reached (alt branch).
                    _weak_vlong = 1.0 - max(0.0, np.tanh(abs(ret_vlong) / 0.03))  # ~1 weak trend, ~0 strong trend
                    _btcvol_partner_boost_bull = 1.0 + 0.07 * _weak_vlong * _btc_vol_rise * max(0.0, np.tanh(_partner_lead / 0.02))
                    _btcvol_partner_boost_bear = 1.0 + 0.07 * _weak_vlong * _btc_vol_rise * max(0.0, np.tanh(-_partner_lead / 0.02))
                    _partnervol_btc_boost_bull = 1.0 + 0.07 * _weak_vlong * _partner_vol_rise * max(0.0, np.tanh(_btc_trend / 0.03))
                    _partnervol_btc_boost_bear = 1.0 + 0.07 * _weak_vlong * _partner_vol_rise * max(0.0, np.tanh(-_btc_trend / 0.03))
                    # Exp2 (architectural, indep): BTC leader DVP x BTC-price-trend-agreement
                    # conjunction boost on ALT entries (the directional-volume column of the
                    # {own,BTC,partner}x{vol,price} grid). _btc_dvp (leader volume-DIRECTION
                    # balance) x /0.03 BTC-trend agreement gate (same as the validated Exp1
                    # BTC-vol-rise boost). When the leader's volume is on the same side as a
                    # BTC-confirmed alt trend entry, broad-market leader participation is
                    # directional -> larger first-bar commitment. Deep-saturated both gates
                    # (near-constant, noise-free, validated safe family). First-bar-only,
                    # +0.05 max. BTC self-referential -> not reached (alt branch).
                    _btcdvp_boost_bull = 1.0 + 0.05 * max(0.0, np.tanh(_btc_dvp / 0.15)) * max(0.0, np.tanh(_btc_trend / 0.03))
                    _btcdvp_boost_bear = 1.0 + 0.05 * max(0.0, np.tanh(-_btc_dvp / 0.15)) * max(0.0, np.tanh(-_btc_trend / 0.03))
                    # Exp3 (architectural, indep): partner-alt DVP x partner-alt-price-momentum-
                    # agreement conjunction boost (partner cell of the DVP column). _partner_dvp
                    # (partner volume-DIRECTION balance) x /0.02 partner-price-agreement gate
                    # (same as the validated Exp2 partner lead-lag boost). When the partner alt's
                    # volume is on the same side as a partner-confirmed alt trend entry, broad
                    # alt-market participation is directional -> larger first-bar commitment.
                    # Deep-saturated both gates (near-constant, noise-free, validated safe
                    # family). First-bar-only, +0.05 max.
                    _partner_dvp = _alt_dvp.get(_partner, 0.0)
                    _partnerdvp_boost_bull = 1.0 + 0.05 * max(0.0, np.tanh(_partner_dvp / 0.15)) * max(0.0, np.tanh(_partner_lead / 0.02))
                    _partnerdvp_boost_bear = 1.0 + 0.05 * max(0.0, np.tanh(-_partner_dvp / 0.15)) * max(0.0, np.tanh(-_partner_lead / 0.02))
                else:
                    _vol_partner_boost_bull = 1.0
                    _vol_partner_boost_bear = 1.0
                    _vol_btc_boost_bull = 1.0
                    _vol_btc_boost_bear = 1.0
                    _btcvol_partner_boost_bull = 1.0
                    _btcvol_partner_boost_bear = 1.0
                    _partnervol_btc_boost_bull = 1.0
                    _partnervol_btc_boost_bear = 1.0
                    _btcdvp_boost_bull = 1.0
                    _btcdvp_boost_bear = 1.0
                    _partnerdvp_boost_bull = 1.0
                    _partnerdvp_boost_bear = 1.0
                # Exp (architectural, indep): close-POSITION-WITHIN-BAR conviction
                # entry boost. NEW data dependency: where the close sits in the bar's
                # own high-low range, close_loc = (close-low)/(high-low) in [0,1]. NO
                # existing primitive reads this — HL2 uses the MIDPOINT (high+low)/2,
                # ATR uses the SPAN (high-low), VWAP voter uses close vs a volume-
                # weighted TYPICAL price (a level deviation, not bar-shape). close_loc
                # is a pure intrabar CONVICTION signal: close near the high = buyers
                # controlled the bar (bullish), close near the low = sellers controlled
                # it (bearish). A trend-aligned entry whose entry bar closed strongly
                # in the trade direction is a higher-conviction trend entry -> larger
                # first-bar commitment captures more of the confirmed trend move ->
                # higher Sharpe in the trend regimes (rally 1.283 binding, crash 1.265
                # return-limited; both are sustained trends whose bars close in the
                # trend direction). Distinct from the saturated volume-participation
                # axis (this is PRICE bar-shape, not volume). 3-bar mean close_loc for
                # noise-robustness (single-bar close position flips under AR(1) noise);
                # deep-saturated gates (/0.15 -> near-constant where it fires, noise-free
                # per the validated safe-family lesson), trend-ALIGNMENT gated (/0.04
                # ret_long so only genuine trend entries boost -> spares sideways chop
                # where close position is mean-reverting noise), first-bar-only, small
                # +0.05 max, bilateral (boost on directional agreement). Direction-
                # agnostic general principle (no regime label). New cross-data-type dep.
                _cl_high = bd.history["high"].values[-3:]
                _cl_low = bd.history["low"].values[-3:]
                _cl_close = closes[-3:]
                _cl_span = np.maximum(_cl_high - _cl_low, 1e-10)
                _close_loc = float(np.mean((_cl_close - _cl_low) / _cl_span))  # [0,1], 3-bar mean
                # Branch step4: revert to ret_long trend gate (step1 was the best composite
                # -0.000003 vs step2 ret_vlong -0.000064) AND add a GRINDING-trend (low-vol)
                # condition. The leak in step1 was into bull-2021 (high-vol uptrend: -0.000270)
                # where sharp continuation bars are exhaustion-prone. close_loc continuation
                # is a GRINDING-trend signal (low vol, persistent) not a sharp-trend signal.
                # Gate on low vol_ratio (vol_ratio below ~1.2 = calm/grinding; rally grinds
                # at low vol, bull-2021 is high-vol sharp). Continuous tanh so no boundary.
                # This is a general bar-shape x vol-regime principle (no regime label): the
                # close-loc continuation signal is weighted by how grinding vs sharp the
                # recent regime is. Sideways (low vol, low trend) still gated off by the
                # trend-alignment term; crash (bear side) gated off by direction.
                _cl_trend_w = max(0.0, np.tanh(abs(ret_long) / 0.04))  # 0 chop, ~1 trend
                _cl_bull_conv = max(0.0, np.tanh((_close_loc - 0.55) / 0.15))  # fires close near high
                _cl_bear_conv = max(0.0, np.tanh((0.45 - _close_loc) / 0.15))  # fires close near low
                # Branch step6: replace vol-based grind gate with EFFICIENCY-RATIO gate.
                # Step4's vol gate (low vol_ratio) separated rally from bull-2021 (high vol)
                # but NOT from sideways (also low vol) -> sideways leak -0.000199. ER
                # (Kaufman, already computed) distinguishes DIRECTIONAL grind (rally, high
                # ER) from CHOPPY mean-reversion (sideways, low ER) at equal low vol.
                # close_loc continuation is a trending-market signal; gate it on path
                # efficiency (high ER = price moved efficiently one way = continuation
                # holds). rally grinding uptrend ER high; sideways chop ER low -> spared.
                # Continuous tanh on _er (no boundary). _er in [0,1], saturate /0.25.
                _cl_er_w = max(0.0, min(1.0, np.tanh(_er / 0.25)))  # ~0 chop, ~1 directional grind
                # Branch step7: multi-day direction gate on the BULL boost. Step6's crash
                # leak (-0.000354) is the bull boost firing on crash dead-cat-bounce longs
                # (sharp bounce: ret_long>0 trend, ER high directional, close near high ->
                # bull boost over-commits to the losing bounce). Require the MULTI-DAY
                # ret_vlong>0 for the bull boost (crash bounces have ret_vlong<0 -> excluded;
                # rally grind has ret_vlong>0 -> kept). ret_vlong is the validated multi-day
                # trend; tanh/0.03 fast-saturating (near-constant, noise-free). Bear boost
                # left ungated by ret_vlong (it is near-inert anyway, and crash shorts are
                # the trend-aligned crash trade). General principle: a close-loc LONG
                # continuation boost requires multi-day uptrend confirmation.
                _cl_bull_vlong = max(0.0, np.tanh(ret_vlong / 0.03))  # multi-day uptrend confirmation
                _close_conv_boost_bull = 1.0 + 0.05 * _cl_trend_w * _cl_er_w * _cl_bull_vlong * _cl_bull_conv
                _close_conv_boost_bear = 1.0 + 0.05 * _cl_trend_w * _cl_er_w * _cl_bear_conv
                # Exp1 (architectural, indep): DIRECTIONAL VOLUME PRESSURE (normalized
                # OBV) trend-aligned entry boost. NEW data axis genuinely orthogonal to
                # every existing volume primitive: VWAP voter reads close vs a volume-
                # weighted TYPICAL price (a LEVEL deviation), vol_rise reads the TREND of
                # TOTAL volume (a magnitude), _vol_z reads a volume LEVEL spike. NONE
                # measures the DIRECTIONAL BALANCE of volume -- whether participating
                # volume is concentrated on up-bars or down-bars. DVP = sum(vol[i] *
                # sign(close[i]-close[i-1])) / sum(vol[i]) over 12 bars, range [-1,+1].
                # +1 = all volume on up-bars (buying pressure), -1 = all on down-bars.
                # close-to-close sign uses only close (noise-perturbed -> legitimately
                # noise-sensitive, NOT the open-price artifact). Mechanism: a grinding
                # rally uptrend has volume on up-bars (DVP>0) even when TOTAL volume is
                # flat (the case vol_rise misses) -> a trend-aligned long confirmed by
                # buy-side volume participation is a higher-quality trend entry -> larger
                # first-bar commitment captures more of the confirmed trend move ->
                # higher Sharpe in the binding regime (rally 1.285 raw, all stab 1.0).
                # Symmetric on crash (sell-side volume confirms downtrend shorts). Mirrors
                # the validated close-loc boost envelope EXACTLY: trend_w (|ret_long|/0.04)
                # x ER grind gate (_er/0.25 - separates rally directional grind from
                # sideways chop at equal low vol) x multi-day ret_vlong>0 bull gate
                # (excludes crash dead-cat-bounce longs whose 20-bar ret_long is positive
                # during sharp bounces but 96-bar ret_vlong negative) x deep-saturated
                # conviction (/0.15 -> near-constant where it fires, noise-free per the
                # validated safe-family lesson). Bear side ungated by ret_vlong (mirrors
                # close-loc; crash shorts are the trend-aligned crash trade). First-bar-
                # only, +0.05 max, bilateral, shrink-side caps at 1.0. Direction-agnostic
                # general principle (no regime label). New cross-data-type dep.
                _dvp_n = 12
                _dvp_c = closes[-_dvp_n - 1:]
                _dvp_v = bd.history["volume"].values[-_dvp_n:]
                _dvp_rets = np.sign(np.diff(_dvp_c))
                _dvp = float(np.sum(_dvp_v * _dvp_rets) / max(np.sum(_dvp_v), 1e-10))
                _dvp_trend_w = max(0.0, np.tanh(abs(ret_long) / 0.04))  # 0 chop, ~1 trend
                _dvp_er_w = max(0.0, min(1.0, np.tanh(_er / 0.25)))  # ~0 chop, ~1 directional grind
                _dvp_bull_vlong = max(0.0, np.tanh(ret_vlong / 0.03))  # multi-day uptrend (excludes crash bounces)
                _dvp_bull_conv = max(0.0, np.tanh(_dvp / 0.15))   # buy-side volume pressure
                _dvp_bear_conv = max(0.0, np.tanh(-_dvp / 0.15))  # sell-side volume pressure
                _dvp_boost_bull = 1.0 + 0.05 * _dvp_trend_w * _dvp_er_w * _dvp_bull_vlong * _dvp_bull_conv
                _dvp_boost_bear = 1.0 + 0.05 * _dvp_trend_w * _dvp_er_w * _dvp_bear_conv
                # Exp1 (architectural): PERSISTENCE-COUNT-gated return-seeking first-
                # bar size boost. The DURATION-fraction _weak_persist (sanctioned
                # untested separator, results.tsv line 1476) gates a small first-bar
                # size boost. Byte-identical when _weak_persist=0 (trend regimes
                # where the count stays low); fires for mixed (persistently weak
                # multi-day trend -> count~1). Mirrors the prior headroom-boost
                # branch's +0.05 magnitude (its mixed +0.0112 was real; the +0.003
                # keep was blocked ONLY by the rally magnitude-gate leak, which this
                # duration-count separator is designed to avoid).
                # Exp2 (architectural, indep): CONVICTION-MARGIN-SCALED persist_boost
                # magnitude. NEW cross-component data dep: the fe6acd4d keep note
                # explicitly states 'MAG could go higher (leak solved)' but a UNIFORM
                # raise risks crash dead-cat-bounce weakenings (the THRESH 0.02 gate
                # excluded MOST but crash -0.0028 was the residual cost at MAG 0.12).
                # Instead of a uniform MAG raise, make the magnitude ADAPTIVE to entry
                # conviction quality: scale the boost up to 1.5x baseline MAG when the
                # entry conviction margin (how far strong-sum sits ABOVE its admission
                # floor) is high -- decisive entries get a bigger weak-trend size boost
                # (they are the high-quality mixed bounce-long winners; mixed is 100pct
                # WR so bigger winners = higher APY = return_bonus at preserved Sharpe).
                # Marginal entries (margin near 0) keep baseline MAG (no extra size on
                # coin-flip-quality entries that a uniform raise would over-commit).
                # This TARGETS the over-commitment risk of a uniform MAG raise: crash
                # dead-cat-bounce entries are MARGINAL-conviction (they barely pass the
                # gate during a counter-trend bounce) -> conviction scale ~0 -> boost
                # stays at baseline -> crash spared; mixed's DECISIVE bounce entries
                # (high conviction, the winners) -> conviction scale ~1 -> boost up to
                # 1.5x. New control flow: persist_boost magnitude depends on conviction
                # margin x weak_persist interaction (was weak_persist only). Amplify-
                # only (floor at baseline MAG; never shrinks below the fe6acd4d keep).
                # Smooth tanh on margin/0.40 (same scale as _bull_conv_atten; no new
                # boundary). Direction-agnostic (uses own-side margin).
                _persist_margin_side = _bull_margin if (_bull_ready and _bull_admit) else _bear_margin
                # Branch step3: gate the conviction-scale on _down_persist<0.75 to EXCLUDE
                # crash (persistent downtrend _down_persist~0.9 per diagnostics) from the
                # amplified boost. crash's trend-aligned SHORTS (100pct WR winners) get
                # weak_persist>0 during multi-week downtrend consolidation stretches AND
                # have high conviction margin -> step1 over-boosted them (crash -0.0013).
                # The separator: crash is a PERSISTENT downtrend (_down_persist~0.9), mixed
                # OSCILLATES (_down_persist~0.5). Gate the conviction amplification on
                # _down_persist<0.75 so crash (0.9) -> scale 1.0 (baseline MAG, byte-identical
                # to fe6acd4d for crash) while mixed (0.5) -> full conviction amplification.
                # _down_persist is the VALIDATED direction-agnostic duration-count signal
                # (the 52a3e671 keep mechanism). Smooth tanh fade on the 0.65 knee (so the
                # 0.5->0.65 region keeps full amplification, fading to 0 by ~0.85 = crash).
                # Amplify-only (floor 1.0 baseline MAG).
                _persist_down_gate = 1.0 - max(0.0, min(1.0, np.tanh((_down_persist - 0.65) / 0.10)))
                _persist_conv_scale = 1.0 + 0.50 * max(0.0, min(1.0, _persist_margin_side / 0.40)) * _persist_down_gate
                # Branch step8: CONSENSUS-AMPLIFIED persist_boost. Phase-3-style combination
                # of two validated mixed-targeting mechanisms: the persist_boost amplifier
                # (fe6acd4d keep, weak_persist-gated) AND the portfolio consensus boost
                # (step2, broad-market agreement). The persist_boost already fires for mixed
                # (weak_persist~1) at magnitude PERSIST_BOOST_MAG * conv_scale. Amplify its
                # magnitude by a SMALL factor (max +20%) when broad-market consensus ALSO
                # agrees with the entry direction (the same consensus_dir signal, weak_persist-
                # gated). Mechanism: a mixed bounce-phase long confirmed by BOTH persistent
                # weak multi-day trend AND broad-market 3-symbol consensus is a higher-quality
                # broad-trend entry -> larger persist_boost magnitude captures more of the
                # confirmed broad move -> higher mixed APY via the persist_boost path (DIFFERENT
                # from the entry-frac path that step2 used; the two paths compose multiplicatively
                # on the first-bar size but the persist_boost ALSO sustains through scale-in via
                # _persist_sustain, so amplifying it may reach mixed through the sustain path
                # that the entry-frac boost couldn't). Byte-identical when consensus absent or
                # entry opposes consensus or weak_persist~0 (rally/crash). Direction-agnostic.
                _persist_consensus_amp = 1.0 + 0.20 * _consensus_strength * _weak_persist * max(0.0, _consensus_dir if (_bull_ready and _bull_admit) else -_consensus_dir)
                _persist_boost = (1.0 + PERSIST_BOOST_MAG * _weak_persist * _persist_conv_scale) * _persist_consensus_amp
                # Exp3 (architectural, indep): TREND-ALIGNED x DD-HEADROOM entry-frac
                # BOOST. Exp1/Exp2 proved the 0.55 cap is inert (_entry_frac_dyn always
                # below 0.55); the real lever is _entry_frac_dyn ITSELF (range ~0.37-0.49).
                # The prior session's persist_boost amplifier raises mixed (weak-trend)
                # but is inert for crash (strong-trend -> weak_persist~0). Crash is the
                # SECOND-lowest regime (0.976, Sh1.31, 100pct WR, DD2.83pct huge headroom
                # below 5pct knee, return-limited). NEW control-flow: boost _entry_frac_dyn
                # for TREND-ALIGNED entries (entry direction matches ret_vlong sign) when
                # the portfolio has DD headroom (at peak equity). Crash trend-aligned shorts
                # (ret_vlong<0, bear entry -> aligned) get a larger first-bar commitment ->
                # higher APY (return_bonus) at preserved Sharpe (100pct WR winners; Sharpe
                # is scale-invariant so bigger trend-aligned entries raise return without
                # lowering Sharpe, and DD stays below the 5pct knee). The DD-headroom gate
                # spares rally pullback entries (rally pullbacks DRIVE the portfolio DD ->
                # during pullback entries dd_frac>0 -> _port_dd_atten<1 -> boost gated off
                # -> rally entry size unchanged -> rally DD stays 5.06pct). Trend-aligned
                # crash entries mostly happen at peak (crash is low-DD) -> boost fires.
                # Sideways (weak ret_vlong -> trend-align gate ~0) byte-identical. Bull
                # trend-aligned longs at peak get the boost (bull DD 2.50pct has headroom;
                # bull is Sh2.11 not return-limited so the gain is modest but safe). The
                # boost is a NEW multiplier on _entry_frac_dyn (not the cap): raises the
                # first-bar frac directly. Gated on multi-day trend-alignment (fast-
                # saturating /0.02 ret_vlong*pos_dir, near-constant noise-free per the
                # validated ct-gate lesson) x DD-headroom (the validated _port_dd_frac
                # signal, EMA-smoothed noise-robust). Direction-agnostic (uses entry
                # direction sign). New cross-component data dep: _entry_frac_dyn depends
                # on trend-alignment x portfolio DD-headroom (was vol-only).
                _frac_dd_headroom = 1.0 - max(0.0, min(1.0, np.tanh(_port_dd_frac / (0.005 * LEVERAGE_K))))
                _frac_trend_align = 0.0  # set per-entry-direction below
                # Exp2: portfolio cross-symbol trend-consensus entry boost (bilateral).
                # _consensus_strength (0..1) x _consensus_dir (+/-1) computed once per bar
                # at the top-level cross-symbol block. Boost first-bar size when the entry
                # direction AGREES with the broad-market consensus direction. Small +0.06 max,
                # deep-saturated (|sum|-2 gate -> near-constant where it fires, noise-free).
                # Byte-identical when consensus absent (<2 symbols) or entry opposes consensus.
                # Branch step1: GATE on _weak_persist (the validated mixed/rally separator
                # from the f6e19151 keep). Opener fired the boost for rally's grinding uptrend
                # (BTC/ETH/SOL all agree -> consensus=+1 -> boost on rally longs -> over-
                # committed -> rally -0.0166, DD 5.06->5.11pct past knee). rally has STRONG
                # multi-day uptrend (ret_vlong solidly positive -> _weak_persist~0); mixed has
                # WEAK/oscillating multi-day trend -> _weak_persist~1. Gate the boost on
                # _weak_persist so it fires for mixed's broad-market bounce-phase longs (the
                # +0.0049 gain source) NOT rally's grinding uptrend (the leak source). Same
                # structural property that solved the entry-frac boost rally-leak wall.
                # Branch step5: BEAR-SIDE consensus boost gated on _down_persist (the validated
                # crash/bear separator) instead of _weak_persist. Crash has STRONG multi-day
                # DOWNTREND (ret_vlong persistently negative -> _weak_persist~0 BUT
                # _down_persist~0.9). The bull-side _weak_persist gate (correct for mixed)
                # BLOCKS the bear-side boost for crash entirely -> crash byte-identical (the
                # f6e19151 keep's entry-frac boost was also inert for crash). Use a SEPARATE
                # gate for the bear side: _down_persist isolates crash (persistent downtrend
                # ~0.9) from bull (transient pullback dips ~0.3, same structural property that
                # made the down_persist duration-count work in the fe6acd4d keep). Crash
                # trend-aligned shorts on broad-market DOWN-LEGS (consensus_dir=-1, all 3
                # symbols down together during multi-month bear) get a bigger first-bar
                # commitment -> higher APY (return-limited floor 0.976, Sh1.31, DD2.83pct
                # huge headroom below 5pct knee). 100pct WR winners -> Sharpe scale-invariant
                # -> return_bonus at preserved Sharpe. Byte-identical when consensus absent,
                # entry opposes consensus, or _down_persist~0 (bull/sideways). Direction-
                # agnostic general principle (no regime label): broad-market downtrend
                # confirmation raises crash short entry quality. New cross-component data dep:
                # bear-side consensus boost depends on _down_persist (was weak_persist).
                _consensus_boost_bull = 1.0 + 0.10 * _consensus_strength * _weak_persist * max(0.0, _consensus_dir)
                _consensus_boost_bear = 1.0 + 0.10 * _consensus_strength * _weak_persist * max(0.0, -_consensus_dir)
                if _bull_ready and _bull_admit:
                    # branch step3->4: re-add bull-side boost GATED ON WEAK multi-day trend.
                    # Exp3 ungated bull boost gave mixed +0.0090 REAL but rally -0.06 DD
                    # cliff (rally DD 5.06->5.22pct past 5pct knee). step3's INSTANTANEOUS
                    # weak-trend gate (1-tanh(|ret_vlong|/0.02)) reduced rally to -0.013 but
                    # STILL leaked: rally pullbacks are TRANSIENT weak-trend episodes
                    # (ret_vlong dips below 0.02 for a few bars) -> instantaneous gate
                    # fires on those bars -> rally DD 5.06->5.086pct still past knee. step4
                    # uses the DURATION-COUNT _weak_persist (fraction of last 48 bars where
                    # |ret_vlong|<0.02) -- the validated separator (fe6acd4d keep) that
                    # solved the SAME rally-pullback-leak wall for the persist_boost.
                    # mixed: PERSISTENTLY weak multi-day trend -> _weak_persist~1. rally:
                    # TRANSIENT dips, re-strengthens between pullbacks -> _weak_persist~0.
                    # The duration-count isolates mixed's sustained weakness from rally's
                    # transient pullback dips (same structural property that made the
                    # persist_boost work for mixed without leaking to rally).
                    _frac_weak = _weak_persist  # ~1 mixed (persistently weak), ~0 rally (transient)
                    _frac_trend_align = max(0.0, np.tanh(ret_vlong / 0.02))  # bull long aligned with uptrend
                    _entry_frac_boost_bull = 1.0 + 0.15 * _frac_trend_align * _frac_weak * _frac_dd_headroom
                    target = size * min(0.55, _entry_frac_dyn * _entry_frac_boost_bull) * _cooldown_factor * _bull_ct_atten * _bull_ct_vlong * _bull_consensus_atten * _bull_quality_atten * _outcome_size_mult *_port_dd_atten * _bull_conv_atten * _churn_size_atten * _churn_ct_atten_bull * _tq_atten * _xasset_bull * _conc_shrink_bull * _net_tilt_shrink_bull * _vol_entry_spike * _vol_decline_shrink * _vd_ct_shrink_bull * _vol_rise_boost_bull * _vol_partner_boost_bull * _vol_btc_boost_bull * _btcvol_partner_boost_bull * _partnervol_btc_boost_bull * _close_conv_boost_bull * _dvp_boost_bull * _btcdvp_boost_bull * _partnerdvp_boost_bull * _streak_ct_shrink_bull * _persist_boost * _consensus_boost_bull * _cv_shrink_bull
                    self._conc_shrink_held[symbol] = _conc_shrink_bull
                    self._vol_shrink_held[symbol] = _vol_entry_spike  # Exp9: cache for scale-in sustain
                    self._cv_shrink_held[symbol] = _cv_shrink_bull  # Exp2 branch: cache for scale-in sustain
                elif _bear_ready and _bear_admit:
                    target = -size * min(0.55, _entry_frac_dyn) * _cooldown_factor * _bear_ct_atten * _bear_ct_vlong * _bear_consensus_atten * _bear_quality_atten * _outcome_size_mult *_port_dd_atten * _bear_conv_atten * _churn_size_atten * _churn_ct_atten_bear * _tq_atten * _xasset_bear * _conc_shrink_bear * _net_tilt_shrink_bear * _vol_entry_spike * _vol_decline_shrink * _vd_ct_shrink_bear * _vol_rise_boost_bear * _vol_partner_boost_bear * _vol_btc_boost_bear * _btcvol_partner_boost_bear * _partnervol_btc_boost_bear * _close_conv_boost_bear * _dvp_boost_bear * _btcdvp_boost_bear * _partnerdvp_boost_bear * _streak_ct_shrink_bear * _persist_boost * _consensus_boost_bear * _cv_shrink_bear
                    self._conc_shrink_held[symbol] = _conc_shrink_bear
                    self._vol_shrink_held[symbol] = _vol_entry_spike  # Exp9: cache for scale-in sustain
                    self._cv_shrink_held[symbol] = _cv_shrink_bear  # Exp2 branch: cache for scale-in sustain
            elif current_pos != 0:
                pos_pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                if current_pos < 0:
                    pos_pnl = -pos_pnl
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)
                # Exp1 (this session): maintain rolling pos_pnl PATH (12-bar) for the
                # MTM-path-efficiency signal consumed at the emission layer. Append the
                # CURRENT pos_pnl each held bar; efficiency computed below at emission.
                _pp_hist = self._pnl_path.get(symbol, [])
                _pp_hist.append(pos_pnl)
                if len(_pp_hist) > 12:
                    _pp_hist = _pp_hist[-12:]
                self._pnl_path[symbol] = _pp_hist

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
                # Architectural (Exp3 this session): trend-gated realized-PnL scale-in
                # ACCELERATION for early winners. Prior session removed a live-CONVICTION
                # scale-in accelerator (it made pace depend on per-bar voter margin =
                # noisy, added bull noise) — but recorded that removing it SLOWED rally
                # (raw 0.393->0.371), proving scale-in PACE moves rally raw. This re-adds
                # an accelerator on a SMOOTHER, realized signal (pos_pnl, cumulative not
                # per-bar voter) and gates it by trend strength so chop (sideways, where
                # early winners mean-revert) is spared. Mechanism: an early-winning
                # position is likely trend-aligned (bull long / crash short / rally long
                # riding the trend) -> reach full size faster to capture more of the
                # winning trend -> higher Sharpe (the raw lever, since all stability
                # factors are now 1.0). One-sided (only positive pos_pnl accelerates;
                # losers keep baseline + adverse-freeze). Smooth tanh on pos_pnl/|stop|.
                # Trend-gated by _trend_strength_w (0 in chop -> no accel, protects
                # sideways mean-reverters; ~1 in trend). Max 0.8 bars faster, floored at
                # 1.5 bars. New control flow on scale-in pace based on realized PnL.
                _win_accel = max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT))) * _trend_strength_w
                # Exp4 (architectural, indep): slope-confirmation gate on the accelerator.
                # Exp3 (baseline a73836dc) helped rally +0.021 but cost bull -0.018: the
                # accelerator grew bull positions right before 2021's sharp corrections.
                # Add a SHORT-TERM slope-confirmation requirement (16-bar OLS log-HL2 slope
                # agreeing with position direction): only accelerate when the near-term
                # slope is still confirming the position. bull's corrections are preceded
                # by slope weakening -> gate off -> bull spared; rally's grinding uptrend
                # has persistent positive slope -> gate on -> rally keeps the gain. One-
                # sided (max(0,...) only reduces accel). New cross-component data dep:
                # scale-in accel now depends on short-term slope agreement (entry voter
                # slope reused), layered on the long-window trend gate.
                _pos_dir_acc = 1.0 if current_pos > 0 else -1.0
                _slope_conf = max(0.0, np.tanh(_lr_slope * _pos_dir_acc / 0.0004))
                _win_accel = _win_accel * _slope_conf
                # Exp5 (architectural, indep): adaptive acceleration floor + stronger
                # magnitude. Exp3/Exp4 validated the accelerator (rally +0.021, bull
                # recovered via slope gate). The fixed 0.8 magnitude rarely saturates the
                # 1.5 floor (pace stays ~1.7), so the floor is not the binding limit —
                # the magnitude is. Make the floor ADAPTIVE: strong trends (rally/bull/
                # crash, high _trend_strength_w) get a lower floor (1.3) allowing more
                # acceleration; chop (sideways) stays at 1.5. Raise magnitude 0.8->1.2.
                # The slope-confirmation gate (Exp4) protects bull from over-acceleration
                # into imminent corrections (gate off when slope weakens). New control
                # flow: acceleration floor depends on trend strength.
                _accel_floor = 1.5 - 0.2 * _trend_strength_w  # 1.5 chop, 1.3 strong trend
                _entry_full_bars_dyn = max(_accel_floor, _entry_full_bars_dyn - 1.2 * _win_accel)
                # Exp4 (architectural, indep): VOL-OF-VOL regime scale-in pace modulation,
                # COUNTER-TREND-AT-MULTI-DAY GATED (refinement of Exp2 which applied to all
                # positions and catastrophically regressed rally -0.369 by slowing rally's
                # trend-aligned longs). Exp2's ONE bright spot was crash +0.026. The fix:
                # apply the cautious scale-in ONLY to counter-trend-at-multi-day positions
                # (the validated _ct_si_gate separator, already computed for _adv_freeze).
                # CT positions during vol-regime transitions are the most uncertain: they
                # fight BOTH the multi-day trend AND a shifting vol regime -> cautious scale-
                # in (let the regime clarify before committing fully). Trend-aligned positions
                # (rally longs, crash trend shorts, bull longs) keep FAST scale-in -> byte-
                # identical to baseline for the trend regimes that Exp2 catastrophically
                # regressed. NEW higher-order data axis (vol-of-vol, 2nd-order vol statistic,
                # absent from strategy) x ct-gate on the scale-in PACE path (not grid-blocked).
                # Continuous tanh on VoV/median_vol; max +0.6 bars slower at deep VoV for ct
                # positions only; floored at _accel_floor. min_stab preserved (ct-gate is
                # noise-immune integer/ret_vlong-derived; VoV is a smooth windowed stat).
                # Direction-agnostic general principle (no regime label): cautious scale-in
                # for counter-trend entries in a vol-flux regime. New cross-timescale x
                # cross-trend data dep at scale-in pace.
                _vov_n = 6
                if len(closes) >= 24:
                    _vov_samples = np.array([
                        float(np.std(np.diff(np.log(closes[-(i + _vov_n) - 1: -(i + 1) + 1])))) if i > 0 else float(np.std(np.diff(np.log(closes[-_vov_n - 1:]))))
                        for i in range(18)
                    ])
                    _vov_med = max(float(np.median(_vov_samples)), 1e-8)
                    _vov = float(np.std(_vov_samples)) / _vov_med
                    _vov_gate = max(0.0, min(1.0, np.tanh((_vov - 0.30) / 0.25)))
                    # _ct_si_gate computed BELOW for _adv_freeze; replicate the multi-day ct
                    # indicator here (scale-in pace block runs before _adv_freeze's gate).
                    _pos_dir_vov = 1.0 if current_pos > 0 else -1.0
                    _ct_vov_gate = max(0.0, np.tanh(-ret_vlong * _pos_dir_vov / 0.01))  # ~0 trend-aligned, ~1 ct-at-multi-day
                    _entry_full_bars_dyn = _entry_full_bars_dyn + 0.6 * _vov_gate * _ct_vov_gate
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
                    _ct_si_gate = max(0.0, np.tanh(-ret_long * _pos_dir_si / 0.04))  # [0,~1] counter-trend (20-bar)
                    # Exp7 (architectural, indep): add MULTI-DAY counter-trend term to the
                    # adverse-freeze gate. The 20-bar _ct_si_gate MISSES rally pullback
                    # shorts: during a pullback ret_long<0 (short-aligned) -> gate=0 -> no
                    # freeze, so the short scales in to full (un-ct_vlong-shrunk) size and
                    # loses bigger. ret_vlong (96-bar OLS, the validated multi-day ct signal
                    # used in ct_vlong shrink / max_hold / target EMA) stays POSITIVE through
                    # rally pullbacks (uptrend intact) -> the short IS counter-trend at the
                    # multi-day scale. Freeze if ct by EITHER measure (max). Catches rally
                    # pullback shorts + crash dead-cat longs (both multi-day ct losers) ->
                    # kept small through scale-in -> smaller losses -> higher Sharpe in the
                    # two low-Sharpe regimes. Trend-aligned (ret_vlong*pos_dir>0 -> 0) and
                    # 20-bar-trend-aligned positions unaffected. Fast-saturating /0.01
                    # (same as other ret_vlong ct gates -> near-constant, noise-free).
                    _ct_si_gate = max(_ct_si_gate, 0.6 * max(0.0, np.tanh(-ret_vlong * _pos_dir_si / 0.01)))
                    _adv_freeze = 0.75 * max(0.0, np.tanh(-pos_pnl / (0.4 * abs(STOP_LOSS_PCT)))) * _ct_si_gate
                    scale_frac = scale_frac * (1.0 - _adv_freeze)
                    # Exp1 (architectural, indep): FEED-FORWARD scale_frac quantization
                    # to a FIXED discrete fraction grid. The continuous scale-in ramp
                    # (ENTRY_INITIAL_FRAC + (1-ENTRY_INITIAL_FRAC)*progress) lands at a
                    # different continuous position value each bar; AR(1) close noise
                    # perturbs the inputs (_entry_full_bars_dyn via _vov_gate, _adv_freeze
                    # via _ct_si_gate) -> the scale_frac value wobbles bar-to-bar -> the
                    # held position value during scale-in differs across the noise
                    # ensemble -> stability tracking error. The prior session PROVED
                    # scale-in consistency raises rally raw +0.012 (real lever) BUT every
                    # TEMPORAL smoothing of scale-in (trend-aligned target EMA, alpha
                    # sweep) crashed bull stability 1.0->0.954 INVARIANT to alpha/vol/
                    # pressure/ER/growing gates -- smoothing-vs-raw tension is FUNDAMENTAL
                    # for trend-aligned (lag holds pullback longs bigger -> bull loss).
                    # Quantization is FEED-FORWARD (no temporal EMA, no lag): round the
                    # scale_frac to a FIXED discrete set so the continuous wobble
                    # collapses onto a FEW stable levels -> fewer distinct scale-in
                    # position values across the noise ensemble -> the rally raw +0.012
                    # consistency gain WITHOUT the bull lag crash. CRITICAL: the grid is a
                    # CONSTANT fraction set (round to nearest 1/N), NOT a width gated on a
                    # noisy quantity (the validated lesson: gating grid width on noisy
                    # _er/churn doubled boundary noise and crushed rally). Rounding to a
                    # fixed fraction grid is noise-IMMUNE in width (the step size is a
                    # constant); only the LEVEL selected can shift, and only when the raw
                    # scale_frac crosses a midpoint (a 1/N relative move -- far larger
                    # than sub-5bps close noise moves the inputs). Applied AFTER the
                    # _adv_freeze multiply so the freeze (real risk-reduction) executes
                    # first and only the residual ramp is quantized.
                    # BRANCH step2: VOL-GATE the quantization off for high-vol sharp
                    # uptrends (bull). Step1 crashed bull stab 1.0->0.826: bull's pullback
                    # longs are HIGH-vol SHARP uptrend entries (vol_ratio>1.0) where the
                    # exact scale-in ramp matters (sharp pullback recovers fast -- a
                    # quantized ramp misses the recovery timing). rally's beneficial
                    # quantization is in a LOW-vol GRIND (vol_ratio<0.8) where scale-in
                    # consistency is pure win. Gate the quantization strength on a vol ramp:
                    # FULL at vol_ratio<=0.8 (calm grind = rally), fading to 0 at
                    # vol_ratio>=1.2 (sharp = bull). Continuous tanh, no boundary. The
                    # validated bull/rally separator (multi-day trend strength was ALSO
                    # tried as the gate in the prior session's scale-in EMA branch and was
                    # INVARIANT -- bull and rally are both trend-aligned uptrends at the
                    # multi-day scale; VOL REGIME is the only axis that separates them:
                    # bull HIGH-vol sharp vs rally LOW-vol grind). Blend quantized<->
                    # continuous scale_frac by the vol gate so high-vol keeps the raw ramp.
                    # BRANCH step3: COUNTER-TREND gate (replaces vol gate which left bull
                    # stab crashed at 0.838). The vol gate was insufficient because bull's
                    # scale-in bars span vol 0.8-1.2 (partial quantization). The rally
                    # stability benefit is specifically on COUNTER-TREND scale-in positions
                    # (rally pullback SHORTS, which are counter-trend at the multi-day scale
                    # -- _ct_si_gate). Trend-aligned scale-in (bull longs, rally longs,
                    # crash shorts) gets byte-identical raw ramp. This isolates the
                    # quantization to exactly the noise-sensitive ct population (rally
                    # pullback shorts) while sparing ALL trend-aligned entries (bull). Uses
                    # the SAME _ct_si_gate already computed above (multi-day ct, fast-
                    # saturating /0.01 near-constant noise-free per validated lesson).
                    _sq_ct_gate = _ct_si_gate  # 0 trend-aligned, ~1 ct-at-multi-day
                    _sq_raw = scale_frac
                    _sq_quant = round(scale_frac * 8.0) / 8.0  # step5: finer 1/8 grid (less raw cost, keeps stability benefit)
                    scale_frac = _sq_raw * (1.0 - _sq_ct_gate) + _sq_quant * _sq_ct_gate
                    # Exp5: sustain the Exp4 entry-time concentration shrink through scale-in
                    # (cached at entry, deterministic). Keeps a concentrated book
                    # proportionally smaller for the whole hold instead of ramping back to
                    # un-shrunk `size` after bar 1. Default 1.0 (no effect) if uncached.
                    _conc_held = self._conc_shrink_held.get(symbol, 1.0)
                    # Exp9: sustain the Exp8 volume-spike entry shrink through scale-in
                    # (cached at entry, deterministic). Keeps a spike-chasing entry smaller
                    # for the whole hold instead of ramping back to un-shrunk `size` after
                    # bar 1. A SHRINK sustained (not a boost) -> smaller giveback on the
                    # spike-chasing trade (opposite of the failed xasset-sustain over-commit).
                    _vol_held = self._vol_shrink_held.get(symbol, 1.0)
                    # Exp2 branch: sustain the counter-velocity entry shrink through scale-in
                    # (cached at entry, deterministic). Keeps a trend-aligned counter-move
                    # entry smaller for the whole hold. Default 1.0 (no effect) if uncached.
                    _cv_held = self._cv_shrink_held.get(symbol, 1.0)
                    # Exp5 (architectural, indep, BRANCH CANDIDATE): DIRECTIONAL
                    # multi-day-downtrend-gated sustained persist_boost. Exp2 (this
                    # session, discarded) showed sustaining _persist_boost through scale-in
                    # raised mixed Sharpe 0.825->0.878 BUT collapsed rally (DD 5.12->30.59pct)
                    # because _weak_persist (the magnitude gate |ret_vlong|<0.02) ALSO saturates
                    # for rally's multi-bar weak-trend stretches. The DIRECTIONAL separator
                    # isolates the two: mixed is a multi-day DOWN year (ret_vlong persistently
                    # NEGATIVE for the held longs), while rally is a multi-day UP trend
                    # (ret_vlong persistently POSITIVE). Gate the sustained boost on a
                    # DIRECTIONAL multi-day downtrend: fire only when ret_vlong<0 (mixed),
                    # NOT when ret_vlong>0 (rally/bull uptrend -> byte-identical). This is
                    # the same validated ret_vlong signal used by the multi-day ct gates
                    # (ct_vlong, ct_hold_sat) but applied to SCALE-IN size sustenance, a new
                    # code path. Combined with _weak_persist (duration count, the keep's
                    # separator) the conjunction requires BOTH persistent weak trend AND
                    # negative multi-day direction -> isolates mixed's down-year oscillating
                    # longs from rally's uptrend pullbacks (which have ret_vlong>0). crash
                    # shorts (ret_vlong<0, pos_dir=-1): the boost applies to the position
                    # magnitude (sustains crash short size) -- but crash shorts are trend-
                    # aligned winning positions; a bigger sustained crash short captures
                    # more of the downtrend -> higher crash APY (crash is return-limited
                    # Sh1.25, DD 2.46pct has huge headroom). Sideways (ret_vlong~0) ->
                    # weak gate ~0 -> byte-identical. New cross-component data dep: scale-in
                    # full_target sustained boost depends on (weak_persist, ret_vlong sign).
                    _persist_down_gate = max(0.0, np.tanh(-ret_vlong / 0.02))  # ~0 uptrend, ~1 downtrend
                    # BRANCH step2 (crash sustained-boost refinement): replace step1's
                    # failed DEEP-downtrend threshold (bull pullback rv dips as deep as
                    # crash's persistent rv -> depth cannot separate) with a DURATION-
                    # based directional negativity gate _down_persist (fraction of last
                    # 48 bars where ret_vlong<0). Crash (multi-month bear -> ~0.9) vs bull
                    # (transient pullback dips -> ~0.3): the 48-bar FRACTION saturates for
                    # crash's persistent negativity but stays low for bull's transient dips
                    # (same structural property that made the magnitude duration-count work
                    # for the keep). Gate the AMPLIFIED magnitude increment (0.10) on
                    # _down_persist>0.6 (persistent downtrend); base 0.12 magnitude keeps
                    # the original /0.02 _persist_down_gate (Exp5 crash gain preserved at
                    # base level). Crash (down_persist~0.9) -> 0.22 amplified; bull (down_
                    # persist~0.3) -> 0.12 base -> byte-identical vs Exp5 keep. New data
                    # dep: amplified magnitude depends on directional-negativity duration.
                    _persist_deep_gate = max(0.0, np.tanh((_down_persist - 0.75) / 0.12))  # step3: 0.6->0.75 exclude bull extended pullbacks
                    # BRANCH step4: replace the BASE _persist_down_gate (instantaneous rv
                    # sign, the bull-leak source per step3 analysis) with the DURATION-based
                    # _down_persist gate for the WHOLE sustained boost (base + amplified).
                    # Step3 proved the bull residual leak (-0.0008) is from the Exp5 keep's
                    # base gate _persist_down_gate=tanh(-ret_vlong/0.02) firing on bull's
                    # transient pullback rv dips (rv briefly <0 during corrections). Route
                    # the ENTIRE sustained boost through the duration gate: fire only when
                    # _down_persist is high (persistent downtrend = crash ~0.9; bull's
                    # transient dips ~0.3 -> gate ~0 -> byte-identical vs Exp5 keep for bull).
                    # This recovers the bull leak AND keeps crash's gain (crash persistent
                    # downtrend -> full gate). Unifies base + amplified on the duration gate
                    # (the amplified increment 0.10 still requires the deeper 0.75 threshold,
                    # the base 0.12 requires the shallower 0.5 threshold so crash's full
                    # sustained gain from Exp5 is preserved).
                    _persist_down_gate_dur = max(0.0, np.tanh((_down_persist - 0.5) / 0.15))  # base gate: persistent downtrend
                    _persist_sustain_mag = PERSIST_BOOST_MAG + 0.10 * _persist_deep_gate
                    _persist_sustain = 1.0 + _persist_sustain_mag * _weak_persist * _persist_down_gate_dur
                    full_target = (size if current_pos > 0 else -size) * _conc_held * _vol_held * _cv_held * _persist_sustain
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
                # Exp1: portfolio-DD-adaptive giveback tightening. As the portfolio draws
                # down from its peak, shrink the effective giveback tolerance so pp_pressure
                # harvests winners faster (locks gains) -> caps DD from riding winners through
                # deep pullbacks. At 5x (rally DD near the 8pct knee) DD relief may now outweigh
                # the return_reward cost of earlier harvest. Continuous tanh on the DD fraction;
                # leverage-coupled scale keeps activation DD-LEVEL invariant; 0 at portfolio peak.
                _port_dd_frac = max(0.0, 1.0 - self._equity_ema / max(self._peak_equity, 1e-10))
                _pp_tighten = 1.0 - PORT_DD_GIVEBACK_TIGHTEN * max(0.0, np.tanh(_port_dd_frac / (PORT_DD_GIVEBACK_SCALE * LEVERAGE_K)))
                _pp_giveback_eff = PEAK_PROFIT_GIVEBACK * _pp_tighten
                _pp_lower = _pp_giveback_eff * (1.0 - _pp_band)
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
                _pp_raw = max(0.0, min(1.0, (_giveback_ratio - _pp_lower) / (_pp_giveback_eff * _pp_band)))
                _pp_pressure = _pp_raw * _pp_activation
                # Exp3 (architectural, indep): TREND-ALIGNED-WINNER pp_pressure attenuation.
                # bull_2021 has 72.9pct WR but PF 0.7 (avg win ~0.26x avg loss); _pp_pressure
                # (giveback) is the BINDING soft-exit for winners (MAX fusion -> changes to
                # non-max terms like _sl_slope_pressure are absorbed, confirmed Exp2). To
                # raise avg win (-> higher PF/Sharpe for bull), let trend-aligned winners
                # ride pullbacks by attenuating _pp_pressure when: (1) trend-aligned at the
                # multi-day scale (ret_vlong*pos_dir>0 -> the uptrend/downtrend is intact
                # through the pullback), (2) in profit (pos_pnl>0 -> a developing winner,
                # NOT a loser — losers byte-identical since pos_pnl<0 -> profit-gate 0),
                # (3) past scale-in (bars_held>ENTRY_FULL_BARS -> only established winners,
                # fresh entries still get full pp protection). NEW cross-component data dep:
                # pp_pressure depends on (trend-align, profit, hold-duration). Max 35pct
                # attenuation (let winner keep up to 35pct more of its peak through pullback);
                # /0.01 ret_vlong scale (near-constant noise-free); tanh profit-gate saturates
                # by ~1x stop in profit. Trend-aligned losers (crash shorts in profit then
                # giving back) are SPARED the attenuation only while in-profit -- once they
                # dip negative the profit-gate goes to 0 -> full pp protection resumes.
                _pos_dir_pp = 1.0 if current_pos > 0 else -1.0
                _ta_winner_gate = max(0.0, np.tanh(ret_vlong * _pos_dir_pp / 0.01)) * max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT))) * (1.0 if bars_held > ENTRY_FULL_BARS else 0.0)
                # Branch step2: GIVEBACK-MAGNITUDE gate. step1 let crash's trend-aligned
                # shorts ride into dead-cat bounces (sharp reversals) -> crash DD 20pct.
                # Distinguish bull's GRADUAL pullbacks (small giveback_ratio, attenuation
                # stays on) from crash's SHARP bounces (large giveback_ratio, attenuation
                # turns off -> pp_pressure fires normally -> exit the reversing short).
                # Full attenuation when giveback_ratio<0.10 (gradual pullback), fading to 0
                # by giveback_ratio=0.30 (sharp reversal). The giveback_ratio is the SAME
                # signal pp_pressure ramps on, so this gates the attenuation OFF precisely
                # when pp_pressure is about to fire strongly -- protects crash's shorts from
                # riding bounces while preserving bull's gradual-pullback winner relief.
                _gb_mag_gate = max(0.0, 1.0 - max(0.0, (_giveback_ratio - 0.15) / 0.20))
                # Branch step3: SHORT-TERM SLOPE-AGAINST gate (faster than giveback). The
                # giveback gate (step2) was too slow for crash bounces -- by the time
                # giveback_ratio>0.30 the bounce has already done damage. Add a slope-
                # against magnitude gate: a SHARP bar-to-bar reversal (large _slope_against)
                # turns off the attenuation immediately, before giveback accumulates. Bull's
                # gradual pullbacks have small slope-against; crash's sharp bounces have
                # large slope-against. Uses the SAME _slope_against already computed for
                # _sl_slope_pressure (no new input). Full attenuation below slope_against
                # 0.0004, fading to 0 by 0.0012 (fast-saturating, distinguishes gradual from
                # sharp reversal). Continuous tanh-style ramp.
                _slope_against_gate = max(0.0, 1.0 - max(0.0, (_slope_against - 0.0004) / 0.0008))
                # Branch step4: LONG-ONLY gate. steps2-3 gates were too slow -- crash's shorts
                # ride bounces faster than giveback/slope gates detect. Structural asymmetry:
                # LONGS in an uptrend riding a pullback (bull) is the canonical let-winners-run
                # scenario; SHORTS in a downtrend riding a bounce (crash) face asymmetric upside
                # risk (downtrends are choppier, bounces sharper). Position direction is a
                # structural property (long/short risk asymmetry), NOT a regime label. Gate the
                # attenuation to LONG positions only -- bull longs get pullback relief, crash
                # shorts get full pp_pressure (exit bounces normally). Shorts byte-identical to
                # baseline. This is the cleanest separator: crash's problem is specifically its
                # trend-aligned SHORTS; bull's benefit is specifically its trend-aligned LONGS.
                _long_only_gate = 1.0 if current_pos > 0 else 0.0
                # Branch step5: UPTREND-PERSISTENCE gate. step4 long-only still blew up crash
                # because ret_vlong (96-bar) FLIPS positive during crash bounces (e.g. Nov 2021
                # bounce off Luna low) -> crash bounce longs become trend-aligned-at-ret_vlong
                # -> attenuation fires -> ride bounce top -> down-trend resumes -> hit stops ->
                # DD explodes. The separator between bull (PERSISTENT uptrend) and crash (TRANSIENT
                # bounce uptrend) is the DURATION of ret_vlong positivity, not the instantaneous
                # sign. Use _down_persist (fraction of last 48 bars where ret_vlong<0, already
                # computed line ~910): bull ~0.3 (transient pullback dips), crash ~0.9 (persistent
                # bear). Gate attenuation to fire only when _down_persist is LOW (persistent uptrend).
                # Full attenuation below 0.40, fading to 0 by 0.60. This is the SAME validated
                # duration-count separator principle (fe6acd4d keep) applied to the pp_pressure
                # attenuation gate. Crash (down_persist~0.9) byte-identical (gate 0); bull
                # (down_persist~0.3) gets full attenuation. Continuous ramp.
                _up_persist_gate = max(0.0, 1.0 - max(0.0, (_down_persist - 0.40) / 0.20))
                _ta_winner_gate = _ta_winner_gate * _gb_mag_gate * _slope_against_gate * _long_only_gate * _up_persist_gate
                # Exp1-3 (this session): magnitude 0.35 -> 0.50 -> 0.65 -> 0.80 (3 KEEPS, +0.0125 composite
                # total; bull -0.3006->-0.2517; crash byte-identical throughout). Decelerating but still
                # crossing +0.003 at 0.80.
                # Exp4 (this session): push magnitude 0.80 -> 0.95. Testing the floor of the magnitude
                # axis. 0.95 leaves only 5% pp_pressure active at full gate pass -- near-complete
                # suppression of giveback-pressure on trend-aligned persistent-uptrend long winners.
                # The gates (giveback_ratio<0.10, slope_against<0.0004, long-only, up_persist<0.40)
                # are the safety: attenuation ONLY fires on gradual pullbacks in persistent uptrends,
                # so sharp reversals (gates off -> attenuation=0) keep full pp protection. If this
                # still crosses +0.003 the axis has headroom; if it drops below, the floor is ~0.80.
                _pp_pressure = _pp_pressure * (1.0 - 0.95 * _ta_winner_gate)

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
                # Exp5 (this session): FAST-SATURATING counter-trend max_hold shortening.
                # Re-attempt of Exp3 (which raised rally raw +0.0064 by cutting ct losers
                # faster via the noise-immune bar-counter, but cost stability -0.0027). The
                # Exp3 failure mode: its shortening amount 2*tanh(-pos_dir*ret_vlong/0.04)
                # sat in the LINEAR tanh region, so noisy ret_vlong made the AMOUNT of
                # shortening — and thus the exit bar — vary across the AR(1) ensemble ->
                # TE up. The codebase's own branch-step-9 lesson (entry ct-shrink): a
                # FAST-saturating gate (scale 0.01) puts rally's solidly-positive multi-day
                # ret_vlong in the FLAT saturated tail of tanh, so the shrink is a near-
                # CONSTANT (sensitivity ~0.4 vs ~5 at scale 0.04) -> a large but NOISE-FREE
                # shortening. The exit then fires at a deterministic bars_held (bar counter
                # is noise-immune; clean and perturbed exit the SAME bar -> zero TE) while
                # keeping the faster-ct-loser-exit raw gain. Trend-aligned holds (pos_dir*
                # ret_vlong>0 -> gate 0) keep max_hold unchanged -> byte-identical; low-
                # ret_vlong sideways spared. New mechanism: near-binary saturated time-cap
                # routing (vs Exp3's mid-slope linear shortening).
                _ct_hold_sat = max(0.0, np.tanh(-(1.0 if current_pos > 0 else -1.0) * ret_vlong / 0.01))
                # Exp3 (architectural, indep): PORTFOLIO-DD-ADAPTIVE max_hold EXTENSION for
                # LONG-ONLY trend-aligned positions. Exp2 (discarded) extended max_hold for
                # ALL trend-aligned positions during DD and catastrophically regressed crash
                # (-1.336, the trend-align gate ret_vlong*pos_dir>0 INCLUDES crash trend
                # shorts -> they rode dead-cat bounces). The structural fix: position
                # direction is a structural property (long/short risk asymmetry, NOT a regime
                # label -- same _long_only_gate pattern used at line ~2602 for pp_pressure
                # attenuation). LONGS in an uptrend during DD face pullback noise that
                # RECOVERS (bull pullback longs, rally grind longs); SHORTS in a downtrend
                # during DD face asymmetric upside risk (crash dead-cat bounces are sharper
                # than bull pullbacks). Gate the DD-extend on LONG-ONLY (current_pos>0) so
                # crash trend shorts are byte-identical (gate 0) while bull/rally trend longs
                # get the extended hold. NEW cross-component data dep: max_hold reads
                # (portfolio-DD state, trend-align, position direction) jointly. DISTINCT
                # from walled held-position de-risk (row-1015: cut HELD positions at a LOSS
                # during DD-pullbacks; this delays time-pressure onset only, pp/slope/stop
                # still fire on real reversals). Byte-identical when _port_dd_atten=1.0
                # (portfolio peak) AND when long-only gate=0 (shorts + ct positions). The
                # trend-align gate uses the SAME fast-saturating /0.01 ret_vlong scale as
                # _ct_hold_sat (near-constant noise-free per the validated lesson). Continuous
                # tanh, no new decision boundary. Direction-asymmetric structural property
                # (long/short risk asymmetry), NOT a regime label.
                _ta_long_gate = 1.0 if current_pos > 0 else 0.0  # long-only structural gate
                _ta_dir = 1.0 if current_pos > 0 else -1.0
                _ta_align = max(0.0, np.tanh(ret_vlong * _ta_dir / 0.01))  # ~0 ct, ~1 trend-aligned
                # branch step3: IN-PROFIT gate to spare sideways losers. step1 regressed
                # sideways -0.0108: sideways longs in brief uptrend stretches that are LOSING
                # (during sideways DD) got the extend -> held losers longer -> realized losses
                # that mean-revert away. The pp_pressure attenuation (line ~2569) uses the SAME
                # in-profit gate (tanh(pos_pnl/|stop|)) to restrict let-winners-run to
                # DEVELOPING WINNERS only (losers byte-identical since pos_pnl<0 -> gate 0).
                # Apply the same principle here: only extend the hold for longs that are
                # CURRENTLY IN PROFIT (a developing winner, not a loser riding the trend down).
                # sideways LOSING longs (pos_pnl<0 -> profit_gate 0 -> no extend -> byte-
                # identical); sideways/bull/rally WINNING longs (pos_pnl>0 -> profit_gate 1
                # -> extend fires). This is the validated fresh-winner-vs-loser separator
                # used by _ta_winner_gate (line 2569) and _grad_gate (line ~3345). Continuous
                # tanh on pos_pnl/|stop| (no decision boundary). Byte-identical for shorts
                # (long gate 0), ct (align 0), losers (profit gate 0).
                _ta_profit_gate = max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # 0 loss, ~1 profit
                _ta_dd_hold_ext_raw = 1.5 * _ta_long_gate * _ta_align * _ta_profit_gate * (1.0 - _port_dd_atten)
                # BRANCH (Exp1 opener, architectural): SMOOTH the hold-extension DECISION
                # INPUT at the source. The keep 47cbe827 added bar-to-bar tracking error at
                # the source: _ta_dd_hold_ext wobbles under AR(1) noise (profit_gate flips
                # around breakeven -> _max_hold jumps -> _time_pressure jumps -> emitted
                # target jumps). The prior session's 3bd20444 keep smoothed the EMITTED
                # target downstream which only partially compensated (bull stability dipped
                # 0.787->0.780). This EMAs the _ta_dd_hold_ext MAGNITUDE itself so _max_hold
                # is temporally stable under AR(1) noise -> the hold-DECISION tracking error
                # is damped at the source. NEW cross-component data dep: _max_hold reads a
                # temporally-smoothed hold-extension magnitude (was the instantaneous product).
                # ASYMMETRIC one-pole low-pass (the stab/raw tension resolution, mirroring the
                # ct-side _target_ema loss-gate and the ta-side pp-harvest exemption patterns):
                # SLOW to RISE (alpha 0.55 low-pass on the rising edge -> dampens the AR(1)
                # wobble that is the tracking-error source, the stability benefit) but FAST to
                # FALL (alpha 0.15 near-instant release -> when profit_gate->0 as a winner
                # turns loser, the hold-extension releases immediately -> losers NOT held
                # longer -> no raw cost on the winner->loser transition). The prior session's
                # symmetric EMA walled here: lag on winner->loser hurt sideways/rally raw.
                # Byte-identical for the never-active population (shorts/ct/losers/outside-DD:
                # raw stays 0 -> EMA converges to 0 -> no extension); continuous, no boundary.
                _prev_he = self._hold_ext_ema.get(symbol, _ta_dd_hold_ext_raw)
                if _ta_dd_hold_ext_raw >= _prev_he:
                    _he_alpha = 0.55  # slow rise: low-pass the wobble (stability). step5 scale-in exemption was byte-identical (the term is rarely active during scale-in), reverted.
                else:
                    _he_alpha = 0.15  # fast fall: release immediately on winner->loser (raw)
                _ta_dd_hold_ext = (1.0 - _he_alpha) * _ta_dd_hold_ext_raw + _he_alpha * _prev_he
                self._hold_ext_ema[symbol] = _ta_dd_hold_ext
                _max_hold = HOLD_DECAY_START + (1.0 / HOLD_DECAY_RATE) + _hold_adj - 2.0 * _ct_hold_sat + _ta_dd_hold_ext
                # Exp (architectural, indep): VOL-NORMALIZED time-pressure activation.
                # NEW data dep in the time-pressure subsystem: max_hold (in BAR units) is
                # currently vol-blind — 6 bars in calm sideways == 6 bars in crash, but 6
                # crash bars = a much larger REAL price move (high vol) than 6 sideways bars.
                # So time-pressure fires "too fast" in real-move terms in high-vol regimes,
                # cutting crash/rally winners before the bigger per-bar move fully plays out
                # -> contributes to crash being return-limited (Sh1.26, 100pct WR, DD3.04pct
                # headroom). Scale max_hold UP with vol_ratio so the hold window is
                # vol-normalized (same amount of REAL price move before time-pressure fires).
                # Continuous tanh on (vol_ratio-1)/0.5, max +12pct at vol_ratio>=1.5; calm
                # (vol_ratio<1) byte-identical (gate floored at 0). New control flow: a vol
                # term in the time-pressure activation. No per-regime labels.
                _vol_hold_ext = max(0.0, np.tanh((vol_ratio - 1.0) / 0.5))
                _max_hold *= 1.0 + 0.12 * _vol_hold_ext
                # Exp2 (this session): VOL-NORMALIZED time-pressure RAMP WIDTH. The fixed
                # 4-bar ramp (_time_pressure onset over 4 bars past _max_hold) treats 4 crash
                # bars (high vol = large real price move) the same as 4 sideways bars (low vol
                # = small move). The _vol_hold_ext above extends _max_hold's LEVEL (+12pct)
                # but the RAMP WIDTH (how gradually pressure ramps from 0->1 once it starts)
                # is still vol-blind. In high-vol a sharper ramp (narrower width) cuts winners
                # fast on the first time-pressure bar; a wider ramp in high-vol lets the
                # position ride the larger per-bar real move more gradually before full
                # time-pressure fires -> crash/rally trend-aligned winners (the high-vol
                # trend regimes) get a smoother de-risk -> higher Sharpe. Distinct from
                # _vol_hold_ext (LEVEL offset, byte-identical when vol_ratio<=1.0): this
                # changes the ramp SHAPE. Continuous tanh on (vol_ratio-1)/0.5, ramp width
                # 4.0 (calm, byte-identical) -> 6.0 (high vol, +50pct wider). New control
                # flow: time-pressure ramp width depends on vol_ratio. Direction-agnostic
                # general principle (no regime label): the de-risk graduation should be
                # vol-normalized in width, not just level.
                # BRANCH step2: lower the vol_ratio onset to 0.8 AND widen max to 8.0. Step1
                # (onset 1.0, max 6.0) gave crash +0.00096 but the tanh/(vol_ratio-1)/0.5
                # saturates by vol_ratio~1.5 -- crash's elevated vol_ratio may sit in the
                # flat tail where the ramp is already near-max, so the +0.00096 is the
                # saturated ceiling. Lowering onset to 0.8 engages the widening EARLIER (at
                # crash's vol_ratio which may be ~1.0-1.3, not past 1.5) and widening max to
                # 8.0 (+100pct) gives more headroom. Sideways/rally (vol_ratio<0.8) stay
                # byte-identical. Tests whether the crash gain scales with ramp width.
                _tp_ramp_w = 4.0 + 4.0 * max(0.0, np.tanh((vol_ratio - 0.8) / 0.4))
                _time_pressure = max(0.0, min(1.0, (bars_held - _max_hold + 3.0) / _tp_ramp_w))

                # PnL-conditioned exit-pressure weighting (architectural change to fusion):
                # In profit (pos_pnl > 0), peak-profit dominates — preserve gains via giveback.
                # In loss (pos_pnl < 0), slope-against dominates — cut losers via momentum reversal.
                # Stop-loss and time pressure stay at unit weight (protective + structural).
                # Smooth transition via tanh of pos_pnl scaled by stop magnitude.
                _pnl_scale = np.tanh(pos_pnl / abs(STOP_LOSS_PCT))   # in [-1, 1]
                # Exp6 (architectural simplification, indep): REMOVED dead _scale_in_w
                # = 1.0 stub. The prior _scale_in_w slope-pressure attenuator (0.5..1.0
                # ramp dampening slope-against during scale-in) was removed in a prior
                # simplification but the =1.0 stub and its multiplicative use in _w_pp
                # remained. Multiplying by 1.0 is a pure no-op. Removing follows the
                # 12b6bc63/Exp4/Exp5 byte-identical simplification precedent. (_scale_in_winning
                # at line ~2440 is a SEPARATE variable, unaffected.)
                _w_slope = 1.0 + 0.15 * max(0.0, -_pnl_scale)  # heavier in loss
                # Architectural: vol-conditioned profit-side _w_pp.
                # Low vol (sideways/rally): _w_pp simplified (no extra boost).
                #   Peak-profit pressure already amplifies via _profit_magnitude + _pp_activation.
                # High vol (crash): restore profit-side amplification — crash recovery profits
                #   are short-lived and need fast giveback locking.
                # Continuous tanh on (vol_ratio - 1.0)/0.4 — smooth transition around vol_ratio=1.
                _vol_w_pp_gate = max(0.0, np.tanh((vol_ratio - 1.0) / 0.4))  # in [0, ~1]
                _w_pp    = (1.0 + 0.20 * max(0.0, _pnl_scale) * _vol_w_pp_gate)
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
                # branch step3: MULTI-DAY-CONFIRMATION boost to the opp-bias trend-align
                # term, WEAK-TREND-GATED (replaces step2's ct-blend which lost mixed's
                # +0.013). Re-analysis of step1 (pure ret_vlong, mixed +0.013): the mixed
                # benefit came from ret_vlong STRENGTHENING opp-atten on mixed's TREND-
                # ALIGNED positions (ret_vlong*pos_dir>0 during mixed's rally phases) ->
                # the 96-bar ret_vlong is more STABLE than the 20-bar ret_long (doesn't
                # flicker off during pullbacks) -> sustained protection -> ride mixed's
                # rally-phase longs longer -> higher mixed Sharpe (NOT dead-capital trim;
                # the prior step2 diagnosis was wrong). Step2's ct-blend gave trend-aligned
                # positions the ret_long term (baseline) -> no change for trend-aligned
                # mixed -> mixed benefit lost. FIX: ADD a ret_vlong confirmation term to
                # the baseline ret_long term (strengthen opp-atten when BOTH agree) so
                # trend-aligned mixed positions get the sustained multi-day protection.
                # GATE on WEAK multi-day trend (|ret_vlong|<0.03, mixed's regime) so
                # STRONG-trend bull (|ret_vlong|~0.027... actually bull ret_vlong ~+0.027
                # is NOT weak) -- use a DIFFERENT gate: the 20-bar ret_long MAGNITUDE.
                # mixed's rally phases have MODEST 20-bar ret_long (weak local trend);
                # bull's grind has LARGE 20-bar ret_long. Gate the boost on modest
                # |ret_long| so bull (strong 20-bar) gets ~no boost (ret_long term already
                # ~1 -> can't boost further) while mixed (modest ret_long -> ret_long term
                # ~0.3-0.6 -> room to boost toward 1). The boost is naturally self-limiting
                # for bull (ret_long term already saturated). max(0, min(1, ret_long_term
                # + boost)) caps at 1.0 -> bull byte-identical (term already 1). Continuous
                # tanh, no boundary. ct positions (ret_vlong*pos_dir<0 -> ret_vlong_term 0)
                # get NO boost -> opp-bias full (fast exit for rally pullback shorts +
                # mixed wrong-side longs that ARE ct). Byte-identical for strong-trend
                # regimes (bull/crash/rally-longs: ret_long term ~1 -> capped -> no change).
                _ret_long_term_vb = max(0.0, np.tanh(ret_long * _pos_dir_vb / 0.05))   # baseline term (20-bar, /0.05)
                _ret_vlong_term_vb = max(0.0, np.tanh(ret_vlong * _pos_dir_vb / 0.04)) # multi-day confirmation (slower /0.04 = smooth, not near-binary)
                # branch step4: VOL-GATE the vlong boost on LOW vol_ratio (calm grind).
                # step3 recovered mixed +0.006 (ret_vlong confirmation sustains opp-atten
                # on mixed's trend-aligned rally-phase longs) BUT bull -0.34 (stab 1.0->
                # 0.881): the 96-bar ret_vlong LAGS bull's sharp 2021 corrections (stays
                # positive while ret_long dips -> boost keeps opp-atten high -> holds
                # losing bull longs through corrections -> stability crash). The
                # bull/mixed separator is VOL REGIME (the validated scale-in quantization
                # keep separator: bull HIGH-vol SHARP vs rally/mixed LOW-vol GRIND). Gate
                # the boost on low vol_ratio so it fires in the calm grind (mixed/rally)
                # and ~off in the sharp high-vol regime (bull). Continuous tanh, no boundary.
                _vlong_vol_gate = max(0.0, min(1.0, (1.2 - vol_ratio) / 0.4))  # ~0 vol_ratio>=1.2, ~1 vol_ratio<=0.8
                _vlong_boost_vb = 0.30 * _vlong_vol_gate * _ret_vlong_term_vb  # small additive boost when multi-day confirms AND low-vol grind
                _trend_align_vb = min(1.0, _ret_long_term_vb + _vlong_boost_vb)
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
                # Exp3 (architectural simplification, this session): removed the dead
                # _ep_pressure (early-profit-lock) and _ar_pressure (adverse-recovery)
                # terms. Both were already zeroed (_ep_pressure=0.0 since the
                # early-profit-lock removal; _ar_pressure=0.0 since the 5cc01688 keep)
                # so they contributed 0 to _soft_max via _w_ep*_ep_pressure and
                # _w_ar*_ar_pressure -- pure dead-code in the MAX-fusion tuple. Removed
                # the zeroed assignments and their tuple entries (6 terms -> 4). Also
                # removed the dead _exit_press_ema state (initialized + popped on exit
                # but never read/written after the exit-pressure EMA removal; only
                # _voter_bias_ema is live). The _mae state is RETAINED (still used by
                # _ts_supp at the tp-harvest gate). Code-structure removal: -2 dead
                # assignments, -2 dead tuple entries, -1 dead state dict + its reset pop.
                # Exp4 (architectural, indep): volume-climax exit pressure (6th soft source).
                # NEW data dependency: volume is used in entry (VWAP voter, calm_boost) but
                # NEVER in the exit subsystem — all 5 existing soft sources (slope/pp/time/
                # ve/ar) use price-derived series only. A volume spike after a winning run
                # is a classic exhaustion/climax signature (rally tops, crash capitulation
                # bounces) -> harvest the winner before the pullback. Distinct from
                # _ve_pressure (vol-of-PRICE expansion, not volume) and _pp_pressure (peak
                # giveback magnitude, not bar-volume). Profit-side only (lock gains at
                # exhaustion; don't punish losers for volume - slope-against handles them).
                # Compute 20-bar volume z-score; activate above ~2 sigma, saturate ~4 sigma.
                # Continuous tanh, no boundary. New exit-pressure source + new control flow
                # in the MAX fusion. Targets rally raw (volume-climax tops precede pullback
                # giveback - the documented rally drag).
                _vol_arr_e = bd.history["volume"].values[-21:-1]
                _vol_mean_e = float(np.mean(_vol_arr_e))
                _vol_std_e = max(float(np.std(_vol_arr_e)), 1e-10)
                _vol_z = (float(bd.history["volume"].values[-1]) - _vol_mean_e) / _vol_std_e
                _vc_pressure = 0.50 * max(0.0, min(1.0, np.tanh((_vol_z - 2.0) / 1.5)))
                _w_vc = max(0.0, _pnl_scale)  # profit-side only

                # ARCHITECTURAL (Exp3, v6 session): BREAK-EVEN STAGNATION EXIT PRESSURE.
                # Under scoring v6 (proper 200-bar warmup), the strategy LOSES in bull/
                # crash/sideways (PF 0.7/0.3/0.4) -- many positions survive scale-in then
                # hover near breakeven (pos_pnl ~ 0) before eventually bleeding to the
                # stop or soft-exit at a loss. There is NO position-level break-even exit:
                # a position that has been held past scale-in but made no progress is dead
                # capital that frequently turns into a loser. NEW exit-pressure source +
                # NEW control flow in the MAX fusion: a soft pressure that fires when
                # bars_held > ENTRY_FULL_BARS (past scale-in) AND |pos_pnl| is within a
                # small band around breakeven. The pressure ramps UP the longer the
                # position stagnates near zero (so a fresh breakeven bar after scale-in is
                # fine, but 4+ breakeven bars signal a stuck position) and ramps DOWN as
                # |pos_pnl| moves away from zero (a profitable position is left alone; a
                # clearly losing position is left to slope-against/stop). Smooth tanh on
                # both axes (no boundary). Weight profit-side-neutral (fires regardless
                # of sign -- a stuck winner is also dead capital, but the |pos_pnl| gate
                # keeps it from firing on real winners). NEW data dep: exit pressure
                # depends on (bars_held, pos_pnl magnitude) jointly -- the position's
                # PnL trajectory shape, absent from all 5 existing soft sources (each
                # reads slope/vol/peak-giveback/time/volume, none reads the LEVEL of
                # pos_pnl relative to breakeven over the hold). Distinct from
                # _pp_pressure (peak-to-current giveback, not absolute level) and
                # _time_pressure (bar count alone, not PnL-gated).
                _be_pnl_band = 0.5 * abs(STOP_LOSS_PCT)  # |pos_pnl| band around breakeven
                _be_near_zero = max(0.0, 1.0 - abs(pos_pnl) / max(_be_pnl_band, 1e-6))  # 1 at BE, 0 outside band
                # branch step12: FASTER hold-gate ramp for the MAE-conditioned path. The
                # baseline ramp (4 bars post-scale-in to saturate) is tuned for the trend-
                # gated path. For the MAE-conditioned CHOP path, sideways's short-lived dead-
                # capital stalls may exit before the 4-bar ramp saturates -> the MAE gate
                # never reaches full strength on the stalls it should cut. Use a FASTER ramp
                # (2 bars) on the MAE path so the pressure fires sooner on deep-MAE stalls.
                # The trend-gate path keeps the baseline 4-bar ramp (crash/bull byte-identical).
                # Continuous, no boundary (just a steeper tanh-equivalent linear ramp).
                _be_hold_gate_trend = max(0.0, min(1.0, (bars_held - ENTRY_FULL_BARS - 1.0) / 4.0))
                _be_hold_gate_mae = max(0.0, min(1.0, (bars_held - ENTRY_FULL_BARS - 1.0) / 2.0))
                # BRANCH step1: GATE the break-even pressure to TRENDING regimes only.
                # Exp3 (ungated) cratered sideways -0.731 (WR 50pct->37pct) because in chop
                # positions legitimately hover near breakeven then RECOVER -- the break-even
                # exit killed mean-reversion recoveries. rally/mixed (trending) BENEFITED
                # (+0.0023/+0.0254 -- dead-capital trim). Use the validated trend separator
                # rsi_trend_str (already used by _w_time/_chop_amp/_accel_floor) to gate:
                # full break-even pressure in trends (rsi_trend_str high), near-zero in chop
                # (rsi_trend_str ~0 -> sideways byte-identical). Continuous tanh ramp on
                # rsi_trend_str/0.20 (saturates by ~0.4 trend strength). This is the SAME
                # separator pattern as _w_time (which is heavier-in-chop for time-pressure,
                # here inverted: break-even fires in trend not chop).
                _be_trend_gate = max(0.0, min(1.0, np.tanh(rsi_trend_str / 0.20)))
                # Exp1 (architectural, indep, this session): MAE-CONDITIONED break-even
                # pressure for CHOP regimes. The trend gate above floors _be_pressure at ~0
                # in sideways (rsi_trend_str~0 -> gate~0) because the ungated version cratered
                # sideways -0.731 (killed mean-reversion recoveries: positions legitimately
                # hover near breakeven then RECOVER in chop). But that throws away ALL
                # break-even protection in sideways, where positions stagnate near BE and
                # bleed to the stop. NEW cross-component data dep: _be_pressure now reads
                # the position's MAE (self._mae, the minimum pos_pnl over the hold) jointly
                # with pos_pnl -- a property of the position's LIFE-CYCLE, not a regime label.
                # Structural separator between the two stagnation cases the trend gate
                # conflates: (a) SHALLOW-MAE stagnation (pos never went significantly under-
                # water, |mae| << stop) = a fresh position finding its level = the mean-
                # reversion recovery the trend gate protects -> keep _be_pressure OFF; (b)
                # DEEP-MAE stagnation (pos went underwater to >= ~0.4*stop then recovered
                # to BE) = a position that already demonstrated it can go underwater and is
                # now stalling at BE -> structurally fragile, likely to re-test the MAE and
                # bleed to the stop -> let _be_pressure fire. The MAE is a position-level
                # structural property (how far underwater it has been), NOT a market-regime
                # classifier -- it generalizes to any chop where a position has a deep-MAE
                # history. Continuous tanh on (-mae)/(|stop|*0.4) (no boundary; fast-saturating
                # so a position with mae<=-0.4*stop gets full activation, shallow-MAE gets ~0).
                # Composes with _be_trend_gate via MAX: trend regimes keep the baseline trend-
                # gated path (byte-identical there since trend gate already saturates to 1.0);
                # chop regimes get the MAE-conditioned path (fires only for deep-MAE). New
                # control flow: a second activation gate on _be_pressure reading a NEW data
                # source (MAE) the baseline _be_pressure did not read. Targets sideways (the
                # negative-Sharpe regime whose score == bare Sharpe; cutting deep-MAE stalls
                # before they bleed to the stop raises sideways Sharpe directly).
                _be_mae_depth = max(0.0, min(1.0, np.tanh(-self._mae.get(symbol, 0.0) / (abs(STOP_LOSS_PCT) * 0.25))))
                _be_mae_gate = max(_be_trend_gate, _be_mae_depth)
                # step12: split hold-gate by path (trend 4-bar ramp, mae 2-bar faster ramp)
                _be_pressure = 0.45 * _be_near_zero * max(_be_hold_gate_trend * _be_trend_gate, _be_hold_gate_mae * _be_mae_depth)
                _w_be = 1.0  # profit-sign-neutral: fires on stuck winners AND losers alike
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
                    _w_vc * _vc_pressure,
                    _w_be * _be_pressure,
                )
                _soft_max = max(_soft_terms)
                # STRUCTURAL_EXPLORATION: subsystem rewrite of the soft-pressure FUSION
                # mechanism. Old: pure element-wise MAX (single dominant term; all other
                # terms DISCARDED regardless of magnitude). This is the documented MAX-
                # fusion absorption wall -- Exp1 (loss-velocity), Exp2 (cross-symbol freq),
                # prior sessions' _w_slope DD-amp ALL byte-identical inert because a second
                # moderately-firing soft term (e.g. loss-velocity 0.4) is fully absorbed by
                # a slightly-larger dominant term (slope-against 0.5): MAX gives 0.5, the
                # 0.4 contributes NOTHING. New core mechanism: CONFIRMATION-AMPLIFIED MAX --
                # _soft_max stays the MAX (preserves single-source noise rejection), but a
                # bounded fraction of the 2nd-highest term is ADDED when multi-source
                # agreement is high: soft_combined = max + _agree_amp * 2nd, where _agree_amp
                # scales with the 2nd/1st ratio (0 at single-source, up to AGREE_MAX at full
                # agreement). Two agreeing mid-magnitude sources (0.5 + 0.45) now yield 0.5 +
                # 0.34 = 0.84 (cuts the loser faster), while a single 0.9 spike yields 0.9 +
                # 0 = 0.9 (no change, noise rejected). Capped at 1.0 to preserve the binary-
                # exit pressure ceiling. The stop-loss floor (max(sl, soft)) is preserved.
                # Distinct from the OLD weighted-sum (removed line ~3076): that ADDED all 6
                # terms unconditionally (correlated noise summed); this adds ONLY the 2nd
                # term, gated on agreement ratio (confirmation-filtered). Net effect: multi-
                # source confirmation AMPLIFIES exit pressure above the single-term MAX,
                # breaking the absorption wall for confirmed multi-source exit signals.
                # WHY OLD MECHANISM AT CEILING: 3 architectural experiments this session
                # (Exp1 loss-velocity pressure, Exp3 MAE-deepening threshold) hit the wall
                # that the MAX-fusion absorbs any new loss-side signal; threshold-based
                # variants (Exp3) collapsed crash because winning shorts share the signal.
                # The fusion MECHANISM itself is the constraint -- rewriting it is the only
                # remaining untested axis on the exit side.
                _sorted_terms = sorted(_soft_terms, reverse=True)
                _ratio_2nd = _sorted_terms[1] / max(_sorted_terms[0], 1e-6) if _sorted_terms[0] > 1e-6 else 0.0
                # Confirmation strength: 0 at single-source, 1 at full agreement
                _agree_gate = max(0.0, min(1.0, np.tanh(_ratio_2nd / 0.30)))
                # Confirmation-amplified MAX: add a bounded fraction of the 2nd term.
                # AGREE_MAX caps the added fraction (0.50 = up to 50% of 2nd term added).
                SOFT_FUSION_AGREE_MAX = 0.50
                # branch step5: VOL-REGIME gate (exclude sideways). steps 1-4 ALL
                # collapsed sideways because sideways is a MEAN-REVERSION regime
                # fundamentally sensitive to exit-pressure amplification in BOTH
                # directions (profit-side amplification over-harvests mean-reverters
                # before their natural peak; loss-side amplification cuts dips that
                # recover). The clean separator between sideways and the trending
                # regimes is VOL_RATIO: sideways 2023 has moderate vol (vol_ratio
                # ~0.8-1.1); crash/bull have high vol (1.2-1.3+). Gate amplification on
                # vol_ratio > 1.15 so it fires ONLY in HIGH-vol regimes (crash/bull)
                # and is byte-identical in moderate-vol chop (sideways) + modest-vol
                # rally. step5 onset 1.0 fixed crash (-0.145, byte-identical) but sideways
                # still collapsed (-1.25) -> sideways vol_ratio passes 1.0 too often.
                # Raise onset to 1.15 to exclude sideways. AGREE_MAX 0.50.
                # branch step6b DIAGNOSTIC FOUND: the min(1.0,...) CLAMP on _soft_max
                # was changing behavior even when _agree_amp=0 -- the original code did
                # NOT clamp _soft_max (weighted terms can exceed 1.0, e.g. weight 1.25 *
                # pressure 0.9 = 1.125), so min(1.0, _soft_max+...) capped it = behavior
                # change. REMOVE the clamp; cap via _agree_amp magnitude (AGREE_MAX 0.50
                # bounds the added term to 0.50*1.0 = 0.50, so _soft_max max ~1.6, which
                # the exit-threshold logic already tolerates since _exit_pressure > 1.0
                # triggers exit anyway).
                _fusion_vol_gate = max(0.0, min(1.0, np.tanh((vol_ratio - 1.15) / 0.15)))
                # Exp2 (this session): TREND-ALIGNED-WINNER gate on the confirmation
                # amplification -- the structural decoupling of the 654588e9 crash-gain/
                # bull-loss coupling. Exp1 (this session, inverse giveback gate) PROVED the
                # giveback-magnitude axis at the 0.10 boundary does NOT separate crash-gain
                # from bull-loss: the inverse gate (ON at <10pct) collapsed crash IDENTICALLY
                # to the forward gate (Exp5 prior session) and the loss-side gate (Exp1 prior)
                # -- all converge to pre-baseline crash -0.143. The two regimes' winning
                # shorts overlap in giveback space. The correct separator is NOT giveback
                # magnitude but the SAME validated trend-aligned-winner property already used
                # to gate pp_pressure ATTENUATION (line ~2689): _ta_winner_gate is 1.0 ONLY
                # for longs in a PERSISTENT uptrend (down_persist<0.40) on a GRADUAL pullback
                # (giveback<0.15 AND slope_against<0.0004) that are past scale-in and in profit
                # -- the canonical "let winners ride" bull-long scenario. It is 0.0 for shorts
                # (long-only gate), for transient bounce uptrends in crashes (up_persist gate),
                # and for sharp reversals (slope/giveback gates). So (1 - _ta_winner_gate)
                # is 1.0 for crash trend-aligned SHORTS (amp full -> bounce-onset realization
                # PRESERVED, the +0.083 crash gain) and 0.0 for bull longs on gradual pullbacks
                # in persistent uptrends (amp MUTED -> the -0.0006 bull loss RECUPERATED).
                # This is a position-level structural property (direction x trend-persistence
                # x pullback-sharpness x hold-duration x profit), NOT a regime label; all
                # constituents are smooth/continuous and already proven stability-robust (the
                # pp_pressure path uses the identical gate with stability passing). NEW cross-
                # component data dep: the fusion amp now reads the trend-aligned-winner gate
                # from the pp_pressure subsystem (a joint position-life-cycle + trend-persistence
                # signal). The vol-gate still confines amp to high-vol regimes, so sideways/
                # rally/mixed remain byte-identical (amp=0 there regardless of this gate).
                _agree_ta_winner_gate = 1.0 - _ta_winner_gate
                _agree_amp = SOFT_FUSION_AGREE_MAX * _agree_gate * _fusion_vol_gate * _agree_ta_winner_gate
                _soft_max = _soft_max + _agree_amp * _sorted_terms[1]  # no clamp (original didn't)
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
                # Branch step 2: chop-only gating. In trends (high abs(ret_long)),
                # single-source pressure signals are real (slope reversal alone =
                # genuine trend break). Mute the attenuator's effect by trend strength
                # so it operates only in chop where single-source spikes ARE noise.
                _chop_atten_w = 1.0 - max(0.0, np.tanh(abs(ret_long) / 0.04))  # 1 in chop, 0 in trend
                # Attenuator: scaled by chop weight — in chop 0.75x at single, 1.0x at agree;
                # in trend approaches 1.0x always (no attenuation)
                _soft_atten = 1.0 - 0.25 * (1.0 - _agree_gate) * _chop_atten_w
                _soft_max = _soft_max * _soft_atten
                # Architectural simplification (this session, branch step3): REMOVE ONLY
                # the exit-pressure EMA (on _soft_max), KEEP the voter_bias EMA (on the
                # additive _voter_bias term). Step1 (remove both): rally +0.003 (exit-
                # pressure EMA redundant given terminal level-EMA alpha0.99; it added
                # double-smoothing lag) BUT bull -0.016 (voter_bias EMA load-bearing for
                # bull ct-short exit timing). Step2 (re-add exit-pressure, keep voter_bias
                # removed): rally gain vanished + bull still regressed = confirmed the two
                # EMAs have OPPOSITE value. exit-pressure EMA = net-negative (redundant,
                # removing helps rally); voter_bias EMA = net-positive (load-bearing for
                # bull). So the keep combination is: remove exit-pressure EMA (capture
                # rally +0.003), keep voter_bias EMA (hold bull at baseline). Removes one
                # EMA state + branch; all ct-gated -> only rally + bull ct-shorts affected.
                _ct_pos_str = max(0.0, np.tanh(-(1.0 if current_pos > 0 else -1.0) * ret_vlong / 0.04))
                _exit_ema_alpha = 0.5 * _ct_pos_str  # 0 trend-aligned, up to 0.5 counter-trend
                _prev_vb = self._voter_bias_ema.get(symbol, _voter_bias)
                _voter_bias = (1.0 - _exit_ema_alpha) * _voter_bias + _exit_ema_alpha * _prev_vb
                self._voter_bias_ema[symbol] = _voter_bias
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
                # Exp4 (architectural, indep): PORTFOLIO-DD lowering of the exit threshold for
                # LOSERS. NEW cross-component data dep: the de-risk full-exit threshold
                # (_exit_thresh, normally 1.0 = full exit when exit_pressure reaches 1.0) is
                # LOWERED for LOSING positions when the PORTFOLIO is in drawdown. Mechanism:
                # a portfolio DD (correlated regime hit, multiple positions losing across
                # symbols) signals an adverse regime where losers are likely to EXTEND
                # further -> trigger the full exit (de-risk ramp to target=0) at LOWER
                # exit_pressure -> cut losers ~1 bar earlier -> smaller realized losses ->
                # higher Sharpe in the negative-Sharpe regimes (bull PF 0.7 despite 71.8pct
                # WR = large losers; bull DD 13pct worst, crash DD 17.85pct past 5pct knee).
                # DISTINCT from Exp2's w_slope DD-amp (MAX-fusion absorbed, byte-identical):
                # this changes the THRESHOLD that converts pressure to action, NOT the
                # pressure sources -- bypasses the MAX-fusion absorption wall (the dominant
                # _sl_pressure term still saturates, but now full-exit triggers at 0.88 not
                # 1.0). Loss-side only (gated on _pnl_scale<0 -> only fires for losers;
                # trend-aligned winners byte-identical, exit_thresh stays 1.0 for them).
                # Byte-identical at portfolio peak (dd_frac=0 -> _port_dd_atten=1.0 ->
                # lowering 0). Uses the validated top-level _port_dd_atten (asymmetric-EMA,
                # leverage-coupled 0.008*LEVERAGE_K scale). Smooth (no boundary), direction-
                # agnostic general principle (no regime label): a losing position during a
                # portfolio drawdown is at correlated-regime-hit risk -> exit sooner.
                if _pnl_scale < 0.0:
                    # branch step5: REPLACE ct-gate with SUSTAINED-LOSS gate. The ct-gate
                    # (step1) excluded all trend-aligned losers (the main bull/crash problem).
                    # MAE-depth (step4) was inert (stop catches deep losers first). New
                    # signal: a loser that has been NEGATIVE for multiple consecutive bars
                    # (sustained loss, not a fresh dip) is likely extending -> cut earlier.
                    # A fresh dip (1-2 bars negative) might recover (crash short bounces,
                    # bull pullback longs that resume) -> protect. Uses the _pnl_path
                    # (12-bar pos_pnl trajectory); sustained_loss = fraction of last K bars
                    # where pos_pnl < 0. K=4 captures multi-bar bleeding (post-scale-in).
                    # Gate fires for sustained losers (any direction, trend-aligned or ct)
                    # while protecting fresh dips. Byte-identical at portfolio peak (dd_frac=0).
                    _pp = self._pnl_path.get(symbol, [])
                    _sustained_loss = 0.0
                    if len(_pp) >= 4:
                        _recent = _pp[-4:]
                        _sustained_loss = sum(1.0 for _p in _recent if _p < 0.0) / 4.0
                    # branch step7: TREND-STRENGTH gate to spare sideways mean-reverters.
                    # Step5 regressed sideways (-0.009): sideways mean-reverters have frequent
                    # sustained-loss bars but RECOVER (chop oscillates). The extending losers
                    # (crash shorts, mixed wrong-side longs) are in TRENDING regimes. Gate
                    # the sustained-loss signal on trend strength (the validated rsi_trend_str
                    # already used by _w_time/_chop_amp/_accel_floor): full effect in trends
                    # (rsi_trend_str high), near-zero in chop (sideways). Sideways byte-
                    # identical; crash/mixed (trending) keep the gain. Continuous tanh ramp.
                    _sustained_loss_trend_gate = max(0.0, min(1.0, np.tanh(rsi_trend_str / 0.20)))
                    _exit_dd_gate = _sustained_loss * _sustained_loss_trend_gate
                    _exit_thresh = _exit_thresh * (1.0 - 0.12 * (1.0 - _port_dd_atten) * _exit_dd_gate)
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
                    # Exp4 (architectural, indep): MULTI-DAY trend-align factor in _ts_supp
                    # (was 20-bar ret_long*pos_dir). DIAGNOSTIC (this session, verifiable): mixed's
                    # LARGE reachable tp_harvest events (pos/equity>0.05, n=443, all cross the grid
                    # ~42 steps) have CLEAN MAE (mean 0.0) -> the MAE factor and deep-peak factor are
                    # both ~1.0 -> _ts_supp ~= tanh(ret_long*pos_dir/0.04). mixed is a multi-day DOWN
                    # year (ret_vlong<0 for held longs) but local 20-bar bounces (ret_long>0 during
                    # chop recoveries) are common; deep peaks form during these bounces -> ret_long*pos_dir
                    # >0 -> _ts_supp -> 1 -> harvest SUPPRESSED (let-run) -> the oscillating paper PnL
                    # is NOT realized -> mixed rides to +30%, gives back to +24%, re-peaks (the intrinsic
                    # return_vol 5.54pct vs 4.4pct net low-Sharpe drag). The 20-bar trend-align factor
                    # MISSES that mixed's longs are COUNTER-TREND at the multi-day scale. Switch to
                    # ret_vlong*pos_dir (the validated 96-bar multi-day trend, fast-saturating /0.04):
                    # mixed longs in a multi-day downtrend (ret_vlong<0, pos_dir=+1) -> product<0 ->
                    # _ts_supp -> 0 -> NO suppression -> FULL harvest -> convert oscillating paper to
                    # realized at the deep peaks -> cut the re-peak "ride again" churn -> lower MTM
                    # oscillation -> higher mixed Sharpe. Crash shorts (ret_vlong<0, pos_dir=-1 ->
                    # product>0) stay suppressed -> BYTE-IDENTICAL. Bull longs (ret_vlong>0,
                    # pos_dir=+1 -> product>0) stay suppressed -> BYTE-IDENTICAL. Rally longs
                    # (ret_vlong>0, +1) stay suppressed -> BYTE-IDENTICAL. The 20-bar ret_long factor
                    # and 96-bar ret_vlong factor AGREE for trend-aligned regimes (both >0 product);
                    # they DISAGREE only for counter-trend-at-multi-day-but-bouncing-locally positions
                    # = exactly mixed's oscillating longs. New cross-timescale data dep: the tp-harvest
                    # trend-extension suppression now keys on multi-day alignment (the scale that
                    # distinguishes a genuine trend extension from a counter-trend bounce). Continuous
                    # tanh, no new decision boundary. ret_vlong is already computed (96-bar OLS,
                    # noise-robust). Targets mixed (binding); protects all trend-aligned regimes.
                    _ts_supp = (1.0 - max(0.0, min(1.0, np.tanh(-self._mae.get(symbol, 0.0) / abs(STOP_LOSS_PCT) / 0.2)))) * max(0.0, np.tanh(ret_vlong * (1.0 if current_pos > 0 else -1.0) / 0.04)) * max(0.0, min(1.0, np.tanh((_tp_ratio - 2.8) / 0.5)))
                    # Exp1 (architectural): portfolio-DD-adaptive relaxation of the
                    # trend-extension harvest suppression. _ts_supp normally PREVENTS
                    # harvesting clean trend-aligned deep-peak winners (let them run).
                    # During portfolio DD (rally pullbacks = the DD source), weaken the
                    # suppression so even clean trend winners get partially harvested ->
                    # lock realized gains at the peak -> the remaining position gives back
                    # less -> caps the DD from riding winners through deep pullbacks. A
                    # SECOND, distinct DD-reduction lever on a different exit path (tp
                    # size scale-down) from the maxed giveback-tolerance tightening.
                    # Byte-identical at portfolio peak (dd_frac=0 -> factor 1.0). Same
                    # leverage-coupled DD-fraction scale as giveback tightening.
                    _dd_tp_relax = 1.0 - PORT_DD_TP_HARVEST_RELAX * max(0.0, np.tanh(_port_dd_frac / (PORT_DD_TP_HARVEST_SCALE * LEVERAGE_K)))
                    _ts_supp = _ts_supp * _dd_tp_relax
                    # Exp5 (architectural, indep): raise tp_harvest base magnitude 0.30 -> 0.45.
                    # Prior session walled magnitude raise at 0.50 (crash stability collapsed
                    # 1.0->0.225): crash's clean trend shorts got over-harvested because _ts_supp's
                    # trend-align factor used 20-bar ret_long, which during crash recovery bounces
                    # (ret_long>0 for shorts = product<0 -> factor 0 -> _ts_supp 0 -> NO suppression
                    # -> full harvest -> over-harvested crash recovery winners). Exp4 KEEP fixed the
                    # root cause: _ts_supp now uses multi-day ret_vlong*pos_dir. Crash shorts in a
                    # multi-day downtrend (ret_vlong<0, pos_dir=-1 -> product>0) -> _ts_supp HIGH ->
                    # harvest SUPPRESSED -> crash protected at the magnitude raise. With crash now
                    # correctly shielded by the multi-day factor, the magnitude ceiling may lift.
                    # mixed's deep peaks (tp_ratio 16.47, MAE clean, already _ts_supp~0 = fully
                    # unsuppressed) get a DEEPER harvest per fire (0.45 vs 0.30 = +50pct) -> more
                    # paper PnL converted to realized per re-peak -> smaller remaining position ->
                    # less giveback on the next oscillation -> lower MTM oscillation -> higher mixed
                    # Sharpe (the binding floor 0.408, byte-identical across all prior tp_harvest
                    # magnitude experiments which crash-wall blocked testing above 0.50). Continuous
                    # (the 0.30 was a fixed scalar, not a gate; changing it scales the existing smooth
                    # tanh activation uniformly). New data dep: none (parameter change riding the Exp4
                    # structural fix that unblocked the crash wall). Targets mixed; crash protected by
                    # the multi-day _ts_supp.
                    _tp_scale = 0.45 * max(0.0, min(1.0, np.tanh((_tp_ratio - 1.6) / 0.6))) * _tp_trend_gate * max(0.0, 1.0 - 1.5 * _ts_supp)
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
                    # Exp3 (architectural, indep): PORTFOLIO-DD-ADAPTIVE de-risk floor
                    # lowering for LOSERS. NEW cross-component data dep on the de-risk
                    # graduation ramp: _de_floor (the lower bound of the graduated partial-
                    # exit ramp) currently depends on PnL sign only (0.55 profit / 0.85 loss).
                    # The Exp4 keep lowers the UPPER bound _exit_thresh for losers during
                    # portfolio DD (full-exit triggers sooner). This is the COMPLEMENTARY
                    # lever on the LOWER bound: during portfolio DD (correlated regime hit),
                    # a LOSING position is at risk of EXTENDING further -> start the graduated
                    # partial-exit ramp EARLIER so the position begins shrinking before
                    # full-exit pressure is reached -> smaller realized losses -> higher
                    # Sharpe in the negative-Sharpe regimes (bull PF 0.7 / crash PF 0.9
                    # despite 67-73pct WR = losers too big; bull DD 12.88pct worst, crash
                    # 17.76pct). DISTINCT from Exp4: that changes the full-exit threshold
                    # (upper bound), this changes the graduation ramp START (lower bound).
                    # Both compose: earlier ramp start (this) + sooner full-exit (Exp4) =
                    # a loser during portfolio DD de-risks gradually from a lower pressure
                    # AND hits full-exit at a lower pressure. Loss-side only (gated on
                    # _pnl_scale<0 -> only fires for losers; trend-aligned winners byte-
                    # identical, _de_floor stays at profit-side 0.55 minus trend relax).
                    # Byte-identical at portfolio peak (dd_frac=0 -> _port_dd_atten=1.0 ->
                    # lowering 0). Uses the validated top-level _port_dd_atten (asymmetric-EMA,
                    # leverage-coupled 0.008*LEVERAGE_K scale). Smooth (no boundary), direction-
                    # agnostic general principle (no regime label): a losing position during a
                    # portfolio drawdown is at correlated-regime-hit risk -> start trimming sooner.
                    # Max lowering 0.10 (0.85 -> 0.75 floor for losers at deep DD) -- smaller
                    # than the 0.30 profit/loss swing to keep the ramp from firing too early
                    # on transient mid-range pressure (which would over-trade fees).
                    # Branch step1: TREND-STRENGTH gate to spare sideways mean-reverters.
                    # Exp3 opener regressed sideways -0.025: sideways mean-reverters have
                    # frequent loser bars during portfolio DD but RECOVER (chop oscillates) ->
                    # trimming them during DD realizes losses that mean-revert away. The
                    # validated separator (Exp4 keep step7, same wall): rsi_trend_str. Full
                    # effect in trends (crash/bull/rally losers are in trending regimes where
                    # a sustained loser EXTENDS), near-zero in chop (sideways mean-reverters
                    # recover). Continuous tanh ramp on rsi_trend_str/0.20 (saturates by ~0.4
                    # trend strength -- the SAME scale as _be_trend_gate / _exit_dd_gate).
                    # Sideways byte-identical; crash/rally/mixed keep gains.
                    # Branch step2: ADD SUSTAINED-LOSS gate (full Exp4 keep gate). Step1 used
                    # trend_gate only -> sideways still regressed (-0.243 vs opener -0.244).
                    # Diagnosis: the trend_gate alone spares sideways CHOP, but sideways DD
                    # episodes often coincide with brief TRENDING stretches within the chop
                    # (a multi-day directional leg in 2023) where rsi_trend_str spikes ->
                    # trend_gate fires -> fresh dips get trimmed. The Exp4 keep's full gate
                    # adds sustained_loss (fraction of last 4 pos_pnl bars negative): a FRESH
                    # dip (1-2 bars negative, likely to recover) is spared; only a SUSTAINED
                    # loser (3-4 bars negative = extending, not mean-reverting) triggers the
                    # floor lowering. This is the validated fresh-dip-vs-extending-loser
                    # separator. Use _exit_dd_gate (already computed above for Exp4) instead
                    # of re-computing -- SAME gate, applied to BOTH the upper bound (Exp4
                    # _exit_thresh) and lower bound (this _de_floor) of the exit graduation.
                    # Sideways fresh dips spared; crash/bull/rally extending losers trimmed.
                    if _pnl_scale < 0.0:
                        # Branch step12: COMBINE step5 baseline (0.13 continuous, captures the
                        # sideways small-DD gain) with a deep-DD BOOST (additional 0.05 that
                        # only fires at deep portfolio DD, captures crash's extended gain).
                        # step5 alone: +0.002270 (sideways +0.0024, crash +0.0075). step11
                        # (deep-DD onset only, mag 0.18): +0.002279 (sideways lost, crash
                        # extended +0.0019). The two gains are at DIFFERENT DD magnitudes:
                        # sideways gain needs small-DD firing (its 4.47pct DD), crash gain
                        # extends with deep-DD firing (its 17.7pct DD). Combining: total
                        # magnitude = 0.13*(1-_port_dd_atten) [continuous, small at small DD]
                        # + 0.05*deep_DD_ramp [additional, only at deep DD]. At sideways small
                        # DD: 0.13*small + 0 = small (sideways gain preserved). At crash deep
                        # DD: 0.13*large + 0.05*~1 = 0.18 (crash extended gain). Continuous
                        # tanh ramp on (1-_port_dd_atten - 0.30)/0.10 for the boost (no
                        # boundary). Byte-identical at portfolio peak (both terms 0).
                        _port_dd_active = max(0.0, np.tanh((1.0 - _port_dd_atten - 0.30) / 0.10))
                        _de_floor -= (0.13 * (1.0 - _port_dd_atten) + 0.07 * _port_dd_active) * _exit_dd_gate
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
                        _dr_x = (_exit_pressure - _de_floor * _exit_thresh) / ((1.0 - _de_floor) * _exit_thresh)
                        _dr_x = max(0.0, min(1.0, _dr_x))
                        # Architectural (Exp2 this session): CONVEX de-risk ramp on the
                        # profit side. The prior LINEAR ramp (_de_risk = 1 - x) de-risks
                        # proportionally to exit pressure across the whole [0,1] band, so
                        # mid-range giveback/slope-against NOISE translates 1:1 into
                        # position-value wobble -> equity-curve tracking error (the
                        # stability penalty's root currency). A CONVEX ramp
                        # (_de_risk = 1 - x^k, k>1) holds near full size through moderate
                        # pressure (absorbing transient mid-range noise without shrinking)
                        # then de-risks sharply as pressure approaches saturation — the
                        # decisive high-pressure cut is preserved (same _de_risk at x=1),
                        # only the mid-range RESPONSE is damped. k scales with profit
                        # (winners get the convex cushion, letting trend-aligned winners
                        # ride pullback noise; losers keep the near-linear fast cut via the
                        # 0.85 loss floor). Continuous (smooth x^k, no new boundary),
                        # direction-agnostic, PnL-modulated via _pnl_scale. New control
                        # flow: exit-decision function shape changes from linear to
                        # profit-convex.
                        # Branch step3: TREND-ALIGNMENT gate on the convex cushion (replaces
                        # step2's failed R^2 gate, which killed bull -0.077). Step1 (uniform
                        # convex in profit) eliminated bull's stability penalty (+0.064) BUT
                        # regressed rally -0.025: the cushion holds rally's COUNTER-TREND
                        # shorts (the losing rally trades) longer through giveback. Gate the
                        # cushion by trend-ALIGNMENT (pos_dir matches ret_long sign): only
                        # trend-aligned winners (bull longs in uptrend, crash shorts in
                        # downtrend, rally longs in uptrend) earn the convex cushion;
                        # counter-trend positions revert to linear fast cut. Preserves bull
                        # (trend-aligned) AND rally's trend longs while cutting rally's ct
                        # shorts fast. General principle (no regime label): the cushion is
                        # earned by trading WITH the long-window trend, not by path shape.
                        # Continuous tanh on (ret_long * pos_dir / 0.04).
                        _dr_pos_dir = 1.0 if current_pos > 0 else -1.0
                        _dr_align = max(0.0, np.tanh(ret_long * _dr_pos_dir / 0.04))  # 0 ct, 1 trend-aligned
                        # Exp4 (architectural, indep): SLOPE-CONFIRMATION gate on the de-risk
                        # convex cushion. The cushion (k>1 -> hold near full size through
                        # moderate giveback, the validated stability lever) was gated only on
                        # trend-ALIGNMENT (ret_long*pos_dir) + profit. The win-accelerator
                        # (line ~1488) is ALREADY gated by _slope_conf (16-bar OLS slope
                        # confirming the position) -- a prior session added it to protect bull
                        # (slope weakens before corrections). Extend the SAME slope-confirmation
                        # to the de-risk cushion: the convex cushion (ride giveback) now requires
                        # the near-term slope to STILL CONFIRM the position; when slope weakens
                        # (trend faltering, pullback deepening), _slope_conf -> 0 -> k -> 1 ->
                        # LINEAR fast cut (exit through giveback faster instead of riding it).
                        # Mechanism: a trend-aligned winner whose slope STILL confirms is a
                        # genuine ongoing trend -> ride the small giveback (cushion); a winner
                        # whose slope has weakened is facing a real near-term reversal -> cut
                        # fast (linear). Consistent with the Exp2 lesson this session (near-term
                        # slope/ret_long is the CORRECT exit signal for rally -- cutting rally
                        # longs when the near-term trend turns protects giveback + stability;
                        # the multi-day ret_vlong was too slow and catastrophic). _slope_conf is
                        # computed unconditionally for held positions (line ~1488, before the
                        # scale-in if); reusing it here adds no new price-derived computation
                        # (just a new control-flow dependency at the de-risk decision). Smooth
                        # tanh (no boundary); direction-agnostic general principle (no regime
                        # label): the giveback-riding cushion is earned by an ONGOING confirmed
                        # slope, not just by long-window trend-alignment. New cross-component
                        # data dep at the de-risk ramp.
                        # Exp5 (architectural, indep): SEPARATE SMOOTHER slope-confirmation for
                        # the de-risk cushion, replacing the shared _slope_conf (single 16-bar
                        # _lr_slope). Exp4 (keep ce66fec6) validated slope-conf on the de-risk
                        # cushion (bull +0.001325, sideways +0.000647) BUT rally regressed
                        # -0.000251: the single 16-bar slope is sensitive to MOMENTARY 1-bar
                        # dips during rally pullbacks -> those dip bars got cut slightly faster
                        # -> missed a bit of trend capture. Use the MULTI-WINDOW _exit_slope
                        # (mean of 12/16/22-bar OLS slopes, already computed at line ~1602 for
                        # the exit subsystem) instead: a momentary dip in one window is averaged
                        # with the other two -> only SUSTAINED slope weakening triggers the
                        # linear cut. Smoother slope -> fewer false momentary-dip cuts on rally's
                        # grinding uptrend (slope persistently confirms across windows) -> recover
                        # the rally regression while keeping the bull/sideways giveback-cut gains
                        # (bull 2021 corrections weaken slope across ALL windows -> still cuts).
                        # Isolates the de-risk ramp from the win-accelerator's calibration (the
                        # accelerator keeps the validated single-16-bar _slope_conf at line 1488;
                        # prior sessions tuned it there -> do NOT change the shared signal). New
                        # separate computation (3-window mean already available -> no new price-
                        # derived reads, just a new gate source at the de-risk decision). Same
                        # /0.0004 scale (comparable magnitude). Smooth tanh, direction-agnostic.
                        _dr_slope_conf = max(0.0, np.tanh(_exit_slope * _dr_pos_dir / 0.0004))
                        _dr_k = 1.0 + DERISK_CONVEX_AMP * max(0.0, _pnl_scale) * _dr_align * _dr_slope_conf  # 1.0 loss/ct/slope-weak, up to ~1.6 trend-aligned+profit+smoother-slope-conf
                        _de_risk = 1.0 - _dr_x ** _dr_k
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
                    # branch step9: VOL-GATED vlong-confirmation boost to opp-gate trend-align
                    # (NEW exit path, additive with step4's opp-bias). The opp-gate graduated
                    # exit (this block) protects trend-aligned+in-profit WINNERS from full
                    # exit on opp-voter spikes, using 20-bar ret_long. For mixed's trend-
                    # aligned rally-phase longs, the 96-bar ret_vlong is more STABLE (doesn't
                    # flicker off during pullbacks) -> sustaining the graduated protection
                    # longer -> ride mixed winners longer (same ride-winners direction as
                    # step4, on the opp-GATE path instead of the opp-BIAS path -> additive).
                    # Same vol-gate (low vol_ratio = mixed/rally grind, sparing bull high-vol)
                    # + same ret_vlong confirmation term. Capped via min(1.0). Byte-identical
                    # when vol-gate=0 (bull) or ret_vlong term=0 (ct positions: full exit fast,
                    # the desired behavior for rally pullback shorts + mixed wrong-side longs).
                    _ret_vlong_term_og = max(0.0, np.tanh(ret_vlong * _pos_dir_og / 0.04))
                    _trend_align_og = min(1.0, _trend_align_og + 0.30 * _vlong_vol_gate * _ret_vlong_term_og)
                    _profit_gate_og = max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # [0, ~1] only profit
                    _grad_gate = _trend_align_og * _profit_gate_og  # both required
                    # Exp2 (architectural, indep): 24-bar OWN-DVP-CONFIRMATION on the
                    # opp-gate exit-FRACTION FLOOR (the uncapped lever). Prior-session
                    # own-DVP branch: 12-bar DVP lowering this floor moved mixed +0.005338
                    # REAL (step1); 24-bar ungated amplified mixed +0.005336 EXTRA but
                    # crashed sideways stability 1.0->0.44 (lag-induced exit-timing
                    # divergence on sideways chop). Exp1 (this session, discarded)
                    # CONFIRMED the structural fix: gating the 24-bar DVP on multi-day
                    # trend-alignment (ret_vlong*pos_dir>0 via _ret_vlong_term_og) SPARES
                    # sideways (ret_vlong~0 -> term~0 -> boost 0 -> byte-identical) -- but
                    # Exp1 routed through _trend_align_og which is capped at 1.0 (absorbed).
                    # HERE the DVP lowers the FLOOR of _opp_exit_frac_grad (uncapped: 0.4
                    # can drop toward 0.25), the SAME lever the 12-bar version moved mixed
                    # on, so the 24-bar amplification has REAL headroom. Gated on the
                    # validated f7af0069 envelope: vol-gate (low-vol grind, spares bull) x
                    # ret_vlong trend-align (spares sideways + ct). Lower exit-frac floor =
                    # less trimming of trend-aligned in-profit winners when own-volume still
                    # confirms -> ride mixed rally-phase longs longer -> higher mixed Sharpe
                    # (the validated ride-winners direction). Byte-identical when vol-gate=0
                    # (bull), ret_vlong term=0 (sideways + ct), or DVP=0. 24-bar own-DVP
                    # computed fresh (sum vol[i]*sign(close[i]-close[i-1])/sum vol[i] over
                    # 24 bars); deep-saturated /0.15 (near-constant where it fires, noise-free).
                    _dvp24_n = 24
                    _dvp24_c = closes[-_dvp24_n - 1:]
                    _dvp24_v = bd.history["volume"].values[-_dvp24_n:]
                    _dvp24_rets = np.sign(np.diff(_dvp24_c))
                    _dvp24 = float(np.sum(_dvp24_v * _dvp24_rets) / max(np.sum(_dvp24_v), 1e-10))
                    _dvp24_conv = max(0.0, np.tanh(_pos_dir_og * _dvp24 / 0.15))
                    # branch step2: add a MULTI-DAY TREND-MAGNITUDE floor to isolate mixed's
                    # rally-phase longs (ret_vlong solidly positive) from sideways noise
                    # (ret_vlong~0). _ret_vlong_term_og=tanh(ret_vlong*pos_dir/0.04) is ~0.5
                    # at ret_vlong=0.02 (sideways noise band) -- enough for the 24-bar DVP lag
                    # to destabilize sideways exit timing (Exp2 stab 1.0->0.44). mixed's
                    # beneficial rally-phase longs sit at ret_vlong~0.04-0.08 (solid uptrend
                    # leg in the oscillating year) -> a magnitude floor at ~0.03 lets mixed
                    # through at full strength while fading sideways to ~0. Continuous tanh
                    # on |ret_vlong*pos_dir|/0.03 (no boundary); fast-saturating so mixed's
                    # solid legs saturate to 1.0 (near-constant, noise-free) while sideways
                    # noise stays in the fade region -> DVP floor-lowering ~0 for sideways.
                    _dvp24_mag_gate = max(0.0, min(1.0, np.tanh((abs(ret_vlong * _pos_dir_og) - 0.03) / 0.02)))
                    _dvp24_floor_lower = 0.15 * _vlong_vol_gate * _ret_vlong_term_og * _dvp24_conv * _dvp24_mag_gate
                    _opp_exit_frac_grad = 0.4 + 0.6 * max(0.0, min(1.0, np.tanh(_opp_margin / 0.30))) - _dvp24_floor_lower
                    # Blend: full exit (1.0) by default, graduated only when both gates hold.
                    _opp_exit_frac = 1.0 + (_opp_exit_frac_grad - 1.0) * _grad_gate
                    target = current_pos * (1.0 - _opp_exit_frac)

            # Exp1 (this session): counter-trend-DIRECTION-gated temporal EMA on the
            # EMITTED position target (the final held-position LEVEL) — a NEW smoothing
            # POINT distinct from the prior exit-pressure EMA (734 keep) and voter_bias
            # EMA (cfc48165 keep), both of which smooth UPSTREAM signals. The held-
            # position target is the confluence of the de-risk ramp + opp-gate partial
            # exit + scale-in + tp-harvest — each contributes position-value variation
            # the upstream pressure/bias EMAs never observe. Stability's tracking error
            # is literally std(clean_ret - pert_ret) of the EQUITY curve, i.e. driven by
            # held-position-value differences between clean and perturbed runs; low-
            # passing the emitted LEVEL for the noise-sensitive counter-trend (rally
            # pullback-short) positions damps that variance at its terminal point,
            # catching ALL upstream resize sources at once. Signed ct gate
            # (-pos_dir*ret_vlong) -> alpha=0 for trend-aligned bull longs / crash
            # shorts -> byte-identical by construction; low-ret_vlong sideways spared.
            # Resizes only: full exits (target==0) and sign flips are risk transitions
            # -> never smoothed (must hit exact target). Reset on full exit.
            # BRANCH step1 (this session): STRENGTHEN the emitted-target EMA (Exp1's
            # proven stability lever, +0.028 at alpha=0.5). Two coordinated changes:
            # (1) ct gate linear /0.04 -> FAST-saturating /0.01 (Exp5's validated lesson:
            #     rally's solidly-positive ret_vlong then sits in the flat tail -> gate is
            #     a near-CONSTANT, so the smoothing strength does not track noise);
            # (2) alpha cap 0.5 -> 0.70 (stronger low-pass on the held ct-position LEVEL).
            # Stronger near-constant smoothing of rally's counter-trend held position
            # value should push rally stability above the 0.708 baseline toward the 0.80
            # knee. Trend-aligned (bull/crash) spared by construction (gate 0 -> alpha 0
            # -> byte-identical); low-ret_vlong sideways spared. Risk: more lag may slow
            # ct-loser exits -> rally raw cost (tension with Exp5's faster-exit raw gain);
            # branch iterates to balance.
            if current_pos != 0 and target != 0 and (current_pos > 0) == (target > 0):
                _pos_dir_te = 1.0 if current_pos > 0 else -1.0
                _ct_te_str = max(0.0, np.tanh(-_pos_dir_te * ret_vlong / 0.01))
                _te_alpha = 0.99 * _ct_te_str  # branch step5: alpha cap 0.97->0.99 (confirm peak)
                # Profit-graduated smoothing (architectural, new data dep on pos_pnl
                # sign). The _target_ema was added when stability was the binding wall
                # (k=0.5); its strong alpha lifts rally stability above the 0.80 knee
                # BUT costs rally raw -- the lag holds counter-trend LOSERS (rally's
                # pullback shorts, the documented losing-trade drag) bigger longer ->
                # larger realized losses -> lower Sharpe. Under k=0.3 the stability
                # benefit is discounted while the raw cost remains, so the trade-off
                # shifted. Weaken the smoothing selectively on LOSING ct positions:
                # losers track the raw (shrinking) target faster -> de-risk/exit
                # sooner -> smaller losses -> rally raw up; WINNING ct holds keep full
                # alpha (preserve the position-value consistency that holds stability
                # above the knee). Smooth tanh on pos_pnl/|stop| (no decision boundary
                # -- profit-continuous); loss-gate ramp 0 profit -> ~1 deep loss, cuts
                # alpha up to 50%. Trend-aligned (gate 0 -> alpha 0) byte-identical.
                _te_loss_gate = max(0.0, -np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # 0 profit, ~1 loss
                _te_alpha = _te_alpha * (1.0 - 0.50 * _te_loss_gate)
                # Exp5 (architectural, indep): CHURN-CONCENTRATED target-EMA alpha. The
                # _target_ema is the proven rally-stab lever (smoothing the counter-trend
                # held-position LEVEL dampens the noise-driven position-value cascade that is
                # rally's tracking-error root). It currently smooths ALL ct positions uniformly
                # (alpha = 0.99 * ct_str * (1 - 0.5*loss_gate)). But rally's binding stability
                # population is the HIGH-CHURN ct re-entries (burst bars, len(_eh)>=3): these
                # are the positions whose scale-in resizes cascade most under AR(1) noise (each
                # re-entry's first-bar + scale-in produces a fresh position-value wobble). A
                # low-churn ct position (a single isolated ct hold) has fewer resize events ->
                # less smoothing needed. Concentrate the smoothing on the HIGH-churn ct
                # population: boost alpha UP (toward the 0.99 cap) when churn is high, keep
                # baseline alpha when churn is low. This should push rally stability past the
                # 0.80 knee more efficiently than uniform smoothing (more smoothing exactly
                # where the cascade is worst) while SPARING the low-churn ct positions (less
                # over-smoothing lag -> less rally raw cost from holding ct losers bigger
                # longer). New cross-component data dep: target-EMA alpha depends on the
                # position's own recent entry density (was ct-str + loss-gate only). Uses the
                # noise-IMMUNE integer churn count len(_eh) (the validated rally-stab gate
                # family: grid/deadband/churn_size_atten all use len(_eh) -- noise-free, no
                # boundary that flips under AR(1)). Continuous tanh on (len(_eh)-1.5)/0.6 (same
                # fast-saturating scale as _churn_dz); max +0.20 alpha boost at deep churn,
                # capped at the 0.99 hard ceiling. Trend-aligned (ct_str 0 -> alpha 0 ->
                # boost multiplies 0) byte-identical; low-churn (len<=1 -> churn boost ~0)
                # byte-identical-ish.
                _te_churn_boost = max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))  # ~0 low churn, ~1 bursting
                _te_alpha = min(0.99, _te_alpha * (1.0 + 0.60 * _te_churn_boost))  # branch step9: 0.40->0.60 (pre-EMA median makes input spike-free -> EMA can smooth harder)
                # Exp1 (architectural, indep, this session): TREND-ALIGNED-WINNER target-EMA.
                # The keep 47cbe827 extended max_hold for long-only trend-aligned in-profit
                # positions (the _ta_dd_hold_ext path). That hold-extension added bar-to-bar
                # position-value tracking error under AR(1) noise -> bull stability dropped
                # 0.803->0.787 (below the 0.80 knee -> stab_factor 0.958 -> bull final score
                # discounted despite positive raw). The prior session-summary explicitly flagged
                # this as the untested lead: "smooth the max_hold extension to recover bull
                # stability (the prior session _target_ema on ct positions is the validated
                # stability lever pattern)."
                #
                # This applies the SAME _target_ema primitive (the proven rally-stab lever) to
                # the MIRROR population: trend-aligned WINNERS (the keep's exact affected
                # population). The ct-side _te_alpha smooths COUNTER-TREND positions (gate
                # _ct_te_str = tanh(-pos_dir*ret_vlong/0.01)); this smooths TREND-ALIGNED
                # positions (gate _ta_te_str = tanh(+pos_dir*ret_vlong/0.01)). The two gates
                # are mutually exclusive by sign (one is ~0 when the other is ~1) -> the two
                # smoothing paths fire on disjoint populations and compose without interference.
                #
                # WINNER-ONLY (profit-graduated, the OPPOSITE of the ct-side loss-gate): bull's
                # deficit is on WINNING longs (the keep extends winners -> the extended hold
                # wobbles under AR(1) -> tracking error). Smooth ONLY winners (pos_pnl>0) so
                # the EMA holds the winning level stable through pullback noise; losers
                # (pos_pnl<0) track the raw shrinking target fast (no exit lag -- avoids the
                # ct-side raw cost that walled uniform smoothing). This is the validated
                # fresh-winner-vs-loser separator (same tanh(pos_pnl/|stop|) gate as
                # _ta_winner_gate line 2569 and _ta_profit_gate line 2698). Continuous tanh,
                # no decision boundary. Byte-identical for ct positions (_ta_te_str=0),
                # trend-aligned losers (profit gate 0), and shorts not in profit.
                #
                # MODEST alpha (0.30 max, vs the ct-side 0.99): bull's deficit is SMALL (0.787
                # vs 0.80 knee, needs ~+0.013 stability) and trend-aligned winners are the
                # dominant bull/rally population -> a large alpha would lag winner->loser
                # transitions and cost raw (the stab/raw tension that walled the ct-side at
                # high alpha before the loss-gate). The profit-gate already eliminates the
                # loser-lag path; 0.30 is enough to dampen the AR(1) position-value wobble
                # (the knee is close, so a small smoothing nudge should clear it).
                _ta_te_str = max(0.0, np.tanh(_pos_dir_te * ret_vlong / 0.01))  # ~0 ct, ~1 trend-aligned (mirror of _ct_te_str)
                _ta_te_profit = max(0.0, np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # 0 loss, ~1 profit (winner-only)
                # Branch step2: SLOPE-CONFIRMATION gate (the validated pattern that protected
                # bull from the win-accelerator Exp4). Step1 (profit-gate alone) catastrophically
                # regressed bull (-0.281): the smoothing lagged bull's winners on winner->reversal
                # transitions (bull PF 0.7 -- winners small, losers big; holding a winner through
                # its reversal turns a small win into a bigger loss). The profit-gate fires on
                # ANY winner, but bull's winners REVERSE (the trend then reverses), while rally/
                # crash trend-aligned winners PERSIST (the slope keeps confirming). Gate the
                # smoothing OFF when the near-term slope stops confirming the position: use the
                # validated multi-window _exit_slope (mean of 12/16/22-bar OLS, smoother than
                # single 16-bar -> only SUSTAINED slope weakening triggers, not momentary dips;
                # the SAME signal + scale /0.0004 as _dr_slope_conf line 3336). When slope still
                # confirms (ongoing trend), smoothing holds the winner level stable through
                # pullback noise; when slope weakens (reversal impending), smoothing turns OFF
                # -> target tracks raw -> exit fires on the reversal at full speed. Spares bull's
                # reversal-prone winners (slope weakens before 2021 corrections) while keeping
                # smoothing on rally/crash persistent-slope winners. Continuous tanh, no boundary.
                _ta_te_slope_conf = max(0.0, np.tanh(_exit_slope * _pos_dir_te / 0.0004))
                # Branch step3: PORTFOLIO-DD gate. Step2 (slope-conf) barely moved bull:
                # bull's grinding uptrend has slope persistently positive -> the gate keeps
                # smoothing ON for ALL bull winners (not just doomed ones) -> cumulative lag
                # across many winners drags APY -5.6pct (vs +2.59 baseline). The slope-conf
                # gate only removes smoothing on reversal bars, but the lag accumulates on
                # the many persistent-slope winner bars. Root cause: the keep's tracking-error
                # source (bull stab 0.787<0.80) occurs ONLY during portfolio DD, when
                # _ta_dd_hold_ext = 1.5 * _ta_long_gate * _ta_align * _ta_profit_gate *
                # (1-_port_dd_atten) is ACTIVE (the extended hold fires). Smoothing the
                # target OUTSIDE DD adds exit-lag with no stability benefit (the keep isn't
                # extending outside DD -> no tracking error to fix). Gate _ta_te_alpha on
                # the SAME (1-_port_dd_atten) DD state: smoothing fires ONLY during
                # portfolio DD (when the keep is actively extending the hold) -> byte-
                # identical to baseline outside DD (no bull winner lag during the normal
                # uptrend) -> only smooths the DD-extended winners (the actual tracking-error
                # population). Continuous (1-_port_dd_atten) in [0,1], no boundary. The keep
                # fires at deep DD (1-_port_dd_atten->1) -> full smoothing; at portfolio peak
                # (1-_port_dd_atten=0) -> no smoothing -> baseline. Uses the SAME top-level
                # _port_dd_atten (asymmetric-EMA, leverage-coupled) as _ta_dd_hold_ext.
                _ta_te_dd = max(0.0, 1.0 - _port_dd_atten)
                # Branch step5: PP-HARVEST EXEMPTION. Step3 (DD gate, best composite -0.076)
                # still lagged bull: bull's trend-aligned winners exit via pp_pressure giveback
                # harvest (deep peak -> giveback -> harvest). Smoothing the target during the
                # giveback phase LAGS the harvest exit -> winner gives back more -> bigger
                # realized loss. The structural separator between bull (smoothing hurts) and
                # crash/sideways/rally (smoothing helps): bull winners exit via pp_pressure
                # (gradual giveback harvest); crash shorts exit via slope-against (sharp
                # reversal) which dominates the smoothed target regardless. Gate _ta_te_alpha
                # OFF when pp_pressure is elevated (giveback harvest in progress): smoothing
                # fires ONLY on winners at their peak (pp_pressure ~0, riding the hold-extension)
                # -> holds the level stable through pullback noise (the keep's tracking-error
                # population); when giveback begins (pp_pressure rises), smoothing turns OFF
                # -> target tracks raw -> harvest fires at full speed. Per-bar structural property
                # (which exit source is binding), NOT a regime label. Continuous ramp over
                # [0, 0.3] pp_pressure (tanh, no boundary) -> smoothing fades out as the harvest
                # engages. Uses the existing _pp_pressure (post-attenuation, line ~2628).
                _ta_te_pp_exempt = max(0.0, 1.0 - max(0.0, _pp_pressure / 0.30))
                _ta_te_alpha = 0.30 * _ta_te_str * _ta_te_profit * _ta_te_slope_conf * _ta_te_dd * _ta_te_pp_exempt
                # Compose: ct-side _te_alpha and _ta_te_alpha fire on disjoint populations
                # (ct vs trend-aligned). Take the max so the active population's smoothing
                # applies; when both are 0 (sideways / weak-trend / losing), no smoothing
                # (target tracks raw -- preserves sideways byte-identical).
                _te_alpha = max(_te_alpha, _ta_te_alpha)
                # Exp3 / BRANCH step6: FEED-FORWARD 3-bar MEDIAN on the RAW target BEFORE
                # the _target_ema (median-then-EMA, replaces the opener's post-EMA median).
                # The _target_ema is a backward-looking LOW-PASS that raises rally stability
                # but incurs LAG on winner->loser transitions (the stab/raw tension). A
                # MEDIAN-of-3 has ZERO LAG for monotone moves (median of 3 rising values =
                # the middle -> tracks the rise) and ONLY rejects isolated AR(1) spikes.
                # Applying it BEFORE the EMA means the EMA smooths an already-spike-FREE
                # signal: the EMA's _prev_te no longer carries the previous bar's spike ->
                # the EMA's lag on spike-driven wobble is reduced -> the stab/raw tension
                # eases (stability gain WITHOUT the raw cost of lagging spike-driven
                # transitions). This is structurally distinct from a post-EMA median (which
                # attenuates the EMA OUTPUT -- downstream of the lag) -- here we attenuate
                # the EMA INPUT, preventing the spike from entering the EMA state at all.
                # Gated to high-churn ct positions (rally's binding population); trend-
                # aligned (ct gate 0) byte-identical by construction. Sign-preserving snap.
                _ct_mf_str = max(0.0, np.tanh(-_pos_dir_te * ret_vlong / 0.01))  # ~0 trend-aligned, ~1 ct-at-multi-day (noise-free fast-saturating)
                _mf_churn = max(0.0, np.tanh((len(_eh) - 1.5) / 0.6))  # ~0 low churn, ~1 bursting (noise-immune integer gate)
                # Exp2 (architectural, indep, this session): EXTEND the feed-forward median
                # filter to TREND-ALIGNED DD-EXTENDED high-churn positions. The prior session
                # documented (untested lead #1) that bull's residual tracking error (stability
                # 0.779<0.80 knee after the _hold_ext_ema keep; the _ta_dd_hold_ext and
                # _hold_adj wobbles were disproven as the bull source by the 0c5ac8c7 keep and
                # Exp1 respectively) lives in "the grid/quantization cascade during pp-harvest
                # phase when the target_EMA pp-exempt turns smoothing OFF" -- i.e. the emitted
                # target reverts to raw grid-quantized levels which wobble under AR(1) -> bull
                # tracking error. The sanctioned lag-free tool (feed-forward median-of-3, zero
                # lag for monotone moves, only rejects isolated AR(1) spikes) was previously
                # gated ct-only. Extend it to the trend-aligned DD-extended population (the
                # keep's _ta_dd_hold_ext fires for long-only trend-aligned in-profit during DD;
                # _ta_te_dd = 1-_port_dd_atten is the SAME DD-state signal the _ta_te_alpha
                # target-EMA uses, so this fires on the exact population the keep extends).
                # NEW data dep: the median filter now reads (trend-align, DD-state, churn)
                # jointly, not just (ct, churn). The median is sign-preserving and lag-free ->
                # no stab/raw tension (unlike the _ta_te_alpha EMA which lags and was gated off
                # in pp-harvest precisely to avoid lagging the harvest exit). Median collapses
                # the AR(1) spike-driven grid wobble WITHOUT lagging the harvest -> targets the
                # documented bull residual source directly. Byte-identical for the never-active
                # population (no DD -> _ta_te_dd 0 -> no extend; shorts/ct/losers per the keep
                # gates). Continuous, no decision boundary. Structural gate (trend-align x DD
                # x high-churn), NOT a regime label.
                # branch step3: add LONG-ONLY gate (mirror _ta_dd_hold_ext's _ta_long_gate).
                # step2 (slope-conf+profit) recovered mixed/sideways stability but crash STILL
                # catastrophic (-1.08): crash trend-aligned SHORTS are in-profit during the
                # downtrend (pass _ta_te_profit) and slope-confirms (pass _ta_te_slope_conf)
                # and are DD-extended (pass _ta_te_dd) -> they pass all 4 gates -> the median
                # lags their exit on dead-cat bounces (crash moves fast, 1 bar of lag on the
                # bounce = catastrophic). The structural separator: LONG vs SHORT risk
                # asymmetry -- longs in an uptrend during DD face pullbacks that RECOVER (bull
                # pullback longs); shorts in a downtrend during DD face dead-cat bounces that
                # are SHARPER (asymmetric upside risk). Same long-only structural property the
                # 0c5ac8c7 keep's _ta_dd_hold_ext uses (_ta_long_gate = 1 if current_pos>0).
                # Excludes crash shorts (the catastrophic regression source) while keeping bull
                # longs (the target). NOT a regime label -- direction-asymmetric structural prop.
                _ta_mf_long = 1.0 if current_pos > 0 else 0.0  # long-only structural gate (mirror _ta_long_gate)
                # branch step5: replace pp-exempt with UP-PERSIST gate. step4 (pp-exempt)
                # fixed crash (-0.157, byte-identical baseline) but bull stability CRASHED
                # (0.509 vs step1's 0.989) -- confirms the bull residual IS in pp-harvest and
                # the median MUST fire there to fix it. But firing during pp-harvest lags
                # crash bounce long exits (catastrophic -1.08 in step1/2/3). The structural
                # separator between bull longs (pp-harvest median HELPS) and crash bounce longs
                # (pp-harvest median HURTS): MULTI-DAY TREND PERSISTENCE. Bull = persistent
                # uptrend (_down_persist low ~0.3, ret_vlong rarely<0, pullbacks RECOVER);
                # crash = persistent downtrend (_down_persist high ~0.9, ret_vlong frequently<0,
                # the bounce IS the reversal). Use the SAME _up_persist_gate the codebase uses
                # for pp_pressure attenuation (line ~2629): full activation when _down_persist
                # <0.40 (persistent uptrend = bull), fading to 0 by 0.60 (crash ~0.9 -> gate 0).
                # This is the validated duration-count separator (fe6acd4d keep principle) --
                # a STRUCTURAL property (trend duration), NOT a regime label. Excludes crash
                # (persistent downtrend) while keeping bull (persistent uptrend). KEEP pp-harvest
                # ACTIVE (no pp-exempt) so the median fires during pp-harvest where the bull
                # benefit lives.
                _ta_mf_str = _ta_te_str * _ta_te_dd * _ta_te_slope_conf * _ta_te_profit * _ta_mf_long * _up_persist_gate  # trend-align x DD x slope-conf x profit x long-only x up-persist (6 gates; up-persist replaces pp-exempt)
                _mf_active = max(_ct_mf_str, _ta_mf_str)
                if _mf_active > 0.0 and _mf_churn > 0.0:
                    _th = self._target_hist.get(symbol, [])
                    _th.append(target)
                    if len(_th) > 3:
                        _th = _th[-3:]
                    self._target_hist[symbol] = _th
                    if len(_th) >= 3:
                        _med = float(np.median(_th))
                        if (_med > 0) == (target > 0) and _med != 0:
                            target = _med
                else:
                    self._target_hist[symbol] = []
                if _te_alpha > 0.0:
                    _prev_te = self._target_ema.get(symbol, target)
                    target = (1.0 - _te_alpha) * target + _te_alpha * _prev_te
                self._target_ema[symbol] = target

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
            # Exp2 (architectural, indep): HOLD-DURATION-CONDITIONED DEADBAND. Sanctioned
            # untested lead from prior session-summary (b00ace8f keep note): "The hold-
            # duration profile could be applied to OTHER emission-layer signals (the grid
            # quantization, the deadband) -- step1 applied it only to the mtm_chop trim."
            # BRANCH CONCLUSION (steps 1-4): the deadband width is EXTREMELY load-bearing --
            # any width modification (widen/narrow, all-bars/early-only/late-only, by-
            # direction, by-PnL) causes massive regime-flipping changes (bull -0.645->+1.681,
            # rally +0.747->-0.889) that CANNOT be isolated by direction or PnL gating. The
            # deadband is at its structural optimum; modifying it trades one regime's gain for
            # another's collapse. Step5 pivots to the OTHER sanctioned emission-layer signal:
            # the GRID QUANTIZATION width (line ~3408), applying the SAME validated hold-
            # duration profile there. The grid quantization rounds the resize target onto a
            # coarse lattice in high churn -- a DIFFERENT mechanism from the deadband (rounding
            # vs snap-to-hold) that may be more tractable to hold-duration-conditioning.
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
                # Branch step5: HOLD-DURATION-CONDITIONED grid width. Apply the validated
                # b00ace8f hold-duration profile to the grid LATTICE SPACING (was churn-only).
                # COARSER grid for early bars (bars<=3: |tanh| large -> profile up to ~1.3 ->
                # larger _grid -> more aggressive rounding -> suppress scale-in wobble); FINER
                # grid for late bars (bars>=8: profile ~0.7 -> smaller _grid -> less rounding
                # -> late-bar trims execute precisely). The OLD uniform grid applied the same
                # lattice spacing to bar-2 and bar-10 resizes alike. Continuous tanh profile
                # (no boundary). Byte-identical when _churn_dz=0 (low-churn regimes use the
                # complementary _calm_gate grid below, untouched).
                # Branch step6: ONE-SIDED early-bar coarsening only. Step5 (two-sided) gave
                # composite +0.017 but mixed regressed -0.017 (Sh 0.627->0.618). Hypothesis:
                # the late-bar FINER grid increased mixed's churn (more distinct position
                # values -> more trades -> fees). Test one-sided: only coarser for early bars
                # (max(...,1.0)), baseline 1.0x for late bars. If mixed recovers, the late-bar
                # finer grid was the culprit; if mixed stays regressed, the early-bar coarsening
                # is absorbing mixed's small-position scale-in.
                _grid_hold_profile = max(1.0, 1.0 - 0.3 * np.tanh((bars_held - 5.0) / 3.0))
                _grid = 0.06 * equity * BASE_POSITION_SIZE * _churn_dz * _grid_hold_profile
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
                # Exp1 (architectural, indep): POSITION-SIZE-CONDITIONED grid exemption
                # on the calm path. The calm grid quantizes same-sign resizes onto a
                # 0.06 lattice to cut noise-driven micro-resize churn in stability-
                # factor-1.0 regimes (the proven crash/sideways raw lever). But for
                # SMALL positions (|current_pos| near or below one grid step), the grid
                # ABSORBS genuine signal-driven reductions: a tp_harvest or de-risk trim
                # that should shrink the position snaps back to the SAME lattice point ->
                # emitted target unchanged -> the reduction never executes (prior
                # session root cause: "mixed positions are SMALL relative to the grid
                # step -- a 16.6pct harvest of a small position is <1 grid step -> snaps
                # to the same quantized lattice point -> emitted target unchanged").
                # mixed's positions sit at ~1 grid step (small BASE*combined_mult in a
                # low-vol down year); crash/sideways/bull sit at 2-4+ grid steps. Exempt
                # small positions (|current_pos| < 1.5 grid steps) from the grid so
                # signal-driven reductions execute; large positions keep the grid (their
                # noise-protection benefit is real, and their reductions are multi-grid
                # so not absorbed). NEW cross-component data dep: calm-grid application
                # depends on |current_pos| relative to the grid step. General principle
                # (grid protects large-position noise; for sub-grid positions the grid
                # only absorbs genuine signal) -- no regime label. The threshold (1.5
                # steps) is a continuous multiple of the same _grid_c, so it scales with
                # equity/BASE (leverage-invariant). Resize-only (entries/exits/flips
                # exempt by _is_resize). Risk: small-position resizes are no longer
                # quantized -> a small noisy resize could emit a sub-$1 trade; the
                # LEVERAGE_K-scaled emission threshold (1.0*LEVERAGE_K below) filters
                # sub-threshold resizes regardless.
                # Branch step4: gate the small-pos exemption on MTM-PATH-CHOP. Step1
                # exempted all small-position resizes (best composite 0.914862) but
                # sideways (which also has small positions) leaked micro-resizes
                # -> 48->57 trades, Sh 2.027->2.019. Step3 reductions-only lost rally's
                # gain without fixing sideways. The clean separator between mixed (where
                # exemption helps) and sideways (where it hurts) is the held position's
                # MTM-path efficiency (the proven mixed/sideways separator at the
                # emission throttle): mixed's small positions are CHOPPY dead capital
                # (low efficiency, whipsaw ~breakeven longs in a down year); sideways's
                # small-position reductions are on SMOOTHER winners (high efficiency).
                # Gate the exemption on chop (1 - MTM-eff over the 12-bar pos_pnl path):
                # exempt only when the held position is choppy (mixed); smooth winners
                # (sideways/bull/crash/rally trend longs) keep the grid. General
                # principle (no regime label): the grid-absorption cost is only worth
                # lifting for positions whose MTM path is itself noise-dominated (where
                # the grid absorbs a genuine reduction that the choppy path needs).
                _ppp_ex = self._pnl_path.get(symbol, [])
                _chop_ex = 0.0
                if len(_ppp_ex) >= 4:
                    _ppa_ex = np.array(_ppp_ex)
                    _net_ex = abs(_ppa_ex[-1] - _ppa_ex[0])
                    _tot_ex = float(np.sum(np.abs(np.diff(_ppa_ex))))
                    _mtm_eff_ex = _net_ex / max(_tot_ex, 1e-10)
                    _chop_ex = max(0.0, min(1.0, 1.0 - _mtm_eff_ex))
                _small_pos_exempt = _chop_ex > 0.30 and abs(current_pos) < 2.0 * _grid_c
                if _grid_c > 0 and not _small_pos_exempt:
                    _qt_c = round(target / _grid_c) * _grid_c
                    if (_qt_c > 0) == (target > 0) and _qt_c != 0:
                        target = _qt_c
            # Branch step2 (behavior-preserving leverage): scale the ABSOLUTE
            # minimum-trade emission threshold by LEVERAGE_K. The \$1 threshold
            # (here + prepare.py line 471, unmodifiable) is an ABSOLUTE dollar
            # floor, not a fraction -> it breaks scale-invariance under leverage:
            # during rally quiet stretches no grid fires so targets are continuous,
            # and sub-\$1 micro-resizes (suppressed at baseline) become ~2x at 2x
            # leverage -> cross \$1 -> emitted as noise-driven trades (rally trades
            # exploded 98->151, WR 84.7->70.9, stability crashed 1.0->0.235).
            # Scaling the emission threshold by LEVERAGE_K (1.0->2.0) filters the
            # sub-\$2 micro-resizes back out (the formerly sub-\$1 set, doubled),
            # restoring the baseline trade set. prepare.py's \$1 execution floor
            # is then moot (strategy already filters <\$2). This makes trade
            # SELECTION leverage-invariant (the last size-dependent decision gate).
            # Exp1 (this session): MTM-path-efficiency reduction throttle (emission
            # layer, downstream of all quantization — the ONLY layer reaching mixed
            # per prior session's root-cause finding). For a same-sign REDUCTION
            # resize (|target|<|current_pos|, not a flip/exit/entry), amplify the
            # trim proportional to how CHOPPY the held position's pos_pnl path is.
            # MTM-path-efficiency = |net| / sum|delta| over the 12-bar pos_pnl path,
            # in [0,1]: HIGH = smooth climber (bull/crash/sideways/rally trend longs),
            # LOW = whipsaw dead-capital (mixed's wrong-side long book). chop = 1-eff;
            # the reduction's distance-to-current is scaled up by (1+AMP*chop), pushed
            # toward (never past) current_pos's own already-reduced target. Smooth
            # winners (chop~0) -> byte-identical. Reduction-only (risk-reducing). The
            # separator last session's trend-alignment emission de-risk lacked: it
            # trimmed bull/crash too; MTM-path-efficiency spares smooth winners by
            # construction. New per-position state + new control flow at emission.
            _is_reduction = (current_pos != 0 and target != 0
                             and (current_pos > 0) == (target > 0)
                             and abs(target) < abs(current_pos))
            if _is_reduction:
                _ppp = self._pnl_path.get(symbol, [])
                if len(_ppp) >= 4:
                    _ppa = np.array(_ppp)
                    _net = abs(_ppa[-1] - _ppa[0])
                    _tot = float(np.sum(np.abs(np.diff(_ppa))))
                    _mtm_eff = _net / max(_tot, 1e-10)  # [0,1]
                    _mtm_chop = max(0.0, min(1.0, 1.0 - _mtm_eff))
                    # Branch step3: LOW-VOL GRIND gate (replaces step2's profit-fade,
                    # which killed rally's gain while not fixing bull). Step2 proved
                    # rally's benefit is trimming choppy GRINDING winners (not just
                    # losers) and bull's loss is NOT winning-trim. The real separator is
                    # VOL REGIME: bull-2021 is a HIGH-vol SHARP uptrend where pullbacks
                    # recover fast (trimming = selling a dip that bounces back -> the
                    # -0.283 loss); rally-2024 is a LOW-vol GRIND where pullbacks are
                    # deeper and giveback-prone (trimming a choppy held position = less
                    # giveback -> the +0.045 gain). Gate the throttle on low vol_ratio:
                    # full at vol_ratio<=0.8 (calm grind), fading to 0 at vol_ratio>=1.3
                    # (sharp/high-vol). bull's high-vol sharp pullbacks -> gate ~0 ->
                    # spared; rally's low-vol grind -> gate ~1 -> trimmed. Continuous
                    # (no boundary). General vol-regime principle (no regime label).
                    _grind_gate = max(0.0, min(1.0, (1.3 - vol_ratio) / 0.5))
                    # Branch step5: STRONG-UPTREND fade (replaces step4's deep-winner
                    # fade, which ate rally without fixing bull). bull -0.155 is the
                    # entire remaining gap; bull's trimmed positions are WINNING longs
                    # in a STRONG multi-day uptrend (ret_vlong ~+0.027) whose pullbacks
                    # recover, so trimming sells dips that bounce. rally's beneficial
                    # trims are in a WEAK uptrend (ret_vlong ~+0.006). Fade the throttle
                    # by trend-aligned multi-day strength: ret_vlong*pos_dir scaled /0.02
                    # -> bull (0.027) fade ~0.13 (spared), rally (0.006) fade ~0.71
                    # (kept). Only trend-ALIGNED strength fades (counter-trend/down gets
                    # max(0,..)=0 -> fade 1 -> full throttle), so mixed's wrong-side longs
                    # in a downtrend (ret_vlong<0) stay fully throttled. Smooth, no
                    # boundary. Documented bull/rally separator (multi-day trend strength).
                    _pos_dir_mtm = 1.0 if current_pos > 0 else -1.0
                    _strong_trend_fade = max(0.0, 1.0 - np.tanh(max(0.0, ret_vlong * _pos_dir_mtm) / 0.02))
                    # Branch step9: WINNER fade (recovers crash -0.0056, the keep-blocker).
                    # crash's throttle drag is trimming WINNING shorts being reduced
                    # (crash is 100pctWR, its reductions are profit-takes on winners ->
                    # the throttle front-runs them = sells the winner early). mixed's
                    # beneficial trims are LOW-PnL dead capital (whippy ~breakeven longs).
                    # Fade the throttle for clear winners: full at pos_pnl<=+0.5*stop,
                    # fading to ~0 at pos_pnl>=+1.5*stop. This is DISTINCT from step2's
                    # symmetric profit-fade (which faded from 0 and killed rally's modest-
                    # PnL grind trims): here the fade ONSET is at +0.5*stop so mixed's
                    # ~breakeven and rally's modest-PnL trims keep full throttle, only
                    # CLEAR winners (crash profit-take shorts) are spared. Smooth.
                    _winner_fade = max(0.0, min(1.0, 1.0 - (pos_pnl / abs(STOP_LOSS_PCT) - 0.5) / 1.0))
                    # STRUCTURAL_EXPLORATION step1: HOLD-DURATION-CONDITIONED trim profile
                    # on the emission-layer reduction throttle. The prior single multiplicative
                    # amplifier (_trim_mult = 1 + AMP*chop*grind*trend_fade*winner_fade) is
                    # at its structural ceiling -- every gate factor has been tuned (grind,
                    # trend_fade, winner_fade each a validated keep). The remaining untested
                    # axis is the POSITION'S LIFE-CYCLE PHASE: the trim should be WEAKER for
                    # early-bar reductions (let the position establish -- a bar-3 choppy
                    # reduction might be a scale-in wobble, not dead capital) and STRONGER
                    # for mid/late-life choppy reductions (a bar-8 choppy reduction IS dead
                    # capital that has bled for 8 bars). NEW cross-component data dep: trim
                    # magnitude reads bars_held jointly with mtm_chop (was chop-only). This
                    # is a subsystem rewrite of the emission throttle's MAGNITUDE FUNCTION
                    # (was a flat single multiplier across all held bars -> now a hold-
                    # duration-profiled multiplier). The OLD mechanism at its structural
                    # ceiling: prior sessions tuned grind/trend_fade/winner_fade to isolate
                    # mixed's beneficial trims from bull/crash/rally regressions, but the
                    # trim MAGNITUDE is uniform across the position's life -- a bar-2 choppy
                    # reduction gets the SAME trim as a bar-10 choppy reduction, even though
                    # the bar-10 one is far more likely to be genuine dead capital. The new
                    # mechanism: _hold_durations factor = 0.5 at bars_held<=3 (light trim,
                    # let position establish), ramping to 1.0 at bars_held>=6 (full trim),
                    # then to 1.4 at bars_held>=12 (aggressive trim on clearly-dead capital).
                    # Continuous tanh profile (no boundary); reduction-only (safe family);
                    # byte-identical when _mtm_chop=0 (smooth winners, chop~0 -> trim_mult
                    # ~1.0 regardless of hold_duration). Direction-agnostic. Targets mixed
                    # (the binding positive regime whose dead-capital longs bleed across
                    # many bars) while sparing early-bar reductions everywhere.
                    _hold_dur_profile = 0.5 + 0.9 * max(0.0, min(1.0, np.tanh((bars_held - 3.0) / 3.0)))
                    # Amplify the reduction distance; clamp so target stays same-sign
                    # and never trims past full close (toward 0, not across it).
                    _trim_mult = 1.0 + MTM_CHOP_TRIM_AMP * _mtm_chop * _grind_gate * _strong_trend_fade * _winner_fade * _hold_dur_profile
                    _new_target = current_pos + (target - current_pos) * _trim_mult
                    if (_new_target > 0) == (current_pos > 0) and abs(_new_target) < abs(current_pos):
                        target = _new_target
            # Branch step13b: emission threshold reduction for small reductions gated
            # on MULTI-DAY TREND DIRECTION (not chop). Step13 (chop gate) was
            # CATASTROPHIC for bull: bull's pullback positions are small AND choppy
            # (sharp pullback = low MTM-eff) so the chop gate fired on them too. The
            # separator that cleanly distinguishes mixed's held longs (in a multi-day
            # DOWNTREND, ret_vlong<0) from bull's pullback longs (in a multi-day
            # UPTREND, ret_vlong>0) is the SIGN of ret_vlong*pos_dir. mixed's wrong-
            # side longs: ret_vlong<0, pos_dir=+1 -> product<0 (counter-trend at
            # multi-day). bull's pullback longs: ret_vlong>0, pos_dir=+1 -> product>0
            # (trend-aligned). Gate the lower emission threshold on COUNTER-TREND-AT-
            # MULTI-DAY small positions (mixed's dead capital) -> bull's trend-aligned
            # pullback longs (product>0) keep the baseline threshold -> NOT over-traded.
            # Fast-saturating /0.01 ret_vlong scale (near-constant, noise-free per the
            # validated branch-step-9 lesson). Same 0.7*LEVERAGE_K floor. General
            # principle: lower emission bar for counter-trend-at-multi-day small
            # reductions (the dead-capital population).
            _emit_thresh = 1.0 * LEVERAGE_K
            _grid_em = 0.06 * equity * BASE_POSITION_SIZE
            if _is_resize and abs(current_pos) < 2.0 * _grid_em:
                _pos_dir_em = 1.0 if current_pos > 0 else -1.0
                _ct_vlong_em = max(0.0, np.tanh(-_pos_dir_em * ret_vlong / 0.01))
                if _ct_vlong_em > 0.50:
                    _emit_thresh = 0.7 * LEVERAGE_K
            if abs(target - current_pos) > _emit_thresh:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    if current_pos != 0:
                        _ep = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                        _exit_pnl_signed = -_ep if current_pos < 0 else _ep
                        self._last_exit_pnl[symbol] = _exit_pnl_signed
                        # Exp3: update portfolio consecutive-loss streak (mirrors
                        # max_consecutive_losses over chronological trade_pnls).
                        if _exit_pnl_signed < 0:
                            self._loss_streak += 1
                        else:
                            self._loss_streak = 0
                    for _d in (self.entry_prices, self.peak_pnl, self.entry_bar, self._smoothed_pnl, self._mae, self._voter_bias_ema, self._target_ema, self._conc_shrink_held, self._vol_shrink_held, self._cv_shrink_held, self._pnl_path, self._target_hist, self._hold_ext_ema):
                        _d.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                    # Branch step2: reset readiness accumulator on full exit so re-entry
                    # must REBUILD conviction from scratch (mimics the baseline 2-bar fresh
                    # persist requirement). Kills the cross-position EMA memory that carried
                    # high conviction through a hold and enabled immediate post-exit re-entry
                    # — the churn source behind the step1 bull raw regression (-0.222).
                    self._entry_accum[symbol] = (0.0, 0.0)
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol], self.peak_pnl[symbol], self.entry_bar[symbol] = mid, 0.0, self.bar_count
                    self._mae[symbol] = 0.0
                    _h = self._entry_bar_history.setdefault(symbol, [])
                    _h.append(self.bar_count)

        return signals
