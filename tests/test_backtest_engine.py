"""
Regression tests for the backtest engine core logic (prepare.py).

These lock in the hand-verified behaviour of run_backtest / compute_score so that
ANY future change to the backtest framework can be checked against known-correct
numbers. Every expected value below is hand-calculated from the fee/slippage
constants — they are NOT snapshots of whatever the engine currently prints, they
are independent ground truth.

Run:  uv run pytest test_backtest_engine.py -v
      (or just  uv run pytest)

WHEN YOU CHANGE prepare.py: run this FIRST. If a test fails, either your change
introduced a regression, or you intentionally changed engine semantics and must
update the expected value here (with a comment explaining why).

History: written after the streak_gate-input bug (fix de9b1e19), where partial
reduces recorded pnl=0 in trade_log and flips/reduces were excluded from
max_consecutive_losses, causing a spurious loss-streak penalty on profitable
strategies. test_reduce_counts_in_stats / test_flip_counts_in_stats guard that.
"""

import math

import pandas as pd
import pytest

from prepare import (
    run_backtest,
    compute_score,
    INITIAL_CAPITAL,
    TAKER_FEE,
    SLIPPAGE_BPS,
    MAX_LEVERAGE,
    Signal,
    BacktestResult,
)

TOL = 1e-3  # tolerance on percentage returns / scores


def _mkdata(prices):
    """Single-symbol BTC series, no intrabar noise (high=low=close=open)."""
    n = len(prices)
    return {
        "BTC": pd.DataFrame({
            "timestamp": [1_600_000_000_000 + i * 3600_000 for i in range(n)],
            "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": [1.0] * n, "funding_rate": [0.0] * n,
        })
    }


# Derived execution constants (must mirror the engine's slippage/fee model)
def _buy_px(p):
    return p * (1 + SLIPPAGE_BPS / 10000)


def _sell_px(p):
    return p * (1 - SLIPPAGE_BPS / 10000)


class _ScriptedStrategy:
    """Emits a target position the first time close hits each trigger price."""
    def __init__(self, steps):
        # steps: list of (trigger_price, target_notional)
        self.steps = list(steps)
        self.idx = 0

    def on_bar(self, bar_data, portfolio):
        bd = bar_data.get("BTC")
        if bd is None or self.idx >= len(self.steps):
            return []
        trigger, target = self.steps[self.idx]
        if abs(bd.close - trigger) < 1e-9:
            self.idx += 1
            return [Signal(symbol="BTC", target_position=target)]
        return []


# ---------------------------------------------------------------------------
# Open + hold through an unrealized loss. Verifies equity = cash + Σ|pos| +
# unrealized is conserved (the +Σ|pos| term is correct because cash is debited
# the notional at open).
# ---------------------------------------------------------------------------
def test_open_hold_unrealized_loss():
    data = _mkdata([100.0, 110.0, 110.0, 99.0, 99.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0)]), data)

    exec_p = _buy_px(110.0)
    fee = 10000 * TAKER_FEE
    unreal = 10000 * (99 - exec_p) / exec_p
    hand_equity = (INITIAL_CAPITAL - fee - 10000) + 10000 + unreal
    expected_ret = (hand_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    assert r.total_return_pct == pytest.approx(expected_ret, abs=TOL)
    # num_trades counts COMPLETED (realizing) trades; this position is opened and
    # held to the end without closing, so there are zero realized trades. The open
    # leg itself must NOT count.
    assert r.num_trades == 0


# ---------------------------------------------------------------------------
# Flip (long -> short). Verifies flip PnL, the DOUBLE fee on |delta| (close old
# + open new), and the cash flow. This is the cost path historically scrutinised
# for correct flip accounting.
# ---------------------------------------------------------------------------
def test_flip_pnl_and_double_fee():
    data = _mkdata([100.0, 110.0, 121.0, 121.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0), (121.0, -10000.0)]), data)

    exec_buy = _buy_px(110.0)
    exec_sell = _sell_px(121.0)
    fee_open = 10000 * TAKER_FEE
    fee_flip = 20000 * TAKER_FEE  # |delta| = |−10000 − 10000| = 20000
    flip_pnl = 10000 * (exec_sell - exec_buy) / exec_buy
    cash = INITIAL_CAPITAL - fee_open - 10000
    cash = cash - fee_flip + 10000 + flip_pnl - 10000
    unreal = -10000 * (121 - exec_sell) / exec_sell
    equity = cash + 10000 + unreal
    expected_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    assert r.total_return_pct == pytest.approx(expected_ret, abs=TOL)
    assert r.flip_count == 1


# ---------------------------------------------------------------------------
# Partial reduce realizes PnL into equity correctly. (The reduce PnL was always
# correct in cash/equity; only its trade_log record was 0 pre-fix.)
# ---------------------------------------------------------------------------
def test_reduce_realizes_into_equity():
    data = _mkdata([100.0, 100.0, 110.0, 110.0])
    r = run_backtest(_ScriptedStrategy([(100.0, 10000.0), (110.0, 5000.0)]), data)

    exec_buy = _buy_px(100.0)
    exec_sell = _sell_px(110.0)
    pnl_reduce = 5000 * (exec_sell - exec_buy) / exec_buy
    cash = INITIAL_CAPITAL - 10000 * TAKER_FEE - 10000  # after open
    cash = cash + 5000 + pnl_reduce - 5000 * TAKER_FEE
    unreal = 5000 * (110 - exec_buy) / exec_buy
    equity = cash + 5000 + unreal
    expected_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    assert r.total_return_pct == pytest.approx(expected_ret, abs=TOL)


# ---------------------------------------------------------------------------
# REGRESSION for fix de9b1e19: partial reduce PnL must be counted in trade
# statistics (win_rate / profit_factor / max_consecutive_losses), not just full
# closes. Before the fix, the reduce below was invisible to the loss-streak
# counter.
# ---------------------------------------------------------------------------
def test_reduce_counts_in_stats():
    data = _mkdata([100.0, 100.0, 110.0, 110.0])
    r = run_backtest(_ScriptedStrategy([(100.0, 10000.0), (110.0, 5000.0)]), data)

    # The reduce locked in a profit; with no losing realizing events the loss
    # streak must be 0 and win_rate must be 100%.
    assert r.max_consecutive_losses == 0
    assert r.win_rate_pct == pytest.approx(100.0, abs=TOL)
    # trade_log reduce row must carry the realized flag and real PnL
    modify_rows = [t for t in r.trade_log if t[0] == "modify"]
    assert len(modify_rows[0]) >= 6, "reduce row missing realized flag (tuple index 5)"
    assert modify_rows[0][5] is True, "reduce row should be realized=True"
    assert abs(float(modify_rows[0][4])) > 1.0, "reduce row should record nonzero PnL"


# ---------------------------------------------------------------------------
# REGRESSION for fix de9b1e19: flip PnL must register in trade stats.
# ---------------------------------------------------------------------------
def test_flip_counts_in_stats():
    data = _mkdata([100.0, 110.0, 121.0, 121.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0), (121.0, -10000.0)]), data)
    # The flip closed a profitable long -> it's a win, no loss streak.
    assert r.max_consecutive_losses == 0
    assert r.win_rate_pct == pytest.approx(100.0, abs=TOL)


# ---------------------------------------------------------------------------
# A LOSING full close must register as a loss in the streak counter. Guards
# against the inverse mistake (over-eager realized filtering).
# ---------------------------------------------------------------------------
def test_losing_close_counts_as_loss():
    # open long @110, close @99 (loss)
    data = _mkdata([100.0, 110.0, 99.0, 99.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0), (99.0, 0.0)]), data)
    assert r.max_consecutive_losses == 1
    assert r.win_rate_pct == pytest.approx(0.0, abs=TOL)


# ---------------------------------------------------------------------------
# Opening a position is NOT a realizing event (must not pollute stats). A single
# open with no close => no realized trades => win_rate 0, streak 0.
# ---------------------------------------------------------------------------
def test_open_is_not_realizing():
    data = _mkdata([100.0, 110.0, 110.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0)]), data)
    assert r.max_consecutive_losses == 0
    # open row carries realized=False
    open_rows = [t for t in r.trade_log if t[0] == "open"]
    assert open_rows[0][5] is False


# ---------------------------------------------------------------------------
# num_trades counts COMPLETED (realizing) trades, not raw order legs. An open
# followed by a close is ONE trade, not two. Guards against reverting to
# len(trade_log) (which would inflate the count and loosen sample_factor and
# the <10-trade hard cutoff).
# ---------------------------------------------------------------------------
def test_num_trades_counts_completed_trades():
    # open long @110, close @121 -> trade_log has [open, close] but only 1 trade
    data = _mkdata([100.0, 110.0, 121.0, 121.0])
    r = run_backtest(_ScriptedStrategy([(110.0, 10000.0), (121.0, 0.0)]), data)
    assert len(r.trade_log) == 2, "trade_log should record both the open and the close leg"
    assert r.num_trades == 1, "num_trades should count the completed trade, not the open leg"


# ---------------------------------------------------------------------------
# compute_score hard cutoffs.
# ---------------------------------------------------------------------------
def test_compute_score_under_10_trades():
    r = BacktestResult(sharpe=2.0, num_trades=5, max_drawdown_pct=1.0,
                       equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL])
    assert compute_score(r) == pytest.approx(-999.0, abs=TOL)


def test_compute_score_over_10pct_dd():
    r = BacktestResult(sharpe=2.0, num_trades=50, max_drawdown_pct=12.0,
                       equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL])
    assert compute_score(r) == pytest.approx(-999.0, abs=TOL)


def test_compute_score_lost_over_15pct():
    r = BacktestResult(sharpe=2.0, num_trades=50, max_drawdown_pct=2.0,
                       equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 0.80])
    assert compute_score(r) == pytest.approx(-999.0, abs=TOL)


# ---------------------------------------------------------------------------
# compute_score multiplicative factors with a known BacktestResult.
# ---------------------------------------------------------------------------
def test_compute_score_factors():
    # equity_curve: 8761 points = 1 year of hourly bars; 5% total return
    ec = [INITIAL_CAPITAL] * 8760 + [INITIAL_CAPITAL * 1.05]
    r = BacktestResult(
        sharpe=1.0, num_trades=50, max_drawdown_pct=5.0,
        return_volatility=0.10, max_consecutive_losses=0,
        total_return_pct=5.0,
        equity_curve=ec,
    )
    signal_quality = math.log(1.0 + 1.0)
    sample_factor = math.sqrt(min(50 / 50.0, 1.0))          # = 1.0
    dd_gate = (1.0 / (1.0 + 5.0 / 100.0)) * math.exp(-max(0.0, 5.0 - 5.0) / 2.0)
    streak_gate = math.exp(-0 / 30.0)                        # = 1.0
    # return_bonus (2026-06-24): direct APY reward, replaces calmar return_reward.
    # fixture: APY=5% (1 year so APY==total). return_bonus = log(1 + APY/10 + 1)
    apy = 5.0  # ann_return (1-year fixture, APY == total_return_pct)
    return_bonus = math.log(1.0 + max(apy, 0.0) / 10.0 + 1.0)
    # vol_gate removed 2026-06-19 (double penalty with Sharpe's std denominator)
    expected = signal_quality * sample_factor * dd_gate * streak_gate * return_bonus
    assert compute_score(r) == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# sharpe <= 0 returns the bare sharpe (continuous negative gradient), NOT a flat
# 0 ("silent band") and NOT a sign-flipped multiplicative score. Guards the M1
# fix. These all pass the hard cutoffs (>=10 trades, DD<=10%, loss<15%).
# ---------------------------------------------------------------------------
def _passing_result(sharpe, dd=2.0, vol=0.3, streak=0):
    # equity ends slightly down (>85% so no hard cutoff) for sharpe<=0 cases
    return BacktestResult(
        sharpe=sharpe, num_trades=50, max_drawdown_pct=dd,
        return_volatility=vol, max_consecutive_losses=streak,
        equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 0.99],
    )


def test_negative_sharpe_returns_bare_sharpe():
    # sharpe <= 0 -> score == sharpe exactly, independent of the risk gates.
    for sh in [0.0, -0.05, -0.5, -2.0, -10.0]:
        assert compute_score(_passing_result(sh)) == pytest.approx(sh, abs=TOL)


def test_negative_region_gradient_not_flat():
    # The old silent band collapsed all sharpe<=0 to 0; now there is a gradient:
    # a near-miss must score strictly higher than a blow-up.
    near_miss = compute_score(_passing_result(-0.05))
    blow_up = compute_score(_passing_result(-5.0))
    assert near_miss > blow_up


def test_negative_region_no_sign_flip_from_risk_gates():
    # Two losing strategies with the SAME sharpe but different risk profiles must
    # score identically (risk gates must NOT invert sign in the negative region).
    low_risk = compute_score(_passing_result(-0.5, dd=1.0, vol=0.1, streak=0))
    high_risk = compute_score(_passing_result(-0.5, dd=9.0, vol=2.0, streak=20))
    assert low_risk == pytest.approx(high_risk, abs=TOL)
    assert low_risk == pytest.approx(-0.5, abs=TOL)


def test_score_continuous_and_monotonic_at_zero():
    # Continuity at sharpe=0 and monotonicity across the boundary.
    s_neg = compute_score(_passing_result(-0.01))
    s_zero = compute_score(_passing_result(0.0))
    s_pos = compute_score(BacktestResult(
        sharpe=0.01, num_trades=50, max_drawdown_pct=2.0,
        return_volatility=0.3, max_consecutive_losses=0,
        equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 1.001],
    ))
    assert s_zero == pytest.approx(0.0, abs=TOL)
    assert s_neg < s_zero <= s_pos  # monotonic across the boundary


def test_hard_cutoff_precedes_negative_region():
    # A negative-sharpe strategy that ALSO breaches a hard cutoff must still get
    # -999 (cutoffs run before the bare-sharpe return).
    r = BacktestResult(
        sharpe=-0.5, num_trades=50, max_drawdown_pct=12.0,  # DD breach
        return_volatility=0.3, max_consecutive_losses=0,
        equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 0.99],
    )
    assert compute_score(r) == pytest.approx(-999.0, abs=TOL)


# ---------------------------------------------------------------------------
# Truncated runs (hit TIME_BUDGET, stopped early) are non-reproducible and
# survivorship-biased -> compute_score must reject them with -999. A normal
# (non-truncated) result with the same metrics must score normally. Guards H3.
# ---------------------------------------------------------------------------
def test_truncated_result_rejected():
    normal = BacktestResult(
        sharpe=2.0, num_trades=50, max_drawdown_pct=2.0,
        return_volatility=0.3, max_consecutive_losses=0,
        equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 1.05],
        truncated=False,
    )
    truncated = BacktestResult(
        sharpe=2.0, num_trades=50, max_drawdown_pct=2.0,
        return_volatility=0.3, max_consecutive_losses=0,
        equity_curve=[INITIAL_CAPITAL, INITIAL_CAPITAL * 1.05],
        truncated=True,
    )
    assert compute_score(normal) > 0, "non-truncated result should score normally"
    assert compute_score(truncated) == pytest.approx(-999.0, abs=TOL)


def test_default_result_not_truncated():
    # Default must be False so normal runs are never accidentally rejected.
    assert BacktestResult().truncated is False


# ---------------------------------------------------------------------------
# Leverage constraint rejects orders that exceed MAX_LEVERAGE.
# ---------------------------------------------------------------------------
def test_leverage_constraint():
    data = _mkdata([100.0, 100.0, 100.0])
    over_lever = INITIAL_CAPITAL * (MAX_LEVERAGE + 5)  # well past the cap
    r = run_backtest(_ScriptedStrategy([(100.0, over_lever)]), data)
    # Order should be skipped -> no position opened -> no trades.
    assert r.num_trades == 0


# ---------------------------------------------------------------------------
# Multi-symbol timestamp misalignment: when a held symbol has no bar at a
# timestamp, mark-to-market must carry-mark it with its last observed close
# instead of dropping its unrealized PnL (which would still count its notional,
# producing a spurious equity swing — the "ghost jump" bug). Guards task1-C1.
# ---------------------------------------------------------------------------
def test_carry_mark_no_ghost_jump_on_missing_bar():
    ts = [1_600_000_000_000 + i * 3600_000 for i in range(5)]
    # BTC present every bar (flat 100, no PnL of its own).
    btc = pd.DataFrame({"timestamp": ts, "open": [100.0]*5, "high": [100.0]*5,
                        "low": [100.0]*5, "close": [100.0]*5,
                        "volume": [1.0]*5, "funding_rate": [0.0]*5})
    # ETH present at ts 0,1,3,4 but MISSING at ts index 2.
    eth_ts = [ts[0], ts[1], ts[3], ts[4]]
    eth = pd.DataFrame({"timestamp": eth_ts, "open": [2000., 2200., 2200., 2200.],
                        "high": [2000., 2200., 2200., 2200.],
                        "low": [2000., 2200., 2200., 2200.],
                        "close": [2000., 2200., 2200., 2200.],
                        "volume": [1.0]*4, "funding_rate": [0.0]*4})

    class OpenETHOnce:
        def __init__(self):
            self.done = False
        def on_bar(self, bd, pf):
            if "ETH" in bd and not self.done:
                self.done = True
                return [Signal(symbol="ETH", target_position=10000.0)]  # $10k long @2000
            return []

    r = run_backtest(OpenETHOnce(), {"BTC": btc, "ETH": eth})
    eq = [float(e) for e in r.equity_curve]
    # After ETH rises to 2200 (unrealized +~$1000), equity should stay flat
    # across the missing-bar gap, NOT dip and recover. Check the tail (post-rise)
    # has no large bar-to-bar swing.
    tail = eq[2:]  # from when ETH has risen onward
    max_swing = max(abs(tail[i+1] - tail[i]) for i in range(len(tail) - 1)) if len(tail) > 1 else 0.0
    assert max_swing < 50.0, f"ghost jump detected: equity tail {tail} has swing {max_swing:.2f}"
