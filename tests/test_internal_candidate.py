"""Tests for extracting internal environment metrics from regime logs."""

from internal_candidate import count_changed_trade_events, parse_regime_scores


def test_parse_regime_scores_includes_recent_fold(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "regime_bull_2021_score: 1.0\n"
        "regime_recent_2026q1_score: 2.0\n"
    )
    assert parse_regime_scores(log) == {
        "bull_2021": 1.0,
        "recent_2026q1": 2.0,
    }


def test_count_changed_trade_events_uses_sequence_diff():
    baseline = [("open", "BTC"), ("close", "BTC"), ("open", "ETH")]
    candidate = [("open", "BTC"), ("close", "BTC"), ("open", "SOL")]
    assert count_changed_trade_events(baseline, candidate) == 2
