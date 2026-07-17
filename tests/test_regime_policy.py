"""Tests for development/holdout time separation and robust composite output."""

from regime_test import HOLDOUT_REGIMES, SEARCH_REGIMES, compute_robust_composite


def test_recent_development_fold_does_not_overlap_holdout():
    search_end = max(r[2] for r in SEARCH_REGIMES)
    holdout_start = min(r[1] for r in HOLDOUT_REGIMES)
    assert search_end < holdout_start
    assert any(r[0] == "recent_2026q1" for r in SEARCH_REGIMES)


def test_legacy_sideways_key_is_described_as_recovery_not_pure_chop():
    sideways = next(r for r in SEARCH_REGIMES if r[0] == "sideways")
    description = sideways[3].lower()
    assert "recovery" in description
    assert "low-vol" in description


def test_recent_fold_is_explicitly_intentionally_overweighted():
    recent = next(r for r in SEARCH_REGIMES if r[0] == "recent_2026q1")
    assert "intentionally overweight" in recent[3].lower()


def test_robust_composite_does_not_reward_single_regime_spike():
    results = [{"score": x} for x in [0, 0, 0, 0, 0, 1]]
    assert compute_robust_composite(results) <= 0
