"""Regression tests for autoresearch anti-overfitting policy."""

import json

import pytest

from research_policy import (
    PromotionBudget,
    complexity_adjusted_threshold,
    evaluate_internal_candidate,
    evaluate_promotion,
    robust_environment_score,
    sanitize_promotion_result,
)


def test_robust_environment_score_rejects_single_environment_spike():
    # Mean is strongly positive, but the median environment is unchanged and the
    # gain is concentrated in one fold. A robust objective must not reward this.
    deltas = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert robust_environment_score(deltas) <= 0.0


def test_internal_candidate_requires_broad_directional_improvement():
    result = evaluate_internal_candidate(
        baseline_scores=[1.0] * 6,
        candidate_scores=[1.0, 1.0, 1.0, 1.0, 1.0, 1.2],
        affected_trades=50,
        complexity_points=0,
    )
    assert not result.passed
    assert "concentration" in result.reasons
    assert "majority" in result.reasons


def test_internal_candidate_accepts_broad_improvement():
    result = evaluate_internal_candidate(
        baseline_scores=[1.0] * 6,
        candidate_scores=[1.01, 1.02, 1.01, 1.03, 1.02, 1.01],
        affected_trades=40,
        complexity_points=1,
    )
    assert result.passed
    assert result.robust_delta > 0


def test_internal_candidate_counts_zero_delta_as_non_regression():
    result = evaluate_internal_candidate(
        baseline_scores=[1.0] * 6,
        candidate_scores=[1.01, 1.01, 1.01, 1.01, 1.0, 1.0],
        affected_trades=40,
        complexity_points=0,
    )
    assert result.passed


def test_complexity_raises_required_gain():
    assert complexity_adjusted_threshold(0) == pytest.approx(0.003)
    assert complexity_adjusted_threshold(4) > complexity_adjusted_threshold(1)


def test_promotion_is_veto_only_and_hides_metrics():
    detailed = evaluate_promotion(
        okx_delta=0.02,
        recent_sharpe_delta=0.1,
        recent_return_delta=0.2,
        recent_trade_ratio=0.9,
        unseen_delta=0.01,
    )
    public = sanitize_promotion_result(detailed)
    assert public == {"promotion_gate": "PASS"}
    assert "okx_delta" not in json.dumps(public)


def test_promotion_failure_closes_mechanism_without_margin_feedback():
    detailed = evaluate_promotion(
        okx_delta=-0.02,
        recent_sharpe_delta=-0.5,
        recent_return_delta=-1.0,
        recent_trade_ratio=0.4,
        unseen_delta=-0.1,
    )
    public = sanitize_promotion_result(detailed)
    assert public == {
        "promotion_gate": "FAIL",
        "action": "CLOSE_MECHANISM_FAMILY",
    }


def test_budget_allows_one_attempt_per_session_and_family(tmp_path):
    path = tmp_path / "budget.json"
    budget = PromotionBudget(path)
    assert budget.consume("session-1", "family-a")
    assert not budget.consume("session-1", "family-b")
    assert not budget.consume("session-2", "family-a")
    assert budget.consume("session-2", "family-b")
