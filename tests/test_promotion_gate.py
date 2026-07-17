"""Tests for the owner-only promotion-gate CLI boundary."""

from promotion_gate import build_public_result


def test_public_result_contains_no_detailed_metric_names():
    public = build_public_result(
        okx_delta=-1.0,
        recent_sharpe_delta=-2.0,
        recent_return_delta=-3.0,
        recent_trade_ratio=0.2,
        unseen_delta=-4.0,
    )
    text = str(public)
    assert public["promotion_gate"] == "FAIL"
    for forbidden in ("okx", "sharpe", "return", "trade_ratio", "unseen", "reason"):
        assert forbidden not in text.lower()
