#!/usr/bin/env python3
"""Owner-only promotion gate boundary.

The evaluator that computes detailed metrics writes them to a private JSONL log and
passes only the sanitized object from ``build_public_result`` to stdout. The agent
must never receive detailed promotion metrics or failure margins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_policy import (
    PromotionBudget,
    evaluate_promotion,
    sanitize_promotion_result,
    write_private_promotion_log,
)


def build_public_result(
    *,
    okx_delta: float,
    recent_sharpe_delta: float,
    recent_return_delta: float,
    recent_trade_ratio: float,
    unseen_delta: float,
) -> dict[str, str]:
    decision = evaluate_promotion(
        okx_delta=okx_delta,
        recent_sharpe_delta=recent_sharpe_delta,
        recent_return_delta=recent_return_delta,
        recent_trade_ratio=recent_trade_ratio,
        unseen_delta=unseen_delta,
    )
    return sanitize_promotion_result(decision)


def main() -> int:
    parser = argparse.ArgumentParser(description="sealed promotion decision")
    parser.add_argument("--session", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--metrics-json", required=True,
                        help="owner-only JSON file produced by validation runner")
    parser.add_argument("--budget", default=".autoresearch/promotion_budget.json")
    parser.add_argument("--private-log", default=".autoresearch/private/promotion.jsonl")
    args = parser.parse_args()

    budget = PromotionBudget(args.budget)
    if not budget.consume(args.session, args.family):
        print(json.dumps({
            "promotion_gate": "FAIL",
            "action": "CLOSE_MECHANISM_FAMILY",
        }))
        return 2

    metrics = json.loads(Path(args.metrics_json).read_text())
    decision = evaluate_promotion(**metrics)
    write_private_promotion_log(args.private_log, decision)
    print(json.dumps(sanitize_promotion_result(decision), sort_keys=True))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
