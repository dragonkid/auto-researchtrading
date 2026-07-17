#!/usr/bin/env python3
"""CLI for the inspectable development-data candidate decision."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from research_policy import evaluate_internal_candidate

_SCORE_RE = re.compile(r"^regime_(.+)_score:\s+([-+0-9.eE]+)$")


def parse_regime_scores(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        match = _SCORE_RE.match(line.strip())
        if match:
            scores[match.group(1)] = float(match.group(2))
    return scores


def count_changed_trade_events(baseline: list[tuple], candidate: list[tuple]) -> int:
    """Count inserted/deleted/replaced events in two ordered trade sequences."""
    matcher = SequenceMatcher(a=baseline, b=candidate, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += (i2 - i1) + (j2 - j1)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_log")
    parser.add_argument("candidate_log")
    parser.add_argument("--affected-trades", type=int, required=True)
    parser.add_argument("--complexity-points", type=int, required=True)
    args = parser.parse_args()

    base = parse_regime_scores(args.baseline_log)
    cand = parse_regime_scores(args.candidate_log)
    names = sorted(set(base) & set(cand))
    if not names:
        raise SystemExit("no matching regime scores")
    decision = evaluate_internal_candidate(
        [base[n] for n in names], [cand[n] for n in names],
        args.affected_trades, args.complexity_points,
    )
    print(json.dumps({
        "passed": decision.passed,
        "robust_delta": decision.robust_delta,
        "required_delta": decision.required_delta,
        "positive_fraction": decision.positive_fraction,
        "gain_concentration": decision.gain_concentration,
        "affected_trades": decision.affected_trades,
        "reasons": decision.reasons,
    }, sort_keys=True))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
