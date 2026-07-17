"""Policy primitives for robust autoresearch selection and sealed promotion.

Development metrics may be inspected and optimized. Promotion metrics are owner-only;
only ``sanitize_promotion_result`` output may be exposed to the research agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics

BASE_KEEP_THRESHOLD = 0.003
COMPLEXITY_PENALTY = 0.001
MIN_AFFECTED_TRADES = 30
MAX_GAIN_CONCENTRATION = 0.70
MIN_POSITIVE_FRACTION = 0.60


@dataclass(frozen=True)
class InternalDecision:
    passed: bool
    robust_delta: float
    required_delta: float
    positive_fraction: float
    gain_concentration: float
    affected_trades: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    okx_delta: float
    recent_sharpe_delta: float
    recent_return_delta: float
    recent_trade_ratio: float
    unseen_delta: float
    reasons: tuple[str, ...]


def _mad(values: list[float]) -> float:
    if not values:
        return math.inf
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def robust_environment_score(deltas: list[float], dispersion_weight: float = 0.5) -> float:
    """Median environment delta minus a robust dispersion penalty.

    A gain isolated to one environment has median zero and therefore cannot pass,
    even when its arithmetic mean is large.
    """
    if not deltas:
        return -math.inf
    return statistics.median(deltas) - dispersion_weight * _mad(deltas)


def complexity_adjusted_threshold(complexity_points: int) -> float:
    return BASE_KEEP_THRESHOLD + max(0, complexity_points) * COMPLEXITY_PENALTY


def _gain_concentration(deltas: list[float]) -> float:
    positive = [max(0.0, d) for d in deltas]
    total = sum(positive)
    if total <= 0:
        return 1.0
    return max(positive) / total


def evaluate_internal_candidate(
    baseline_scores: list[float],
    candidate_scores: list[float],
    affected_trades: int,
    complexity_points: int,
) -> InternalDecision:
    if len(baseline_scores) != len(candidate_scores) or not baseline_scores:
        raise ValueError("baseline_scores and candidate_scores must have equal non-zero length")

    deltas = [c - b for b, c in zip(baseline_scores, candidate_scores)]
    robust_delta = robust_environment_score(deltas)
    required = complexity_adjusted_threshold(complexity_points)
    positive_fraction = sum(d > 0 for d in deltas) / len(deltas)
    concentration = _gain_concentration(deltas)
    reasons: list[str] = []
    if robust_delta < required:
        reasons.append("robust_delta")
    if positive_fraction < MIN_POSITIVE_FRACTION:
        reasons.append("majority")
    if concentration > MAX_GAIN_CONCENTRATION:
        reasons.append("concentration")
    if affected_trades < MIN_AFFECTED_TRADES:
        reasons.append("affected_trades")

    return InternalDecision(
        passed=not reasons,
        robust_delta=robust_delta,
        required_delta=required,
        positive_fraction=positive_fraction,
        gain_concentration=concentration,
        affected_trades=affected_trades,
        reasons=tuple(reasons),
    )


def evaluate_promotion(
    *,
    okx_delta: float,
    recent_sharpe_delta: float,
    recent_return_delta: float,
    recent_trade_ratio: float,
    unseen_delta: float,
) -> PromotionDecision:
    """Owner-side veto gate. Metrics must never be fed back to the agent."""
    reasons: list[str] = []
    if okx_delta < 0:
        reasons.append("okx")
    if recent_sharpe_delta < -0.30:
        reasons.append("recent_sharpe")
    if recent_return_delta < -0.50:
        reasons.append("recent_return")
    if not 0.70 <= recent_trade_ratio <= 1.30:
        reasons.append("recent_trade_count")
    if unseen_delta < 0:
        reasons.append("unseen_token")
    return PromotionDecision(
        passed=not reasons,
        okx_delta=okx_delta,
        recent_sharpe_delta=recent_sharpe_delta,
        recent_return_delta=recent_return_delta,
        recent_trade_ratio=recent_trade_ratio,
        unseen_delta=unseen_delta,
        reasons=tuple(reasons),
    )


def sanitize_promotion_result(decision: PromotionDecision) -> dict[str, str]:
    if decision.passed:
        return {"promotion_gate": "PASS"}
    return {
        "promotion_gate": "FAIL",
        "action": "CLOSE_MECHANISM_FAMILY",
    }


class PromotionBudget:
    """Durable one-attempt-per-session and one-attempt-per-family budget."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"sessions": [], "families": []}
        return json.loads(self.path.read_text())

    def consume(self, session_id: str, family: str) -> bool:
        state = self._load()
        if session_id in state["sessions"] or family in state["families"]:
            return False
        state["sessions"].append(session_id)
        state["families"].append(family)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        return True


def write_private_promotion_log(path: str | Path, decision: PromotionDecision) -> None:
    """Append detailed owner-only metrics; never print them to agent stdout."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(decision), sort_keys=True) + "\n")
