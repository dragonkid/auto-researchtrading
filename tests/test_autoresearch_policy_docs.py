"""Static policy assertions for the autoresearch prompts and outer loop."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_normal_prompt_has_candidate_promotion_and_branch_cap():
    text = (ROOT / "program-stateless.md").read_text()
    for phrase in (
        "candidate_keep",
        "promoted_keep",
        "robust_composite",
        "hard maximum of 5 total steps",
        "CLOSE_MECHANISM_FAMILY",
        "recent_2026q1",
    ):
        assert phrase in text


def test_council_uses_same_sealed_policy():
    text = (ROOT / "program-council.md").read_text()
    for phrase in ("candidate_keep", "promoted_keep", "robust_composite", "one-shot"):
        assert phrase in text


def test_prompts_do_not_treat_temporal_windows_as_pure_regimes():
    normal = (ROOT / "program-stateless.md").read_text()
    council = (ROOT / "program-council.md").read_text()
    assert "temporal development environments" in normal
    assert "not pure" in normal
    assert "temporal development environments" in council
    assert "Do not propose calendar/environment detectors" in council


def test_outer_loop_runs_promotion_before_next_agent_round():
    text = (ROOT / "autoresearch.sh").read_text()
    promotion = text.index("promotion_runner.py")
    agent = text.index('Working directory: $PROJECT_DIR')
    assert promotion < agent
    assert ".autoresearch/private" in text


def test_outer_loop_restores_promoted_baseline_not_head_strategy():
    text = (ROOT / "autoresearch.sh").read_text()
    assert "active_baseline_hash" in text
    assert "git checkout -- strategy.py" not in text
