"""Tests for candidate/promotion status parsing used by the outer loop."""

from promotion_runner import latest_candidate, resolved_candidate_row


def test_latest_candidate_requires_candidate_keep_after_last_promotion(tmp_path):
    p = tmp_path / "results.tsv"
    p.write_text(
        "h1\t0\t0\t0\t0\t0\t0\t0\t0\tpromoted_keep\told\n"
        "h2\t0\t0\t0\t0\t0\t0\t0\t0\tcandidate_keep\tnew\n"
    )
    assert latest_candidate(p) == ("h1", "h2")


def test_latest_candidate_returns_none_after_candidate_already_resolved(tmp_path):
    p = tmp_path / "results.tsv"
    p.write_text(
        "h1\t0\t0\t0\t0\t0\t0\t0\t0\tcandidate_keep\tnew\n"
        "h1\t0\t0\t0\t0\t0\t0\t0\t0\tdiscard_oos\tfailed\n"
    )
    assert latest_candidate(p) is None


def test_resolved_candidate_row_preserves_scores_and_changes_only_status(tmp_path):
    p = tmp_path / "results.tsv"
    p.write_text(
        "h1\t1\t2\t3\t4\t5\t6\t7\t8\t9\tcandidate_keep\tdesc\n"
    )
    row = resolved_candidate_row(p, "promoted_keep", "sealed pass")
    assert row.split("\t") == [
        "h1", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "promoted_keep", "sealed pass",
    ]
