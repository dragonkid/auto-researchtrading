#!/usr/bin/env python3
"""Run sealed promotion validation in an isolated worktree.

This is owner-side infrastructure. It never exposes detailed metrics on stdout;
stdout is exactly the sanitized PASS/FAIL object from promotion_gate.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RESOLVED = {"promoted_keep", "discard_oos"}


def _rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        cols = line.split("\t")
        if len(cols) >= 10:
            rows.append(cols)
    return rows


def latest_candidate(path: str | Path) -> tuple[str, str] | None:
    rows = _rows(Path(path))
    candidate_idx = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][-2] == "candidate_keep":
            candidate_idx = i
            break
    if candidate_idx is None:
        return None
    if any(r[-2] in RESOLVED for r in rows[candidate_idx + 1:]):
        return None
    baseline = None
    for r in reversed(rows[:candidate_idx]):
        if r[-2] in {"promoted_keep", "keep"}:
            baseline = r[0]
            break
    if baseline is None:
        baseline = rows[candidate_idx][0]
    return baseline, rows[candidate_idx][0]


def previous_baseline(path: str | Path) -> str | None:
    pair = latest_candidate(path)
    return pair[0] if pair else None


def resolved_candidate_row(path: str | Path, status: str, description: str) -> str:
    """Clone the pending candidate row, replacing only status/description."""
    if status not in RESOLVED:
        raise ValueError(f"invalid resolved status: {status}")
    rows = _rows(Path(path))
    for row in reversed(rows):
        if row[-2] == "candidate_keep":
            return "\t".join(row[:-2] + [status, description])
    raise ValueError("no pending candidate row")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--session", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--validator", default="scripts/promotion_validate.py")
    args = parser.parse_args()

    pair = latest_candidate(args.results)
    if pair is None:
        print(json.dumps({"promotion_gate": "NONE"}))
        return 0
    baseline, candidate = pair

    root = Path.cwd()
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-promotion-"))
    metrics_file = tmp / "metrics.json"
    private_log = root / ".autoresearch/private/promotion.jsonl"
    budget = root / ".autoresearch/promotion_budget.json"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(tmp / "repo"), candidate],
            cwd=root, check=True, stdout=subprocess.DEVNULL,
        )
        validator = root / args.validator
        run = subprocess.run(
            [sys.executable, str(validator), baseline, candidate, str(metrics_file)],
            cwd=tmp / "repo", capture_output=True, text=True,
        )
        if run.returncode != 0:
            print(json.dumps({
                "promotion_gate": "FAIL",
                "action": "CLOSE_MECHANISM_FAMILY",
            }))
            return 1
        gate = subprocess.run(
            [sys.executable, str(root / "promotion_gate.py"),
             "--session", args.session, "--family", args.family,
             "--metrics-json", str(metrics_file), "--budget", str(budget),
             "--private-log", str(private_log)],
            cwd=root, text=True, capture_output=True,
        )
        print(gate.stdout.strip())
        return gate.returncode
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tmp / "repo")],
                       cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
