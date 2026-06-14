# autotrader — multi-step experiment session

Autonomous trading strategy research on Hyperliquid perpetual futures.
You will run **independent experiments + 1 exploration branch** per session. The outer shell script invokes you once per "round" — each round is a coherent research arc. The session has two phases: independent exploration (find a promising direction) and branch deepening (iterate on it without reverting).

## Context

This project adapts Karpathy's autoresearch pattern for trading strategy discovery.
The owner has existing production strategies designed for tick-level market making (20-second intervals). Those strategies underperform when ported to hourly directional trading on this backtest harness.

Your job: **improve the current strategy in `strategy.py`** through iterative experimentation within a single session.

## What you CAN do

- Modify `strategy.py` — this is the only file you edit. Everything is fair game.

## What you CANNOT do

- Modify `prepare.py`, `backtest.py`, `regime_test.py`, `noise_test.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at holdout data (2025-01 onwards).

### Phase priority rule
Focus on maximizing composite_score (= mean regime scores - 0.5*std). Stability test is ENABLED — applies to regimes with positive score.

## Session protocol

You run **independent experiments (exit after 5 consecutive architectural discards) + one exploration branch (no fixed depth, terminates on stagnation)** per session. Independent and branch budgets do not share slots.

Each experiment may be a **single-variable change OR a multi-variable structural change** — whichever is appropriate for the hypothesis. Multi-variable changes are especially encouraged for stability improvements, where architectural modifications (voter weighting, signal fusion, ensemble method changes) inherently require coordinated edits. You can **use insights from earlier experiments in this session to choose your next direction**, and after step 2 you may attempt **combination experiments** that merge two independently-validated improvements. **Exploration branches** (see below) let you iterate on a promising direction without reverting — use them when a regime-level signal is strong.

### Phase 1: Read context (once per session)

1. Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline -n 30`.
2. **Analyze**: What worked? What failed? What hasn't been tried? **Saturation check**: grep `results.tsv` descriptions for directions you're considering. If a direction has 10+ prior experiments with mostly `discard`, it's saturated — switch to something structurally different.
3. **Incremental viability check**: count the most recent non-architectural experiments (descriptions without "architectural") and their keep rate. If the last 10+ non-architectural experiments are ALL discards, incremental changes are exhausted at this baseline — all experiments this session MUST be architectural (new code structure, new control flow, new data dependencies). If non-architectural keeps exist in recent history, you may attempt incremental experiments freely.

### PARAMETER-SPACE SATURATION RULE

If the last 5 branches (from results.tsv BRANCH SUMMARY lines) all operate within the SAME parameter space — position sizing (SIZE/FRAC/GATE_FLOOR/GATE_SCALE/CONF_MIN), sigmoid width, MIN_VOTES/FLIP_MIN_VOTES tuning — the parameter space is SATURATED. Your next experiment MUST change the DECISION ARCHITECTURE itself.

"Architectural" does NOT mean "multi-variable parameter change." It means the CODE STRUCTURE changes — new functions, new control flow, new data dependencies between components. If your change can be described as "adjust parameter X from A to B" or "combine parameters X+Y+Z at different values", it is NOT architectural.

### Phase 2: Experiment loop

For each experiment:

**Exit rule:**
- 5 consecutive discards, all architectural → stop session.
- 5 consecutive discards, not all architectural → continue with architectural experiments until you've attempted at least 5 architectural, then stop.
- A `keep` resets the consecutive discard count.

1. **Propose one change**: Pick one specific, testable idea. After your first experiment in this session, you may base your next idea on the regime-level insights you just observed (e.g., "Exp A showed sideways +0.24 but crash -1.44 — try a different condition that protects crash").
2. **Implement**: Edit `strategy.py` with your change.
3. **Commit**:
   ```
   git commit -m "exp: <short description of what you changed>" \
              -m "Hypothesis: <1-2 sentences on the mechanism>" \
              -m "Expected: <which regime(s) should benefit>"
   ```
4. **Backtest**: `uv run regime_test.py > run.log 2>&1`
5. **Parse results**: `grep "^composite_score:\|^raw_composite:\|^mean_score:\|^std_score:\|^regime_\|^min_stability:" run.log`
   - Key fields per regime: `regime_X_score` (final), `regime_X_raw_score` (before penalties), `regime_X_stability_factor`, `regime_X_flip_streak_gate`
6. **Record** (mandatory — do NOT skip): Append one row to `results.tsv` for EVERY experiment. This is not optional.

   **Keep/discard rules:**

   An experiment qualifies as `keep` if ALL of the following are met:
   - `composite_score` improved vs baseline (any positive delta counts — strategy is currently net-negative, every improvement matters).
   - `mean_score` improved vs baseline (average per-regime score must go up).
   - No regime's `max_dd_pct` exceeds 95% (hard safety net only).

   See the Scoring formula section below for the full `compute_score` formula.

   **Computing scores:** `regime_test.py` outputs `composite_score:`, `raw_composite:`, `mean_score:`, per-regime `score` / `raw_score` / `stability_factor` / `flip_streak_gate`, and other metrics directly.

   If keep: append a `keep` line with all per-regime scores. The new baseline for ALL subsequent experiments is now this keep. **CRITICAL: after a keep, you MUST compare the next experiment against this new keep's scores, not the session-start baseline.** Read the last `keep` row in results.tsv to get the current baseline values.
   If discard: **check exploration branch eligibility** (see below). If not eligible, run `git revert --no-edit HEAD`, append a `discard` line. NEVER use `git reset --hard`.

### Exploration branches (iterate before reverting)

When an experiment does NOT meet full keep criteria but shows architectural promise, you MAY keep the change and open an **exploration branch** instead of reverting. This lets you iterate on a promising direction — fixing weak regimes, tuning the new mechanism — before the final keep/discard decision.

**Entry conditions (any ONE is sufficient):**
- At least one regime score improved (any positive delta on a per-regime score)
- The experiment introduces a fundamentally new mechanism (not a parameter tweak)

You do NOT need the experiment to "almost pass" — the point is to allow bold architectural exploration. If you can articulate WHY the regression happened and HOW a follow-up change could fix it, that's sufficient justification to open a branch.

**Rules:**
- **Justification required**: State explicitly (1) what the new architecture does differently, (2) which regime regressed and why, (3) your hypothesis for fixing it in the next step.
- **No fixed max depth**: the branch continues as long as you're making progress. Branch budget is INDEPENDENT of independent exploration — they don't share slots. If 7 consecutive branch steps show no improvement (raw_composite delta ≤ 0 vs the previous branch step), terminate the branch early. There is no other depth ceiling — a branch that keeps improving can iterate as many steps as needed.
- **Each iteration**: commit normally, run regime_test, record in results.tsv with prefix `branch:` (e.g., `branch: fix rally regression from linreg slope gate`). Each branch step may freely modify strategy.py — you're iterating on the new architecture, not just tweaking one parameter.
- **Success (keep the branch)**: if at any point during the branch the FULL keep criteria are met vs the **current baseline** (the most recent `keep` row in results.tsv), it's a real `keep`. Record as `keep` and update baseline. All subsequent experiments compare against this new keep.
- **Failure (revert the branch)**: if the branch terminates without meeting keep criteria (either via 7 consecutive no-progress steps or because you decided to stop), revert ALL branch commits back to the most recent `keep` in results.tsv. **CRITICAL: You MUST only revert your own experiment commits. Run `git log --oneline` and count ONLY commits with messages starting with "exp:" or "branch step". Use `git revert --no-edit HEAD~N..HEAD` where N = that count. NEVER revert commits with "feat:", "fix:", or "doc:" prefixes — those are infrastructure changes by the project owner. If such commits are interleaved, revert your commits individually instead of using a range. NEVER modify or delete existing `keep` rows in results.tsv — they are permanent records.** Record ONE summary `discard` line explaining the branch attempt and why it failed.
- **One branch per session**: you may only open one exploration branch per round. After a branch concludes (keep or revert), the session ends. The next round starts fresh with full context from results.tsv. Note: if branch reverts and you still have unused independent-exploration slots, the session still ends — branch revert is a strong enough signal that the round should conclude and reset with fresh context.
- **Exit rule interaction**: branch experiments count as architectural if the opening experiment was architectural.
- **Intermediate regression is OK**: within a branch, stability may temporarily drop further as you restructure. Only the FINAL state of the branch is judged against keep criteria vs original baseline. Don't abandon a branch just because step 2 made things worse — you have as many steps as needed to recover, until the 7-step stagnation guard triggers.

**Typical session shape:**
- Independent exploration: normal discard/revert cycle while searching for a promising direction
- Once a direction shows promise → open exploration branch (separate budget, no fixed depth)
- Branch concludes (keep or revert) → session ends

**Example flow (success):**
1. Exp 1 (independent): smoothed equity → flat, discard
2. Exp 2 (independent): linreg slope deadzone gate → bull +0.47, rally -0.0003 → **open exploration branch**
3. Branch step 2: add rally-protective condition → rally fixed, sideways -0.001 → continue
4. Branch step 3: tune sideways parameters → sideways fixed, crash -0.001 → continue
5. Branch step 4: crash-protective adjustment → all regimes pass → **KEEP** (vs original baseline)

**Example flow (failure with full exploration):**
1-3. Exp 1-3 (independent): three different directions, all discard
4. Exp 4 (independent): new exit mechanism → crash +0.5, sideways -0.010 → **open exploration branch**
5-13. Branch steps 2-10: iterate on sideways fixes (fast exit path, regime-specific logic, hybrid exit, etc.) → sideways stays at -0.002 throughout
14. After 7 consecutive no-progress steps → revert all branch commits, record discard, session ends

7. **Decide next step**:
   - If you have a clear follow-up insight from the regime breakdown → continue to next experiment.
   - If you've found a keep and want to try combining it with another idea → continue.
   - If you've exhausted your ideas or hit the exit rule (5 consecutive architectural discards) without opening a branch → exit. If a branch concluded → exit.

### Phase 3: Combination experiments (optional)

After running at least 2 independent experiments and observing their regime-level effects, you MAY attempt a combination:
- **Prerequisite**: at least one of the component ideas showed a promising signal (e.g., improved a target regime even if overall was discard due to regression elsewhere).
- **Combination = applying two ideas together** in a single `strategy.py` edit. Still counts as one experiment, still gets one results.tsv row.
- **Attribution**: in the commit message, reference which prior experiments you're combining (e.g., "Combines the EMA spread filter (sideways +0.24) with vol_ratio gating (crash protection)").
- **Same keep/discard rules apply** — no special treatment for combinations.

### Session end

After your last experiment (or when the exit rule triggers), exit. The outer loop will invoke you again for the next round.

## Results TSV format

**results.tsv is append-only.** Never delete, modify, or rewrite existing rows. Only append new rows at the end. **CRITICAL: never use write/overwrite on results.tsv — use shell `echo >> results.tsv` or equivalent append operation. If you need to fix a row, append a corrected row with a note; do NOT edit or remove the original. Keep rows are permanent historical records.**

New schema (10 columns, tab-separated):
```
commit	score	mean_score	std_score	bull_2021	crash_bear	sideways	rally_2024	status	description
```

Legacy rows (6 columns) may remain in the file for historical reference but are ignored when computing the per-regime baseline. Always append new rows using the 10-column schema.

- `score` = composite_score (mean - 0.5*std)
- `mean_score` = average across 4 regimes
- `std_score` = std across regimes (lower = more consistent)
- `bull_2021 / crash_bear / sideways / rally_2024` = per-regime scores extracted from lines matching `^regime_<name>_score:` in run.log (e.g., `regime_bull_2021_score: 27.123456` → store `27.12`)
- Append one line per experiment. Use the short commit hash, or `-` for discarded.

## Scoring formula

Each regime is scored via `compute_score()`, then combined:

```
score = signal_quality × sample_factor × dd_gate × turnover_gate × vol_gate × streak_gate

signal_quality = log(1 + max(sharpe, 0))
sample_factor = sqrt(min(num_trades / 50, 1))
dd_gate = 1/(1 + DD%) × exp(-max(0, DD%-5)/10)
turnover_gate = 1 / (1 + trades_per_day / 10)
vol_gate = 1 / (1 + return_volatility)
streak_gate = exp(-max_consecutive_losses / 30)

Hard cutoffs: <10 trades → -999, >10% drawdown → -999, lost >15% → -999

Composite score = mean(regime_scores) - 0.5 * std(regime_scores)
```

Multiplicative structure: any dimension being terrible collapses the entire score.
The DD penalty is a smooth exponential — no cliff at any specific DD level. DD 5%→no penalty, 8%→0.74x, 10%→0.55x.
The turnover gate penalizes trading frequency: 5 tpd→0.67, 10→0.50, 40→0.20.
The composite rewards strategies that perform **consistently across all market conditions**.

Search regimes (4 non-overlapping periods):
- bull_2021: 2021-01 ~ 2021-10 (bull market)
- crash_bear: 2021-11 ~ 2022-12 (Luna/FTX crash + deep bear)
- sideways: 2023-01 ~ 2023-12 (sideways recovery)
- rally_2024: 2024-01 ~ 2024-12 (ETF + election rally)

## Primary Objective: Maximize composite_score

Stability penalty (applied per-regime when score > 0, uses AR(1) correlated noise test):

- stability < 0.70 → factor = (stab/0.80) × 0.50
- stability 0.70–0.79 → factor = (stab/0.80) × 0.75
- stability ≥ 0.80 → factor = stab/0.80, capped at 1.0

### Diagnostic-first approach (optional, recommended for new sessions)

If results.tsv has < 10 entries or you haven't seen flip-rate data from prior sessions, diagnose noise sensitivity first:
1. Read `strategy.py` and identify voters/signals using hard thresholds on price-derived values
2. Estimate how far typical signal values sit from decision boundaries
3. Run the flip-rate diagnostic (see reference below) to quantify per-voter noise sensitivity

If results.tsv already contains diagnostic insights from prior sessions (grep for "flip rate", "noise", "voter sensitivity"), you may skip re-running diagnostics and proceed directly to experimentation.

### How to evaluate experiments
- Check `composite_score` and `mean_score` — both must improve vs baseline
- Check per-regime scores — check both `raw_score` and final `score` to understand penalty impact
- Check `regime_X_flip_count` and `regime_X_flip_pnl` — these are significant cost contributors
- The ONLY hard constraint is: no regime MaxDD > 95%

**Score decomposition fields** (output by `regime_test.py` for each regime):
- `regime_X_raw_score` — score BEFORE stability and flip_streak penalties
- `regime_X_stability_factor` — multiplier applied by stability penalty (1.0 = no penalty, <1.0 = penalized)
- `regime_X_flip_streak_gate` — multiplier applied by flip streak penalty (1.0 = no penalty)
- Final score = raw_score × stability_factor × flip_streak_gate

**Use raw_score to diagnose WHERE the problem is.** If a regime has high raw_score but low final score, the problem is noise sensitivity (low stability), not strategy quality. Focus on making signals more robust (smooth thresholds, wider margins from decision boundaries) rather than changing the signal logic itself. If raw_score is low, the strategy genuinely underperforms in that regime.

## Stability constraints (guard rails, not objectives)

Design strategies with smooth thresholds to minimize stability penalty.

**Do NOT use open price as a "stable" signal source.** The noise test only perturbs close (then adjusts high/low). Open appears noise-immune but this is an artifact of the test methodology, not a real property. In live trading, open is equally noisy.
**HL2 in noise test.** HL2=(high+low)/2 is tested with AR(1) correlated noise (high: std 8bps, low: std 12bps). HL2-based signals have comparable noise exposure to close-based signals. No discount needed.

**Hard binary regime switches are forbidden.** A strategy that detects "current regime = sideways" and switches to a different code path creates boundary noise that destroys stability (the switch point itself is noise-sensitive). More importantly, the AR(1) correlated noise test CANNOT detect regime-detection overfitting — a smooth regime classifier (e.g., 100-bar volatility average) will pass stability tests while being perfectly overfit to the 4 known backtest regimes. This is the one form of overfitting our test harness does not catch.

**Required approach:** use continuous/gradual transitions. Parameters should scale smoothly with regime indicators rather than switching between discrete modes. The transition must be gradual enough that there is no identifiable "switch point" to overfit.

### Flip-rate diagnostic (reference)

Run this once per session if you need to identify the noisiest voter. Adapt the skeleton to instrument your actual voters:

```python
# diagnostic: per-voter flip rate under AR(1) correlated noise
# Instrument vote computation to extract individual voter booleans
# Compare: voter_clean[i] != voter_perturbed[i] → flip
# Report table: Voter | Flip rate | Bars affected
# Target: identify voters with flip_rate > 0.03 (3%)
```

Do NOT hardcode "proven ineffective" conclusions here — read results.tsv each round to discover what has been tried. Only methodology-level blind spots (like open price artifact and HL2 overestimation above) belong in this file.

**funding_rate data is all zeros in the current dataset.** The Binance spot OHLCV data used for backtesting does not include funding rates (funding_rate column = 0.0 for all bars). Any strategy component that reads `bd.history["funding_rate"]` will receive constant zeros — it cannot provide a real signal. A "voter" based on funding_rate=0 will produce a fixed constant bias (not a data-responsive signal). Do not use funding_rate as a signal source until real per-exchange funding data is integrated.

## Data available

- BTC, ETH, SOL hourly OHLCV + funding rates
- History buffer: last 500 bars via `bar_data[symbol].history` DataFrame
- Columns: timestamp, open, high, low, close, volume, funding_rate

## Overfitting hygiene

These rules exist because this branch has accumulated hundreds of experiments. At that scale, selection bias dominates — any single +0.01 improvement is statistically fragile, and the search regimes themselves are effectively in-sample. Violating these rules causes meta-overfit that the regime-regression gate cannot catch.

- **Do NOT read commit bodies of prior experiments.** Use only `git log main..HEAD --oneline` (subjects only) and `results.tsv`. Commit bodies hold past hypotheses — reading them narrows your proposal space to "slight variants of what was tried," amplifying selection bias.
- **Do NOT base your idea on holdout findings.** The holdout (2025-01+) is never read by you. But also: if you notice phrasing in this prompt or in code comments that references specific holdout events (e.g., "the single-hour DD on 2025-03-02"), treat those as off-limits — do NOT design a rule targeting them. Any insight derived from holdout data is information leakage, regardless of who performed the analysis.
- **Prefer mechanism-backed changes over parameter sweeps.** At this experiment count, moving a parameter by 10% and finding +0.01 is likely noise. Adding a new signal with a clear mechanism, or removing a component that should be redundant, is a more honest trial.
- **Respect saturation signals.** If step 2 finds a direction has 10+ prior tries with mostly discards, do not submit another tuning variant.

## Guidelines

- **One idea per experiment** (each experiment tests one hypothesis in isolation — but that hypothesis may require coordinated multi-variable edits if it's architectural). The exception is Phase 3 combination experiments, which explicitly merge two independently-validated improvements.
- If you have no ideas, re-read `strategy.py` carefully and look for parameters to tune or signals to add/remove.
- All else equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code is not worth it.
- **Simplification experiments are as valuable as additions.** Try removing a voter, disabling a sizing multiplier, or deleting dead code. If the score holds or improves, keep the simpler version. Complexity has a hidden cost: it hurts out-of-sample generalization.
- **Do NOT inline constants or compress code cosmetically.** Named constants improve readability. Inlining a named constant into its usage site is NOT a valid simplification.
- **Use your session context wisely.** Your advantage over single-experiment mode is that you can observe patterns across experiments within this session. If experiment 1 shows sideways +0.24 but crash -1.44, experiment 2 should try to preserve the sideways gain while protecting crash — not start from scratch on an unrelated direction.
- Do NOT ask for confirmation. You are fully autonomous for this session.
