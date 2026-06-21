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
Focus on maximizing composite_score (= mean regime scores - 0.3*std). Stability test is ENABLED — applies to regimes with positive score.

## Session protocol

You run **independent experiments (exit after 5 consecutive architectural discards) + one exploration branch (no fixed depth, terminates on stagnation)** per session. Independent and branch budgets do not share slots.

Each experiment may be a **single-variable change OR a multi-variable structural change** — whichever is appropriate for the hypothesis. Multi-variable changes are especially encouraged for stability improvements, where architectural modifications (voter weighting, signal fusion, ensemble method changes) inherently require coordinated edits. You can **use insights from earlier experiments in this session to choose your next direction**, and after step 2 you may attempt **combination experiments** that merge two independently-validated improvements. **Exploration branches** (see below) let you iterate on a promising direction without reverting — use them when a regime-level signal is strong.

### Phase 1: Read context (once per session)

1. Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline -n 30`.
   **Baseline integrity check (do this first):** the current baseline is the most recent `keep` row in results.tsv. Verify it actually reflects HEAD: the keep row's commit hash should be an ancestor of HEAD (`git merge-base --is-ancestor <hash> HEAD`). If the most recent `session-summary` row declares a NEWER baseline ("NEW BASELINE <hash> <score>" / "SUPERSEDES ...") than the last `keep` row, the keep row was lost in a past rollback — trust the session-summary's declared baseline hash + score instead, and immediately re-append a proper `keep` row for that commit so the next session reads it correctly. Going forward, rollbacks use `git checkout -- strategy.py` (path-scoped) and never touch results.tsv, so keep rows will no longer disappear.
2. **Analyze**: What worked? What failed? What hasn't been tried? **Saturation check**: grep `results.tsv` descriptions for directions you're considering. If a direction has 10+ prior experiments with mostly `discard`, it's saturated — switch to something structurally different.
3. **Extract structural knowledge from prior sessions**: grep `results.tsv` for `session-summary` rows and descriptions containing "load-bearing", "CROSS-EXPERIMENT", or "CONCLUSION". Prior sessions have already proven which components are load-bearing (removing them regresses a regime) and which directions are dead ends. Do NOT re-attempt a removal/change that a prior session already confirmed catastrophic. Build on confirmed mechanism insights instead of rediscovering them.
4. **Incremental viability check**: count the most recent non-architectural experiments (descriptions without "architectural") and their keep rate. If the last 10+ non-architectural experiments are ALL discards, incremental changes are exhausted at this baseline — all experiments this session MUST be architectural (new code structure, new control flow, new data dependencies). If non-architectural keeps exist in recent history, you may attempt incremental experiments freely.

### PARAMETER-SPACE SATURATION RULE

If the last 5 branches (from results.tsv BRANCH SUMMARY lines) all operate within the SAME parameter space — position sizing (SIZE/FRAC/GATE_FLOOR/GATE_SCALE/CONF_MIN), sigmoid width, MIN_VOTES/FLIP_MIN_VOTES tuning — the parameter space is SATURATED. Your next experiment MUST change the DECISION ARCHITECTURE itself.

"Architectural" does NOT mean "multi-variable parameter change." It means the CODE STRUCTURE changes — new functions, new control flow, new data dependencies between components. If your change can be described as "adjust parameter X from A to B" or "combine parameters X+Y+Z at different values", it is NOT architectural.

### COMPONENT-SATURATION RULE (escalate to subsystem redesign)

Single-component changes (add one pressure source, remove one voter, gate one threshold) operate WITHIN the existing subsystem structure. If multiple recent sessions have confirmed (via session-summary rows) that every single component is load-bearing — i.e., the strategy sits in a tight local optimum where add/remove/modify of any individual component regresses at least one regime — then component-level exploration is saturated.

At that point a valid experiment may be a **subsystem redesign**: a coordinated rewrite of an entire decision subsystem (e.g., the exit-pressure fusion layer, the voter-aggregation mechanism, the position-sizing pipeline, the entry-admission gate) where multiple interdependent components are replaced together by a structurally different mechanism. This is still ONE experiment (one commit, one results.tsv row, fully revertable via `git revert`) — it is judged by the same keep/discard rules. The point is that some improvements are only reachable by changing several coupled components at once; a redesign that replaces a whole subsystem is a legitimate single hypothesis even though its diff is large.

Do NOT target a specific regime by name in the redesign (that would be regime-detection overfitting — forbidden). Redesign the subsystem's general mechanism; let the regime effects fall out of the backtest.

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
   - `composite_score` improved vs baseline by **at least +0.003** (absolute delta). Sub-noise improvements (<0.003) are `discard` even if positive — they are in-sample noise-floor and accumulate overfitting without meaningful signal. This threshold was restored 2026-06-20 after "any positive delta" let ~27 keeps through at avg +0.0008 each, none of which improved cross-exchange Sharpe.
   - `mean_score` improved vs baseline (average per-regime score must go up).
   - No regime score dropped more than 50% vs baseline (e.g., baseline 1.023 → floor is 0.512). If any regime breaches this gate, the experiment cannot be a direct KEEP — open an exploration branch to recover the regressed regime first.
   - No regime's `max_dd_pct` exceeds 10% (hard safety net).

   See the Scoring formula section below for the full `compute_score` formula.

   **Computing scores:** `regime_test.py` outputs `composite_score:`, `raw_composite:`, `mean_score:`, per-regime `score` / `raw_score` / `stability_factor` / `flip_streak_gate`, and other metrics directly.

   If keep: append a `keep` line with all per-regime scores. The new baseline for ALL subsequent experiments is now this keep. **CRITICAL: after a keep, you MUST compare the next experiment against this new keep's scores, not the session-start baseline.** Read the last `keep` row in results.tsv to get the current baseline values.
   If discard: **check exploration branch eligibility** (see below). If not eligible, roll back strategy.py ONLY with `git checkout <last-keep-commit> -- strategy.py && git commit -m "discard: <reason>" strategy.py` (find the last keep commit via `git log --oneline | grep -m1 keep` or the most recent `keep`/baseline row's commit hash). Then append a `discard` line to results.tsv. **NEVER use `git revert` or `git reset --hard`** — see the results.tsv durability rule below for why.

   **results.tsv DESCRIPTION HYGIENE (write only verifiable content):** the description field is BOTH a historical record AND a live input every future session reads as established fact — so an unverified guess written here propagates as truth. The current baseline's own `keep`-row description is inherited by every subsequent session. Therefore:
   - **DO record:** (a) reproducible measured numbers from THIS experiment (per-regime raw/score/stability deltas, Sharpe, trade counts), and (b) the keep/discard decision basis (which gate/criterion was met or breached).
   - **Causal attribution requires direct numerical evidence from THIS experiment.** A correlational observation (entry-side share, win rate, trade count) may NOT be promoted to a causal claim ("X is the drag/killer/cause of regime Y"). Win rate ≠ PnL contribution; entry-side share ≠ marginal Sharpe contribution. If you have not measured the causal link directly (e.g. direction-split realized PnL, a counterfactual backtest), state the observation as correlation only.
   - **Do NOT write speculative cross-experiment diagnoses** (e.g. naming a mechanism "the X killer" for a regime). These get inherited by the baseline keep row and misdirect every future session toward an unverified premise. Precedent: one baseline keep row promoted a correlational observation (entry-side share + win rate) into a named causal diagnosis without ever measuring the causal link by direction-split PnL; when finally measured it was wrong, and it had already misdirected ~20 sessions. State only what you measured; let later sessions form their own hypotheses from the raw numbers rather than inheriting a conclusion.

   **results.tsv DURABILITY RULE (root-cause fix for lost keep rows):** results.tsv must NEVER be touched by a rollback. Past sessions lost `keep` rows because the experiment commit bundled strategy.py + the results.tsv row together, and `git revert HEAD~N..HEAD` then rolled back BOTH — erasing the keep row and corrupting the baseline for the next session. To prevent this permanently:
   - Roll back ONLY strategy.py, using `git checkout <last-keep-commit> -- strategy.py` (restores the file to the keep state without rewriting history and without touching results.tsv). NEVER `git revert` an experiment — revert operates on whole commits and will take results.tsv rows with it.
   - results.tsv is strictly append-only and is never the target of any rollback. Every keep/discard/branch/session-summary row, once appended and committed, stays forever.
   - It is fine for one commit to contain both strategy.py and the results.tsv row, because rollback now uses `git checkout -- strategy.py` (path-scoped) instead of `git revert` (commit-scoped). The append for the rollback's own `discard` row is a fresh, separate append on top.

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
- **Failure (revert the branch)**: if the branch terminates without meeting keep criteria (either via 7 consecutive no-progress steps or because you decided to stop), roll back **strategy.py ONLY** to the most recent `keep` state: `git checkout <last-keep-commit> -- strategy.py && git commit -m "discard: branch <name> reverted to <keep-hash>" strategy.py`. Find `<last-keep-commit>` from the most recent `keep`/baseline row in results.tsv (its commit hash column). **CRITICAL: do NOT use `git revert HEAD~N..HEAD` — that is what historically destroyed keep rows, because the range includes the commits that appended them to results.tsv. `git checkout -- strategy.py` is path-scoped: it restores only the strategy file and never rewrites results.tsv or any other file.** results.tsv stays append-only — every prior keep/branch/discard row is preserved. Then append ONE summary `discard` line explaining the branch attempt and why it failed. NEVER `git reset --hard`. NEVER edit or delete existing rows in results.tsv — they are permanent records.
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

### Structural-exploration mode (for breaking structural rigidity)

When the strategy is at a **structural local optimum** — every incremental angle has been documented dead (alpha-trade wall, noise-trajectory wall, etc.) and conventional experiments keep hitting the same walls — the normal keep criteria (which require mean_score improvement and no regime dropping >50%) will reject ANY structural change on step 1, because rewriting a subsystem temporarily disrupts multiple regimes at once. This mode gives structural changes room to rebuild.

**Trigger conditions (ALL must be met):**
1. The current session has 3+ consecutive architectural discards (proving conventional exploration is saturated).
2. The experiment is a **subsystem rewrite**, not a gate/tweak/parameter change. "Subsystem rewrite" means replacing the CORE MECHANISM of one decision subsystem — e.g. replacing MAX-fusion with weighted-sum, replacing voter-count with voter-confidence-product, replacing the linear de-risk ramp with a completely different exit function form. Adding a new soft source, adding a gate, adjusting a parameter, or sustaining an existing shrink through a new path is NOT a subsystem rewrite — those are conventional experiments judged under normal criteria.
3. You explicitly mark the experiment description with prefix `STRUCTURAL_EXPLORATION:` and state (a) which subsystem is being rewritten, (b) what the new core mechanism is, (c) why the old mechanism is at its structural ceiling.

**Relaxed first-step criteria (ONLY for the opening STRUCTURAL_EXPLORATION experiment):**
- `mean_score` regression up to **−0.05** allowed (vs the hard "must improve" of normal keep).
- `composite_score` regression up to **−0.08** allowed (vs +0.003 improvement required for normal keep).
- The 50% per-regime gate is RETAINED (no regime may drop >50% — structural changes must not catastrophically break a regime).
- `max_dd_pct` > 10% hard cutoff RETAINED (safety net).
- These relaxations apply ONLY to step 1. Once the branch is open, subsequent steps are judged against normal keep criteria (or the branch's own progress — see below).

**Branch mechanics:**
- The structural-exploration experiment auto-opens a branch (does NOT count as the session's one exploration branch for normal-experiment budget — it IS the branch).
- Budget: **15 steps** (not 7). Historical structural successes took 14 steps to rebuild balance.
- Stagnation guard: **10 consecutive no-progress steps** (raw_composite delta ≤ 0 vs previous step) → terminate. This is looser than the 7-step guard for normal branches, because structural rebuilds have plateaus.
- Progress requirement: raw_composite must be **monotonically non-decreasing over any 3-step window** (a step may regress, but 3 consecutive regressions = terminate). This allows exploration detours while preventing drift.
- Success: if at any step the FULL normal keep criteria are met vs the original baseline (composite +0.003, mean improved, all gates pass), it's a real `keep`.
- Failure: if the branch terminates without meeting keep criteria, revert strategy.py ONLY to the last keep, record `STRUCTURAL_EXPLORATION` discard summary.

**Frequency limit:** at most ONE structural-exploration branch per session. If it reverts, the session ends — do not open a second one.

**What this is NOT:**
- NOT a license to do reckless rewrites. The 50% regime gate and 10% DD cutoff still apply.
- NOT for parameter sweeps. If your change can be described as "adjust X from A to B", it's a conventional experiment.
- NOT for adding components. Adding a new soft source / gate / data dep is conventional, even if "architectural".

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

- `score` = composite_score (mean - 0.3*std)
- `mean_score` = average across 4 regimes
- `std_score` = std across regimes (lower = more consistent)
- `bull_2021 / crash_bear / sideways / rally_2024` = per-regime scores extracted from lines matching `^regime_<name>_score:` in run.log (e.g., `regime_bull_2021_score: 27.123456` → store `27.12`)
- Append one line per experiment. Use the short commit hash, or `-` for discarded.

## Scoring formula

Each regime is scored via `compute_score()`, then combined:

```
score = signal_quality × sample_factor × dd_gate × streak_gate × return_reward

signal_quality = log(1 + max(sharpe, 0))
sample_factor = sqrt(min(num_trades / 50, 1))
dd_gate = 1/(1+DD%) × exp(-max(0,DD%-8)/2)   # soft@8%, scale=2 (0-8% mild, 8%+ steep)
streak_gate = exp(-max_consecutive_losses / 30)
return_reward = log(1 + min(calmar, 10)/10 + 1)   # calmar = APY/MaxDD; risk-adjusted, stops leverage farming

Hard cutoffs: <10 trades → -999, >10% drawdown → -999, lost >15% → -999

Composite score = mean(regime_scores) - 0.3 * std(regime_scores)
```

**vol_gate removed (2026-06-19):** the former `vol_gate = 1/(1+return_volatility)` was a double penalty — `return_volatility` (= `std(returns)*sqrt(8760)`) is the same std already in Sharpe's denominator. Measured across all historical keeps it was a near-constant 0.970-0.985 dampener (<1.4% spread between regimes), never changing ranking, only shifting all scores uniformly by ~2.6%. At equal Sharpe it preferred lower-vol (= lower-return) strategies — a misincentive. Sharpe is now the sole vol-adjustment.

**std penalty lowered (2026-06-19):** `k` lowered 0.5 → 0.3. At k=0.5, 72% of composite gains came from std reduction; agent over-optimized for consistency at the expense of mean return. k=0.3 keeps consistency reward (prevents abandoning weakest regime) while giving mean-improvement room. Pure k=0 was rejected (rewards 3-strong-1-weak fragility).

**return_reward added (2026-06-20, revised 2026-06-21):** `log(1+min(calmar,10)/10+1)` factor (range 0.693-1.0), where calmar = APY/MaxDD (risk-adjusted return). The original absolute-APY form let the agent farm score by raising LEVERAGE_K — leverage scales APY and DD proportionally (APY/DD stays flat, Sharpe stays flat = pure scale-up, no signal improvement). The risk-adjusted form stops this: leverage → calmar unchanged → return_reward unchanged → no score gain. Genuine signal-quality improvements (Sharpe up at same DD → APY/DD up) are rewarded 3x more than under absolute-APY (verified: +0.029 vs +0.009 for Sharpe 1.89→2.0). Uses APY not raw total_return because regime windows differ in length (bull 273d, crash 426d, sideways 365d, rally 366d).

**dd_gate soft_start raised 5→8, scale 10→2 (2026-06-20):** the 5-8% DD range was over-penalized (5→8% lost 28% under soft@5), discouraging the agent from accepting 5-8% DD to capture more return. New curve: 0-8% only the mild 1/(1+DD) base (8% → 0.926, just -7.4%); 8%+ steep exp penalty (scale=2: 9% → 0.557, 10% → 0.334). Hard cutoff at 10% unchanged. Real strategies sit at DD 0.45-1.70% (far below 8%), so this mainly opens up the 5-8% range for return-seeking experiments.

Note: trades incur real fees (5bps taker + 1bp slippage) in the backtest, so transaction cost is already reflected in Sharpe. There is no separate turnover penalty — if higher trade frequency raises post-fee Sharpe, that is genuine alpha and is rewarded. `sample_factor` only enforces a minimum sample size (50 trades); it does not penalize high frequency.

Multiplicative structure: any dimension being terrible collapses the entire score.
The DD penalty is a smooth exponential — no cliff at any specific DD level. DD 5%→no penalty, 8%→0.74x, 10%→0.55x.
The composite rewards strategies that perform **consistently across all market conditions**.

Search regimes (4 non-overlapping periods):
- bull_2021: 2021-01 ~ 2021-10 (bull market)
- crash_bear: 2021-11 ~ 2022-12 (Luna/FTX crash + deep bear)
- sideways: 2023-01 ~ 2023-12 (sideways recovery)
- rally_2024: 2024-01 ~ 2024-12 (ETF + election rally)

## Primary Objective: Maximize composite_score

Stability penalty (applied per-regime when score > 0, uses AR(1) correlated noise test):

- Continuous linear ramp: stability_factor = clamp((stability - 0.50) / (0.80 - 0.50), 0, 1)
- stability ≤ 0.50 → factor 0.0 (effectively rejected)
- stability 0.50–0.80 → factor ramps linearly 0.0 → 1.0
- stability ≥ 0.80 → factor 1.0 (no penalty)

Every stability value has a usable gradient — improving stability always raises the factor, no tier cliffs.

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
- Hard constraints (see Keep/discard rules above): no regime score drop > 50% vs baseline, no regime MaxDD > 10%

**Score decomposition fields** (output by `regime_test.py` for each regime):
- `regime_X_raw_score` — score BEFORE stability and flip_streak penalties
- `regime_X_stability_factor` — multiplier applied by stability penalty (1.0 = no penalty, <1.0 = penalized)
- `regime_X_flip_streak_gate` — multiplier applied by flip streak penalty (1.0 = no penalty)
- Final score = raw_score × stability_factor × flip_streak_gate

**Use raw_score to diagnose WHERE the problem is.** If a regime has high raw_score but low final score, the problem is noise sensitivity (low stability), not strategy quality. Focus on making signals more robust (smooth thresholds, wider margins from decision boundaries) rather than changing the signal logic itself. If raw_score is low, the strategy genuinely underperforms in that regime.

## Stability constraints (guard rails, not objectives)

Design strategies with smooth thresholds to minimize stability penalty.

**Do NOT use open price as a "stable" signal source.** The noise test only perturbs close (then adjusts high/low). Open appears noise-immune but this is an artifact of the test methodology, not a real property. In live trading, open is equally noisy.
**HL2 in noise test.** HL2=(high+low)/2 is tested with AR(1) correlated noise (high: std 4bps, low: std 5bps — empirically calibrated 2026-06 to demeaned cross-source random dispersion, see noise_test.py header). HL2-based signals have comparable noise exposure to close-based signals. No discount needed.

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
