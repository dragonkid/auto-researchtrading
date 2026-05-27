# autotrader — multi-step experiment session

Autonomous trading strategy research on Hyperliquid perpetual futures.
You will run **up to 10 experiments** in this session, building on your findings iteratively. The outer shell script invokes you once per "round" — each round is a coherent research arc. Use the extra slots for exploration branches — when a direction shows promise, iterate on it rather than reverting immediately.

## Context

This project adapts Karpathy's autoresearch pattern for trading strategy discovery.
The owner has existing production strategies designed for tick-level market making (20-second intervals). Those strategies underperform when ported to hourly directional trading on this backtest harness.

Your job: **improve the current strategy in `strategy.py`** through iterative experimentation within a single session.

## What you CAN do

- Modify `strategy.py` — this is the only file you edit. Everything is fair game.

## What you CANNOT do

- Modify `prepare.py`, `backtest.py`, `regime_test.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at holdout data (2025-01 onwards).

### Phase priority rule
When min_stability < 0.90: at least 3 of 5 experiments MUST target stability (use stability keep path). Remaining 2 may target composite.

## Session protocol

You run up to 10 experiments per session. Each experiment may be a **single-variable change OR a multi-variable structural change** — whichever is appropriate for the hypothesis. Multi-variable changes are especially encouraged for stability improvements, where architectural modifications (voter weighting, signal fusion, ensemble method changes) inherently require coordinated edits. You can **use insights from earlier experiments in this session to choose your next direction**, and after step 2 you may attempt **combination experiments** that merge two independently-validated improvements. **Exploration branches** (see below) let you iterate on a promising direction without reverting — use them when a regime-level signal is strong.

### Phase 1: Read context (once per session)

1. Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline -n 30`.
2. **Analyze**: What worked? What failed? What hasn't been tried? **Saturation check**: grep `results.tsv` descriptions for directions you're considering. If a direction has 10+ prior experiments with mostly `discard`, it's saturated — switch to something structurally different.

### ESCALATION RULE (multi-variable architectural change)

If the last 3+ stability-targeted experiments in `results.tsv` all achieved < +0.005 stability gain, single-parameter threshold tuning is **exhausted**. Your next stability experiment MUST be a **multi-variable architectural change** — e.g., replace binary voting with weighted ensemble, restructure exit logic, change signal fusion method, add hysteresis layers, or redesign position sizing. Do NOT repeat incremental threshold/parameter tweaks that have been proven to plateau.

**This rule is enforced by the exit rule below:** you cannot exit a session without having attempted at least 2 architectural changes. The escalation rule checks BOTH the current session's discards AND the tail of `results.tsv` from prior sessions — if the last 3+ results across sessions are sub-threshold discards, escalation is already active from experiment 1.

### Phase 2: Experiment loop

For each experiment:

**Exit rule (escalation-gated):**
- 3 consecutive discards → your next experiment MUST be a **multi-variable architectural change** (layers 3-4 from the stability methodology: hysteresis, confidence margin, weighted ensemble, abstain zone, etc.). Single-parameter tweaks are forbidden after 3 discards.
- 5 consecutive discards → stop session ONLY IF at least 2 of those 5 were architectural changes (multi-variable, touching decision boundaries or signal fusion). If fewer than 2 were architectural, you MUST continue with architectural experiments until you've attempted at least 2, up to the hard cap of 10 experiments.
- A discard with stability ≥ +0.005 counts as "progress" and resets the counter — keep iterating on that direction.
- A `keep` resets everything.

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
6. **Record** (mandatory — do NOT skip): Append one row to `results.tsv` for EVERY experiment. This is not optional.

   **Keep/discard rules (stability-first):**

   An experiment qualifies as `keep` if ALL of the following are met:
   - `min_stability` improved by **at least +0.005** vs baseline.
   - No regime's `max_dd_pct` exceeds the **absolute DD cap** (see below). These caps are fixed and do NOT drift with baseline updates.
   - `raw_composite` ≥ **8.0** (composite score calculated WITHOUT tiered penalty — use pre-penalty regime scores to compute mean - 0.5*std + simplicity_bonus).

   **Absolute DD caps (hard ceiling, never changes):**
   - bull_2021: ≤ 7.8%
   - crash_bear: ≤ 6.9%
   - sideways: ≤ 5.6%
   - rally_2024: ≤ 6.0%

   **Revenue decline is acceptable.** A keep that improves stability but reduces composite score is valid as long as raw_composite stays ≥ 8.0. Do NOT discard experiments solely because composite decreased.

   **Computing raw_composite:** `regime_test.py` now outputs `raw_composite:` directly (pre-penalty composite). Just read it from `run.log` alongside `composite_score:`. No manual computation needed.

   **Composite keep path (only when min_stability ≥ 0.90):** Once stability reaches 0.90+, an alternative keep path opens: `composite_score` improved by at least +0.03 vs baseline, with no DD cap violation. This allows revenue optimization after the stability goal is achieved.

   If keep: append a `keep` line with all per-regime scores. The new baseline for subsequent experiments in this session is now this keep.
   If discard: **check exploration branch eligibility** (see below). If not eligible, run `git revert --no-edit HEAD`, append a `discard` line. NEVER use `git reset --hard`.

### Exploration branches (iterate before reverting)

When an experiment does NOT meet full keep criteria but shows architectural promise, you MAY keep the change and open an **exploration branch** instead of reverting. This lets you iterate on a promising direction — fixing weak regimes, tuning the new mechanism — before the final keep/discard decision.

**Entry conditions (any ONE is sufficient):**
- At least one regime score improved by ≥ +0.3
- At least one regime stability improved by ≥ +0.002
- Overall min_stability is unchanged or only slightly worse (≥ -0.003) AND the change introduces a fundamentally new mechanism (not a parameter tweak)
- The experiment improves composite/raw_composite meaningfully (+0.03) with stability regression explainable and plausibly fixable

You do NOT need the experiment to "almost pass" — the point is to allow bold architectural exploration. If you can articulate WHY the regression happened and HOW a follow-up change could fix it, that's sufficient justification to open a branch.

**Rules:**
- **Justification required**: State explicitly (1) what the new architecture does differently, (2) which regime regressed and why, (3) your hypothesis for fixing it in the next step.
- **Max depth**: 7 consecutive experiments on the branch (including the initial one that opened it). You get 6 more attempts to iterate on the new architecture.
- **Each iteration**: commit normally, run regime_test, record in results.tsv with prefix `branch:` (e.g., `branch: fix rally regression from linreg slope gate`). Each branch step may freely modify strategy.py — you're iterating on the new architecture, not just tweaking one parameter.
- **Success (keep the branch)**: if at any point during the branch the FULL keep criteria are met vs the **original baseline** (not branch-internal baseline), it's a real `keep`. Record as `keep` and update baseline.
- **Failure (revert the branch)**: if after max depth the keep criteria are still not met, revert ALL branch commits back to the original baseline: `git revert --no-edit HEAD~N..HEAD` (where N = number of branch commits). Record ONE summary `discard` line explaining the branch attempt and why it failed.
- **One branch per session**: you may only open one exploration branch per round. After a branch concludes (keep or revert), the session ends. The next round starts fresh with full context from results.tsv.
- **Exit rule interaction**: branch experiments count as architectural if the opening experiment was architectural.
- **Intermediate regression is OK**: within a branch, stability may temporarily drop further as you restructure. Only the FINAL state of the branch is judged against keep criteria vs original baseline. Don't abandon a branch just because step 2 made things worse — you still have step 3-7 to recover.

**Typical session shape:**
- Experiments 1-3: independent explorations (normal discard/revert cycle)
- Experiment 3-4: promising direction found → open exploration branch
- Experiments 4-10 (branch): iterate on new architecture, fixing weak regimes one by one
- Branch concludes → session ends

**Example flow (success):**
1. Exp 2: linreg slope deadzone gate → bull +0.47, rally -0.0003 → open exploration branch
2. Exp 3 (branch): add rally-protective condition → rally fixed, sideways -0.001 → continue
3. Exp 4 (branch): tune sideways parameters → sideways fixed, crash -0.001 → continue
4. Exp 5 (branch): crash-protective adjustment → all regimes pass → KEEP (vs original baseline)

**Example flow (failure):**
1. Exp 3: new exit mechanism → crash +0.5, sideways -0.010 → open exploration branch
2. Exp 4 (branch): add fast sideways exit path → sideways -0.005 (improved) → continue
3. Exp 5 (branch): alternative sideways fix → sideways -0.004 → continue
4. Exp 6 (branch): regime-specific sideways logic → sideways -0.003 → continue
5. Exp 7 (branch): different approach to sideways → sideways -0.003 → continue
6. Exp 8 (branch): hybrid exit for sideways → sideways -0.002 → continue
7. Exp 9 (branch): final attempt → sideways -0.002 (still failing) → max depth reached → revert all 7, record discard, session ends

7. **Decide next step**:
   - If you have a clear follow-up insight from the regime breakdown → continue to next experiment.
   - If you've found a keep and want to try combining it with another idea → continue.
   - If you've exhausted your ideas or hit 10 experiments → exit.

### Phase 3: Combination experiments (optional)

After running at least 2 independent experiments and observing their regime-level effects, you MAY attempt a combination:
- **Prerequisite**: at least one of the component ideas showed a promising signal (e.g., improved a target regime even if overall was discard due to regression elsewhere).
- **Combination = applying two ideas together** in a single `strategy.py` edit. Still counts as one experiment, still gets one results.tsv row.
- **Attribution**: in the commit message, reference which prior experiments you're combining (e.g., "Combines the EMA spread filter (sideways +0.24) with vol_ratio gating (crash protection)").
- **Same keep/discard rules apply** — no special treatment for combinations.

### Session end

After your last experiment (or when the exit rule triggers), exit. The outer loop will invoke you again for the next round.

## Results TSV format

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

Each regime is scored via multiplicative `compute_score()`, then combined:

```
Base score = log(1+sharpe)         # signal quality
           × sqrt(trade_factor)    # sample sufficiency
           × 1/(1 + DD%)           # base drawdown gate
           × exp(-max(0, DD%-5)/10) # soft DD penalty (mild slope above 5%)
           × 1/(1 + vol)           # volatility gate
           × exp(-streak/30)       # consecutive loss gate

Per-regime score = base_score × log(1 + annual_return% / 100)   # return gate

Hard cutoffs: <10 trades → -999, >20% drawdown → -999, lost >25% → -999

Composite score = mean(regime_scores) - 0.5 * std(regime_scores) + simplicity_bonus
Simplicity bonus = max(0, (500 - effective_LOC)) * 0.001   # reward shorter strategy.py
```

The simplicity bonus rewards removing dead code and unnecessary complexity. Effective LOC counts non-empty, non-comment lines in strategy.py. Each line removed below 500 adds +0.001 to composite.

Multiplicative structure: any dimension being terrible collapses the entire score.
The DD penalty is a smooth exponential — no cliff at any specific DD level. DD 5%→no penalty, 8%→0.74x, 10%→0.61x, 15%→0.37x.
The return gate prevents gaming via position-size reduction (smaller positions improve DD/vol gates but reduce returns).
The composite rewards strategies that perform **consistently across all market conditions**.

Search regimes (4 non-overlapping periods):
- bull_2021: 2021-01 ~ 2021-10 (bull market)
- crash_bear: 2021-11 ~ 2022-12 (Luna/FTX crash + deep bear)
- sideways: 2023-01 ~ 2023-12 (sideways recovery)
- rally_2024: 2024-01 ~ 2024-12 (ETF + election rally)

## Primary Objective: Signal Stability (min_stability ≥ 0.90)

**Stability is the #1 priority.** The scoring applies a **tiered penalty**:
- stability < 0.80 → 50% penalty: factor = (stab/0.85) × 0.50 (e.g., 0.70 → 0.41, loses 59%)
- stability 0.80–0.89 → 25% penalty: factor = (stab/0.85) × 0.75 (e.g., 0.82 → 0.72, loses 28%)
- stability ≥ 0.90 → no penalty: factor = stab/0.85, capped at 1.0

Each tier crossing yields a massive score boost. Reaching 0.80 = +40% per regime. Reaching 0.90 = another +33% per regime. Target: 0.90+.

**Do NOT conclude that "stability requires fundamentally different architecture and is too risky."** That reasoning is a trap — it leads to endless base-performance tweaks that never close the gap. Structural changes to improve stability ARE the highest-ROI experiments available.

**Multi-variable structural changes are explicitly allowed** for stability work. You are NOT limited to single-parameter tweaks. Diagnose the noise sensitivity source first, then propose whatever scope of change is needed — including architectural modifications that touch multiple components simultaneously.

### Diagnostic-first approach (optional, recommended for new sessions)

If results.tsv has < 10 entries or you haven't seen flip-rate data from prior sessions, diagnose noise sensitivity first:
1. Read `strategy.py` and identify voters/signals using hard thresholds on price-derived values
2. Estimate how far typical signal values sit from decision boundaries
3. Run the flip-rate diagnostic (see reference below) to quantify per-voter noise sensitivity

If results.tsv already contains diagnostic insights from prior sessions (grep for "flip rate", "noise", "voter sensitivity"), you may skip re-running diagnostics and proceed directly to experimentation.

### How to evaluate stability experiments
- Check `regime_X_stability` in the output — ALL four should improve toward 0.85+
- A stability gain of +0.005 is worth pursuing even if composite drops significantly — revenue decline is acceptable as long as raw_composite ≥ 8.0 and DD caps are not violated
- The ONLY hard constraints are: DD caps (bull ≤7.8%, crash ≤6.9%, sideways ≤5.6%, rally ≤6.0%) and raw_composite ≥ 8.0

## Stability improvement approaches (priority when min_stability < 0.90)

**Do NOT use open price as a "stable" signal source.** The noise test only perturbs close (then adjusts high/low). Open appears noise-immune but this is an artifact of the test methodology, not a real property. In live trading, open is equally noisy.
**HL2 stability gains are overstated.** HL2=(high+low)/2 receives ~half the perturbation of close. Acceptable use: multi-point aggregations (e.g., linreg over 16 bars). Unacceptable use: single-point comparisons or magnitude calculations. Discount reported HL2 stability gains by ~50%.

### Choosing your approach

There is no single correct path to stability. Choose based on your analysis of results.tsv and the current architecture:

**Incremental (within current architecture):**
- Per-voter hysteresis, confidence margins, abstain zones
- Input denoising (smoothing before voters)
- Aggregate decision margin (not just majority)
- Remove/replace the noisiest voter entirely

**Structural (new architecture):**
- Replace binary voting with weighted/continuous signals
- Redesign exit logic entirely (not RSI-based)
- Different signal fusion method (regression, scoring, probabilistic)
- Fundamentally different entry/exit decision mechanism
- Replace voter-based decisions with continuous confidence scores

If incremental approaches are saturated (check results.tsv — 10+ discards in that direction), switch to structural. **Do not keep iterating on approaches that have been proven to plateau.**

### Flip-rate diagnostic (reference)

Run this once per session if you need to identify the noisiest voter. Adapt the skeleton to instrument your actual voters:

```python
# diagnostic: per-voter flip rate under ±5bps noise
# Instrument vote computation to extract individual voter booleans
# Compare: voter_clean[i] != voter_perturbed[i] → flip
# Report table: Voter | Flip rate | Bars affected
# Target: identify voters with flip_rate > 0.03 (3%)
```

Do NOT hardcode "proven ineffective" conclusions here — read results.tsv each round to discover what has been tried. Only methodology-level blind spots (like open price artifact and HL2 overestimation above) belong in this file.

## Data available

- BTC, ETH, SOL hourly OHLCV + funding rates
- History buffer: last 500 bars via `bar_data[symbol].history` DataFrame
- Columns: timestamp, open, high, low, close, volume, funding_rate

## Overfitting hygiene

These rules exist because this branch has accumulated 190+ experiments. At that scale, selection bias dominates — any single +0.01 improvement is statistically fragile, and the search regimes themselves are effectively in-sample. Violating these rules causes meta-overfit that the regime-regression gate cannot catch.

- **Do NOT read commit bodies of prior experiments.** Use only `git log main..HEAD --oneline` (subjects only) and `results.tsv`. Commit bodies hold past hypotheses — reading them narrows your proposal space to "slight variants of what was tried," amplifying selection bias.
- **Do NOT base your idea on holdout findings.** The holdout (2025-01+) is never read by you. But also: if you notice phrasing in this prompt or in code comments that references specific holdout events (e.g., "the single-hour DD on 2025-03-02"), treat those as off-limits — do NOT design a rule targeting them. Any insight derived from holdout data is information leakage, regardless of who performed the analysis.
- **Prefer mechanism-backed changes over parameter sweeps.** At this experiment count, moving a parameter by 10% and finding +0.01 is likely noise. Adding a new signal with a clear mechanism, or removing a component that should be redundant, is a more honest trial.
- **Respect saturation signals.** If step 2 finds a direction has 10+ prior tries with mostly discards, do not submit another tuning variant.

## Guidelines

- **One variable per experiment** (within a session you run multiple experiments, but each tests one idea in isolation). The exception is Phase 3 combination experiments, which explicitly merge two independently-validated ideas.
- If you have no ideas, re-read `strategy.py` carefully and look for parameters to tune or signals to add/remove.
- All else equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code is not worth it.
- **Simplification experiments are as valuable as additions.** Try removing a voter, disabling a sizing multiplier, or deleting dead code. If the score holds or improves, keep the simpler version. Complexity has a hidden cost: it hurts out-of-sample generalization.
- **Do NOT inline constants or compress code for LOC bonus.** Named constants improve readability. The simplicity bonus rewards removing dead logic, not cosmetic code compression. Inlining a named constant into its usage site is NOT a valid simplification.
- **Use your session context wisely.** Your advantage over single-experiment mode is that you can observe patterns across experiments within this session. If experiment 1 shows sideways +0.24 but crash -1.44, experiment 2 should try to preserve the sideways gain while protecting crash — not start from scratch on an unrelated direction.
- Do NOT ask for confirmation. You are fully autonomous for this session.
