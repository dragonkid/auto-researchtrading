# 信号稳定性指标 v3 — Noise Robustness for Research Scoring

## Context

实盘分析发现：CryptoCompare (回测) 和 Bitget (实盘) 之间有 2-4 bps 系统性基差，但其 mean level 对 return 计算无影响（constant multiplier 约分掉）。真正导致信号分歧的是基差的 **bar-to-bar 变化**（std ≈ 1.5 bps）。这种变化改变了 momentum return 的计算结果，在 threshold 边界处翻转信号。

结果：5 次 flip cascade，净损失 -$6.98 (35% of total loss in 45h)。

当前 research 框架无法检测 threshold sensitivity。需要在 composite score 中加入 signal stability 维度。

## 关键推导：为什么用 i.i.d. per-bar noise

1. **Constant offset 无效** — 所有 bar 乘同一常数 → return `(c[-1]*m - c[-12]*m) / (c[-12]*m)` 约分 → 信号不变
2. **真实失败模式** — 基差的 bar-to-bar 变化（~1.5 bps std）才改变 return
3. **i.i.d. noise 是保守上界** — ±5 bps i.i.d. 比真实 ~1.5 bps 变化严格得多
4. **通过 i.i.d. ±5bps 测试的策略** → 必然能承受真实数据源差异

## 设计

### 方法：i.i.d. Per-Bar Noise + Full Simulation

每个 regime 跑 1 次 clean backtest + 10 次 perturbed backtest。

Perturbation 方式:
- 对每个 symbol 的每根 bar，独立加 `Uniform(-5, +5) bps` 噪声到 close
- 扰动应用到 **整个 DataFrame**（策略看到的 500-bar history 全部被扰动）
- 同时调整 H/L 保持 OHLC 一致性
- 用固定 seed (42 + trial_idx) 保证确定性可复现

比较 clean 和 perturbed 的 **equity curve**:
```
tracking_error = std(clean_hourly_return - perturbed_hourly_return)
normalized_te  = tracking_error / clean_volatility
stability      = 1 - mean(normalized_te across 10 trials)
```

### 为什么用 full simulation 而非 fixed portfolio

- 真正的 PnL 损失来自 **flip**（持仓方向翻转）
- Flip 需要有 existing position → fixed portfolio (空仓) 永远不会触发 flip 分支
- Full simulation 让 portfolio 状态自然演化，才能检测 flip sensitivity

### 状态 cascade 处理

Full simulation 下，第一个 divergence 会改变后续 portfolio 状态（cooldown timer, positions），导致后续 bar 的 divergence 不独立。

处理方式: **Tracking error 天然吸收 cascade**。
- Tracking error = std(clean_return - pert_return)，一次 flip 会导致后续几十 bar 的 return 持续偏移
- 这比 count-based 指标更准确：一次 flip 后 portfolio 方向相反，每个后续 bar 的 return diff 都贡献到 TE
- 但 TE 是 std（不是 sum），所以持续的小偏移（constant offset from different position size）不会无限膨胀
- 多次 trial (n=10) 平均降低单次 cascade 的统计影响
- TE 归一化 by clean_vol → 无量纲，跨 regime 可比较

### 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `noise_test.py` | 新建 | stability 计算 (i.i.d. noise, full simulation) |
| `regime_test.py` | 修改 | worker 调 noise_test, composite 加 stability_penalty |
| `program-stateless.md` | 修改 | 告知 agent stability 输出格式 |

### noise_test.py 核心逻辑

```python
import numpy as np
from prepare import run_backtest, BacktestResult

N_TRIALS = 10
NOISE_BPS = 5.0

def compute_signal_stability(data: dict, clean_result: BacktestResult) -> float:
    """
    Full-simulation stability via equity-curve tracking error.

    Compares clean vs perturbed equity curves (naturally timestamp-aligned,
    one entry per bar) to avoid the index-misalignment problem of trade-log
    comparison.

    Returns: 1.0 (perfectly stable) to 0.0 (completely divergent).
    """
    from strategy import Strategy

    clean_eq = np.array(clean_result.equity_curve)
    if len(clean_eq) < 10:
        return 1.0

    clean_ret = np.diff(clean_eq) / np.where(clean_eq[:-1] > 0, clean_eq[:-1], 1.0)
    clean_vol = clean_ret.std()
    if clean_vol < 1e-10:
        return 1.0

    tracking_errors: list[tuple[bool, float]] = []  # (is_correlated, te)

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(42 + trial)
        correlated = (trial < N_TRIALS // 2)  # first 5: correlated across symbols, last 5: i.i.d.
        perturbed_data = _perturb_data(data, NOISE_BPS, rng, correlated=correlated)
        pert_result = run_backtest(Strategy(), perturbed_data)

        pert_eq = np.array(pert_result.equity_curve)

        # Early termination (liquidation) = worst-case divergence
        if len(pert_eq) < 0.8 * len(clean_eq):
            tracking_errors.append((correlated, 3.0 * clean_vol))
            continue

        n = min(len(clean_eq), len(pert_eq))
        if n < 10:
            continue

        pert_ret = np.diff(pert_eq[:n]) / np.where(pert_eq[:n-1] > 0, pert_eq[:n-1], 1.0)
        clean_ret_aligned = clean_ret[:n-1]

        # Tracking error = std of return differences
        diff = clean_ret_aligned - pert_ret
        te = diff.std()
        tracking_errors.append((correlated, te))

    if not tracking_errors:
        return 1.0

    # Separate correlated vs i.i.d. modes, take worst-case to avoid dilution
    corr_tes = [te for is_corr, te in tracking_errors if is_corr]
    iid_tes = [te for is_corr, te in tracking_errors if not is_corr]
    mean_te = max(
        sum(corr_tes) / len(corr_tes) if corr_tes else 0.0,
        sum(iid_tes) / len(iid_tes) if iid_tes else 0.0,
    )

    # Normalize: tracking_error / clean_vol → 0 = identical, 1+ = fully divergent
    normalized_te = mean_te / clean_vol

    return max(0.0, min(1.0, 1.0 - normalized_te))


def _perturb_data(data: dict, noise_bps: float, rng, correlated: bool = False) -> dict:
    """Apply per-bar noise to close (and adjust H/L) for all symbols.

    correlated=True: same noise sequence for all symbols (tests cross-asset cascade).
    correlated=False: independent noise per symbol (tests per-symbol threshold sensitivity).
    """
    # Verify symbols have similar length (correlated mode assumes row-aligned timestamps)
    lengths = [len(df) for df in data.values()]
    if max(lengths) - min(lengths) > 5:
        raise ValueError(f"symbol bar count mismatch ({max(lengths) - min(lengths)}) too large for correlated noise")

    # Pre-generate common noise for correlated mode
    max_len = max(lengths)
    common_noise = rng.uniform(-noise_bps, noise_bps, size=max_len) / 10000.0 if correlated else None

    result = {}
    for sym, df in data.items():
        new_df = df.copy()
        n = len(new_df)
        noise = common_noise[:n] if correlated else rng.uniform(-noise_bps, noise_bps, size=n) / 10000.0
        new_df['close'] = new_df['close'] * (1.0 + noise)
        new_df['high'] = new_df[['high', 'close']].max(axis=1)
        new_df['low'] = new_df[['low', 'close']].min(axis=1)
        result[sym] = new_df
    return result
```

**为什么用 tracking error 而非 trade-log 比较：**
- Trade log 没有 timestamp，按 sequential index 比较时一笔额外交易导致后续 ALL indices 错位 → divergence 虚高
- Equity curve 天然按 bar 对齐，无 alignment 问题
- Tracking error (std of return difference / clean_vol) 是金融标准指标，直接衡量 noise 导致的经济偏移
- 同决策但 price 微变 → TE 极小；flip 导致方向反转 → TE 显著跳升

### regime_test.py 修改

`_run_regime_worker()` 末尾加（传入已有的 `result` 避免重复 clean backtest）:
```python
from noise_test import compute_signal_stability
stability = compute_signal_stability(data, result)

# Multiplicative penalty: stability < 0.85 → proportional score reduction
# Only apply to positive scores — preserve -999 sentinel for hard-fail detection
stability_factor = min(1.0, max(0.0, stability / 0.85))
if score > 0:
    score = score * stability_factor
```

worker 返回 dict 加 stability 字段:
```python
return {
    ...
    "stability": stability,
    "stability_factor": stability_factor,
}
```

parseable 输出（在**主进程**末尾的 per-regime 循环中，不在 worker 中 print）:
```python
# 加在现有 regime_{name}_score / regime_{name}_sharpe 之后
print(f"regime_{r['name']}_stability: {r['stability']:.6f}")
```

`compute_composite_score()` 只做汇总诊断输出:
```python
stabilities = [r.get("stability", 1.0) for r in results]
min_stability = min(stabilities)
print(f"min_stability: {min_stability:.1%}")
```

注意: 不在 `_run_regime_worker()` 中 print — subprocess stdout 会和主进程交错。

### program-stateless.md 修改

Phase 2 grep 指令改为:
```bash
grep "^composite_score:\|^mean_score:\|^std_score:\|^regime_\|^min_stability:" run.log
```

说明: `regime_{name}_stability:` 已被 `^regime_` 捕获; `min_stability:` 额外匹配作为诊断汇总。

### 参数

| 参数 | 值 | 理由 |
|------|-----|------|
| NOISE_BPS | 5.0 | 实测基差 std ~1.5 bps, ×3 作为保守 stress test |
| N_TRIALS | 10 | 性能平衡: 5 correlated + 5 i.i.d. = 10 trials × 4 regimes = 40 extra backtests |
| seed | 42 + trial | 确定性，每个 trial 不同但可复现 |
| noise_mode | mixed | trial 0-4: correlated (同噪声 across symbols), trial 5-9: i.i.d. |
| penalty_type | multiplicative | `score *= min(1.0, stability / 0.85)` |
| penalty_threshold | 0.85 | tracking error 归一化后 85% 以下开始惩罚 |
| penalty_scope | per-regime | 在 score 产生的同层级扣减，不做全局聚合 |
| liquidation_penalty | 3x clean_vol | perturbed equity < 80% clean length → TE = 3 * clean_vol |

### Penalty 效果

Multiplicative penalty — 每个 regime 独立根据自己的 stability 缩放 score：

| regime stability | factor | 对 score=25 的影响 |
|------------------|--------|-------------------|
| 95% | 1.00 | 25.0 (无影响) |
| 85% | 1.00 | 25.0 (刚好不罚) |
| 80% | 0.94 | 23.5 (-1.5) |
| 70% | 0.82 | 20.6 (-4.4) |
| 50% | 0.59 | 14.7 (-10.3) |

**为什么 multiplicative 而非 additive：**
- 实际 per-regime score 在 19-33 范围（非设计时假设的 5-8）
- Multiplicative 按比例缩放，自动适应不同 score 量级
- stability=0.70 时扣掉 ~18% score（约 4-6 分），足以影响 keep/discard 决策

**为什么 per-regime 而非 global penalty：**
- 不稳定的 regime 其 score 本身就不可信 → 在产生 score 的同一层扣减
- 如果只有 sideways 不稳定（70%），只有 sideways 的 score 被扣；其他 regime 不受影响
- 如果多个 regime 都不稳定，composite 通过 mean 自然累积惩罚
- 不需要 min/mean 的选择问题 — 每个 regime 为自己的稳定性负责

**为什么混合 correlated + i.i.d. 噪声：**
- Correlated (trial 0-4): 所有币种同 bar 加相同噪声 → 测试 cross_asset_agree cascade（真实数据源偏差通常全局性）
- i.i.d. (trial 5-9): 每币种独立噪声 → 测试 per-symbol threshold sensitivity
- 两种 failure mode 都覆盖，无额外计算成本

### 性能

- 每 regime: 1 clean (已有，复用) + 10 perturbed = 10 × 额外 `run_backtest`
- 4 regimes × 10 = 40 extra runs (ProcessPoolExecutor 4 workers 并行)
- 每 worker 内 10 runs 串行，每 run ~4-5s → 每 worker ~45-50s
- 估计 ~50-60s total wall time (8-core Mac, 4 parallel workers)
- 每轮 research (5 experiments): ~5 min (原 ~2.5 min)
- 整个 autoresearch session (5-10 轮): ~25-50 min (原 ~12-25 min)

## 验证

1. `uv run python noise_test.py` — 当前策略 stability 应 > 85%（TE/vol < 0.15）
2. 临时设 `BASE_THRESHOLD = 0.001` — stability 应显著下降（<70%）因为极小 threshold 对 noise 极敏感
3. `uv run regime_test.py` — 输出含 `stability:` per-regime + `min_stability:` + `stability_penalty:`
4. 确认 penalty=0 when min_stability >= 85%，penalty > 0 when < 85%
5. 用 `score-v3-run1` branch tag 启动 research, 确认 agent grep 正确 parse
