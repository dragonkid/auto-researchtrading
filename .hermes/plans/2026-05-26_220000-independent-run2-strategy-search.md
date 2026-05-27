# Plan: Independent Run2 — From-Scratch Strategy Search

## Goal

启动一个与 run1 完全隔离的新 autoresearch run，使用相同的评分基础设施（prepare.py, regime_test.py, noise_test.py），但策略从极简骨架开始，让 agent 自由探索全新的架构方向。

## 当前状态

- Run1: `~/Coding/auto-researchtrading/`，分支 `autotrader/score-v3-run1`，min_stability=0.730，700+ 实验
- Run1 天花板：RSI+linreg 架构下 stability 0.725-0.730 是极限
- 评分系统：multiplicative score (sharpe × dd_gate × vol_gate × ...) + regime consistency + noise stability
- 数据路径：`~/.cache/autotrader/data/`（绝对路径，目录无关）

## 方案：完全独立目录 + 独立 git

```
~/Coding/auto-researchtrading/       ← run1（不动）
~/Coding/auto-researchtrading-v2/    ← run2（全新 repo，无历史）
```

### 为什么不用 git worktree / branch

- Agent 会读 git log 获取上下文 → 看到 run1 历史会复制方向
- results.tsv 在 worktree 间可见
- 完全独立 = 零污染

## 需要复制的文件（评分基础设施）

| 文件 | 用途 | 是否修改 |
|------|------|----------|
| `prepare.py` | 数据加载 + backtest 引擎 | 不改 |
| `regime_test.py` | 多 regime 评分 + composite | 不改 |
| `noise_test.py` | 稳定性测试 | 不改 |
| `autoresearch.sh` | 外层循环 | 微调（指向新 program.md） |
| `program-council.md` | Council mode | 可选复制 |

数据文件在 `~/.cache/autotrader/data/` — 绝对路径，自动共享。

## 不复制的文件

- `strategy.py` — 从极简骨架开始
- `results.tsv` — 从零开始
- `program-stateless.md` — 全新写
- 所有 git history — 不存在
- `STRATEGIES.md`, `POST.md` 等 — 不相关

## 新文件设计

### 1. strategy.py（极简骨架）

目标：能通过 backtest 跑通，但策略极简（买入持有 or 随机），让 agent 有一个能 commit/revert 的起点。

```python
"""
Minimal strategy skeleton.
Implement on_bar() to generate trading signals.
Available data: bar_data[symbol].{open,high,low,close,volume,funding_rate,history}
                portfolio.{cash,positions,entry_prices,equity}
Return: list of Signal(symbol, target_position)
"""
import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]
POSITION_SIZE = 0.10  # fraction of equity per position

class Strategy:
    def __init__(self):
        self.bar_count = 0

    def on_bar(self, bar_data, portfolio):
        signals = []
        self.bar_count += 1
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash

        # TODO: Replace with actual signal logic
        # This baseline does nothing (flat)
        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            signals.append(Signal(symbol=symbol, target_position=0.0))

        return signals
```

设计意图：
- Agent 第一步必须写一个真正的策略才能得到 >0 的分数
- 不暗示任何特定架构（没有 RSI、没有均线）
- 保留 `ACTIVE_SYMBOLS` 约定让 agent 知道交易什么
- 让 agent 看到 `bar_data` 和 `portfolio` 的接口

### 2. program-stateless-v2.md（开放式引导）

关键设计原则：
- **不提任何具体策略方向**（不提 RSI、趋势跟随、均值回归等）
- **保留评分解读**（agent 需要知道怎么读 regime_test 输出）
- **保留 session protocol**（实验流程、git commit、results.tsv 格式）
- **保留 keep 规则**（但调整阈值适应从零开始的场景）
- **强调 noise stability 的重要性**（这是最终门槛）
- **允许大胆的架构探索**

Keep 规则调整（关键区别）：
- **Bootstrap 阶段（composite < 5.0）**：composite > 0 即可 keep（快速建立 baseline）
- **成长阶段（composite 5.0-8.0）**：composite +0.5 或 stability +0.01 可 keep
- **成熟阶段（composite ≥ 8.0）**：回到 run1 的标准（stability +0.005 或 composite +0.03）

这确保 agent 在初期快速迭代，不会被严格 threshold 卡住。

### 3. autoresearch-v2.sh

基本跟 run1 相同，区别：
- `--system-prompt-file` 指向 `program-stateless-v2.md`
- TAG 默认 `score-v3-run2`
- Council threshold 可以放宽（初期不需要 council，前 20 轮让 agent 自由发挥）

### 4. results.tsv

只有 header，从零开始：
```
commit	composite	mean_score	std_score	bull	crash	sideways	rally	status	description
```

## 并行运行

```
tmux pane 1: cd ~/Coding/auto-researchtrading && ./autoresearch.sh score-v3-run1 50
tmux pane 2: cd ~/Coding/auto-researchtrading-v2 && ./autoresearch.sh score-v3-run2 50
```

两个进程完全独立，不共享任何 mutable state。

## 实施步骤

```bash
# 1. 创建新目录
mkdir ~/Coding/auto-researchtrading-v2
cd ~/Coding/auto-researchtrading-v2

# 2. 初始化独立 git repo
git init

# 3. 复制基础设施（只读文件）
cp ~/Coding/auto-researchtrading/prepare.py .
cp ~/Coding/auto-researchtrading/regime_test.py .
cp ~/Coding/auto-researchtrading/noise_test.py .
cp ~/Coding/auto-researchtrading/program-council.md .

# 4. 创建新文件
# - strategy.py（极简骨架）
# - program-stateless-v2.md（开放式引导）
# - autoresearch.sh（微调版）
# - results.tsv（空 header）

# 5. 初始 commit
git add -A
git commit -m "init: scoring infrastructure + minimal strategy skeleton"

# 6. 创建分支
git checkout -b autotrader/score-v3-run2

# 7. 启动
./autoresearch.sh score-v3-run2 50 | tee /tmp/autoresearch-v2.txt
```

## 风险 & 权衡

| 风险 | 缓解 |
|------|------|
| Agent 从零开始可能浪费 20+ 轮写出低质量策略 | Bootstrap 阶段 keep 规则宽松，允许快速迭代 |
| Agent 可能重新发现 RSI（很常见的方向） | 不算风险 — 如果 agent 独立发现相同方向并用不同架构实现，可能绕过 run1 的天花板 |
| Agent 可能写出过拟合策略 | noise_test 会自动惩罚 — 这是评分系统的核心设计 |
| Council mode 在初期可能过早触发 | 提高初期 council threshold（前 20 轮 10 consecutive discards 才触发）|
| prepare.py 依赖 scipy/numpy — 新目录可能缺环境 | 用相同的 uv/venv 环境，或 symlink |

## Open Questions

1. **Claude 配置隔离？** — 两个 run 共用 `~/.claude-autoresearch` 应该没问题（无 mutable state 冲突），还是需要 `~/.claude-autoresearch-v2`？
2. **是否给 agent 提供 `prepare.py` 的接口文档？** — 或者让它自己读。倾向让它自己读（更自然）。
3. **DD cap 是否保留？** — 建议保留（7.8/6.9/5.6/6.0），这是核心约束。但初期可以放宽到 20%（prepare.py 的 hard cutoff），让 agent 先找到好方向再收紧。
4. **stability 初期目标？** — 建议不设初期 stability 门槛，让 composite 主导前期探索。stability penalty 本身已经内置在评分中（<0.80 打 50% 折），agent 会自然被引导。
