# 本地运行与讲解验收

先用 VS Code 打开这个项目文件夹，在 Terminal 中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

预期 16 项测试通过。

```bash
python scripts/run_experiments.py
```

结果和三张图生成在 `reports/`。数据全部来自模拟，固定种子用于复现；没有真实交易所行情。

## 三个亲手实验

1. 手算：中间价 100，持仓 5，风险厌恶 0.1，波动率 2，剩余时间 1。保留价应为 100 - 5×0.1×4 = 98。解释为何有多头库存时想降低报价。
2. 运行 `python -m pytest -q tests/test_mmsim.py -k pnl_decomposition`，说明“现金+期末持仓市值”为什么等于“价差收益+库存收益”。
3. 比较 `jumps_only` 和 `informed_flow`，再比较训练选出的 buffer 在新种子上的结果。解释这是模拟环境中的检验，不能称为真实市场样本外收益。

## 模型的重要边界

噪声成交是每个时间步、每一边最多一次的 Bernoulli 近似，概率为 clip(A×exp(-k×距离)×dt,0,1)，不是精确 Poisson 到达次数。知情交易者被假设提前知道下一次跳跃；报价策略没有这个预知权限。

`informed_pnl` 是成交价相对跳跃后中间价的诊断，不能再加到总 P&L 中，否则会重复计数。

先解释对称报价与库存偏移，再解释知情交易，不要先背“风险减半”和“收益改善 8%-29%”这些结果。
