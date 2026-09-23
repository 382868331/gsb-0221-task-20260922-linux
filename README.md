# dirichlet_eb — 层级多项计数的确定性经验贝叶斯

在离线、小样本、无第三方依赖的条件下，对多组 K 类计数拟合**共同对称 Dirichlet
先验**的浓度参数 α（每类先验均为 α），并给出每组的后验均值、方差与新样本
类别预测。全部计算确定性、无随机数、无网络。

- **Python**：3.14.7（仅标准库，`math`、`dataclasses`、`enum`、`decimal` 仅测试参考使用）
- **依赖**：无。装好 Python 3.14.7 后无需 `pip install` 任何包
- **平台**：Linux 原生运行；不使用账号、密钥、网络、硬件、数据库、中间件、云 API 或 Docker

## 快速命令

```bash
python -m unittest discover -s tests -v   # 必要测试（47 个），约 2–4 秒（Decimal 参考开销）
python demo.py                            # 固定输入演示，约 0.05 秒（远低于 8 秒上限）
```

## 模型与数学

对组 g ∈ {1,…,G}、类别 k ∈ {1,…,K}：

```
θ_g ~ Dirichlet(α, …, α)                    # 对称 Dirichlet，单一浓度 α
n_g | θ_g ~ Multinomial(N_g, θ_g)
```

单组边际似然（Dirichlet–multinomial / Pólya 归一化常数之比）：

```
L_g(α) = Γ(Kα)/Γ(Kα + N_g) · ∏_k Γ(α + n_gk)/Γ(α)
log_evidence(α) = Σ_g log L_g(α)
```

- **空组（N_g = 0）的似然恒为 1**，对证据贡献严格的 0，不参与 α 估计；其
  后验/预测直接返回先验。
- 组内后验：`θ_g | n_g ~ Dirichlet(α + n_g1, …, α + n_gK)`
  - 后验均值：`m_k = (α + n_gk)/(Kα + N_g)`
  - 后验方差：`v_k = m_k(1 − m_k)/(Kα + N_g + 1)`
  - 一个新样本的类别预测：`P(category = k) = m_k`
  - `predicted_category = argmax m_k`（并列取最小下标）

## 接口

```python
from dirichlet_eb import (
    fit, log_evidence, log_evidence_per_group, compute_posteriors, validate_counts,
    FitResult, GroupPosterior, CandidateMode, OptStatus,
    InputValidationError, IdentifiabilityError, SearchBudgetExhaustedError, EBMError,
    ALPHA_MIN, ALPHA_MAX, MAX_CATEGORIES, MAX_GROUPS,
)
```

- `log_evidence(alpha, counts, k=None) -> float`
  池化对数边际似然。`counts` 为计数序列的序列；`k` 可显式给出，否则按首组长度推断。
- `log_evidence_per_group(alpha, counts, k=None) -> (tuple[float,...], rows, k)`
  每组对数似然（空组为 0.0）、校验后的行与 K。
- `compute_posteriors(alpha, counts, k=None) -> tuple[GroupPosterior, ...]`
  每组后验结构（空组返回先验）。
- `fit(counts, *, k=None, grid_points=201, tolerance=1e-10, max_iterations=200) -> FitResult`

`FitResult` 字段（frozen dataclass）：

| 字段 | 含义 |
|---|---|
| `alpha` | α̂（端点状态时为字面值 `1e-4` 或 `1e4`） |
| `log_evidence` | α̂ 处的池化对数证据 |
| `status` | `"interior"` / `"lower_endpoint"` / `"upper_endpoint"` |
| `k`, `group_count`, `nonempty_group_count` | 维度与组数 |
| `grid_points`, `candidate_count` | 网格点数与局部候选盆地数 |
| `function_evaluations`, `iterations` | 实际目标评估次数与局部搜索迭代总数（搜索限制可见） |
| `candidates` | 每个候选盆地的 `CandidateMode(log_alpha, bracket, log_evidence, status)` |
| `groups` | 每组 `GroupPosterior` |

### 搜索方法（确定性，括区间一维）

1. 在 `t = ln α ∈ [ln 1e−4, ln 1e4]` 上取 `grid_points`（默认 201）个均匀格点；
2. 噪声感知地找**全部**局部极大盆地：相邻格点证据差 ≤ 噪声带 `1e-10`
   （见下）视为同一平台；真正的双峰必须被深于该带的山谷分开；
3. 对每个盆地在其网格括号 `[t_{i−1}, t_{i+1}]`（端点盆地为边界格）内做
   **黄金分割**最小化 `−log evidence`，**任何评估都不越出显式区间**；
4. 取全局最大；返回端点/内部状态。所有候选盆地都保留在 `candidates` 中。

## 输入上限与错误语义

- `2 ≤ K ≤ 6`（`MAX_CATEGORIES=6`；`K=1` 时多项不携带 θ 信息，`fit` 抛
  `IdentifiabilityError`），`1 ≤ G ≤ 40`（`MAX_GROUPS=40`）；
- 计数必须为非负 **int**：拒绝 float、负数；`bool` 是 `int` 子类但被显式拒绝；
- 所有数值参数拒绝 `NaN`/`+Inf`/`−Inf`（`ValueError`，消息定位到参数）；
- 长度不一致、空批次、超限等抛 `InputValidationError`（`EBMError` 子类）；
- **整批先校验后计算**：非法批次在任何结果产生前抛出，不存在部分结果；
  返回结构均为 frozen dataclass。

### “预算耗尽”与“证明无解”严格分开

- `IdentifiabilityError`（数学结论，非搜索失败）：
  - 全部组为空（证据恒为 1）；
  - `K = 1`；
  - 每个非空组至多 1 个观测——单次抽取的边际似然恒为 `1/K`，
    证据在整个区间上常数。
- `SearchBudgetExhaustedError`（资源限制）：某盆地黄金分割在
  `max_iterations` 内未把括号收缩到 `tolerance`。`demo.py` 实际触发这两种情况。

## 浮点结果误差口径（不以宽容差掩盖边界）

所有浮点为 IEEE-754 float64（独立参考用 `decimal` 50 位）。

- **证据值**：核心用 `lgamma` 乘积式。对照测试用 K=2 的**积分多项式二项展开**
  （`B` 比值在 Decimal 50 位下做有限精确乘除，不调用被测核心）：
  - 小计数、α ∈ [1e−4, 1e4] 范围内 `|Δ log L| ≤ 1e-10`（测试断言）；
  - 远离高 α 端时实测约 1e-15～1e-12；
  - 接近 α=1e4 时，`lgamma` 参数 `Kα+N` 的输入舍入（ulp(4e4)≈7e-12，经
    ψ≈O(10) 放大）会产生约 1e-10 量级的对数涟漪，这是 float64 输入量化的
    下界，不是优化容差。
- **优化精度**：黄金分割停止于 log α 括号宽 `tolerance=1e-10`，对应 α 相对
  误差约 1e-10；测试中内部最优与独立优化器（801 点 log 网格 + 独立黄金分割）
  的 α 相对一致 ≤ 1e-6。
- **端点判定噪声带**：`1e-10` 对数证据、`1e-3` log α 距离。真实信号低于该带
  （如证据在 1e4 附近仅以 ~1e-11 爬升）在 float64 下无法与端点平台区分，
  此时按区间端点报告；这是已量化、已记录的判定规则。
- **后验矩**：与闭式 Beta 公式一致 ≤ 1e-12。
- 搜索**绝不**在 [1e−4, 1e4] 之外评估（测试检查括号边界）。

## 设计取舍

- 单一共同浓度 α（题设对称 Dirichlet）；证据为各组成对独立之和，因此**复制
  同一组 G 次不改变 α̂**（证据整体乘 G，最优点不变；有测试）。
- 默认 201 个 log 均匀格点：8 个数量级跨度下每步约 0.092 log 单位，配合每盆地
  黄金分割，默认预算内即可收敛（典型 fit 数毫秒）。
- 多峰处理保留为通用机制（网格→全部盆地→各自括区间局部优化）。在本模型与
  本区间的 3000 组随机小计数确定性扫描中**未观测到真实多峰**（证据普遍单峰），
  因此多峰路径以构造形状在 `tests/test_search.py` 中直接单测。
- 端点平台用噪声带合并，避免 lgamma 量化涟漪被报成多个虚假“峰”。

## 已知限制（本题数学子集，不宣称行业标准兼容）

- 只实现对称单参数 Dirichlet–multinomial 层级；不支持非对称 α 向量、
  类别相关协变量或跨组部分池化。
- α̂ 为边际似然点估计（MLE）；后验矩条件于 α̂，**不传播 α 的不确定性**，
  也不提供 α 的区间/后验抽样。
- 预测分布只给“一个”新样本；未提供 m 个未来样本的 Dirichlet–multinomial
  展开。
- 计数上限软保护为 1e300（超过报 `InputValidationError`），以保证 `lgamma`
  不溢出；题面规模（K≤6、G≤40、小样本）远在此内。
- 端点判定在信号 < 1e-10 对数证据时遵循上述噪声带约定。

## 文件结构

```
dirichlet_eb/__init__.py     # 公共 API 导出
dirichlet_eb/core.py         # 证据、后验、网格+黄金分割、错误类型（核心算法）
tests/reference.py           # 独立 K=2 参考：Decimal 精确二项展开 + 独立优化器
tests/test_evidence.py       # 证据/后验对照独立参考
tests/test_fit.py            # α̂、状态、空组、复制、端点
tests/test_contract.py       # 输入契约、错误语义、预算、确定性
tests/test_search.py         # 多峰盆地检测与黄金分割括号纪律
demo.py                      # 固定输入：正常拟合（显示收缩）+ 实际触发的异常边界
README.md
```
