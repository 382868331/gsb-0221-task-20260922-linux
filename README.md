# himeb — 层级多项计数的确定性经验贝叶斯

纯 Python 标准库实现的小样本层级模型估计模块：多组 K 类计数共享一个对称
Dirichlet 先验 `Dirichlet(alpha,…,alpha)`，通过最大化（Dirichlet-）多项边际
似然（证据）估计浓度参数 `alpha`，并给出每组的后验均值、方差与新样本类别
预测。

- **Python**：3.14.7（仅用标准库 `math`、`dataclasses`、`typing`）
- **依赖**：无。安装 Python 后无需 `pip install` 任何东西，完全离线。
- **平台**：Linux 原生运行；不访问网络、账号、数据库、中间件、云或 Docker。

## 运行命令

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m unittest discover -s tests -v   # 必要测试（含独立参考对照）
python demo_iteration.py                  # 固定输入演示，通常远低于 90 秒
```

`.venv` 只用于锁定 Linux 原生 Python 3.14.7 解释器；仓库没有第三方依赖，
因此不运行 `pip install`。若系统的 `python` 已经是 3.14.7，也可直接运行上面的
测试和演示命令。旧基线演示仍保留为 `python demo.py`。

## 模型与数学

对组 g 的计数 `x_g = (x_g1,…,x_gK)`，n_g = Σ x_gk：

```
theta_g ~ Dirichlet(alpha,…,alpha)
x_g | theta_g ~ Multinomial(n_g, theta_g)
```

单组对数证据（边际化 theta）：

```
log p(x|alpha) = log C(n; x)
  + logGamma(K*alpha) - logGamma(n + K*alpha)
  + Σ_k [ logGamma(x_k + alpha) - logGamma(alpha) ]
```

- 多组证据为各组之和；**空组（n=0）证据恒为 0，不参与求和**，其后验只返回
  先验。
- 另有一个退化事实：n=1 的组在对称先验下 `p(x=e_k|alpha) = 1/K` 与 alpha
  无关（单抽类别的概率就是先验质量）。当所有非空组都是 n=1 时，证据为
  精确常数 `-G1*log(K)`。

后验为 `Dirichlet(alpha + x_k)`：

```
mean_k       = (alpha + x_k) / (K*alpha + n)
variance_k   = a_k (A - a_k) / (A^2 (A + 1)),  a_k = alpha+x_k, A = K*alpha+n
predictive_k = mean_k                  # 一步多项预测 = 后验均值
predicted    = argmax(predictive)      # 平局取最小下标
```

## 部分（合并类别）观测

样本 g 可给出原 K 类的一个完整划分 `B_g = {S_g1,…,S_gJ}` 以及每个集合的计数
`y_gj = Σ_{k∈S_gj} x_gk`。不同样本可以采用不同划分，但所有样本的原始 K 必须
相同。对称 Dirichlet 的聚合性质使每个大小为 m 的类别集合对应先验参数
`m*alpha`，总浓度仍为 `K*alpha`；因此合并后的 DM 证据为：

```
log p(y|alpha) = log C(n; y_1,…,y_J)
  + logGamma(K*alpha) - logGamma(n + K*alpha)
  + Σ_j [ logGamma(y_j + |S_j|*alpha) - logGamma(|S_j|*alpha) ]
```

不引入第二套非对称参数；大小为 m 的集合只使用 `m*alpha`。完全细分时，每个
集合 `|S|=1`，公式与旧完整计数证据逐项相同。

公开入口：

```python
from himeb import partial_observation, fit_alpha_partial

obs = partial_observation([
    ((0, 1), 4),  # 原类别 0、1 合并计数为 4
    ((2,), 1),
])
result = fit_alpha_partial([obs, other_obs])
result.fit.alpha
result.fit.status
result.contributions          # 每组在拟合 alpha 处的证据贡献
```

`partial_observation` 会拒绝：空集合、类别重复、遗漏原类别、集合交叠、负/非
整数/bool 计数、各样本 K 不一致等情况。直接构造 dataclass 也会在计算前重新
校验。

**全合并**观测只有一个聚合类别，已知总数后概率恒为 1，证据恒为 0。任何
`n=1` 观测落入大小 m 的集合时，概率恒为 `m/K`，也与 alpha 无关。空样本证据
为 0。若一批观测全部属于这些恒常情形，`fit_alpha_partial` 直接返回
`status='flat'`、`alpha=1.0`、`converged=True`、`budget_exhausted=False`，且
不运行搜索器；`alpha=1.0` 只是 flat 约定，不是识别出的估计值。混合批次中仍有
信息样本时，复用原 `fit_alpha` 的同一网格+黄金分割搜索器拟合共享 alpha；无
信息样本的恒常贡献仍保留在总证据和 `contributions` 中。

## 优化：括区间一维搜索

在 `t = log(alpha)` 的**显式闭区间** `[1e-4, 1e4]`（常量 `ALPHA_MIN/MAX`）
上最大化总对数证据：

1. 区间上 201 个（默认）均匀 t 网格求值；
2. 每个网格局部极大（含平台节点）作为候选模，在其相邻网格构成的括号
   `[t_{i-1}, t_{i+1}]` 内做**保括号黄金分割**细化（容差 `T_TOL=1e-12`）；
3. 两个端点始终作为候选并精确比较；
4. 所有候选都保留在结果里（值、alpha、括号、迭代次数、是否收敛），最高者胜出。

`FitResult.status` 明确区分四种结局：

| status | 含义 |
|---|---|
| `interior` | 内部极大 |
| `lower_endpoint` / `upper_endpoint` | 极大落在区间端点（边界状态，不外延） |
| `flat` | **证明性无唯一解**：完整计数中全空/非空全 n=1；部分观测中空样本、n=1 或全合并使全部证据为常数；不运行搜索，`alpha=1.0` |

**搜索预算耗尽与证明无解严格分开**：内部细化用尽 `max_iter` 仍未达到容差时
`converged=False`、`budget_exhausted=True`（连续函数在闭区间上必有最大值，
所以这只是预算问题）；`flat` 则 `converged=True` 且 `budget_exhausted=False`。

## 公开接口

```python
from himeb import (
    log_evidence, log_evidence_one,   # 证据（任意正 alpha；区间只管优化器）
    fit_alpha,                        # 仅拟合 alpha -> FitResult
    maximize_on_interval,             # 通用多峰一维最大化引擎 f(alpha)
    posterior_group,                  # 单组后验均值/方差/预测
    analyze,                          # 拟合 + 全部组后验 -> EBResult
    ALPHA_MIN, ALPHA_MAX, MAX_K, MAX_GROUPS,

    # 部分观测（合并类别）
    partial_observation,
    log_evidence_partial,
    partial_evidence_breakdown,       # total + 每组 contributions
    fit_alpha_partial,                # 复用原搜索器 -> PartialFitResult
)
```

结果均为不可变 dataclass（`Candidate`、`FitResult`、`GroupPosterior`、
`EBResult`）；所有函数为纯函数、无状态，非法批次在产出任何结果前整体抛错，
不存在部分变更。

```python
result = analyze([(4, 0, 1), (0, 3, 0), (0, 0, 0)])
result.fit.alpha            # 拟合浓度
result.fit.status           # interior / *_endpoint / flat
result.fit.candidates       # 全部候选（含 bracket、iterations、converged）
result.posteriors[0].mean   # 后验均值
result.posteriors[0].variance
result.posteriors[0].predictive
```

## 输入上限与错误语义

- 类数 `2 <= K <= 6`；组数 `1 <= G <= 40`；所有组长度必须一致。
- 部分观测中每组必须给出原 K 类的一个完整划分；组间划分可以不同，但 K 必须
  相同。集合不可为空、重复、遗漏或交叠。
- 计数为非负 `int`，单个计数上限 `2**53`（超出无法作为 float64 精确参与证据
  计算，直接拒绝）。
- 数值输入（alpha、目标函数返回值）拒绝 `NaN`/`±Infinity`；alpha 必须 > 0。
- **bool 不是整数**：计数、`grid_points`、`max_iter` 传 `True/False` 抛
  `TypeError`。
- 类型不符抛 `TypeError`，取值非法抛 `ValueError`；错误信息带可定位坐标
  （如 `group 1 class 0: ...`、`partial observation 2 block 0 class 1: ...`），
  不静默纠正、不钳制。
- `grid_points >= 3`、`max_iter >= 1`。

## 浮点误差口径（不用宽容差掩盖边界）

全部计算为 IEEE-754 binary64（多个量的求和用 `math.fsum`）。

- **证据**：当整数增量 m ≤ 128 时，
  `log Gamma(x+m)/Gamma(x)` 用恒等式 `Σ_{j=0}^{m-1} log(x+j)` 精确求和，
  避免 alpha=1e4 处两个大 lgamma 相消；m 更大时用 `lgamma` 差。精度由下列
  独立手段实测确认（非估计）：
  - alpha=1（均匀先验）时任意 x 都有精确事实 `p(x)=1/(n+1)`，n≤400 上
    生产证据与该闭式值绝对差 **≤ 1.5e-13**；
  - n≤128、alpha=1e4（最易相消的角落）与独立数值积分差 **< 2e-13**；
  - 平滑大计数（n≤400）与自适应 Simpson 积分差 **≤ 4e-11**；
  - 极大计数（n≈2e15）结果仍有限，误差为 float64 固有水平
    （约 `eps·|对数项|`）。
- **独立参考对照**（`tests/reference.py`，不 import 被测包）：
  - 二项（K=2）似然因子做多项式展开后逐项 beta 积分（带符号 log-sum-exp）。
    该交错和在题设“小计数”n ≤ 8、alpha 覆盖 `[1e-4,1e4]` 上与生产实现绝对差
    **≤ 2e-8**（此 2e-8 是交错求和**参考本身**的误差，不是生产实现的误差；
    n 更大时交错相消使参考失效，已在“已知限制”说明，故大 n 改用上面的
    精确恒等式/积分核对）；
  - 第二条独立路线为 [0,1] 上自适应 Simpson 直接积分（alpha ≥ 1、被积函数
    不单点尖峰时有效），绝对差 **≤ 2e-9**；
  - 后验矩另用独立的 beta 积分比 `B(a+1,b)/B(a,b)` 交叉核对（alpha ≤ 10）。
  - 合并观测的独立参考枚举所有满足各集合总数的小总数细分类计数向量，分别计算
    细分类 DM 证据后 log-sum-exp；这是在事件集合上直接求和，不调用生产中的
    合并公式。当前测试在 n≤6、K=3、alpha 覆盖 `[1e-4,1e4]` 时与生产合并证据
    绝对差 **≤ 2e-12**；完全细分观测还与旧完整计数证据精确相等。
  生产实现与参考的期望值由独立公式各自算出；“往返一致”仅作补充。
- **优化**：内部模在 t 空间定位到 `1e-12` 括号宽度；端点为精确函数值。
  拟合 alpha 与独立 4001 点密扫描的相对差 ≤ 2%（密扫描自身分辨率有限），
  证据在拟合点自洽到 1e-10。
- **后验矩**：纯乘除，相对误差为机器精度量级（~1e-15）；均值之和在
  1e-12 内归一。

## 设计取舍

- **在 log alpha 空间搜索**：Dirichlet 证据跨 8 个数量级，t 空间使括号按
  比例收缩、黄金分割不会因尺度退化；端点固定不外推。
- **网格 + 局部、保留全部搜索痕迹**：支持多峰目标，且不把“网格分辨率”这一
  限制藏起来——每个候选带回 log-alpha 括号、迭代数和收敛标志。
- **关于自然多峰**：对 K=2 全部 n≤4 组型子集做了穷举、并对 12 万余组
  2–4 型混合（n≤20，K=2..4）随机扫描，**未观察到**对称 Dirichlet–多项
  证据出现显著多峰（其结构接近单峰）；多峰能力因此用通用引擎
  `maximize_on_interval` + 合成双模目标在测试中验证（两个内模均被找到并细化）。
- **空组 / 单抽组不“拟合噪声”**：证据贡献为解析常数时直接判 `flat`，
  而不是让数值搜索在机器噪声里随便挑一个点。
- **全合并不提供 alpha 信息**：对称 Dirichlet 聚合后只知道所有类别的总数，
  事件概率恒为 1；因此它与单抽、空组一样在纯常批中产生证明性 `flat`，但在
  混合批次中作为恒常贡献保留。
- **部分观测不返回原类后验**：只看到合并计数时，组内类别的后验无法唯一展开；
  因此本增量只提供合并证据、共享 alpha 拟合和逐组证据贡献。原完整计数的
  `posterior_group` / `analyze` 接口和行为不变。
- 只实现本题契约的数学子集（对称先验、单一共享 alpha、上述上限），不宣称
  任何行业标准协议兼容。

## 已知限制

- 网格可能漏掉**窄于一个网格单元且峰顶落在格点之间**的模；`tests` 中有用
  3 点网格故意漏模、2001 点网格找到的对照用例。增大 `grid_points` 可缓解。
- 先验固定为对称形式、全组共享单一 alpha；不支持非对称基率、组级协变量或
  层级超先验。
- 枚举参考只适合小总数（测试使用 n≤6、K≤3）；类别数或总数增大时枚举组合数
  会快速增长。生产合并公式本身不受该参考枚举规模限制，但仍受 K≤6、G≤40 和
  单个计数 2**53 的接口边界约束。
- 部分观测不会臆造集合内部各类的分配，也不提供基于合并计数的原类后验。
- `flat` 情形无数据信息，按约定返回 `alpha=1.0` 供后验计算，不代表估计值。
- 参考展开式在 n 很大且 alpha 很小时交错相消明显，因此参考测试限定在
  题设的小计数范围；生产实现用整数 log 和不受此限。
- 独立 Simpson 积分参考在被积函数宽度 ≪ 1（如 a=1、x=0、n=400 时尖峰
  宽约 1/n）时，固定递归深度（30 层）不足以分辨，会严重低估；这是参考
  数值方法的限制（此时改用 `p(x)=1/(n+1)` 闭式事实核对），不是生产证据的
  误差。
- 模块只做计算，不提供 CLI 产品、网站、看板或持久化。

## 仓库结构

```
himeb/core.py        # 核心：证据、优化器、后验、校验
himeb/__init__.py    # 公开接口
tests/reference.py   # 独立参考（展开 / 数值积分 / beta 矩 / 密扫描）
tests/test_*.py      # 校验、证据与后验、优化、合并枚举参考测试
demo.py              # 旧基线固定输入演示
demo_iteration.py    # 本题增量：混合划分、flat 与异常边界演示
```
