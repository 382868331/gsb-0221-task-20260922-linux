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
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试（含独立参考对照）
./.venv/bin/python demo.py                            # 固定输入演示，< 1 秒
```

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
| `flat` | **证明性无唯一解**：全空组（证据恒 0）或非空组全为 n=1（证据为常数）；不运行搜索，`alpha=1.0` |

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
    posterior_predictive,            # 后验预测事件概率与条件期望（m<=100）
   analyze,                          # 拟合 + 全部组后验 -> EBResult
    ALPHA_MIN, ALPHA_MAX, MAX_K, MAX_GROUPS, MAX_PREDICTIVE_M,
)
```

结果均为不可变 dataclass（`Candidate`、`FitResult`、`GroupPosterior`、
`EBResult`、`PredictiveResult`）；所有函数为纯函数、无状态，非法批次在产出任何结果前整体抛错，
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

## 后验预测：下一批 m 次观测落入计数区间的概率

共同先验拟合完成后，对某组已观测计数 `x`，其后验为
`Dirichlet(a_k)`，`a_k = alpha + x_k`；下一批 `m` 次计数 `y` 因此服从
**Dirichlet-multinomial（复合多项）分布**，而不是把后验均值代进普通多项分布：

```
p(y | x, alpha) = C(m; y) * Gamma(A) / Gamma(A + m)
                  * prod_k Gamma(a_k + y_k) / Gamma(a_k),   A = sum_k a_k
```

入口 `posterior_predictive(counts, alpha, m, bounds)` 返回不可变
`PredictiveResult`：事件概率、`log_probability`、后验参数
`posterior_params = alpha + counts`，以及**该事件条件下**每类期望计数
`conditional_mean`。计算以本次传入（即拟合得到）的 `alpha` 为条件，**不重新
拟合、不引入非对称先验或 alpha 的超先验**；拟合端点状态仍保留在 `FitResult`
中不变。

**核心算法：按类别 × 累计计数的 log 域动态规划。** 把上面的分布按类别分解为
逐类因子 `w_k(c) = Gamma(a_k+c)/(Gamma(a_k) c!)`，公共归一化常数
`Z = Gamma(A+m)/(Gamma(A) m!)` 在事件概率比中消去。DP 逐类做受限卷积
（`dp[c]` = 前 k 类计数之和为 c 且均在各自界内的 log 总权重），时间
O(K·m·区间宽度)，`m <= 100`、`K <= 6` 时毫秒级，**不枚举 K 维计数单纯形**。
所有累加走 log-sum-exp，事件概率小到 ~1e-17 也不会下溢；
`Gamma(x+m)/Gamma(x)` 沿用证据模块的精确 log 和。

条件均值用“留一类重组”计算：
`E[y_k·1_事件] = sum_c c·w_k(c)·F_<k(·)·S_>k(·)`，其中 `F/S` 是同一套 DP 的
前向/后缀表，同样在 log 域完成；数值上 `sum_k conditional_mean = m`。

边界约定：

- `bounds` 为每类一个闭区间 `(lower_k, upper_k)`，要求
  `0 <= lower <= upper <= m`；长度必须等于 K。
- **上下界总和不可能达到 m 时直接返回零**（不运行 DP）：当
  `sum(lower) > m` 或 `sum(upper) < m`，界内不存在总和为 m 的计数向量。
- 空事件：`probability = 0.0`、`log_probability = -inf`、
  `is_empty_event = True`，`conditional_mean` 全部为 `nan`——零概率事件下的
  条件均值**未定义**，不用 0 冒充。
- **m = 0 单独处理**：唯一可能的计数向量是全 0，若所有 `lower_k = 0`
  （此时 upper 也只能是 0）则概率 1、条件均值全 0。
- `0 <= m <= MAX_PREDICTIVE_M = 100`；`m` 为 int，`bool` 按整数契约拒绝。

```python
from himeb import fit_alpha, posterior_predictive
fit = fit_alpha([(4, 0, 1), (0, 3, 0), (2, 2, 1)])
r = posterior_predictive((4, 0, 1), fit.alpha, 20, [(4, 14), (0, 6), (2, 10)])
r.probability          # 事件概率
r.log_probability      # log 域概率（极小事件不下溢）
r.conditional_mean     # 事件条件下的每类期望计数，和为 m
r.posterior_params     # (alpha+x_0, ...)，即 DM 后验参数
```


## 输入上限与错误语义

- 类数 `2 <= K <= 6`；组数 `1 <= G <= 40`；所有组长度必须一致。
- 计数为非负 `int`，单个计数上限 `2**53`（超出无法作为 float64 精确参与证据
  计算，直接拒绝）。
- 数值输入（alpha、目标函数返回值）拒绝 `NaN`/`±Infinity`；alpha 必须 > 0。
- **bool 不是整数**：计数、`grid_points`、`max_iter` 传 `True/False` 抛
  `TypeError`。
- 类型不符抛 `TypeError`，取值非法抛 `ValueError`；错误信息带可定位坐标
  （如 `group 1 class 0: ...`），不静默纠正、不钳制。
- `grid_points >= 3`、`max_iter >= 1`。
- 后验预测 `0 <= m <= 100`（`MAX_PREDICTIVE_M`）；每个界
  `0 <= lower_k <= upper_k <= m`，界对数必须等于 K。

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
  生产实现与参考的期望值由独立公式各自算出；“往返一致”仅作补充。
- **优化**：内部模在 t 空间定位到 `1e-12` 括号宽度；端点为精确函数值。
  拟合 alpha 与独立 4001 点密扫描的相对差 ≤ 2%（密扫描自身分辨率有限），
  证据在拟合点自洽到 1e-10。
- **后验矩**：纯乘除，相对误差为机器精度量级（~1e-15）；均值之和在
 1e-12 内归一。
- **后验预测 DP**：全部权重与卷积在 log 域（log-sum-exp）累加，公共归一化
  常数在概率比中消去；事件概率与条件均值在小 m（m ≤ 8）上与**全枚举独立
  参考** `dm_event_enumeration`（逐向量有限乘积，见 `tests/reference.py`）
  逐组随机 300 例对照（K=2..6、alpha∈[1e-4,1e4]、含空事件与 m=0），
  概率绝对差 ≤ 1e-9、条件均值绝对差 ≤ 1e-8；全范围事件概率为 1（1e-12）、
  条件均值之和为 m（1e-6 相对口径）；极小可行事件（~1e-17）经 log 域仍可
  重建。误差主体是 binary64 级别的舍入，未用宽容差掩盖边界语义。

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
- 只实现本题契约的数学子集（对称先验、单一共享 alpha、上述上限），不宣称
 任何行业标准协议兼容。
- **后验预测用复合分布而非“均值代入”**：后验 `Dirichlet(alpha+x)` 与多项
  采样复合为 Dirichlet-multinomial，正确反映 theta 不确定性带来的
  过离散；条件均值经“留一类”前向/后缀 DP 一次算出，不额外做 K 次重复 DP。
  预测严格以调用方给出的拟合 alpha 为条件（端点解保持其端点状态），本入口
  不重新拟合、不加非对称先验或超先验，也不新增拟合器。

## 已知限制

- 网格可能漏掉**窄于一个网格单元且峰顶落在格点之间**的模；`tests` 中有用
  3 点网格故意漏模、2001 点网格找到的对照用例。增大 `grid_points` 可缓解。
- 先验固定为对称形式、全组共享单一 alpha；不支持非对称基率、组级协变量或
  层级超先验。
- `flat` 情形无数据信息，按约定返回 `alpha=1.0` 供后验计算，不代表估计值。
- 参考展开式在 n 很大且 alpha 很小时交错相消明显，因此参考测试限定在
  题设的小计数范围；生产实现用整数 log 和不受此限。
- 独立 Simpson 积分参考在被积函数宽度 ≪ 1（如 a=1、x=0、n=400 时尖峰
  宽约 1/n）时，固定递归深度（30 层）不足以分辨，会严重低估；这是参考
  数值方法的限制（此时改用 `p(x)=1/(n+1)` 闭式事实核对），不是生产证据的
  误差。
- 模块只做计算，不提供 CLI 产品、网站、看板或持久化。
- 后验预测上限 `m <= 100`、`K <= 6`（与原类数上限一致）；更大 m/K 不在
  本契约内，直接拒绝而不静默截断。
- 空事件的条件期望未定义（返回 `nan`，不返回 0）；m=0 只有全零向量一种
  结果，已单独分支处理。
- DP 为精确 log 域卷积（除 log/exp 本身的 binary64 舍入外无离散化近似），
  但结果仍以单一拟合 alpha 为条件，未对 alpha 的拟合不确定性积分。

## 仓库结构

```
himeb/core.py        # 核心：证据、优化器、后验、校验
himeb/predictive.py  # 后验预测：Dirichlet-multinomial log 域 DP
himeb/__init__.py    # 公开接口
tests/reference.py   # 独立参考（展开 / 数值积分 / beta 矩 / 密扫描 / DM 枚举）
tests/test_*.py      # 校验、证据与后验对照、优化与层级测试
demo.py              # 固定输入演示
```
