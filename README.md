# himeb — 层级多项计数的确定性经验贝叶斯

纯 Python 标准库实现的小样本层级模型估计模块：多组 K 类计数共享一个
Dirichlet 先验，支持两种参数化：

- **对称**：`Dirichlet(alpha,…,alpha)`，单一浓度 `alpha`，在显式区间
  `[ALPHA_MIN, ALPHA_MAX]` 上对 `log(alpha)` 最大化边际似然；
- **每类别独立浓度**（本次新增）：`Dirichlet(alpha_1,…,alpha_K)`，各
  `alpha_k > 0` 独立，在盒子 `[ALPHA_MIN, ALPHA_MAX]^K` 上做**有限迭代**
  循环坐标上升，输出总对数边际似然、各类别浓度拟合和每组后验均值/方差/
  预测，并显式报告收敛或预算耗尽。

- **Python**：3.14.7（仅用标准库 `math`、`dataclasses`、`typing`）
- **依赖**：无。安装 Python 后无需 `pip install` 任何东西，完全离线。
- **平台**：Linux 原生运行；不访问网络、账号、数据库、中间件、云或 Docker。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 全部测试（含独立参考对照）
python demo.py                            # 固定输入演示，约 0.8 秒
```

## 模型与数学

对组 g 的计数 `x_g = (x_g1,…,x_gK)`，n_g = Σ x_gk：

```
theta_g ~ Dirichlet(alpha_1,…,alpha_K)
x_g | theta_g ~ Multinomial(n_g, theta_g)
```

单组对数证据（边际化 theta），记 A = Σ_k alpha_k：

```
log p(x|alpha) = log C(n; x)
  + logGamma(A) - logGamma(A + n)
  + Σ_k [ logGamma(x_k + alpha_k) - logGamma(alpha_k) ]
```

对称模型是 `alpha_k = alpha`、`A = K*alpha` 的特例，两条代码路径给出
逐位相同的证据值（测试断言相等）。

- 多组证据为各组之和；**空组（n=0）证据恒为 0，不参与求和**，其后验只返回
  先验。
- 对称退化事实：n=1 的组在对称先验下 `p(x=e_k|alpha) = 1/K` 与 alpha 无关；
  当所有非空组都是 n=1 时对称证据为精确常数 `-G1*log(K)`。
- **每类别模型下单抽不退化**：`p(x=e_k) = alpha_k/A` 依赖浓度向量。对称的
  两个类各观测一次时证据沿 `alpha_0=alpha_1` 光线恒为 `-2 log 2`（平坦岭，
  不是唯一最优点），拟合不会声称 `flat`。

后验为 `Dirichlet(alpha_k + x_k)`：

```
mean_k       = (alpha_k + x_k) / (A + n)
variance_k   = a_k (A_tot - a_k) / (A_tot^2 (A_tot + 1)),  a_k = alpha_k+x_k, A_tot = A+n
predictive_k = mean_k                  # 一步多项预测 = 后验均值
predicted    = argmax(predictive)      # 平局取最小下标
```

## 对称优化：括区间一维搜索

在 `t = log(alpha)` 的**显式闭区间** `[1e-4, 1e4]`（常量 `ALPHA_MIN/MAX`）
上最大化总对数证据：

1. 区间上 201 个（默认）均匀 t 网格求值；
2. 每个网格局部极大（含平台节点）作为候选模，在其相邻网格构成的括号
   `[t_{i-1}, t_{i+1}]` 内做**保括号黄金分割**细化（容差 `T_TOL=1e-12`）；
3. 两个端点始终作为候选并精确比较；
4. 所有候选都保留在结果里（值、alpha、括号、迭代次数、是否收敛），最高者胜出。

黄金分割在函数值顶平坦到机器精度时，对探针平局**同时收缩两端**
（`[a,b] -> [c,d]`），避免单向收缩让括号从最优点上漂走；细化结束返回括号
中点，保证返回点与最优点的 t 距离不超过 `T_TOL/2`。

`FitResult.status` 明确区分四种结局：

| status | 含义 |
|---|---|
| `interior` | 内部极大 |
| `lower_endpoint` / `upper_endpoint` | 极大落在区间端点（边界状态，不外延） |
| `flat` | **证明性无唯一解**：全空组（证据恒 0）或非空组全为 n=1（对称证据为常数）；不运行搜索，`alpha=1.0` |

**搜索预算耗尽与证明无解严格分开**：内部细化用尽 `max_iter` 仍未达到容差时
`converged=False`、`budget_exhausted=True`；`flat` 则 `converged=True` 且
`budget_exhausted=False`。

## 每类别浓度拟合：有限迭代循环坐标上升

`fit_alphas` 在盒子 `[ALPHA_MIN, ALPHA_MAX]^K` 上最大化总对数证据：

1. 从 `init`（默认全 1）开始，每轮 sweep 依次固定其余坐标，对单个
   `alpha_j` 用与对称拟合**同一个**网格+黄金分割引擎做一维最大化；
2. 每个一维切片严格凹：对 n_g>0 的组，二阶导
   `[ψ'(A) − ψ'(A+n_g)] + [ψ'(α_j+x_gj) − ψ'(α_j)] < 0`（trigamma 严格
   递减），故每步沿该坐标找到唯一的盒内全局最大；联合证据同样严格凹
   （Hessian = c·11ᵀ + 负定对角，c<0），驻点唯一，坐标上升不会困在局部极大；
3. **正浓度靠闭盒子保证，不做钳制**：盒外的点从不求值，`init` 必须在盒内；
4. 每轮 sweep 比较总证据增益，低于
   `max(1e-13, EVIDENCE_REL_TOL·|证据|)`（`EVIDENCE_REL_TOL=1e-13`）即判
   收敛；否则继续，最多 `sweeps`（默认 60）轮。

`AlphasFitResult.status` 三种结局：

| status | 含义 |
|---|---|
| `converged` | 增益在有限 sweep 内降到容差以下；`converged=True` |
| `max_sweeps_reached` | **用尽有限迭代预算仍未平稳**：`converged=False`、`budget_exhausted=True`，返回当前最优近似点与 `sweeps_used` |
| `flat` | 全空组（证据恒 0），不搜索，返回全 1 |

另有 `at_box: tuple[bool,...]` 逐类报告该浓度是否停在盒子边界。两种典型的
非内点情形都被测试覆盖：

- **从未被观测的类别**：其边际证据随 `alpha_k` 单调下降，坐标最大落在
  `ALPHA_MIN`（`at_box[k]=True`），属正常收敛；
- **边界 MLE**：若各组都把全部计数压在同一类，经验贝叶斯 MLE 逃向无穷
  浓度（点质量先验），任何有限盒子内都没有驻点，坐标上升持续爬向
  `ALPHA_MAX` 并在预算耗尽时如实报告 `max_sweeps_reached`，而不是谎称收敛。

## 公开接口

```python
from himeb import (
    # 对称（原有，签名与行为不变）
    log_evidence, log_evidence_one,   # 证据（任意正 alpha；区间只管优化器）
    fit_alpha,                        # 仅拟合 alpha -> FitResult
    maximize_on_interval,             # 通用多峰一维最大化引擎 f(alpha)
    posterior_group,                  # 单组后验均值/方差/预测
    analyze,                          # 拟合 + 全部组后验 -> EBResult
    # 每类别独立浓度（本次新增）
    log_evidence_dirichlet,           # 向量证据 log p(x|alpha_1..alpha_K)
    fit_alphas,                       # 坐标上升拟合 -> AlphasFitResult
    posterior_group_dirichlet,        # 向量先验下的单组后验
    analyze_dirichlet,                # 拟合 + 全部组后验 -> DirichletEBResult
    ALPHA_MIN, ALPHA_MAX, MAX_K, MAX_GROUPS,
    DEFAULT_SWEEPS, DEFAULT_DIRICHLET_GRID_POINTS,
)
```

结果均为不可变 dataclass（`Candidate`、`FitResult`、`GroupPosterior`、
`EBResult`、`AlphasFitResult`、`DirichletGroupPosterior`、`DirichletEBResult`）；
所有函数为纯函数、无状态，非法批次在产出任何结果前整体抛错，不存在部分变更。

```python
result = analyze_dirichlet([(4, 0, 1), (0, 3, 0), (0, 0, 0)])
result.fit.alphas            # 各类别拟合浓度 (alpha_1, ..., alpha_K)
result.fit.log_evidence      # 拟合点总对数边际似然
result.fit.status            # converged / max_sweeps_reached / flat
result.fit.converged         # 是否在有限迭代内达到增益容差
result.fit.sweeps_used       # 实际使用的坐标轮数
result.fit.at_box            # 每类是否停在 [ALPHA_MIN, ALPHA_MAX] 边界
result.posteriors[0].mean    # 后验均值
result.posteriors[0].variance
result.posteriors[0].predictive
```

## 输入上限与错误语义

- 类数 `2 <= K <= 6`；组数 `1 <= G <= 40`；所有组长度必须一致。
- 计数为非负 `int`，单个计数上限 `2**53`（超出无法作为 float64 精确参与证据
  计算，直接拒绝）。
- 浓度（标量 alpha 或向量 alphas 的每个分量、init 的每个分量）拒绝
  `NaN`/`±Infinity`，必须严格 > 0；`init` 分量还必须落在
  `[ALPHA_MIN, ALPHA_MAX]` 内（证据函数本身接受任意正浓度，盒子只管优化器）。
- alphas 向量长度必须等于该批次的 K，且自身在 2..6 之间。
- **bool 不是整数/数值**：计数、`grid_points`、`max_iter`、`sweeps`、浓度传
  `True/False` 抛 `TypeError`。
- 类型不符抛 `TypeError`，取值非法抛 `ValueError`；错误信息带可定位坐标
  （如 `group 1 class 0: ...`、`alpha class 1: ...`、`init class 1: ...`），
  不静默纠正、不钳制。
- `grid_points >= 3`、`max_iter >= 1`、`sweeps >= 1`。

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
  - 二项（K=2）似然因子做多项式展开后逐项 beta 积分（带符号 log-sum-exp），
    含对称与一般非对称 `(a,b)` 形式。该交错和在题设“小计数”n ≤ 8、
    alpha 覆盖 `[1e-4,1e4]` 的**适度区域**与生产实现绝对差
    **≤ 2e-8**（此 2e-8 是交错求和**参考本身**的误差；a≫b、n 稍大时交错
    相消使参考失效，已在“已知限制”说明）；
  - **Pólya 瓮顺序因子**（`evidence_polya` / 一般 K 类 `evidence_polya_k`）：
    把 beta/Polya 分布的概率写成一串**正分数**连乘（用 `-log1p` 防止大
    alpha 下的相消），完全没有交错求和，因此在展开失效的极端角落（如
    a0=1e4、a1=1e-4、n≤8）仍准确，与生产证据全网格实测绝对差
    **≤ 2e-14**；期望值与生产实现分别独立计算；
  - 第三条独立路线为 [0,1] 上自适应 Simpson 直接积分（对称与非对称
    `evidence_quadrature_ab`；被积函数不单点尖峰时有效），绝对差 **≤ 2e-9**；
  - 后验矩另用独立的 beta 积分比 `B(a+1,b)/B(a,b)` 交叉核对（含非对称形式）；
  - 一维标量拟合与独立 4001 点密扫描对照；**每类别 K=2 拟合**与独立的
    201×201 二维 log 密扫描（用 Pólya 路线）对照到该扫描自身分辨率，
    并额外用独立 digamma 检验三维内点处每个偏导 ≈ 0（≤ 1e-5）。
  “往返一致”仅作补充。
- **优化**：内部模在 t 空间定位到 `1e-12` 括号宽度；端点为精确函数值。
  拟合 alpha 与独立 4001 点密扫描的相对差 ≤ 2%（密扫描自身分辨率有限），
  证据在拟合点自洽到 1e-10。
- **每类别拟合**：坐标点的一维切片由同一引擎定位；默认预算下内点批次
  数十轮 sweep 收敛（实测约 30–60 轮），独立 digamma 梯度 ≤ 6e-6；类别置换
  两次独立拟合的浓度相对差 ≤ 1e-5、证据差 ≤ 1e-10（有限迭代的固有分辨率，
  已在测试中按此口径断言而非用宽容差掩盖）。
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
- **每类别浓度用坐标上升而非 K 维网格**：K≤6 时联合证据严格凹（唯一驻点），
  循环坐标上升把 K 维问题化为 K 次复用的一维保括号搜索，代价是
  `O(sweeps·K·grid)` 次证据求值；有限预算、增益容差、逐类贴边标志都写进
  返回值，不把“还没跑完”伪装成收敛。
- **正浓度不外推、不钳制**：所有优化只在闭盒子 `[1e-4,1e4]^K` 内求值；
  MLE 在无穷远（如所有组纯落同一类）时报告预算耗尽与爬升方向，由调用方
  决定加大 `sweeps` 还是解释为边界 MLE，而不是静默返回一个盒边点冒充驻点。
- 只实现本题契约的数学子集：对称标量先验与每类别独立浓度、上述上限；
  按本次范围**不做**组间树状共享、多层浓度联合估计，也不支持组级协变量或
  层级超先验，不宣称任何行业标准协议兼容。

## 已知限制

- 网格可能漏掉**窄于一个网格单元且峰顶落在格点之间**的模；`tests` 中有用
  3 点网格故意漏模、2001 点网格找到的对照用例。增大 `grid_points` 可缓解。
- 对称先验/每类别先验均不含非对称基率的超先验结构；`alpha_k` 之间独立，
  不建模浓度的层级共享（树状/多层联合估计不在本次范围内）。
- `flat` 情形无数据信息，按约定返回浓度全 1（对称为 `alpha=1.0`）供后验
  计算，不代表估计值。
- **每类别拟合是有限迭代坐标上升**：返回点是驻点的近似（实测梯度
  ~1e-6 量级、置换重复拟合相对差 ~1e-5），不是解析 MLE；当经验贝叶斯 MLE
  在无穷远边界时，任何有限 `sweeps` 都会以 `max_sweeps_reached` 结束——
  这是如实报告而非错误。坐标上升在近平坦岭（如对称数据的对称光线）上会
  之字形慢行；对称初始化可精确停在光线上。
- 参考展开式在 n 很大或 `a0`/`a1` 极度悬殊时交错相消明显（如
  a0=1e4、a1=0.01、n≥2 时参考可整体错号），因此展开参考测试限定在其准确
  包络内；极端角落改用无交错求和的 Pólya 瓮分数连乘参考核对。
- 独立 Simpson 积分参考在被积函数宽度 ≪ 1（如 a=1、x=0、n=400 时尖峰
  宽约 1/n）时，固定递归深度（30 层）不足以分辨，会严重低估；这是参考
  数值方法的限制（此时改用 `p(x)=1/(n+1)` 闭式事实核对），不是生产证据的
  误差。
- 模块只做计算，不提供 CLI 产品、网站、看板或持久化。

## 仓库结构

```
himeb/core.py        # 核心：对称/每类别证据、两套优化器、后验、校验
himeb/__init__.py    # 公开接口
tests/reference.py   # 独立参考（展开 / Pólya 瓮 / 数值积分 / beta 矩 / 密扫描）
tests/test_*.py      # 校验、证据与后验对照、优化、每类别浓度测试
demo.py              # 固定输入演示
```
