# 从矩阵分析视角理解mHC和AttnRes的信息量

英文版见 [residual_analysis_en.md](residual_analysis_en.md)

在Kimi-K3权重发布之际，又仔细读了一遍[Attention Residuals(AttnRes)论文](https://arxiv.org/abs/2603.15031)，反过来也更理解了[DS-V4 mHC](https://arxiv.org/abs/2512.24880)的数学本质。

AttnRes论文的"6.2 Residual Connections as Structured Matrices"章节给出了Residual Connection各类改进变种(包括mHC和block AttnRes在内)的有趣数学特性，但没有详细展开。本文补上数学证明、代码验证、具体样例、并聊聊背后的数学物理渊源。

改进残差连接还有更多工作在infra层面，尤其是pipeline并行、重计算等等，待后续填坑。

# 目录

- [数学记号](#notation)
- [残差连接的矩阵视角](#matrix-view)
  - [Residual, mHC, AttnRes的递归版本与矩阵版本](#recurrent-and-matrix)
  - [和Linear Attention的惊人相似](#linear-attention)
- [数值实验: 定量衡量跨层传递的信息量 (off-diagonal rank)](#numerical-rank)
- [更严谨的定义与证明 (semi-separable order)](#semisep-order)
  - [半可分矩阵的历史渊源](#semisep-history)
  - [mHC的SS order推导](#mhc-proof)
  - [AttnRes的SS order推导](#attnres-proof)
- [参考文献](#references)

<a id="notation"></a>
# 数学记号

各论文的习惯记号并不一致：mHC原论文的扩展系数记作 $n=4$，在AttnRes论文里是 $m$；hidden state有的写 $x$ 有的写 $h$，mHC的投影矩阵有时叫 $\mathcal H_{\text{pre}}/\mathcal H_{\text{post}}/\mathcal H_{\text{res}}$、有时叫 $\alpha/\beta/A$……为免混淆，本文**统一用AttnRes论文的记号**，并在表里标注它跟mHC原论文的对应：

| 记号 | 含义 | 与mHC原论文的对应 |
|---|---|---|
| $L$ | Transformer总层数 (FFN和Attention各算一层) | — |
| $l,\,i$ | 层的下标； $l$ 为**接收层**、 $i$ 为**来源层**，恒有 $i\le l$ | — |
| $\mathbf h_l\in\mathbb R^{d}$ | 第 $l$ 层的输入 hidden state | — |
| $f_l(\cdot)$ | 第 $l$ 层的非线性变换 (Attention或FFN的forward，不含residual相加) | — |
| $\mathbf v_l\in\mathbb R^{d}$ | 第 $l$ 层输出 $f_l(\mathbf h_l)$；约定 $\mathbf v_0=\mathbf h_1$ 为embedding | $v_i$ |
| $d$ | hidden dimension | — |
| $m$ | mHC的stream扩展系数 (DS-V4取 $4$) | 原文记作 $n$ |
| $\mathbf M\in\mathbb R^{L\times L}$ | **Depth Mixing Matrix**，(严格)下三角； $\mathbf M_{l,i}$ = 接收层 $l$ 分给来源输出 $\mathbf v_i$ 的权重， $\mathbf h_l=\sum_{i\lt l}\mathbf M_{l,i}\mathbf v_i$ | 论文记作 $\mathbf M_{i\to l}$ |
| $\boldsymbol\alpha_l,\,\boldsymbol\beta_i\in\mathbb R^{m}$ | mHC的读 / 写投影向量 | $\mathcal H_{\text{pre}}=\alpha$ (把 $m$ 条stream混成输入)， $\mathcal H_{\text{post}}=\beta$ (把输出写回stream) |
| $\mathbf A_k\in\mathbb R^{m\times m}$ | mHC的stream转移矩阵 (被约束为doubly stochastic) | $\mathcal H_{\text{res}}$ |
| $\mathbf w_l,\,\mathbf k_i\in\mathbb R^{d}$ | Full AttnRes的query / key | — |
| $\phi(\mathbf w,\mathbf k)=\exp(\mathbf w^{\!\top}\mathbf k)$ | AttnRes的(softmax) kernel | — |
| $N,\,S$ | Block AttnRes的block数 / block size， $N=L/S$ | — |
| $t$ | 截断位置(cut)：把 $1..t$ 层与 $t{+}1..L$ 层切开 | — |
| SS order | semiseparable order：所有off-diagonal block $\mathbf M[t{:},{:}t]$ 中rank的最大值 | — |

<a id="matrix-view"></a>
# 残差连接的矩阵视角

<a id="recurrent-and-matrix"></a>
## Residual, mHC, AttnRes的递归版本与矩阵版本

先看经典残差。把递推 $\mathbf h_l=\mathbf h_{l-1}+\mathbf v_{l-1}$ 展开，就是 $\mathbf h_l=\sum_{i\lt l}\mathbf v_i$：第 $l$ 层的输入，就是它之前所有层输出的等权求和。写成一个矩阵方程，把全部层的 $\mathbf h_l$ 和 $\mathbf v_i$ 一次性联系起来：

```math
\mathbf{h}_l=\sum_{i\lt l}\mathbf{v}_i
\quad\Longleftrightarrow\quad
\mathbf{H}=\mathbf{M}^{\mathrm{res}}\mathbf{V},\qquad
\mathbf{M}^{\mathrm{res}}=\begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
\vdots & & \ddots & \ddots & \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}.
```

这个严格下三角的全 1 矩阵就是离散的累加(cumulative-sum)算子，形式上类似sequence维度里attention的causal mask。numpy确认展开等于矩阵乘：

```python
import numpy as np
rng = np.random.default_rng(0)
L, d = 6, 5
V = rng.standard_normal((L, d))
H_rec = np.zeros((L, d))
for l in range(1, L):
    H_rec[l] = H_rec[l-1] + V[l-1]              # 逐层递归 h_l = h_{l-1} + v_{l-1}
M_res = np.tril(np.ones((L, L)), -1)            # 严格下三角全 1
assert np.allclose(H_rec, M_res @ V)            # 递归展开 == 矩阵乘   -> 成立
```

> **脚注（关于下三角的对角线）。** 本文把接收层和来源层用同一个层计数（ $\mathbf h_l=\sum_{i\lt l}\mathbf M_{l,i}\mathbf v_i$），所以 $\mathbf M$ 是严格下三角：对角元 $\mathbf M_{l,l}=0$，因为第 $l$ 层读不到自己还没算出来的输出 $\mathbf v_l$。AttnRes 论文 Figure 9 则把 embedding $\mathbf v_0$ 对到第 0 列、 $\mathbf h_1$ 对到第 0 行，行、列下标错开一位，于是"读上一层"的权重（普通残差为 1、mHC 为 $\boldsymbol\beta_{l-1}^{\!\top}\boldsymbol\alpha_l$）落到了主对角线上，画出来就成了带对角的下三角（普通残差恰好是可逆的全 1 cumsum 矩阵）。两者是同一个映射，只差列下标平移一位；后文的 SS order 只看严格下三角的 off-diagonal 子块，与对角元无关，两种约定结论一致。

推广一下：**任何残差变种都能写成 $\mathbf h_l=\sum_{i\lt l}\mathbf M_{l,i}\,\mathbf v_i$**，其中 $\mathbf M_{l,i}$ 是接收层 $l$ 分给来源层 $i$ 输出的权重。这个下三角矩阵 $\mathbf M$ 叫 **Depth Mixing Matrix**。变种的区别就在于 $\mathbf M$ 怎么来——固定常数、可学参数、还是随输入动态计算：

![Depth Mixing Matrix from AttnRes paper](fig/depth_mixing_def.png)

普通残差对应的 $\mathbf M$ 全是 1；mHC、AttnRes，以及早年的[Highway Residual](https://arxiv.org/abs/1612.07771)等，都是给这张矩阵换上更有表达力的权重。借用AttnRes论文的总结表格：

![AttnRes论文, Table 5](fig/residual_compare_table.png)

论文给出了划红线几种变种对应的 $\mathbf M$，比全 1 下三角丰富得多：

![AttnRes论文, Figure 9](fig/matrix_view.png)

**Full AttnRes：深度上的softmax attention。** 一种直观且表达力强的改进：给每个接收层 $l$ 配一个可学的query $\mathbf w_l$，对之前所有层的输出做一次softmax注意力，权重由动态输入决定：

$$\mathbf M^{\text{Attn}}_{l,i}=\frac{\exp\!\big(\mathbf w_l^{\!\top}\mathbf k_i\big)}{\sum_{j\lt l}\exp\!\big(\mathbf w_l^{\!\top}\mathbf k_j\big)},\qquad \mathbf k_i=\mathrm{RMSNorm}(\mathbf v_i).$$

（ $\mathbf k_i$ 上的RMSNorm是工程细节，跟本文关心的矩阵结构无关，后面不再提。）这就是Figure 9里的Full AttnRes：一整行都是softmax权重，无low-rank约束。

**mHC：把单条residual流拓宽成 $m$ 条stream。** mHC(或原始Hyper-Connections)不在depth维度用softmax，而是让残差流变宽：用一个 $\mathbf H_l\in\mathbb R^{d\times m}$ 的state沿深度传递，每层做三件事——**读**（用 $\boldsymbol\alpha_l$ 把 $m$ 条stream混成一个 $d$ 维输入 $\mathbf H_{l-1}\boldsymbol\alpha_l$）、**写**（把 $f$ 的输出用 $\boldsymbol\beta_l^{\!\top}$ 分发回各条stream）、**转移**（右乘 $\mathbf A_l$ 让 $m$ 条stream彼此混合）：

$$\mathbf H_l=\mathbf H_{l-1}\mathbf A_l+\underbrace{f_{l-1}\!\big(\mathbf H_{l-1}\boldsymbol\alpha_{l-1}\big)}_{=\,\mathbf v_{l-1}\in\mathbb R^{d}}\boldsymbol\beta_{l-1}^{\!\top}.$$

把递归展开（来源 $i$ 写进来之后，往后每走一层就被右乘一个 $\mathbf A$），读出 $\mathbf h_l=\mathbf H_{l-1}\boldsymbol\alpha_l$ 时，每个来源 $\mathbf v_i$ 前面的系数就是

$$\boxed{\ \mathbf M^{\text{mHC}}_{l,i}=\boldsymbol\beta_i^{\!\top}\,\mathbf A^{\times}_{i+1\to l}\,\boldsymbol\alpha_l\ },\qquad
\mathbf A^{\times}_{i+1\to l}:=\mathbf A_{i+1}\mathbf A_{i+2}\cdots\mathbf A_l\ \ (l{=}i\text{ 时约定为单位阵}).$$

即AttnRes论文Eq.(10)的形式（ $\pm1$ 的下标差别取决于"转移"与"写入"谁先谁后，不影响结论）。 $L=4$ 时它长成Figure 9里(m)HC那个下三角。注意矩阵的特殊形状：一个行向量 $\boldsymbol\beta_i^{\!\top}$、一段矩阵连乘 $\mathbf A^{\times}$、一个列向量 $\boldsymbol\alpha_l$ 夹在一起。含义是：任意两层 $i\to l$ 之间的耦合，都由同一串 $m\times m$ 转移矩阵生成，只是首尾换上各层自己的读 / 写向量。这个结构是后面推导的关键。

用numpy简单验证递归vs矩阵形式的等价（这里 $\mathbf A$ 用随机矩阵即可；mHC实际会用Sinkhorn迭代把 $\mathbf A$ 约束成doubly stochastic，那只影响深层的数值稳定性，跟矩阵结构无关）：

```python
m = 4
alpha = rng.standard_normal((L, m)); beta = rng.standard_normal((L, m))
A = rng.standard_normal((L, m, m)) / np.sqrt(m)
def Aprod(i, l):                                 # A_{i+1} A_{i+2} ... A_l  (l<=i 时为 I)
    Q = np.eye(m)
    for k in range(i+1, l+1): Q = Q @ A[k]
    return Q
M_mhc = np.array([[beta[i] @ Aprod(i, l) @ alpha[l] if i <= l else 0.
                   for i in range(L)] for l in range(L)])
H, x = np.zeros((d, m)), np.zeros((L, d))
for l in range(L):
    H = H @ A[l] + np.outer(V[l], beta[l])       # 转移 + 写入
    x[l] = H @ alpha[l]                          # 读出
assert np.allclose(x, M_mhc @ V)                 # 逐层递归 == unrolled 矩阵形式   -> 成立
```

<a id="linear-attention"></a>
## 和Linear Attention的惊人相似

有意思的是，这张跨*layer维度*的矩阵 $\mathbf M$，和*跨sequence维度*attention里那张masked score matrix，有几乎等同的数学性质和变种形态。

对照Mamba2论文的矩阵视角，mHC的 $\mathbf M$ 和它长得几乎一模一样（后文证明确实等价）：

![Figure 2 from Mamba2 paper](fig/mamba_ssm_visual.png)

Linear attention的基本知识可参考[Mamba2博客](https://goombalab.github.io/blog/2024/mamba2-part2-theory/#causal-linear-attention)和[DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/)。和本文有关的核心结论：causal linear attention有两种等价写法——矩阵并行形式，和线性递推形式：

$$
\underbrace{\mathbf O=\big(\mathbf L\circ\mathbf Q\mathbf K^{\!\top}\big)\mathbf V}_{\text{矩阵并行形式}}
\qquad\Longleftrightarrow\qquad
\underbrace{\mathbf S_t=\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^{\!\top},\quad \mathbf o_t=\mathbf S_t\mathbf q_t}_{\text{线性递推形式，state }\mathbf S\in\mathbb R^{d\times d}}.
$$

"矩阵mixer $\Leftrightarrow$ 有限state的递归"这个对偶，是semiseparable矩阵理论的核心（Mamba2 SSD）。常见的改进方向：

- **加decay / gating**（GLA、Mamba2）： $\mathbf S_t=a_t\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^{\!\top}$，mask变成带衰减的版本；
- **加matrix transition + rank-1 update**（DeltaNet）： $\mathbf S_t=\mathbf S_{t-1}(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^{\!\top})+\beta_t\mathbf v_t\mathbf k_t^{\!\top}$。这个"state × 转移 + 秩1写入"的形状，十分类似mHC的递归公式——无非把sequence轴换成了depth轴： $\mathbf A_k(\mathcal H_{\text{res}})$ 当DeltaNet的转移矩阵， $\boldsymbol\beta f(\cdot)$ 当那个秩1写入。

精辟类比：

> **mHC 就是"旋转90°"、跑在depth维度上的 (gated / DeltaNet) linear attention；Full AttnRes 则是旋转90°、跑在depth维度上的 softmax attention。** 二者之别，是长序列建模的老问题——**linear vs softmax**。

"旋转90°"这个观点最早是[Attention Residuals 回忆录](https://spaces.ac.cn/archives/11664)提出的——"HC 相当于'旋转90度'的 DeltaNet"，把linear attention那套sequence维度玩熟了的招式（gating、delta rule、matrix-valued state、递归与矩阵两种等价形式）搬到了depth维度；上面的"精辟类比"只是把细节补全（mHC 对 gated/DeltaNet linear、Full AttnRes 对 softmax）。严格来说，"DeltaNet用到depth维度"对应的是[Deep Delta Learning(DDL)论文](https://arxiv.org/abs/2601.00417)，也在前面那张变种表里。

<a id="numerical-rank"></a>
# 数值实验: 定量衡量跨层传递的信息量 (off-diagonal rank)

有了这么多变种，怎么定量、直观地衡量它们比经典残差强在哪？做个朴素的实验：在 $L$ 层网络里选一个截断位置 $t$，看" $1..t$ 层"往" $t{+}1..L$ 层"传信息有多灵活。这份灵活度，由Depth Mixing Matrix的一个*非对角子块 (off-diagonal block)* $\mathbf M[t{:},\,{:}t]$（下三角的左下角，行 $\ge t$、列 $\lt t$）决定：系数全是 1，后面的层就只能无差别地照单全收（经典残差）；系数越自由，"挑着吸收"的空间越大。而**子块 $\mathbf M[t{:},\,{:}t]$ 的 rank 衡量了这份自由度**：

- 普通残差： $\mathbf M[t{:},\,{:}t]$ 全是 1，写成外积 $\mathbf 1\mathbf 1^{\!\top}$，rank 恒为 1，自由度最低。
- Full AttnRes： $\mathbf M[t{:},\,{:}t]$ 由跨层softmax算出，不受low-rank约束。该块尺寸 $(L{-}t)\times t$，故 $\mathrm{rank}\le\min(t,\,L{-}t)$，在 $t=L/2$ 处取到最大 $\lfloor L/2\rfloor$，自由度最高。
- mHC：最大off-diagonal rank可以证明**恰好等于 $m$**（下一节证）—— $m$ 越大越灵活。
- Block AttnRes：夹在残差与Full AttnRes之间， $N$ 越大越灵活（后文详述）。

先用数值验证。`numpy.linalg.matrix_rank` 内部就是对子块做SVD、数非零奇异值，扫遍所有截断 $t$：

```python
import numpy as np
rng = np.random.default_rng(0)
L = 16
def rms(x): return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)

# (1) residual: 全 1 下三角
M_res = np.tril(np.ones((L, L)))

# (2) mHC, m 条 stream: M[l,i] = beta_i^T (A_{i+1}...A_l) alpha_l
def M_mhc(m):
    a = rng.standard_normal((L, m)); b = rng.standard_normal((L, m))
    A = rng.standard_normal((L, m, m)) / np.sqrt(m)
    def P(l, i):
        Q = np.eye(m)
        for k in range(i+1, l+1): Q = Q @ A[k]
        return Q
    return np.array([[b[i] @ P(l, i) @ a[l] if i <= l else 0.
                      for i in range(L)] for l in range(L)])

# (3) Full AttnRes: 深度上的 softmax
w = rng.standard_normal((L, 32)); k = rms(rng.standard_normal((L, 32)))
Lo = np.where(np.tril(np.ones((L, L))) > 0, w @ k.T, -np.inf)
M_attn = np.exp(Lo - np.nanmax(Lo, 1, keepdims=True))
M_attn = np.where(np.isfinite(Lo), M_attn, 0.); M_attn /= M_attn.sum(1, keepdims=True)

# semiseparable order = 遍历所有 cut, 取 off-diagonal block 的最大 rank
order = lambda M: max(np.linalg.matrix_rank(M[t:, :t], tol=1e-9) for t in range(1, L))
print("residual :", order(M_res))         # 1
print("mHC m=2  :", order(M_mhc(2)))       # 2
print("mHC m=4  :", order(M_mhc(4)))       # 4
print("AttnRes  :", order(M_attn))         # 8 = floor(L/2)
```

把截断 $t$ 从 $1$ 扫到 $L$，再对中点截断的子块做SVD看奇异值谱，矩阵的表达能力就一目了然（(a) rank随cut变化；(b) 中点子块的奇异值谱；(c) Block AttnRes的order如何随参数在 $[N,N{+}S]$ 内变化，见后文）：

![off-diagonal rank 数值扫描](fig/rank_numerical.png)

结果和AttnRes论文（未给证明）的结论一致：residual恒为 1，mHC饱和在stream数 $m$，Full AttnRes爬到最大的 $\lfloor L/2\rfloor$，Block AttnRes则夹在 $[N,N{+}S]$：

![rank_compare](fig/rank_compare.png)

细节澄清：

- 本文和AttnRes论文所说的rank，指的是 $\mathbf M$ 某个 off-diagonal 子块的秩，而非整个矩阵的秩。整个 $\mathbf M$ 的秩对这个比较没意义（只反映有没有对角 / 次对角元），有信息量的是子块 $\mathbf M[t{:},\,{:}t]$ 能有多高的rank。
- 截断 $t$ 可取 $1$ 到 $L{-}1$，我们遍历所有 $t$、取rank最大的那个子块，得到一个整数metric。为啥用 `max` 而不用 `mean`？当经验指标都行，但 `max` 有更漂亮的理论性质，下节展开。

<a id="semisep-order"></a>
# 更严谨的定义与证明 (semi-separable order)

上文所用的"所有下三角off-diagonal子块里rank的最大值"，在数值分析里有个正式名字：**semi-separable order**。借用Mamba2论文的定义：

![Definition 3.1 from Mamba2 paper](fig/semisep_def.png)

有的AI论文里也叫它"semiseparable rank"，本文沿用数值分析文献的叫法"semiseparable order" / "SS order"。

<a id="semisep-history"></a>
## 半可分矩阵(Semi-separable matrix)的历史渊源

一个 $n\times n$ 下三角矩阵是 **$k$-semiseparable** 的（Mamba2 定义里把这个 order 记作 $N$；为跟矩阵大小 $n$ 以及后面 Block AttnRes 的 $N$ 区分，这里改用 $k$），意思是：对角线以下的任意子块，rank都不超过 $k$——也就是每个子块都能写成一个"瘦长"的外积 $\mathbf U\mathbf V^{\!\top}$（ $\mathbf U,\mathbf V$ 只有 $k$ 列）。当 $k\ll n$ 时，矩阵信息高度集中：原本 $O(n^3)$ 的矩阵乘、求逆、QR 等运算，都能找到 $O(n)$ （更精确是 $O(kn)$ ）量级的快速算法。

这类矩阵经典的来源，是**一维差分算子的逆**（Gantmacher & Krein；Vandebril 等，见文末参考文献）。举个例子：一根两端固定的弦受载荷 $f$，稳态形变 $u$ 满足泊松方程 $-u''=f$。用三点差分离散二阶导，得到一个稀疏的三对角矩阵 $\mathbf T=\mathrm{tridiag}(-1,2,-1)$（每个节点只跟左右邻居耦合，很局域）。它的逆却是**稠密**的，且有干净的闭式解：

$$\big(\mathbf T^{-1}\big)_{ij}=\frac{\min(i,j)\,\big(n{+}1-\max(i,j)\big)}{n+1}.$$

妙处在于：取任意一个off-diagonal子块（左半节点 $j$ 对右半节点 $i$ 的影响， $i\gt j$，此时 $\min=j,\max=i$），元素恰好拆成"只跟行有关"乘"只跟列有关"：

$$\big(\mathbf T^{-1}\big)_{ij}=\underbrace{\tfrac{n{+}1-i}{n+1}}_{\text{只含行 }i}\cdot\underbrace{j}_{\text{只含列 }j}\quad(i\gt j),$$

整个子块就是一个列向量乘一个行向量——外积、rank 1。所以 $\mathbf T^{-1}$ 是 1-semiseparable。物理直觉也朴素：载荷和固定端之间没有外力、弦是直线，载荷左边所有点都落在同一条直线段上，形变形状相同、只差一个斜率标量——一个数就把载荷的全部影响传了过去，这条通道就是rank 1。

回到残差：普通残差的 $\mathbf M$ 也是这个故事。 $\mathbf h_l=\mathbf h_{l-1}+\mathbf v_l$ 是离散的一阶差分方程，它的逆正好是全 1 下三角（cumsum），同样是 1-semiseparable。两个例子放一起看（左列是稀疏局域的差分算子，中列是它稠密的逆，右列是单位脉冲的响应）：

```python
import numpy as np, sympy as sp
n = 6
T = sp.Matrix(n, n, lambda i, j: 2 if i == j else (-1 if abs(i-j) == 1 else 0))
Tinv = np.array(T.inv().tolist(), float)
c = n // 2; blk = Tinv[c:, :c]                       # 左半 -> 右半 的 off-diagonal 块
u = (n+1 - np.arange(c+1, n+1)) / (n+1); v = np.arange(1., c+1)
assert np.allclose(blk, np.outer(u, v))              # 块 == 外积 u v^T  -> rank 1  (泊松/扩散)

D = np.eye(n) - np.diag(np.ones(n-1), -1)            # 一阶差分
assert np.allclose(np.linalg.inv(D), np.tril(np.ones((n, n))))   # D^-1 = 全 1 下三角 = residual
```

![两个 order-1 的半可分矩阵：残差 = 一阶差分之逆，泊松 = 二阶差分之逆](fig/pde_semisep.png)

两者rank都是 1，但值不同：残差是纯积分器，脉冲响应是阶跃（某层的信号往后每层都以权重 1 累加，永不衰减）；泊松的逆则是两端夹住的"帐篷"、线性衰减到边界。order只数"有几条通道"，不管衰减快慢。而残差的那条通道就是**running sum**（residual stream本身），这也是为什么整套残差只需一次 $O(L)$ 前向就能算完——跟linear attention / SSM能做到线性复杂度，是同一个道理。

<a id="mhc-proof"></a>
## mHC的SS order推导

来证明 **mHC的SS order恰好是 $m$**。回到前面unroll出的

$$\mathbf M^{\text{mHC}}_{l,i}=\boldsymbol\beta_i^{\!\top}\big(\mathbf A_{i+1}\cdots\mathbf A_l\big)\boldsymbol\alpha_l.$$

> **Side note.** 这种"行向量–矩阵连乘–列向量"的写法，在数值分析里叫 SSS（sequentially semiseparable）的 generator form，是半可分矩阵的标准表示之一。

固定一个截断 $t$，看子块 $\mathbf M[t{:},\,{:}t]$（行 $l\ge t$、列 $i\lt t$）。里面每个 $(l,i)$ 都满足 $i\lt t\le l$，那串转移连乘恰好横跨 $t$，可以在 $t$ 处一刀切开、左右分家：

$$\mathbf A_{i+1}\cdots\mathbf A_l=\underbrace{\big(\mathbf A_{i+1}\cdots\mathbf A_t\big)}_{\text{只依赖列 }i}\ \underbrace{\big(\mathbf A_{t+1}\cdots\mathbf A_l\big)}_{\text{只依赖行 }l}.$$

切点 $t$ 固定，左半只跟 $i$ 有关、右半只跟 $l$ 有关。于是每个元素成了两个 $m$ 维向量的内积：

$$\mathbf M_{l,i}=\mathbf v_i^{\!\top}\mathbf u_l,\qquad
\mathbf u_l=\big(\mathbf A_{t+1}\cdots\mathbf A_l\big)\boldsymbol\alpha_l,\quad
\mathbf v_i=\big(\mathbf A_{i+1}\cdots\mathbf A_t\big)^{\!\top}\boldsymbol\beta_i.$$

把 $\mathbf u_l$ 摞成 $\mathbf U\in\mathbb R^{(L-t)\times m}$、 $\mathbf v_i$ 摞成 $\mathbf V\in\mathbb R^{t\times m}$，整个子块就是 $\mathbf M[t{:},\,{:}t]=\mathbf U\mathbf V^{\!\top}$——一个内维只有 $m$ 的外积，所以 $\mathrm{rank}\le m$，且对每个 $t$ 都成立。这对应Mamba2论文的Lemma 3.3：

![Lemma 3.3 of Mamba 2 Paper](fig/semisep_quick_proof.png)

反过来，只要generator不退化（几乎所有情形）， $\mathbf U$、 $\mathbf V$ 都是列满秩的，子块的rank就恰好是 $m$、不会更低——严格说，"某个 $m\times m$ 子式非零"是一个不恒为零的多项式，只在测度零的参数上取零。两头一夹，**SS order $=m$**。直觉：截断处只有一个 $m$ 维的state在浅层和深层之间流过，所以最多 $m$ 份浅层信息能活着到达深层。

用sympy一次把两件事都验掉（符号确认子块 $=\mathbf U\mathbf V^{\!\top}$、且 $m\times m$ 子式行列式是非零多项式），再用随机数值确认rank真的是 $m$（而非 $m{-}1$）：

```python
import sympy as sp, numpy as np
m, L, t = 2, 5, 3
A = [sp.Matrix(m, m, lambda r, c: sp.Symbol(f'a{k}_{r}{c}')) for k in range(L)]
B = [sp.Matrix(m, 1, lambda r, c: sp.Symbol(f'b{k}_{r}')) for k in range(L)]
C = [sp.Matrix(m, 1, lambda r, c: sp.Symbol(f'c{k}_{r}')) for k in range(L)]
def Ap(i, l):                                    # A_{i+1} ... A_l
    P = sp.eye(m)
    for k in range(i+1, l+1): P = P * A[k]
    return P
M   = sp.Matrix(L, L, lambda l, i: (B[i].T * Ap(i, l) * C[l])[0] if i <= l else 0)
blk = M[t:, :t]
U = sp.Matrix.hstack(*[Ap(t, l) * C[l] for l in range(t, L)])         # 行因子 u_l  (m x (L-t))
V = sp.Matrix.hstack(*[Ap(i, t).T * B[i] for i in range(t)])         # 列因子 v_i  (m x t)
assert sp.simplify(blk - U.T * V) == sp.zeros(*blk.shape)            # 子块 == U^T V  -> rank <= m
assert sp.simplify(M[2:4, 0:2].det()) != 0                          # m x m 子式是非零多项式 -> 泛型下 rank = m

rng = np.random.default_rng(0); mm, LL, ct = 3, 12, 6
a = rng.standard_normal((LL, mm)); b = rng.standard_normal((LL, mm)); Ar = rng.standard_normal((LL, mm, mm))
def P(i, l):
    Q = np.eye(mm)
    for k in range(i+1, l+1): Q = Q @ Ar[k]
    return Q
Mn = np.array([[b[i] @ P(i, l) @ a[l] if i <= l else 0. for i in range(LL)] for l in range(LL)])
assert np.linalg.matrix_rank(Mn[ct:, :ct], tol=1e-9) == mm          # 数值: 子块 rank 恰为 m
```

这就是"state size 为 $m$ 的linear-attention / SSM恰好是 $m$-semiseparable"在depth轴上的翻版。只有在极端特殊的情况下——比如所有 $\boldsymbol\beta_i$ 恰好共线、或 $\mathbf A$ 把state压进了一个更低维的子空间——order才会掉到 $m$ 以下；这属于退化情形，训练好的mHC只要stream没塌缩就碰不上，所以说order"等于 $m$"是泛型意义下的结论。

另外，以上结论对**mHC-lite同样成立**。[mHC-lite](https://arxiv.org/abs/2601.05732)保留了mHC的整个递归，只是把 $\mathbf A$ 的构造从Sinkhorn近似的doubly-stochastic，换成了置换矩阵的精确凸组合（Birkhoff–von Neumann）。但SS order只由state维度 $m$ 决定，HC / mHC / mHC-lite都是 $m{=}4$，所以同一套推导给出的order都是 $m$。 $\mathbf A$ 上的约束（无约束 / 近似 / 精确doubly-stochastic）管的是稳定性，跟order无关。

<a id="attnres-proof"></a>
## AttnRes的SS order推导

**Full AttnRes是满阶的，也没有有限的SSS表示。** 通俗说就是"有了softmax、加了exponential，就再也压不成low-rank了"。背后的原因是：softmax kernel $\phi(\mathbf q,\mathbf k)=\exp(\mathbf q^{\!\top}\mathbf k)$ 没有有限维的feature map $\varphi$ 能让 $\exp(\mathbf q^{\!\top}\mathbf k)=\varphi(\mathbf q)^{\!\top}\varphi(\mathbf k)$。对照linear attention：它的score是先过一个**固定 $r$ 维的feature map $\varphi$**（ $\varphi$ 可以是非线性的）再做内积， $\mathbf M_{\text{lin}}=\varphi(\mathbf Q)\varphi(\mathbf K)^{\!\top}$ 必然 rank $\le r$、跟层数 $L$ 无关；而 softmax 的 $\exp(\mathbf Q\mathbf K^{\!\top})$ 不经过任何有限feature map，逐元素 $\exp$ 把低秩打散、直接满秩。用一个真实的linear-attention feature map（elu+1）验证—— $\varphi$ 是非线性变换，但 $\mathbf M_{\text{lin}}$ 的rank仍卡在 $r$（ $=4\ll L$），softmax 则满秩：

```python
import numpy as np
rng = np.random.default_rng(0)
L, r = 8, 4
Q = rng.standard_normal((L, r)); K = rng.standard_normal((L, r))
phi = lambda x: np.where(x > 0, x + 1.0, np.exp(x))     # elu+1 feature map (非线性), R^r -> R^r
M_lin  = phi(Q) @ phi(K).T                              # 经 r 维特征做内积 -> rank <= r
M_soft = np.exp(Q @ K.T)                                # softmax kernel (未归一化)
print(np.linalg.matrix_rank(M_lin,  tol=1e-9))    # 4 = r   (linear kernel, 远小于 L=8)
print(np.linalg.matrix_rank(M_soft, tol=1e-9))    # 8 = L   (softmax kernel, full)
```

于是AttnRes的off-diagonal子块都是满秩的，SS order取到因果 $L\times L$ 矩阵所能允许的最大值：

$$\text{order}=\max_{t}\ \mathrm{rank}\,\mathbf M[t{:},\,{:}t]=\max_t\ \min(t,\,L{-}t)=\big\lfloor L/2\big\rfloor.$$

论文Figure 8画出的真实depth attention map也印证了这个"稠密但有结构"的样子（对角局域性 + embedding列的attention sink）：

![Figure 8 from AttnRes paper: real depth-attention maps](fig/AttnRes_score.png)

**Block AttnRes是工程上的折中，order落在 $[N,N{+}S]$。** 它把每 $S$ 层压成一个block-summary（用普通residual求和），然后只对 $N=L/S$ 个summary做attention。要严格算rank不容易，但机理清楚、也能数值确认：

- 一个已经封口(sealed)的block里， $S$ 个来源共享同一个block级别的key，它们在 $\mathbf M$ 里对应的 $S$ 列相同——每个封口block只贡献 rank 1，至多 $N$ 条通道；
- 当前还没封口的block里，每层仍逐个来源地attend，最多再添 $S$ 条通道。

两者相加就得到 order $\in[N,N{+}S]$： $N{=}1$ 退回普通残差， $N{=}L$ 逼近Full AttnRes。数值上确认封口block确实rank 1，且上界 $N{+}S$ 在所有随机种子下都成立（实际order贴着 $N$，始终小于Full AttnRes的 $\lfloor L/2\rfloor$）：

```python
import numpy as np
rng = np.random.default_rng(0)
def rms(x): return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)
def M_block_attn(L, S):                          # N = L // S 个 block, 每个 block 大小 S
    B = L // S; d = 32
    kblk = rms(rng.standard_normal((B, d)))       # 每个 sealed block 一个共享 key
    ksrc = rms(rng.standard_normal((L, d)))       # 当前 block 内每个 source 各自 key
    w = rng.standard_normal((L, d)); M = np.zeros((L, L))
    for l in range(L):
        bl = l // S
        lg = [w[l] @ kblk[b] for b in range(bl)] + [w[l] @ ksrc[i] for i in range(bl*S, l+1)]
        a = np.exp(np.array(lg) - max(lg)); a /= a.sum()
        for b in range(bl): M[l, b*S:(b+1)*S] = a[b]     # sealed block: S 列相同 -> rank 1
        M[l, bl*S:l+1] = a[bl:]                          # 当前 block: 逐 source
    return M
order = lambda M: max(np.linalg.matrix_rank(M[t:, :t], tol=1e-9) for t in range(1, M.shape[0]))
assert np.linalg.matrix_rank(M_block_attn(24, 4)[8:, 0:4], tol=1e-9) == 1   # sealed block 确为 rank 1
for L, S in [(16, 4), (24, 4), (32, 4)]:
    N = L // S; ords = [order(M_block_attn(L, S)) for _ in range(30)]
    assert all(o <= N + S for o in ords)          # 上界 N+S 恒成立
    print(f"L={L} S={S} N={N}: order in {min(ords)}..{max(ords)} <= N+S={N+S}, band [{N},{N+S}]")
    # 输出: L=16 S=4 N=4: order in 4..4 <= 8, band [4,8]
    #       L=24 S=4 N=6: order in 6..6 <= 10, band [6,10]
    #       L=32 S=4 N=8: order in 7..7 <= 12, band [8,12]
```

对应"数值实验"章节图里的 subplot (c)： $L{=}24$ 下取 $N{=}4,6,8$，每条曲线随截断呈锯齿状（block内爬升、一封口就跌回），峰值落在阴影 $[N,N{+}S]$ 带里， $N$ 越大整条曲线越往上抬——始终卡在残差的 $1$ 和Full AttnRes的 $\lfloor L/2\rfloor$ 之间。所以Block AttnRes可以用参数 $N$ 调节表达能力： $N$ 调大，深层能挑着读取的浅层信息越多，代价是要跨截断搬运的summary也越多。

<a id="references"></a>
# 参考文献

- Attention Residuals — [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)（Table 5 & Fig. 9：把residual变种归到structured matrix）
- mHC: Manifold-Constrained Hyper-Connections — [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)；Hyper-Connections — [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)；mHC-lite — [arXiv:2601.05732](https://arxiv.org/abs/2601.05732)
- Dao & Gu, *Transformers are SSMs* (SSD / Mamba-2) — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)；[理论博客](https://goombalab.github.io/blog/2024/mamba2-part2-theory/)；[DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- Vandebril, Van Barel & Mastronardi, *Matrix Computations and Semiseparable Matrices* (Johns Hopkins UP, 2008)；Gantmacher & Krein, *Oscillation Matrices and Kernels* —— 半可分矩阵的经典起源与 $O(n)$ 快速算法
