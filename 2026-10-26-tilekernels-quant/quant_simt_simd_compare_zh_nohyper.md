# SIMT比SIMD更好编程？基于TileKernels/quant算子的深入对比分析

本文正文**纯手敲，无AI生成**。AI agent仅用于示例代码仓，可复现本文结果：
- CA model仿真单核算子VF，没有NPU卡的同学也可以跑，带Dockerfile一键复现： https://github.com/PTO-ISA/tilelang-puzzles-ascend/tree/simt_compare
- 完整多核流水+上板：https://github.com/PTO-ISA/TileKernels-PTO

参考自DeepSeek官方样例：
- https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md
- https://github.com/deepseek-ai/TileKernels/tree/v2.1.0-release/tile_kernels/quant

带超链接的GitHub版blog见 <sup>[1]</sup>

# 写作动机

多年来，笔者一直不能理解许多人把"NPU编程困难"归咎于"不支持CUDA/SIMT"。我琢磨着，CUDA经常要写warp shuffle/vote/sync/mma不也没有SIMT的便利性了？而CPU上过去十几年手搓AVX/Neon/SVE也没人抱怨难写？"SIMD NPU"比"SIMD CPU"确实难写的地方在software pipelining，所幸TileLang原生支持了昇腾<sup>[2]</sup>，算子哥只要用`T.Pipelined`自动排软件流水，不用痛苦地手排set_flag/wait_flag和cross-core flag这堆semaphore了。(感谢DeepSeek源神！)

但NPU算子依然看着比CUDA复杂。根据DS开源的TileKernels样例<sup>[3]</sup>，每一个`*_asc.py`算子比相应的`*_cuda.py`冗长**约两倍**。这里不是手搓software pipelining的锅了 (已经被`T.Pipelined`, `T.Persistent`, `T.Stage`等隐藏起来)，而是由于`with T.SimdVF():`内大量使用底层SIMD指令，例如`vld`, `vsts`, `vadds`, `vdup`等等 (见`tilelang.ascend.language.simd`<sup>[4]</sup>）。GPU代码则是`T.Parallel` scope内的标量粒度操作，看着更简单。本文放下SIMD vs SIMT之争的主观偏见，基于具体算法pattern，客观分析语法gap到底在哪。

<img src="fig/hybrid_simt_simd.png" alt="hybrid_simt_simd" width="50%" />

Ascend 950支持了SIMD与SIMT混合编程<sup>[5]</sup>，由上图所示。我们可以把同一个算子用SIMD和SIMT各写一遍，逐行对比。SIMT模式的编程体验和早期CUDA接近，只是性能差于SIMD（后文有实测；这里不像优化CUDA那样做好coalesce就能持平SIMD/warp，而是指令吞吐就不同），且指令种类有缺失 (e.g. FP4转换在SIMT模式没有，只在SIMD有)。在SIMD模式下，得进一步区分是"**SIMD本身**难写" (宽泛的结论) 还是"**ASC/CCE的SIMD抽象**太底层所以难写" (更谨慎的结论，说明SIMD概念模型还能抢救)

为了对tilelang社区做些贡献，PTO团队思考了两种方案：
- 一是维持在SIMD编程范畴，但是抽象掉ASC/CCE里难写的SIMD指令，让编程体验更接近SIMD CPU，也就是PTO Virtual micro Instruction (pto.vmi)<sup>[6]</sup>。使用tilelang社区PTO后端codegen_pto.cc<sup>[7]</sup>，用户可以设置 `target='pto'` 从tilelang codegen生成PTO-DSL (类似tilelang -> CuteDSL的lowering路径，和Cute共享CodeGenTileLangPY<sup>[8]</sup>基类)，也可以直接编写或修改底层的PTO-DSL<sup>[9]</sup>。
- 二是给用户营造更像SIMT的幻觉，允许`T.Parallel`这种标量+SIMT式的语法，把更多的困难留给编译器，见PR 3512<sup>[10]</sup> （嗅到了远古的"SIMT-to-SIMD transpilation"送命题的味道？所幸，不需要整个CUDA编译到NPU，只是Vector Function内部，相信我们优秀的编译器团队可以搞定）

以上设计是众多同事日夜奋战的结果 (自行看tilelang git committer，不抢credit)。个人一项贡献是在今年初 (顶着反对声音，咳咳。。) 搓了一版 PTO-DSL前端原型<sup>[11]</sup>，证明不依靠厚重的bisheng/clang前端也可以用Python->MLIR搓算子，且框架层可以*做薄、透明可魔改*。该DSL后续已重构优化并合入开源的PTO Assembler<sup>[12]</sup>与9/30后的官方CANN包。

# 大纲

- 为何选量化算法为切入点
- 数学基础复习
    - 量化算法变种
    - 极简torch代码示例
- 按运算pattern分类
    - SIMD已足够简单直观的case
    - SIMT和SIMD区别微小的case
    - SIMT有略微优势的case
    - SIMT有明显优势的case
    - 特殊情况
- 性能对比 (VF仿真)
- 遗留问题讨论
- 引用链接

<a id="why-quantize"></a>
# 为何选量化算法为切入点

原因一：国庆开源的TileKernels昇腾适配里，quant算子占很大比例<sup>[13]</sup>。这些都是真实场景有用的算子，有足够的复杂度，例如FP8 cast考虑了输出的内存排布和DeepGEMM对齐，无缝对接。性能也好，大shape性能可以打到950DT roofline的80%左右（见benchmark<sup>[14]</sup>），对950PR已经远远超标了（即HBM bound，不是vector bound）。在Semianalysis的性能实测里<sup>[15]</sup>，tilelang的quantize算子数倍快于Triton-ascend, dequant算子数倍快于CANN包内置算子。

![semi_950_kernel_quant](fig/semi_950_kernel_quant.png)

原因二：同类的quant算子，在ops-nn仓的AscendC的写法十分酸爽，语法有很大改良空间：
- MXFP量化，一个核心文件1600行，还没算上其他变种模板 https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis.h
- MXFP反量化，一个核心文件800行 https://gitcode.com/cann/ops-nn/blob/master/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h

这类算子有个有趣的特征：数学公式简单，比attention/MoE等融合算子简单得多（无非是 分组reduce_max，分组乘scale，dtype转换, 再pack一下scaling factor省内存；反量化就是顺序倒过来且不用reduce），写成pytorch不到10行，CUDA代码也不复杂 (唯一丑点的是如果手写 `__shfl_*_sync`等等，所幸tilelang把warp操作藏在`T.reduce_*` primitive里了)。写成NPU算子却过分冗长 -- 哪怕tilelang已经化简了很多复杂度，也要花100~200行捣腾SIMD寄存器；而AscendC还要手写流水和模板，自然就上千行了。

Ascend 950量化逃不开的运算是 `Reg::Interleave`<sup>[16]</sup> (`S.vintlv`) 和 `Reg::Deinterleave`<sup>[17]</sup> (`S.vdintlv`)，以及相关的`Reg::LoadDist::DIST_DINTLV_B32` (`S.vld2(..., dist="DINTLV_B32"`) 等等。这是由于一个SIMD寄存器长度永远是256 Byte，无论类型(FP32/FP16/BF16/FP8)，所以type cast前后会导致寄存器**数量**的变化，而不是**长度**的变化。这就是为什么`T.vcvt`参数<sup>[18]</sup>会有`PART_EVEN`/`PART_ODD`，甚至`PART_P0`/`PART_P1`/`PART_P2`/`PART_P3`。AscendC `Reg::Cast`文档<sup>[19]</sup>有示意图：

<img src="fig/cast_parts.png" alt="cast_parts" width="70%" />

为了处理这些奇偶交错的parts，需要时不时用`vintlv`/`vdintlv`转换成其他SIMD计算/store等指令更prefer的layout。如下图示意：

<img src="fig/asc_interleave.png" alt="asc_interleave" width="50%" />
<img src="fig/asc_deinterleave.png" alt="asc_deinterleave" width="50%" />

`pto.vmi`一个核心简化是把`vcvt`的行为变成reg长度的变化，(虚拟)寄存器数量依然是一个，使得程序员的mental model更像torch `.to(dtype)`或者numpy`.astype`。是否足够简单？用具体例子展开。

<a id="math-basics"></a>
# 数学基础复习

<a id="quant-variants"></a>
## 量化算法变种

考虑常用的groupwise量化算子 (排除mixed precision、预排序、旋转、vector quantization等花活儿)，还能调整的无非是：每个scaling factor管多少个数 (按行/按列/按行列，block多大)，scaling factor用什么精度格式 (E8M0, FP32, etc.)。 直观示意 `TileKernels/quant`里的`per_token_cast`, `per_channel_cast`, `per_block_cast`三种情况：

<img src="fig/quant_dim_compare_fig.png" alt="quant algo compare fig" width="50%" />

<img src="fig/quant_dim_compare_txt.png" alt="quant algo compare text" width="70%" />

（截图自论文 Pretraining large language models with MXFP4 on Native FP4 Hardware<sup>[20]</sup>）

TileKernels/quant还有更细节的变种：
- scaling factor (sf) 是按`[batch, group]`存，还是转置的`[group, batch]`存 (`use_tma_aligned_col_major_sf=True`)
- E8M0的sf是否要按每两个数压成int16类型 (`use_packed_ue8m0=True`)，否则用FP32(E8M23)或BF16(E8M7)浪费Mantissa bit
- 有些场景，单个算子会*融合量化和反量化*:
    - FP8 + sf反量化后，马上重新量化成FP4 + sf，或相反路径FP4->FP32->FP8 （当`per_token_cast`输入是`QuantTensor` tuple类型的时候）
    - per_token格式反量化后，重新量化成per_channel格式（当`per_channel_cast`输入是`QuantTensor` tuple的时候）
    - BF16/FP32量化成FP8/FP4以后，马上反量化回BF16/FP32，实现fake-quant的效果，用于quantization-aware training （`per_token_cast_and_cast_back`文件）

<a id="torch-examples"></a>
## 极简torch代码示例

`per_{token/channel/block}_cast`维度变种，在pytorch/数学视角是trivial的区别：

```python
def torch_per_token_cast(x: torch.Tensor, group_size: int = 32):
    m, k = x.shape
    # 沿着行，按group_size分组 ，输出`grouped` shape [m, k // G, G]
    grouped = x.float().view(m, k // group_size, group_size)  
    # 分组找absmax，输出`amax` shape [m, k // G]
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN) 
    sf = amax / E4M3_MAX  # 按 E4M3_MAX = 448.0 归一化; shape仍为 [m, k // G]
    q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)  # 分组按sf缩放然后转低精度
        ).view(m, k).to(torch.float8_e4m3fn)  # 输出shape 从 [m, k // G, G] 回归为 [m, k]
    return q, sf

def torch_per_channel_cast(x: torch.Tensor, group_tokens: int = 32):
    m, k = x.shape
    # 沿着列按group_tokens分组 ，输出`grouped` shape为 [m // G, G, k]
    grouped = x.float().view(m // group_tokens, group_tokens, k)  
     # 分组找absmax，输出`amax` shape [m // G, k]
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX  # 按 E4M3_MAX = 448.0 归一化; shape仍为 [m // G,  k]
    q = (grouped * (E4M3_MAX / amax).unsqueeze(1)  # 分组按sf缩放然后转低精度
        ).view(m, k).to(torch.float8_e4m3fn)  # 输出shape 从 [m // G, G, k] 回归为 [m, k]
    return q, sf

def torch_per_block_cast(x: torch.Tensor, block: tuple = (32, 32)):
    m, k = x.shape
    bm, bk = block
    # 沿着行和列，各自按group_size分组 ，输出`tiles` shape为[M // bm, K // bk, bm, bk]
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    # 被行和列分割成的2D groups里，每个组找absmax，输出`amax` shape 为[M // bm, K // bk]
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
     # 分组按sf缩放然后转低精度，这里用unsqueeze帮助broadcast：[M // bm, K // bk, 1, 1] 和 [M // bm, K // bk, bm, bk] 相乘
    quant = tiles * (E4M3_MAX / amax).unsqueeze(-1).unsqueeze(-1)
     # 输出shape 从  [M // bm, K // bk, bm, bk] 回归为 [m, k]
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    return q, sf
```

(完整脚本见 puzzles/torch/quant<sup>[21]</sup>)

上面省略了一些细节，这里补充:

1. 为什么要`clamp` (`E4M3_CLAMP_MIN = 1e-4`)？考虑某一组x都为0的极端情况，如果不clamp，会导致`sf = amax = 0`, `q = 0/0 = nan`。

2. 真实业务算子还要考虑corner case：最后一个group不能被group_size整除的情况，要填充(padding)到group_size的倍数。

3. 上面代码示例输出的scaling factor `sf`是高精度FP32 (E8M23)。按照MXFP8的定义，sf应该只取E8的部分，并且pack进int16或int32数组省内存(因为没有原生的"E8M0"类型数组)。对应TileKernels/quant里的`round_sf=True`和`use_packed_ue8m0=True`同时成立。

`round`这一步依赖bit操作(左移右移, and/or)。真实tilelang kernel里也有**同样的位运算**，例如CUDA后端的 `get_sf_and_inv`<sup>[22]</sup>，或者Ascend后端的`compute_scale`<sup>[23]</sup>.

*此处把数学部分搞清楚，后面理解算子就容易了，无非是把torch API换成等价的NPU指令。*

**3.1. 向上舍入(FP32 -> E8M0)**

```python
# 假设已按前面示例算出了分组的最大值`amax`，FP32格式

# float类型不允许bit操作 (几乎所有编程语言的通用规范)，所以归一化后需要re-interpret成int类型才能操作
# 这里只是重新解读同一块bits，不是真的cast成int32
# (注意区别torch的`.view` vs `.to`，或C++的 reinterpret_cast vs static_cast)
bits = (amax / E4M3_MAX).view(torch.int32)

# 这里是所谓的 "ceil-log2 bit trick"，把FP32 sf向上舍入到最邻近的E8M0表示
exp_sf = ((bits - 1) >> 23) + 1 - 127
```

只要理解上面的`exp_sf`公式，就理解了quant算子的主要数学trick。这里是为了把高精度的FP32 scaling factor*向上舍入*到E8M0格式。

假设 $s$ 代表代码里`amax / E4M3_MAX`的FP32计算值， $s_f$ 代表舍入后的 E8M0数值. $s_f$ 按照浮点数定义，必须是base-2的指数表达：

$$
\mathrm{s_f} = 2^{\left\lceil \log_2 s \right\rceil}
$$

(这里公式都是单个标量的数学运算，带入torch都是elementwise操作，无需考虑tensor shape)

这里有 $s_f \ge s$，因为 $s_f$ 本身满足下面恒等式，不需要ceil取整:

$$
\mathrm{s_f} = 2^{\log_2 s_f} \ge 2^{\log_2 s}
$$

$s$已知，怎么算 ${\left\lceil \log_2 s \right\rceil}$ 呢？先考虑更简单的`floor-log2`: ${\left\lfloor \log_2 s \right\rfloor}$ ，用 `bits >> 23 - 127` 就能算出来。FP32的bit是`S1E8M23`, `bits >> 23`把8位指数挪到了最右边。因为是absmax, sign bit一定是0，就只剩`E8`的部分。

<img src="fig/fp32_format.png" alt="FP32 format" width="50%" />

注意float32的`E8`存的是*biased exponent*，代表的实数是 $s = 1.M \times 2^{E - 127}$，所以`log2(s) = (E − 127) + log2(1.m)`，要把`E8`减掉`127`才是${\left\lfloor \log_2 s \right\rfloor}$。

回忆经典的C++ ceil-division `(x - 1) / y + 1` (正数情况，对比floor-division `x / y`)。这里的ceil-log2 vs floor-log2是类似的trick，把`bits >> 23`改成`(bits - 1) >> 23) + 1`，就相当于ceil-log2了。先`-1`把“刚好不被向下舍入的数”(除法情况就是x为y的整数倍，log2情况就是2^n整次幂)变成“会被向下舍入的数”，然后把向下舍入的结果+1修回来 -- 使得“刚好整除的数/刚好是2^n的数”计算结果恰好不变，其他数值相当于被向上舍入。biased exponent`-127`的部分依旧，完整公式就是： ${\left\lceil \log_2 s \right\rceil}$ 等于 `((bits - 1) >> 23) + 1 - 127`

为什么不直接用floor-log2把 $s$ 向下舍入呢？这样scale factor的值域太小，`q` 会超过FP8的表达上限 `E4M3_MAX = 448.0`。

**3.2 如果只round不pack**

前面算出的`exp_sf = ((bits - 1) >> 23) + 1 - 127`是位移操作得到的“E8M0的指数bits”在int32类型下的view。别忘了我们要把原始输入x分组除以scaling factor，才得到FP8 `q`的值。`int32`格式的`exp_sf`和`float32`的`x`不能直接乘（直接乘的自动type promotion就是real cast不是bit view了），而得左移 `<< 23`挪回FP32的指数位上，再重新view (re-interpret cast) 回`float32`，才能正确地和`x`相乘。不仅torch得这么写，真实GPU/NPU算子同理。

```python
# 左移23位之前，得把减掉的前面127再加回去，满足FP32 encoding定义
sf = ((127 + exp_sf) << 23).view(torch.float32)
# 1/sf直接用2^{-E}表达，bit操作放到正确的指数位，无需浮点除法
sf_inv = ((127 - exp_sf) << 23).view(torch.float32)

# 记得，x量化成q之前要用”round后的sf”去分组除，不是原始的高精度sf，这样(q, sf)这对tuple才能对得上原始x
# `grouped`变量是前面reshape过的`x`
q = (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
```

小quiz：为什么3.1节做了右移23位，3.2节又左移23位，不就抵消了等于啥也没干？不如直接把mantissa bits mask掉，指数bit不动，也能提取8位exponent？这么做的问题是没法正确表达“log2向上取整”的算法，还得根据M是否为0走不同分支。

**3.3. 如果pack（使E8M0只占8 bit），再多算一步**

不存在原生的`torch.e8m0`数组类型，TileKernels quant算子用`torch.int16`数组存储E8M0的bit，每一个int16标量代表两个scaling factor:

```python
# 注意这里是.to, 不是.view，是确实把int32 cast成了只有8位的uint8
e8m0 = (exp_sf + 127).to(torch.uint8) 
lo = e8m0[..., 0::2].to(torch.int16)  # 16bit的右半边8位
hi = e8m0[..., 1::2].to(torch.int16)  # 16bit的左半边8位
packed = lo | (hi << 8)  # 左右取or，拼起来
# `packed`变量被写回global memory
```

<a id="patterns"></a>
# 按运算pattern分类

DS开源的TileKernels把上面提到的很多变种放进了一个kernel文件里，通过许多`config` fields在kernel内做compile-time branching。笔者从中拆出了独立的小pattern tilelang-puzzles-ascend<sup>[24]</sup>，仿照tilelang-puzzles<sup>[25]</sup>的从易到难、循序渐进风格，便于学习理解。quant类算子按复杂度和算法pattern，分成了几类：

| | cast_back | per_token | per_block | per_channel |
|---|---|---|---|---|
| 01 | FP8 + FP32 scale, either out dtype | raw FP32 scale | raw, 32×32 tile | raw, reduce along M |
| 02 | packed UE8M0 | `round_sf` (pow2) | `round_sf` + packed | packed along **M** |
| 03 | `sf_block=(32,32)` | packed UE8M0 | FP4 output | requant + bf16 |
| 04 | `sf_block=(32,1)` | fp32 in / FP4 out | column-major scales | composed |
| 05 | FP4 input | column-major scales | split + composed | |
| 06 | col-major, composed | `sf_only`/`cast_only`, requant | | |
| 07 | | bf16 compute, composed | | |

每个变种包含了4种实现：
- `ASC`提取自DS官方TileKernels SIMD实现，但截取单个分支变种；
- `PTO`来自我司PTO团队移植的简化SIMD写法; 
- `SIMT`大多提取自DS官方CUDA实现，个别做了微小改动，可运行在Ascend SIMT模式；
- `torch`是纯数学参照，提取自unit test reference。

本文所有样例不考虑VF之外的GM<->UB DMA (`T.copy`)、多核scheduling部分 (`T.Persistent`)等等 -- 这些部分兼容了成熟的tilelang GPU语法，不需要NPU特化的API。

先high-level地统计下 `SimdVF`和`SimtVF`区域的代码行数，粗略近似编程复杂度：

<img src="fig/vf_lines.png" alt="VF lines" width="100%" />

对于“SIMD友好类算子”，比如`per_token_cast`类多是“内存连续的32个数做reduce"，PTO SIMD代码几乎和SIMT一样短，已经足够简单；ASC的冗长是因为物理指令太底层，而非"广义SIMD概念"的锅。而“访存跳跃类”算子，例如“transpose to column major”，SIMT的行数确实更短。总的来说，涉及到transpose、gather、2D/3D broadcast and reduce、irregular strides、算法侧interleave等等灵活访存的pattern，SIMT代码确实更简短、易理解。原则上，SIMD抽象可以跳出1D局限，可以进一步增加high-dimensional reduce/broadcast/gather/scatter等 torch/numpy-like高阶API，但毕竟扩大了API surface，多出一堆函数让用户去记/让agent去查。（API也不是越多越好。。我相信算法哥第一次看到`torch.gather`/`torch.scatter`的高维语义都得楞个几秒钟，如果搞出更复杂的API也不好卖。。）

<a id="simd-simple"></a>
## SIMD已足够简单直观的case

### `per_token_cast/01` -- 内存连续的分组reduce

SIMT写法是：

```python
# 注：`x_ub = T.alloc_shared((hidden,), T.bfloat16)`在VF外分配，值从 GM T.copy 而来

# 下面代码为 `with T.SimtVF(threads=THREADS):` scope内部的片段
x_l = T.alloc_fragment((num_groups, group_size), T.float32)
amax_l = T.alloc_fragment((num_groups,), T.float32)

# 把1D的 `x_ub`从UB读到reg，reshape成 2D的 `x_l`，并从bfloat16转成float32
# 类比torch的: x_l = x_ub.reshape(num_groups, group_size).to(torch.float32)
for g, j in T.Parallel(num_groups, group_size):
    x_l[g, j] = T.Cast("float32", x_ub[g * group_size + j])

# 沿着`group_size`轴reduce， 类比torch的: `amax_l = x_l.amax(dim=1)`
T.reduce_absmax(x_l, amax_l, dim=1)
```

PTO SIMD写法是：

```python
# 同上，`x_ub = T.alloc_shared((hidden,), T.bfloat16)` 在VF外分配
# 下面代码为 `with T.SimdVF():` scope内部的片段

# 同上，Load UB -> Reg, cast bfloat16 to float32
x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")

# 分组reduce-max (vcmax带group参数)，结果写回UB
V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
         amax_ub[group])
```

vcmax指令<sup>[26]</sup> + `vabs` 等价于SIMT代码里reshape + `T.reduce_absmax`的效果。CA模型显示，运行这段SIMD代码比SIMT模式*少花近一半的cycle*，见后文的性能总结。

ASC SIMD写法，则难读得多。这是由于前文解释的“一个SIMD物理寄存器无论type永远是256 Byte长，所以type cast会导致寄存器**数量**的变化”。由于这个限制，在AscendC代码随处可见“用2~4个寄存器表示一组数学含义相同的变量，调用2～4次重复的API”。

```python
a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                    T.float32))
# Each register holds two groups, so reduce half at a time.
S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high),
        dist="ONEPT_B32")
S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),
        dist="ONEPT_B32")
S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high),
        dist="ONEPT_B32")
```

这个低层次的代码是PTO希望抽象掉的，就不展开解释了。有兴趣可参见更底层的Micro Instruction Spec<sup>[27]</sup>.

<a id="simt-simd-similar"></a>
## SIMT和SIMD区别微小的case

### `per_token/02`, `cast_back/02` -- 倒腾bit运算

如果代码行数消耗在复杂数学公式，大部分是elementwise运算，SIMD或SIMT写出来都一样长。前文用torch讲解的“FP32 -> E8M0向上舍入”就是典型。

上文torch写过的逻辑都在tilelang代码原封不动地出现了：`CLAMP_MIN` 防止“0除以0”；用torch `.view(int32)`也即tilelang `T.reinterpret("int32")`允许浮点数的bit运算；用"ceil-log2 bit trick" 也即 `((bits - 1) >> 23) + 1`算向上舍入的log2; FP32 biased exponent格式要减去 `127`（下面的`254`是`127 - exp_sf = 127 - (biased - 127)`合并出来的)：

```python
# `with T.SimtVF(threads=THREADS):` scope内部的片段
for g in T.Parallel(num_groups):
    amax = T.max(amax_l[g], E4M3_CLAMP_MIN)
    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    sf_l[g] = T.reinterpret("float32", biased << 23)
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
```

当前PTO代码长这样：

```python
# `with T.SimdVF():` scope内部的片段

# 注：`V.vbrc(const)`把标量broadcast成SIMD寄存器，相当于ASC样例的`S.vdup`
clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes))
one = V.vbrc(T.uint32(1), size=lanes)
shift = V.vbrc(T.uint32(23), size=lanes)
bits = V.vinterpret_cast(
    V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
inv = V.vinterpret_cast(
    V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
```

看着比SIMT啰嗦，是因为缺少符号重载的语法糖罢了. 假设我们重载了 `__add__`, `__mul__`, `__sub__`, `__rsub__`, `__rshift__`, `__lshift__`, 并且把参与运算的python标量`const`自动promote成 `V.vbrc(T.uint32(const), size=lanes)`，SIMD代码看起来同样简洁：

```python
# `with T.SimdVF():` scope内部的片段
clamped = V.vmax(amax, E4M3_CLAMP_MIN)
bits = V.vinterpret_cast(clamped * inv_qmax, "uint32")
biased = ((bits - 1) >> 23) + 1
sf = V.vinterpret_cast(biased << 23, "float32")
inv = V.vinterpret_cast((254 - biased) << 23, "float32")
```

还省了一行`T.Parallel`本身。

### `per_channel/01` serial reduction 

对严格串行的运算，用SIMT和SIMD并无区别。典型例子是per_channel量化，需要沿着列(内存非连续)顺序地找max。

CUDA-like SIMT:

```python
# `T.SimtVF(threads=THREADS)` scope内部的片段

# 沿着 x_ub[row, c] 的`row`维度串行地对比，记录最大的absmax值
for c in T.Parallel(hidden):
    amax = T.alloc_var(T.float32)
    # 初始值为E4M3_CLAMP_MIN，结果不会小于它，相当于torch.clamp的效果
    amax = E4M3_CLAMP_MIN
    for row in T.serial(group_tokens):
        amax = T.max(amax,
                     T.abs(T.Cast("float32", x_ub[row, c])))
    sf_ub[c] = amax / E4M3_MAX
    inv_ub[c] = E4M3_MAX / amax
```

PTO SIMD:

```python
# `alloc_local`分配长度为1的局部向量寄存器数组；`acc[0]`可被反复赋值，用作循环累加器
acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
# 同上，记录初始值为E4M3_CLAMP_MIN
acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
for row in T.serial(group_tokens):
    v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                      "float32"), mask)
    # lane c holds the max over tokens seen so far
    acc[0] = V.vmax(acc[0], v, mask)
V.vstore(V.vdiv(acc[0], qmax, mask), sf_ub[col])
V.vstore(V.vdiv(qmax, acc[0], mask), inv_ub[col])
# 注：`col`为外层循环的index，按SIMD长度循环处理整段向量
# `qmax`为scope外面创建的、填充了`E4M3_MAX=448.0`值的SIMD寄存器 
# 即 `qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)`
```

PTO语法略有冗余的地方，只需加点语法糖：把`V.vdiv`重载成 `/`；把创建标量`acc[0]`的API简化；把默认`mask`省略（full lane, 不mask）；再沿用上一小节`per_token_cast`建议的自动promote `python常量 -> V.vbrc` -- 看着就和SIMT示例同样简短，唯一区别是把`for c in T.Parallel(hidden)`吸收进向量操作API：

```python
# 加了语法糖的SIMD mock style
amax = V.alloc_var(T.float32)  # represents scalar, not vector reg
amax = E4M3_CLAMP_MIN
for row in T.serial(group_tokens):
    v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES), "float32"))
    amax = V.vmax(amax, v)
V.vstore(amax / qmax, sf_ub[col])
V.vstore(qmax / amax, inv_ub[col])
```

<a id="simt-slight-advantage"></a>
## SIMT有略微优势的case

涉及到strided memory access，但是stride比较regular的情况 -- SIMD代码并不复杂，只是需要知道专门的API名字；SIMT稍微直观一些。

### `cast_back/01` group broadcast

考虑1D反量化，把长度为`hidden/group_size`的scaling factor分组乘到长度为`hidden`的`q`，每`group_size`(通常是32)个q共享一个sf。SIMT语法：

```python
for i in T.Parallel(hidden):
    o_l[i] = q_l[i] * sf_ub[i // group_size]
```

Group-Query Attention也有类似算法pattern，多个query共享一个kv，相当于要把kv broadcast才能和q乘。

cast_back_cuda.py<sup>[28]</sup>泛化到二维broadcast，简单两行代码支持{1D/2D, 按行/按列}的反量化：

```python
for i, j in T.Parallel(TILE_M, TILE_K):
    out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens, j // num_per_channels]
```

`i // group_size`实现了对`sf`的分组broadcast. 在PTO SIMD模式，load-broadcast是专门的指令：

```python
scale = V.vload(sf_ub[group], size=LANES, stride=1,
                dist_mode="brc", group=groups_per_strip)
scaled = V.vmul(values, scale, mask)
```

只要用户或agent知道这个API存在，代码也是两行。

cast_back_asc.py<sup>[29]</sup>的SIMD指令冗长2倍。用户/agent需要知道`vld`有个`BRC_B32` mode，并且手动合并两个物理寄存器。

```python
lo = S.vld(sf_ub[group], dist="BRC_B32")
hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
scale = S.vsel(lo, hi, mask_low)
scaled = S.vmul(values, scale)
```

### `per_channel/02` algorithm-level interleave

考虑"E8M0格式pack到INT16数组"这步。如果沿着行pack是连续读写；如果沿着列pack，就得把两个不相邻的数挪到相邻位置，涉及非连续的write。SIMT代码示意：
```python
for c in T.Parallel(hidden):
    packed_ub[PACK_FACTOR * c] = sf_rows_ub[0, c]
    packed_ub[PACK_FACTOR * c + 1] = sf_rows_ub[1, c]
```

相当于transpose，把"跨行的非连续sf"转成连续packing，化简自更泛化的tilelang SIMT代码 `tile_kernels/quant/common.py`的`get_sf_index`<sup>[30]</sup>：

```python
    if config.sf_col_pack:
        return m_idx // pack_factor, k_idx * pack_factor + m_idx % pack_factor
    if config.use_packed_ue8m0 and config.use_tma_aligned_col_major_sf:
        return k_idx // pack_factor, m_idx * pack_factor + k_idx % pack_factor
    if config.use_tma_aligned_col_major_sf:
        return k_idx, m_idx
    return m_idx, k_idx
```

这是一个非常regular的`[2, N]`转置，对应SIMD指令其实是`interleave`:

```python
a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
lo, _ = V.vintlv(a, b, byte_mask)
V.vstore(lo, packed_ub[col * PACK_FACTOR])
```

参见最开头讲解`AscendC:Interleave`的示意图。**这里和type cast无关，而是算法要求的内存layout变换。** PTO可以省略掉AscendC里和type cast有关的interleave，*但是省不了此处的*。ASC写起来也一样：

```python
a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
lo, _ = S.vintlv(a, b)
S.vsts(packed_ub[col * PACK_FACTOR], lo, dist="NORM_B8")
```

参考自 `per_channel_cast_asc.py`的`pack_sf_rows`<sup>[31]</sup>

这个少有的"PTO SIMD并未简化ASC"的情形。SIMT确实好理解些，无需用户了解`vintlv`指令及其行为。

<a id="simt-clear-advantage"></a>
## SIMT有明显优势的case

### `per_token/05` transpose

上个case是特殊的`[2, N] -> [N, 2]` transpose，如果考虑更泛化的transpose，那么SIMD模式就不能用专门的`vintlv`了，而得用更泛化的`vgather`。在SIMT模式依然是简单的标量下标`A[j, i] = ...`

当`out_config.use_tma_aligned_col_major_sf = True`，要把scaling factor transpose成 `[batch, group] -> [group, batch]`再store出去。`per_token_cast_asc.py`在`transpose_vectors()`里调用`S.vgather2`实现transpose<sup>[32]</sup>.

CUDA/SIMT写法十分简单，只要改一行。不需要知道vgather指令行为，不需要构造gather index。

```python
# 和前文展示过的 `per_token/02` bit运算trick一模一样，只改了最后一行
for g in T.Parallel(num_groups):
    amax = T.max(amax_l[g], E4M3_CLAMP_MIN)
    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
    # 不transpose就是store回`[sub, g]`下标，transpose就是下标互换一下
    sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)
```

对比PTO和ASC的SIMD gather写法：

```python
# PTO
V.vstore(V.vgather(sf_dense_ub[0, base], idx, gmask),
         sf_out_ub[base, 0])
# ASC
S.vsts(sf_out_ub[base, 0],
       S.vgather2(sf_dense_ub[0, base], idx), dist="NORM_B32")

```

`gather`只是一行，麻烦的是`idx`的构造。per_token_cast_asc.py内<sup>[33]</sup>根据是否packing (占FP32还是只占8位E8M0)，根据`pack_factor` (这里等于2)，构造transpose用的下标：

```python
    def init_transpose_indices(indices):
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                lane = S.vci(0, T.int16)
                output_row = S.vand(T.reinterpret(lane, 'uint16x128'), S.vdup(block_m - 1, T.uint16))
                group_in_vector = T.reinterpret(S.vshrs(lane, log2_block_m), 'uint16x128')
                source_indices = S.vadd(S.vmuls(output_row, sf_stride // pack_factor), group_in_vector)
                S.vsts(indices[0], source_indices, dist='NORM_B16')
            else:
                ...
```

由于这里不涉及 `vcvt` 引起的寄存器数量改变，PTO代码和这段ASC几乎一样，无法进一步简化。用语法糖重载`vand`, `vdup`, `vadd`, `vmuls`, `vshrs` 可以缩短代码，但远不如SIMT“把`x[i, j]`改成`x[j, i]`”那么简单：

```python
# init_transpose_indices的语法糖简化
lane = S.vci(0, T.int16)
output_row = T.reinterpret(lane, 'uint16x128') & (block_m - 1)
group_in_vector = T.reinterpret((lane >> log2_block_m), 'uint16x128')
source_indices = output_row * (sf_stride // pack_factor) + group_in_vector
S.vsts(indices[0], source_indices, dist='NORM_B16')
```

这是由于SIMD寄存器抽象只有1D。在Reg层面，PTO只抽象了长度，而不是维度。如果在SIMD Register File这层memory hierarchy也定义"2D"甚至"高维tile"，支持`tile.transpose()`和`tile.max(dim=1)`等高维语义，那么这里的SIMD代码会变成一行 `vstore(sf_out_ub, sf_reg.transpose())`。当然，编译器需要把`transpose` API映射到`vgather`指令（包括gather下标构造）

接下来的例子也是2D tile操作↓

### `per_block/01` 2D reduction

`per_block_cast`做跨行和列的reduce，SIMT写法为"先reshape/transpose, 然后按inner轴reduce"：

```python
for i, j in T.Parallel(bm, bk):
    flat_l[0, i * bk + j] = T.Cast("float32", x_ub[i, j])
T.reduce_absmax(flat_l, tile_l, dim=1)
```

完整版见per_block_cast_cuda.py<sup>[34]</sup>

```python
for i, j in T.Parallel(block_m, block_k):
    block_offset = (i % (num_threads * num_vectorize // block_k) * num_per_channels + j % num_per_channels) // num_vectorize
    amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset] = T.max(
        amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset],
        T.abs(x_fragment[i, j]),
    )
T.reduce_max(amax_reducer, amax_fragment, dim=2)
```

`T.Parallel`里的标量操作相当于`reshape + permute`，可回顾本文最开头的torch示例 `.view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)`。`T.reduce_max`这行是tilelang primitive, 不是SIMT语法，带`dim`参数实现分组reduce。

PTO SIMD的写法是 行内并行reduce(`V.vcmax`)，列间串行reduce (`T.serial`)

```python
# 沿着行，分组1D reduce，中间结果写到run_ub
for chunk in T.serial(num_chunks):
    v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                              size=LANES), "float32"), mask)
    V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])

# 最后把分组结果再做一次reduce，得到32x32 tile的absmax
partials = V.vload(run_ub[0], size=LANES)
tile_amax = V.vcmax(partials, mask_chunks, group=1)
```

如同前一个case的transpose样例，如果PTO支持2D tile的reshape/permute以及`max(dim=1)`，代码会像torch一样简短。但如果不断增加所谓的“高阶SIMD API”，和CUDA前端就越来越不一样了，对社区而言是增益还是负担？

<a id="special-cases"></a>
## 特殊情况

### `per_token/04` FP4 not supported in SIMT

最后是一个Ascend SIMT模式写不了的特殊情况：cast to FP4 和 cast back from FP4。GPU上有`__nv_fp4_e2m1`系列的primitive<sup>[35]</sup>, tilelang `per_token_cast_cuda.py`也支持`T.cast`到`T.float4_e2m1fn`类型；只是Ascend950 SIMT ISA里没有FP4转换指令，只能在SIMD模式下转。

SimtVF里对int类型进行bit操作，可以software-emulate出E2M1格式
```python
bits = T.reinterpret("int32", v)
# bit 3 of (bits >> 28) is bit 31 of bits: the sign nibble, without
# depending on whether >> sign-extends.
sign_nib = (bits >> 28) & 0x8
exps = (bits >> 23) & 0xFF
mant = bits & 0x7FFFFF
```

完整可运行代码见 simt/per_token/04_fp32_in_fp4_out.py<sup>[36]</sup>

但性能非常慢，比原生SIMD慢90倍cycle，见下一节性能测试。只要把cast这一步写成SIMD，其他都保留SIMT，就能从慢90倍变成只慢1.26倍。

```python
# 大部分代码是SIMT
with T.SimtVF(threads=THREADS):
    ...
    # 用`T.Parallel` 和 `T.reduce_absmax`做reduce和scaling，前面都讲解过

# 最后一步cast to FP4留给SIMD做
with T.SimdVF():
    for pair in T.serial(hidden // PAIR):
        S.vsts(q_ub[pair * PAIR],
                S.vcvt(S.vld(bf_ub[pair * PAIR]),
                        T.float4_e2m1fn),
                dist="PK4_B32")
```

<a id="perf-vf"></a>
# 性能对比 (VF仿真)

由于最近连不上机器了，先用CA模型跑一下Vector Function的cycle数，硬件实测下次再补。用`T.Kernel(1)` launch单核算子，不统计`T.copy` DMA的开销，不考虑GM<->UB流水、多核schedule的优化水平，纯粹看SIMD vs SIMT有区别的局部代码。使用 `msopprof simulator`，见 msOpProf Simulator Mode User Guide<sup>[37]</sup>。

本节原始数据在quant_vf_cycles.json<sup>[38]</sup>，结果可在CPU + Docker环境复现，完全由TileLang DSL + msprof内置CA模型测出；无需手写AscendC，无需真实NPU。

相同算法，相同shape，SIMT的cycle数多1.2 ~ 10倍不等。注意这只是VF仿真的结果，不代表真实硬件上的性能差距。如果VF耗时很短，算子是memory bound，那么cycle数多点少点都不会影响实测耗时。

<img src="fig/simt_cycles.png" alt="FP32 format" width="60%" />

- 第一行90倍是由于SIMT bit操作硬算FP4，属于特殊outlier。
- SIMT相对SIMD慢3倍以上的case，例如`per_channel` & `per_block` ，通常也是SIMT语法明显比SIMD简单的case。SIMD要挑选合适的 broadcast, interleave, gather指令，而SIMT捣鼓两行下标就完事。
    - 如果编译器能自动把这些case从SIMT DSL前端映射到SIMD硬件后端 (类似于把GPU的coalescing做到软件里；或者能处理高维复杂pattern的auto-vectorization compiler)，就能既兼容GPU CUDA-like代码又发挥Ascend 950的峰值性能

同样用SIMD，对比PTO vs ASC。这里基于`vmi-v0.1.9` PTOAS版本<sup>[39]</sup>，后续版本可能有更多优化。

<img src="fig/pto_cycles.png" alt="FP32 format" width="60%" />

Again，VF快30%不代表上板快30%。很多情况ASC基线已经优化到memory-bound了，再省VF时间也看不出效果。

根据之前做的profiling，PTO更快的case是因为lowering pass找到了更优的指令顺序和种类。虽然计算指令(vadd/vmul)省不了，但是捣腾layout的指令例如interleave/pack经常有多重等价写法。省掉几条interleave指令就化简了指令依赖关系，能使得OoO pipeline的bubble更少。

MXFP8->BF16反量化为例，profiling图：

手写ASC/CCE, VF耗时 144 us

<img src="fig/asc_vf_prof.png" alt="asc_vf_prof" width="100%" />

通过PTO.VMI lower, VF耗时 113 us

<img src="fig/pto_vf_prof.png" alt="pto_vf_prof" width="100%" />

统计发现`VADD`都是16个，`VMUL`都是32个，而`VINTLV`在ASC用了20个，在PTO只有8个，Pipeline bubble更少

理论上，根据PTO的lowering结果，再用ASC手写一版一样的指令顺序，也能做到相同性能。PTO编译器把常用的变换沉淀到pass里，不用反复浪费token或人力去寻优了。

<a id="open-questions"></a>
# 遗留问题讨论

在Agent编程的年代，本文花大篇幅讨论"降低编程复杂度"真的重要吗？目前阶段，"代码易于人类理解 & review & 维护"依然是重要加分项，至少看不懂的代码我还不敢merge。。。虽然写算子这行当几乎被agent取代完了，但agent写完你总得读懂(~~以及背锅~~)不是？

看到一篇slime v0.3.0: 面向 Agent 时代<sup>[40]</sup>讨论了"Agent 辅助的项目维护"，深以为然：

    在我看来，使用 coding agent 编程主要有 2 个主要的风险，

    一个是代码的巨幅膨胀形成的个人注意力 DDoS：代码产量上去了，但维护者 review、理解和消化这些改动的能力没有同步提升，最终只能心有余而力不足。

    另一个是人的参与感降低导致的 ownership 缺失：当开发者不再真正理解代码为什么这样设计，也不再把代码看作自己的作品，项目的架构质量就会在一次次“看起来能跑”的改动中慢慢劣化。

还有另一个常见疑问是，`T.Parallel` -> PTO SIMD的自动转换，如果用户写出*非常访存不友好*的奇葩pattern (`x[i, j]`这种灵活标量语法可以写得放飞自我)，会不会有编译器分析搞不定的case，或者转换结果不是最优？肯定有，我可以造出arbitrarily bad meaningless test cases。。但lowering pass可以做成best-effort：如果转成功了就是免费午餐提升性能，如果无法分析可以fallback到SIMT模式运行、或报错&回到用户手写SIMD的开发方式，对用户也没有负面影响。类似`torch.compile`的设计逻辑，能处理就是赚，不能处理至少可以fallback to eager。

相比于 `T.Parallel` + 标量下标的SIMT式语法，对NPU编译器更轻松的肯定是*扩展high-level SIMD API surface*：添加一堆类似torch/numpy的高维reshape/transpose/reduce/broadcast/gather/scatter等等，不需要lifting只需要lowering。借用Paul Kelly老爷子的一张图<sup>[41]</sup>，下山可比上山容易多了：

<img src="fig/Helicopter.png" alt="Helicopter" width="40%" />

其实Triton的`tl.max(axis=...)`, `tl.reshape`，`tl.broadcast`已满足"高维tile API"的要求。可惜原生Triton的`tl.load(pointers)`过于不亲和NPU的memory hierarchy和DMA参数。Tilelang的`T.copy`和多层次的`T.alloc_xxx`更贴合NPU；唯独`T.Parallel`语法，在GPU上是个很好的设计 (能利用hardware-based coalescing不用白不用啊)，到SIMD NPU则成了跨硬件兼容的挑战。。

<a id="reference-links"></a>
# 引用链接

```
[1] https://github.com/learning-chip/ai-system-research-blogs/blob/main/2026-10-26-tilekernels-quant/quant_simt_simd_compare_zh.md
[2] https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md
[3] https://github.com/deepseek-ai/TileKernels/tree/v2.1.0-release/tile_kernels/
[4] https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/language/simd.py
[5] https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_10052.html
[6] https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-vmi-Instruction-SPEC.md
[7] https://github.com/tile-ai/tilelang/blob/main/src/ascend/codegen/codegen_pto.cc
[8] https://github.com/tile-ai/tilelang/blob/424792eba01d887805fda8f71c847aa7749e6643/src/ascend/codegen/codegen_pto.h#L22
[9] https://github.com/hw-native-sys/PTOAS/blob/v0.68/ptodsl/docs/user_guide/14-vmi-virtual-instruction-set.md
[10] https://github.com/tile-ai/tilelang/pull/3512
[11] https://github.com/huawei-csl/pto-dsl
[12] https://github.com/hw-native-sys/PTOAS
[13] https://github.com/deepseek-ai/TileKernels/tree/v2.1.0-release/tile_kernels/quant
[14] https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md#benchmark
[15] https://newsletter.semianalysis.com/i/218361806/from-fast-kernels-to-reusable-kernel-engineering
[16] https://gitcode.com/cann/asc-devkit/blob/v9.1.1/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/MaskReg%E8%AE%A1%E7%AE%97/Interleave-11.md
[17] https://gitcode.com/cann/asc-devkit/blob/v9.1.1/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/MaskReg%E8%AE%A1%E7%AE%97/DeInterleave-12.md
[18] https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/language/simd.py#L1247-L1251
[19] https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/en/api/SIMD-API/basic_api/reg_vector_compute/type_conversion/Cast-45.md#constraints
[20] https://arxiv.org/abs/2605.09825
[21] https://github.com/PTO-ISA/tilelang-puzzles-ascend/tree/simt_compare/puzzles/torch/quant/answer
[22] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/common.py#L270-L282
[23] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/per_token_cast_asc.py#L59-L70
[24] https://github.com/PTO-ISA/tilelang-puzzles-ascend/tree/simt_compare
[25] https://github.com/tile-ai/tilelang-puzzles
[26] https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-vmi-Instruction-SPEC.md#ptovmivcmax--ptovmivcmin
[27] https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-micro-Instruction-SPEC.md
[28] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/cast_back_cuda.py#L78-L79
[29] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/cast_back_asc.py#L123-L128
[30] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/common.py#L285-L294
[31] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/per_channel_cast_asc.py#L222
[32] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/per_token_cast_asc.py#L96-L113
[33] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/per_token_cast_asc.py#L79-L93
[34] https://github.com/deepseek-ai/TileKernels/blob/v2.1.0-release/tile_kernels/quant/per_block_cast_cuda.py#L77-L83
[35] https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html
[36] https://github.com/PTO-ISA/tilelang-puzzles-ascend/blob/simt_compare/puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py
[37] https://gitcode.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md
[38] https://github.com/PTO-ISA/tilelang-puzzles-ascend/blob/simt_compare/perf/quant_vf_cycles.json
[39] https://github.com/hw-native-sys/PTOAS/releases/tag/vmi-v0.1.9
[40] https://zhuanlan.zhihu.com/p/2044533166694732929
[41] https://wp.doc.ic.ac.uk/spo/
```
