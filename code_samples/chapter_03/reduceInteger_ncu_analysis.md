# reduceInteger ncu 分析

## 测试环境

本报告分析对象：

- 程序路径: `code_samples/chapter_03/reduceInteger`
- 源文件: `code_samples/chapter_03/reduceInteger.cu`
- 默认输入规模: `1 << 24`，即 `16,777,216` 个 `int`
- 默认 block size: `512`
- 编译选项: 当前 `Makefile` 使用 `nvcc -g -G -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_03
make reduceInteger
```

注意：`-G` 会生成 debug device code，Nsight Compute 会提示 debug mode。该模式适合学习源码、观察控制流和调试，但会明显影响真实性能。因此本文把重点放在不同 reduction 写法的相对差异，尤其是 warp 分化的来源、规模和优化方向。

当前撰写环境执行 `./reduceInteger` 时返回 `no CUDA-capable device is detected`，因此本文没有填入实际 `ncu` 数值。下面给出完整采集命令和基于代码结构的分化推导；在有 CUDA GPU 的机器上运行命令即可补齐实测表格。

## 程序结构概览

`reduceInteger` 使用同一份随机整数输入，分别运行多种并行归约 kernel：

| Kernel | 主要策略 | 每个 block 覆盖元素数 | 主要观察点 |
| --- | --- | ---: | --- |
| `reduceNeighbored` | 相邻配对，`tid % (2 * stride) == 0` | `blockDim.x` | 分支分化最明显，且有取模开销 |
| `reduceNeighboredLess` | 相邻配对，但把活跃线程映射到连续 `tid` | `blockDim.x` | 分化显著减少 |
| `reduceInterleaved` | 交错配对，`tid < stride` | `blockDim.x` | 活跃线程连续，访存和控制流更规整 |
| `reduceUnrolling2` | 每线程先合并 2 个元素，再 interleaved 规约 | `2 * blockDim.x` | block 数减半，减少后续部分和数量 |
| `reduceUnrolling4` | 每线程先合并 4 个元素，再 interleaved 规约 | `4 * blockDim.x` | 更高的串行预合并，更多并行层级被消除 |
| `reduceUnrolling8` | 每线程先合并 8 个元素，再 interleaved 规约 | `8 * blockDim.x` | 进一步降低 block 数和同步轮数的相对开销 |
| `reduceUnrollWarps8` | `unrolling8` + 最后一个 warp 手工展开 | `8 * blockDim.x` | 消除最后几轮循环和 `__syncthreads()` |
| `reduceCompleteUnrollWarps8` | `unrolling8` + block 内大步长规约完全展开 + warp 展开 | `8 * blockDim.x` | 消除循环控制开销 |
| `reduceCompleteUnroll<iBlockSize>` | 模板完全展开 | `8 * blockDim.x` | block size 是编译期常量，便于编译器删分支 |

所有 GPU kernel 都只完成第一层 block 内归约：每个 block 写出一个部分和到 `g_odata[blockIdx.x]`。最终的 `gpu_sum` 由主机端循环累加所有 block 的部分和得到。

## nvprof 指标到 ncu 指标的替换

书中较老版本常用 `nvprof` 观察分支效率、warp 执行效率和全局内存吞吐。当前可以用 Nsight Compute CLI 采集更底层的 SASS 和 scheduler 指标。

建议采集的核心指标：

- `gpu__time_duration.sum`: kernel 执行时间，单位通常为 ns。
- `smsp__sass_branch_targets_threads_divergent.sum`: SASS 分支目标中发生分化的线程数累计。
- `smsp__sass_average_branch_targets_threads_uniform.pct`: 分支目标线程一致性的比例。
- `smsp__inst_executed_op_branch.sum`: 执行的分支指令数量。
- `smsp__inst_executed.sum`: 执行的总指令数量。
- `smsp__thread_inst_executed_per_inst_executed.ratio`: 平均每条执行指令对应的线程数。越接近 32，warp lane 利用率越高；明显偏低通常说明存在 predication、分支分化或尾部低活跃度。
- `smsp__warps_launched.sum`: 启动的 warp 数，用于把分化累计值归一化理解。
- `smsp__warp_issue_stalled_barrier_per_warp_active.pct`: warp 因 barrier 等待而停顿的比例，可观察 `__syncthreads()` 的成本。
- `smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct`: warp 因分支解析停顿的比例。
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`: L1TEX 侧全局 load 请求吞吐。
- `dram__bytes_read.sum.per_second`: DRAM 实际读吞吐。
- `dram__bytes_write.sum.per_second`: DRAM 实际写吞吐。

采集命令模板：

```sh
ncu \
  --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warps_launched.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second \
  --kernel-name regex:<kernel_name> \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./reduceInteger 512
```

单独采集某个 kernel 的示例：

```sh
ncu \
  --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warps_launched.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct \
  --kernel-name regex:reduceNeighbored \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./reduceInteger 512
```

如果只想对比所有 reduction kernel 的耗时，可以先采集时间指标：

```sh
ncu \
  --metrics gpu__time_duration.sum \
  --kernel-name regex:reduce \
  --launch-count 9 \
  --print-units base \
  --print-metric-name name \
  ./reduceInteger 512
```

注意：`ncu` 下程序打印的 `elapsed` 会被 profiler 插桩和 replay 显著放大。普通运行时间应以直接运行 `./reduceInteger 512` 的输出为准；硬件事件和分化指标以 `ncu` 表格为准。

## 普通运行结果记录表

在有 CUDA GPU 的机器上直接运行：

```sh
./reduceInteger 512
```

预期程序会输出每个 kernel 的 `gpu_sum`，这些值都应与 `cpu_sum` 一致。可以把实测时间填入下表：

| Kernel | grid 配置 | block 配置 | 普通运行时间 | `gpu_sum == cpu_sum` |
| --- | ---: | ---: | ---: | --- |
| `reduceNeighbored` | `32768` | `512` | 待测 | 待测 |
| `reduceNeighboredLess` | `32768` | `512` | 待测 | 待测 |
| `reduceInterleaved` | `32768` | `512` | 待测 | 待测 |
| `reduceUnrolling2` | `16384` | `512` | 待测 | 待测 |
| `reduceUnrolling4` | `8192` | `512` | 待测 | 待测 |
| `reduceUnrolling8` | `4096` | `512` | 待测 | 待测 |
| `reduceUnrollWarps8` | `4096` | `512` | 待测 | 待测 |
| `reduceCompleteUnrollWarps8` | `4096` | `512` | 待测 | 待测 |
| `reduceCompleteUnroll<512>` | `4096` | `512` | 待测 | 待测 |

默认规模 `1 << 24` 与 `512`、`2 * 512`、`4 * 512`、`8 * 512` 都整除，因此默认配置下 unrolling kernel 的边界判断理论上不会出现尾部不完整 block。这能让分析重点集中在规约模式本身，而不是尾部越界处理。

## ncu 结果记录表

建议先用默认 `blocksize=512` 填入以下表格：

| Kernel | `gpu__time_duration.sum` | `divergent.sum` | `uniform.pct` | branch inst | thread/inst | barrier stall | branch resolving stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reduceNeighbored` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceNeighboredLess` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceInterleaved` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceUnrolling2` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceUnrolling4` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceUnrolling8` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceUnrollWarps8` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceCompleteUnrollWarps8` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| `reduceCompleteUnroll<512>` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

内存吞吐可以单独记录：

| Kernel | L1TEX global load 吞吐 | DRAM 读吞吐 | DRAM 写吞吐 | 备注 |
| --- | ---: | ---: | ---: | --- |
| `reduceNeighbored` | 待测 | 待测 | 待测 | 多轮全局内存原地写回 |
| `reduceNeighboredLess` | 待测 | 待测 | 待测 | 访存位置仍是相邻配对 |
| `reduceInterleaved` | 待测 | 待测 | 待测 | 右半段折叠到左半段 |
| `reduceUnrolling2` | 待测 | 待测 | 待测 | 初始读取 2 份输入 |
| `reduceUnrolling4` | 待测 | 待测 | 待测 | 初始读取 4 份输入 |
| `reduceUnrolling8` | 待测 | 待测 | 待测 | 初始读取 8 份输入 |
| `reduceUnrollWarps8` | 待测 | 待测 | 待测 | 最后 warp 阶段展开 |
| `reduceCompleteUnrollWarps8` | 待测 | 待测 | 待测 | 循环完全展开 |
| `reduceCompleteUnroll<512>` | 待测 | 待测 | 待测 | 模板实例化 |

## 并行规约中的分化来源

并行规约的核心模式是“每一轮只让一部分线程继续工作”。随着 stride 变化，活跃线程数量不断减少。这个过程天然会带来两个问题：

1. 活跃线程比例下降。即使没有分支分化，后几轮也只有很少线程真正做加法，其余线程等待同步。
2. 活跃线程在 warp 内的分布不同。如果活跃线程和非活跃线程混在同一个 warp 中，就会产生 warp divergence。

CUDA warp 通常包含 32 个连续线程。一个 warp 内的线程如果对同一个分支条件给出不同结果，硬件需要分别执行不同路径，并通过掩码控制哪些 lane 生效。对 reduction 这类循环来说，分化不仅增加控制流成本，还会让平均每条指令真正工作的线程数下降。

### `reduceNeighbored`

核心条件：

```cpp
if ((tid % (2 * stride)) == 0)
{
    idata[tid] += idata[tid + stride];
}
```

默认 `blockDim.x = 512`，一个 block 有 16 个 warp。各轮 stride 的活跃线程模式如下：

| stride | 活跃 tid 模式 | 活跃线程数/block | 典型 warp 内形态 | 分化特征 |
| ---: | --- | ---: | --- | --- |
| 1 | `0, 2, 4, ...` | 256 | 每个 warp 16 个 lane 活跃、16 个 lane 非活跃 | 所有 warp 分化 |
| 2 | `0, 4, 8, ...` | 128 | 每个 warp 8 个 lane 活跃 | 所有 warp 分化 |
| 4 | `0, 8, 16, ...` | 64 | 每个 warp 4 个 lane 活跃 | 所有 warp 分化 |
| 8 | `0, 16, 32, ...` | 32 | 每个相关 warp 2 个 lane 活跃 | 所有 warp 分化 |
| 16 | `0, 32, 64, ...` | 16 | 每个 warp 1 个 lane 活跃 | 所有 warp 分化 |
| 32 | `0, 64, 128, ...` | 8 | 一半 warp 有 1 个 lane 活跃，另一半全不活跃 | 部分 warp 分化 |
| 64 | `0, 128, 256, 384` | 4 | 更少 warp 有 1 个 lane 活跃 | 部分 warp 分化 |
| 128 | `0, 256` | 2 | 只有两个 warp 各 1 个 lane 活跃 | 部分 warp 分化 |
| 256 | `0` | 1 | 只有第一个 warp 的 lane 0 活跃 | 一个 warp 分化 |

这个 kernel 非常适合作为“坏例子”观察分化。前几轮每个 warp 都有交错的活跃 lane，硬件必须在同一条控制流上反复用掩码屏蔽一半、四分之三、八分之七的 lane。后几轮虽然分化 warp 数减少，但每轮仍有 `__syncthreads()`，大量非活跃线程继续参与同步等待。

除此之外，`tid % (2 * stride)` 本身也比简单比较更重。即使编译器能把部分 2 的幂取模优化成位运算，这个条件仍然比 `tid < stride` 更复杂。这个 kernel 的性能损失来自两部分：warp 内 lane 利用率低，以及分支条件计算更复杂。

### `reduceNeighboredLess`

核心条件：

```cpp
int index = 2 * stride * tid;

if (index < blockDim.x)
{
    idata[index] += idata[index + stride];
}
```

它没有改变数学上的相邻配对规约，但改变了“哪些 tid 参与本轮工作”。默认 `blockDim.x = 512` 时：

| stride | 条件等价形式 | 活跃线程数/block | warp 形态 | 分化特征 |
| ---: | --- | ---: | --- | --- |
| 1 | `tid < 256` | 256 | 前 8 个 warp 全活跃，后 8 个 warp 全不活跃 | 无明显 warp 内分化 |
| 2 | `tid < 128` | 128 | 前 4 个 warp 全活跃 | 无明显 warp 内分化 |
| 4 | `tid < 64` | 64 | 前 2 个 warp 全活跃 | 无明显 warp 内分化 |
| 8 | `tid < 32` | 32 | 第 0 个 warp 全活跃 | 无明显 warp 内分化 |
| 16 | `tid < 16` | 16 | 第 0 个 warp 只有 16 lane 活跃 | 1 个 warp 分化 |
| 32 | `tid < 8` | 8 | 第 0 个 warp 只有 8 lane 活跃 | 1 个 warp 分化 |
| 64 | `tid < 4` | 4 | 第 0 个 warp 只有 4 lane 活跃 | 1 个 warp 分化 |
| 128 | `tid < 2` | 2 | 第 0 个 warp 只有 2 lane 活跃 | 1 个 warp 分化 |
| 256 | `tid < 1` | 1 | 第 0 个 warp 只有 1 lane 活跃 | 1 个 warp 分化 |

关键变化是：前几轮的活跃线程变成连续 tid。对 warp 来说，要么整个 warp 都满足条件，要么整个 warp 都不满足条件。只有当活跃线程数小于一个 warp 时，才不可避免地在第 0 个 warp 内分化。

因此它相比 `reduceNeighbored` 的预期变化是：

- `smsp__sass_branch_targets_threads_divergent.sum` 明显下降。
- `smsp__thread_inst_executed_per_inst_executed.ratio` 上升。
- branch resolving stall 可能下降。
- 总分支指令数量不一定大幅下降，因为循环轮数仍然相同。
- barrier stall 仍然存在，因为每轮仍有 `__syncthreads()`。

这个版本体现了 reduction 优化中的一个重要原则：不一定先改变算法的数学结构，只要重新安排活跃线程，使它们在 warp 内连续，就可以显著降低分化。

### `reduceInterleaved`

核心条件：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
{
    if (tid < stride)
    {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();
}
```

默认 `blockDim.x = 512` 时：

| stride | 活跃线程数/block | warp 形态 | 分化特征 |
| ---: | ---: | --- | --- |
| 256 | 256 | 前 8 个 warp 全活跃 | 无明显 warp 内分化 |
| 128 | 128 | 前 4 个 warp 全活跃 | 无明显 warp 内分化 |
| 64 | 64 | 前 2 个 warp 全活跃 | 无明显 warp 内分化 |
| 32 | 32 | 第 0 个 warp 全活跃 | 无明显 warp 内分化 |
| 16 | 16 | 第 0 个 warp 半活跃 | 1 个 warp 分化 |
| 8 | 8 | 第 0 个 warp 8 lane 活跃 | 1 个 warp 分化 |
| 4 | 4 | 第 0 个 warp 4 lane 活跃 | 1 个 warp 分化 |
| 2 | 2 | 第 0 个 warp 2 lane 活跃 | 1 个 warp 分化 |
| 1 | 1 | 第 0 个 warp 1 lane 活跃 | 1 个 warp 分化 |

这个分化模式和 `reduceNeighboredLess` 很接近，但代码更简单，条件是直接的 `tid < stride`。它从一开始就做右半段向左半段的折叠，活跃线程始终是连续低编号 tid。

从分化角度看，`reduceInterleaved` 的主要优势是前半程完全避免 warp 内交错活跃 lane。真正不可避免的分化只发生在 `stride < warpSize` 的最后几轮，因为此时一个 block 中剩余的有效工作量已经少于一个 warp。

从同步角度看，它仍然每轮调用 `__syncthreads()`。当 stride 进入 16、8、4、2、1 时，只有第 0 个 warp 的一部分线程在工作，但整个 block 的所有线程仍要参与 barrier。这是后续 warp unrolling 优化的直接动机。

### `reduceUnrolling2/4/8`

以 `reduceUnrolling8` 为例：

```cpp
unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
int *idata = g_idata + blockIdx.x * blockDim.x * 8;

if (idx + 7 * blockDim.x < n)
{
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    ...
    int b4 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
}
```

unrolling 的核心思想是让一个线程在进入 block 内规约前先串行合并多个元素。默认 `size = 1 << 24`、`blocksize = 512` 时：

| Kernel | 启动 block 数 | 相比未展开版本 | 每线程预合并元素数 |
| --- | ---: | ---: | ---: |
| 未展开 | 32768 | 1x | 1 |
| `reduceUnrolling2` | 16384 | 1/2 | 2 |
| `reduceUnrolling4` | 8192 | 1/4 | 4 |
| `reduceUnrolling8` | 4096 | 1/8 | 8 |

这类优化对分化的影响不只是“减少 block 数”。更重要的是，它把原本需要更多 block、更多部分和、更多同步轮次承担的工作，提前变成每个线程内部的串行加法。线程内部串行加法不会产生 warp 分支分化。

默认输入规模能被展开倍数整除，因此 `idx + 7 * blockDim.x < n` 对所有启动线程都为真。也就是说，默认配置下 unrolling8 的预合并边界判断不会造成 warp 内真假混杂。若换成不能整除的输入规模，最后一个 block 可能出现部分线程越界，届时这个边界判断会产生尾部 warp 分化。

unrolling 后仍然有 interleaved 的 block 内规约，所以最后 `stride < 32` 的几轮分化和 barrier 等待依旧存在。它主要优化的是：

- 减少启动 block 数。
- 减少 `g_odata` 中部分和数量。
- 增加每个线程的独立工作量，提高指令级并行机会。
- 降低同步和分支控制开销在总工作量中的占比。

### `reduceUnrollWarps8`

核心变化：

```cpp
for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
{
    if (tid < stride)
    {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();
}

if (tid < 32)
{
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid +  8];
    vmem[tid] += vmem[tid +  4];
    vmem[tid] += vmem[tid +  2];
    vmem[tid] += vmem[tid +  1];
}
```

这个版本把循环停在 `stride > 32`，即只用循环规约到 64 个值。剩下的 64 到 1 的规约交给第 0 个 warp 手工展开。

它的意义非常关键：最后几轮是 reduction 中最“低效”的阶段。以 `blockDim.x = 512` 为例，普通 interleaved 版本最后几轮是：

| stride | 真正做加法的线程数 | 是否需要 `__syncthreads()` | lane 利用率 |
| ---: | ---: | --- | ---: |
| 16 | 16 | 是 | 50.00% |
| 8 | 8 | 是 | 25.00% |
| 4 | 4 | 是 | 12.50% |
| 2 | 2 | 是 | 6.25% |
| 1 | 1 | 是 | 3.125% |

warp 展开后，第 0 个 warp 的 32 个线程执行固定加法序列。虽然每一步仍然有越来越少的有效数据依赖，但它避免了循环分支和 block 级同步。对一个 warp 内的线程来说，硬件按 SIMT 模式推进，不需要 `__syncthreads()` 来同步整个 block。

`volatile` 指针用于约束编译器不要把这些内存访问过度缓存到寄存器中，确保每一步能读到上一行写回的结果。这个示例使用的是全局内存原地规约，不是共享内存版本；在更现代的 CUDA 写法中，最后一个 warp 的规约也常用 warp shuffle 指令实现。

预期 `ncu` 变化：

- `barrier stall` 下降，因为最后 5 到 6 次 block 级同步被移除。
- branch 指令数量下降，因为最后几轮循环控制被展开。
- `gpu__time_duration.sum` 通常低于 `reduceUnrolling8`。
- 分化指标不一定降到 0，因为 `if (tid < 32)` 本身在 block 层面只有第 0 个 warp 为真，其余 warp 为假；对单个 warp 来说，这个分支是 uniform 的，但 SASS 层面还可能统计其他控制流。

### `reduceCompleteUnrollWarps8`

核心变化：

```cpp
if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
__syncthreads();

if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
__syncthreads();

if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
__syncthreads();

if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
__syncthreads();
```

这个版本把原本的循环写成固定的规约步骤。对于默认 `blockDim.x = 512`：

- `blockDim.x >= 1024` 为假，跳过 512 步。
- `blockDim.x >= 512` 为真，执行 256 步。
- `blockDim.x >= 256` 为真，执行 128 步。
- `blockDim.x >= 128` 为真，执行 64 步。
- 最后 64 到 1 由 warp 展开完成。

它消除了循环递减、循环条件判断、stride 更新等控制流成本。但这里的 `blockDim.x` 是运行时值，编译器不一定能完全删除不可能执行的分支。它仍然比循环版更直观地表达了固定 block size 下的规约结构。

### `reduceCompleteUnroll<iBlockSize>`

模板版把 block size 变成编译期常量：

```cpp
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(...)
{
    if (iBlockSize >= 1024 && tid < 512) ...
    if (iBlockSize >= 512 && tid < 256) ...
    if (iBlockSize >= 256 && tid < 128) ...
    if (iBlockSize >= 128 && tid < 64) ...
}
```

默认 `blocksize = 512` 时，主机端调用：

```cpp
reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
```

这样编译器在实例化 `reduceCompleteUnroll<512>` 时已经知道：

- `iBlockSize >= 1024` 永远为假，可以删除。
- `iBlockSize >= 512`、`>= 256`、`>= 128` 永远为真，可以简化条件。

因此模板版通常是本文件中控制流最精简的版本。它体现了 CUDA reduction 优化中的常见手法：如果某些参数在运行时其实只会取少数固定值，把它们提升为模板参数可以让编译器生成专门版本。

## 默认 blocksize=512 的分化对比推导

只看 block 内规约阶段，默认 `blockDim.x = 512` 时，各版本的分化趋势可以概括如下：

| Kernel | 前半程分化 | 最后一个 warp 分化 | 同步轮数 | 控制流复杂度 | 预期表现 |
| --- | --- | --- | ---: | --- | --- |
| `reduceNeighbored` | 高，活跃 lane 交错分布 | 高 | 9 | `tid %` 条件较重 | 最慢或接近最慢 |
| `reduceNeighboredLess` | 低，活跃 warp 连续 | 不可避免 | 9 | index 计算较简单 | 明显优于 neighbored |
| `reduceInterleaved` | 低，活跃 warp 连续 | 不可避免 | 9 | `tid < stride` 最简单 | 通常优于前两者 |
| `reduceUnrolling2` | 同 interleaved | 不可避免 | 9 | 额外预合并 | 通常优于 interleaved |
| `reduceUnrolling4` | 同 interleaved | 不可避免 | 9 | 更多预合并 | 通常继续改善 |
| `reduceUnrolling8` | 同 interleaved | 不可避免 | 9 | 更多预合并 | block 数最少 |
| `reduceUnrollWarps8` | 低 | 通过 warp 展开降低开销 | 3 | 手工展开 | 通常优于 unrolling8 |
| `reduceCompleteUnrollWarps8` | 低 | 通过 warp 展开降低开销 | 3 | 无循环 | 通常更优 |
| `reduceCompleteUnroll<512>` | 低 | 通过 warp 展开降低开销 | 3 | 编译期删分支 | 通常最好或接近最好 |

这里的“同步轮数”按默认 `blockDim.x = 512` 估算：

- 普通 interleaved 从 `256` 到 `1`，共有 9 轮，每轮一次 `__syncthreads()`。
- warp 展开版本只在 `256`、`128`、`64` 这些跨 warp 阶段同步，最后 `32` 到 `1` 不再做 block 级同步。

这正是并行规约优化的主线：

1. 先把活跃线程从交错分布改成连续分布，减少 warp divergence。
2. 再通过每线程多元素预合并减少 block 数和部分和数量。
3. 再把最后一个 warp 的低效阶段展开，减少同步和循环开销。
4. 最后用完全展开和模板参数让编译器进一步删除控制流。

## 为什么 `reduceNeighbored` 分化最直观

`reduceNeighbored` 的第一轮 stride 为 1：

```cpp
if ((tid % 2) == 0)
```

一个 warp 中 lane 0、2、4、...、30 为真，lane 1、3、5、...、31 为假。硬件执行 if 路径时只启用偶数 lane，奇数 lane 空闲。执行完后，整个 warp 继续到 `__syncthreads()`。这意味着这条加法指令的有效 lane 利用率只有 50%。

第二轮 stride 为 2：

```cpp
if ((tid % 4) == 0)
```

一个 warp 中只有 lane 0、4、8、...、28 为真，有效 lane 利用率下降到 25%。

第三轮 stride 为 4，有效 lane 利用率下降到 12.5%。这种模式非常适合用来说明“线程总数很多”不等于“每条指令都充分利用了 32 个 lane”。并行规约越到后面，理论并行度越低；如果活跃 lane 又交错分布，就会更早暴露分化成本。

`reduceNeighboredLess` 和 `reduceInterleaved` 的改进点就是让前几轮变成：

```cpp
if (tid < 256)
if (tid < 128)
if (tid < 64)
if (tid < 32)
```

此时前 8、4、2、1 个 warp 是整 warp 活跃，后面的 warp 是整 warp 不活跃。对单个 warp 而言，分支结果一致，因此不会出现同一 warp 内一半 lane 走 if、一半 lane 跳过 if 的情况。

## `__syncthreads()` 与分化的关系

`__syncthreads()` 本身不是分支分化，但它会放大 reduction 后半程低活跃度的成本。原因是 block 内所有线程都必须到达 barrier，哪怕本轮只有 1 个线程真正做了加法。

以 `blockDim.x = 512` 的普通 interleaved 规约为例，最后五轮的有效工作总共只有：

```text
16 + 8 + 4 + 2 + 1 = 31 次加法
```

但每轮都要让 512 个线程参与同步。warp 展开优化正是针对这个问题：当剩余工作已经落在一个 warp 内时，不再使用 block 级同步，而是利用 warp 内隐式同步执行固定加法序列。

因此分析 reduction 时不要只看 `divergent.sum`。还应该同时看：

- barrier stall 是否下降。
- 总分支指令是否下降。
- kernel 时间是否下降。
- thread/inst ratio 是否改善。

有时某个优化并不显著降低 SASS 层面的 divergent 计数，但仍然通过减少同步和循环控制获得性能提升。

## 内存访问角度

本示例没有使用共享内存，而是在全局内存 `g_idata` 上原地规约。这让代码更容易对照书中优化步骤，但也意味着每一轮规约都包含全局内存读写。真实高性能 reduction 通常会先把数据加载到 shared memory，或者使用 warp shuffle 做 warp 内规约。

几个版本的内存访问特点：

- `reduceNeighbored` 和 `reduceNeighboredLess` 都是相邻配对，前几轮访问 `idata[tid]` 与 `idata[tid + stride]`。区别主要在控制流，不是数据总量。
- `reduceInterleaved` 每轮把右半段折叠到左半段，访问模式更规整，条件也更简单。
- `unrolling2/4/8` 在 kernel 开头引入更多全局 load，但这些 load 是每个线程串行执行，减少了后续 block 数和部分和数量。
- warp 展开和完全展开主要减少控制流与同步，不改变初始全局数据量。

用 `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` 和 `dram__bytes_read.sum.per_second` 对比时，应注意这些 kernel 执行时间很短，且 `ncu` replay 可能影响绝对吞吐。更稳妥的做法是同时看 kernel 时间、指令数、分支分化和 barrier stall。

## 不同 block size 的分析建议

主程序允许传入 block size：

```sh
./reduceInteger 1024
./reduceInteger 512
./reduceInteger 256
./reduceInteger 128
./reduceInteger 64
```

模板完全展开版只支持源码中 `switch` 列出的 `1024`、`512`、`256`、`128`、`64`。如果传入其他值，最后一个模板 kernel 不会正常选择匹配实例，不适合作为有效性能样本。

不同 block size 会改变规约轮数：

| block size | warp 数/block | 普通规约轮数 | 进入 warp 展开前的跨 warp轮数 | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 32 | 10 | 4 | block 最大，规约轮数最多 |
| 512 | 16 | 9 | 3 | 默认配置 |
| 256 | 8 | 8 | 2 | 同步轮数更少 |
| 128 | 4 | 7 | 1 | 更快进入 warp 阶段 |
| 64 | 2 | 6 | 0 或 1 | 大部分工作很快落入一个 warp |

block size 并不是越大越好。更大的 block 可以让单个 block 覆盖更多元素，但也可能降低 SM 上同时驻留的 block 数，增加每个 block 内的同步等待。reduction 的调优通常需要在以下因素之间平衡：

- 每个 block 的线程数。
- 每个线程预合并的元素数。
- SM occupancy。
- 全局内存吞吐。
- barrier stall。
- 最后几轮低活跃度的成本。

## 可复现实验命令

普通运行：

```sh
cd code_samples/chapter_03
make reduceInteger
./reduceInteger 512
./reduceInteger 256
./reduceInteger 1024
```

对比分化最明显的三个 kernel：

```sh
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct --kernel-name regex:reduceNeighbored --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct --kernel-name regex:reduceNeighboredLess --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct --kernel-name regex:reduceInterleaved --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
```

对比 unrolling 和 warp 展开：

```sh
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name regex:reduceUnrolling8 --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name regex:reduceUnrollWarps8 --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
ncu --metrics gpu__time_duration.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name regex:reduceCompleteUnrollWarps8 --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
```

采集内存吞吐：

```sh
ncu --metrics gpu__time_duration.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second --kernel-name regex:reduceCompleteUnroll --launch-count 1 --print-units base --print-metric-name name ./reduceInteger 512
```

## 总结

`reduceInteger` 这组 kernel 展示了并行规约优化从“控制流分化”到“同步开销”再到“循环展开”的完整路径。

最初的 `reduceNeighbored` 使用 `tid % (2 * stride)` 选择活跃线程，导致同一个 warp 内 lane 交错活跃，是观察 warp divergence 的典型反例。`reduceNeighboredLess` 和 `reduceInterleaved` 通过让活跃线程连续化，使前半程大多数 warp 要么全活跃、要么全不活跃，从而显著减少 warp 内分化。

`unrolling2/4/8` 把多个输入元素的归约提前到单个线程内部完成，减少 block 数和部分和数量，使同步和控制流开销在总工作中的占比下降。`reduceUnrollWarps8` 进一步针对最后一个 warp 的低活跃度阶段，去掉最后几轮 block 级同步和循环控制。`reduceCompleteUnrollWarps8` 与模板版 `reduceCompleteUnroll<iBlockSize>` 则通过完全展开减少循环分支；模板版还让编译器基于编译期 block size 删除无关分支。

因此，这个示例体现的核心结论是：并行规约的性能瓶颈不只是加法数量，而是活跃线程如何分布在 warp 中、低活跃度阶段是否仍在支付 block 级同步成本，以及编译器能否看清固定规约结构并消除控制流开销。
