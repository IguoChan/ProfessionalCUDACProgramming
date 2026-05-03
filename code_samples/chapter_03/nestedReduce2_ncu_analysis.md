# nestedReduce2 ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_03/nestedReduce2`
- 源文件: `code_samples/chapter_03/nestedReduce2.cu`
- 默认输入规模: `2048 * 512`，即 `1,048,576` 个 `int`
- 默认 block size: `512`
- 编译选项: `nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED`

编译命令：

```sh
cd code_samples/chapter_03
make nestedReduce2
```

`nestedReduce2.cu` 仍保留了带设备端 `cudaDeviceSynchronize()` 的 `gpuRecursiveReduce`，所以和前两个 nested reduction 示例一样，需要 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 才能在 CUDA 12.9 + `sm_89` 下编译。编译 warning 来自这个旧 CDP1 API。

## 程序结构概览

`nestedReduce2` 在 `nestedReduceNosync` 基础上新增 `gpuRecursiveReduce2`：

| Kernel | 主要策略 | 观察点 |
| --- | --- | --- |
| `reduceNeighbored` | 单层 kernel，相邻配对 block 内规约 | 非动态并行基线 |
| `gpuRecursiveReduce` | 每个 block 各自启动递归 child kernel，并显式设备端同步 | 动态并行 + 同步开销 |
| `gpuRecursiveReduceNosync` | 每个 block 各自启动递归 child kernel，但去掉同步 | 减少同步开销，但有时序风险 |
| `gpuRecursiveReduce2` | 只有 `blockIdx.x == 0 && threadIdx.x == 0` 启动下一层 child grid | 大幅减少 child kernel launch 数量 |

`gpuRecursiveReduce2` 的核心结构：

```cpp
idata[threadIdx.x] += idata[threadIdx.x + iStride];

if(threadIdx.x == 0 && blockIdx.x == 0)
{
    gpuRecursiveReduce2<<<gridDim.x, iStride / 2>>>(g_idata, g_odata,
            iStride / 2, iDim);
}
```

和 `gpuRecursiveReduceNosync` 不同，`gpuRecursiveReduce2` 每一层 child kernel 仍使用 `gridDim.x` 个 block 覆盖所有 block 的部分数据，但整个 grid 只有一个线程负责启动下一层。因此默认 `block=512` 时，递归层数约为 8 层，child launch 数也约为 8 次，而不是 `2048 * 8` 次。

## 普通运行结果

默认配置：

```sh
./nestedReduce2
```

结果：

| 实现 | grid 配置 | block 配置 | 普通运行时间 | sum |
| --- | ---: | ---: | ---: | ---: |
| CPU recursive reduce | - | - | `0.000877 sec` | `1048576` |
| `reduceNeighbored` | `2048` | `512` | `0.012335 sec` | `1048576` |
| `gpuRecursiveReduce` | `2048` | `512` | `0.012517 sec` | `1048576` |
| `gpuRecursiveReduceNosync` | `2048` | `512` | `0.006737 sec` | `1048576` |
| `gpuRecursiveReduce2` | `2048` | `512` | `0.000410 sec` | `1048576` |

本次默认运行中所有 GPU 结果都与 CPU 结果一致。

## ncu 采集命令

本报告采集以下指标：

- `gpu__time_duration.sum`: kernel 执行时间，单位为 ns。
- `smsp__inst_executed.sum`: 执行的总指令数。
- `smsp__inst_executed_op_branch.sum`: 执行的分支指令数。
- `smsp__sass_branch_targets_threads_divergent.sum`: SASS 分支目标中发生分化的线程数累计。
- `smsp__sass_average_branch_targets_threads_uniform.pct`: 分支目标线程一致性的比例。
- `smsp__thread_inst_executed_per_inst_executed.ratio`: 平均每条指令对应的活跃线程数。
- `smsp__warp_issue_stalled_barrier_per_warp_active.pct`: warp 因 barrier 等待而停顿的比例。

示例命令：

```sh
ncu \
  --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct \
  --kernel-name gpuRecursiveReduce2 \
  --launch-count 1 \
  --csv \
  --page raw \
  --print-units base \
  ./nestedReduce2 2048 512
```

## ncu 结果

默认配置 `./nestedReduce2 2048 512`：

| Kernel | `gpu__time_duration.sum` | `inst executed` | `branch inst` | `divergent.sum` | `uniform.pct` | `thread/inst` | `barrier stall` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpuRecursiveReduce2` | `415456 ns` | `38996796` | `10796102` | `16392` | `94.16%` | `25.09` | `0.45%` |

对照前一份报告中的默认配置数据：

| Kernel | 普通运行时间 | `ncu` 父 kernel 时间 | `barrier stall` |
| --- | ---: | ---: | ---: |
| `gpuRecursiveReduce` | `0.012517 sec` | `13.885 ms` | `38.51%` |
| `gpuRecursiveReduceNosync` | `0.006737 sec` | `7.932 ms` | `0.16%` |
| `gpuRecursiveReduce2` | `0.000410 sec` | `0.415 ms` | `0.45%` |

`ncu` 下程序打印的 `elapsed` 会被 profiler 插桩和 replay 放大；普通运行时间以上一节为准。

## 与 nestedReduce / nestedReduceNosync 的区别

`nestedReduce` 的同步版本中，每个初始 block 都会递归启动自己的 child kernel 链。默认 `grid=2048, block=512` 时，每个 block 约 8 层递归，child launch 数量约为 `2048 * 8 = 16384` 次，并且每层还有设备端同步。

`nestedReduceNosync` 删除了设备端同步和 block 同步，所以减少了 barrier stall，但仍然是每个 block 各自启动递归链。它减少的是“同步成本”，没有从根上减少 child launch 数量。

`nestedReduce2` 改变了递归粒度：每一层只由整个 grid 中的一个线程启动下一层 child grid，child grid 继续用 `gridDim.x` 个 block 覆盖所有数据块。默认配置下 child launch 数约为 8 次，远少于 `nestedReduce` / `nestedReduceNosync` 的数万次。因此普通运行时间降到 `0.000410 sec`，比前两个动态并行版本快很多。

这个版本仍有一个重要前提：父 kernel 启动 child grid 时，没有显式等待同一层所有 block 都完成 `idata[threadIdx.x] += idata[threadIdx.x + iStride]`。也就是说，`blockIdx.x == 0 && threadIdx.x == 0` 可能在其它 block 完成本层写回前启动下一层。当前测试中结果正确，但从跨 block 同步语义看，这种写法依赖动态并行 launch 的实际调度时序，不应当当作严格可靠的通用规约模板。

如果目标是性能和正确性，`nestedReduce2` 已经比前两个 nested 版本更接近合理方向，因为它显著减少了 launch 数量；但高性能 reduction 仍更应使用单层/少层 kernel、共享内存、warp shuffle、unrolling 和模板展开，而不是依赖递归 child kernel。

## 可复现实验命令

普通运行：

```sh
make nestedReduce2
./nestedReduce2
```

Nsight Compute：

```sh
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name gpuRecursiveReduce2 --launch-count 1 --csv --page raw --print-units base ./nestedReduce2 2048 512
```
