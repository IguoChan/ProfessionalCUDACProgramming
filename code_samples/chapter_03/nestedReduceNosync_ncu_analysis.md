# nestedReduceNosync ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_03/nestedReduceNosync`
- 源文件: `code_samples/chapter_03/nestedReduceNosync.cu`
- 默认输入规模: `2048 * 512`，即 `1,048,576` 个 `int`
- 默认 block size: `512`
- 编译选项: `nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED`

编译命令：

```sh
cd code_samples/chapter_03
make nestedReduceNosync
```

`nestedReduceNosync.cu` 保留了 `nestedReduce.cu` 中带设备端 `cudaDeviceSynchronize()` 的 `gpuRecursiveReduce`，因此仍然需要 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 才能在 CUDA 12.9 + `sm_89` 下编译。编译时会看到设备端 `cudaDeviceSynchronize()` 已废弃的 warning，这是预期现象。

## 程序结构概览

`nestedReduceNosync` 在 `nestedReduce` 的基础上新增了一个无同步版本：

| Kernel | 主要策略 | 观察点 |
| --- | --- | --- |
| `reduceNeighbored` | 单层 kernel，相邻配对 block 内规约 | 作为非动态并行基线 |
| `gpuRecursiveReduce` | 每层递归 child launch 后调用设备端 `cudaDeviceSynchronize()`，并用 `__syncthreads()` 同步 block | 动态并行 + 显式设备端同步成本 |
| `gpuRecursiveReduceNosync` | 线程完成本层加法后，由 `tid == 0` 立即启动下一层 child kernel | 去掉设备端同步和 block 同步后的开销变化 |

`gpuRecursiveReduceNosync` 的核心差异：

```cpp
if(istride > 1 && tid < istride)
{
    idata[tid] += idata[tid + istride];

    if(tid == 0)
    {
        gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
    }
}
```

它没有在 child kernel launch 前执行 `__syncthreads()`，也没有在 launch 后调用设备端 `cudaDeviceSynchronize()`。

## 普通运行结果

默认配置：

```sh
./nestedReduceNosync
```

结果：

| 实现 | grid 配置 | block 配置 | 普通运行时间 | sum |
| --- | ---: | ---: | ---: | ---: |
| CPU recursive reduce | - | - | `0.000895 sec` | `1048576` |
| `reduceNeighbored` | `2048` | `512` | `0.012420 sec` | `1048576` |
| `gpuRecursiveReduce` | `2048` | `512` | `0.012529 sec` | `1048576` |
| `gpuRecursiveReduceNosync` | `2048` | `512` | `0.006660 sec` | `1048576` |

本次运行中三个 GPU 结果都与 CPU 结果一致。需要注意，`gpuRecursiveReduceNosync` 虽然在这次测试中得到正确结果，但源码缺少 child launch 前的 block 同步，理论上存在读写时序风险。

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
  --kernel-name gpuRecursiveReduceNosync \
  --launch-count 1 \
  --csv \
  --page raw \
  --print-units base \
  ./nestedReduceNosync 2048 512
```

## ncu 结果

默认配置 `./nestedReduceNosync 2048 512`：

| Kernel | `gpu__time_duration.sum` | `inst executed` | `branch inst` | `divergent.sum` | `uniform.pct` | `thread/inst` | `barrier stall` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reduceNeighbored` | `55168 ns` | `26605568` | `2783232` | `196608` | `90.32%` | `24.12` | `17.15%` |
| `gpuRecursiveReduce` | `13885408 ns` | `400902350` | `98717778` | `26624` | `97.84%` | `15.91` | `38.51%` |
| `gpuRecursiveReduceNosync` | `7931840 ns` | `228170188` | `60095282` | `26624` | `97.22%` | `9.62` | `0.16%` |

`ncu` 下程序打印的 `elapsed` 会被 profiler 插桩和 replay 放大。上表中的 `gpu__time_duration.sum` 用于同一采集配置下横向比较；普通运行时间以上一节为准。

## 与 nestedReduce 的区别

`nestedReduce.cu` 的 `gpuRecursiveReduce` 在每一层递归中有三步：

1. 当前 block 中活跃线程写回 `idata[tid] += idata[tid + istride]`。
2. `__syncthreads()` 等待当前 block 内所有写回完成。
3. `tid == 0` 启动 child kernel，并调用设备端 `cudaDeviceSynchronize()` 等 child 完成，然后再次 `__syncthreads()`。

`nestedReduceNosync.cu` 新增的 `gpuRecursiveReduceNosync` 把这些同步删掉，只让 `tid == 0` 在自己完成写回后立即 launch child kernel。这样显著减少了 barrier 和设备端同步开销。实测中普通运行从 `0.012529 sec` 降到 `0.006660 sec`，`ncu` 的父 kernel 时间从 `13.885 ms` 降到 `7.932 ms`，`barrier stall` 也从 `38.51%` 降到 `0.16%`。

但这个优化有明确风险：下一层 child kernel 会读取 `idata[0..istride-1]`，这些位置本应由当前层的 `tid < istride` 线程写完。没有 `__syncthreads()` 时，`tid == 0` 可能在其它线程完成写回前就启动 child kernel。当前测试能得到正确结果，不代表这个写法在所有 GPU、编译选项、输入规模和调度条件下都严格可靠。

因此，`gpuRecursiveReduceNosync` 的价值是展示“设备端同步开销有多重”。它比 `nestedReduce` 的同步版本快，但它不是一个严谨的通用 reduction 实现。如果要追求性能和正确性，仍应优先使用 `reduceInteger` 中的 unrolling、warp 展开、模板完全展开，或者使用共享内存 / warp shuffle 的规约版本。

## 可复现实验命令

普通运行：

```sh
make nestedReduceNosync
./nestedReduceNosync
```

Nsight Compute：

```sh
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name reduceNeighbored --launch-count 1 --csv --page raw --print-units base ./nestedReduceNosync 2048 512
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name gpuRecursiveReduce --launch-count 1 --csv --page raw --print-units base ./nestedReduceNosync 2048 512
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name gpuRecursiveReduceNosync --launch-count 1 --csv --page raw --print-units base ./nestedReduceNosync 2048 512
```
