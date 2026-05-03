# nestedReduce ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_03/nestedReduce`
- 源文件: `code_samples/chapter_03/nestedReduce.cu`
- 默认输入规模: `2048 * 512`，即 `1,048,576` 个 `int`
- 默认 block size: `512`
- 编译选项: `nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED`

编译命令：

```sh
cd code_samples/chapter_03
make nestedReduce
```

注意：`-G` 会生成 debug device code，适合学习和调试，但性能明显不同于 release 构建。`nestedReduce` 使用 CUDA Dynamic Parallelism，在 kernel 内部再次启动 kernel，因此需要 `-rdc=true`。

当前 CUDA 12.9 中，设备端 `cudaDeviceSynchronize()` 已废弃，默认不再暴露给 device code。这个示例保留了书中的 CDP1 写法，所以 Makefile 对 `nestedReduce` 单独加了 `-DCUDA_FORCE_CDP1_IF_SUPPORTED`。该写法能在 `sm_89` 上编译运行，但编译器会提示它在 `sm_90+` 上不可用，后续更现代的实现应避免在设备端调用 `cudaDeviceSynchronize()`。

## 程序结构概览

`nestedReduce` 对同一份输入运行两个 GPU kernel：

| Kernel | 主要策略 | 观察点 |
| --- | --- | --- |
| `reduceNeighbored` | 单层 kernel，在每个 block 内用相邻配对完成规约 | 有 `tid % (2 * stride)` 导致的 warp 分化和 block 内同步 |
| `gpuRecursiveReduce` | 父 kernel 每个 block 启动一条递归 child-kernel 链 | 展示动态并行、设备端 kernel launch 和设备端同步开销 |

输入数组初始化后又被设为全 1：

```cpp
h_idata[i] = (int)( rand() & 0xFF );
h_idata[i] = 1;
```

因此默认规模下 `cpu_sum`、`gpu Neighbored` 和 `gpu nested` 的结果都应为 `1,048,576`。

## 普通运行结果

默认配置：

```sh
./nestedReduce
```

结果：

| 实现 | grid 配置 | block 配置 | 普通运行时间 | sum |
| --- | ---: | ---: | ---: | ---: |
| CPU recursive reduce | - | - | `0.000876 sec` | `1048576` |
| `reduceNeighbored` | `2048` | `512` | `0.012300 sec` | `1048576` |
| `gpuRecursiveReduce` | `2048` | `512` | `0.012238 sec` | `1048576` |

小规模配置：

```sh
./nestedReduce 8 512
```

结果：

| 实现 | grid 配置 | block 配置 | 普通运行时间 | sum |
| --- | ---: | ---: | ---: | ---: |
| CPU recursive reduce | - | - | `0.000005 sec` | `4096` |
| `reduceNeighbored` | `8` | `512` | `0.012218 sec` | `4096` |
| `gpuRecursiveReduce` | `8` | `512` | `0.000472 sec` | `4096` |

普通运行时间来自程序输出。GPU 时间包含 kernel launch、同步和 debug 编译带来的开销，不是单纯的设备指令执行时间。

## ncu 采集命令

采集的核心指标：

- `gpu__time_duration.sum`: kernel 执行时间，单位为 ns。
- `smsp__inst_executed.sum`: 执行的总指令数。
- `smsp__inst_executed_op_branch.sum`: 执行的分支指令数。
- `smsp__sass_branch_targets_threads_divergent.sum`: SASS 分支目标中发生分化的线程数累计。
- `smsp__sass_average_branch_targets_threads_uniform.pct`: 分支目标线程一致性的比例。
- `smsp__thread_inst_executed_per_inst_executed.ratio`: 平均每条指令对应的活跃线程数。
- `smsp__warp_issue_stalled_barrier_per_warp_active.pct`: warp 因 barrier 等待而停顿的比例。

采集普通 block 内规约：

```sh
ncu \
  --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct \
  --kernel-name regex:reduceNeighbored \
  --launch-count 1 \
  --csv \
  --page raw \
  --print-units base \
  ./nestedReduce 2048 512
```

采集递归父 kernel：

```sh
ncu \
  --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct \
  --kernel-name regex:gpuRecursiveReduce \
  --launch-count 1 \
  --csv \
  --page raw \
  --print-units base \
  ./nestedReduce 2048 512
```

`ncu` 下程序打印的 `elapsed` 会被 profiler 插桩和 replay 放大。分析普通运行耗时时看上一节，分析硬件事件时看下面表格。

## ncu 结果

默认配置 `./nestedReduce 2048 512`：

| Kernel | `gpu__time_duration.sum` | `inst executed` | `branch inst` | `divergent.sum` | `uniform.pct` | `thread/inst` | `barrier stall` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reduceNeighbored` | `55136 ns` | `26605568` | `2783232` | `196608` | `90.32%` | `24.12` | `17.14%` |
| `gpuRecursiveReduce` | `13754784 ns` | `400166000` | `98469135` | `26624` | `97.84%` | `15.93` | `34.97%` |

补充采样：`./nestedReduce 8 512` 下只采集 `gpuRecursiveReduce` 父 kernel，`gpu__time_duration.sum = 461696 ns`，`barrier stall = 61.71%`。这个小规模配置能更直观看出递归 kernel launch 和设备端同步在总耗时中的占比。

## 结果分析

`reduceNeighbored` 和 `reduceInteger` 中同名 kernel 的控制流模式一致。它用 `tid % (2 * stride) == 0` 选择活跃线程，早期 stride 会让同一个 warp 中部分 lane 活跃、部分 lane 不活跃，因此 `uniform.pct` 只有 `90.32%`，`thread/inst` 也只有 `24.12`。

`gpuRecursiveReduce` 的目标不是最快规约，而是展示动态并行。默认 `block=512` 时，每个初始 block 会从 `512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2` 递归。也就是说，每个初始 block 会发起 8 次 child kernel launch；默认 `grid=2048` 时，总 child launch 约为 `2048 * 8 = 16384` 次。大量设备端 kernel launch 和 `cudaDeviceSynchronize()` 是它的主要成本来源。

从 `ncu` 表看，`gpuRecursiveReduce` 的源码级分支分化并不比 `reduceNeighbored` 更严重，`uniform.pct` 反而更高。但它的总指令数和分支指令数远高于 `reduceNeighbored`，`barrier stall` 也更高。这说明这里的主要瓶颈不是单个规约循环的 warp divergence，而是动态并行运行时、递归控制流、设备端同步和大量小 kernel 的调度成本。

`gpuRecursiveReduce` 中每一层 child kernel 的规模都会减半。越到后面，kernel 中活跃工作越少，但仍要付出一次 device-side launch 和同步代价。因此这种写法适合作为 CUDA Dynamic Parallelism 的教学样例，不适合作为高性能规约的首选实现。实际优化时通常应优先使用单层或少层 kernel、warp shuffle、共享内存规约，或者像 `reduceInteger` 中的 unrolling 版本那样减少同步和部分和数量。

## 可复现实验命令

普通运行：

```sh
make nestedReduce
./nestedReduce
./nestedReduce 8 512
```

Nsight Compute：

```sh
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name regex:reduceNeighbored --launch-count 1 --csv --page raw --print-units base ./nestedReduce 2048 512
ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_branch.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_barrier_per_warp_active.pct --kernel-name regex:gpuRecursiveReduce --launch-count 1 --csv --page raw --print-units base ./nestedReduce 2048 512
```
