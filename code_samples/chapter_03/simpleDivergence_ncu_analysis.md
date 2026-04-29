# simpleDivergence ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_03/simpleDivergence`

编译命令：

```sh
cd code_samples/chapter_03
make simpleDivergence
```

## 基准运行

命令：

```sh
./simpleDivergence 256 1048576
```

结果：

| Kernel | 时间 |
| --- | ---: |
| `warmup` | 0.219 ms |
| `mathKernel1_divergent` | 79.600 ms |
| `mathKernel2_uniform` | 40.656 ms |
| `mathKernel3` | 0.014 ms |
| `mathKernel4` | 0.011 ms |

`mathKernel1_divergent` 和 `mathKernel2_uniform` 都包含大量 `sinf/cosf/tanf` 运算，因此耗时远高于轻量版本。前者按 `tid % 2` 分支，同一个 warp 内偶数和奇数线程走不同路径；后者按 `warp_id` 分支，同一个 warp 内线程路径一致。

## ncu 采集命令

使用的核心指标：

- `smsp__sass_branch_targets_threads_divergent.sum`: SASS 分支目标中发生分化的线程数累计。
- `smsp__sass_average_branch_targets_threads_uniform.pct`: 分支目标线程一致性的比例。
- `smsp__inst_executed_op_branch.sum`: 执行的分支指令数。
- `smsp__inst_executed.sum`: 执行的总指令数。

命令模板：

```sh
ncu \
  --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum \
  --kernel-name regex:<kernel_name> \
  --launch-count 1 \
  --csv \
  --page raw \
  --print-units base \
  ./simpleDivergence 256 <size>
```

实际采集命令：

```sh
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum --kernel-name regex:mathKernel1_divergent --launch-count 1 --csv --page raw --print-units base ./simpleDivergence 256 65536
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum --kernel-name regex:mathKernel2_uniform --launch-count 1 --csv --page raw --print-units base ./simpleDivergence 256 65536
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum --kernel-name regex:mathKernel3 --launch-count 1 --csv --page raw --print-units base ./simpleDivergence 256 1048576
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__inst_executed_op_branch.sum,smsp__inst_executed.sum --kernel-name regex:mathKernel4 --launch-count 1 --csv --page raw --print-units base ./simpleDivergence 256 1048576
```

## ncu 结果

| Kernel | size | `gpu__time_duration.sum` | `smsp__inst_executed.sum` | `smsp__inst_executed_op_branch.sum` | `smsp__sass_average_branch_targets_threads_uniform.pct` | `smsp__sass_branch_targets_threads_divergent.sum` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mathKernel1_divergent` | 65,536 | 9,435,488 ns | 1,509,117,296 | 281,495,672 | 86.42% | 28,276,048 |
| `mathKernel2_uniform` | 65,536 | 6,865,696 ns | 756,624,944 | 141,373,504 | 86.28% | 14,337,000 |
| `mathKernel3` | 1,048,576 | 10,176 ns | 4,521,984 | 425,984 | 71.43% | 65,536 |
| `mathKernel4` | 1,048,576 | 6,944 ns | 2,113,536 | 147,456 | 100.00% | 0 |

注意：`ncu` 下的时间不是普通运行时间。Nsight Compute 会插桩并可能进行 replay，所以这里的 `gpu__time_duration.sum` 主要用于同一采集配置下的相对比较。

## 结果分析

`mathKernel3` 与 `mathKernel4` 是最清晰的对照组。`mathKernel3` 的条件来自 `tid % 2 == 0`，同一个 warp 内线程 ID 连续，偶数线程和奇数线程会走不同分支，因此 `divergent.sum = 65536`。`mathKernel4` 使用 `tid >> 5` 得到 warp 编号，同一个 warp 内条件完全一致，因此 `uniform.pct = 100%` 且 `divergent.sum = 0`。

`mathKernel1_divergent` 更能体现分化的性能影响，因为它把同样的 `tid % 2` 分支放在重计算循环外。发生分化时，一个 warp 需要分别执行 if 和 else 两条路径，分支体越重，串行化成本越容易被观测到。基准运行中它约为 79.600 ms，而 `mathKernel2_uniform` 约为 40.656 ms。

`mathKernel2_uniform` 的源码级分支是按 `warp_id` 做的，同一个 warp 不会因这个 `if` 分化。但 `ncu` 统计的是 SASS 层面的所有分支，`sinf/cosf/tanf` 的库实现内部也包含数据相关分支，所以它仍然出现了非零的 `divergent.sum`。因此分析重计算版本时，不能把所有分化指标都归因于源码中的 `if`。

当前 `main` 没有启动 `mathKernel1` 和 `mathKernel2`，所以本次 `ncu` 不能直接采集它们。可用 `mathKernel3` 类比 `mathKernel1` 的轻量分化行为，用 `mathKernel4` 类比 `mathKernel2` 的 warp 粒度一致分支行为。
