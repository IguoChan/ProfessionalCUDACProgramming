# readSegmentUnroll ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_04/readSegmentUnroll`
- 源文件: `code_samples/chapter_04/readSegmentUnroll.cu`
- 采集数组规模: `1 << 26` 个 `float`
- 默认 block 大小: `512`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_04
make readSegmentUnroll
```

普通运行：

```sh
./readSegmentUnroll 0 512 26
./readSegmentUnroll 11 512 26
./readSegmentUnroll 32 512 26
```

## 程序结构概览

`readSegmentUnroll.cu` 在 `readSegment.cu` 的基础上增加了两个循环展开版本，用于观察减少 block 数量后，对非对齐 global load 的影响：

| Kernel | grid 配置 | 每个线程处理元素数 | 主要行为 |
| --- | ---: | ---: | --- |
| `readOffset` | `grid.x` | 1 | 每个线程读取 `A[i + offset]` 和 `B[i + offset]`，写入 `C[i]` |
| `readOffsetUnroll2` | `grid.x / 2` | 2 | 每个线程处理两个相隔 `blockDim.x` 的元素 |
| `readOffsetUnroll4` | `grid.x / 4` | 4 | 每个线程处理四个相隔 `blockDim.x` 的元素 |

核心差异在于 unroll 版本减少了 kernel 中参与调度的 block 数量，但总的全局内存读写元素数不变。因此它主要影响调度和指令开销，不会从根本上改变同一 offset 下的 global load 访问形态。

## 普通运行结果

默认使用 `block=512`、`nElem=1 << 26`：

| offset | warmup 时间 | `readOffset` 时间 | `unroll2` 时间 | `unroll4` 时间 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001027 sec | 0.001204 sec | 0.001224 sec | 0.001252 sec |
| 11 | 0.002538 sec | 0.001203 sec | 0.001270 sec | 0.001282 sec |
| 32 | 0.001177 sec | 0.001272 sec | 0.001223 sec | 0.001225 sec |

三组运行均没有输出 `Arrays do not match.`，说明 `readOffset`、`readOffsetUnroll2` 和 `readOffsetUnroll4` 的 GPU 结果均与 CPU 参考结果一致。

普通运行时间显示，三个 kernel 都在约 `1.2 ms` 左右。unroll2 和 unroll4 没有带来明显加速，说明这个样例主要受全局内存带宽和缓存行为影响，单纯减少 block 数量并不一定改善端到端时间。

## ncu 采集命令

一次运行中过滤所有名字包含 `readOffset` 的 kernel，并采集三次 launch：

```sh
ncu \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second \
  --kernel-name regex:readOffset \
  --launch-count 3 \
  --print-units base \
  --print-metric-name name \
  ./readSegmentUnroll 11 512 26
```

分别将最后一行参数替换为 `0 512 26`、`11 512 26`、`32 512 26` 采集三个 offset。

注意：`ncu` 下程序自身打印的 `elapsed` 会被 profiler 插桩和 replay 明显放大，分析 kernel 时间应看 `gpu__time_duration.sum`。

## ncu 指标记录

### offset = 0

| Kernel | ncu kernel 时间 | L1TEX global load sectors | DRAM read sectors | global load 吞吐 | DRAM 读吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `readOffset` | 0.860256 ms | 16,777,216 | 17,020,504 | 624.08 GB/s | 633.13 GB/s |
| `readOffsetUnroll2` | 0.829664 ms | 16,777,216 | 16,817,192 | 647.09 GB/s | 648.64 GB/s |
| `readOffsetUnroll4` | 0.843488 ms | 16,777,216 | 16,833,504 | 636.49 GB/s | 638.62 GB/s |

### offset = 11

| Kernel | ncu kernel 时间 | L1TEX global load sectors | DRAM read sectors | global load 吞吐 | DRAM 读吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `readOffset` | 0.875104 ms | 20,971,516 | 16,972,172 | 766.87 GB/s | 620.62 GB/s |
| `readOffsetUnroll2` | 0.830528 ms | 20,971,516 | 16,815,584 | 808.03 GB/s | 647.90 GB/s |
| `readOffsetUnroll4` | 0.858880 ms | 20,971,516 | 16,835,684 | 781.35 GB/s | 627.26 GB/s |

### offset = 32

| Kernel | ncu kernel 时间 | L1TEX global load sectors | DRAM read sectors | global load 吞吐 | DRAM 读吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `readOffset` | 0.852608 ms | 16,777,208 | 16,902,260 | 629.68 GB/s | 634.37 GB/s |
| `readOffsetUnroll2` | 0.855616 ms | 16,777,208 | 16,855,100 | 627.47 GB/s | 630.38 GB/s |
| `readOffsetUnroll4` | 0.841760 ms | 16,777,208 | 16,785,364 | 637.80 GB/s | 638.11 GB/s |

## 结果分析

`offset=11` 的 L1TEX global load sectors 稳定为 `20,971,516`，而 `offset=0` 和 `offset=32` 都约为 `16,777,216`。这个趋势和 `readSegment` 一致：`offset=11` 让 warp 的连续 `float` 读取从非对齐地址开始，导致更多 sector 请求；`offset=32` 等价于偏移 `128 bytes`，重新落回对齐边界，因此接近 `offset=0`。

unroll2 和 unroll4 没有减少 L1TEX global load sectors。原因是三个 kernel 读取的逻辑数据范围相同，offset 造成的每个 warp 访存对齐状态也相同。unroll 只是让一个线程串行处理多个元素，并减少启动的 block 数量，不会改变总的 load sector 需求。

从 `gpu__time_duration.sum` 看，unroll2 在本次采集中略快一些：

| offset | `unroll2 / readOffset` 时间比例 | `unroll4 / readOffset` 时间比例 |
| ---: | ---: | ---: |
| 0 | 0.964x | 0.981x |
| 11 | 0.949x | 0.981x |
| 32 | 1.004x | 0.987x |

这个差异很小，不能简单解释为 unroll 越多越快。当前 kernel 每个元素只有两次 load、一次 add 和一次 store，主要瓶颈仍然在内存访问。unroll2 可能减少了一部分调度和索引开销；unroll4 增加了更多边界判断和指令序列，收益不如 unroll2 稳定。

DRAM read sectors 在三个 offset 下都接近 `16.8M`，没有随 `offset=11` 增加 25%。这与 `readSegment` 的观察一致：额外 sector 更清楚地体现在 L1TEX global load 请求上，DRAM 侧会受到 L2 cache、sector 合并和 profiler replay 行为影响。

## 结论

`readSegmentUnroll` 说明，unrolling 可以减少 block 数和部分调度/循环相关开销，但不能消除非对齐读取带来的 global load sector 增加。对于这个样例，`offset=11` 的 L1TEX load sectors 相比对齐访问增加约 25%，而 unroll2/unroll4 的 sector 数与基础版本相同。

如果继续扩展实验，建议增加更多 offset，例如 `1`、`2`、`16`、`31`、`32`，并对每个 kernel 重复采样多次取中位数。这样可以更稳定地区分随机波动、cache 状态和 unrolling 对执行时间的实际影响。
