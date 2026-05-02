# sumMatrix ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_03/sumMatrix`
- 矩阵规模: `16384 x 16384`
- 编译选项: 当前 `Makefile` 使用 `nvcc -g -G -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_03
make sumMatrix
```

注意：`-G` 会生成 debug device code，Nsight Compute 会提示 debug mode。该模式便于学习和调试，但性能可能明显低于 release 编译结果。因此本报告重点比较不同 block 形状之间的相对差异。

## nvprof 指标到 ncu 指标的替换

书中命令：

```sh
nvprof --metrics achieved_occupancy ./sumMatrix <dimx> <dimy>
nvprof --metrics gld_throughput ./sumMatrix <dimx> <dimy>
```

在 Nsight Compute CLI 中可以改为：

```sh
ncu \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name regex:sumMatrixOnGPU2D \
  --launch-count 1 \
  ./sumMatrix <dimx> <dimy>
```

```sh
ncu \
  --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
  --kernel-name regex:sumMatrixOnGPU2D \
  --launch-count 1 \
  ./sumMatrix <dimx> <dimy>
```

本报告实际同时采集了以下指标：

- `gpu__time_duration.sum`: `ncu` 记录的 kernel 执行时间，单位为 ns。
- `sm__warps_active.avg.pct_of_peak_sustained_active`: achieved occupancy，表示实际活跃 warp 相对峰值活跃 warp 的比例。
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`: L1TEX 侧全局 load 请求吞吐量，可作为 `nvprof gld_throughput` 的直接替代观察项。
- `dram__bytes_read.sum.per_second`: 实际 DRAM 读吞吐量，用于辅助判断全局 load 最终落到显存的带宽情况。

采集命令模板：

```sh
ncu \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second,gpu__time_duration.sum \
  --kernel-name regex:sumMatrixOnGPU2D \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./sumMatrix <dimx> <dimy>
```

`ncu` 下程序打印的 `elapsed` 会显著变大，因为 profiler 会插桩和 replay。分析运行时间时应优先看普通运行结果；分析硬件指标时看 `ncu` 表格。

## 普通运行结果

以下时间来自直接运行 `./sumMatrix dimx dimy`。程序没有输出 `Arrays do not match.`，说明这Developer: Reload Window些配置下 GPU 结果均与 CPU 参考结果一致。

| block 配置 | grid 配置 | 每 block 线程数 | 普通运行时间 |
| --- | --- | ---: | ---: |
| `(32, 32)` | `(512, 512)` | 1024 | 4.25005 ms |
| `(32, 16)` | `(512, 1024)` | 512 | 3.66497 ms |
| `(16, 16)` | `(1024, 1024)` | 256 | 3.65400 ms |
| `(16, 32)` | `(1024, 512)` | 512 | 4.01902 ms |
| `(64, 16)` | `(256, 1024)` | 1024 | 4.80795 ms |
| `(16, 64)` | `(1024, 256)` | 1024 | 4.60100 ms |
| `(64, 8)` | `(256, 2048)` | 512 | 3.67904 ms |
| `(8, 64)` | `(2048, 256)` | 512 | 3.82304 ms |
| `(32, 8)` | `(512, 2048)` | 256 | 3.67808 ms |
| `(8, 32)` | `(2048, 512)` | 256 | 3.75795 ms |
| `(128, 8)` | `(128, 2048)` | 1024 | 4.31418 ms |
| `(8, 128)` | `(2048, 128)` | 1024 | 4.94099 ms |

## ncu 指标结果

吞吐量按十进制换算为 GB/s。`ncu` 时间来自 `gpu__time_duration.sum`，与普通运行时间不是同一种测量方式，只适合在相同 profiler 配置下横向比较。

| block 配置 | ncu kernel 时间 | achieved occupancy | global load 吞吐 | DRAM 读吞吐 |
| --- | ---: | ---: | ---: | ---: |
| `(32, 32)` | 4.479 ms | 58.10% | 479.41 GB/s | 483.06 GB/s |
| `(32, 16)` | 3.570 ms | 82.03% | 601.53 GB/s | 605.40 GB/s |
| `(16, 16)` | 3.583 ms | 86.39% | 599.30 GB/s | 602.67 GB/s |
| `(16, 32)` | 3.630 ms | 83.67% | 591.66 GB/s | 595.21 GB/s |
| `(64, 16)` | 4.393 ms | 59.58% | 488.88 GB/s | 493.78 GB/s |
| `(16, 64)` | 4.735 ms | 57.93% | 453.50 GB/s | 456.79 GB/s |
| `(64, 8)` | 3.558 ms | 82.94% | 603.64 GB/s | 606.99 GB/s |
| `(8, 64)` | 3.752 ms | 84.22% | 572.36 GB/s | 576.20 GB/s |
| `(32, 8)` | 3.543 ms | 83.74% | 606.11 GB/s | 609.29 GB/s |
| `(8, 32)` | 3.674 ms | 87.14% | 584.54 GB/s | 588.24 GB/s |
| `(128, 8)` | 4.368 ms | 60.35% | 491.69 GB/s | 494.93 GB/s |
| `(8, 128)` | 5.004 ms | 58.02% | 429.16 GB/s | 432.95 GB/s |

## 结果分析

这个 kernel 每个元素执行两次全局 load 和一次全局 store，算术只有一次浮点加法，属于明显的内存带宽型 kernel。不同配置的主要差别不在计算量，而在线程组织如何影响 SM 上的活跃 warp 数量、warp 内访存形态，以及 block 调度粒度。

`(32, 16)`、`(16, 16)`、`(64, 8)`、`(32, 8)` 是表现较好的配置。它们的 achieved occupancy 都在 82% 到 86% 左右，global load 吞吐接近 600 GB/s，普通运行时间也集中在 3.65 ms 到 3.68 ms 附近。这说明这些配置能让 SM 保持足够多的活跃 warp，同时全局读取也比较高效。

`(32, 32)`、`(64, 16)`、`(16, 64)`、`(128, 8)`、`(8, 128)` 每个 block 都是 1024 个线程，但 occupancy 只有约 58% 到 60%。原因是 RTX 4090 每个 block 最多 1024 线程，而每个 SM 最多 1536 线程。一个 1024 线程 block 已经占用了大量线程名额，单个 SM 通常只能同时驻留一个这样的 block，剩余的 512 个线程容量无法再放入第二个 1024 线程 block，因此实际活跃度受限。这个现象正好解释了为什么“最大 block 线程数”不等于“最高性能”。

`dimx` 对访存合并很重要。矩阵按行优先存储，连续的 `ix` 对应连续内存地址。CUDA warp 内线程的线性编号先沿 `threadIdx.x` 增长，再沿 `threadIdx.y` 增长，因此较大的 `blockDim.x` 更容易让同一个 warp 覆盖连续列，形成更自然的合并访问。`(32, 8)` 和 `(64, 8)` 都表现很好；相反，`(8, 128)` 的 `blockDim.x` 太小，一个 warp 会跨多行访问，每 8 个线程换一行，访存合并形态更差，再叠加 1024 线程 block 的低 occupancy，最终 global load 吞吐最低，普通运行时间也最慢。

`(16, 16)` 虽然 `blockDim.x` 只有 16，但每个 block 只有 256 个线程，调度粒度更灵活，SM 上可以同时驻留更多 block，因此 achieved occupancy 最高。它的普通运行时间与 `(32, 16)` 非常接近，说明在这个样例中，较高活跃度可以弥补一部分访存形态上的差异。

`(16, 32)` 和用户提供的普通运行结果中略慢于 `(32, 16)`。两者都是 512 线程 block，但 `(32, 16)` 的 x 维正好是一整个 warp 的宽度，warp 更倾向于覆盖同一行中的连续 32 个元素；`(16, 32)` 中一个 warp 通常覆盖两行各 16 个元素，访存请求会被拆成更多片段，因此吞吐略低。

总体结论：这个矩阵加法 kernel 更推荐从 `(32, 8)`、`(32, 16)`、`(64, 8)`、`(16, 16)` 这类配置开始调参。避免默认认为 1024 线程 block 一定更快；在本测试中，1024 线程 block 往往因为驻留 block 数受限而降低 achieved occupancy，并不占优。

## 可复现实验命令

普通运行：

```sh
./sumMatrix 32 32
./sumMatrix 32 16
./sumMatrix 16 16
./sumMatrix 16 32
./sumMatrix 64 16
./sumMatrix 16 64
./sumMatrix 64 8
./sumMatrix 8 64
./sumMatrix 32 8
./sumMatrix 8 32
./sumMatrix 128 8
./sumMatrix 8 128
```

Nsight Compute 指标采集示例：

```sh
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second,gpu__time_duration.sum --kernel-name regex:sumMatrixOnGPU2D --launch-count 1 --print-units base --print-metric-name name ./sumMatrix 32 16
```
