# readSegment ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Version: 13.0, CUDA Toolkit: 12.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_04/readSegment`
- `ncu` 采集时间: 2026-05-09 09:07:47 CST
- `ncu` 采集数组规模: `1 << 26` 个 `float`
- 默认 block 大小: `512`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_04
make clean
make
```

普通运行：

```sh
./readSegment 0
./readSegment 11
./readSegment 32
```

用户最初在 RTX 4090 上的普通运行结果，当时程序输出数组规模为 `1 << 20`：

| offset | grid 配置 | block 配置 | warmup 时间 | readOffset 时间 |
| ---: | --- | --- | ---: | ---: |
| 0 | `(2048, 1)` | `(512, 1)` | 0.000186 sec | 0.000010 sec |
| 11 | `(2048, 1)` | `(512, 1)` | 0.000123 sec | 0.000011 sec |
| 32 | `(2048, 1)` | `(512, 1)` | 0.000192 sec | 0.000011 sec |

程序没有输出 `Arrays do not match.`，说明 GPU 结果与 CPU 参考结果一致。

本次 `ncu` 采集时，当前 `readSegment.cu` 中的数组规模已经改为 `1 << 26`，因此普通输出中的 grid 配置变为 `(131072, 1)`：

| offset | grid 配置 | block 配置 | warmup 时间 | ncu 下 readOffset 输出时间 |
| ---: | --- | --- | ---: | ---: |
| 0 | `(131072, 1)` | `(512, 1)` | 0.006537 sec | 1.288480 sec |
| 11 | `(131072, 1)` | `(512, 1)` | 0.007776 sec | 0.678972 sec |
| 32 | `(131072, 1)` | `(512, 1)` | 0.007370 sec | 0.980625 sec |

注意：`ncu` 下程序自己打印的 `readOffset elapsed` 会被 profiler 插桩和 replay 明显放大，不应作为真实性能时间使用。分析 kernel 时间应看 `gpu__time_duration.sum`。

## 程序行为

`readSegment.cu` 用同一个 kernel 访问不同 offset 的数组位置：

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = i + offset;

if (k < n) C[i] = A[k] + B[k];
```

每个线程执行两次全局 load 和一次全局 store：

- 从 `A[k]` 读取一个 `float`
- 从 `B[k]` 读取一个 `float`
- 向 `C[i]` 写入一个 `float`

`offset` 改变的是读取地址 `A[k]` 和 `B[k]` 的起始位置，不改变写入地址 `C[i]`。因此这个样例主要观察 offset 对全局 load 合并访问的影响。

## nvprof 指标到 ncu 指标的替换

书中使用：

```sh
nvprof --metrics gld_transactions ./readSegment <offset>
```

`nvprof gld_transactions` 关注 global load 产生的内存事务数量。在 Nsight Compute 中，建议用下面两个指标替代观察：

| 目标 | ncu 指标 | 含义 |
| --- | --- | --- |
| L1TEX 侧 global load 请求扇区数 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | global load 请求的 sector 数，最接近观察 load transaction 增减 |
| DRAM 实际读 sector 数 | `dram__sectors_read.sum` | 最终落到 DRAM 的读 sector 数，受缓存命中影响 |

补充采集：

| 目标 | ncu 指标 | 含义 |
| --- | --- | --- |
| kernel 时间 | `gpu__time_duration.sum` | profiler 记录的 kernel 时间 |
| global load 吞吐 | `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` | L1TEX 侧 global load 吞吐 |
| DRAM 读吞吐 | `dram__bytes_read.sum.per_second` | DRAM 侧实际读吞吐 |

`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 统计的是请求到 L1TEX 的 global load sectors；`dram__sectors_read.sum` 统计的是 DRAM 实际读取 sectors。两者不一定相同：如果数据命中 L2 或被 profiler replay/cache 状态影响，DRAM 读 sector 可能不能直接代表指令层面的 global load 请求形态。

## ncu 采集命令

为了避免 `warmup` kernel 污染 `readOffset` 的 cache 状态，采集时直接按 kernel 名过滤 `readOffset`，并跳过同名 kernel 之前的其他 launch。

```sh
ncu \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second \
  --kernel-name regex:readOffset \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./readSegment 0
```

```sh
ncu \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second \
  --kernel-name regex:readOffset \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./readSegment 11
```

```sh
ncu \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second \
  --kernel-name regex:readOffset \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./readSegment 32
```

如果想降低 cache 对 `readOffset` 的影响，可以在采集时禁用 profiler 的 cache control：

```sh
ncu \
  --cache-control none \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second \
  --kernel-name regex:readOffset \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./readSegment 11
```

注意：`--cache-control none` 表示 profiler 不主动清理 cache，更接近普通运行时的缓存状态；默认 `--cache-control all` 会让 profiler 在 replay pass 之间控制 cache，便于指标稳定，但不等同于普通连续执行。

## 指标记录表

本次在 RTX 4090 上实际执行了上一节的 `ncu` 命令，结果如下：

| offset | ncu kernel 时间 | L1TEX global load sectors | DRAM read sectors | global load 吞吐 | DRAM 读吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.846080 ms | 16,777,216 | 16,858,928 | 634.54 GB/s | 637.63 GB/s |
| 11 | 0.853216 ms | 20,971,516 | 16,870,316 | 786.54 GB/s | 632.72 GB/s |
| 32 | 0.848352 ms | 16,777,208 | 16,856,492 | 632.84 GB/s | 635.83 GB/s |

相对 `offset=0`：

| offset | L1TEX global load sectors 比例 | ncu kernel 时间比例 |
| ---: | ---: | ---: |
| 0 | 1.000x | 1.000x |
| 11 | 1.250x | 1.008x |
| 32 | 1.000x | 1.003x |

## 结果分析

从用户最初的普通运行时间看，`offset=11` 并没有比 `offset=0` 明显更慢，`readOffset` 都在约 `10 us` 到 `11 us`。本次 `ncu` 结果则显示，`offset=11` 的 L1TEX global load sectors 从 `16,777,216` 增加到 `20,971,516`，约增加 `25.0%`。这说明非对齐 offset 确实增加了 global load 请求扇区数，只是时间差很小。

第一，`warmup` 和 `readOffset` 的访问模式完全相同：

```cpp
warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
```

当 `warmup` 用同一个 `offset` 先读了一遍 `A[k]` 和 `B[k]`，紧接着执行的 `readOffset` 很可能命中 L2 cache。因此第二个 kernel 的普通运行时间更多反映缓存命中后的执行时间，而不是从 DRAM 读取时的事务差异。

第二，用户最初普通运行时数组规模较小。`nElem = 1 << 20` 时，每个输入数组约 `4 MiB`，两个输入数组约 `8 MiB`。对 RTX 4090 来说，这个数据规模很容易受到 L2 cache、kernel launch、同步和计时粒度影响。`readOffset` 的时间只有十几微秒，单次测量差异容易被噪声覆盖。

第三，`offset=32` 对 `float` 来说不是随机偏移。`float` 为 4 bytes，`offset=32` 等价于地址偏移 `128 bytes`，通常会重新落在对齐边界上。因此 `offset=32` 往往更接近 `offset=0`，不应被理解为比 `offset=11` 更严重的非对齐访问。

## 指标趋势

本次 profiler 采集到的 `readOffset` global load sectors 与理论预期一致：

- `offset=0`：warp 内线程读取连续且对齐的 `float`，L1TEX global load sectors 为 `16,777,216`。
- `offset=11`：起始地址偏移 `44 bytes`，warp 的连续读取会跨越更多 sector，L1TEX global load sectors 增加到 `20,971,516`，约为 `offset=0` 的 `1.25x`。
- `offset=32`：起始地址偏移 `128 bytes`，重新落回对齐边界，L1TEX global load sectors 为 `16,777,208`，基本等同于 `offset=0`。

这里最关键的现象是：`offset=11` 的 transaction/sector 数明显增加，但 `gpu__time_duration.sum` 只从 `0.846080 ms` 增加到 `0.853216 ms`，约 `1.008x`。也就是说，在 RTX 4090 上，非对齐访问的额外 sector 数可以通过 `ncu` 指标观察到，但并没有等比例转化为 kernel 时间。

`dram__sectors_read.sum` 在三个 offset 下都接近 `16.86M`，没有像 L1TEX sectors 那样增加 25%。这说明额外的 global load sector 请求主要体现在 L1TEX 侧的请求形态上，最终 DRAM 侧读 sector 受 L2 cache、sector 合并、profiler replay 和内存系统行为影响，并不会简单等于 `nvprof gld_transactions` 的变化。

## 改进实验建议

为了更清楚地复现书中 `gld_transactions` 的趋势，可以做以下调整：

1. 保持当前 `nElem = 1 << 26`，减少 kernel launch 和计时噪声占比。
2. 避免 `warmup` 使用与 `readOffset` 相同的数据范围，例如 warmup 访问另一个较大的 buffer。
3. 对每个 offset 重复执行多次，记录中位数，而不是只看一次 `elapsed`。
4. 重点比较 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`，而不是只比较 `gpu__time_duration.sum`。
5. 分别测试 `offset=0`、`1`、`2`、`11`、`16`、`31`、`32`，观察哪些偏移重新落回对齐边界。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make
./readSegment 0
./readSegment 11
./readSegment 32
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment 0
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment 11
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__sectors_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,dram__bytes_read.sum.per_second --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment 32
```

## RTX 4090 上的 `-Xptxas -dlcm=cg` 结论

书中用来禁用一级缓存的编译选项：

```sh
nvcc -O2 -arch=sm_89 -Xptxas -dlcm=cg -o readSegment_cg readSegment.cu
```

在 RTX 4090 上仍然是有效选项。当前环境中 `ptxas --help` 显示：

```text
--def-load-cache (-dlcm)
        Default cache modifier on global/generic load.

--def-store-cache (-dscm)
        Default cache modifier on global/generic store.
```

因此需要注意语义边界：`-dlcm=cg` 控制的是 global/generic load 的默认 cache modifier。它不是 global store 的开关，也不能把它直接解释为“所有一级缓存都被禁用”。如果讨论 store cache modifier，`ptxas` 另有 `-dscm` 选项。

本章 `Makefile` 增加了两个显式实验目标：

```sh
cd code_samples/chapter_04
make readSegment-cache
```

等价于分别执行：

```sh
nvcc -O2 -arch=sm_89 -Xptxas -dlcm=ca -o readSegment_ca readSegment.cu
nvcc -O2 -arch=sm_89 -Xptxas -dlcm=cg -o readSegment_cg readSegment.cu
```

### SASS 对比

使用 `cuobjdump --dump-sass` 检查 `readOffset`，RTX 4090/SM 8.9 上的差异如下：

| 编译方式 | global load 指令 | global store 指令 |
| --- | --- | --- |
| 默认编译 | `LDG.E` | `STG.E` |
| `-Xptxas -dlcm=ca` | `LDG.E.STRONG.SM` | `STG.E` |
| `-Xptxas -dlcm=cg` | `LDG.E` | `STG.E` |

这说明在当前 CUDA 12.9 + RTX 4090 上，`-dlcm=ca` 会改变 load 指令的 cache modifier；`-dlcm=cg` 与默认编译在这个样例中生成的 load 指令一致；三者的 store 指令都没有因为 `-dlcm` 改变。

### 普通运行结果

数组规模为 `1 << 26`，`offset=0`：

| 程序 | 编译选项 | warmup 时间 | readOffset 时间 |
| --- | --- | ---: | ---: |
| `readSegment_default` | 默认编译 | 0.000984 sec | 0.000878 sec |
| `readSegment_ca` | `-Xptxas -dlcm=ca` | 0.000987 sec | 0.000880 sec |
| `readSegment_cg` | `-Xptxas -dlcm=cg` | 0.000995 sec | 0.000879 sec |

普通运行时间基本相同。这个样例的工作负载是连续读两个数组并连续写一个数组，时间主要受整体内存系统、L2、调度和测量噪声影响；不应只凭单次普通运行时间判断 L1 cache modifier 是否生效。

### `ncu` 指标结果

采集命令：

```sh
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment_ca 0
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment_cg 0
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:readOffset --launch-count 1 --print-units base --print-metric-name name ./readSegment_default 0
```

结果：

| 程序 | ncu kernel 时间 | L1TEX global load sectors | L1TEX global store sectors | DRAM read sectors | DRAM write sectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `readSegment_default` | 0.835232 ms | 16,777,216 | 8,388,608 | 16,852,536 | 7,401,408 |
| `readSegment_ca` | 0.837120 ms | 16,777,216 | 8,388,608 | 16,868,056 | 7,414,176 |
| `readSegment_cg` | 0.840512 ms | 16,777,216 | 8,388,608 | 16,904,332 | 7,433,848 |

三种编译方式的 L1TEX load/store sector 数相同，DRAM sectors 只有小幅波动，kernel 时间也非常接近。结合 SASS 结果，本次实验结论是：

1. RTX 4090 上仍可使用 `-Xptxas -dlcm=cg`。
2. `-dlcm=cg` 的准确含义是设置 global/generic load 的默认 cache modifier，不能用它解释 store 行为。
3. 对 `readSegment.cu` 这个连续访问样例，RTX 4090 的默认 load 指令已经与 `-dlcm=cg` 结果一致；显式使用 `-dlcm=cg` 不会带来可见性能变化。
4. 如果要演示 cache modifier 是否生效，反汇编比单次运行时间更直接；本次 `-dlcm=ca` 的 `LDG.E.STRONG.SM` 与 `-dlcm=cg` 的 `LDG.E` 就是直接证据。
