# writeSegment ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Toolkit: 12.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_04/writeSegment`
- 数组规模: `1 << 20` 个 `float`
- 默认 block 大小: `512`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译和运行：

```sh
cd code_samples/chapter_04
make writeSegment
./writeSegment 0
./writeSegment 11
./writeSegment 32
```

## 程序行为

`writeSegment.cu` 观察不同 offset 对全局写访问的影响：

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = i + offset;

if (k < n) C[k] = A[i] + B[i];
```

每个线程执行两次连续 global load，并向带 offset 的位置执行一次 global store。和 `readSegment.cu` 相反，这里 offset 改变的是写入地址 `C[k]`，读取地址 `A[i]` 和 `B[i]` 保持连续对齐。

程序还包含 `writeOffsetUnroll2` 和 `writeOffsetUnroll4`，用于观察减少 block 数后每个线程写多个位置的表现。unroll kernel 的索引使用 `blockIdx.x * blockDim.x * unroll`，并且每个 unroll 写入都有独立边界判断，避免尾部 offset 元素被错误跳过。

## 普通运行结果

本次普通运行没有输出 `Arrays do not match.`，说明 GPU 结果与 CPU 参考结果一致。

| offset | writeOffset 时间 | unroll2 时间 | unroll4 时间 |
| ---: | ---: | ---: | ---: |
| 0 | 0.000010 sec | 0.000078 sec | 0.000076 sec |
| 11 | 0.000012 sec | 0.000106 sec | 0.000077 sec |
| 32 | 0.000160 sec | 0.000076 sec | 0.000018 sec |

单次普通运行时间很短，容易受到 launch、同步和缓存状态影响；判断非对齐写访问的事务变化应优先看 `ncu` 的 L1TEX store sector 指标。

## ncu 采集命令

```sh
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 0
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 11
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 32
```

## ncu 结果

| offset | ncu kernel 时间 | L1TEX global load sectors | L1TEX global store sectors | DRAM read sectors | DRAM write sectors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.011008 ms | 262,144 | 131,072 | 263,500 | 0 |
| 11 | 0.011136 ms | 262,142 | 163,838 | 263,452 | 0 |
| 32 | 0.030656 ms | 262,136 | 131,068 | 265,908 | 0 |

相对 `offset=0`：

| offset | L1TEX global store sectors 比例 | ncu kernel 时间比例 |
| ---: | ---: | ---: |
| 0 | 1.000x | 1.000x |
| 11 | 1.250x | 1.012x |
| 32 | 1.000x | 2.785x |

## 结果分析

`offset=11` 把写入地址偏移了 `44 bytes`，warp 内连续写入会跨越更多 sector，因此 L1TEX global store sectors 从 `131,072` 增加到 `163,838`，约为 `1.25x`。这与 `readSegment` 中非对齐读访问增加 load sectors 的趋势一致，只是这里变化体现在 store sectors。

`offset=32` 对 `float` 来说是 `128 bytes` 偏移，通常重新落回对齐边界，因此 store sectors 基本回到 `offset=0` 水平。

本次 `dram__sectors_write.sum` 为 `0`，不代表 kernel 没有写入，而是说明该指标在这个短 kernel 的采集窗口内没有直接反映最终写回行为。对这个样例，观察写访问合并形态应以 `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` 为主。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make writeSegment
./writeSegment 0
./writeSegment 11
./writeSegment 32
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 0
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 11
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum --kernel-name regex:writeOffset --launch-count 1 --print-units base --print-metric-name name ./writeSegment 32
```
