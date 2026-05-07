# sumArrayZerocpy ncu 分析

## 测试环境

- 测试时间: 2026-05-07 17:20:00 CST
- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Toolkit: 12.9, V12.9.41
- `ncu`: Nsight Compute 2025.2.0.0
- `nsys`: Nsight Systems 2025.1.3
- 程序路径: `code_samples/chapter_04/sumArrayZerocpy`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_04
make clean
make
```

普通运行：

```sh
./sumArrayZerocpy
./sumArrayZerocpy 22
```

输出：

```text
Using Device 0: NVIDIA GeForce RTX 4090 Vector size 1024 power 10  nbytes    4 KB
Using Device 0: NVIDIA GeForce RTX 4090 Vector size 4194304 power 22  nbytes   16 MB
```

程序没有输出 `Arrays do not match!`，说明普通 device memory 路径和 zero-copy 路径的结果都与 CPU 参考结果一致。

## 程序行为

这个样例对比两种向量加法路径：

1. 普通 device memory 路径
   - host 使用 `malloc` 分配 `h_A`、`h_B`
   - device 使用 `cudaMalloc` 分配 `d_A`、`d_B`、`d_C`
   - 使用两次 `cudaMemcpyHostToDevice` 把 `A`、`B` 拷到 GPU
   - 启动 `sumArrays`
   - 使用一次 `cudaMemcpyDeviceToHost` 拷回 `C`

2. zero-copy 路径
   - host 使用 `cudaHostAlloc(..., cudaHostAllocMapped)` 分配 page-locked mapped memory
   - 使用 `cudaHostGetDevicePointer` 得到 GPU 可访问的 host memory 地址
   - `sumArraysZeroCopy` 直接读取映射到 GPU 地址空间的 host memory
   - 只把输出 `C` 从 device memory 拷回 host

本次 profiling 使用参数 `22`，即每个数组 `1 << 22` 个 `float`，单个数组大小为 `16,777,216` bytes，约 `16.00 MiB` 或 `16.777 MB`。

## ncu 指标采集

普通 device memory kernel：

```sh
ncu \
  --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second \
  --kernel-name regex:sumArrays \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./sumArrayZerocpy 22
```

Zero-copy kernel：

```sh
ncu \
  --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second \
  --kernel-name regex:sumArraysZeroCopy \
  --launch-count 1 \
  --print-units base \
  --print-metric-name name \
  ./sumArrayZerocpy 22
```

## ncu 结果

吞吐量按十进制换算为 GB/s。`ncu` 时间来自 `gpu__time_duration.sum`。

| Kernel | 输入位置 | ncu kernel 时间 | achieved occupancy | global load 吞吐 | global store 吞吐 | DRAM 读吞吐 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sumArrays` | device memory | 0.038 ms | 84.75% | 883.38 GB/s | 441.69 GB/s | 886.86 GB/s |
| `sumArraysZeroCopy` | mapped host memory | 1.584 ms | 72.30% | 21.19 GB/s | 10.59 GB/s | 2.79 GB/s |

按 `ncu` 的 kernel 时间看，zero-copy kernel 比普通 device memory kernel 慢约 `41.7x`：

```text
1.583520 ms / 0.037984 ms = 41.7
```

## Nsight Systems 采集

采集命令：

```sh
nsys profile --trace=cuda --stats=true --force-overwrite=true -o sumArrayZerocpy_nsys ./sumArrayZerocpy 22
```

本次采集时，`nsys` 提示 CPU IP/backtrace sampling 和 CPU context switch tracing 在当前环境不可用，但 CUDA trace、kernel 统计和 memcpy 统计正常生成。

## nsys kernel 统计

来自 `nsys` 的 `cuda_gpu_kern_sum`：

| Kernel | 次数 | GPU 时间 |
| --- | ---: | ---: |
| `sumArrays` | 1 | 0.011 ms |
| `sumArraysZeroCopy` | 1 | 1.534 ms |

`nsys` 与 `ncu` 的绝对时间不同，因为 `ncu` 会插桩并可能 replay kernel。两者都显示同一个趋势：zero-copy kernel 明显慢于普通 device memory kernel。

## nsys memcpy 统计

来自 `nsys` 的 `cuda_gpu_mem_time_sum` 和 `cuda_gpu_mem_size_sum`：

| 方向 | 次数 | 总传输大小 | 总时间 | 平均时间 |
| --- | ---: | ---: | ---: | ---: |
| Host-to-Device | 2 | 33.554 MB | 2.108 ms | 1.054 ms |
| Device-to-Host | 2 | 33.554 MB | 2.196 ms | 1.098 ms |

这里的 Host-to-Device 两次拷贝来自普通 device memory 路径中的 `A` 和 `B`。Zero-copy 路径没有为 `A` 和 `B` 执行显式 Host-to-Device 拷贝。Device-to-Host 有两次，是因为普通路径和 zero-copy 路径都把输出 `C` 拷回 host 以检查结果。

## 结果分析

这个样例想说明的核心区别不是 pinned memory 本身，而是 zero-copy mapped memory 的访问模型。普通路径先把输入数组复制到 device global memory，kernel 后续从 GPU 显存读取，访存吞吐高，kernel 很快。Zero-copy 路径省掉了输入数组的显式 memcpy，但 kernel 每次读取 `A` 和 `B` 都要通过映射地址访问 host memory，实际数据仍然跨 PCIe 传输，因此 kernel 访存吞吐明显低。

对一次性、少量数据，zero-copy 可以减少代码里的显式拷贝，也可能减少端到端延迟。对本例这种每个元素都要从输入数组读取两次并做大规模顺序处理的带宽型 kernel，把数据先拷到 device memory 再计算更快。

`sumArraysZeroCopy` 的 global load 吞吐只有约 `21.19 GB/s`，接近主机和 GPU 之间互连带宽级别；`sumArrays` 从 device memory 读取，global load 吞吐约 `883.38 GB/s`。这正好体现了两者的瓶颈不同：前者受 PCIe/host memory 访问限制，后者主要受 GPU device memory 带宽限制。

总体结论：zero-copy 的优势是省掉显式 Host-to-Device 拷贝和一份 device 输入内存；代价是 kernel 直接读 host memory 时带宽和延迟都远不如读 device global memory。它适合小数据、一次性访问、集成 GPU 或对拷贝开销更敏感的场景，不适合本例这种大数组、高吞吐、重复顺序读取的计算。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make
./sumArrayZerocpy
./sumArrayZerocpy 22
ncu --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second --kernel-name regex:sumArrays --launch-count 1 --print-units base --print-metric-name name ./sumArrayZerocpy 22
ncu --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second --kernel-name regex:sumArraysZeroCopy --launch-count 1 --print-units base --print-metric-name name ./sumArrayZerocpy 22
nsys profile --trace=cuda --stats=true --force-overwrite=true -o sumArrayZerocpy_nsys ./sumArrayZerocpy 22
```
