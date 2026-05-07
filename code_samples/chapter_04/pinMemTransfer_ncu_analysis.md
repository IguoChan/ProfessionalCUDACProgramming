# pinMemTransfer ncu 分析

## 测试环境

- 测试时间: 2026-05-07 16:53:28 CST
- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Toolkit: 12.9, V12.9.41
- `ncu`: Nsight Compute 2025.2.0.0
- `nsys`: Nsight Systems 2025.1.3
- 程序路径: `code_samples/chapter_04/pinMemTransfer`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_04
make clean
make
```

普通运行：

```sh
./pinMemTransfer
```

输出：

```text
./pinMemTransfer starting at device 0: NVIDIA GeForce RTX 4090 memory size 4194304 nbyte 16.00MB canMap 1
```

`canMap 1` 表示当前设备支持映射 host memory。本程序实际使用 `cudaMallocHost` 分配 page-locked host memory，然后执行一次 Host-to-Device 和一次 Device-to-Host 的同步 `cudaMemcpy`。

## 程序行为

`pinMemTransfer.cu` 没有启动 CUDA kernel。它主要执行以下 CUDA Runtime API：

- `cudaSetDevice`
- `cudaGetDeviceProperties`
- `cudaMallocHost`
- `cudaMalloc`
- `cudaMemcpy(..., cudaMemcpyHostToDevice)`
- `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
- `cudaFree`
- `cudaFreeHost`
- `cudaDeviceReset`

每次 `cudaMemcpy` 传输 `1 << 22` 个 `float`，也就是 `16,777,216` bytes，约 `16.00 MiB` 或 `16.777 MB`。

## ncu 结果

实际执行的 `ncu` 命令：

```sh
ncu --target-processes all --print-units base --print-metric-name name ./pinMemTransfer
```

`ncu` 输出：

```text
==PROF== Connected to process .../pinMemTransfer
./pinMemTransfer starting at device 0: NVIDIA GeForce RTX 4090 memory size 4194304 nbyte 16.00MB canMap 1
==PROF== Disconnected from process .../pinMemTransfer
==WARNING== No kernels were profiled.
```

结论与 `memTransfer` 相同：这个样例没有 kernel launch，`ncu` 可以附加进程，但没有 kernel 级指标可采集。若目标是查看 memcpy/API 时间，应使用 Nsight Systems。

## Nsight Systems 采集

采集命令：

```sh
nsys profile --trace=cuda --stats=true --force-overwrite=true -o pinMemTransfer_nsys ./pinMemTransfer
```

本次采集时，`nsys` 提示 CPU IP/backtrace sampling 和 CPU context switch tracing 在当前环境不可用，但 CUDA trace 和 memcpy 统计正常生成。

## CUDA API 统计

来自 `nsys` 的 `cuda_api_sum`：

| CUDA API | 调用次数 | 总时间 | 平均时间 | 占 API 时间比例 |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceReset` | 1 | 43.978 ms | 43.978 ms | 86.4% |
| `cudaHostAlloc` | 1 | 2.863 ms | 2.863 ms | 5.6% |
| `cudaMemcpy` | 2 | 1.968 ms | 0.984 ms | 3.9% |
| `cudaFreeHost` | 1 | 1.118 ms | 1.118 ms | 2.2% |
| `cudaGetDeviceProperties` | 1 | 0.867 ms | 0.867 ms | 1.7% |
| `cudaFree` | 1 | 0.055 ms | 0.055 ms | 0.1% |
| `cudaMalloc` | 1 | 0.052 ms | 0.052 ms | 0.1% |

`cudaHostAlloc` 和 `cudaFreeHost` 比普通 `malloc/free` 更重，因为它们需要向 CUDA Runtime 注册或解除注册 page-locked host memory。这部分是分配成本，不是单次 memcpy 的设备传输时间。

## GPU memcpy 统计

来自 `nsys` 的 `cuda_gpu_mem_time_sum` 和 `cuda_gpu_mem_size_sum`：

| 方向 | 次数 | 传输大小 | GPU memcpy 时间 | 估算带宽 |
| --- | ---: | ---: | ---: | ---: |
| Host-to-Device | 1 | 16.777 MB | 1.048 ms | 16.00 GB/s |
| Device-to-Host | 1 | 16.777 MB | 0.893 ms | 18.79 GB/s |

带宽按十进制计算：

```text
bandwidth = bytes / seconds / 1e9
```

## 与 memTransfer 对比

同一环境下，上一份 `memTransfer_ncu_analysis.md` 中的 pageable memory 结果如下：

| 程序 | Host memory 类型 | HtoD 时间 | HtoD 带宽 | DtoH 时间 | DtoH 带宽 |
| --- | --- | ---: | ---: | ---: | ---: |
| `memTransfer` | pageable, `malloc` | 0.986 ms | 17.01 GB/s | 1.093 ms | 15.35 GB/s |
| `pinMemTransfer` | pinned, `cudaMallocHost` | 1.048 ms | 16.00 GB/s | 0.893 ms | 18.79 GB/s |

本次单次采样中，pinned memory 的 Device-to-Host 更快，Host-to-Device 略慢。由于每个方向只传输一次 16 MiB，结果会受上下文状态、PCIe 链路状态、profiler 开销和系统负载影响。更严谨的比较应循环多次传输、跳过 warm-up，并统计平均值或中位数。

## 结果分析

`pinMemTransfer` 演示的是 page-locked host memory 对数据传输路径的影响。Pinned host memory 不需要在传输前临时分页锁定或额外 staging，通常更适合高吞吐、重复性的 Host 和 Device 数据传输，尤其是与 `cudaMemcpyAsync` 和 stream 配合时。

需要注意的是，pinned memory 的申请和释放成本明显高于普通 host memory。本次 API 统计中，`cudaHostAlloc` 约 `2.863 ms`，`cudaFreeHost` 约 `1.118 ms`。因此 pinned memory 更适合长期复用的 buffer，不适合在高频路径中反复申请和释放。

总体结论：对这个无 kernel 的样例，`ncu` 仍然只能确认没有 kernel 可 profile；实际 memcpy 统计应看 `nsys`。本次结果显示 pinned memory 改善了 Device-to-Host 方向的有效带宽，但单次 16 MiB 测量不足以代表稳定上限。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make
./pinMemTransfer
ncu --target-processes all --print-units base --print-metric-name name ./pinMemTransfer
nsys profile --trace=cuda --stats=true --force-overwrite=true -o pinMemTransfer_nsys ./pinMemTransfer
```
