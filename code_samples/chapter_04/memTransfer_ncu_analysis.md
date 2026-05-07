# memTransfer ncu 分析

## 测试环境

- 测试时间: 2026-05-07 10:05:34 CST
- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Toolkit: 12.9, V12.9.41
- `ncu`: Nsight Compute 2025.2.0.0
- `nsys`: Nsight Systems 2025.1.3
- 程序路径: `code_samples/chapter_04/memTransfer`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译命令：

```sh
cd code_samples/chapter_04
make clean
make
```

普通运行：

```sh
./memTransfer
```

输出：

```text
./memTransfer starting at device 0: NVIDIA GeForce RTX 4090 memory size 4194304 nbyte 16.00MB
```

## 程序行为

`memTransfer.cu` 没有启动 CUDA kernel。它只执行以下 CUDA Runtime API：

- `cudaSetDevice`
- `cudaGetDeviceProperties`
- `cudaMalloc`
- `cudaMemcpy(..., cudaMemcpyHostToDevice)`
- `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
- `cudaFree`
- `cudaDeviceReset`

每次 `cudaMemcpy` 传输 `1 << 22` 个 `float`，也就是 `16,777,216` bytes，约 `16.00 MiB` 或 `16.777 MB`。

## nvprof 与 ncu 的对应关系

书中通常用 `nvprof ./memTransfer` 查看 CUDA API 调用和 GPU memcpy 时间。这个样例没有 kernel，所以不能直接用 Nsight Compute 取得类似 `nvprof` 的 memcpy 时间表。

实际执行的 `ncu` 命令：

```sh
ncu --target-processes all --print-units base --print-metric-name name ./memTransfer
```

`ncu` 输出：

```text
==PROF== Connected to process .../memTransfer
./memTransfer starting at device 0: NVIDIA GeForce RTX 4090 memory size 4194304 nbyte 16.00MB
==PROF== Disconnected from process .../memTransfer
==WARNING== No kernels were profiled.
```

结论：`ncu` 是 kernel 级 profiler。对 `memTransfer` 这种只包含 `cudaMemcpy`、不包含 kernel launch 的程序，`ncu` 可以启动并附加到进程，但没有可采集的 kernel 指标，因此不会给出 memcpy 的时间统计。

## 实际替代工具

如果目标是复现书中 `nvprof` 对 memcpy 的统计，当前 NVIDIA 工具链里更合适的是 Nsight Systems CLI：

```sh
nsys profile --trace=cuda --stats=true --force-overwrite=true -o memTransfer_nsys ./memTransfer
```

本次采集时，`nsys` 提示 CPU IP/backtrace sampling 和 CPU context switch tracing 在当前环境不可用，但 CUDA trace 和 memcpy 统计正常生成。

## CUDA API 统计

来自 `nsys` 的 `cuda_api_sum`：

| CUDA API | 调用次数 | 总时间 | 平均时间 | 占 API 时间比例 |
| --- | ---: | ---: | ---: | ---: |
| `cudaDeviceReset` | 1 | 45.307 ms | 45.307 ms | 93.4% |
| `cudaMemcpy` | 2 | 2.232 ms | 1.116 ms | 4.6% |
| `cudaGetDeviceProperties` | 1 | 0.847 ms | 0.847 ms | 1.7% |
| `cudaMalloc` | 1 | 0.064 ms | 0.064 ms | 0.1% |
| `cudaFree` | 1 | 0.053 ms | 0.053 ms | 0.1% |

`cudaDeviceReset` 占比较高，是因为它会销毁当前进程的 CUDA context 并执行清理工作。它不是数据传输成本。

## GPU memcpy 统计

来自 `nsys` 的 `cuda_gpu_mem_time_sum` 和 `cuda_gpu_mem_size_sum`：

| 方向 | 次数 | 传输大小 | GPU memcpy 时间 | 估算带宽 |
| --- | ---: | ---: | ---: | ---: |
| Host-to-Device | 1 | 16.777 MB | 0.986 ms | 17.01 GB/s |
| Device-to-Host | 1 | 16.777 MB | 1.093 ms | 15.35 GB/s |

带宽按十进制计算：

```text
bandwidth = bytes / seconds / 1e9
```

## 结果分析

这个样例的核心开销是两次同步 `cudaMemcpy`。在本次测试中，Host-to-Device 比 Device-to-Host 略快，分别约为 `17.01 GB/s` 和 `15.35 GB/s`。两次传输大小相同，因此时间差直接反映了方向上的有效带宽差异。

由于主机内存使用普通 `malloc` 分配，源/目标 host buffer 是 pageable memory。CUDA Runtime 在同步 `cudaMemcpy` 时通常需要经过内部 staging 过程，实际带宽会低于 page-locked host memory 的理想传输表现。若要进一步测试 PCIe 传输上限，可以新增使用 `cudaMallocHost` 或 `cudaHostAlloc` 的版本，再与当前 pageable memory 版本对比。

总体结论：对 `memTransfer` 这个无 kernel 的样例，`ncu` 的可观测结果是“没有 kernel 可 profile”。如果目的是替代书中的 `nvprof` memcpy/API 时间表，应使用 `nsys --trace=cuda --stats=true`；如果后续样例包含 kernel，再用 `ncu` 采集 occupancy、memory throughput、SM utilization 等 kernel 指标。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make
./memTransfer
ncu --target-processes all --print-units base --print-metric-name name ./memTransfer
nsys profile --trace=cuda --stats=true --force-overwrite=true -o memTransfer_nsys ./memTransfer
```

