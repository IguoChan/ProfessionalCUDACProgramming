# Chapter 02 Summary

## 本章定位

第二章开始系统学习 CUDA 编程模型。当前目录中的示例围绕设备查询、grid/block/thread 层次结构、线程索引计算、主机和设备内存拷贝、向量加法、矩阵加法以及不同执行配置展开。

## 示例脉络

- `checkDeviceInfor.cu`
  - 查询 CUDA 设备数量、驱动和运行时版本、计算能力、全局内存、共享内存、寄存器数量、warp size、最大线程数和 grid/block 维度限制。

- `defineGridBlock.cu`
  - 用 `dim3` 定义 block 和 grid。
  - 展示常见的向上取整公式：`(nElem + block.x - 1) / block.x`。

- `checkDimension.cu`
  - 在主机端和设备端分别打印 `gridDim`、`blockDim`、`blockIdx`、`threadIdx`。
  - 帮助理解 CUDA 的内建变量。

- `checkThreadIndex.cu`
  - 使用二维 block 和二维 grid 访问矩阵。
  - 将 `(threadIdx, blockIdx, blockDim)` 映射为全局坐标 `(ix, iy)` 和一维数组下标 `idx = iy * nx + ix`。

- `sumArraysOnHost.c`
  - CPU 端顺序向量加法，作为后续 GPU 版本的对照。

- `sumArraysOnGPU-small-case.cu`
  - 最小规模 GPU 向量加法。
  - 覆盖 `cudaMalloc`、`cudaMemcpy`、kernel 启动、结果拷回和 CPU/GPU 结果校验。

- `sumArraysOnGPU-timer.cu`
  - 在向量加法基础上加入计时。
  - 展示 kernel 异步启动时需要用 `cudaDeviceSynchronize()` 包住计时区间。

- `sumMatrixOnGPU-1D-grid-1D-block.cu`
  - 用一维 grid 和一维 block 处理二维矩阵。
  - 每个线程负责一个列索引，并在 kernel 内循环遍历多行。

- `sumMatrixOnGPU-2D-grid-1D-block.cu`
  - 用二维 grid 和一维 block 处理矩阵。
  - `blockIdx.y` 映射矩阵行，`threadIdx.x + blockIdx.x * blockDim.x` 映射列。

- `sumMatrixOnGPU-2D-grid-2D-block.cu`
  - 用二维 grid 和二维 block 直接映射矩阵二维坐标。
  - 是矩阵类问题中更自然的线程组织方式。

## 核心知识点

1. CUDA 执行层次结构。
   - grid 由多个 block 组成。
   - block 由多个 thread 组成。
   - thread 通过 `threadIdx`、`blockIdx`、`blockDim` 和 `gridDim` 计算自己负责的数据位置。

2. `dim3` 可以表达一维、二维、三维执行配置。
   - 未显式设置的维度默认为 1。
   - 对矩阵问题，二维 grid/block 通常更贴合数据形状。

3. 线程索引必须配合边界检查。
   - grid 常用向上取整覆盖全部数据。
   - 最后一个 block 可能包含越界线程，因此 kernel 中需要 `if (idx < N)` 或 `if (ix < nx && iy < ny)`。

4. 主机和设备内存是分开的。
   - 主机端使用 `malloc/free`。
   - 设备端使用 `cudaMalloc/cudaFree`。
   - 数据通过 `cudaMemcpy` 在 Host 和 Device 之间传输。

5. 正确性验证是 CUDA 示例的基本闭环。
   - 先在 CPU 上计算参考结果。
   - 再运行 GPU kernel。
   - 最后把 GPU 结果拷回并逐元素比较。

6. 计时需要考虑 CUDA 异步执行。
   - kernel launch 本身通常很快返回。
   - 计量 kernel 执行时间时必须同步，否则测到的主要是启动开销。

## 运行方式

```sh
cd code_samples/chapter_02
make
./checkDeviceInfor
./checkThreadIndex
./sumArraysOnGPU-timer
./sumMatrixOnGPU-2D-grid-2D-block
```

## 学习收获

本章的关键是把“数据下标”和“线程层次”建立稳定对应关系。向量加法训练一维索引，矩阵加法训练二维索引；掌握这些之后，后续性能优化和复杂 kernel 都是在这个基础上改进线程组织、访存模式和同步方式。
