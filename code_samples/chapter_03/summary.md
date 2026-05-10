# Chapter 03 Summary

## 本章定位

第三章进入 CUDA 执行模型和性能优化。当前目录的示例重点覆盖设备资源查询、矩阵加法执行配置、线程束分化、并行归约优化、warp 级优化、模板展开、动态并行以及 Nsight Compute 指标分析。

## 示例脉络

- `simpleDeviceQuery.cu`
  - 查询 SM 数量、常量内存、共享内存、寄存器数量、warp size、每个 block 和每个 SM 的线程上限。
  - 这些信息决定了 block size、占用率和资源约束的基本边界。

- `sumMatrix.cu`
  - 用二维 grid 和二维 block 执行矩阵加法。
  - 支持通过命令行传入 `dimx dimy`，便于比较不同 block 形状对 kernel 时间的影响。
  - 对应 `sumMatrix_ncu_analysis.md` 中的 profiling 思路。

- `simpleDivergence.cu`
  - 对比同一 warp 内分支条件不同和 warp 粒度分支一致的情况。
  - `mathKernel1_divergent` 使用 `tid % 2`，同一 warp 中奇偶线程走不同路径，会产生线程束分化。
  - `mathKernel2_uniform` 使用 `tid / warpSize`，同一 warp 内分支一致，避免 warp 内分化。
  - `mathKernel3` 说明把条件保存为谓词变量并不会自动消除分化。
  - `mathKernel4` 使用 warp 编号构造一致分支，并修正了位运算优先级问题。

- `reduceInteger.cu`
  - 系统比较多种整数数组并行归约策略。
  - 从相邻配对、改进相邻配对、交错配对逐步过渡到 unrolling、warp 展开、shuffle 和模板完全展开。
  - 每个 GPU kernel 都与 CPU 递归规约结果对照，保证优化前后结果一致。

- `nestedHelloWorld.cu` 和 `nestedHelloWorldNew.cu`
  - 演示 CUDA Dynamic Parallelism，也就是在设备端 kernel 中继续启动 child kernel。
  - 新版本支持更灵活的 grid/block 配置，并使用边界检查控制递归规模。

- `nestedReduce.cu`
  - 用动态并行实现递归归约。
  - 每个 block 各自启动 child kernel，并使用设备端 `cudaDeviceSynchronize()` 等待子 kernel。
  - 展示动态并行的表达能力，同时暴露 child launch 和设备端同步开销。

- `nestedReduceNosync.cu`
  - 去掉设备端同步，减少部分开销。
  - 代码运行可能得到正确结果，但缺少必要同步时存在读写时序风险。

- `nestedReduce2.cu`
  - 只由一个线程启动下一层 child grid，大幅减少 child kernel launch 数量。
  - 相比前两个 nested reduction 版本更快，但仍不能替代常规高性能 reduction 模板。

## 核心知识点

1. GPU 执行的基本单位是 warp。
   - 一个 warp 通常包含 32 个线程。
   - 同一 warp 内如果线程走不同分支，硬件会用执行掩码串行化不同路径。
   - 分支本身不是问题，关键是分支条件在 warp 内是否一致。

2. block 配置会影响性能。
   - `sumMatrix.cu` 可以用不同 `dimx dimy` 实验 block 形状。
   - 合理配置需要同时考虑线程数上限、warp 对齐、访存合并、寄存器和共享内存压力。

3. 并行归约的优化路径很典型。
   - 相邻配对写法直观，但容易产生分支分化。
   - 交错配对让活跃线程连续，通常更高效。
   - unrolling 让每个线程先处理多个元素，减少 block 数和循环次数。
   - 最后一个 warp 内可以展开同步，减少 `__syncthreads()` 开销。
   - `__shfl_down_sync()` 可以在寄存器之间做 warp 内数据交换，避免部分内存读写。
   - 模板参数把 block size 变成编译期常量，有利于编译器删除无关分支。

4. 同步既保证正确性，也带来成本。
   - block 内数据依赖需要 `__syncthreads()`。
   - kernel 计时需要主机端同步或 CUDA event。
   - 动态并行中的设备端同步成本很高，且新 CUDA 版本中旧 CDP1 设备端同步 API 已不推荐使用。

5. 动态并行适合表达递归结构，但不一定适合高性能归约。
   - `nestedReduce` 类示例更适合理解 parent/child kernel 关系。
   - 对 reduction 这类经典数据并行问题，单层或少层 kernel 配合 unrolling、warp shuffle 和共享内存通常更可靠、更高效。

6. profiling 需要区分普通运行时间和 profiler 下的采集时间。
   - `ncu` 插桩和 replay 会放大程序打印的 elapsed time。
   - 性能结论应结合普通运行、kernel 指标和源码结构一起判断。

## 推荐实验命令

```sh
cd code_samples/chapter_03
make
./simpleDeviceQuery
./sumMatrix 32 32
./sumMatrix 16 16
./simpleDivergence
./reduceInteger 512
./nestedReduce2
```

## 学习收获

本章的主线是从“能并行运行”走向“理解为什么快或慢”。线程束分化解释了控制流成本，归约示例展示了经典 kernel 优化路线，动态并行说明了 GPU 端启动 kernel 的能力和代价。后续学习应继续围绕访存合并、共享内存、占用率、warp 级原语和 profiling 指标建立稳定的性能判断能力。
