# Chapter 01 Summary

## 本章定位

第一章是 CUDA 编程环境和最小可运行程序的入门章。当前目录中的 `hello.cu` 通过一个非常小的 kernel 展示了 CUDA 程序的基本形态：主机端 `main` 负责启动设备端函数，GPU 线程在 kernel 中执行代码。

## 代码示例

- `hello.cu`
  - 使用 `__global__` 声明设备端 kernel。
  - 通过 `hello<<<1, 10>>>()` 从 CPU 端启动 GPU kernel。
  - 使用 `cudaDeviceSynchronize()` 等待 GPU 执行完成，避免程序提前退出导致设备端 `printf` 没有完整输出。

## 核心知识点

1. CUDA 程序同时包含主机端代码和设备端代码。
   - 主机端代码运行在 CPU 上，负责内存管理、kernel 启动和同步。
   - 设备端代码运行在 GPU 上，通常写成 `__global__` kernel。

2. kernel 启动使用三尖括号语法。
   - `<<<grid, block>>>` 是 CUDA C 扩展语法。
   - 本章示例中 `<<<1, 10>>>` 表示启动 1 个线程块，每个线程块 10 个线程。

3. kernel 启动默认是异步的。
   - CPU 发起 kernel 后不会自动等待 GPU 完成。
   - 对于简单示例，`cudaDeviceSynchronize()` 可以强制主机端等待设备端完成。

4. GPU 端可以使用 `printf` 辅助观察线程执行。
   - `printf` 适合学习和调试，不适合高性能路径。
   - 多线程输出顺序不应被当作严格执行顺序。

## 运行方式

```sh
cd code_samples/chapter_01
make
./hello
```

## 学习收获

本章建立了 CUDA 程序的最小心智模型：CPU 负责调度，GPU 负责并行执行；kernel 通过执行配置决定启动多少线程；同步是理解 CUDA 程序行为的第一步。
