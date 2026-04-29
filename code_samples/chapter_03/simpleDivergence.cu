#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 */

/*
 * mathKernel1 用 tid 的奇偶性作为分支条件。
 *
 * CUDA 以线程束（warp）为基本调度单位，一个 warp 通常包含 32 个连续
 * threadIdx。对于同一个 warp 内的线程，tid 也是连续的：0, 1, 2, ...
 * 因此条件 tid % 2 == 0 会让相邻线程走不同路径：偶数线程执行 if 分支，
 * 奇数线程执行 else 分支。
 *
 * 当一个 warp 内同时存在满足和不满足条件的线程时，硬件不能真正同时执行
 * 两条分支路径，而是先屏蔽一部分线程执行其中一条路径，再切换掩码执行另
 * 一条路径，最后汇合。这就是线程束分化。该函数的每个 warp 基本都会被
 * 拆成两组线程执行，所以可以直接体现分支条件导致的 warp divergence。
 */
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
 * mathKernel2 用 (tid / warpSize) 的奇偶性作为分支条件。
 *
 * warpSize 是一个 warp 中的线程数量，通常为 32。tid / warpSize 会得到
 * 当前线程所在的 warp 编号：tid 0~31 得到 0，32~63 得到 1，以此类推。
 * 因为同一个 warp 内所有线程计算出的 warp 编号相同，所以它们会整体进入
 * 同一个 if 或 else 分支。
 *
 * 这种写法仍然有分支，但分支在 warp 粒度上是一致的，不会让同一个 warp
 * 内的线程走不同路径。因此它可以作为 mathKernel1 的对照，用来说明“是否
 * 分化”取决于同一个 warp 内的分支条件是否一致，而不是代码里是否出现 if。
 */
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
 * mathKernel1_divergent 是一个更容易观察性能差异的分化版本。
 *
 * 分支条件仍然是 tid % 2 == 0，所以同一个 warp 内偶数线程和奇数线程会
 * 走不同路径。与 mathKernel1 相比，这里在两个分支中加入了 1000 次数学
 * 运算。这样做不是为了改变分化原理，而是为了放大分化成本：发生分化后，
 * warp 需要分别执行 if 路径和 else 路径，两个路径中的长循环会被串行化
 * 地消耗更多指令周期。
 *
 * 如果分支体很短，计时结果可能被启动开销、访存或调度噪声掩盖；增加计算
 * 量后，分化导致的额外执行路径更容易在 elapsed time 中体现出来。
 */
__global__ void mathKernel1_divergent(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    
    // 同一 warp 内 tid 奇偶交错，因此这一分支会造成线程束分化。
    if (tid % 2 == 0) {
        for(int i = 0; i < 1000; i++) { // 增加计算循环
            val += sinf(cosf(tanf((float)tid))); // 一些“昂贵”的数学运算
        }
        c[tid] = val + 100.0f;
    } else {
        for(int i = 0; i < 1000; i++) {
            val -= cosf(sinf(tanf((float)tid)));
        }
        c[tid] = val + 200.0f;
    }
}

/*
 * mathKernel2_uniform 是 mathKernel1_divergent 的无分化对照版本。
 *
 * warp_id = tid / warpSize 表示线程所属的 warp 编号。同一个 warp 内的
 * 32 个线程拥有相同 warp_id，因此判断 warp_id % 2 == 0 时，整个 warp
 * 会统一选择 if 或 else 分支。
 *
 * 这里的两个分支也包含大量数学运算，计算量与 mathKernel1_divergent 接近。
 * 对比两者的运行时间时，主要差异来自分支是否在同一个 warp 内分裂：本函数
 * 每个 warp 只执行一条分支路径，而分化版本中的每个 warp 需要依次执行两条
 * 分支路径。
 */
__global__ void mathKernel2_uniform(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    int warp_id = tid / warpSize;
    
    // 无分化：整个 warp 的 warp_id 相同，因此所有线程执行同一条路径。
    if (warp_id % 2 == 0) {
        for(int i = 0; i < 1000; i++) {
            val += sinf(cosf(tanf((float)tid)));
        }
        c[tid] = val + 100.0f;
    } else {
        for(int i = 0; i < 1000; i++) {
            val -= cosf(sinf(tanf((float)tid)));
        }
        c[tid] = val + 200.0f;
    }
}

/*
 * mathKernel3 把分支条件先保存到布尔变量 ipred 中，再用两个独立的 if
 * 分别处理 true 和 false 情况。
 *
 * ipred 的值仍然来自 tid % 2 == 0，所以同一个 warp 内仍然会出现偶数线程
 * 为 true、奇数线程为 false 的情况。从逻辑上看，这与 mathKernel1 一样会
 * 产生线程束分化。
 *
 * 该函数用于说明：把条件表达式提前保存为谓词变量，并不会自动消除分化。
 * 只要同一个 warp 内线程对该谓词的取值不同，硬件仍需要使用执行掩码分别
 * 处理不同线程集合。
 */
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
 * mathKernel4 使用位移得到 warp 编号，再按 warp 编号选择分支。
 *
 * tid >> 5 等价于 tid / 32。因为常见 CUDA 设备的 warpSize 为 32，所以
 * itid 表示当前线程所在的 warp 编号。同一个 warp 内的所有线程拥有相同
 * itid，条件 ((itid & 0x01) == 0) 会让整个 warp 统一进入 if 或 else。
 *
 * 这种写法与 mathKernel2 的思想相同：分支可以存在，但分支条件必须在 warp
 * 内保持一致，才能避免同一个 warp 被拆成多条执行路径。它也展示了使用位
 * 运算按 warp 分组时，必须显式加括号，避免 C 运算符优先级改变判断含义。
 */
__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if ((itid & 0x01) == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
 * warmingup 用于预热 GPU 和计时路径。
 *
 * 第一次启动 kernel 可能包含上下文初始化、缓存状态变化等额外开销。先运行
 * 一个简单 kernel，可以让后续用于比较分化和无分化版本的计时更稳定。
 *
 * 这里采用与 mathKernel2 类似的 warp 粒度分支，避免预热本身引入明显的
 * warp 内分化；它的目的不是测试性能，而是让设备进入较稳定的执行状态。
 */
__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


/*
 * main 负责配置测试规模、分配 GPU 内存、使用 CUDA event 计时，并依次运行
 * 分化与无分化版本的 kernel。
 *
 * 对比时重点观察：
 * 1. mathKernel1_divergent：同一 warp 内线程按 tid 奇偶分裂，分支体又较重，
 *    因此更容易体现线程束分化的时间成本。
 * 2. mathKernel2_uniform：按 warp_id 分支，同一 warp 内路径一致，是无分化
 *    对照组。
 * 3. mathKernel3：展示保存谓词变量并不能改变分化本质。
 * 4. mathKernel4：展示通过 warp 编号构造一致分支条件可以避免 warp 内分化。
 */
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size - 默认 1000 万个元素，避免分配过大内存
    int size = 10000000;
    int blocksize = 256;

    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size = atoi(argv[2]);

    printf("Data size %d (Memory Size: %.2f MB)\n", size, (size * sizeof(float)) / (1024.0 * 1024.0));

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // 创建 CUDA 事件用于高精度计时
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float elapsedTime = 0.0f; // 毫秒

    // run a warmup kernel to remove overhead
    CHECK(cudaEventRecord(start, 0));
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("warmup      <<< %8d %4d >>> elapsed %.3f ms \n", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    // run kernel 1 (有分化)
    CHECK(cudaEventRecord(start, 0));
    mathKernel1_divergent<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("mathKernel1_divergent <<< %8d %4d >>> elapsed %.3f ms \n", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    // run kernel 2 (无分化)
    CHECK(cudaEventRecord(start, 0));
    mathKernel2_uniform<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("mathKernel2_uniform <<< %8d %4d >>> elapsed %.3f ms \n", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    // run kernel 3 (有分化)
    CHECK(cudaEventRecord(start, 0));
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("mathKernel3 <<< %8d %4d >>> elapsed %.3f ms \n", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    // run kernel 4 (无分化) - 注意：修复了原代码的运算符优先级bug
    CHECK(cudaEventRecord(start, 0));
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("mathKernel4 <<< %8d %4d >>> elapsed %.3f ms \n", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    // 销毁事件对象
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // free gpu memory and reset device
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
