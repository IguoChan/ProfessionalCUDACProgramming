#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * 本示例对整数数组做并行归约求和，用同一组输入依次比较多种 CUDA
 * reduction 写法：
 *
 * 1. neighbored pair：相邻配对规约，代码直观但会产生明显的线程束分支分化。
 * 2. neighbored less：把活跃线程重新映射到连续 tid，减少分支分化。
 * 3. interleaved pair：从 blockDim.x / 2 开始对半折叠，访存模式更规整。
 * 4. unrolling 2/4/8：每个线程先累加多个全局内存元素，减少 block 数量和
 *    kernel 内循环工作量。
 * 5. warp unrolling：当规约只剩最后一个 warp 时，利用 warp 内线程隐式同步，
 *    展开最后几步加法，减少 __syncthreads() 开销。
 * 6. complete unroll：对固定 block size 的规约步骤做完全展开，让编译器消除
 *    循环控制开销。
 *
 * 注意：这些 kernel 都直接在全局内存 g_idata 上做 block 内原地规约。每个
 * block 最终只向 g_odata 写出一个部分和，主机端随后再把所有部分和累加成
 * 最终结果，并与 CPU 递归规约结果比较。
 */

// CPU 端递归归约，用作 GPU 结果的参考答案。
int recursiveReduce(int *data, int const size)
{
    // 递归终止条件：数组已经被规约到只剩一个元素。
    if (size == 1) return data[0];

    // 每一轮把后半段加到前半段，所以有效问题规模缩小一半。
    int const stride = size / 2;

    // 在原数组上就地规约：data[i] 保存 data[i] + data[i + stride]。
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // 继续规约前半段，直到只剩最终和。
    return recursiveReduce(data, stride);
}

// 相邻配对规约：第 0、2、4... 号线程先工作，下一轮第 0、4、8... 号线程工作
// tid % (2 * stride) 会让同一个 warp 中部分线程执行 if、部分线程跳过，
// 因而产生较明显的 warp divergence。
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // tid 是 block 内局部线程编号，idx 是全局输入数组下标。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // idata 指向当前 block 负责的全局内存片段，后续用 tid 做片段内索引。
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 输入长度不是 blockDim.x 整数倍时，越界线程直接退出。
    if (idx >= n) return;

    // stride 从 1 开始翻倍，每轮把距离 stride 的相邻元素加到左侧元素。
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 只有每个 2 * stride 分组中的第一个线程参与本轮加法。
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // 下一轮依赖本轮写回的中间结果，必须等待 block 内所有活跃线程完成。
        __syncthreads();
    }

    // 每个 block 的规约结果保存在该片段第 0 个位置，由 tid 0 写到输出数组。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 改进的相邻配对规约：仍按相邻元素配对，但用 index = 2 * stride * tid
// 让活跃线程集中在较小的连续 tid 范围内，减少 warp 内分支分化。
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata,
                                      unsigned int n)
{
    // block 内编号和全局输入下标。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 block 负责的输入片段起始地址。
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 防止最后一个 block 访问数组末尾之外的元素。
    if(idx >= n) return;

    // 每轮仍然把 index + stride 累加到 index。
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 把连续的 tid 映射到本轮真正要处理的元素下标。
        int index = 2 * stride * tid;

        // 只有映射后的下标仍在当前 block 片段内时才做加法。
        if (index < blockDim.x)
        {
            // stride = 1, tid: 0 ~ blockDim.x/2-1 号线程工作
            // stride = 2, tid: 0 ~ blockDim.x/4-1 号线程工作
            // stride = 4, tid: 0 ~ blockDim.x/8-1 号线程工作
            // ...
            // stride = blockDim.x/2, tid: 只有 0 号线程工作
            idata[index] += idata[index + stride];
        }

        // 保证本轮所有写入在下一轮读取前可见。
        __syncthreads();
    }

    // 写出当前 block 的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 交错配对规约：从 blockDim.x / 2 开始，每轮把右半段加到左半段。
// 活跃线程条件是 tid < stride，因此活跃线程始终是连续的一段，分支分化更少。
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // block 内线程编号与全局输入下标。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 block 的全局内存片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 越界线程不参与规约。
    if(idx >= n) return;

    // stride 每轮减半，逐步把右侧元素折叠到左侧。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        // 前 stride 个线程参与本轮加法。
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // 下一轮读取 idata[tid]，所以这里需要 block 级同步。
        __syncthreads();
    }

    // 当前 block 的最终部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 每个 block 覆盖 2 * blockDim.x 个元素：每个线程先合并两个相隔 blockDim.x
// 的元素，再按 interleaved 方式规约。这样 block 数量减半，减少 kernel 调度和
// 后续主机端累加的部分和数量。
__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n)
{
    // idx 指向该线程负责的第一份输入。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 一个 block 的逻辑输入片段现在是 2 * blockDim.x 个元素。
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // 先在全局内存中合并两份数据；边界检查保证第二份输入存在。
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    // 确保所有线程的预合并结果都写回后，再进入 block 内规约。
    __syncthreads();

    // 预合并后只需要规约 idata[0..blockDim.x-1]。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // 每一轮都依赖上一轮的写回结果。
        __syncthreads();
    }

    // 写出当前 block 覆盖的 2 * blockDim.x 个元素的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling4：每个线程先串行累加 4 个元素，然后 block 内再规约。
// 相比 unrolling2，读取更多连续分段，进一步减少参与后续规约的 block 数量。
__global__ void reduceUnrolling4 (int *g_idata, int *g_odata, unsigned int n)
{
    // 每个 block 覆盖 4 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // 只有当第 4 份数据也没有越界时才一次读取 4 个元素。
    if (idx + 3 * blockDim.x < n)
    {
        // 临时变量帮助编译器把多次全局内存读取组织成寄存器中的串行加法。
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // 对每个线程预合并后的 blockDim.x 个值继续规约。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // 等待本轮规约写回。
        __syncthreads();
    }

    // 当前 block 覆盖的 4 * blockDim.x 个元素的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling8：每个线程先合并 8 个元素，是本文件后续优化版本的基础。
__global__ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    // 每个 block 覆盖 8 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 先让每个线程串行累加 8 个相隔 blockDim.x 的元素。
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // 再对预合并后的 blockDim.x 个值做标准 interleaved 规约。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // 本轮所有线程写完后才能进入下一轮。
        __syncthreads();
    }

    // 当前 block 覆盖的 8 * blockDim.x 个元素的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling8 + warp 展开：block 内规约只循环到 stride > 32。
// 当剩下 64 个值时，最后一个 warp 的 32 个线程完成固定 6 步加法，
// 省掉最后几轮循环判断和 __syncthreads()。
__global__ void reduceUnrollWarps8 (int *g_idata, int *g_odata, unsigned int n)
{
    // 每个 block 覆盖 8 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 每个线程先串行合并 8 个元素。
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // 只规约到 stride == 64 为止；最后 64 个值交给一个 warp 展开处理。
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // stride 大于 warpSize 时，参与线程可能跨多个 warp，仍需要显式同步。
        __syncthreads();
    }

    // 最后一个 warp 内的线程天然以 SIMT 方式同步执行，所以不再调用
    // __syncthreads()。volatile 防止编译器把这些全局内存访问过度缓存到寄存器，
    // 保持每一步都能读到前一步写回的值。
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // 写出当前 block 的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling8 + warp shuffle：前半段仍用全局内存和 __syncthreads() 做跨 warp
// 规约；最后一个 warp 改用 __shfl_down_sync() 在寄存器之间传值，避免 volatile
// 内存读写。
__global__ void reduceUnrollWarps8New (int *g_idata, int *g_odata,
        unsigned int n)
{
    // 每个 block 覆盖 8 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 每个线程先串行合并 8 个元素。
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // 跨 warp 阶段仍然需要 block 级同步。
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // 剩下 64 个值时，只让第一个 warp 工作。每个 lane 先合并一对值，
    // 之后通过 shuffle 从更高 lane 读取寄存器值，完成 32 -> 1 的规约。
    if (tid < 32)
    {
        int sum = idata[tid] + idata[tid + 32];
        unsigned int mask = 0xffffffff;

        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum,  8);
        sum += __shfl_down_sync(mask, sum,  4);
        sum += __shfl_down_sync(mask, sum,  2);
        sum += __shfl_down_sync(mask, sum,  1);

        if (tid == 0) g_odata[blockIdx.x] = sum;
    }
}

// 完全展开版：在 unrolling8 和 warp 展开的基础上，把 block 内大步长规约
// 也写成固定 if 语句，避免 for 循环控制开销。这里使用运行时 blockDim.x
// 判断，因此编译器不能像模板版那样完全消除无关分支。
__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // 每个 block 覆盖 8 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 每个线程先合并 8 个输入元素。
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // 下列 if 对应 stride=512、256、128、64 的规约步骤。
    // 每一步后都同步，因为这些步骤可能由多个 warp 协同完成。
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // 最后 64 -> 1 的阶段在一个 warp 内手工展开。
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // 写出当前 block 的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 模板完全展开版：iBlockSize 是编译期常量，编译器可以在实例化时删除
// 不可能执行的分支，并对固定 block size 生成更精简的代码。
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // 每个 block 覆盖 8 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 每个线程先串行合并 8 个输入元素。
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // 这些条件依赖编译期常量 iBlockSize，未命中的分支会被优化掉。
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // 最后一个 warp 内展开固定加法序列。
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // 写出当前 block 的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling2 + warp 展开：和 reduceUnrollWarps8 的思路相同，但每个线程
// 只预合并 2 个元素，用于观察预展开倍数变化对性能的影响。
__global__ void reduceUnrollWarps (int *g_idata, int *g_odata, unsigned int n)
{
    // 每个 block 覆盖 2 * blockDim.x 个元素。
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 当前 block 的逻辑输入片段起点。
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // 每个线程先合并两个元素。
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // 规约到只剩最后一个 warp。
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // 跨 warp 规约阶段需要显式同步。
        __syncthreads();
    }

    // 展开最后一个 warp 的 64 -> 1 规约。
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // 写出当前 block 的部分和。
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // 固定使用 0 号 GPU，并打印设备信息，便于比较不同机器上的运行结果。
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // 输入元素总数。1 << 24 表示 16,777,216 个 int。
    int size = 1 << 24;
    printf("    with array size %d  ", size);

    // 默认每个 block 使用 512 个线程；也可以通过命令行参数覆盖。
    int blocksize = 512;

    if(argc > 1)
    {
        // 例如 ./reduceInteger 256 会用 256 线程/block 启动各个 kernel。
        blocksize = atoi(argv[1]);
    }

    // grid 是未展开版本需要的 block 数。unrolling2/4/8 会分别使用
    // grid.x / 2、grid.x / 4、grid.x / 8 个 block。
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // h_idata 保存原始输入，h_odata 保存 GPU 每个 block 写出的部分和，
    // tmp 是 CPU 递归规约使用的输入副本，因为 recursiveReduce 会原地修改数据。
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // 初始化输入数组。限制到 0..255 可以避免本例规模下 int 求和溢出。
    for (int i = 0; i < size; i++)
    {
        // 只保留低 8 位。
        h_idata[i] = (int)( rand() & 0xFF );
    }

    // CPU 参考实现会破坏输入数据，因此使用 tmp 副本。
    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // d_idata 是 GPU 输入和原地规约缓冲；d_odata 存放每个 block 的部分和。
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // CPU 递归规约，既用于计时，也用于后续校验 GPU 结果。
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1：相邻配对规约，作为最基础的 GPU 版本。
    // 每次测试前都重新拷贝 h_idata，因为 kernel 会原地修改 d_idata。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    // 每个 block 只输出一个部分和，拷回主机后再做最后一层求和。
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2：改进的相邻配对规约，活跃线程更连续，减少分支分化。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3：交错配对规约，通常比 neighbored pair 有更好的执行效率。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4：每个线程先处理 2 个元素，所以只需要 grid.x / 2 个 block。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5：每个线程先处理 4 个元素，进一步减少 block 数。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6：每个线程先处理 8 个元素，是后续 warp 展开版本的基础。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 7：unrolling8 加最后一个 warp 手工展开。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 8：用 __shfl_down_sync() 重写最后一个 warp 的规约。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8New<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu ShflWarp8   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9：unrolling8 + warp 展开 + block 内规约完全展开。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 10：模板完全展开版。根据命令行指定的 blocksize 选择匹配实例，
    // 让 iBlockSize 成为编译期常量。
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    default:
        printf("Unsupported block size for reduceCompleteUnroll: %d\n",
               blocksize);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // 释放主机端内存。
    free(h_idata);
    free(h_odata);
    free(tmp);

    // 释放设备端内存。
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // 重置设备，方便 profiling 工具拿到完整 trace。
    CHECK(cudaDeviceReset());

    // 这里只检查最后一个 kernel 的结果；前面各个 kernel 的 gpu_sum 已经打印，
    // 可以直接与 cpu_sum 对照。
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
