#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

/*
 * 本示例演示二维矩阵逐元素加法：C = A + B。
 *
 * CPU 版本通过双层循环按行、列遍历矩阵，GPU 版本使用二维 grid 和二维 block
 * 将矩阵元素映射到 CUDA 线程。程序会先在主机端计算参考结果，再在设备端执行
 * kernel，最后把 GPU 结果拷回主机并逐元素比较，用于验证线程索引和边界判断
 * 是否正确。
 */

/*
 * 返回当前墙钟时间，单位为秒。
 *
 * 该函数用于粗略测量主机端初始化、CPU 计算和 GPU kernel 执行耗时。GPU kernel
 * 是异步启动的，因此测量 kernel 时间时必须在启动前后调用 cudaDeviceSynchronize，
 * 保证计时区间真正覆盖设备端执行时间。
 */
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

/*
 * 初始化一段主机端浮点数组。
 *
 * 每个元素使用低 8 位随机数生成 0.0 到 25.5 之间的数值。这里不追求随机分布质量，
 * 只需要为矩阵加法准备可重复检查的非零输入数据。
 */
void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

/*
 * 在 CPU 上计算矩阵逐元素加法。
 *
 * A、B、C 都按行优先的一维数组存储，nx 表示每行元素个数，ny 表示行数。
 * 外层循环遍历行，内层循环遍历列；每处理完一行，就把三个指针同时向后移动
 * nx 个元素，指向下一行的开头。该结果作为 GPU 计算的参考答案。
 */
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

/*
 * 比较 CPU 参考结果和 GPU 计算结果。
 *
 * hostRef 保存 CPU 版本结果，gpuRef 保存从设备端拷回的结果，N 是总元素个数。
 * 如果任意元素差值超过 epsilon，就输出第一个不匹配的位置对应的数值并提示失败。
 * 本例做的是同顺序的浮点加法，理论上 CPU 与 GPU 结果应完全一致；epsilon 用于
 * 保留浮点比较的容忍度。
 */
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

/*
 * 使用二维 grid 和二维 block 在 GPU 上执行矩阵逐元素加法。
 *
 * blockIdx/threadIdx 的 x 维映射矩阵列索引 ix，y 维映射矩阵行索引 iy。
 * 一维数组下标 idx = iy * NX + ix，对应行优先存储中的第 iy 行第 ix 列。
 * grid 尺寸通过向上取整覆盖整个矩阵，因此边界 block 中可能存在越界线程，
 * 必须使用 ix < NX && iy < NY 保护实际访存。
 */
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    /*
     * 选择 CUDA 设备。
     *
     * 本例固定使用 0 号设备，并通过 CHECK 包装 CUDA Runtime API 调用，保证
     * 设备查询或设置失败时能立刻打印错误位置和原因。
     */
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    /*
     * 设置矩阵规模。
     *
     * nx 和 ny 分别表示矩阵宽度和高度，默认都是 2^14。矩阵总元素数为 nxy，
     * 字节数为 nBytes。该规模足够大，便于观察不同 block 配置对 kernel 时间
     * 的影响。
     */
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    /*
     * 分配主机端内存。
     *
     * h_A 和 h_B 是输入矩阵，hostRef 保存 CPU 参考结果，gpuRef 保存 GPU 结果。
     * 四个数组大小相同，均为 nx * ny 个 float。
     */
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    /*
     * 初始化主机端输入数据。
     *
     * A 和 B 使用随机数填充，参考结果和 GPU 结果缓冲区稍后清零，避免旧数据影响
     * 结果验证。
     */
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    /*
     * 在 CPU 上计算参考结果。
     *
     * 这一步与 GPU kernel 实现相同的矩阵加法逻辑，用于之后检查 GPU 结果是否正确。
     */
    iStart = cpuSecond();
    sumMatrixOnHost (h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;

    /*
     * 分配设备端全局内存。
     *
     * d_MatA 和 d_MatB 保存输入矩阵，d_MatC 保存 kernel 输出。所有 CUDA API
     * 调用都使用 CHECK 检查，便于定位显存不足或运行时错误。
     */
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    /*
     * 将输入数据从主机端拷贝到设备端。
     *
     * GPU kernel 只读取 d_MatA 和 d_MatB，并把结果写入 d_MatC。
     */
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    /*
     * 设置 kernel 启动参数。
     *
     * 默认 block 为 32 x 32，也可以通过命令行传入 dimx 和 dimy，例如：
     * ./sumMatrix 16 16。grid 使用向上取整计算，保证矩阵所有元素都至少被一个
     * 线程覆盖。
     */
    int dimx = 32;
    int dimy = 32;

    if(argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /*
     * 启动 GPU kernel 并测量执行时间。
     *
     * 第一次同步用于清空前面 CUDA 操作的影响；kernel 启动后再次同步，确保计时
     * 结束时设备端计算已经完成。CHECK(cudaGetLastError()) 用于捕获 kernel 启动
     * 配置错误，例如 block 中线程数超过设备限制。
     */
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %.5f ms\n", grid.x,
           grid.y,
           block.x, block.y, iElaps * 1000.0);
    CHECK(cudaGetLastError());

    /*
     * 将 GPU 结果拷回主机端。
     *
     * 后续 checkResult 在主机端比较 hostRef 和 gpuRef。
     */
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    /*
     * 验证 GPU 结果。
     *
     * 如果没有输出 "Arrays do not match."，说明 GPU 结果与 CPU 参考结果一致。
     */
    checkResult(hostRef, gpuRef, nxy);

    /*
     * 释放设备端全局内存。
     */
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    /*
     * 释放主机端内存。
     */
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    /*
     * 重置 CUDA 设备。
     *
     * 对简单示例程序来说，退出前 reset 可以清理当前进程创建的 CUDA 上下文，
     * 便于性能分析工具得到完整的运行信息。
     */
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
