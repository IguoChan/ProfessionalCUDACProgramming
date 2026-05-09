# simpleMath AoS/SoA ncu 分析

## 测试环境

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- Driver: 580.95.05
- CUDA Toolkit: 12.9
- `ncu`: Nsight Compute 2025.2.0.0
- 程序路径: `code_samples/chapter_04/simpleMathAoS` 和 `code_samples/chapter_04/simpleMathSoA`
- 数组规模: `1 << 22` 个元素
- block 大小: `128`
- 编译选项: 当前 `Makefile` 使用 `nvcc -O2 -arch=sm_89`

编译和普通运行：

```sh
cd code_samples/chapter_04
make simpleMathAoS simpleMathSoA
./simpleMathAoS 128
./simpleMathSoA 128
```

## 程序行为

`simpleMathAoS.cu` 使用 array of structures：

```cpp
struct innerStruct
{
    float x;
    float y;
};
```

kernel 对每个元素读取 `x/y`，分别加上常量，再写回另一个 `innerStruct` 数组：

```cpp
innerStruct tmp = data[i];
tmp.x += 10.f;
tmp.y += 20.f;
result[i] = tmp;
```

`simpleMathSoA.cu` 使用 structure of arrays：

```cpp
struct InnerArray
{
    float x[LEN];
    float y[LEN];
};
```

kernel 分别访问连续的 `x` 数组和 `y` 数组：

```cpp
float tmpx = data->x[i];
float tmpy = data->y[i];
result->x[i] = tmpx + 10.f;
result->y[i] = tmpy + 20.f;
```

## 普通运行结果

本次普通运行没有输出 `Arrays do not match.`，说明 GPU 结果与 CPU 参考结果一致。

| 程序 | kernel | grid | block | 普通运行时间 |
| --- | --- | ---: | ---: | ---: |
| `simpleMathAoS` | `testInnerStruct` | 32,768 | 128 | 0.000039 sec |
| `simpleMathSoA` | `testInnerArray` | 32,768 | 128 | 0.000040 sec |

普通运行时间非常接近。这个样例每个元素只做很少计算，单次计时容易受到缓存状态、launch 和同步开销影响；判断内存访问形态应优先看 `ncu` sector 指标。

## ncu 采集命令

```sh
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second --kernel-name regex:testInnerStruct --launch-count 1 --print-units base --print-metric-name name ./simpleMathAoS 128
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second --kernel-name regex:testInnerArray --launch-count 1 --print-units base --print-metric-name name ./simpleMathSoA 128
```

## ncu 结果

| 程序 | ncu kernel 时间 | L1TEX global load sectors | L1TEX global store sectors | DRAM read sectors | DRAM write sectors | global load 吞吐 | global store 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `simpleMathAoS` | 0.043456 ms | 2,097,152 | 2,097,152 | 1,052,104 | 191,424 | 1544.29 GB/s | 1544.29 GB/s |
| `simpleMathSoA` | 0.044064 ms | 1,048,576 | 1,048,576 | 1,053,772 | 206,644 | 761.49 GB/s | 761.49 GB/s |

相对 AoS：

| 程序 | L1TEX global load sectors 比例 | L1TEX global store sectors 比例 | ncu kernel 时间比例 |
| --- | ---: | ---: | ---: |
| `simpleMathAoS` | 1.000x | 1.000x | 1.000x |
| `simpleMathSoA` | 0.500x | 0.500x | 1.014x |

## 结果分析

SoA 的 L1TEX global load/store sectors 都是 AoS 的一半。对这个 kernel 来说，SoA 把 `x` 和 `y` 分成两个连续数组，warp 访问 `x[i]` 时是一段连续 `float`，访问 `y[i]` 时也是另一段连续 `float`。AoS 则以 `{x, y}` 交错布局访问，`x/y` 被组合在结构体元素中，L1TEX 侧统计到的 sector 数更高。

不过本次 RTX 4090 上的 `gpu__time_duration.sum` 几乎相同：AoS 为 `0.043456 ms`，SoA 为 `0.044064 ms`。这说明在这个小型算术 kernel 中，sector 数差异没有直接转化为可见的 kernel 时间差异。原因包括工作量很小、缓存命中、写回行为和 profiler replay 都会影响最终时间。

`dram__sectors_read.sum` 在两种布局下都约为 `1.05M`，没有像 L1TEX sectors 一样减半。这里应把 L1TEX sector 指标理解为 SM/L1TEX 侧的请求形态；DRAM 侧指标还受 L2、合并、写回和 profiler 采样窗口影响。

## 可复现实验命令

```sh
cd code_samples/chapter_04
make clean
make simpleMathAoS simpleMathSoA
./simpleMathAoS 128
./simpleMathSoA 128
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second --kernel-name regex:testInnerStruct --launch-count 1 --print-units base --print-metric-name name ./simpleMathAoS 128
ncu --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second --kernel-name regex:testInnerArray --launch-count 1 --print-units base --print-metric-name name ./simpleMathSoA 128
```
