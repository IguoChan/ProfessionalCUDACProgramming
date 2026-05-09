# Professional CUDA C Programming

本仓库用于学习和实践 CUDA C 编程，代码按章节组织，包含设备查询、线程索引、网格与线程块配置、向量加法、矩阵加法、线程分支以及内存访问等基础示例。

## 项目结构

```text
code_samples/
├── common/       # 公共工具函数与 CUDA 错误检查宏
├── chapter_01/   # CUDA Hello World 示例
├── chapter_02/   # 线程模型、设备信息、数组与矩阵计算示例
├── chapter_03/   # 分支、设备查询和矩阵计算示例
└── chapter_04/   # 全局变量、内存传输、固定内存、数据布局、零拷贝和读写偏移示例
```

每个章节目录都有独立的 `Makefile`，示例程序通常由同名 `.cu` 或 `.c` 文件编译生成。

## 环境要求

- NVIDIA GPU
- 已安装 NVIDIA 驱动
- 已安装 CUDA Toolkit，并确保 `nvcc` 可用
- Linux 环境下的 `make` 和 `gcc`

可以使用以下命令检查 CUDA 编译器：

```sh
nvcc --version
```

## 编译与运行

进入对应章节目录后执行 `make`：

```sh
cd code_samples/chapter_02
make
```

运行生成的示例程序：

```sh
./sumArraysOnGPU-timer
```

清理当前章节生成的可执行文件：

```sh
make clean
```

当前 Makefile 默认使用 `-arch=sm_89` 编译。如果你的 GPU 架构不匹配，请根据设备计算能力修改对应章节的 `Makefile`。

chapter 04 的 `readSegment` 还提供了用于比较 global load cache modifier 的实验目标：

```sh
cd code_samples/chapter_04
make readSegment-cache
./readSegment_ca 0
./readSegment_cg 0
```

其中 `readSegment_ca` 使用 `-Xptxas -dlcm=ca` 编译，`readSegment_cg` 使用 `-Xptxas -dlcm=cg` 编译。详细结论见 `code_samples/chapter_04/readSegment_ncu_analysis.md`。

## 示例说明

- `chapter_01/hello.cu`：最小 CUDA 程序示例。
- `chapter_02/checkDeviceInfor.cu`：查看 CUDA 设备信息。
- `chapter_02/checkThreadIndex.cu`：观察线程、线程块和网格索引。
- `chapter_02/sumArraysOnGPU-*.cu`：比较 CPU 与 GPU 上的数组加法。
- `chapter_02/sumMatrixOnGPU-*.cu`：使用不同网格和线程块布局进行矩阵加法。
- `chapter_03/simpleDivergence.cu`：演示线程分支对执行行为的影响。
- `chapter_03/sumMatrix.cu`：矩阵加法示例。
- `chapter_04/globalVariable.cu`：演示设备端全局变量的读写。
- `chapter_04/memTransfer.cu`：演示主机与设备之间的基础内存传输。
- `chapter_04/pinMemTransfer.cu`：比较可分页内存与页锁定内存的传输性能。
- `chapter_04/readSegment.cu`：通过可配置偏移量观察非对齐读访问的执行表现。
- `chapter_04/readSegment_ncu_analysis.md`：记录 RTX 4090 上的非对齐读访问和 `-dlcm` cache modifier 实验。
- `chapter_04/simpleMathAoS.cu`：使用 AoS 数据布局处理 `x/y` 成员。
- `chapter_04/simpleMathSoA.cu`：使用 SoA 数据布局处理 `x/y` 数组。
- `chapter_04/simpleMath_layout_ncu_analysis.md`：合并记录 AoS 与 SoA 数据布局在 RTX 4090 上的 `ncu` 对比。
- `chapter_04/sumArrayZerocpy.cu`：演示零拷贝内存进行数组加法。
- `chapter_04/writeSegment.cu`：通过可配置偏移量观察非对齐写访问及 unroll 写入的执行表现。
- `chapter_04/writeSegment_ncu_analysis.md`：记录 RTX 4090 上的非对齐写访问和 store sector 指标实验。

## 开发建议

- 新增示例时，将代码放入对应的 `code_samples/chapter_##/` 目录。
- 通用工具函数优先放入 `code_samples/common/common.h`。
- CUDA API 调用建议使用 `CHECK(...)` 宏进行错误检查。
- 提交前运行相关章节的 `make clean && make`，并手动执行受影响的示例程序。

## 参考

本仓库内容围绕 CUDA C 编程基础展开，适合作为阅读 CUDA 教材或学习 GPU 并行编程时的配套练习代码。
