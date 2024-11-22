# Warp Level Programming

## 1
张量计算是Warp Level的指令，每个Volta TensorCore需要8个thread提供数据，一个warp中仅有一半的线程会参与计算。猜想这样的设计也是受制于Register File的带宽，根据Cuda Core的数量（16个FP32 Core)推断，每个cycle寄存器应该能传输512bits，恰是2x8x2即32个FP16数据，也就是2个4x4的矩阵。

[ref](https://zhuanlan.zhihu.com/p/2148596914)



## 2
通过完整的执行Warp同时使用多个 Tensor Core。warp 内的线程提供更大的 16x16x16 矩阵运算以供 Tensor Core 处理。
CUDA 将这些操作公开为 CUDA C++ WMMA API 中的扭曲级矩阵操作。这些 C++ 接口提供专门的矩阵加载、矩阵乘法和累加以及矩阵存储操作，以在 CUDA C++ 程序中有效利用 Tensor Core

CUDA Core 针对各种并行计算任务进行了优化，更适合于通用并行计算任务。

首先，深度学习任务其不仅仅是矩阵运算，还有很多的并行计算，这就看瓶颈在哪里了。如果瓶颈时在并行计算，那么这种类型的深度学习任务可能更适合CUDA Core。

其次，Tensor Core的使用是有些限制的，对于GEMM计算其效果很好，其次其输入和输出数据类型需要是半精度或单精度，矩阵的维度最好是 8 的倍数。

[ref](https://zhuanlan.zhihu.com/p/678893340)

