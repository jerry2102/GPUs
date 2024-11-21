# Volta Architecture

Volta是Nvidia 2017年推出的新一代GPU架构。在Volta架构中，Nvidia引入多项新技术，如Tensor Core、Independent Thread Scheduling等，来加速深度学习的发展。

## Streaming Multiprocessor(SM)
Volta的SM较上一代Pascal有很多改进：
- CUDA Cores数量，同Pascal一样，Volta每个SM包含64个FP32 cores和32个FP64 core。然而和Pascal架构中，FP32和INT32共享同一个CUDA core不同，Volta SM包含了独立的FP32 core和INT32 core（如下图），从而Volta架构中FP32指令和INT32指令可以同时执行，这提高了Volta架构的指令吞吐，同时也缩减了FMA指令的指令延迟，特别的加快了混和了int地址计算和和float浮点运算的的循环操作将因此受益。
    >Unlike Pascal GPUs, which could not execute FP32 and INT32 instructions simultaneously, the Volta GV100 SM includes separate FP32 and INT32 cores, allowing simultaneous execution of FP32 and INT32 operations at full throughput, while also increasing instruction issue throughput. Dependent instruction issue latency is also reduced for core FMA (Fused Multiply-Add) math operations, requiring only four clock cycles on Volta, compared to six cycles on Pascal.
    Many applications have inner loops that perform pointer arithmetic (integer memory address calculations) combined with floating-point computations that will benefit from simultaneous execution of FP32 and INT32 instructions. Each iteration of a pipelined loop can update addresses (INT32 pointer arithmetic) and load data for the next iteration while simultaneously processing the current iteration in FP32.

- Processing Blocks的划分：
    - Pascal将每个SM划分为2个Processing Blocks，Volta将每个SM划分成4个Processing Blocks，对比如下。虽然划分方式不同，Volta每个SM的Register File总容量和Pascal一样都是256KB（16K * 4Byte/ProcessingBlock * 4 ProcessingBlock/SM = 256KB/SM）。Volta还实现了新的L0 Instruction Cache，较Pascal的Instruction Buffer效率更高。整体上来说，Volta的这种划分方式可以提高SM利用率和整体性能。

        | Architecuture | Pascal | Volta |
        |---------------|--------|-------|
        | Processing Blocks / SM | 2 | 4 |
        | FP32 core / PB  | 32 | 16|
        | INT32 core / PB | *_无独立的INT32 core_ | 16|
        | FP64 core / PB  | 16 | 8|        
        | Tensor core / PB  | - | 2 |        
        | Warp Scheduler / PB | 1 | 1|
        | Dispatch Unit / PB  | 2 | 1|
        | Register File | 128Kb | 64Kb|
        | Instruction Buffer / PB | Y | - |
        | L0 Instruction Cache / PB | - | Y |


- L1 Data Cache：Volta将每个SM中的L1 DataCache和Shared Memory进行了合并，使得L1 Cache的访问效率像Shared Memory一样高（Shared Memory的访问效率较高？）。合并后的容量是128KB/SM。如果kernel没有使用到Shared Memory，则这128KB都会被当成L1 Data Cache使用，来对streaming访问模式提供高带宽缓存支持，或者对频繁访问的数据提供低延迟访问支持，通过这种机制Volta缩小了显式Shared Memory和不使用shared memory直接访问device memory这两种操作键的性能gap（即使没有显式使用Shared Memory，Volta可以隐式的利用L1 DataCache来加速对Global Memory的访问，就像用户显示使用Shared Memory一样）。Nvidia做实验验证，在Volta架构下，一些计算任务不使用shared memory较使用shared memory性能损失只有7%，而对Pascal等架构其性能损失能达到30%。同时Volta的L1 Cache/Shared Memory也实现了对写的支持，进一步加速了性能。

    > The L1 In Volta functions as a high-throughput conduit for streaming data while simultaneously providing high-bandwidth and low-latency access to frequently reused data—the best of both worlds. This combination is unique to Volta and delivers more accessible performance than in the past.
     With Volta GV100, the merging of shared memory and L1 delivers a high-speed path to global memory capable of streaming access with unlimited cachemisses in flight. Prior NVIDIA GPUs only performed load caching, while GV100 introduces writecaching (caching of store operations) to further improve performance.

<center><img src="images/Volta-GV100-Streaming-Multiprocessor.png" alt="Volta Streaming Multiprocessor" height="400" /></center>

## Tensor Core
Volta架构中引入了全新的矩阵乘加专用计算单元。中每个SM有8个Tensor Core，即每个Processing Block或者每个Warp Scheduler含有2个Tensor Core，每个Tensor Core每时钟周期可以处理64个FMA操作，一个SM中8个Tensor Core每时钟周期可以处理```64 * 8 = 512```个FMA操作。


**Q: Volta Tensor Core算力是Pascal的12倍是如何计算出来的？**  
**A**: Volta Tensor Core算力是Pasacl的12倍源于[Volta Architecture Whitepaper](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)中的如下段落：
>Tesla V100’s Tensor Cores deliver up to 125 Tensor TFLOPS for training and inference
applications. Tensor Cores provide up to 12x higher peak TFLOPS on Tesla V100 that can be
applied to deep learning training compared to using standard FP32 operations on P100. 

12倍的性能增益是以整张Tesla V100的Tensor Core算力和Tesla P100的CUDA Core算力进行比较的，以Tesla V100 GPU的tensor Core理论算力和Tesla P100 GPU的CUDA Core理论算力进行比较计算的，12倍综合了单SM引入Tensor Core带来的算力增益、GPU卡的主频增益和SM数量增益。参考[Volta Architecture Whitepaper](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)文档Table 1中的Tesla各型号GPU参数，计算如下：

``` shell
# Volta Tensor Core算力 (FMA是两个指令，所以最后要乘2)
1530 Mhz * 10^6 * 80 SM * 8 TensorCore/SM * 64 FMA/cycle * 2 = 125.33 TFLOPS
#Pascal FP32 CUDA Core算力(FMA是两个指令，所以最后要乘2)
1480 MHz * 10^6 * 56 SM * 64 FP32 Core/SM * 2 = 10.6 TFLOPS
125.33 / 10.6 = 11.8 ≈ 12
# 单个SM Tensor Core算力增益为Pascal中所有CUDA Core的8倍 x 主频增益 x SM数量增益
8 * (1530 Mhz / 1480 Mhz) * (80 SM / 56 SM) = 11.8  ≈ 12
```





## Independent Thread Scheduling(ITS)
Pascal及以前的架构，其SIMT执行模型，将32个线程组成一个warp作为一个最小的调度单元。当程序中出现diverge时，warp中threads的执行路径会按divergence串行执行，直到所有thread重新聚合。该机制的根本原因在于，pre-Pascal架构，每个warp只有一个程序计数器（PC），同一warp的32个线程共享相同的PC，这导致同一warp中threads diverge时，只能串行执行，而diverge的分支逻辑间如果有数据依赖时，将导致warp死所等问题。
``` cpp
if (condition) {
    A;  // 如果执行A/B的threads对执行X/Y的线程有数据依赖，将导致warp在A/B卡死。
    B;
} else {
    X;
    Y;
}
```
Volta为每个线程设置了一个独立的程序计数器（PC），使每个线程有自己独立的执行环境。运行时Warp Scheduler仍然以warp为基本单位，每个时钟周期执行一条指令，但是不同分支的线程有平等的调度机会，即使前序分支阻塞了，后序分支只要满足其分支条件就可以执行。这解决了pre-Pascal架构中由于分支串行机制导致的死锁问题。（为了最大化并行执行效率，Volta引入了Schedule Optimizer来判决如何对一个warp中的活跃线程分组为SIMT执行单元。这样同一个warp中的线程可以以sub-warp的粒度进行diverge和recoverge，同时Convergence Optimizer仍然会将执行相同代码的线程分组在一起以获得最大并行效率）
> Volta’s independent thread scheduling allows the GPU to yield execution of any thread, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. To maximize parallel efficiency, Volta includes a schedule optimizer which determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity, while the convergence optimizer in Volta will still group together threads which are executing the same code and run them in parallel for maximum efficiency


Volta ITS的引入，使得pre-Pascal架构中广泛采用的Implicit Warp Level Synchronize机制变得，因而在Volta中引入了显式的Warp Level Primitive。

Volta ITS的引入，使得Volta可以做到任意线程的组合，也成为Cooperative Group机制得以实现的基础。

### Cooperative Group