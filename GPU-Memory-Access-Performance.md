# GPU Memory Access Performance


# Device Memory
CUDA异构编程模型假设每个GPU设备都有自己的独立内存即显存。在GPU中也实现了多级Cache的显存层次结构，即：
- 显存: 
- L2 Cache: 
- L1 Cache/Shared Memory: 
- L0 Instruction Cache: 


## 主存和显存间数据传输效率

- host端使用page locked(锁页内存)，锁页内存最大的好处是锁页内存和显存间数据拷贝可以和核函数的执行并行。
- 每次发起D2H和H2D传输都存在overhead，将尽可能多的小数据量传输封装成一个较大数据的单次传输，比多次的小数据传输性能更好。
    > Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately.


## Global Memory
GPU的全局内存驻留在物理设备内存中，设备内存是通过32/64/128字节的内存事务来访问的，并且需要各自按32/64/128字节对齐，才能发起32/64/128字节内存段的事务访问。

当warp执行全局内存访存指令时，它会根据warp内每个线程访问的word大小以及warp内所有线程的访存地址分布，将warp内线程的访存合并为一个活多个内存事务。通常来说，warp的一次访存需求产生的访存事务越多，除线程访问字之外，传输的未使用字越多，有效传输带宽越低。比如，当warp中每个线程的4-byte内存访问地址连续，且首地址按32-byte对齐，只会产生4个32-byte的内存事务，且每个事务的32byte均为该warp所需的有效数据；若warp中每个线程需要访问一个4-byte字，但每个thread间内存地址跨步为32，会导致每个线程都发起了一个32-byte的访存事务，内存带宽降低为最大带宽的1/8。

> Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions.
<br><br>
When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread’s 4-byte access, throughput is divided by 8. [1]


## Coalesced Access to Global Memory(全局内存合并访问)

在CUDA编程中，一个重要的性能优化点是合并全局内存的访问，GPU会尽可能的将同一warp中线程的全局内存访问合并成最少得访存事务。

对于计算能力6.0及以上的设备，访存合并的要求可以简单总结为：同一warp中线程的并发访存会被合并为多个事务，这些事务的数量等于可以服务warp中所有线程访存要求的32-byte字节事务的最小数量。

对于计算能力为6.0及以上的设备，L1 cache是默认开启的，无论全局读是否被缓存在L1 cache中，访存是按长度为32字节的内存段操作的。对于计算能力5.2的设备，对全局内存的L1 cache可以选择性的开启，若开启了L1 cache，所需要的访存事务数量等于长度为128字节的内存段的数量，128字节内存段在warp coalesced memory size较小时，会导致较大的访存带宽浪费，因而在计算能力6.0及随后版本，将内存transaction的内存segment调小成了32 Byte。

> A very important performance consideration in programming for CUDA-capable GPU architectures is the coalescing of global memory accesses. Global memory loads and stores by threads of a warp are coalesced by the device into as few as possible transactions.
<br><br>
For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.
<br><br>
For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required 128-byte aligned segments.
<br><br>
On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not. [2]


## How to access global memory efficently in CUDA C/C++ kernels
2013年的文章，年代较老，介绍了Compute Capability 1.0 1.1 1.2 2.0等相关设备的内存访问特性。
对计算能力为1.0和1.1的设备，内存地址对齐和内存合并都是非常重要的高性能访存要求，未对齐到32自己的内存访问其内存带宽降低到对齐情况下的1/8（降低了87.2%），推论起来是在CC 1.0/1.1的访存实现中，所有未按32字节对齐的访存，都会产生一个访存事务，没有对同一个warp中不同threads的访存进行合并。，从CC1.2/1.3开始，非32字节对齐的内存访问，会被合并成尽可能少的访存事务，所以内存带宽降低了大概1/2。而在CC 2.0及更新的架构中，引入了L1 Cache且Cache Line是128字节，即使warp访存没有按32字节对齐，但L1 Cache的引入，将当前warp按memory transaction访存所读取到到对当前warp无用的数据，会在L1 cache缓存起来，来加速随后的warp的内存访问，所以整体的访存带宽非常高。

所以从CC 2.0开始，mis-aligned对访存效率影响不再是主要问题，但strided memory access（跨步内存访问）一直都是需要关注的低效访存模式。


## 扩展阅读-如何理解访存事务的内存块的大小/Cache Line
在早期计算能力较低的GPU架构，不同CC的warp size和访存事务，cache line都在迭代和变化，比如在CC1.0/1.1中没有访存合并的能力，在CC1.x中没有L1 Cache。在CC6.0及之后，GPU的L1/SharedMemory -> L2 Cache -> Global Memory层次结构稳定下来，合并访存、CacheLine等机制也已完善稳定，因为在[2]中从CC6.0开始，NV GPU的访存特性有了同一的表述。

对Pascal（CC6.0）及之后的架构，GPU中L1/L2 Cache都是按sector访问的，每个sector的粒度是32字节，对应一个32字节的内存段。而L1 Cache每条128字节的cache line由4个sectors构成，cache查找进行时可以对4路sector进行同时查找，即tag查找的粒度是128字节，某个section的查找miss，并不意味着这一个cache line要被整体覆盖和填充，GPU只会从下一级存储加载所丢失的那个32字节内存段。这种模式可以有效降低无效内存传输导致的带宽浪费。[4]~[7]


> In modern GPUs (say, Pascal and newer) both the L1 and L2 cache can be populated sector-by-sector. The minimum granularity is 1 sector or 32 bytes. The cache line tag, however, applies to 4 sectors (in each case) that comprise the 128-byte cache line. You can adjust L2 cache granularity.

另外[7]中提到一个cuda线程在一个读取指令可以访问1、2、4、8、16字节数据，即一个cuda线程一次最多可以读取一个int4结构。



**************
参考文献

[1] [CUDA C Programming Guide-Device Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)</span>


[2] [Cuda C best Practices Guide-Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

[3] [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

[4] [the granularity of L1 and L2 caches](https://forums.developer.nvidia.com/t/the-granularity-of-l1-and-l2-caches/290065)

[5] [behavior of L1/L2 caches](https://forums.developer.nvidia.com/t/behavior-of-l1-l2-caches/255293)

[6] [Pascal L1 cache](https://forums.developer.nvidia.com/t/pascal-l1-cache/49571)

[7] [what are cuda global memory 32- 64 128-byte transactions](https://stackoverflow.com/questions/72147025/what-are-cuda-global-memory-32-64-and-128-byte-transactions)