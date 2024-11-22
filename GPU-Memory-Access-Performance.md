# GPU Memory Access Performance


## Global Memory
Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions.

When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread’s 4-byte access, throughput is divided by 8.

[1] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses



## Coalesced Access to Global Memory(全局内存合并访问)
A very important performance consideration in programming for CUDA-capable GPU architectures is the coalescing of global memory accesses. Global memory loads and stores by threads of a warp are coalesced by the device into as few as possible transactions.

Note

High Priority: Ensure global memory accesses are coalesced whenever possible.

The access requirements for coalescing depend on the compute capability of the device and are documented in the CUDA C++ Programming Guide.

For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.

For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required 128-byte aligned segments.

Note

On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not.

[1] https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html



## Behavior of L1/L2 Caches

从Pascal架构，ComputeCacity 6.0开始，

Q: I was reading about the L1 and L2 caches load and store and I have found that if there is a miss in L1 for a load instruction, L1 will get only the sector (32byte) of the 128 cache line from L2. But why do we say that the granularity of a fetching is 128byte? In which case do we fetch 128bytes? and what is the advantage of getting only 1 sector in a cache miss?

> In Fermi/Kepler days, a miss on the L1 triggered a 128byte request to the L2. Somewhere between Maxwell and Pascal this changed to a 32-byte granularity.
<br></br>
You’ll fetch 128 bytes if you have a request that needs 128 bytes. For example if you have a warp-wide load of a float or int per thread, adjacent. The advantage of getting only 1 sector on a cache miss needs to be considered in the case of a warp request that only needs 32 bytes or less. In that case, it is preferable to request 32 bytes rather than 128.

[1] [ref](https://forums.developer.nvidia.com/t/behavior-of-l1-l2-caches/255293)


## How to access global memory efficently in CUDA C/C++ kernels
2013年的文章，年代较老，介绍了Compute Capability1.2 1.3 2.0相关设备的内存访问特性。

[1] https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/

## what are cuda global memory 32-,64- and 128- byte transaction
Both of these need to be understood in the context of a CUDA warp. All operations are issued warp-wide, and this includes instructions that access memory.

An individual CUDA thread can access 1,2,4,8,or 16 bytes in a single instruction or transaction. When considered warp-wide, that translates to 32 bytes all the way up to 512 bytes. The GPU memory controller can typically issue requests to memory in granularities of 32 bytes, up to 128 bytes. Larger requests (say, 512 bytes, considered warp wide) will get issued via multiple "transactions" of typically no more than 128 bytes.

Modern DRAM memory has the design characteristic that you don't typically ask for a single byte, you request a "segment" typically of 32 bytes at a time for typical GPU designs. The division of memory into segments is fixed at design time. As a result, you can request either the first 32 bytes (the first segment) or the second 32 bytes (the second segment). You cannot request bytes 16-47 for example. This is all a function of the DRAM design, but it manifests in terms of memory behavior.

The diagram(s) depicts the behavior of each thread in a warp. Individually, they are depicted by the gray/black arrows pointing upwards. Each arrow represents the request from a thread, and the arrow points to a relative location in memory that that thread would like to load or store.

The diagrams are presented in comparison to each other to show the effect of "alignment". When considered warp-wide, if all 32 threads are requesting bytes of data that belong to a single segment, this would require the memory controller to retrieve only one segment to satisfy the request. This would arguably be the most efficient possible behavior (and therefore data organization as well as access pattern, considered warp-wide) for a single request (i.e. a single load or store instruction).

However if the addresses emanating from each thread in the warp result in a pattern depicted in the 2nd figure, this would be "unaligned", and even though you are effectively asking for a similar data "footprint", the lack of alignment to a single segment means the memory controller will need to retrieve 2 segments from memory to satisfy the request.

That is the key point of understanding associated with the figure. But there is more to the story than that. Misaligned access is not necessarily as tragic (performance cut in half) as this might suggest. The GPU caches play a role here, when we consider these transactions not just in the context of a single warp, but across many warps.

To get a more complete and orderly treatment of these topics, I suggest referring to various training material. It's by no means the only one, but unit 4 of this training series will cover the topic in more detail.


[1] https://stackoverflow.com/questions/72147025/what-are-cuda-global-memory-32-64-and-128-byte-transactions

## Pascal L1 Cache
https://forums.developer.nvidia.com/t/pascal-l1-cache/49571


## Cache Behavior in compute capability 7.5
- L1 cache line size is 128 bytes divided into 4 32 byte sectors. On a miss only addressed sectors will be fetched from L2.
- L2 cache line size is 128 bytes divided into 4 32 byte sectors.

[1] https://stackoverflow.com/questions/63497910/cache-behaviour-in-compute-capability-7-5

## The granularity of L1 and L2 caches
[1] https://forums.developer.nvidia.com/t/the-granularity-of-l1-and-l2-caches/290065


| arch code | Fermi | Kepler | Maxwell | Pascal | Volta | Turning | Ampere | Ada Lovelace | Hopper |
|-----------|-------|--------|---------|--------|-------|---------|--------|--------------|--------|
|compute capacity| sm_20 | sm_30 | sm_50 | sm_60| sm_70 |  sm_75  |  sm_80 |     sm_89    |  sm_90 |  
 