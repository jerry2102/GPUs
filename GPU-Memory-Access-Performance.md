# GPU Memory Access Performance


## Global Memory合并访存

Global Memory A

>The access requirements for coalescing depend on the compute capability of the device and are documented in the CUDA C++ Programming Guide.
<br><br>
For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.
<br><br>
For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required 128-byte aligned segments.





    On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not.


ABC

***