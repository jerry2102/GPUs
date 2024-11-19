#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <vector_types.h>
using namespace cooperative_groups;

#define FULL_MASK 0xffffffff

/*---------------------------- reduce sum kernel for a single thread block -----------------------*/
__global__ void reduce_sum(int* temp) {
    thread_block block = this_thread_block();
    int lane = block.thread_rank();
    
    int length = block.size();
    int middle = block.size() / 2;
    int middle_length = middle + (length & 1);
    for (; middle > 0; ) {
        if (lane < middle) {
            temp[lane] += temp[lane + middle_length];
        }
        length = middle_length;
        middle = length / 2;
        middle_length = middle + (length & 1);
    }
}

__global__ void reduce_sum_by_warp_level_primitives(int* temp, int32_t count) {
    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < count);
    if (threadIdx.x < count) {
        int val = temp[threadIdx.x];
        // clear temp[0], using it for storing result
        if (threadIdx.x == 0) temp[0] = 0;
        for (int offset = 16; offset > 0; offset /=2) {
            val += __shfl_down_sync(mask, val, offset);
        }
            
        if (threadIdx.x % 32 == 0) {
            atomicAdd(temp, val);
        }
    }
}

__global__ void reduce_sum_by_active_warps(int* temp, int32_t count) {
    if (threadIdx.x < count) {
        // using __activemask is incorrect as is would result in partial sums instead of total sum
        // tyhe CUDA execution model does not guarantee that all threads taking the branch together will 
        // execute the __activemask() together.
        // Implicit lock-step execution is not guaranteed
        uint32_t mask = __activemask();
        int val = temp[threadIdx.x];
        if (threadIdx.x == 0) temp[0] = 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (threadIdx.x % 32 == 0) {
            atomicAdd(temp, val);
        }
    }

}

__global__ void reduce_sum_using_manual_syncwarp(int* temp, int32_t count) {
    uint32_t tid = threadIdx.x;

    __shared__ int32_t shmem[32];
    shmem[tid] = temp[tid];
    __syncwarp();

    int v = shmem[tid];
    v += shmem[tid+16]; __syncwarp();
    shmem[tid] = v;     __syncwarp();
    v += shmem[tid+8];  __syncwarp();
    shmem[tid] = v;     __syncwarp();
    v += shmem[tid+4];  __syncwarp();
    shmem[tid] = v;     __syncwarp();
    v += shmem[tid+2];  __syncwarp();
    shmem[tid] = v;     __syncwarp();
    v += shmem[tid+1];  __syncwarp();
    shmem[tid] = v;

//    shmem[tid] += shmem[tid + 16]; __syncwarp();
//    shmem[tid] += shmem[tid + 8]; __syncwarp();
//    shmem[tid] += shmem[tid + 4]; __syncwarp();
//    shmem[tid] += shmem[tid + 2]; __syncwarp();
//    shmem[tid] += shmem[tid + 1]; __syncwarp();

    if (tid == 0) temp[0] = shmem[0];

}

__global__ void shuffle_values(int* temp, int32_t count) {
    float val = threadIdx.x;

    if (threadIdx.x % 32 <16) {
        val = __shfl_xor_sync(0xFFFFFFFF, val, 16);
    }
    else {
        val = __shfl_xor_sync(0xFFFFFFFF, val, 16);
    }
    printf("thread: %d, swpped: %f\n", threadIdx.x, val);
}

void test_reduce_sum() {
    int datas[] = {
        1,2,3,4,5,6,7,8,
        9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,
        25,26,27,28,29,30,31,32,
        33,34,35,36,37,38,39,40
    };
    int* gdatas = nullptr;
    cudaMalloc((void**)&gdatas, sizeof(datas));
    cudaMemcpy((char*)gdatas, (char*)datas, sizeof(datas), cudaMemcpyHostToDevice);
    //reduce_sum<<<1, 33>>>(gdatas);
    //reduce_sum_by_warp_level_primitives<<<1, 33>>>(gdatas, 33);
    //reduce_sum_by_active_warps<<<1, 40>>>(gdatas, 40);
    //reduce_sum_using_manual_syncwarp<<<1, 32>>>(gdatas, 32);
    shuffle_values<<<1, 32>>>(gdatas, 32);
    int* result = (int*)malloc(sizeof(datas));
    cudaMemcpy((char*)result, (char*)gdatas, sizeof(datas), cudaMemcpyDeviceToHost);
    std::cout << result[0] << ", " << result[1] << std::endl;
    cudaFree(gdatas);
}
/*================================================================================================*/

/*------------------------------- reduce sum using thread block ----------------------------------*/
// 
__device__ int reduce_sum(thread_group g, int* temp, int val) {
    int lane = g.thread_rank();
    int length = g.size();
    int middle = length / 2;
    int middle_length = middle + (length & 1);
    for (/**/; middle > 0; /**/) {
        temp[lane] = val;
        g.sync();
        if (lane < middle) {
            val += temp[lane + middle_length];
        }
        length = middle_length;
        middle = length / 2;
        middle_length = middle + (length & 1);
        g.sync();
    }
    return val;
}

// each thread calculate for 
__device__ int thread_sum(int* input, int n) {
    int sum = 0;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx < (n + 3) / 4; idx += blockDim.x * gridDim.x) {
        int4 in = ((int4*)input)[idx];
        sum += in.x + in.y + in.z + in.w;
    }
    // for the last thread of the last block
    if (idx == (n + 3) / 4) {
        for (int t = 4 * idx; t < n; ++t) {
            sum += input[t];
        }
    }

    return sum;
}

__global__ void sum_kernel_block(int* sum, int* input, int n) {
    // each kernel calculate 4 consecutive items
    int single_thread_sum = thread_sum(input, n);
    extern __shared__ int temp[];
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, single_thread_sum);

    if (g.thread_rank() == 0) {
        atomicAdd(sum, block_sum);
    }
}

__global__ void sum_kernel_by_tile32(int* sum, int* input, int n) {
    int single_thread_sum = thread_sum(input, n);
    extern __shared__ int temp[];
    auto g = this_thread_block();
    int tile_idx = g.thread_rank() / 32;
    int* tile_temp = &temp[32 * tile_idx];
    auto tile32 = tiled_partition(g, 32);
    
    int tile_sum = reduce_sum(tile32, tile_temp, single_thread_sum);
    
    if (tile32.thread_rank() == 0) atomicAdd(sum, tile_sum);

}

template<int tilesize> 
__device__ int reduce_sum_tile_shfl(thread_block_tile<tilesize> g, int val) {
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;
}

template<int tilesize> 
__global__ void sum_kern_tile_shfl(int* sum, int* input, int n) {
    int single_thread_sum = thread_sum(input, n);
    auto tile = tiled_partition<tilesize>(this_thread_block());
    int tile_sum = reduce_sum_tile_shfl<tilesize>(tile, single_thread_sum);
    if (tile.thread_rank() == 0) atomicAdd(sum, tile_sum);
}

void check_sum_kernel_block() {
    int n = 1 << 24;
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    int sharedBytes = blockSize * sizeof(int);
    int* hostData = (int*)malloc(n * sizeof(int));
    std::iota(hostData, hostData + n, 1);
    int* deviceData = nullptr;
    cudaMalloc((void**)&deviceData, (n + 4) * sizeof(int));
    cudaMemset(deviceData, 0, sizeof(int));
    // WARN: use deviceData[0] store sum result, 
    // but considering the alignment requirement of int4, we use deviceData + 4 as the beginning address of input
    // odd thing may happen is alignment is not satisfied
    cudaMemcpy((char*)(deviceData + 4), hostData, n * sizeof(int), cudaMemcpyHostToDevice);
    
    //sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(deviceData, deviceData + 4, n);
    //sum_kernel_by_tile32<<<nBlocks, blockSize, sharedBytes>>>(deviceData, deviceData + 4, n);
    sum_kern_tile_shfl<32><<<nBlocks, blockSize>>>(deviceData, deviceData + 4, n);

    int hostResult;
    cudaMemcpy((char*)&hostResult, (char*)deviceData, sizeof(int), cudaMemcpyDeviceToHost);

    printf("result: %d, %d\n", hostResult, std::accumulate(hostData, hostData + n, 0));
    cudaFree(deviceData);
}

// mock a global kernel calling the device reduce_sum function
__global__ void sum_kernel_block2(int* result, int* input, int n) {
    int idx = threadIdx.x;
    extern __shared__ int temp[];
    int mysum = 0;
    if (idx < n) {
        auto g = this_thread_block();
        mysum = reduce_sum(g, temp, input[idx]);
        printf("sum result, thread idx %d, %d \n", threadIdx.x, mysum);
    }
    if (idx == 0) *result = mysum;
}

void check_reduce_sum() {
    int datas[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    int* gdatas = nullptr;
    cudaMalloc((void**)&gdatas, sizeof(datas));
    cudaMemcpy((char*)gdatas, (char*)datas, sizeof(datas), cudaMemcpyHostToDevice);
    int* result = (int*)malloc(sizeof(datas));
    const int size = 13;
    sum_kernel_block2<<<1, size, size * sizeof(int)>>>(gdatas, gdatas, size);
    cudaMemcpy((char*)result, (char*)gdatas, sizeof(datas), cudaMemcpyDeviceToHost);
    std::cout << result[0] << std::endl;
    cudaFree(gdatas);
    
}
/*===============================================================================================*/



int main() {
    test_reduce_sum();
    //check_sum_kernel_block();
}   
