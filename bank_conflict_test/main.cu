#include "zee_utils.h"
#include "zee_ndarray.h"

__global__ void hello_from_device() {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;

    if (tx==0 && ty==0) {
        printf("[%d, %d]: hello from device\n", tx, ty);
    }

}

__global__ void conflict_kernel(float * shared_store) {
    const int shared_load_size = 1024;
    __shared__ float shared_load[shared_load_size];   // 32*32 banks


    // 从shared memory中加载数据到global memory中
    // 前16个线程和后16个线程产生bank conflict
    int shared_load_idx = threadIdx.x*2;


    // 前16个线程和后16个线程存在交错，不会产生bank conflict了
    //- int shared_load_idx = threadIdx.x*2+threadIdx.x/16;


    shared_store[threadIdx.x] = shared_load[shared_load_idx];
}

int main() {
    printf("Hello World\n");
    float * d_shared_store;
    cudaMalloc(&d_shared_store, sizeof(float)*32);
    // hello_from_device<<<1, 32>>>();
    conflict_kernel<<<1, 32>>>(d_shared_store);
    cudaDeviceSynchronize();
    cudaFree(d_shared_store);
}
