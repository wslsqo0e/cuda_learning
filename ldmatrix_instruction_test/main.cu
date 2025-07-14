#include "zee_utils.h"
#include "zee_ndarray.h"
#include <cuda_fp16.h>

__global__ void hello_from_device() {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;

    if (tx==0 && ty==0) {
        printf("[%d, %d]: hello from device\n", tx, ty);
    }
}

union __align__(4) HalfPack {
    uint32_t u32;
    half h[2];
};

__global__ void ld_matrix_test() {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;

    __shared__ half smem[16*16];

    if (tx==0 && ty==0) {
        for (int i = 0; i < 256; i++) {
            smem[i] = __float2half(static_cast<float>(i));
        }
    }
    __syncthreads();

    HalfPack regs[4];
    half* smem_ptr = smem + (tx%16)*16 + (tx/16)*8;
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // TO: R1(0, 1), R2(128, 129), R3(8, 9), R4(136, 137)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(regs[0].u32), "=r"(regs[1].u32), "=r"(regs[2].u32), "=r"(regs[3].u32)
                 : "r"(smem_int_ptr));
    // 添加trans会按照列的方向加载，同一个寄存器中的两个half元素也不在挨着了
    // T0: R1(0, 16), R2(128, 144), R3(8, 24), R4(136, 152)
    //- asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    //-              : "=r"(regs[0].u32), "=r"(regs[1].u32), "=r"(regs[2].u32), "=r"(regs[3].u32)
    //-              : "r"(smem_int_ptr));

    printf("[%d, %d]: R1(%.0f, %.0f), R2(%.0f, %.0f), R3(%.0f, %.0f), R4(%.0f, %.0f)\n", tx, ty,
          __half2float(regs[0].h[0]), __half2float(regs[0].h[1]),
          __half2float(regs[1].h[0]), __half2float(regs[1].h[1]),
          __half2float(regs[2].h[0]), __half2float(regs[2].h[1]),
          __half2float(regs[3].h[0]), __half2float(regs[3].h[1]));
}

int main() {
    printf("Hello World\n");
    hello_from_device<<<1, 32>>>();
    ld_matrix_test<<<1, 32>>>();
    cudaDeviceSynchronize();
}
