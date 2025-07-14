#include "zee_utils.h"
#include "zee_ndarray.h"
#include <cuda_fp16.h>

union __align__(4) HalfPack {
    uint32_t u32;
    half h[2];
};

__global__ void ld_matrix_with_bank_conflict() {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;

    __shared__ half smem[32*64];  // 每行刚好是128Bytes，占据32个Bank

    if (tx==0 && ty==0) {
        for (int i = 0; i < 32*64; i++) {
            smem[i] = __float2half(static_cast<float>(i));
        }
    }
    __syncthreads();

    HalfPack regs[4];
    // 取smem左上角部分[16, 16]大小的数据
    half* smem_ptr = smem + (tx%16)*64 + (tx/16)*8;
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // TO: R1(0, 1), R2(512, 513), R3(8, 9), R4(520, 521)
    // 分析其bank conflict 情况
    // 一次加载至多128Bytes，所以至少4个wavefronts, T0~T7, T8~T15, T16~T23, T24~T32
    // T0~T7，每次加载都有bank conflict (其加载的行地址都是偏差128Bytes的整数倍)
    // 所以T0~T7 需要8次 wavefronts
    // 其它thread group类似，总计需要 32次 wavefronts
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(regs[0].u32), "=r"(regs[1].u32), "=r"(regs[2].u32), "=r"(regs[3].u32)
                 : "r"(smem_int_ptr));

    printf("[%d, %d]: R1(%.0f, %.0f), R2(%.0f, %.0f), R3(%.0f, %.0f), R4(%.0f, %.0f)\n", tx, ty,
          __half2float(regs[0].h[0]), __half2float(regs[0].h[1]),
          __half2float(regs[1].h[0]), __half2float(regs[1].h[1]),
          __half2float(regs[2].h[0]), __half2float(regs[2].h[1]),
          __half2float(regs[3].h[0]), __half2float(regs[3].h[1]));
}

int main() {
    printf("Hello World\n");
    ld_matrix_with_bank_conflict<<<1, 32>>>();
    cudaDeviceSynchronize();
}
