#pragma once

#include "zee_utils.h"

static __global__ void mat_transpose_kernel_v0(const float* idata, float* odata, int M, int N) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 对一个warp来说，idata的读取是连续的(连续16个)，odata的写入是非顺序的
    if (y < M && x < N) {
        odata[x * M + y] = idata[y * N + x];
    }
}

static void mat_transpose_v0(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(zee::ceil_div(N, BLOCK_SZ), zee::ceil_div(M, BLOCK_SZ));
    mat_transpose_kernel_v0<<<grid, block>>>(idata, odata, M, N);
}

template<int BLOCK_SZ>
static __global__ void mat_transpose_kernel_v1(const float* idata, float* odata, int M, int N) {
    const int tdx = threadIdx.x;
    const int tdy = threadIdx.y;

    const int xi = blockIdx.x * blockDim.x + tdx;
    const int yi = blockIdx.y * blockDim.y + tdy;

    __shared__ float smem[BLOCK_SZ][BLOCK_SZ];
    if (yi < M && xi < N)
      smem[tdy][tdx] = idata[yi*N + xi];
    __syncthreads();

    const int xo = blockIdx.x * blockDim.x + tdy;
    const int yo = blockIdx.y * blockDim.y + tdx;

    // bank-conflict 分析:
    // tdx: 0~15, tdy 0, tdx: 0~15, tdy 1 属于一个warp
    // 访问 smem + tdx*16+tdy 的地址，只要tdx差2，则访问地址差32，会导致bank-conflict
    // 所以总共有16路的banck-confict
    // 有个padding解决方案， 初始化 smem[BLOCK_SZ][BLOCK_SZ+1]
    // 访问地址变为 smem + tdx*17+tdy，一般不会出现bank-conflict了，但laneID==0 和 laneID==31 的情况
    // 访问地址 smem+0 和 smem+15*17+1 = smem+256，还是有一路bank-conflict，可以忽略
    // 参见 mat_transpose_kernel_v2
    if (xo < N && yo < M)
      odata[xo*M + yo] = smem[tdx][tdy];
}

static void mat_transpose_v1(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(zee::ceil_div(N, BLOCK_SZ), zee::ceil_div(M, BLOCK_SZ));
    mat_transpose_kernel_v1<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}

template<int BLOCK_SZ>
static __global__ void mat_transpose_kernel_v2(const float* idata, float* odata, int M, int N) {
    const int tdx = threadIdx.x;
    const int tdy = threadIdx.y;

    const int xi = blockIdx.x * blockDim.x + tdx;
    const int yi = blockIdx.y * blockDim.y + tdy;

    __shared__ float smem[BLOCK_SZ][BLOCK_SZ+1];
    if (yi < M && xi < N)
      smem[tdy][tdx] = idata[yi*N + xi];
    __syncthreads();

    const int xo = blockIdx.x * blockDim.x + tdy;
    const int yo = blockIdx.y * blockDim.y + tdx;

    // bank-conflict 分析:
    // tdx: 0~15, tdy 0, tdx: 0~15, tdy 1 属于一个warp
    // 访问 smem + tdx*16+tdy 的地址，只要tdx差2，则访问地址差32，会导致bank-conflict
    // 所以总共有16路的banck-confict
    // 有个padding解决方案， 初始化 smem[BLOCK_SZ][BLOCK_SZ+1]
    // 访问地址变为 smem + tdx*17+tdy，一般不会出现bank-conflict了，但laneID==0 和 laneID==31 的情况
    // 访问地址 smem+0 和 smem+15*17+1 = smem+256，还是有一路bank-conflict，可以忽略
    // 参见 mat_transpose_kernel_v2
    if (xo < N && yo < M)
      odata[xo*M + yo] = smem[tdx][tdy];
}

static void mat_transpose_v2(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(zee::ceil_div(N, BLOCK_SZ), zee::ceil_div(M, BLOCK_SZ));
    mat_transpose_kernel_v2<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}

// 增加每个线程处理的元素数量
template<int BLOCK_SZ, int NUM_PER_THREAD>
static __global__ void mat_transpose_kernel_v3(const float* idata, float* odata, int M, int N) {
    const int tdx = threadIdx.x;
    const int tdy = threadIdx.y;

    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;

    const int row_stride = BLOCK_SZ/NUM_PER_THREAD;

    __shared__ float smem[BLOCK_SZ][BLOCK_SZ+1];

    if (bx+tdx < N) {
        #pragma unroll
        for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
            int row = tdy+z;
            if (by+row < M) {
                smem[row][tdx] = idata[(by+row)*N + bx+tdx];
            }
        }
    }
    __syncthreads();

    // 最好画图对照着看下，这里记住 smem 已经被填满了，然后如何让每个线程操作四个元素的搬运
    // 一个循环内，warp搬运的global memory需要时连续的
    // 性能提升非常明显：因为线程执行指令变多了，增加了计算强度，数据搬运和计算之间更加平衡了
    // 一次搬运的数据更多了，体现为pipeline中空泡变少了
    if (by+tdx < M) {
        #pragma unroll
        for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
            int row = tdy+z;
            if (bx+row < N) {
                odata[(bx+row)*M + by+tdx] = smem[tdx][row];
            }
        }
    }
}

static void mat_transpose_v3(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    constexpr int NUM_PER_TRHEAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_TRHEAD);
    dim3 grid(zee::ceil_div(N, BLOCK_SZ), zee::ceil_div(M, BLOCK_SZ));
    mat_transpose_kernel_v3<BLOCK_SZ, NUM_PER_TRHEAD><<<grid, block>>>(idata, odata, M, N);
}

// 在v3的基础上减少分支判断
// 当一个block明显在矩阵内部时，无需额外的边界判断
// 实际判断一个block是否在矩阵内部，也引入了额外的判断，相比之下分支条件倒是多了
// 性能相比v3有所下降
template<int BLOCK_SZ, int NUM_PER_THREAD>
static __global__ void mat_transpose_kernel_v4(const float* idata, float* odata, int M, int N) {
    const int tdx = threadIdx.x;
    const int tdy = threadIdx.y;

    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;

    const int row_stride = BLOCK_SZ/NUM_PER_THREAD;

    __shared__ float smem[BLOCK_SZ][BLOCK_SZ+1];

    if (bx+tdx < N) {
        if (by+BLOCK_SZ <= M) {  // 整个block的行部分在矩阵内，但这个判断每个线程都需要执行，感觉总的判断数没有减少
            #pragma unroll
            for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
                int row = tdy+z;
                smem[row][tdx] = idata[(by+row)*N + bx+tdx];
            }
        } else {
            #pragma unroll
            for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
                int row = tdy+z;
                if (by+row < M) {
                    smem[row][tdx] = idata[(by+row)*N + bx+tdx];
                }
            }
        }

    }
    __syncthreads();

    // 最好画图对照着看下，这里记住 smem 已经被填满了，然后如何让每个线程操作四个元素的搬运
    // 一个循环内，warp搬运的global memory需要时连续的
    // 性能提升非常明显：因为线程执行指令变多了，增加了计算强度，数据搬运和计算之间更加平衡了
    // 一次搬运的数据更多了，体现为pipeline中空泡变少了
    if (by+tdx < M) {
        if (bx+BLOCK_SZ <= M) {   // 整个block的行部分在矩阵内
            #pragma unroll
            for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
                int row = tdy+z;
                odata[(bx+row)*M + by+tdx] = smem[tdx][row];
            }
        } else {
            #pragma unroll
            for (int z = 0; z < BLOCK_SZ; z+=row_stride) {
                int row = tdy+z;
                if (bx+row < N) {
                    odata[(bx+row)*M + by+tdx] = smem[tdx][row];
                }
            }
        }

    }
}

static void mat_transpose_v4(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    constexpr int NUM_PER_TRHEAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_TRHEAD);
    dim3 grid(zee::ceil_div(N, BLOCK_SZ), zee::ceil_div(M, BLOCK_SZ));
    mat_transpose_kernel_v4<BLOCK_SZ, NUM_PER_TRHEAD><<<grid, block>>>(idata, odata, M, N);
}
