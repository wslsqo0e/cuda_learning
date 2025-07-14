#include <stdio.h>
#include "zee_utils.h"

int main() {
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
              &sm_count, cudaDevAttrMultiProcessorCount, dev_id));
    // 30个sm
    printf("sm_count: %d\n", sm_count);

    int max_smem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
            &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // 每个sm有100KB的共享内存
    printf("max_smem_per_sm: %d\n", max_smem_per_sm);

    int max_regs_num_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
            &max_regs_num_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev_id));
    // 65536个32bit寄存器 4*65536=256KB
    printf("max_regs_num_per_sm: %d\n", max_regs_num_per_sm);

    int max_thread_num_per_block = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
            &max_thread_num_per_block, cudaDevAttrMaxThreadsPerBlock, dev_id));
    // 1024
    printf("max_thread_num_per_block: %d\n", max_thread_num_per_block);

    int max_thread_num_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
            &max_thread_num_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev_id));
    // 1536 大约 48*32
    printf("max_thread_num_per_sm: %d\n", max_thread_num_per_sm);
}
