#include <stdio.h>
#include <algorithm>
#include "zee_utils.h"

struct SMInfo {
    int cudaCoresPerSM;
    int tensorCoresPerSM;
    const char* note;
};

static std::string lower(const std::string &s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c){ return std::tolower(c); });
    return r;
}

// 返回估算值：每个 SM 的 CUDA core 数和 Tensor Core 数（若不确定返回 0）
SMInfo getSMInfo(int major, int minor, const char* deviceNameC) {
    std::string deviceName = deviceNameC ? deviceNameC : "";
    std::string name = lower(deviceName);

    // 默认返回（如果没有匹配项会是 0,0）
    SMInfo info{0,0,"unknown"};

    // Hopper (H100) -- CC 9.0: 128 CUDA cores/SM, 4 Tensor Cores/SM (4th-gen TC).
    if (major == 9) {
        info = {128, 4, "Hopper-family (9.x) - e.g. H100 (ref: NVIDIA H100 docs)"};
        return info;
    }

    // Ampere / Ada / Turing / Volta / Pascal 等（major == 8 / 7 / 6 / 5 / 3）
    if (major == 8) {
        // handle common minors
        if (minor == 0) {
            // GA100 (A100): 64 CUDA cores/SM, 4 Tensor Cores/SM.（A100 whitepaper）.
            info = {64, 4, "Ampere GA100 (A100) - 64 cores/SM, 4 TC/SM"};
            return info;
        }
        if (minor == 6) {
            // GA10x (consumer Ampere, e.g. RTX 30 series) -> commonly 128 CUDA cores/SM, 4 TC/SM
            info = {128, 4, "Ampere GA10x (RTX30 series) - ~128 cores/SM, 4 TC/SM"};
            return info;
        }
        if (minor == 9) {
            // Ada Lovelace (compute capability 8.9): 128 CUDA cores/SM, 4 Tensor Cores/SM.（Ada whitepaper）
            info = {128, 4, "Ada Lovelace (8.9) - 128 cores/SM, 4 TC/SM"};
            return info;
        }
        // other 8.x (fallback): try to guess from name
        if (name.find("a100") != std::string::npos || name.find("ga100") != std::string::npos) {
            return {64, 4, "Ampere GA100 (detected by name)"};
        }
        if (name.find("rtx") != std::string::npos || name.find("3090") != std::string::npos ||
            name.find("3080") != std::string::npos || name.find("3070") != std::string::npos) {
            return {128, 4, "Ampere GA10x (detected by name)"};
        }
        // fallback conservative guess
        info = {64, 4, "Ampere (unknown minor) - guessed 64/SM (use device name or whitepaper to refine)"};
        return info;
    }

    if (major == 7) {
        // Volta (7.0): V100 -> 64 cores/SM, Tensor cores exist (gen1) – often 8 TC/SM (Volta/Turing gen)
        if (minor == 0) { info = {64, 8, "Volta (7.0) - V100 style (64 cores/SM, 8 TC/SM)"}; return info; }
        // Turing (7.5): many Turing SMs have 64 CUDA cores/SM and 8 Tensor Cores/SM
        if (minor == 5) { info = {64, 8, "Turing (7.5) - consumer/server Turing"}; return info; }
        // generic 7.x fallback
        info = {64, 8, "7.x family (Volta/Turing) - guessed 64 cores/SM, 8 TC/SM"};
        return info;
    }

    if (major == 6) {
        // Pascal family: GP100 had 64 cores/SM; many GP1xx/GP10x consumer chips have 128/SM.
        // We'll attempt to distinguish by device name:
        if (name.find("p100") != std::string::npos || name.find("gp100") != std::string::npos) {
            info = {64, 0, "Pascal GP100 (P100) - 64 cores/SM (server)"}; return info;
        }
        // Consumer Pascal GPUs (GP104 etc) had 128 cores/SM
        if (name.find("gtx") != std::string::npos || name.find("geforce") != std::string::npos) {
            info = {128, 0, "Pascal consumer (GP104 etc) - 128 cores/SM"}; return info;
        }
        // conservative fallback: 64
        info = {64, 0, "Pascal (6.x) fallback - 64 cores/SM"};
        return info;
    }

    if (major == 5) {
        // Maxwell: commonly 128 CUDA cores/SM (no Tensor Cores)
        info = {128, 0, "Maxwell (5.x) - 128 cores/SM (no tensor cores)"};
        return info;
    }

    if (major == 3) {
        // Kepler: commonly 192 CUDA cores/SM (no tensor cores)
        info = {192, 0, "Kepler (3.x) - 192 cores/SM"};
        return info;
    }

    // older or unknown -- leave zeros (caller should handle)
    return info;
}


int main() {
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
              &sm_count, cudaDevAttrMultiProcessorCount, dev_id));
    // 30个sm
    printf("sm_count: %d\n", sm_count);

    int max_smem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
            &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // 每个sm有100KB的共享内存
    printf("max_smem_per_sm: %d(Bytes)\n", max_smem_per_sm);

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

    SMInfo smInfo = getSMInfo(prop.major, prop.minor, prop.name);

    int totalCudaCores = smInfo.cudaCoresPerSM * prop.multiProcessorCount;
    int totalTensorCores = smInfo.tensorCoresPerSM * prop.multiProcessorCount;

    printf("Per SM: %d CUDA cores, %d Tensor Cores\n",
           smInfo.cudaCoresPerSM, smInfo.tensorCoresPerSM);
    printf("Total : %d CUDA cores, %d Tensor Cores\n",
           totalCudaCores, totalTensorCores);

        // 主频，单位 MHz，需要转成 GHz
    double baseClockGHz = prop.clockRate / 1e6;

    // CUDA Core 理论 FP32 算力
    // 每 CUDA Core 每周期 2 FLOPs (FMA = mul+add)
    double fp32TFLOPS = totalCudaCores * 2.0 * baseClockGHz / 1e3;

    // Tensor Core 理论 FP16 算力 (简化：每 Tensor Core 每周期 64 FLOPs)
    // 实际值依架构不同，这里用常见估算公式
    double fp16TensorTFLOPS = totalTensorCores * 128.0 * baseClockGHz / 1e3;

    printf("Base Clock (reported): %.2f GHz\n", baseClockGHz);
    printf("Theoretical FP32 (CUDA cores): %.2f TFLOPS\n", fp32TFLOPS);
    printf("Theoretical FP16 (Tensor Cores): %.2f TFLOPS\n", fp16TensorTFLOPS);

}
