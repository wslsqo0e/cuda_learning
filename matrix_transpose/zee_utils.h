#pragma once
#include <chrono>   // C++11 high-resolution timer
#include <string>
#include <vector>
#include <iostream>
#include <numeric>


#include <cuda_runtime.h>

#define MY_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + message + \
                                     " (" __FILE__ ":" + std::to_string(__LINE__) + ")"); \
        } \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace zee {

inline size_t ceil_div(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// 定义一个别名，方便使用高精度时钟
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds; // 以纳秒为单位进行计时，确保精度

// 函数模板：用于测量任意函数F的执行时间
// F: 要测量的函数类型
// Args: 函数F的参数类型
template <typename F, typename... Args>
double profile_function(std::string prefix_str,
                        int warmup_iterations,
                        int measure_iterations,
                        F&& func,
                        Args&&... args) {

    // 1. Warmup 阶段
    // std::cout << "Starting warmup for " << warmup_iterations << " iterations..." << std::endl;
    for (int i = 0; i < warmup_iterations; ++i) {
        func(std::forward<Args>(args)...); // 调用函数，注意使用 std::forward 完美转发参数
    }
    // std::cout << "Warmup finished." << std::endl;

    // 2. 测量阶段
    std::vector<double> durations_ms; // 存储每次执行的毫秒数

    // std::cout << "Starting measurement for " << measure_iterations << " iterations..." << std::endl;
    for (int i = 0; i < measure_iterations; ++i) {
        TimePoint start_time = Clock::now(); // 记录开始时间
        func(std::forward<Args>(args)...); // 调用函数
        TimePoint end_time = Clock::now();   // 记录结束时间

        // 计算持续时间并转换为毫秒
        Duration duration_ns = std::chrono::duration_cast<Duration>(end_time - start_time);
        durations_ms.push_back(duration_ns.count() / 1000000.0); // 纳秒转毫秒
    }
    // std::cout << "Measurement finished." << std::endl;

    // 3. 统计结果
    double total_duration_ms = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
    double average_duration_ms = total_duration_ms / measure_iterations;

    // (可选) 计算中位数和标准差，提供更全面的统计
    // std::sort(durations_ms.begin(), durations_ms.end());
    // double median_duration_ms = durations_ms[measure_iterations / 2];

    // double sum_sq_diff = 0.0;
    // for (double d : durations_ms) {
    //     sum_sq_diff += (d - average_duration_ms) * (d - average_duration_ms);
    // }
    // double std_dev_ms = std::sqrt(sum_sq_diff / measure_iterations);

    std::cout << "------------------------------------------" << std::endl;
    // std::cout << prefix_str << ": Total measured duration: " << total_duration_ms << " ms" << std::endl;
    std::cout << prefix_str << ": Average duration over " << measure_iterations << " runs: " << average_duration_ms << " ms" << std::endl;
    // std::cout << "Median duration: " << median_duration_ms << " ms" << std::endl;
    // std::cout << "Standard deviation: " << std_dev_ms << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return average_duration_ms;
}

template <typename F, typename... Args>
double profile_cuda_func(std::string prefix_str,
                         int warmup_iterations,
                         int measure_iterations,
                         F&& func,
                         Args&&... args) {

    // 1. Warmup 阶段
    // std::cout << "Starting warmup for " << warmup_iterations << " iterations..." << std::endl;
    for (int i = 0; i < warmup_iterations; ++i) {
        func(std::forward<Args>(args)...); // 调用函数，注意使用 std::forward 完美转发参数
    }
    cudaDeviceSynchronize();
    // std::cout << "Warmup finished." << std::endl;

    // 2. 测量阶段
    std::vector<double> durations_ms; // 存储每次执行的毫秒数
    float ms;
    cudaEvent_t startEvent, stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // std::cout << "Starting measurement for " << measure_iterations << " iterations..." << std::endl;
    for (int i = 0; i < measure_iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(startEvent,0));
        func(std::forward<Args>(args)...); // 调用函数
        CUDA_CHECK(cudaEventRecord(stopEvent,0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));

        durations_ms.push_back(ms); // 纳秒转毫秒
    }
    // std::cout << "Measurement finished." << std::endl;

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // 3. 统计结果
    double total_duration_ms = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
    double average_duration_ms = total_duration_ms / measure_iterations;

    // (可选) 计算中位数和标准差，提供更全面的统计
    // std::sort(durations_ms.begin(), durations_ms.end());
    // double median_duration_ms = durations_ms[measure_iterations / 2];

    // double sum_sq_diff = 0.0;
    // for (double d : durations_ms) {
    //     sum_sq_diff += (d - average_duration_ms) * (d - average_duration_ms);
    // }
    // double std_dev_ms = std::sqrt(sum_sq_diff / measure_iterations);

    std::cout << "------------------------------------------" << std::endl;
    // std::cout << prefix_str << ": Total measured duration: " << total_duration_ms << " ms" << std::endl;
    std::cout << prefix_str << ": Average duration over " << measure_iterations << " runs: " << average_duration_ms << " ms" << std::endl;
    // std::cout << "Median duration: " << median_duration_ms << " ms" << std::endl;
    // std::cout << "Standard deviation: " << std_dev_ms << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return average_duration_ms;
}

}
