#pragma once

#include <iostream>
#include <random>
#include <iomanip>
#include <vector>

#include "zee_utils.h"

namespace zee {

template<typename T>
static __global__ void _init_kernel(T* devPtr, const T value, const size_t N)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; tidx < N; tidx += stride) {
        devPtr[tidx] = value;
    }
}

template<typename T>
static __global__ void _copy_kernel(T* tgtPtr, T* srcPtr, const size_t N)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; tidx < N; tidx += stride) {
        tgtPtr[tidx] = srcPtr[tidx];
    }
}

enum NDDevice {
  CPU,
  GPU
};

template <typename T>
struct NDArray {
  T* data_ = NULL;

  // 最右边的维度变化最快
  std::vector<size_t> shape_;

  NDDevice device_ = CPU;
  size_t len_ = 0;


  NDArray(const std::vector<size_t>& shape, NDDevice device=CPU) {
    initialize(shape, device);
  }

  NDArray(size_t M, size_t N, NDDevice device=CPU) {
    initialize({M, N}, device);
  }

  NDArray(const std::vector<size_t>& shape, T value, NDDevice device=CPU) {
    initialize(shape, device);
    set_value(value);
  }

  NDArray(size_t M, size_t N, T value, NDDevice device=CPU) {
    initialize({M, N}, device);
    set_value(value);
  }

  // 拷贝构造
  NDArray(const NDArray<T>& other) {
    initialize(other.shape, other.device);

    if (device_ == CPU) {
      for (size_t i = 0; i < len_; i++) {
        data_[i] = other.data_[i];
      }
    } else {
      _copy_kernel<<<ceil_div(len_, 256), 256>>>(data_, other.data_, len_);
    }
  }
  NDArray(NDArray<T>&& other) {
    this->data_ = other.data_;
    swap(this->shape_, other.shape_);
    this->device_ = other.device_;
    this->len_ = other.len_;

    other.data_ = NULL;
    other.len_ = 0;
  }

  ~NDArray() {
    if (device_ == CPU) {
      free(data_);
    } else {
      cudaFree(data_);
    }

    data_ = NULL;
    len_ = 0;
  }

  // 打印矩阵
  void print() const {
    MY_ASSERT(shape_.size() <= 2, "print of NDArray only support dimension <= 2");

    if (shape_.size() == 0) {
      std::cout << std::endl;
      return;
    }

    T* data_host_ = NULL;
    if (device_ == CPU) {
      data_host_ = data_;
    } else {
      data_host_ = (T*)malloc(sizeof(T)*len_);
      CUDA_CHECK(cudaMemcpy(data_host_, data_, sizeof(T)*len_, cudaMemcpyDeviceToHost));
    }


    if (shape_.size() == 1) {
      for (size_t i = 0; i < len_; i++) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << data_host_[i] << " ";
      }
    } else if (shape_.size() == 2) {
      for (size_t r = 0; r < shape_[0]; ++r) {
          for (size_t c = 0; c < shape_[1]; ++c) {
              std::cout << std::setw(6) << std::fixed << std::setprecision(2) << data_host_[r*shape_[1] + c] << " ";
          }
          std::cout << std::endl;
      }
    }

    if (device_ == GPU) {
      free(data_host_);
    }
  }

  // oeprator 索引
  T& operator[](size_t i) {
    MY_ASSERT(device_ == CPU, "index not supported for GPU NDArray");
    return data_[i];
  }

  const T& operator[](size_t i) const {
    MY_ASSERT(device_ == CPU, "index not supported for GPU NDArray");
    return data_[i];
  }

  T& operator()(size_t i, size_t j) {
    MY_ASSERT(device_ == CPU, "index not supported for GPU NDArray");
    MY_ASSERT(shape_.size() == 2, "index (i, j) only support for NDArray with dimension of 2");
    return data_[i*shape_[shape_.size()-1] + j];
  }

  const T& operator()(size_t i, size_t j) const {
    MY_ASSERT(device_ == CPU, "index not supported for GPU NDArray");
    MY_ASSERT(shape_.size() == 2, "index (i, j) only support for NDArray with dimension of 2");
    return data_[i*shape_[shape_.size()-1] + j];
  }

  NDArray<T> transpose() {
    MY_ASSERT(device_ == CPU, "transpose not supported for GPU NDArray");
    MY_ASSERT(shape_.size() >= 2, "transpose only supported for dimension of NDArray larger than or equal 2");

    size_t M = shape_[shape_.size()-2];
    size_t N = shape_[shape_.size()-1];

    std::vector<size_t> new_shape;

    size_t stride = M*N;
    size_t num = 1;
    for (size_t i = 0; i < shape_.size()-2  ; i++) {
      num *= shape_[i];
      new_shape.push_back(shape_[i]);
    }
    new_shape.push_back(N);
    new_shape.push_back(M);

    NDArray<T> ret(new_shape, device_);

    for (size_t k = 0; k < num; k++) {
      T* base_data = data_ + k*stride;
      T* o_base_data = ret.data_ + k*stride;
      for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
          o_base_data[j*M + i] = base_data[i*N + j];
        }
      }
    }
    return std::move(ret);
  }

  // 随机初始化矩阵元素
  void random_init(T min_val = 0, T max_val = 100) {
    T* host_data_;
    if (device_ == CPU) {
      host_data_ = data_;
    } else {
      host_data_ = (T*)malloc(sizeof(T)*len_);
    }

    // 使用 C++11 的随机数生成器
    std::random_device rd; // 硬件熵源
    std::mt19937 gen(rd()); // Mersenne Twister 随机数引擎

    // 根据类型选择合适的分布
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> distrib(min_val, max_val);
        for (size_t i = 0; i < len_; ++i) {
            host_data_[i] = distrib(gen);
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        // 浮点数通常在 [min_val, max_val) 之间均匀分布
        std::uniform_real_distribution<T> distrib(min_val, max_val);
        for (size_t i = 0; i < len_; ++i) {
            host_data_[i] = distrib(gen);
        }
    } else {
        std::cerr << "Warning: random_init not supported for this data type." << std::endl;
    }

    if (device_ == GPU) {
      CUDA_CHECK(cudaMemcpy(data_, host_data_, sizeof(T)*len_, cudaMemcpyHostToDevice));
      free(host_data_);
    }
  }

  NDArray<T>& operator= (T val) {
    set_value(val);
    return *this;
  }

  bool operator == (const NDArray& other) {
    static_assert(std::is_integral_v<T>, "operator == olny support for NDArray with integral type");
    MY_ASSERT(device_ == CPU, "operator == not supported for GPU NDArray");

    if (this->shape_ != other.shape_) {
      return false;
    }
    for (size_t i = 0; i < len_; i++) {
      if (this->data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }

  size_t size() {
    return len_;
  }

  inline size_t M() {
    MY_ASSERT(shape_.size() == 2, "M() only support for NDArray with dimension of 2");
    return shape_[shape_.size() - 2];
  }

  inline size_t N() {
    MY_ASSERT(shape_.size() == 2, "N() only support for NDArray with dimension of 2");
    return shape_[shape_.size() - 1];
  }

  NDArray<T> cpu() {
    MY_ASSERT(device_ == GPU, "toCPU() only support GPU NDArray");
    NDArray<T> ret(this->shape_, CPU);
    CUDA_CHECK(cudaMemcpy(ret.data_, data_, sizeof(T)*len_, cudaMemcpyDeviceToHost));
    return std::move(ret);
  }

private:
  inline void initialize(const std::vector<size_t>& shape, NDDevice device) {
    this->shape_ = shape;
    this->device_ = device;

    if (shape.size() == 0) {
      this->len_ = 0;
    } else {
      this->len_ = 1;
      for (auto& i : this->shape_) {
        this->len_ *= i;
      }
    }

    if (device_ == CPU) {
      data_ = (T*)malloc(sizeof(T)*len_);
    } else {
      CUDA_CHECK(cudaMalloc(&data_, sizeof(T)*len_));
    }

  }

  inline void set_value(T value) {
    if (device_ == CPU) {
      for (size_t i = 0; i < len_; i++) {
        data_[i] = value;
      }
    } else {
      _init_kernel<<<ceil_div(len_, 256), 256>>>(data_, value, len_);
    }
  }

};
}
