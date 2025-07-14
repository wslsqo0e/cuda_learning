#include "matrix_transpose.cuh"
#include "zee_ndarray.h"

void check_transpose_correction_v0() {
  const int M = 32;
  const int N = 18;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);
  mat_transpose_v0(A.data_, B.data_, M, N);

  zee::NDArray<float> A_C = A.cpu();
  zee::NDArray<float> B_C = B.cpu();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (A_C(i, j) != B_C(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("check_transpose_correction_v0 success!\n");
}


void check_transpose_correction_performance_v0() {
  const int M = 2300;
  const int N = 1500;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);

  double ms = zee::profile_cuda_func("v0", 2, 5, mat_transpose_v0, A.data_, B.data_, M, N);
  printf("Bandwidth: %.2f GB/s\n\n", M*N*4*2/(ms/1000)/1024/1024/1024);
}

void check_transpose_correction_v1() {
  const int M = 32;
  const int N = 18;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);
  mat_transpose_v1(A.data_, B.data_, M, N);

  zee::NDArray<float> A_C = A.cpu();
  zee::NDArray<float> B_C = B.cpu();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (A_C(i, j) != B_C(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("check_transpose_correction_v1 success!\n");
}

void check_transpose_correction_performance_v1() {
  const int M = 2300;
  const int N = 1500;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);

  double ms = zee::profile_cuda_func("v1", 2, 5, mat_transpose_v1, A.data_, B.data_, M, N);

  printf("Bandwidth: %.2f GB/s\n\n", M*N*4*2/(ms/1000)/1024/1024/1024);
}

void check_transpose_correction_v2() {
  const int M = 32;
  const int N = 18;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);
  mat_transpose_v2(A.data_, B.data_, M, N);

  zee::NDArray<float> A_C = A.cpu();
  zee::NDArray<float> B_C = B.cpu();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (A_C(i, j) != B_C(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("check_transpose_correction_v2 success!\n");
}

void check_transpose_correction_performance_v2() {
  const int M = 2300;
  const int N = 1500;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);

  double ms = zee::profile_cuda_func("v2", 2, 5, mat_transpose_v2, A.data_, B.data_, M, N);

  printf("Bandwidth: %.2f GB/s\n\n", M*N*4*2/(ms/1000)/1024/1024/1024);
}

void check_transpose_correction_v3() {
  const int M = 32;
  const int N = 18;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);
  mat_transpose_v3(A.data_, B.data_, M, N);

  zee::NDArray<float> A_C = A.cpu();
  zee::NDArray<float> B_C = B.cpu();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (A_C(i, j) != B_C(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("check_transpose_correction_v3 success!\n");
}

void check_transpose_correction_performance_v3() {
  const int M = 2300;
  const int N = 1500;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);

  double ms = zee::profile_cuda_func("v3", 2, 5, mat_transpose_v3, A.data_, B.data_, M, N);

  printf("Bandwidth: %.2f GB/s\n\n", M*N*4*2/(ms/1000)/1024/1024/1024);
}

void check_transpose_correction_v4() {
  const int M = 32;
  const int N = 18;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);
  mat_transpose_v4(A.data_, B.data_, M, N);

  zee::NDArray<float> A_C = A.cpu();
  zee::NDArray<float> B_C = B.cpu();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (A_C(i, j) != B_C(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("check_transpose_correction_v4 success!\n");
}

void check_transpose_correction_performance_v4() {
  const int M = 2300;
  const int N = 1500;
  zee::NDArray<float> A(M, N, zee::GPU);
  zee::NDArray<float> B(N, M, zee::GPU);

  double ms = zee::profile_cuda_func("v4", 2, 5, mat_transpose_v4, A.data_, B.data_, M, N);

  printf("Bandwidth: %.2f GB/s\n\n", M*N*4*2/(ms/1000)/1024/1024/1024);
}

int main() {
  check_transpose_correction_v0();
  check_transpose_correction_performance_v0();

  check_transpose_correction_v1();
  check_transpose_correction_performance_v1();

  check_transpose_correction_v2();
  check_transpose_correction_performance_v2();

  check_transpose_correction_v3();
  check_transpose_correction_performance_v3();

  check_transpose_correction_v4();
  check_transpose_correction_performance_v4();

  printf("Hello World\n");
}
