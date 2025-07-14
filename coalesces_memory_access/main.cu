#include "zee_utils.h"

// 访问偏移 s 的global memory 地址
template<typename T>
__global__ void offset(T* a, int s)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

// 访问stride 为 s 的 global memory 地址
template<typename T>
__global__ void stride(T* a, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template<typename T>
void runTest(int deviceId, int nMB) {
  int blockSize = 256;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;

  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  CUDA_CHECK(cudaMalloc(&d_a, n * 33 * sizeof(T)));
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));
  printf("Offset, Bandwidth (GB/s):\n");

  // warm_up
  offset<<<n/256, 256>>>(d_a, 0);

  for (int i = 0; i <= 32; i++) {
    CUDA_CHECK(cudaMemset(d_a, 0.0, n * 33 * sizeof(T)));

    CUDA_CHECK( cudaEventRecord(startEvent,0));
    offset<<<n/256, 256>>>(d_a, i);
    CUDA_CHECK(cudaEventRecord(stopEvent,0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");
  stride<<<n/256, 256>>>(d_a, 1); // warm up

  for (int i = 1; i <= 32; i++) {
    CUDA_CHECK(cudaMemset(d_a, 0.0, n * 33 * sizeof(T)));

    CUDA_CHECK( cudaEventRecord(startEvent,0));
    stride<<<n/256, 256>>>(d_a, i);
    CUDA_CHECK(cudaEventRecord(stopEvent,0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));
  cudaFree(d_a);
}


int main(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaSetDevice(deviceId));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);

  printf("%s Precision\n", bFp64 ? "Double" : "Single");

  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}
