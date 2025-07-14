#include "zee_utils.h"
#include "zee_ndarray.h"

__global__ void hello_from_device() {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;

    if (tx==0 && ty==0) {
        printf("[%d, %d]: hello from device\n", tx, ty);
    }

}

int main() {
    printf("Hello World\n");
    hello_from_device<<<1, 32>>>();
    cudaDeviceSynchronize();
}
