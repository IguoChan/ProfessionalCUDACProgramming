#include <stdio.h>

__global__ void hello() {
    printf("Hello, CUDA from the GPU!\n");
}

int main() {
    // Launch the kernel with a single block and a single thread
    hello<<<1, 10>>>();

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}