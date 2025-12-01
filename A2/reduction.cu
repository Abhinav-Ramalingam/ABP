#include <iostream>
#include <vector>
#include <cstdlib>  // for atoi

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

// Kernel to set vector elements to a constant
__global__ void set_vector(const int N, const float val, float *x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = val;
}

// Simple reduction kernel: sum elements of 'x' into 'result'
__global__ void reduce_sum_atomic(const float *x, float *result, int N)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: load into shared memory
    sdata[tid] = (idx < N) ? x[idx] : 0.0f;
    __syncthreads();

    // Step 2: parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Step 3: one atomic add per block
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

int main(int argc, char **argv)
{
    // Read N from command line, else default = 10000
    int N = 10000;
    if (argc > 1)
        N = std::atoi(argv[1]);

    std::cout << "Reducing vector of size " << N << std::endl;

    // Allocate vector on GPU
    float *d_v;
    cudaMalloc(&d_v, N * sizeof(float));

    // Fill vector with 1.0
    int n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    set_vector<<<n_blocks, BLOCK_SIZE>>>(N, 1.0f, d_v);

    // Allocate result on GPU
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Launch reduction kernel
    reduce_sum_atomic<<<n_blocks, BLOCK_SIZE>>>(d_v, d_result, N);

    // Copy result back
    float h_result = 0.0f;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print
    std::cout << "Sum of vector elements: " << h_result << std::endl;

    // Cleanup
    cudaFree(d_v);
    cudaFree(d_result);

    return 0;
}
