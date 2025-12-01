#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include "cublas_v2.h"

// Kernel to initialize a matrix with a simple pattern (Toeplitz-like)
__global__ void set_matrix(const int M, const int N, float* A) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M * N) {
        int row = idx % M;
        int col = idx / M;
        A[idx] = float(row - col + 1); // simple pattern
    }
}

// Function to run a single matrix-matrix multiplication case
void run_case(int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Initialize matrices on GPU
    int threads = 256;
    int blocksA = (M * N + threads - 1) / threads;
    int blocksB = (N * K + threads - 1) / threads;
    set_matrix<<<blocksA, threads>>>(M, N, d_A);
    set_matrix<<<blocksB, threads>>>(N, K, d_B);
    cudaDeviceSynchronize();

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    auto t1 = std::chrono::steady_clock::now();
    // C = alpha*A*B + beta*C
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, K, N,
                &alpha,
                d_A, M,
                d_B, N,
                &beta,
                d_C, M);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double gflops = 2.0 * M * N * K / time_sec / 1e9;

    std::cout << "Matrix sizes: " << M << " x " << N << " * " << N << " x " << K
              << ", GFLOPS: " << gflops << "\n";

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Sweep benchmark
    for (int n = 100; n <= 5000; n = int(n * 1.5))
        run_case(n, n, n);

    return 0;
}
