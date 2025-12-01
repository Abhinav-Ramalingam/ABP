#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16  // better as 16x16 blocks for 2D kernels
#endif

unsigned long long compute_repeats(int M, int N) {
    unsigned long long total_work_target = 100000000ULL;
    unsigned long long n_repeat = std::max(1ULL, total_work_target / (unsigned long long)(M * N));
    return n_repeat;
}


// Kernel: C = A * B
__global__ void matmul_kernel(int M, int N, int K,
                            const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Co  l in C

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float a = A[row + j * M];   // A(row,j), col-major
            float b = B[j + col * N];   // B(j,col), col-major
            sum += a * b;
        }
        C[row + col * M] = sum; // C(row,col), col-major
    }
}

// Kernel: initialize matrix with a pattern
__global__ void set_matrix(const int M, const int N, float* A) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M * N) {
        int row = idx % M;
        int col = idx / M;
        A[idx] = float(row - col + 1); // Toeplitz-like pattern
    }
}

void run_case(int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    int threads = 256;
    int blocksA = (M * N + threads - 1) / threads;
    int blocksB = (N * K + threads - 1) / threads;

    set_matrix<<<blocksA, threads>>>(M, N, d_A);
    set_matrix<<<blocksB, threads>>>(N, K, d_B);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    unsigned long long n_repeat = compute_repeats(M, N);
    auto t1 = std::chrono::steady_clock::now();

    for (unsigned long long rep = 0; rep < n_repeat; ++rep) 
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    // double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double avg_time = total_time / n_repeat; // average time per kernel

    std::vector<float> h_C(M * K);
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    double flops = 2.0 * M * N * K; // 2 flops per multiply-add
    double gflops = flops / avg_time / 1e9;

    std::cout << M << ", " << gflops << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "MatrixSize,GFlops" << std::endl;
    for (int n = 100; n <= 5000; n += 200)
        run_case(n, n, n);

    return 0;
}
