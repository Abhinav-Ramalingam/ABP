#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16  // better as 16x16 blocks for 2D kernels
#endif

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

    auto t1 = std::chrono::steady_clock::now();
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    double time_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    std::vector<float> h_C(M * K);
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    double flops = 2.0 * M * N * K; // 2 flops per multiply-add
    double gflops = flops / time_sec / 1e9;

    std::cout << "Matrix sizes: " << M << "x" << N << " * " << N << "x" << K
            << " -> " << M << "x" << K << ", ";
    std::cout << "Time: " << time_sec << " s, "
            << "GFLOPS: " << gflops << std::endl;

    // Write matrix to file in readable format
    std::string outfname = "mmab1_out.txt";
    std::ofstream ofs(outfname);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            ofs << h_C[i + j * M]; // column-major
            if (j < K - 1) ofs << "\t"; // tab between elements
        }
        ofs << "\n"; // newline at end of row
    }
    ofs.close();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    run_case(5, 5, 5);
    return 0;
}
