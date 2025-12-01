#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include "cublas_v2.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

unsigned long long compute_repeats(int M, int N) {
    unsigned long long total_work_target = 100000000ULL;
    unsigned long long n_repeat = std::max(1ULL, total_work_target / (unsigned long long)(M * N));
    return n_repeat;
}


// Kernel to initialize a vector to a constant
__global__ void set_vector(const int N, const float val, float *x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = val;
}

// Kernel to initialize a matrix (Hilbert)
__global__ void set_matrix(const int M, const int N, float *A)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M * N)
    {
        int row = idx % M;       // row index
        int col = idx / M;       // column index (column-major)
        A[idx] = float(row - col + 1);  // Toeplitx Matrix
    }
}

void run_case(int M, int N) {

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    int threadsPerBlock = BLOCK_SIZE;
    int blocksA = (M * N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksX = (N + threadsPerBlock - 1) / threadsPerBlock;

    set_matrix<<<blocksA, threadsPerBlock>>>(M, N, d_A);
    set_vector<<<blocksX, threadsPerBlock>>>(N, 1.0f, d_x);
    cudaDeviceSynchronize();

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    unsigned long long n_repeat = compute_repeats(M, N);
    auto t1 = std::chrono::steady_clock::now();

    for (unsigned long long rep = 0; rep < n_repeat; ++rep) {
        cublasStatus_t stat = cublasSgemv(
            handle,
            CUBLAS_OP_N,   // non-transposed: y = A * x
            M, N,
            &alpha,
            d_A, M,        // lda = M (since column-major)
            d_x, 1,
            &beta,
            d_y, 1);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS operation failed\n";
            std::abort();
        }
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();


    // Copy result (optional, if you want to verify)
    std::vector<float> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Bandwidth calculation
    // double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double avg_time = total_time / n_repeat;

    
    double bytes = double(M) * N * sizeof(float) + double(N + M) * sizeof(float);
    double bw = bytes / avg_time / 1e9;

    std::cout << "[cuBLAS] Matrix size: " << M << " x " << N << ", ";
    std::cout << "Time: " << avg_time << " s, Bandwidth: " << bw << " GB/s" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}


int main() {
    std::cout << std::fixed << std::setprecision(6);

    for (int M = 100; M <= 10000; M += 300)
        run_case(M, M);

    
    return 0;
}
