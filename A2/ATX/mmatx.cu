#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

unsigned long long compute_repeats(int M, int N) {
    unsigned long long total_work_target = 100000000ULL;
    unsigned long long n_repeat = std::max(1ULL, total_work_target / (unsigned long long)(M * N));
    return n_repeat;
}

// Kernel: y = A^T * x
__global__ void matvec_transpose_kernel(
    const int M, const int N, 
    const float* A, const float* x, float* y)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int col = blockIdx.x;

    if (col >= N) return; // one block per column

    float sum = 0.0f;

    // Each thread processes a strided set of rows
    for (int row = tid; row < M; row += blockDim.x)
        sum += A[row + M * col] * x[row];

    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 writes final result to global memory
    if (tid == 0)
        atomicAdd(&y[col], sdata[0]);
}

// Kernel to initialize a vector
__global__ void set_vector(const int N, const float val, float* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = val;
}

// Kernel to initialize a matrix (column-major)
__global__ void set_matrix(const int M, const int N, float* A)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M * N)
    {
        int row = idx % M;
        int col = idx / M;
        A[idx] = float(row - col + 1); // simple non-symmetric matrix
    }
}

// Host function
void run_case(int M, int N)
{
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float)); // size matches rows of A
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemset(d_y, 0, N * sizeof(float)); // important for atomicAdd

    int threadsPerBlock = BLOCK_SIZE;
    int blocksA = (M * N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksX = (M + threadsPerBlock - 1) / threadsPerBlock;

    set_matrix<<<blocksA, threadsPerBlock>>>(M, N, d_A);
    set_vector<<<blocksX, threadsPerBlock>>>(M, 1.0f, d_x);
    cudaDeviceSynchronize();

    // One block per column
    dim3 blocksY(N);
    dim3 threadsY(threadsPerBlock);

    unsigned long long n_repeat = compute_repeats(M, N);
    auto t1 = std::chrono::steady_clock::now();

    for (unsigned long long rep = 0; rep < n_repeat; ++rep) 
        matvec_transpose_kernel<<<blocksY, threadsY>>>(M, N, d_A, d_x, d_y);

    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    // double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double avg_time = total_time / n_repeat; // average time per kernel

    std::vector<float> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    double bytes = double(M) * N * sizeof(float) + double(M + N) * sizeof(float);
    double bw = bytes / avg_time / 1e9;

    std::cout << "Matrix size: " << M << " x " << N << ", ";
    std::cout << "Time: " << avg_time << " s, Bandwidth: " << bw << " GB/s" << std::endl;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main()
{
    std::cout << std::fixed << std::setprecision(6);

    for (int M = 200; M <= 5000; M += 160)
        run_case(M, M);

    return 0;
}
