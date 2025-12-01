#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

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

    auto t1 = std::chrono::steady_clock::now();
    matvec_transpose_kernel<<<blocksY, threadsY>>>(M, N, d_A, d_x, d_y);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    std::vector<float> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Write output to file
    std::string outfname = "mmatx1_out.txt";
    std::ofstream ofs(outfname);
    if (!ofs) {
        std::cerr << "Failed to open " << outfname << " for writing\n";
    } else {
        ofs.setf(std::ios::scientific);
        ofs.precision(8);
        for (int i = 0; i < M; ++i) ofs << h_y[i] << "\n";
        ofs.close();
        std::cout << "Wrote " << M << " values to " << outfname << std::endl;
    }

    double bytes = double(M) * N * sizeof(float) + double(M + N) * sizeof(float);
    double bw = bytes / time_sec / 1e9;

    std::cout << "Matrix size: " << M << " x " << N << ", ";
    std::cout << "Time: " << time_sec << " s, Bandwidth: " << bw << " GB/s" << std::endl;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main()
{
    std::cout << std::fixed << std::setprecision(6);

    run_case(5000, 5000); // final larger case
    return 0;
}
