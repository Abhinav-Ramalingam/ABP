#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib> 
#include <fstream>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Kernel: y = A * x
__global__ void matvec_kernel(const int M, const int N, const float *A, const float *x, float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M)
    {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col)
            sum += A[row + M * col] * x[col]; // Column-major access
        y[row] = sum;
    }
}

// Kernel to initialize a vector to a constant
__global__ void set_vector(const int N, const float val, float *x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = val;
}

// Kernel to initialize a matrix to a constant
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

    int blocksY = (M + threadsPerBlock - 1) / threadsPerBlock;
    auto t1 = std::chrono::steady_clock::now();
    matvec_kernel<<<blocksY, threadsPerBlock>>>(M, N, d_A, d_x, d_y);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::vector<float> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Write output to file
    std::string outfname = "mmax1_out.txt";
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

    double bytes = double(M) * N * sizeof(float) + double(N + M) * sizeof(float);
    double bw = bytes / time_sec / 1e9;

    std::cout << "Matrix size: " << M << " x " << N << ", ";
    std::cout << "Time: " << time_sec << " s, Bandwidth: " << bw << " GB/s" << std::endl;

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}


int main() {

    run_case(10, 10); // final exact case

    return 0;
}
