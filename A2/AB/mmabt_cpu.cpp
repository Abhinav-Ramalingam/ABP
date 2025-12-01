#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath> // For abs() for a hypothetical check

// Function to calculate C = A * B on the CPU
void matmul_cpu(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C)
{
    // A is M x N, B is N x K, C is M x K
    // All matrices are assumed to be stored in Column-Major order (matching CUDA)

    for (int col = 0; col < K; ++col) {
        for (int row = 0; row < M; ++row) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                // Access A[row, j] (Column-Major)
                float a = A[row + j * M];
                
                // Access B[j, col] (Column-Major)
                float b = B[j + col * N];
                
                sum += a * b;
            }
            // Write C[row, col] (Column-Major)
            C[row + col * M] = sum;
        }
    }
}

// Function to initialize a matrix with the Toeplitz-like pattern
void set_matrix_cpu(int M, int N, std::vector<float>& A) {
    A.resize(M * N);
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < M; ++row) {
            int idx = row + col * M;
            A[idx] = float(row - col + 1);
        }
    }
}

void run_case(int M, int N, int K) {
    std::vector<float> h_A, h_B, h_C;

    // 1. Initialize Matrices
    set_matrix_cpu(M, N, h_A);
    set_matrix_cpu(N, K, h_B);
    h_C.resize(M * K, 0.0f); // Initialize C to zero

    // 2. Timing the Matrix Multiplication
    auto t1 = std::chrono::steady_clock::now();
    
    matmul_cpu(M, N, K, h_A, h_B, h_C);
    
    auto t2 = std::chrono::steady_clock::now();

    double time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    
    // 3. Performance Calculation
    double flops = 2.0 * M * N * K; // 2 FLOPs per multiply-add
    double gflops = flops / time_sec / 1e9;

    // Output M and GFLOPS for plotting
    std::cout << M << ", " << gflops << std::endl;
}


int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "MatrixSize,GFlops" << std::endl; 
    
    // Run case for square matrices from 100 to 5000 with 200 increments
    for (int n = 100; n <= 5000; n += 200)
        run_case(n, n, n);

    return 0;
}

