#include <cstring> 
#include <cstdlib> 
#include <iostream>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <omp.h>

// Define the dimensions for the J and A matrices
constexpr int J_DIM = 3;
constexpr int A_DIM = 4;
constexpr int FLOPS_PER_ELEMENT = 69; // Based on prior analysis

/**
 * @brief Template function to run the Finite Element matrix computation benchmark.
 * * @tparam Number The floating-point type (float or double).
 * @tparam Layout The memory layout (LayoutLeft or LayoutRight).
 * @param N_elements The number of element matrices to process.
 */
template <typename Number, typename Layout>
void Run_Benchmark(int N_elements)
{
    // Execution and Memory Space definitions (Portable)
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    // Define Rank-3 Views (Number***) for Host
    using ViewJHost = Kokkos::View<Number ***, Layout, Kokkos::HostSpace>; 
    using ViewAHost = Kokkos::View<Number ***, Layout, Kokkos::HostSpace>; 

    // Allocate Host Views: N x J_DIM x J_DIM (Input) and N x A_DIM x A_DIM (Output)
    ViewJHost J_host("J_host", N_elements, J_DIM, J_DIM);
    ViewAHost A_host("A_host", N_elements, A_DIM, A_DIM);

    // Initialize Jacobians on host (using a simple, non-singular test matrix)
    for (int e = 0; e < N_elements; ++e)
    {
        // Inverse Jacobian entries: [3, 1, 1; 1, 3, 1; 1, 1, 3]
        J_host(e, 0, 0) = 3; J_host(e, 0, 1) = 1; J_host(e, 0, 2) = 1;
        J_host(e, 1, 0) = 1; J_host(e, 1, 1) = 3; J_host(e, 1, 2) = 1;
        J_host(e, 2, 0) = 1; J_host(e, 2, 1) = 1; J_host(e, 2, 2) = 3;
    }

    // --- Device Views ---
    // Define Rank-3 Views for Device in the portable MemSpace
    using ViewJDevice = Kokkos::View<Number ***, Layout, MemSpace>;
    using ViewADevice = Kokkos::View<Number ***, Layout, MemSpace>;

    // Allocate Device Views
    ViewJDevice J_dev("J_dev", N_elements, J_DIM, J_DIM);
    ViewADevice A_dev("A_dev", N_elements, A_DIM, A_DIM);

    // --- H -> D Transfer ---
    Kokkos::Timer timer_h_to_d;
    Kokkos::deep_copy(J_dev, J_host);
    Kokkos::fence();
    double time_h_to_d = timer_h_to_d.seconds();

    // Calculate bytes transferred for H->D (Input only: J_host data)
    size_t bytes_h_to_d = (size_t)N_elements * J_DIM * J_DIM * sizeof(Number);
    double gb_s_h_to_d = (double)bytes_h_to_d / time_h_to_d / 1e9;

    // --- Parallel Kernel (Computation) ---
    Kokkos::Timer timer_comp;

    Kokkos::parallel_for("ComputeElementMatrices", Kokkos::RangePolicy<ExecSpace>(0, N_elements), KOKKOS_LAMBDA(const int e) {
        // Use thread-local fixed-size arrays for register/L1 cache optimization
        Number J_local[J_DIM][J_DIM];
        Number A_local[A_DIM][A_DIM];

        // Copy input from global View to local register array
        for(int i=0; i < J_DIM; i++) {
            for(int j=0; j < J_DIM; j++) {
                J_local[i][j] = J_dev(e, i, j);
            }
        }
        
        // Calculate Cofactors and Determinant Inverse
        Number C0 = J_local[1][1] * J_local[2][2] - J_local[1][2] * J_local[2][1];
        Number C1 = J_local[1][2] * J_local[2][0] - J_local[1][0] * J_local[2][2];
        Number C2 = J_local[1][0] * J_local[2][1] - J_local[1][1] * J_local[2][0];
        Number inv_J_det = J_local[0][0] * C0 + J_local[0][1] * C1 + J_local[0][2] * C2;
        Number d = (Number(1.0)/Number(6.0)) / inv_J_det;

        // Calculate the components of the G-matrix (G = J^-T * J^-1 * d)
        Number G0 = d * (J_local[0][0] * J_local[0][0] + J_local[1][0] * J_local[1][0] + J_local[2][0] * J_local[2][0]);
        Number G1 = d * (J_local[0][0] * J_local[0][1] + J_local[1][0] * J_local[1][1] + J_local[2][0] * J_local[2][1]);
        Number G2 = d * (J_local[0][0] * J_local[0][2] + J_local[1][0] * J_local[1][2] + J_local[2][0] * J_local[2][2]);
        Number G3 = d * (J_local[0][1] * J_local[0][1] + J_local[1][1] * J_local[1][1] + J_local[2][1] * J_local[2][1]);
        Number G4 = d * (J_local[0][1] * J_local[0][2] + J_local[1][1] * J_local[1][2] + J_local[2][1] * J_local[2][2]);
        Number G5 = d * (J_local[0][2] * J_local[0][2] + J_local[1][2] * J_local[1][2] + J_local[2][2] * J_local[2][2]);

        // Populate the 4x4 Element Matrix A (with symmetries)
        A_local[0][0] = G0;       A_local[0][1] = G1;     A_local[0][2] = G2;     A_local[0][3] = -G0-G1-G2;
        A_local[1][0] = G1;       A_local[1][1] = G3;     A_local[1][2] = G4;     A_local[1][3] = -G1-G3-G4;
        A_local[2][0] = G2;       A_local[2][1] = G4;     A_local[2][2] = G5;     A_local[2][3] = -G2-G4-G5;
        A_local[3][0] = A_local[0][3];  A_local[3][1] = A_local[1][3];  A_local[3][2] = A_local[2][3];
        A_local[3][3] = G0 + Number(2.0) * G1 + Number(2.0) * G2 + G3 + Number(2.0) * G4 + G5;

        // Copy output from local register array back to global View
        for(int i=0; i < A_DIM; i++) { 
            for(int j=0; j < A_DIM; j++) {
                A_dev(e, i, j) = A_local[i][j];
            }
        }
    });

    Kokkos::fence();
    double time_comp = timer_comp.seconds();

    // --- D -> H Transfer ---
    Kokkos::Timer timer_d_to_h;
    Kokkos::deep_copy(A_host, A_dev);
    Kokkos::fence();
    double time_d_to_h = timer_d_to_h.seconds();

    // Calculate bytes transferred for D->H (Output only: A_host data)
    size_t bytes_d_to_h = (size_t)N_elements * A_DIM * A_DIM * sizeof(Number);
    double gb_s_d_to_h = (double)bytes_d_to_h / time_d_to_h / 1e9;

    // std::cout << "\nFinal 4x4 A matrix (last element):\n";
    // for(int i=0; i < A_DIM; i++) {
    //     for(int j=0; j < A_DIM; j++) {
    //         std::cout << A_host(N_elements-1, i, j) << "\t";
    //     }
    //     std::cout << "\n";
    // }


    // --- Performance Reporting ---
    const char* layout_str = std::is_same<Layout, Kokkos::LayoutLeft>::value ? "LayoutLeft" : "LayoutRight";
    const char* number_str = std::is_same<Number, double>::value ? "double" : "float";

    // printf("\n--- Benchmark Results for %s, %s ---\n", number_str, layout_str);
    // printf("N = %d elements\n", N_elements);
    // printf("H->D Transfer Time: %f s\n", time_h_to_d);
    // printf("Computation Time:   %f s\n", time_comp);
    // printf("D->H Transfer Time: %f s\n", time_d_to_h);

    double total_flops = (double)N_elements * FLOPS_PER_ELEMENT;
    double m_elements_per_s = (double)N_elements / time_comp / 1e6;
    double gflops_s = total_flops / time_comp / 1e9;

    // Total data processed by the kernel (Input J + Output A)
    size_t bytes_per_element_comp = (size_t)(J_DIM * J_DIM + A_DIM * A_DIM) * sizeof(Number);
    double total_bytes_comp = (double)N_elements * bytes_per_element_comp; 
    double gb_s_comp = total_bytes_comp / time_comp / 1e9; // Total data touched / Comp Time

    printf("%d, %f, %f, %f, %f, %f\n",
       N_elements, m_elements_per_s, gflops_s, gb_s_comp, gb_s_h_to_d, gb_s_d_to_h);
}

/**
 * @brief Runs the benchmark across a range of N_elements, starting at N_start and stopping at N_limit, doubling N_elements each time.
 * @tparam Number The floating-point type (float or double).
 * @tparam Layout The memory layout (LayoutLeft or LayoutRight).
 * @param N_start The starting number of elements (lower bound, inclusive).
 * @param N_limit The ending number of elements (upper bound, inclusive).
 */
template <typename Number, typename Layout>
void Run_Benchmarks(int N_start, int N_limit)
{
    printf("N_elements, M Elements/s, GFlop/s, GB/s(Comp), GB/s(H->D), GB/s(D->H)\n");
    int N_current = N_start;
    
    // N_stride=2: double the size in each iteration
    while (N_current <= N_limit) {
        Run_Benchmark<Number, Layout>(N_current);
        
        // Prevent overflow and ensure N_current is updated correctly
        if (N_current > N_limit / (float) 1.5) {
             N_current = N_limit + 1; // Exit loop after final run
        } else {
            N_current *= (float) 1.5; // Double the element count
        }
    }
}

/**
 * @brief Prints the command line usage information.
 */
void PrintUsage(const char* prog_name) {
    printf("Usage: %s [-N <num>] [-Nlimit <num>] [-dtype <type>] [-layout <type>]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -N <num>       Starting number of elements (default: 1000). Also serves as Nlimit if not specified.\n");
    printf("  -Nlimit <num>  Maximum number of elements (upper bound, inclusive) for the benchmark sweep.\n");
    printf("  -dtype <type>  Data type for floating point numbers (default: float). Options: float, double.\n");
    printf("  -layout <type> Memory layout for Kokkos Views (default: left). Options: left, right.\n");
    printf("  -h             Print this help message.\n");
}

int main(int argc, char *argv[])
{
    // --- Default Values ---
    int N_elements = 1000;
    int N_limit = 0; // Will default to N_elements if not specified
    std::string dtype_str = "float";
    std::string layout_str = "left";
    constexpr int N_MAX = 40000000;

    // --- Command Line Parsing ---
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-h") == 0)
        {
            PrintUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
        {
            N_elements = std::atoi(argv[i + 1]);
            i++; 
        }
        else if (strcmp(argv[i], "-Nlimit") == 0 && i + 1 < argc)
        {
            N_limit = std::atoi(argv[i + 1]);
            i++; 
        }
        else if (strcmp(argv[i], "-dtype") == 0 && i + 1 < argc)
        {
            dtype_str = argv[i + 1];
            i++; 
        }
        else if (strcmp(argv[i], "-layout") == 0 && i + 1 < argc)
        {
            layout_str = argv[i + 1];
            i++; 
        }
    }
    
    // If N_limit was not specified, set it equal to N_elements
    if (N_limit == 0) {
        N_limit = N_elements;
    }

    // --- Validation and Error Handling ---
    if (N_elements <= 0 || N_elements > N_MAX) {
        fprintf(stderr, "Error: -N must be a positive integer, max %d.\n", N_MAX);
        PrintUsage(argv[0]);
        return 1;
    }
    
    if (N_limit < N_elements || N_limit > N_MAX) {
         fprintf(stderr, "Error: -Nlimit must be greater than or equal to -N, and max %d.\n", N_MAX);
        PrintUsage(argv[0]);
        return 1;
    }

    if (dtype_str != "float" && dtype_str != "double") {
        fprintf(stderr, "Error: -dtype must be 'float' or 'double'.\n");
        PrintUsage(argv[0]);
        return 1;
    }
    
    if (layout_str != "left" && layout_str != "right") {
        fprintf(stderr, "Error: -layout must be 'left' or 'right'.\n");
        PrintUsage(argv[0]);
        return 1;
    }

    int n_threads = 16;
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(n_threads);
    Kokkos::initialize(settings);  // no argc/argv needed

    // Use a lambda to handle the template selection based on runtime strings
    auto launch_benchmark = [&]() {
        if (dtype_str == "float") {
            if (layout_str == "left") {
                Run_Benchmarks<float, Kokkos::LayoutLeft>(N_elements, N_limit);
            } else { // layout_str == "right"
                Run_Benchmarks<float, Kokkos::LayoutRight>(N_elements, N_limit);
            }
        } else { // dtype_str == "double"
            if (layout_str == "left") {
                Run_Benchmarks<double, Kokkos::LayoutLeft>(N_elements, N_limit);
            } else { // layout_str == "right"
                Run_Benchmarks<double, Kokkos::LayoutRight>(N_elements, N_limit);
            }
        }
    };
    
    launch_benchmark();

    Kokkos::finalize();
    return 0;
}

