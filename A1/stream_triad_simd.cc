
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vectorization.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void compute_triad(const std::size_t N,
                   const float       a,
                   const float *     x,
                   const float *     y,
                   float *           z)
{
  constexpr unsigned int simd_width = VectorizedArray<float>::n_array_elements;
  // round N down to nearest multiple of SIMD width
  const std::size_t N_regular = N / simd_width * simd_width;
#pragma omp parallel for
  for (std::size_t i = 0; i < N_regular; i += simd_width)
    {
      VectorizedArray<float> x_vec, y_vec;
      x_vec.load(x + i);
      y_vec.load(y + i);
      const VectorizedArray<float> z_vec = a * x_vec + y_vec;
      z_vec.store(z + i);
    }

  // remainder
  for (std::size_t i = N_regular; i < N; ++i)
    z[i] = a * x[i] + y[i];
}


float *get_aligned_vector_pointer(const bool          align,
                                  const unsigned int  N,
                                  std::vector<float> &x)
{
  const unsigned int simd_length = VectorizedArray<float>::n_array_elements;
  x.reserve(N + simd_length - 1);
  // allocate entries on NUMA nodes
#pragma omp parallel for
  for (unsigned int i = 0; i < N + simd_length - 1; ++i)
    x[i] = 0;
  x.resize(N + simd_length - 1);

  if (align == false)
    return x.data();
  else
    {
      const std::size_t alignment_offset =
        reinterpret_cast<std::size_t>(x.data()) % (simd_length * sizeof(float));
      return x.data() +
             (simd_length - alignment_offset / sizeof(float)) % simd_length;
    }
}


// Run the actual benchmark
void benchmark_triad(const bool        align,
                     const std::size_t N,
                     const long long   repeat)
{
  std::vector<float> v1_vec, v2_vec, v3_vec;
  float *            v1 = get_aligned_vector_pointer(align, N, v1_vec);
  float *            v2 = get_aligned_vector_pointer(align, N, v2_vec);
  float *            v3 = get_aligned_vector_pointer(align, N, v3_vec);

  for (unsigned int i = 0; i < N; ++i)
    v1[i] = static_cast<float>(rand()) / RAND_MAX;
  for (unsigned int i = 0; i < N; ++i)
    v2[i] = static_cast<float>(rand()) / RAND_MAX;

  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
        compute_triad(N, 13.f, v1, v2, v3);
      const volatile float result = v3[0] + v3[N - 1];

      // prevent compiler to report about an unused variable
      (void)result;

      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8)
            << 1e-9 * 3 * sizeof(float) * N / best << " GB/s" << std::endl;
}

int main(int argc, char **argv)
{
  if (argc % 2 == 0)
    {
      std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << "-min 100 -max 1e8 -repeat -1" << std::endl;
      std::abort();
    }

  long N_min  = 8;
  long N_max  = -1;
  bool align  = false;
  long repeat = -1;
  // parse from the command line
  for (int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-min")
        N_min = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-max")
        N_max = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-repeat")
        repeat = std::atoll(argv[l + 1]);
      else if (option == "-align")
        align = std::atoi(argv[l + 1]);
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }
  if (N_min < 1)
    {
      std::cout << "Expected positive size for min argument, got " << N_min
                << std::endl;
      return 0;
    }

  if (N_max < N_min)
    N_max = N_min;

  const unsigned int n_vect_floats = VectorizedArray<float>::n_array_elements;
  const unsigned int n_vect_bits   = 8 * sizeof(float) * n_vect_floats;

  std::cout << "Intrinsics-based vectorization over " << n_vect_floats
            << " floats = " << n_vect_bits << " bits"
#ifdef _OPENMP
            << " with " << omp_get_max_threads() << " OpenMP threads "
#endif
            << std::endl;

  for (long n = N_min; n <= N_max; n = (1 + n * 1.1))
    {
      // round up to nearest multiple of 8
      n = (n + 7) / 8 * 8;
      benchmark_triad(align, n, repeat);
    }

  return 0;
}
