
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

void compute_triad(const std::size_t N,
                   const float       a,
                   const float *     x,
                   const float *     y,
                   float *           z)
{
#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i)
    z[i] = a * x[i] + y[i];
}


// Run the actual benchmark
void benchmark_triad(const unsigned long N, const long long repeat)
{
  std::vector<float> v1(N), v2(N), v3(N);

  for (std::size_t i = 0; i < N; ++i)
    v1[i] = static_cast<float>(rand()) / RAND_MAX;
  for (std::size_t i = 0; i < N; ++i)
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
        compute_triad(N, 13.f, v1.data(), v2.data(), v3.data());
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
                << "-min 100 -max 1e8 -repeat -1 -align 0" << std::endl;
      std::abort();
    }

#ifdef _OPENMP
  std::cout << "Using " << omp_get_max_threads() << " OpenMP threads "
            << std::endl;
#endif

  long N_min  = 8;
  long N_max  = -1;
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

  for (long n = N_min; n <= N_max; n = (1 + n * 1.1))
    {
      // round up to nearest multiple of 8
      n = (n + 7) / 8 * 8;
      benchmark_triad(n, repeat);
    }

  return 0;
}


