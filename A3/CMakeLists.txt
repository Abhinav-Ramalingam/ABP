cmake_minimum_required(VERSION 3.10)
project(FE_Kokkos LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to your Kokkos installation
# Change this to install_cpu for CPU build
# set(Kokkos_DIR "${CMAKE_CURRENT_LIST_DIR}/../install_cpu/lib64/cmake/Kokkos")
set(Kokkos_DIR "${CMAKE_CURRENT_LIST_DIR}/../install_gpu/lib64/cmake/Kokkos")
# Find Kokkos package
find_package(Kokkos REQUIRED)

# Create executable
add_executable(fed fed.cpp)

# Link Kokkos
target_link_libraries(fed Kokkos::kokkos)

# Enable Kokkos features if needed
target_compile_definitions(fed PRIVATE USE_KOKKOS)

