# CMake version
cmake_minimum_required(VERSION 3.21 FATAL_ERROR) # at least 3.18 required for icpx at least 3.21 required for hip
project(parallel-mandelbrot LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND BASE_OPTIONS
    # Enable Warnings
    "-Wall"
    "-Wextra"
    "-Wshadow"
    "-Wfloat-equal"
    "-Wconversion"
    # Enable Debug Symbols
    "-g"
)

include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Normal build
add_subdirectory(src)

# GPU
add_subdirectory(gpu/cuda)
add_subdirectory(gpu/rocm)
add_subdirectory(gpu/sycl)