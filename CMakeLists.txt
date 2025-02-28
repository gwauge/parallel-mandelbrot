# CMake version
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # at least 3.18 required for icpx
project(parallel-mandelbrot)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Normal build
add_subdirectory(src)
