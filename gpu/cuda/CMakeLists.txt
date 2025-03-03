set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_ARCHITECTURES sm_75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Xcompiler -fno-exceptions")
project(parallel-mandelbrot-cuda LANGUAGES CXX CUDA)
set(CUDA_ARCHITECTURES sm_75)

list(APPEND BASE_OPTIONS
    "-O3"
    "-ffast-math"
)

add_executable(cuda_sp ../../main.cpp cuda_sp.cu)
target_compile_options(cuda_sp PRIVATE ${BASE_OPTIONS})
target_link_options(cuda_sp PRIVATE ${BASE_OPTIONS})

add_executable(cuda_dp ../../main.cpp cuda_dp.cu)
target_compile_options(cuda_dp PRIVATE ${BASE_OPTIONS})
target_link_options(cuda_dp PRIVATE ${BASE_OPTIONS})