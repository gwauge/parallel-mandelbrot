#include "mandelbrot.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>

#ifdef DEBUG_CUDA
#define CUDA_CALL(call) do {                                                  \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":"    \
                  << __LINE__ << " : " << cudaGetErrorString(err) << std::endl;\
        exit(err);                                                            \
    }                                                                         \
} while(0)
#else
#define CUDA_CALL(call) (void)(call)
#endif

using namespace std;
using namespace std::chrono;

// Kernel: compute Mandelbrot set using a 2D grid.
__global__ void mandelbrot_kernel_2D(uint16_t* __restrict__ image) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < HEIGHT && col < WIDTH) {
        float cr = col * STEP + MIN_X;
        float ci = row * STEP + MIN_Y;
        uint16_t iter = 1;
        float zr = 0.0f, zi = 0.0f;

        // Compute Mandelbrot iterations: z = z^2 + c.
        for (; iter <= ITERATIONS; iter++) {
            float new_zr = zr * zr - zi * zi + cr;
            float new_zi = 2.0f * zr * zi + ci;
            zr = new_zr;
            zi = new_zi;
            if (zr * zr + zi * zi >= 4.0f)
                break;
        }
        image[row * WIDTH + col] = iter;
    }
}

int64_t mandelbrot_computation(ofstream &matrix_out, bool output) {
    // Allocate host memory.
    const size_t num_pixels = HEIGHT * WIDTH;
    uint16_t *h_image = new uint16_t[num_pixels];
    auto start = steady_clock::now();

    // Allocate device memory.
    uint16_t *d_image;
    CUDA_CALL(cudaMalloc(&d_image, num_pixels * sizeof(uint16_t)));

    // Set up a 2D grid: using 8x8 threads per block.
    dim3 blockDim(8, 8);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch the kernel using CUDA triple chevron syntax.
    mandelbrot_kernel_2D<<<gridDim, blockDim>>>(d_image);

    // Synchronize to ensure kernel execution is complete.
    CUDA_CALL(cudaDeviceSynchronize());

    // Copy the result back to host.
    CUDA_CALL(cudaMemcpy(h_image, d_image, num_pixels * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    auto end = steady_clock::now();
    int64_t difference = duration_cast<milliseconds>(end - start).count();
    cerr << "Time elapsed: " << difference << " milliseconds." << endl;

    if (output) {
        // Write the computed matrix to the output stream (CSV format).
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                matrix_out << h_image[row * WIDTH + col];
                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << "\n";
        }
        matrix_out.close();
    }

    CUDA_CALL(cudaFree(d_image));
    delete[] h_image;

    return difference;
}
