#include "mandelbrot.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>

#ifdef DEBUG_HIP
#define HIP_CALL(call) do {                                                    \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
        std::cerr << "HIP error in " << #call << " at " << __FILE__ << ":"      \
                  << __LINE__ << " : " << hipGetErrorString(err) << std::endl; \
        exit(err);                                                             \
    }                                                                          \
} while(0)
#else
#define HIP_CALL(call) (void)(call)
#endif

using namespace std;
using namespace std::chrono;

// Adapted kernel using a 2D grid, __restrict__ for the pointer, and precomputed values.
__global__ void mandelbrot_kernel_2D(uint16_t* __restrict__ image) {
    // Compute 2D thread indices.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < HEIGHT && col < WIDTH) {
        double cr = col * STEP + MIN_X;
        double ci = row * STEP + MIN_Y;
        int iter = 1;
        double zr = 0.0f, zi = 0.0f;

        // Compute Mandelbrot iterations: z = z^2 + c.
        for (; iter <= ITERATIONS; ++iter) {
            double new_zr = zr * zr - zi * zi + cr;
            double new_zi = 2.0f * zr * zi + ci;
            zr = new_zr;
            zi = new_zi;
            if (zr * zr + zi * zi >= 4.0f)
                break;
        }
        // Write the iteration count for the current pixel.
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
    HIP_CALL(hipMalloc(&d_image, num_pixels * sizeof(uint16_t)));
    
    // Set up a 2D grid: using 16x16 threads per block (256 threads per block).
    dim3 blockDim(8, 8);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);
    
    
    // Launch the optimized kernel.
    hipLaunchKernelGGL(mandelbrot_kernel_2D, gridDim, blockDim, 0, 0, d_image);
    
    // Synchronize to ensure kernel execution is complete.
    HIP_CALL(hipDeviceSynchronize());
    
    // Copy the result back to host.
    HIP_CALL(hipMemcpy(h_image, d_image, num_pixels * sizeof(uint16_t), hipMemcpyDeviceToHost));
    
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
    
    HIP_CALL(hipFree(d_image));
    delete[] h_image;
    
    return difference;
}
