#include <mandelbrot.hpp>
#include <hip/hip_runtime.h>

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

// Kernel that computes Mandelbrot for a tile of the image.
// The tile's upper-left corner is specified by (tile_row_offset, tile_col_offset).
__global__ void mandelbrot_kernel_tile(uint16_t* __restrict__ image,
                                       uint image_width,
                                       uint image_height,
                                       uint tile_row_offset,
                                       uint tile_col_offset) {
    // Compute local thread indices within the tile.
    uint local_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint local_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Compute the corresponding global image coordinates.
    uint global_x = tile_col_offset + local_x;
    uint global_y = tile_row_offset + local_y;
    
    if (global_x < image_width && global_y < image_height) {
        // Map pixel to complex plane.
        float cr = (float) global_x * (float) STEP + (float) MIN_X;
        float ci = (float) global_y * (float) STEP + (float) MIN_Y;
        uint16_t iter = 1;
        float zr = 0.0f, zi = 0.0f;
        
        // Compute the Mandelbrot iterations.
        for (; iter < ITERATIONS; ++iter) {
            float new_zr = zr * zr - zi * zi + cr;
            float new_zi = 2.0f * zr * zi + ci;
            zr = new_zr;
            zi = new_zi;
            if (zr * zr + zi * zi >= 4.0f)
                break;
        }
        // Write result into the global image.
        image[global_y * image_width + global_x] = iter;
    }
}

int64_t mandelbrot_computation(ofstream &matrix_out, bool output) {
    const size_t num_pixels = WIDTH * HEIGHT;

    uint16_t* h_image = new uint16_t[num_pixels];
    auto start = steady_clock::now();

    
    // Allocate device memory for the output image.
    uint16_t* d_image;
    HIP_CALL(hipMalloc(&d_image, num_pixels * sizeof(uint16_t)));
    
    // Choose a tile size (adjustable as needed).
    const int tile_width = 1024;
    const int tile_height = 1024;
    
    // Block dimensions for each tile kernel launch.
    dim3 localBlockDim(16, 16);
    
    
    // Loop over the image by tiles.
    for (int row_offset = 0; row_offset < HEIGHT; row_offset += tile_height) {
        for (int col_offset = 0; col_offset < WIDTH; col_offset += tile_width) {
            // Compute current tile's actual size (for edge cases).
            uint32_t current_tile_width = (uint32_t) std::min(tile_width, WIDTH - col_offset);
            uint32_t current_tile_height = (uint32_t) std::min(tile_height, HEIGHT - row_offset);
            
            // Compute grid dimensions for the current tile.
            dim3 localGridDim((current_tile_width + localBlockDim.x - 1) / localBlockDim.x,
                         (current_tile_height + localBlockDim.y - 1) / localBlockDim.y);
            
            // Launch the kernel for this tile.
            hipLaunchKernelGGL(mandelbrot_kernel_tile,
                               localGridDim, localBlockDim, 0, 0,
                               d_image, WIDTH, HEIGHT, row_offset, col_offset);
            //HIP_CALL(hipDeviceSynchronize());
        }
    }
    
    // Copy the computed image from device to host.
    HIP_CALL(hipMemcpy(h_image, d_image, num_pixels * sizeof(uint16_t), hipMemcpyDeviceToHost));
    
    auto end = steady_clock::now();
    int64_t elapsed = duration_cast<milliseconds>(end - start).count();
    std::cerr << "Time elapsed: " << elapsed << " milliseconds." << std::endl;
    
    if (output) {
        // Write the image as a CSV file.
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
    
    return elapsed;
}
