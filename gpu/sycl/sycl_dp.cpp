#include <mandelbrot.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int64_t mandelbrot_computation(ofstream &matrix_out, bool output) {
    // Allocate image storage (iteration counts)
    uint16_t* const image = new uint16_t[HEIGHT * WIDTH];

    // Start timing the computation.
    auto start = chrono::steady_clock::now();

    // Create a SYCL queue using the SYCL 2020 default selector.
    queue q(default_selector_v);

    {
        // Create a buffer over the image array.
        buffer<uint16_t, 1> image_buf(image, range<1>(HEIGHT * WIDTH));

        // Submit the kernel to the queue.
        q.submit([&](handler &h) {
            // Get write access to the buffer.
            auto image_acc = image_buf.get_access<access::mode::write>(h);

            // Launch a 2D kernel for each pixel.
            h.parallel_for(range<2>(HEIGHT, WIDTH), [=](id<2> idx) {
                int row = idx[0];
                int col = idx[1];

                // Map pixel coordinate to a point in the complex plane.
                // Using macros from mandelbrot.hpp for MIN_X, STEP, etc.
                double cr = col * STEP + MIN_X;
                double ci = row * STEP + MIN_Y;
                int iter = 0;
                double zr = 0.0f, zi = 0.0f;

                // Compute Mandelbrot iterations: z = z^2 + c.
                for (int i = 1; i <= ITERATIONS; i++) {
                    double new_zr = zr * zr - zi * zi + cr;
                    double new_zi = 2.0f * zr * zi + ci;
                    zr = new_zr;
                    zi = new_zi;
                    iter = i;
                    if (zr * zr + zi * zi >= 4.0f)
                        break;
                }
                // Write the iteration count for the current pixel.
                image_acc[row * WIDTH + col] = iter;
            });
        }); // End of queue submission—buffer destructor will wait for completion.
    }

    // End timing.
    auto end = chrono::steady_clock::now();
    int64_t difference = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cerr << "Time elapsed: " << difference << " milliseconds." << std::endl;

    if (output) {
        // Write the computed image to the output stream (CSV format).
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                matrix_out << image[row * WIDTH + col];
                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << "\n";
        }
        matrix_out.close();
    }

    delete[] image; // Free allocated memory.
    return difference;
}