
#include "mandelbrot.hpp"
#define SIMD_LEN 4
#define ALIGNED_LENGTH ((HEIGHT * WIDTH) / SIMD_LEN + 1) * SIMD_LEN

int64_t mandelbrot_computation(ofstream &matrix_out, bool output)
{

    //uint16_t *const image = new uint16_t[ALIGNED_LENGTH];
    uint16_t* image = (uint16_t*) std::aligned_alloc(sizeof(float) * SIMD_LEN, ALIGNED_LENGTH * sizeof(uint16_t));

    const auto start = chrono::steady_clock::now();

    #pragma omp parallel for simd aligned(image:sizeof(float)*SIMD_LEN) simdlen(SIMD_LEN) safelen(SIMD_LEN) schedule(simd:dynamic, 128)
    for (uint pos = 0; pos < ALIGNED_LENGTH; ++pos) {
        const double c_re = pos % WIDTH * STEP +  MIN_X;
        const double c_im = pos / WIDTH* STEP + MIN_Y;

        double z_re = c_re;
        double z_im = c_im;
        uint16_t iter = 1;

        for (; iter < ITERATIONS; ++iter) {
            double zr2 = z_re * z_re;
            double zi2 = z_im * z_im;
            if (zr2 + zi2 >= 4.0f)
                break;
            double new_z_re = zr2 - zi2 + c_re;
            double new_z_im = 2.0f * z_re * z_im + c_im;
            z_re = new_z_re;
            z_im = new_z_im;
        }
        image[pos] = iter;
    }

    const auto end = chrono::steady_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cerr << "Time elapsed: "
         << difference
         << " milliseconds." << endl;
    if (output) { 
        for (int row = 0; row < HEIGHT; row++)
        {
            for (int col = 0; col < WIDTH; col++)
            {
                matrix_out << image[row * WIDTH + col];

                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << endl;
        }
        matrix_out.close();
    }

    delete[] image; // It's here for coding style, but useless
    return difference;
}
