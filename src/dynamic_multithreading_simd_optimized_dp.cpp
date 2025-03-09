#include "mandelbrot.hpp"

int64_t mandelbrot_computation(ofstream &matrix_out, bool output)
{
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();
    #pragma omp parallel for simd schedule(simd:dynamic, 128)
    for (uint pos = 0; pos < WIDTH * HEIGHT; ++pos) {
        const complex<double> c((double)(pos % WIDTH) *  STEP +  MIN_X, (double)(pos / WIDTH)* STEP +  MIN_Y);

        // z = z^2 + c
        complex<double> z = c;
        uint16_t iter = 1;
        for (; iter < ITERATIONS ; ++iter)
        {
            if (norm(z) >= 4) {
                break;
            }
            z = z * z + c;
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

    //delete[] image; // It's here for coding style, but useless
    return difference;
}
