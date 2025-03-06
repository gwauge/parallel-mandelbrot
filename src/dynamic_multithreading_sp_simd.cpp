#include "mandelbrot.hpp"

int64_t mandelbrot_computation(ofstream &matrix_out, bool output)
{
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();
    #pragma omp parallel for simd collapse(2) schedule(dynamic, 100)
    for (uint row = 0; row < HEIGHT; ++row) {
        for (uint col = 0; col < WIDTH; ++col) {
            const complex<float> c((float)(col * STEP + MIN_X), (float)(row * STEP + MIN_Y));

            // z = z^2 + c
            complex<float> z(0, 0);
            for (uint16_t i = 1; i <= ITERATIONS; ++i)
            {
                z = pow(z, 2) + c;

                // If it is convergent
                image[row*WIDTH+col] = i;
                if (abs(z) >= 2)
                {
                    break;
                }
            }
        }
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
