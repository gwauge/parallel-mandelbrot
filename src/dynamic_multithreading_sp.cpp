#include "mandelbrot.hpp"

#ifdef DYNAMIC_MULTITHREADING_SP
int64_t mandelbrot_computation(ofstream &matrix_out)
{
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int pos = 0; pos < HEIGHT * WIDTH; ++pos)
    {

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<float> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<float> z(0, 0);
        for (int i = 1; i <= ITERATIONS; ++i)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            image[pos] = i;
            if (abs(z) >= 2)
            {
                break;
            }
        }
    }
    const auto end = chrono::steady_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cerr << "Time elapsed: "
         << difference
         << " milliseconds." << endl;
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

    delete[] image; // It's here for coding style, but useless
    return difference;
}
#endif