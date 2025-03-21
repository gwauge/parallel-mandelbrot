#include "mandelbrot.hpp"

#ifdef NO_COMPLEX_DP
int64_t mandelbrot_computation(ofstream &matrix_out, bool output)
{
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int pos = 0; pos < HEIGHT * WIDTH; ++pos)
    {

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const double cr = col * STEP + MIN_X;
        const double ci = row * STEP + MIN_Y;


        // z = z^2 + c
        double zr = 0;
        double zi = 0;

        for (uint16_t k = 1; k <= ITERATIONS; ++k) {
            double zr2 = zr*zr;
            double zi2 = zi*zi;
            double zrzi = zr * zi;

            zr = (zr2 - zi2) + cr;
            zi = (zrzi + zrzi) + ci;
            zr2 = zr*zr;
            zi2 = zi*zi;
            double mag2 = zr2 + zi2;
            image[pos] = k;
            if(mag2 >= 4.0) {
                break;
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
#endif
