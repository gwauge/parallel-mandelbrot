#include <immintrin.h>
#include "mandelbrot.hpp"

#ifdef AVX2_DP_OPTIMIZED

int64_t mandelbrot_computation(ofstream &matrix_out)
{
    //uint64_t *const image = new uint64_t[HEIGHT * WIDTH];
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();

    __m256d xmin = _mm256_set1_pd(MIN_X);
    __m256d ymin = _mm256_set1_pd(MIN_Y);
    __m256d step = _mm256_set1_pd(STEP);

    // 2**2 (We calculate the squared magnitude to eliminate the square root operation)
    __m256d threshold = _mm256_set1_pd(4);
    __m256i one = _mm256_set1_epi64x(1);
    __m256d col_increment = _mm256_set_pd(3, 2, 1, 0);
    __m256d minus_one = _mm256_set1_pd(-1);

    #pragma omp parallel for schedule(dynamic, 100)
    for (int pos = 0; pos < HEIGHT * WIDTH; pos += 4) {
            const int row = pos / WIDTH;
            const int col = pos % WIDTH;

            __m256d mcol = _mm256_add_pd(_mm256_set1_pd(col), col_increment);
            __m256d mrow = _mm256_set1_pd(row);

            __m256d cr = _mm256_add_pd(_mm256_mul_pd(mcol, step), xmin);
            __m256d ci = _mm256_add_pd(_mm256_mul_pd(mrow, step), ymin);
            
            // Skip first iteration because 0**2=0
            __m256d zr = cr;
            __m256d zi = ci;

            __m256d mk = _mm256_set1_epi64x(1);

            for (uint16_t k = 1; k < ITERATIONS; ++k) {
                // z**2 + c
                __m256d zr2 = _mm256_mul_pd(zr, zr);
                __m256d zi2 = _mm256_mul_pd(zi, zi);
                __m256d zrzi = _mm256_mul_pd(zr, zi);
                __m256d mag2 = _mm256_add_pd(zr2, zi2);

                __m256d mask = _mm256_cmp_pd(mag2, threshold, _CMP_LT_OS);
                mk = _mm256_add_epi64(_mm256_and_ps(mask, one), mk);

                // zr1 = zr0 * zr0 - zi0 * zi0 + cr 
                // zi1 = zr0 * zi0 + zr0 * zi0 + ci
                if (_mm256_testz_pd(mask, minus_one)) {
                    break;
                }

                zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
                zi = _mm256_add_pd(_mm256_add_pd(zrzi, zrzi), ci);

            }

            uint16_t *dst = image + row * WIDTH + col; 
            uint16_t *src = (uint16_t*) &mk;

            dst[0] = src[0];
            dst[1] = src[4];
            dst[2] = src[8];
            dst[3] = src[12];
    }

    const auto end = chrono::steady_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cerr << "Time elapsed: "
         << difference
         << " milliseconds." << endl;

    for (uint row = 0; row < HEIGHT; row++)
    {
        for (uint col = 0; col < WIDTH; col++)
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