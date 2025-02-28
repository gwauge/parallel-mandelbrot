#include <immintrin.h>
#include "mandelbrot.hpp"

#ifdef AVX2_SP

int64_t mandelbrot_computation(ofstream &matrix_out, bool output)
{
    uint16_t *const image = new uint16_t[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();

    __m256 xmin = _mm256_set1_ps(MIN_X);
    __m256 ymin = _mm256_set1_ps(MIN_Y);
    __m256 step = _mm256_set1_ps((float)STEP);

    // 2**2 (We calculate the squared magnitude to eliminate the square root operation)
    __m256 threshold = _mm256_set1_ps(4);
    __m256 one = _mm256_set1_ps(1);
    __m256 col_increment = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
    __m256 minus_one = _mm256_set1_ps(-1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int pos = 0; pos < HEIGHT * WIDTH; pos += 8) {
            const int row = pos / WIDTH;
            const int col = pos % WIDTH;

            __m256 mcol = _mm256_add_ps(_mm256_set1_ps((float)col), col_increment);
            __m256 mrow = _mm256_set1_ps((float)row);

            __m256 cr = _mm256_add_ps(_mm256_mul_ps(mcol, step), xmin);
            __m256 ci = _mm256_add_ps(_mm256_mul_ps(mrow, step), ymin);
            
            // Skip first iteration because 0**2=0
            __m256 zr = _mm256_set1_ps(0);
            __m256 zi = _mm256_set1_ps(0);

            __m256 mk = _mm256_set1_ps(1);

            for (uint16_t k = 1; k < ITERATIONS; ++k) {
                // z**2 + c
                __m256 zr2 = _mm256_mul_ps(zr, zr);
                __m256 zi2 = _mm256_mul_ps(zi, zi);
                __m256 zrzi = _mm256_mul_ps(zr, zi);

                // zr1 = zr0 * zr0 - zi0 * zi0 + cr 
                // zi1 = zr0 * zi0 + zr0 * zi0 + ci
                zr = _mm256_add_ps(_mm256_sub_ps(zr2, zi2), cr);
                zi = _mm256_add_ps(_mm256_add_ps(zrzi, zrzi), ci);

                // Increment k 
                zr2 = _mm256_mul_ps(zr, zr);
                zi2 = _mm256_mul_ps(zi, zi);
                __m256 mag2 = _mm256_add_ps(zr2, zi2);

                __m256 mask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
                mk = _mm256_add_ps(_mm256_and_ps(mask, one), mk);

                if (_mm256_testz_ps(mask, minus_one)) {
                    break;
                }
            }

            __m256i pixel = _mm256_cvtps_epi32(mk);
            uint16_t *dst = image + row * WIDTH + col; 
            uint16_t *src = (uint16_t*) &pixel;

            dst[0] = src[0];
            dst[1] = src[2];
            dst[2] = src[4];
            dst[3] = src[6];
            dst[4] = src[8];
            dst[5] = src[10];
            dst[6] = src[12];
            dst[7] = src[14];
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