#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#ifndef RESOLUTION
#define RESOLUTION 1000
#endif

#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#ifndef ITERATIONS
// Maximum 65535 due to datatype uint16_t
#define ITERATIONS 1000 // Maximum number of iterations
#endif

using namespace std;

int64_t mandelbrot_computation(ofstream &matrix_out, bool output);
