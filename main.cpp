#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include "mandelbrot.hpp"

using namespace std;

int main(int argc, char **argv)
{

    ofstream matrix_out;

    if (argc < 2)
    {
        //cerr << "Please specify the output file as a parameter." << endl;
        //return -1;
        cout << mandelbrot_computation(matrix_out, false) << endl;
        return 0;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cerr << "Unable to open file." << endl;
        return -2;
    }

    cout << mandelbrot_computation(matrix_out, true) << endl;

    return 0;
}