#!/bin/bash
set -e
# ===== HPC mandelbrot benchmark =====

# Check if script is run from within benchmarks directory
if [ ! -f "benchmark.sh" ]; then
    echo "Please run this script from within the benchmarks directory"
    exit 1
fi

# change to root directory
cd .. || exit

# === Build project ===
echo "Building mandelbrot benchmark"

# check if --skip-build flag is set
if [ "$1" == "--skip-build" ]; then
    echo "Skipping build"
else
    bash build.sh # run build script
fi

# === Run benchmarks ===
echo "Running benchmarks"

cd benchmarks || exit

BENCHMARK_FILE="benchmark.csv"
./benchmark.py --output $BENCHMARK_FILE -y

echo "Benchmark finished"


# === Generate plots ===
echo "Generating plots"

source ../.venv/bin/activate
python generate_plots.py --filename $BENCHMARK_FILE

echo "Plots generated"


# === Exit ===
echo "Exiting"
exit 0
