#!/bin/bash
set -e
# ===== HPC mandelbrot benchmark =====

# Check if script is run from within benchmarks directory
if [ ! -f "benchmark.sh" ]; then
    echo "Please run this script from within the benchmarks directory"
    exit 1
fi

cd .. || exit

# === Build project ===
echo "Building mandelbrot benchmark"

source /opt/intel/oneapi/setvars.sh

if [ ! -d "build" ]; then
    mkdir build
fi

# Change to build directory
cd build || exit

# Run cmake
cmake ..

# Compile files
make -j

# Change back to root directory
cd .. || exit


# === Run benchmarks ===
echo "Running benchmarks"

cd benchmarks || exit

BENCHMARK_FILE="benchmark.csv"

# remove old benchmark.csv
if [ -f $BENCHMARK_FILE ]; then
    # ask user if old benchmark.csv should be removed
    echo "Old $BENCHMARK_FILE found. Remove? (y/n)"
    read remove
    if [ $remove == "y" ]; then
        rm $BENCHMARK_FILE
        echo "Removed old $BENCHMARK_FILE"

        # Create new benchmark.csv
        echo "file,threads,time" >> $BENCHMARK_FILE
    else
        echo "Keeping old $BENCHMARK_FILE and aborting"
        exit 0
    fi
else
    echo "file,threads,time" >> $BENCHMARK_FILE
fi

# run each file in bin directory N times for warmup and M times for measurement
# execution time is printed to stdout and should be saved to csv

N=5 # num warmup runs
M=10 # num measurement runs

for file in ../build/bin/*; do
    # extract filename from path
    filename=$(basename $file)

    echo -e "\tRunning $filename"

    # run each executable with different thread count
    # for THREADS in 1 2 4 8 16; do
    for THREADS in 4 8 16; do
        echo -e "\t\tThread count: $THREADS"
        for _ in $(seq 1 $N); do
            OMP_NUM_THREADS=$THREADS $file result > /dev/null
        done

        echo -e "\t\t\tFinished warmup"
        for _ in $(seq 1 $M); do
            # save output to variable, redirect stderr to /dev/null
            output=$(OMP_NUM_THREADS=$THREADS $file result 2>/dev/null)
            # append output to csv
            echo "$filename,$THREADS,$output" >> $BENCHMARK_FILE
        done
    done
done

echo "Benchmark finished"


# === Generate plots ===
echo "Generating plots"

source .venv/bin/activate
python generate_plots.py --filename $BENCHMARK_FILE

echo "Plots generated"


# === Exit ===
echo "Exiting"
exit 0
