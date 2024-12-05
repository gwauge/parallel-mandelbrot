=== HPC mandelbrot benchmark ===

echo

# Clean directory
# make  clean

# Compile files
make

# Run the benchmark
echo "Running benchmarks"

# remove old benchmark.csv
if -f benchmark.csv; then
    # ask user if old benchmark.csv should be removed
    echo "Old benchmark.csv found. Remove? (y/n)"
    read remove
    if [ $remove == "y" ]; then
        rm benchmark.csv
        echo "Removed old benchmark.csv"
    else
        echo "Keeping old benchmark.csv and aborting"
        exit 0
    fi
fi

# run each file in bin directory N times for warmup and M times for measurement
# execution time is printed to stdout and should be saved to csv

N=5
M=10
THREADS=1

for file in bin/*; do
    echo "\tRunning $file"
    for i in $(seq 1 $N); do
        $file > /dev/null
    done
    echo "\t\tFinished warmup"
    for i in $(seq 1 $M); do
        # save output to variable
        output=$(OMP_NUM_THREADS=$THREADS $file)
        # append output to csv
        echo "$file,$THREADS,$output$" >> benchmark.csv
    done
done

echo "Benchmark finished"
