# parallel-mandelbrot
Parallel implementation of mandelbrot. Created as part of the final assignment for the class "High Performance Computing" at UniGe.

![Mandelbrot Set](mandelbrot.png)

## Dependencies

To be able to build all binaries you need a working intel oneAPI basekit installation version 2025.0.4, a working CUDA installation and working AMD ROCm installation. If you only want to build some binaries, you can comment out the includes in the CMakeLists.txt file in the root directory.

## How to run

```bash
source /opt/intel/oneapi/setvars.sh --include-intel-llvm
mkdir build && cd build
cmake ..
make clean
make run
```

## Other scripts

`visualize.py`

Script that reads the output of a binary and creates a picture using matplotlib.

`visualize_opengl.py`
Script that reads the output of a binary and creates a picture using vispy with opengl acceleration for larger interactive rendering.

`benchmarks/`
Scripts to build and run all binaries for CPUs and scripts to generate plots from the gathered data that show speedup/efficiency etc.

`gpu_benchmarks/`
Benchmark script for gpu binaries and scripts to generate plots from that data.

`advisor/`
Script to execute advisor runs with roofline analysis and create snapshot for each binary and csv report for later analysis.

`result_compare/`
Write output files for all binaries and a script to analyze the output to find out which outputs are equal, to check correctness.

## Final report
This repository contains a final report in the file [`final_report.pdf`](final_report.pdf) that explains the implementations and discusses the results.
Data included in the report, such as benchmarking and Intel Advisor reports can be found on the `report-data` branch.
