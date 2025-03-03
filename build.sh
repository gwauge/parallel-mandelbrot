#!/bin/bash
set -e

# Script to build the project

# check if inside CI environment
if [ ! -n "$CI" ]; then
    source /opt/intel/oneapi/setvars.sh --include-intel-llvm --force
fi

# Check if script is run from root directory
if [ ! -f "main.cpp" ]; then
    echo "Please run this script from the root directory"
    exit 1
fi

# Check if build directory exists
if [ -d "build" ]; then
    rm -r build
fi

mkdir build

# Change to build directory
cd build || exit

# Run cmake
cmake ..

# Compile files
make -j
