#!/bin/bash
set -e

# Script to build the project

source /opt/intel/oneapi/setvars.sh

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
