project(parallel-mandelbrot)

# CMake version
cmake_minimum_required(VERSION 3.20 FATAL_ERROR) # at least 3.20 required for icpx

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER icpx) # use Intel C++ compiler

# specify binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

add_compile_options(
    # warnings
    -Wall -Wextra -Wshadow -Wfloat-equal -Wconversion
    # optimization
    -qopenmp
    -O3
    -xHost
)

link_libraries(-qopenmp)

# Add the executables
file(GLOB SRC_FILES src/*.cpp)

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)
    add_executable(${FILE_NAME} main.cpp ${SRC_FILE})
    string(TOUPPER ${FILE_NAME} DEF_NAME) # capitalize the file name
    target_compile_definitions(${FILE_NAME} PRIVATE ${DEF_NAME}) # add definition
endforeach()
