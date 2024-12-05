project(parallel-mandelbrot)

# CMake version
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # at least 3.18 required for icpx

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER icpx) # use Intel C++ compiler

include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND BASE_OPTIONS
    # Enable Warnings
    "-Wall"
    "-Wextra"
    "-Wshadow"
    "-Wfloat-equal"
    "-Wconversion"
    # Enable OpenMP
    "-qopenmp"
    # Enable Debug Symbols
    "-g"
    )

list(APPEND UNOPTIMIZED "-O0")
list(APPEND OPTIMIZED "-O3")
list(APPEND HOST "-O3" "-xHost")
list(APPEND OPTIONS "UNOPTIMIZED" "OPTIMIZED" "HOST")

# Add the executables
file(GLOB SRC_FILES src/*.cpp)

message("All targets:")
foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)
    string(TOUPPER ${FILE_NAME} DEF_NAME) # capitalize the file name
    foreach(OPTION ${OPTIONS})
        string(CONCAT TARGET_NAME ${FILE_NAME} "_" ${OPTION})
        message(${TARGET_NAME})
        add_executable(${TARGET_NAME} main.cpp ${SRC_FILE})
        target_compile_definitions(${TARGET_NAME} PRIVATE ${DEF_NAME}) # add definition
        target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${${OPTION}})
        target_link_libraries(${TARGET_NAME} "-qopenmp")
    endforeach()
endforeach()
