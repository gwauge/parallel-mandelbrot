# CMake version
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # at least 3.18 required for icpx
project(parallel-mandelbrot)

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

list(APPEND O2 
    "-O2" 
    "-march=core-avx2" 
)
list(APPEND O2_STRICT 
    "-O2" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O2_PRECISE
    "-O2" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O3 
    "-O3" 
    "-march=core-avx2" 
)
list(APPEND O3_STRICT 
    "-O3" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O3_PRECISE
    "-O3" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O1 
    "-O1" 
    "-march=core-avx2" 
)
list(APPEND O1_STRICT 
    "-O1" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O1_PRECISE
    "-O1" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O0 
    "-O0" 
    "-march=core-avx2" 
)
list(APPEND O0_STRICT 
    "-O0" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O0_PRECISE
    "-O0" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND OPTIONS 
    "O2"
    "O2_STRICT"
    "O2_PRECISE"
    "O3"
    "O3_STRICT"
    "O3_PRECISE"
    "O1"
    "O1_STRICT"
    "O1_PRECISE"
    "O0"
    "O0_STRICT"
    "O0_PRECISE"
)

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
