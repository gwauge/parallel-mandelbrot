project(parallel-mandelbrot-rocm LANGUAGES CXX HIP)
       
list(APPEND BASE_OPTIONS
    "-O3"
    "-ffast-math"
)

set(CMAKE_HIP_ARCHITECTURES gfx1103)
file(GLOB SRC_FILES *.hip)

message("ROCM Targets:")

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)
    #string(TOUPPER ${FILE_NAME} DEF_NAME) # capitalize the file name
    message(${FILE_NAME})
    add_executable(${FILE_NAME} ../../main.cpp ${SRC_FILE})
    target_compile_options(${FILE_NAME} PRIVATE ${BASE_OPTIONS})
    set_target_properties(${FILE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocm)
endforeach()