set(CMAKE_CXX_COMPILER icpx) # use Intel C++ compiler

list(APPEND BASE_OPTIONS
    "-fsycl"
    "-O3"
)

list(APPEND ROCM_OPTIONS
    "-fsycl-targets=amdgcn-amd-amdhsa"
    "-Xsycl-target-backend"
    "--offload-arch=gfx1103"
)

list(APPEND CUDA_OPTIONS
    "-fsycl-targets=nvptx64-nvidia-cuda"
)

list(APPEND MULTI_OPTIONS
    "-fsycl-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa"
    "-Xsycl-target-backend=amdgcn-amd-amdhsa"
    "--offload-arch=gfx1103"
)

file(GLOB SRC_FILES *.cpp)

message("SYCL Targets:")

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)

    string(CONCAT TARGET_NAME ${FILE_NAME} "_cpu")
    message(${TARGET_NAME})
    add_executable(${TARGET_NAME} ../../main.cpp ${SRC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS})
    target_link_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl)

    string(CONCAT TARGET_NAME ${FILE_NAME} "_rocm")
    message(${TARGET_NAME})
    add_executable(${TARGET_NAME} ../../main.cpp ${SRC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${ROCM_OPTIONS})
    target_link_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${ROCM_OPTIONS})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl)

    string(CONCAT TARGET_NAME ${FILE_NAME} "_cuda")
    message(${TARGET_NAME})
    add_executable(${TARGET_NAME} ../../main.cpp ${SRC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${CUDA_OPTIONS})
    target_link_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${CUDA_OPTIONS})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl)

    string(CONCAT TARGET_NAME ${FILE_NAME} "_multi")
    message(${TARGET_NAME})
    add_executable(${TARGET_NAME} ../../main.cpp ${SRC_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${MULTI_OPTIONS})
    target_link_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${MULTI_OPTIONS})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl)

endforeach()