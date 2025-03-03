set(CMAKE_CXX_COMPILER icpx) # use Intel C++ compiler

list(APPEND BASE_OPTIONS 
    # Enable OpenMP
    "-qopenmp"
)

list(APPEND O2 
    "-O2" 
    "-march=core-avx2" 
)
list(APPEND O2-STRICT 
    "-O2" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O2-PRECISE
    "-O2" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O3 
    "-O3" 
    "-march=core-avx2" 
)
list(APPEND O3-STRICT 
    "-O3" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O3-PRECISE
    "-O3" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O1 
    "-O1" 
    "-march=core-avx2" 
)
list(APPEND O1-STRICT 
    "-O1" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O1-PRECISE
    "-O1" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND O0 
    "-O0" 
    "-march=core-avx2" 
)
list(APPEND O0-STRICT 
    "-O0" 
    "-fp-model=strict"
    "-march=core-avx2" 
)
list(APPEND O0-PRECISE
    "-O0" 
    "-fp-model=precise"
    "-march=core-avx2" 
)
list(APPEND OPTIONS 
    "O2"
    "O2-STRICT"
    "O2-PRECISE"
    "O3"
    "O3-STRICT"
    "O3-PRECISE"
    "O1"
    "O1-STRICT"
    "O1-PRECISE"
    "O0"
    "O0-STRICT"
    "O0-PRECISE"
)

# Add the executables
file(GLOB SRC_FILES *.cpp)

message("All targets:")
foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILE_NAME ${SRC_FILE} NAME_WE)
    string(TOUPPER ${FILE_NAME} DEF_NAME) # capitalize the file name
    foreach(OPTION ${OPTIONS})
        string(CONCAT TARGET_NAME ${FILE_NAME} "_" ${OPTION})
        message(${TARGET_NAME})
        add_executable(${TARGET_NAME} ../main.cpp ${SRC_FILE})
        target_compile_definitions(${TARGET_NAME} PRIVATE ${DEF_NAME}) # add definition
        target_compile_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${${OPTION}})
        target_link_options(${TARGET_NAME} PRIVATE ${BASE_OPTIONS} ${${OPTION}})
        target_link_libraries(${TARGET_NAME} "-qopenmp")
    endforeach()
endforeach()