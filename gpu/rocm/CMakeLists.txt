set(CMAKE_CXX_COMPILER hipcc) # use Intel C++ compiler
        
add_compile_options(-O3 --offload-arch=gfx1103 -DNDEBUG -ffast-math)
add_executable(rocm_sp ../../main.cpp rocm_sp.cpp)
add_executable(rocm_sp_tiled ../../main.cpp rocm_sp_tiled.cpp)
add_executable(rocm_dp ../../main.cpp rocm_dp.cpp)