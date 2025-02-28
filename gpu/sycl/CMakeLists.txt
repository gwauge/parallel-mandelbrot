set(CMAKE_CXX_COMPILER icpx) # use Intel C++ compiler

add_executable(sycl_cpu_sp ../../main.cpp sycl_sp.cpp)
target_compile_options(sycl_cpu_sp PRIVATE -fsycl)
target_link_options(sycl_cpu_sp PRIVATE -fsycl)

add_executable(sycl_cpu_dp ../../main.cpp sycl_dp.cpp)
target_compile_options(sycl_cpu_dp PRIVATE -fsycl)
target_link_options(sycl_cpu_dp PRIVATE -fsycl)

# AMD ROCM target
add_executable(sycl_rocm_sp ../../main.cpp sycl_sp.cpp)
target_link_options(sycl_rocm_sp PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1103)
target_compile_options(sycl_rocm_sp PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1103)

add_executable(sycl_rocm_dp ../../main.cpp sycl_dp.cpp)
target_link_options(sycl_rocm_dp PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1103)
target_compile_options(sycl_rocm_dp PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1103)