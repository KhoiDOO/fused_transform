cmake_minimum_required(VERSION 3.18)
project(fused_transform LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(csrc)

add_executable(main main.cpp)
target_link_libraries(
    main PRIVATE sample_p_cube os transform simple_matmul utils CUDA::cudart
)

add_executable(bench bench.cpp)
target_link_libraries(
    bench PRIVATE sample_p_cube os transform simple_matmul utils CUDA::cudart
)