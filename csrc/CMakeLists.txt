cmake_minimum_required(VERSION 3.10)

add_library(utils SHARED utils.c)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(sample_p_cube SHARED sample_p_cube.c)
target_include_directories(sample_p_cube PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(os SHARED os.c)
target_include_directories(os PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(transform SHARED transform.c)
target_include_directories(transform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(simple_matmul SHARED simple_matmul.cu)
target_include_directories(simple_matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(simple_matmul PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
)