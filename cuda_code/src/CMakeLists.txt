cmake_minimum_required(VERSION 3.8)

add_definitions(-D_USE_MATH_DEFINES)
find_package(OpenCV 4.2.0 REQUIRED EXACT)
find_package(CUDA)
include(FindCUDA)

cuda_add_executable(surface_normal_estimation_cvpr main.cu stdafx.h)
target_link_libraries(surface_normal_estimation_cvpr ${OpenCV_LIBS} )

set(CUDA_NVCC_FLAGS; -arch=sm_21 )#cmake






