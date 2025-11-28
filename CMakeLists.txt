cmake_minimum_required(VERSION 3.18)

# First, find CUDA
find_package(CUDAToolkit 12.6 REQUIRED)

# Set compilers
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")

# Declare project
project(planar_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (adjust for your GPU)
# RTX 4060 = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11
add_subdirectory(pybind11 EXCLUDE_FROM_ALL)
# If pybind11 is not a subdirectory, use: find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)
# Link CUDA libraries
target_link_libraries(planar_cuda PRIVATE CUDA::cudart)
set(CUDA_SOURCES
    src/cuda_utils/vector_ops.cu
)

# Create Python module
pybind11_add_module(planar_cuda
    src/cpp_binding/binding.cpp
    ${CUDA_SOURCES}
)

# Link CUDA libraries
target_link_libraries(planar_cuda PRIVATE ${CUDA_LIBRARIES})

# Set output directory
set_target_properties(planar_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA specific flags
target_compile_options(planar_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
    >
)

# Windows-specific: ensure correct CUDA DLL paths
if(WIN32)
    set_target_properties(planar_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
