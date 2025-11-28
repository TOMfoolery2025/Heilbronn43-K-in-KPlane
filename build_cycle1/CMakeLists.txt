# CMakeLists.txt for Cycle 1: Geometry Verification
# Builds planar_cuda module with pybind11 and CUDA support
#
# Build Instructions:
#   mkdir -p build_cycle1
#   cd build_cycle1
#   cmake ..
#   cmake --build . --config Release
#
# Output: planar_cuda.pyd (Windows) or planar_cuda.so (Linux)

cmake_minimum_required(VERSION 3.18)
project(planar_cuda LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4060 (sm_89)

# ============================================================================
# Find Dependencies
# ============================================================================

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Try pip-installed pybind11
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Build Module
# ============================================================================

# Create Python module from CUDA source
pybind11_add_module(planar_cuda 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_utils/planar_cuda.cu
)

# Set CUDA properties
set_target_properties(planar_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA runtime
target_link_libraries(planar_cuda PRIVATE CUDA::cudart)

# Include directories
target_include_directories(planar_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${Python_INCLUDE_DIRS}
)

# Compiler options
target_compile_options(planar_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -Xcompiler=/MD
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/EHsc /MD>
        $<$<CXX_COMPILER_ID:GNU>:-fPIC>
    >
)

# ============================================================================
# Installation
# ============================================================================

# Copy to build_artifacts directory
add_custom_command(TARGET planar_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:planar_cuda>
        ${CMAKE_CURRENT_SOURCE_DIR}/../build_artifacts/planar_cuda$<TARGET_FILE_SUFFIX:planar_cuda>
    COMMENT "Copying planar_cuda module to build_artifacts/"
)

# ============================================================================
# Testing (Optional)
# ============================================================================

# Print configuration
message(STATUS "==============================================")
message(STATUS "  Cycle 1: Geometry Verification Build")
message(STATUS "==============================================")
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Python Include: ${Python_INCLUDE_DIRS}")
message(STATUS "pybind11: ${pybind11_DIR}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "==============================================")
