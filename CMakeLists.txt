cmake_minimum_required(VERSION 3.18)
project(llm-engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_OPENBLAS "Use OpenBLAS for BLAS operations" ON)
option(USE_EIGEN "Use Eigen for linear algebra operations" OFF)
option(ENABLE_SIMD "Enable SIMD optimizations" OFF)
option(ENABLE_TESTS "Build unit tests" ON)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(nlohmann_json QUIET) # For Safetensors and HTTP server

# Eigen is optional - find it if available and user wants it
if(USE_EIGEN)
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        message(WARNING "Eigen3 not found, falling back to simple implementation")
        set(USE_EIGEN OFF)
    endif()
endif()

if(USE_EIGEN AND Eigen3_FOUND)
    message(STATUS "Using Eigen for linear algebra operations")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(USE_OPENBLAS)
    find_package(BLAS REQUIRED)
    find_path(OPENBLAS_INCLUDE_DIR
        NAMES openblas_config.h
        HINTS ${BLAS_INCLUDE_DIRS}
    )
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Protobuf generated files (for ONNX parsing)
# Note: In a real implementation, you would generate these from .proto files
# For now, we'll compile the ONNX loader as a stub

# Compiler flags
if(ENABLE_SIMD)
    add_compile_options(-mavx2 -mfma)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${Protobuf_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/app.cpp
    src/tensor.cpp
    src/alloc.cpp
    src/loaders/onnx_loader.cpp
    src/loaders/gguf_loader.cpp
    src/loaders/safetensors_loader.cpp
    src/kernels/gemm_ref.cpp
    src/kernels/q4_rowwise.cpp
    src/kernels/optimized/simd_gemm.cpp
    src/kernels/optimized/flash_attention.cpp
    src/transformer/transformer.cpp
    src/tokenizer/sentencepiece_wrapper.cpp
    src/util/threadpool.cpp
    src/util/profiler.cpp
    src/http_server.cpp
    src/batch_processor.cpp
)

# Create executable
add_executable(infer ${SOURCES})

# Link libraries
target_link_libraries(infer ${Protobuf_LIBRARIES})
if(USE_EIGEN AND Eigen3_FOUND)
    target_link_libraries(infer Eigen3::Eigen)
endif()
if(nlohmann_json_FOUND)
    target_link_libraries(infer nlohmann_json::nlohmann_json)
endif()

if(USE_EIGEN AND Eigen3_FOUND)
    target_compile_definitions(infer PRIVATE USE_EIGEN)
endif()

if(USE_OPENBLAS)
    target_link_libraries(infer ${BLAS_LIBRARIES})
    target_include_directories(infer PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_compile_definitions(infer PRIVATE USE_OPENBLAS)
endif()

if(ENABLE_SIMD)
    target_compile_definitions(infer PRIVATE ENABLE_SIMD)
endif()

# Tests
if(ENABLE_TESTS)
    add_executable(unit_tests tests/unit_tests.cpp ${SOURCES})
    target_link_libraries(unit_tests ${Protobuf_LIBRARIES})
    if(USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(unit_tests Eigen3::Eigen)
        target_compile_definitions(unit_tests PRIVATE USE_EIGEN)
    endif()
    if(USE_OPENBLAS)
        target_link_libraries(unit_tests ${BLAS_LIBRARIES})
        target_include_directories(unit_tests PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_compile_definitions(unit_tests PRIVATE USE_OPENBLAS)
    endif()
    if(ENABLE_SIMD)
        target_compile_definitions(unit_tests PRIVATE ENABLE_SIMD)
    endif()
endif()

# Create necessary directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_baselines)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/loaders)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/optimized)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/transformer)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/util)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools)
