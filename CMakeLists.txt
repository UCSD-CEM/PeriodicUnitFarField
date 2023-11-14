# Fnagzhou Ai @ Prof. Vitaliy Lomakin Group
# Sep 26th, 2023, Willoughby Hills, Ohio, USA
# Cmake file for cross-platform compilation
# Windows platform in principle should use Ninja generator
# Check Visual studio project properties carefully before using
cmake_minimum_required(VERSION 3.24)

# Set host arch and generator, AMD x64 only
if (MSVC)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

set(CMAKE_CUDA_ARCHITECTURES all)

# Release type
option(BUILD_DEBUG CACHE OFF)
if(BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE DEBUG)
else()
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

project(PUFF VERSION 0.0.1 LANGUAGES CXX CUDA)

# Basic CXX flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    # pass definition USE_OPENMP to compiler
    add_definitions(-DUSE_OPENMP)
    # pass -fopenmp to host compiler
    if(MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/openmp\"")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif()
endif()

# CUDA part
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
# set CUDA enviroment
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -use_fast_math -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

# MKL part
set(MKL_LINK static)
set(MKL_ARCH intel64)
set(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL CONFIG REQUIRED)

# Third party libraries
# cusp
set(CUSP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cusplibrary)
# cxxopts
set(CXXOPTS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)


# Set output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# if BUILD_BENCHMARK is on
option(BUILD_BENCHMARK CACHE ON)
if(BUILD_BENCHMARK)
    add_executable(benchmark_host_device ${CMAKE_SOURCE_DIR}/benchmarks/benchmark_host_device.cu)
    target_include_directories(benchmark_host_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(benchmark_host_device PUBLIC ${CUSP_INCLUDE_DIR})
    target_include_directories(benchmark_host_device PUBLIC ${CXXOPTS_INCLUDE_DIR})
    target_compile_options(benchmark_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(benchmark_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(benchmark_host_device PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(benchmark_host_device PUBLIC 
                                                CUDA::cufftw
                                                CUDA::cufft
                                                CUDA::cudart
                                                )
    #link OpenMP
    if(OpenMP_CXX_FOUND)
        target_include_directories(benchmark_host_device PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(benchmark_host_device PUBLIC OpenMP::OpenMP_CXX)
    endif()
    # CUDA Architecture
    set_property(TARGET benchmark_host_device PROPERTY CUDA_ARCHITECTURES all)
endif()


# if BUILD_EXAMPLE is on
option(BUILD_EXAMPLE CACHE ON)
if(BUILD_EXAMPLE)
    add_executable(example_host_device ${CMAKE_SOURCE_DIR}/examples/example_host_device.cu)
    target_include_directories(example_host_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(example_host_device PUBLIC ${CUSP_INCLUDE_DIR})
    target_include_directories(example_host_device PUBLIC ${CXXOPTS_INCLUDE_DIR})
    target_compile_options(example_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(example_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(example_host_device PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(example_host_device PUBLIC 
                                            CUDA::cufftw
                                            CUDA::cufft
                                            CUDA::cudart
                                            )
    #link OpenMP
    if(OpenMP_CXX_FOUND)
        target_include_directories(example_host_device PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(example_host_device PUBLIC OpenMP::OpenMP_CXX)
    endif()
    # CUDA Architecture
    set_property(TARGET example_host_device PROPERTY CUDA_ARCHITECTURES all)
endif() 


# if BUILD_TEST is on
option(BUILD_TEST CACHE ON)
if(BUILD_TEST)
    # GTest package
    include(FetchContent)
    FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    # Test
    add_executable(test_host_device ${CMAKE_SOURCE_DIR}/tests/test_host_device.cu)
    target_link_libraries(test_host_device PUBLIC GTest::gtest_main)
    target_include_directories(test_host_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(test_host_device PUBLIC ${CUSP_INCLUDE_DIR})
    target_include_directories(test_host_device PUBLIC ${CXXOPTS_INCLUDE_DIR})
    target_compile_options(test_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(test_host_device PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(test_host_device PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(test_host_device PUBLIC 
                                            CUDA::cufftw
                                            CUDA::cufft
                                            CUDA::cudart
                                            )
    #link OpenMP
    if(OpenMP_CXX_FOUND)
        target_include_directories(test_host_device PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(test_host_device PUBLIC OpenMP::OpenMP_CXX)     
    endif()
    gtest_discover_tests(test_host_device)
    # CUDA Architecture
    set_property(TARGET test_host_device PROPERTY CUDA_ARCHITECTURES all)
endif()