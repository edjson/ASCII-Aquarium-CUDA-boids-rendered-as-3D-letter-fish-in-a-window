cmake_minimum_required(VERSION 3.24)
project(koi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(koi src/main.cu)
target_compile_options(koi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> -Wall -Wextra)
