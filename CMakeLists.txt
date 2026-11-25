cmake_minimum_required(VERSION 3.24)
project(koi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(koi src/main.cu src/sort.cu)
target_compile_options(koi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> -Wall -Wextra)

# SDL2 opens the window; SDL2_image write the png screenshots

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)            # SDL2_ttf draws the letters once, at startup
target_link_libraries(koi PRIVATE SDL2::SDL2 SDL2_image::SDL2_image SDL2_ttf::SDL2_ttf)