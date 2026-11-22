#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(1);                                               \
        }                                                               \
    } while (0)
