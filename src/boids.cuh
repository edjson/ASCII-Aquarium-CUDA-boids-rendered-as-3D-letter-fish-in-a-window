#pragma once 
#include "config.cuh"
#include "vec3.cuh"

// Which cell a cooridnate falls in, clamped to the grid
__device__ inline int cellOf(float p, float cellSize, int g){
    return min(max((int)(p / cellSize), 0), g - 1);
}

// 3D cells coords -> flat cell id. x varies fastest
__device__ inline int cellIndex(int x, int y, int z, int3 grid){
    return((z * grid.y + y) * grid.x + x);
}


__global__ void k_assign(const float3* pos, int* cellId, int* perm, int N, int3 grid, float cellSize){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;
    }

    float3 p = pos[i];
    int cx = cellOf(p.x, cellSize, grid.x);
    int cy = cellOf(p.y, cellSize, grid.y);
    int cz = cellOf(p.z, cellSize, grid.z);
    cellId[i] = cellIndex(cx, cy, cz, grid);
    perm[i] = i;
}

__global__ void k_bounds(const int* cellId, int* cellStart, int* cellEnd, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;
    }

    int c = cellId[i];
    if(i == 0 || cellId[i - 1] != c){
        cellStart[c] = i;
    }
    if(i == N -1 || cellId[i + 1] != c){
        cellEnd[c] = i + 1;
    }
}

__global__ void k_reorder(const float3* pos, const float3* vel, const int* fishId, const int* perm,
    float3* posOut, float3* velOut, int* idOut, int N){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= N){
        return;
    }

    int j = perm[k];
    posOut[k] = pos[j];
    velOut[k] = vel[j];
    idOut[k] = fishId[j];
}