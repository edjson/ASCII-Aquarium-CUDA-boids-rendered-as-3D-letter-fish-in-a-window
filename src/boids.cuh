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