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
