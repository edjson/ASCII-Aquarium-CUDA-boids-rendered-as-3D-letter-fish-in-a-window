#pragma once
#include "boids.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <random>
#include <utility>
#include <vector>

struct Sim {
    int     N;
    float3  tank;                    // pond size, world units
    float3 *pos, *vel;               // state, cell-sorted after each step
    float3 *posSort, *velSort;       // k_reorder output, k_boids input
    int    *fishId, *idSort;         // stable identity, travels with its fish
    int    *cellId, *perm;           // sort keys and values
    int    *cellStart, *cellEnd;     // fish range [start, end) of each cell
    int3    grid;
    int     nCells;
    float   cellSize;
    float   time;                    // seconds simulated, drives the wander
};

static void simInit(Sim& s, int N, float3 tank){
    s.N        = N;
    s.tank     = tank;
    s.cellSize = VIEW_R;
    s.grid     = make_int3((int)(tank.x / s.cellSize) + 1,
                           (int)(tank.y / s.cellSize) + 1,
                           (int)(tank.z / s.cellSize) + 1);
    s.nCells   = s.grid.x * s.grid.y * s.grid.z;

    size_t f3 = (size_t)N * sizeof(float3);
    size_t i1 = (size_t)N * sizeof(int);
    size_t gc = (size_t)s.nCells * sizeof(int);
    CUDA_CHECK(cudaMalloc(&s.pos,       f3));
    CUDA_CHECK(cudaMalloc(&s.vel,       f3));
    CUDA_CHECK(cudaMalloc(&s.posSort,   f3));
    CUDA_CHECK(cudaMalloc(&s.velSort,   f3));
    CUDA_CHECK(cudaMalloc(&s.fishId,    i1));
    CUDA_CHECK(cudaMalloc(&s.idSort,    i1));
    CUDA_CHECK(cudaMalloc(&s.cellId,    i1));
    CUDA_CHECK(cudaMalloc(&s.perm,      i1));
    CUDA_CHECK(cudaMalloc(&s.cellStart, gc));
    CUDA_CHECK(cudaMalloc(&s.cellEnd,   gc));

}