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