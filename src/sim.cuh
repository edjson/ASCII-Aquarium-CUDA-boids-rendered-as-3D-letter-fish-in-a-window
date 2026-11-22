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

// Fish start in the middle box, clear  of the wall margins, swimming level.
// call again to reseed.

static void simSeed(Sim& s, unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto inside = [&](float len, float m){
        return(m + (U(rng) * (len - (2.0f * m))));
    };

    std::vector<float3> hp(s.N), hv(s.N);
    std::vector<int>    hid(s.N);
    for(int i = 0; i < s.N; i++){
        float x  = inside(s.tank.x, WALL_M);
        float y  = inside(s.tank.y, WALL_MY);
        float z  = inside(s.tank.z, WALL_M);
        float a  = U(rng) * 6.2831853f;                            // heading in the x-z plane
        float sp = MIN_SPEED + (U(rng) * (SPEED - MIN_SPEED));     // starting speed
        hp[i]    = make_float3(x, y, z);
        hv[i]    = make_float3(cosf(a) * sp, 0.0f, sinf(a) * sp);  // level: no y speed
        hid[i]   = i;                                              // fish i starts in slot i
    }
    CUDA_CHECK(cudaMemcpy(s.pos, hp.data(), s.N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.vel, hv.data(), s.N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.fishId , hid.data(), s.N * sizeof(int), cudaMemcpyHostToDevice));
    s.time = 0.0f;
}