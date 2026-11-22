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

//soft wall: inside the margin m, push back toward the middle, gorwing as t^2

__device__ inline float wallPush(float p, float hi, float m){
    float a = 0.0f;
    if(p < m){
        float t = 1.0f - (p / m);
        a += (t * t);
    }
    if(p > (hi - m)){
        float t = 1.0f - ((hi - p) / m);
        a -= (t * t);
    }
    return(a * WALL_W);
}

// Hard wall, only a saftey net: mirror the pisiont point velcity inward.

__device__ inline void reflect(float& p, float& v, float hi){
    if(p < 0.0f){
        p = -p;
        v = fabsf(v);
    }
    if(p > hi){
        p = (2.0f * hi) - p;
        v = -fabsf(v);
    }
    p = fminf(fmaxf(p, 0.0f), hi);
}

__global__ void k_boids(const float3* posIn, const float3* velIn, const int* fishId,
    float3* posOut, float3* velOut, const int* cellStart, const int* cellEnd,
    int N, float dt, float time, float3 tank, int3 grid, float cellSize){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= N){
            return;
        }
        float3 p = posIn[i], v = velIn[i];
        float3 sep = {0, 0, 0}, ali = {0, 0, 0}, coh = {0, 0, 0};
        int n = 0;

        int cx = cellOf(p.x, cellSize, grid.x);
        int cy = cellOf(p.y, cellSize, grid.y);
        int cz = cellOf(p.z, cellSize, grid.z);

        for(int oz = -1; oz <= 1; oz++){                 // 3x3x3 neighbor cells
            for(int oy = -1; oy <= 1; oy++){
                for(int ox = -1; ox <= 1; ox++){
                    int nx = cx + ox, ny = cy + oy, nz = cz + oz;
                    if(nx < 0 || ny < 0 || nz < 0 ||
                       nx >= grid.x || ny >= grid.y || nz >= grid.z){
                        continue;                        // cell is off the edge
                    }

                    int c = cellIndex(nx, ny, nz, grid);
                    int s = cellStart[c];
                    if(s < 0){
                        continue;                        // empty cell
                    }

                    for(int j = s, e = cellEnd[c]; j < e; j++){
                        if(j == i){
                            continue;                    // skip myself
                        }
                        float3 d = posIn[j] - p;
                        float d2 = dot(d, d);
                        if(d2 > (VIEW_R * VIEW_R)){
                            continue;                    // too far to see
                        }

                        if(d2 < (SEP_R * SEP_R)){        // push away: 1 when touching, 0 at SEP_R
                            sep -= d * (rsqrtf(fmaxf(d2, 1e-4f)) - (1.0f / SEP_R));
                        }
                        ali += velIn[j];
                        coh += posIn[j];
                        n++;
                    }
                }
            }
        }
        // Steering: what the koi wants
        float3 acc = sep * SEP_W;
        if(n > 0){
            float inv = 1.0f / n;
            acc += ((((ali * inv) - v) * ALI_W) +            // match the average heading
                    (((coh * inv) - p) * COH_W)); 
        }

        // Wander: each koi's own slow nudge

        float id = (float)fishId[i];
        acc.x += WANDER_W * sinf((time * 0.22f) + (id * 2.4f));
        acc.y += 0.5f * sinf((time * 0.17f) + (id * 1.7f));   // optional: slow depth drift
        acc.z += WANDER_W * sinf((time * 0.31f) + (id * 4.4f));
        
        //cap steering: smooth turns
        float a = length(acc);
        if(a > MAX_FORCE){
            acc = acc * (MAX_FORCE / a);
        }

        // Not steering, so not capped; the walls always win

        acc.x += wallPush(p.x, tank.x, WALL_M);
        acc.y += wallPush(p.y, tank.y, WALL_MY);
        acc.z += wallPush(p.z, tank.z, WALL_M);

        // Cruise: ease back to level swimming at CRUISE speed
        float h = sqrtf((v.x * v.x) + (v.z * v.z));
        float3 goal = make_float3(v.x, 0.0f, v.z) * (CRUISE / fmaxf(h, 1e-4f));
        acc += (goal - v) * CRUISE_W; 

        //Move: integrate, limit the climb angle, clamp the speed
        v += acc * dt;
        h = sqrtf((v.x * v.x) + (v.z * v.z));
        v.y = fminf(fmaxf(v.y, -MAX_CLIMB * h), MAX_CLIMB * h);
        float sp = length(v);
        if(sp < 1e-4f){
            v = make_float3(MIN_SPEED, 0, 0);
            sp = MIN_SPEED;
        }
        v = v * (fminf(fmaxf(sp, MIN_SPEED), SPEED) / sp);

        p += v * dt;
        reflect(p.x, v.x, tank.x);
        reflect(p.y, v.y, tank.y);
        reflect(p.z, v.z, tank.z);

        posOut[i] = p;
        velOut[i] = v;
}
        
