#pragma once
#include "config.cuh"
#include "vec3.cuh"

// Everyting sized by the window. Cuda draaws into fb; host is the pinned
// copy that SDL shows and then S key saves. Row 0 is the top of the screen

struct Screen{
    int    W, H;   // pixels
    uchar4 *fb;     // device, W * H
    uchar4 *host;   // pinned host copy of fb
    int cols, rows;
    unsigned char *glyph;
    float3 *color;
};

static Screen screenInit(int W, int H){
    Screen sc{};
    sc.W = W;
    sc.H = H;
    size_t bytes = (size_t)W * H * sizeof(uchar4);
    CUDA_CHECK(cudaMalloc(&sc.fb, bytes));
    CUDA_CHECK(cudaMallocHost(&sc.host, bytes)); // pinned; faster copies
    sc.cols = W / CW;                                 // whole cell only
    sc.rows = H / CH;
    size_t cells = (size_t)sc.cols * sc.rows;
    CUDA_CHECK(cudaMalloc(&sc.glyph, cells));
    CUDA_CHECK(cudaMalloc(&sc.color, cells * sizeof(float3)));
    CUDA_CHECK(cudaMemset(sc.glyph, 0, cells));
    CUDA_CHECK(cudaMemset(sc.color, 0, cells * sizeof(float3)));
    return sc;
}

static void screenFree(Screen& sc){
    cudaFree(sc.fb);
    cudaFreeHost(sc.host);
    cudaFree(sc.glyph);
    cudaFree(sc.color);
}

// Waits for the kernels, then copies the finsihed frame to the host.
static void screenDownload(Screen&sc){
    CUDA_CHECK(cudaMemcpy(sc.host, sc.fb, (size_t)sc.W * sc.H * sizeof(uchar4),
                            cudaMemcpyDeviceToHost));
}

const dim3 TILE(32, 8);
static dim3 blocksFor(int w, int h){
    return dim3((w + TILE.x -1) / TILE.x, (h + TILE.y - 1) / TILE.y);
}

// Color 0..1 to one byte, clamped. rgba() is the only place that knows the
// byte order: uchar4 x, y, z, w = R, G, B, A in memory, SDL's RGA32.

__device__ inline unsigned char u8(float v){
    return (unsigned char)(255.0f * fminf(fmaxf(v, 0.0f), 1.0f));
}

__device__ inline uchar4 rgba(float3 c){
    return make_uchar4(u8(c.x), u8(c.y), u8(c.z), 255);
}

// TEMP test pattern (Step 3 replaces it): red grows to the right, green
// grows downward, blue pulses with the sim clock.
__global__ void k_gradient(Screen sc, float t){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= sc.W || y >= sc.H){
        return;
    }

    float3 c = make_float3((float)x / (sc.W - 1), (float)y / (sc.H - 1),
                    0.5f + 0.5f * sinf(2.0f * t));
    sc.fb[y * sc.W + x] = rgba(c);
}

// TEMP draw evey fish as a white 5x5 dow seen form above

__global__ void k_dots(Screen sc, const float3* pos, int N, float3 tank){
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;
    }

    int px = (int)(pos[i].x / tank.x * (sc.W - 1));
    int py = (int)(pos[i].z / tank.z * (sc.H - 1));
    for(int dy = -2; dy <= 2; dy++){
        for(int dx = -2; dx <= 2; dx++){
            int x = px + dx, y = py + dy;
            if(x >= 0 && x < sc.W && y >= 0 && y < sc.H){
                sc.fb[y * sc.W + x] = make_uchar4(255, 255, 255, 255);
            }
        }
    }
}