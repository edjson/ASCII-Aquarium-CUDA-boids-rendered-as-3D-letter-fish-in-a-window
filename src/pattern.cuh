#pragma once
#include "model.cuh"

// Pattern colors
enum{C_WHITE, C_GOLD, C_BLACK, C_RED};
enum { K_OGON, K_KOHAKU, K_TANCHO, K_SHOWA, K_BEKKO, K_CALICO, NKIND };
constexpr int NBLOB = 8;             // blobs per koi, each a float4 {u, v, radius, color}

struct Patterns{
    float4 *blob;
    int *style;
};

static Patterns patternInit(int nFish){
    Patterns p{};
    CUDA_CHECK(cudaMalloc(&p.blob,  (size_t)nFish * NBLOB * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&p.style, (size_t)nFish * sizeof(int)));
    return p;
}

// A small interger hash: the same number in always  give the same number out,
// on the CPU and on the GPU. hash01 turns it into a float from 0 - 1.

__host__ __device__ inline unsigned hashU(unsigned x){
    x ^= x >> 16;  x *= 0x7feb352dU;
    x ^= x >> 15;  x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

__host__ __device__ inline float hash01(unsigned x){
    return (hashU(x) >> 8) * (1.0f / 16777216.0f);
}

static void patternFree(Patterns& p){
    cudaFree(p.blob);
    cudaFree(p.style);
}

// Give every koi its variety and blobs, on the host, and upload them. Call
// again to reseed. Prints how many blobs each of the first six koi got.
static void patternSeed(Patterns& p, int nFish, unsigned seed){
    std::vector<float4> hb((size_t)nFish * NBLOB, make_float4(0, 0, 0, 0));   // radius 0: unused
    std::vector<int>    hs(nFish);                                            // host copy of style
    std::printf("seed %u blobs:", seed);
    for(int id = 0; id < nFish; id++){
        unsigned s = hashU((unsigned)id * 2654435761u + seed * 40503u);   // this koi's own stream
        auto rnd = [&](){
            s = hashU(s + 1u);
            return (s >> 8) * (1.0f / 16777216.0f);                       // 0..1
        };
        auto pick = [&](int lo, int hi){
            return lo + (int)(rnd() * (hi - lo + 1));                     // lo..hi
        };
        int kind = id % NKIND;       // TEMP (Step 5 picks at random): one koi of each variety
        int base = C_WHITE, fin = C_WHITE, n = 0;
        float4* b = &hb[(size_t)id * NBLOB];             // this koi's 8 blobs

        // k patches in a row along the back, from u0, to u1: radius r0 to r1,
        // each up to dv off the top line of the back.
        auto chain = [&](int k, float u0, float u1, float r0, float r1, float dv, int color){
            for(int i = 0; i < k && n < NBLOB; i++){
                float u = u0 + (u1 - u0) * (i + 0.2f + 0.6f * rnd()) / k;
                float v = 0.5f + dv * (2.0f * rnd() - 1.0f);
                b[n++] = make_float4(u, v, r0 + (r1 - r0) * rnd(), (float)color);
            }
        };

        switch(kind){
            case K_OGON:
                base = fin = C_GOLD;
                break;
            case K_KOHAKU:
                chain(pick(2, 4), 0.07f, 0.74f, 0.10f, 0.16f, 0.04f, C_RED);
                break;
            case K_TANCHO:
                b[n++] = make_float4(0.105f, 0.5f, 0.062f, (float)C_RED);
                break;
            case K_SHOWA:
                base = C_BLACK;
                chain(3,          0.04f, 0.78f, 0.10f, 0.15f, 0.10f, C_WHITE);
                chain(pick(2, 3), 0.06f, 0.70f, 0.09f, 0.14f, 0.06f, C_RED);
                break;
            case K_BEKKO:
                chain(pick(5, 7), 0.22f, 0.76f, 0.035f, 0.060f, 0.08f, C_BLACK);
                break;
            case K_CALICO:
                chain(pick(2, 3), 0.07f, 0.70f, 0.09f, 0.14f, 0.04f, C_RED);
                chain(pick(3, 5), 0.24f, 0.76f, 0.035f, 0.055f, 0.09f, C_BLACK);
                break;
        }
        hs[id] = kind | base << 8 | fin << 16;
        if(id < 6){
            std::printf(" %d", n);
        }
    }
    std::printf("\n");
    CUDA_CHECK(cudaMemcpy(p.blob,  hb.data(), hb.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.style, hs.data(), hs.size() * sizeof(int),    cudaMemcpyHostToDevice));
}